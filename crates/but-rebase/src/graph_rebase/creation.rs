use std::collections::HashMap;

use anyhow::{Context as _, Result, bail};
use but_core::{RefMetadata, commit::SignCommit};

use crate::graph_rebase::{
    Checkout, Editor, Pick, RevisionHistory, Selector, Step, StepGraph, StepGraphIndex,
    SuccessfulRebase, placements,
};

#[derive(Clone)]
/// Options for the editor.
pub struct GraphEditorOptions {
    /// Determines how cherry-picked commits are signed.
    pub default_sign_commit: SignCommit,
    /// References whose segment should be forced mutable.
    ///
    /// The editor always contains every segment in the workspace graph, with
    /// only those reachable from `HEAD` being mutable. Use this to force a
    /// segment that isn't reachable from `HEAD` to be mutable so it can be
    /// rewritten.
    pub extra_mutable_refs: Vec<gix::refs::FullName>,
}

impl Default for GraphEditorOptions {
    fn default() -> Self {
        Self {
            default_sign_commit: SignCommit::IfSignCommitsEnabled,
            extra_mutable_refs: vec![],
        }
    }
}

/// Creates an editor out of the workspace graph.
impl<'ws, 'meta, M: RefMetadata> Editor<'ws, 'meta, M> {
    /// Creates an editor out of the workspace graph with the default options.
    pub fn create(
        workspace: &'ws mut but_graph::Workspace,
        meta: &'meta mut M,
        repo: &gix::Repository,
    ) -> Result<Self> {
        Self::create_with_opts(workspace, meta, repo, &GraphEditorOptions::default())
    }

    /// Creates an editor out of the workspace graph with the specified options.
    pub fn create_with_opts(
        workspace: &'ws mut but_graph::Workspace,
        meta: &'meta mut M,
        repo: &gix::Repository,
        options: &GraphEditorOptions,
    ) -> Result<Self> {
        // The editor graph is built NATIVELY: the ref-placement ledger derives from the
        // segment graph and create_native builds picks straight from the carried CommitGraph.
        let ledger = placements::derive(workspace, repo, options)?;
        let (graph, references, checkouts) = create_native(workspace, repo, options, &ledger)?;
        Ok(Self {
            graph,
            initial_references: references,
            checkouts,
            repo: repo.clone().with_object_memory(),
            history: RevisionHistory::new(),
            workspace,
            meta,
        })
    }
}

/// Build the editor graph NATIVELY: picks and their ordered parent arrays straight from the
/// carried [`but_graph::CommitGraph`], references and their positions from the placement
/// ledger — no segment walk, no temporary ref edges, no strip pass.
fn create_native(
    workspace: &but_graph::Workspace,
    repo: &gix::Repository,
    options: &GraphEditorOptions,
    ledger: &placements::RefPlacements,
) -> Result<(StepGraph, Vec<gix::refs::FullName>, Vec<Checkout>)> {
    let Some(cg) = workspace.graph.commit_graph() else {
        bail!("native creation requires the graph to carry its CommitGraph");
    };
    let workspace_commit_id = workspace
        .graph
        .managed_entrypoint_commit(repo)?
        .map(|c| c.id);

    let mut graph = StepGraph::new();
    let mut pick_by_id = HashMap::<gix::ObjectId, StepGraphIndex>::new();
    for id in cg.commit_ids() {
        let mut pick = if workspace_commit_id == Some(id) {
            Pick::new_workspace_pick(id)
        } else {
            let mut pick = Pick::new_pick(id);
            pick.sign_commit = options.default_sign_commit;
            pick
        };
        pick.mutable = ledger.mutable_commits.contains(&id);
        let ix = graph.add_node(Step::Pick(pick));
        pick_by_id.insert(id, ix);
    }
    // ARENA IDENTITY (dissolve invariant): the pick arena mirrors the CommitGraph arena
    // index-for-index — Node(i) is commit i. The dissolve swaps the former for the latter on
    // the strength of this.
    debug_assert!(
        cg.commit_ids()
            .enumerate()
            .all(|(i, id)| graph.commit_id(StepGraphIndex::Node(i)) == Some(id)),
        "native pick arena must mirror the CommitGraph arena index-for-index"
    );

    for id in cg.commit_ids().collect::<Vec<_>>() {
        let ix = pick_by_id[&id];
        let raw_parents = &cg.node(id).expect("iterating graph ids").commit.parent_ids;
        // A parent outside the graph means the traversal was partial here — preserve the raw
        // parent list so the rebase keeps the commit's real ancestry.
        if !raw_parents.is_empty() && raw_parents.iter().any(|p| cg.node(*p).is_none()) {
            graph.set_preserved_parents(ix, Some(raw_parents.clone()));
        }
        // The ws commit takes its LANE slots from the ledger (one per workspace lane, dups
        // and all); everything else wires the PRESENT parents in parent order — the same
        // presence filter the segment walk's parent-fixup pass applies.
        if workspace_commit_id == Some(id) {
            for parent in ledger.ws_parents.as_deref().unwrap_or_default() {
                graph.push_parent(ix, pick_by_id[parent]);
            }
        } else {
            for parent in cg.parents(id) {
                graph.push_parent(ix, pick_by_id[&parent]);
            }
        }
    }

    // Two passes: refs stack top-down in the ledger (a ref's `below` has a HIGHER index), so
    // every node must exist before positions can name it.
    let mut ref_by_name = HashMap::<gix::refs::FullName, StepGraphIndex>::new();
    for placed in &ledger.refs {
        let ix = graph.add_reference(placed.name.clone(), placed.mutable);
        ref_by_name.insert(placed.name.clone(), ix);
    }
    for placed in &ledger.refs {
        // Unborn refs (no anchor) keep no stored position.
        let Some(anchor_id) = placed.anchor else {
            continue;
        };
        let node = ref_by_name[&placed.name];
        let Some(&anchor) = pick_by_id.get(&anchor_id) else {
            bail!("ledger anchor {anchor_id} is not a commit in the graph");
        };
        let below =
            match &placed.below {
                Some(name) => Some(*ref_by_name.get(name).with_context(|| {
                    format!("ledger below {name} is not a reference in the graph")
                })?),
                None => None,
            };
        let mut approach = Vec::with_capacity(placed.approach.len());
        for (source, slot) in &placed.approach {
            let Some(&source_ix) = pick_by_id.get(source) else {
                bail!("ledger approach source {source} is not a commit in the graph");
            };
            approach.push((source_ix, *slot));
        }
        graph.set_position(node, anchor, &approach, placed.ambiguous, below);
    }

    let references = ledger
        .refs
        .iter()
        .filter(|r| r.mutable)
        .map(|r| r.name.clone())
        .collect();
    let checkouts = ledger
        .head_refs
        .iter()
        .map(|name| {
            let Some(&id) = ref_by_name.get(name) else {
                bail!("ledger head ref {name} is not a reference in the graph");
            };
            Ok(Checkout::Head {
                selector: Selector { id },
                merge_base_override: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    crate::graph_rebase::positions::debug_assert_positions_total(&graph);
    Ok((graph, references, checkouts))
}

impl<'ws, 'meta, M: RefMetadata> SuccessfulRebase<'ws, 'meta, M> {
    /// Converts a SuccessfulRebase back into another editor for multi-step operations.
    ///
    /// This is the normalization path for callers that want to chain
    /// additional editor-based operations and need the editor graph plus
    /// in-memory repository to agree on ancestry.
    pub fn into_editor(self) -> Editor<'ws, 'meta, M> {
        Editor {
            graph: self.graph,
            initial_references: self.initial_references,
            checkouts: self.checkouts,
            repo: self.repo,
            history: self.history,
            workspace: self.workspace,
            meta: self.meta,
        }
    }
}
