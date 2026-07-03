use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};
use but_core::{RefMetadata, commit::SignCommit};
use but_graph::Direction;
use but_graph::{Commit, SegmentIndex};
use petgraph::visit::EdgeRef as _;

use crate::graph_rebase::{
    Checkout, Edge, Editor, Pick, RevisionHistory, Selector, Step, StepGraph, StepGraphIndex,
    SuccessfulRebase, util,
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
        let native = super::native_creation::create_native(workspace, repo, options, meta)?;
        // Transitional oracle: BUT_REBASE_NATIVE=assert also runs the legacy segment walk and
        // panics with precise diffs if it disagrees in canonical form (see canonical.rs).
        if std::env::var("BUT_REBASE_NATIVE").ok().as_deref() == Some("assert") {
            let (graph, references, head_selectors, immutable_references) =
                segment_walk_parts(workspace, repo, options)?;
            let old_c = super::canonical::canonical_form(&graph);
            let new_c = super::canonical::canonical_form(&native.graph);
            let mut diffs = new_c.diff_against(&old_c);
            let sorted = |v: &[gix::refs::FullName]| {
                let mut v: Vec<_> = v.iter().map(|r| r.to_string()).collect();
                v.sort();
                v
            };
            if sorted(&native.references) != sorted(&references) {
                diffs.push(format!(
                    "references {:?} != {:?}",
                    sorted(&native.references),
                    sorted(&references)
                ));
            }
            let selector_labels = |g: &StepGraph, sels: &[Selector]| {
                let mut v: Vec<String> = sels
                    .iter()
                    .map(|s| match &g[s.id] {
                        Step::Reference { refname } => refname.to_string(),
                        other => format!("{other:?}"),
                    })
                    .collect();
                v.sort();
                v
            };
            let old_sel = selector_labels(&graph, &head_selectors);
            let new_sel = selector_labels(&native.graph, &native.head_selectors);
            if new_sel != old_sel {
                diffs.push(format!("head_selectors {new_sel:?} != {old_sel:?}"));
            }
            let mut imm_old: Vec<_> = immutable_references.iter().map(|r| r.to_string()).collect();
            let mut imm_new: Vec<_> = native
                .immutable_references
                .iter()
                .map(|r| r.to_string())
                .collect();
            imm_old.sort();
            imm_new.sort();
            if imm_new != imm_old {
                diffs.push(format!("immutable {imm_new:?} != {imm_old:?}"));
            }
            if !diffs.is_empty() {
                panic!(
                    "NATIVE_EDITOR_DIVERGENCE ({} lines):\n{}",
                    diffs.len(),
                    diffs.join("\n")
                );
            }
        }
        Ok(Self {
            graph: native.graph,
            initial_references: native.references,
            // TODO(CTO): We need to eventually list all worktrees that we own
            // here so we can `safe_checkout` them too.
            checkouts: native
                .head_selectors
                .into_iter()
                .map(|selector| Checkout::Head {
                    selector,
                    merge_base_override: None,
                })
                .collect(),
            repo: repo.clone().with_object_memory(),
            history: RevisionHistory::new(),
            immutable_references: native.immutable_references,
            workspace,
            meta,
        })
    }
}

/// The legacy editor-graph construction: runs of nodes per but-graph SEGMENT, connected via the
/// segment edges, then a repair pass rewiring any pick whose graph-derived parents disagree with
/// the commit's real parent array. Kept as the transitional oracle for `native_creation`.
#[expect(clippy::type_complexity)]
fn segment_walk_parts(
    workspace: &but_graph::Workspace,
    repo: &gix::Repository,
    options: &GraphEditorOptions,
) -> Result<(
    StepGraph,
    Vec<gix::refs::FullName>,
    Vec<Selector>,
    HashSet<gix::refs::FullName>,
)> {
    {
        let entrypoint = workspace.graph.entrypoint()?;

        let mut mutable_entrypoints = vec![entrypoint.segment.id];

        for ref_name in &options.extra_mutable_refs {
            let Some((segment, _)) = workspace
                .graph
                .segment_and_commit_by_ref_name(ref_name.as_ref())
            else {
                bail!("Failed to find corresponding segment for {ref_name}");
            };
            mutable_entrypoints.push(segment.id);
        }

        // Segments reachable from a mutable entrypoint (following parent edges)
        // may be rewritten. Every other segment is still included in the
        // editor, but as immutable.
        let mut mutable_segments = HashSet::new();
        for entrypoint in mutable_entrypoints {
            workspace.graph.visit_all_segments_including_start_until(
                entrypoint,
                Direction::Outgoing,
                |segment| !mutable_segments.insert(segment.id),
            );
        }

        // The editor contains every commit the graph contains, so we iterate
        // over all segments rather than only those reachable from an entrypoint.
        let segments_to_add = workspace.graph.segments().collect::<Vec<_>>();

        let workspace_commit_id = workspace
            .graph
            .managed_entrypoint_commit(repo)?
            .map(|c| c.id);

        let mut commits: Vec<Commit> = vec![];
        let mut commit_to_idx = HashMap::<gix::ObjectId, SegmentIndex>::new();
        let mut commit_to_pick_ix = HashMap::<gix::ObjectId, StepGraphIndex>::new();
        let mut graph = StepGraph::new();
        let mut head_selectors = vec![];
        let mut references = vec![];
        struct NodeSegment {
            nodes: Vec<StepGraphIndex>,
        }

        let mut segments = HashMap::<SegmentIndex, NodeSegment>::new();

        for sid in segments_to_add {
            let segment = &workspace.graph[sid];
            let mutable = mutable_segments.contains(&sid);
            let mut nodes = vec![];

            if let Some(reference) = segment.ref_name() {
                let refname = reference.to_owned();
                // Only mutable references are tracked for potential deletion.
                if mutable {
                    references.push(refname.clone());
                }
                let ix = graph.add_node(Step::Reference {
                    refname: refname.clone(),
                    mutable,
                });
                if Some(reference) == entrypoint.segment.ref_name() {
                    head_selectors.push(Selector {
                        id: ix,
                        revision: 0,
                    });
                }
                nodes.push(ix);
            }

            for commit in &segment.commits {
                commits.push(commit.clone());
                commit_to_idx.insert(commit.id, segment.id);

                let refs = commit
                    .refs
                    .iter()
                    .map(|r| r.ref_name.clone())
                    .collect::<Vec<_>>();

                for reference in refs {
                    if mutable {
                        references.push(reference.to_owned());
                    }
                    let ix = graph.add_node(Step::Reference {
                        refname: reference.clone(),
                        mutable,
                    });
                    if let Some(previous_ix) = nodes.last() {
                        graph.add_edge(*previous_ix, ix, Edge { order: 0 });
                    }
                    nodes.push(ix);
                }

                let mut pick = if workspace_commit_id == Some(commit.id) {
                    Pick::new_workspace_pick(commit.id)
                } else {
                    let mut pick = Pick::new_pick(commit.id);
                    pick.sign_commit = options.default_sign_commit;
                    pick
                };
                pick.mutable = mutable;
                let ix = graph.add_node(Step::Pick(pick));
                commit_to_pick_ix.insert(commit.id, ix);
                if let Some(previous_ix) = nodes.last() {
                    graph.add_edge(*previous_ix, ix, Edge { order: 0 });
                }
                nodes.push(ix);
            }

            if nodes.is_empty() {
                tracing::debug!("Empty node added - this is probably impossible");
                let ix = graph.add_node(Step::None);
                nodes.push(ix);
            }

            segments.insert(segment.id, NodeSegment { nodes });
        }

        let commit_ids = commits.iter().map(|c| c.id).collect::<HashSet<_>>();

        for c in &commits {
            let has_no_parents = c.parent_ids.is_empty();
            let missing_parent_steps = c.parent_ids.iter().any(|p| !commit_ids.contains(p));

            // If the commit has parents, but at least one of them is not
            // in the graph, this means but-graph did a partial traversal
            // and we want to preserve the commit as it is.
            if !has_no_parents && missing_parent_steps {
                let Some(idx) = commit_to_pick_ix.get(&c.id) else {
                    bail!("BUG: Listed commit does not have corresponding idx.");
                };

                let Step::Pick(pick) = &mut graph[*idx] else {
                    bail!("BUG: Listed commit does not have corresponding pick step.");
                };

                pick.preserved_parents = Some(c.parent_ids.clone());
            };
        }

        // Rebase edge order = the destination's position in the source commit's parent list.
        let parents_by_commit: HashMap<gix::ObjectId, &[gix::ObjectId]> = commits
            .iter()
            .map(|c| (c.id, c.parent_ids.as_slice()))
            .collect();

        for sidx in segments.keys() {
            let Some(source) = segments.get(sidx).and_then(|n| n.nodes.last()) else {
                continue;
            };

            // but-graph yields outgoing edges in parent order, so iterate as-is. The counter below
            // gives commit-less empty branches distinct, increasing orders — so the StepGraph never
            // has tied parent orders and needs no insertion-order tie-break.
            let edges = workspace.graph.edges_directed(*sidx, Direction::Outgoing);
            let mut empty_branch_count = 0usize;
            'inner: for edge in edges {
                let Some(target) = segments.get(&edge.target()).and_then(|n| n.nodes.first())
                else {
                    tracing::warn!(
                        "Dropping parent edge for segment {sidx:?}: edge target {:?} has no nodes",
                        edge.target()
                    );
                    continue 'inner;
                };

                // A real parent gets its index in the source commit's parent array. A dst with no
                // commit id (a commit-less empty branch) can't be indexed, so it's placed after the
                // real parents — and each one bumps the counter so siblings get distinct orders.
                let parents = edge
                    .weight()
                    .src_id()
                    .and_then(|src| parents_by_commit.get(&src).copied());
                let real_parent_index = parents
                    .zip(edge.weight().dst_id())
                    .and_then(|(parents, dst)| parents.iter().position(|p| *p == dst));
                let order = match real_parent_index {
                    Some(idx) => idx,
                    None => {
                        let o = parents.map_or(0, |p| p.len()) + empty_branch_count;
                        empty_branch_count += 1;
                        o
                    }
                };
                // Whether a parent edge lands on a Reference node or directly on a Pick is
                // load-bearing: upstream integration classifies refs by reachability from the
                // target's node, treating every ref a merge routes through as integrated history.
                // So a merge rejoining a lane from outside (the target's merge into a stack) must
                // bypass empty-branch chains spliced above its parent — those refs belong to the
                // workspace spine and must stay unreachable from the target, or integration would
                // drop them. A merge into a commit-holding named segment (the merged branch's own
                // tip) keeps its chain: that ref genuinely is integrated. The walk-era builder
                // produced this shape by accident (mis-ordered empty edges tripped the repair
                // pass below, which rewires to picks); this encodes it deliberately.
                let is_merge = edge.weight().src_id().is_some()
                    && edge.weight().src_id() != workspace_commit_id
                    && parents.is_some_and(|p| p.len() > 1);
                let bypass_to_pick = if !is_merge {
                    None
                } else if let Some(dst) = edge.weight().dst_id() {
                    let spliced_above = workspace
                        .graph
                        .edges_directed(edge.target(), Direction::Incoming)
                        .any(|e| {
                            let src = &workspace.graph[e.source()];
                            src.commits.is_empty() && src.ref_name().is_some()
                        });
                    (spliced_above
                        && workspace.graph[edge.target()].commits.first().map(|c| c.id)
                            == Some(dst))
                    .then(|| commit_to_pick_ix.get(&dst))
                    .flatten()
                } else {
                    // The edge enters an empty segment chain (the walk's inline splice): resolve
                    // through it to the first commit-holding segment below.
                    let mut tsel = edge.target();
                    let mut hops = 0usize;
                    loop {
                        if let Some(c) = workspace.graph[tsel].commits.first() {
                            break commit_to_pick_ix.get(&c.id);
                        }
                        hops += 1;
                        let next = workspace
                            .graph
                            .edges_directed(tsel, Direction::Outgoing)
                            .next()
                            .map(|e| e.target());
                        match next {
                            Some(next) if hops <= 1000 => tsel = next,
                            _ => break None,
                        }
                    }
                };
                let target = bypass_to_pick.unwrap_or(target);
                graph.add_edge(*source, *target, Edge { order });
            }
        }

        // Repair pass: a pick recreated with wrong parents gets written into rewritten
        // commits, so verify that the parents derived from the graph (resolved through
        // reference chains and empty splices) equal each commit's real parent array. On
        // mismatch, drop the pick's edges and rewire it directly to its parents' picks in
        // real order — the chain routing can't be trusted at that point. Rarely fires now
        // that empty-branch edges carry distinct orders; kept as a cheap tripwire against
        // history corruption. The workspace commit is exempt: it is rebuilt from the
        // projected lanes, not from its stored parent array.
        for c in &commits {
            if Some(c.id) == workspace_commit_id {
                continue;
            }

            let Some(&pick_ix) = commit_to_pick_ix.get(&c.id) else {
                continue;
            };

            // Skip commits with preserved parents (partial traversal — already handled above)
            if let Step::Pick(Pick {
                preserved_parents: Some(_),
                ..
            }) = &graph[pick_ix]
            {
                continue;
            }

            // Resolve what the graph thinks are the parents of this pick
            let graph_parents = util::collect_ordered_parents(&graph, pick_ix);
            let graph_parent_ids: Vec<gix::ObjectId> = graph_parents
                .iter()
                .filter_map(|idx| match &graph[*idx] {
                    Step::Pick(Pick { id, .. }) => Some(*id),
                    _ => None,
                })
                .collect();

            if graph_parent_ids == c.parent_ids {
                continue;
            }

            tracing::warn!(
                "but-graph inconsistent with the commit graph.\nParents for commit {} do not match.\n\nFound:{:?}\nExpected:{:?}\n\nThese IDs may be in memory, but may be helpful for debugging.",
                c.id,
                graph_parent_ids
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>(),
                c.parent_ids
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>(),
            );

            let outgoing_edge_ids: Vec<_> = graph
                .edges_directed(pick_ix, petgraph::Direction::Outgoing)
                .map(|e| e.id())
                .collect();
            for edge_id in outgoing_edge_ids {
                graph.remove_edge(edge_id);
            }

            'inner: for (order, parent_id) in c.parent_ids.iter().enumerate() {
                let Some(&target_ix) = commit_to_pick_ix.get(parent_id) else {
                    tracing::warn!(
                        "Dropping parent edge for commit {} (parent fix): parent {parent_id} not found in pick map",
                        c.id
                    );
                    continue 'inner;
                };

                graph.add_edge(pick_ix, target_ix, Edge { order });
            }
        }

        Ok((graph, references, head_selectors, immutable_references))
    }
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
