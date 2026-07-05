use std::collections::{HashMap, HashSet};

use anyhow::{Context as _, Result, bail};
use but_core::{RefMetadata, commit::SignCommit};
use but_graph::{Commit, SegmentIndex};

use crate::graph_rebase::{
    Checkout, Editor, Pick, RevisionHistory, Selector, Step, StepGraph, StepGraphIndex,
    SuccessfulRebase, placements, util,
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
        // `BUT_REBASE_NATIVE=assert` additionally runs the legacy segment-walk build and
        // panics on any canonical divergence (ledger AND graph); `=0` keeps the legacy graph
        // (escape hatch).
        let (graph, references, checkouts) =
            match std::env::var("BUT_REBASE_NATIVE").ok().as_deref() {
                Some("0") => create_via_segment_walk(workspace, repo, options)?,
                Some("assert") => {
                    let (old_graph, old_references, old_checkouts) =
                        create_via_segment_walk(workspace, repo, options)?;
                    let workspace_commit_id = workspace
                        .graph
                        .managed_entrypoint_commit(repo)?
                        .map(|c| c.id);
                    let extracted =
                        placements::extract(&old_graph, &old_checkouts, workspace_commit_id)?;
                    let derived = placements::derive(workspace, repo, options)?;
                    placements::assert_ledger_parity(&extracted, &derived);
                    let (native_graph, native_references, native_checkouts) =
                        create_native(workspace, repo, options, &derived)?;
                    placements::assert_native_parity(
                        &old_graph,
                        &old_checkouts,
                        &old_references,
                        &native_graph,
                        &native_checkouts,
                        &native_references,
                        workspace_commit_id,
                    )?;
                    (native_graph, native_references, native_checkouts)
                }
                _ => {
                    let ledger = placements::derive(workspace, repo, options)?;
                    create_native(workspace, repo, options, &ledger)?
                }
            };
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

/// The LEGACY editor-graph build — runs of step nodes per segment, rank-ordered edges, the
/// parent fixup, then the finalize strip. Kept only as the `BUT_REBASE_NATIVE=assert` oracle
/// counterpart and the `=0` escape hatch; production creation is native.
fn create_via_segment_walk(
    workspace: &but_graph::Workspace,
    repo: &gix::Repository,
    options: &GraphEditorOptions,
) -> Result<(StepGraph, Vec<gix::refs::FullName>, Vec<Checkout>)> {
    // This first creates runs of nodes and associates them with the
    // but-graph segments. We then do a second pass over all the segments
    // and use the but_graph to connect up the runs. Finally, we validate
    // that each Pick step's parents match the commit's actual parents,
    // and if not, we disconnect and rewire directly to the correct
    // parent commits.

    // TODO(CTO): Look into traversing "in workspace" segments that are not
    // reachable from the entrypoint TODO(CTO): Look into stopping at the
    // common base
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
                but_graph::Direction::Outgoing,
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
                let ix = graph.add_reference(refname.clone(), mutable);
                if Some(reference) == entrypoint.segment.ref_name() {
                    head_selectors.push(Selector { id: ix });
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
                    let ix = graph.add_reference(reference.clone(), mutable);
                    if let Some(previous_ix) = nodes.last() {
                        graph.push_parent(*previous_ix, ix);
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
                    graph.push_parent(*previous_ix, ix);
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

                if graph.commit_id(*idx).is_none() {
                    bail!("BUG: Listed commit does not have corresponding pick step.");
                }

                graph.set_preserved_parents(*idx, Some(c.parent_ids.clone()));
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

            // but-graph yields outgoing edges in parent order, so iterate as-is. The keys below
            // rank real parents by their index in the source commit's parent array and commit-less
            // empty branches after them (distinct, increasing) — a ranking, not final slots: the
            // ranks can have gaps (an empty branch may stand in front of a real parent, or ALL legs
            // may be commit-less refs over one base), so the sorted ranks compact by push order.
            let edges = workspace
                .graph
                .edges_directed(*sidx, but_graph::Direction::Outgoing);
            let mut empty_branch_count = 0usize;
            let mut ranked_targets = Vec::new();
            'inner: for edge in edges {
                let Some(target) = segments.get(&edge.target()).and_then(|n| n.nodes.first())
                else {
                    tracing::warn!(
                        "Dropping parent edge for segment {sidx:?}: edge target {:?} has no nodes",
                        edge.target()
                    );
                    continue 'inner;
                };

                let parents = edge
                    .weight()
                    .src_id()
                    .and_then(|src| parents_by_commit.get(&src).copied());
                let real_parent_index = parents
                    .zip(edge.weight().dst_id())
                    .and_then(|(parents, dst)| parents.iter().position(|p| *p == dst));
                let rank = match real_parent_index {
                    Some(idx) => idx,
                    None => {
                        let o = parents.map_or(0, |p| p.len()) + empty_branch_count;
                        empty_branch_count += 1;
                        o
                    }
                };
                ranked_targets.push((rank, *target));
            }
            ranked_targets.sort_by_key(|(rank, _)| *rank);
            for (_, target) in ranked_targets {
                graph.push_parent(*source, target);
            }
        }

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
            }) = graph.step_view(pick_ix)
            {
                continue;
            }

            // Resolve what the graph thinks are the parents of this pick
            let graph_parents = util::collect_ordered_parents(&graph, pick_ix);
            let graph_parent_ids: Vec<gix::ObjectId> = graph_parents
                .iter()
                .filter_map(|idx| graph.commit_id(*idx))
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

            let mut fixed_parents = Vec::with_capacity(c.parent_ids.len());
            'inner: for parent_id in &c.parent_ids {
                let Some(&target_ix) = commit_to_pick_ix.get(parent_id) else {
                    tracing::warn!(
                        "Dropping parent edge for commit {} (parent fix): parent {parent_id} not found in pick map",
                        c.id
                    );
                    continue 'inner;
                };
                fixed_parents.push(target_ix);
            }
            graph.set_parents(pick_ix, fixed_parents);
        }

        crate::graph_rebase::positions::initialize_positions_and_strip_ref_edges(&mut graph);
        crate::graph_rebase::positions::debug_assert_positions_total(&graph);

        // TODO(CTO): We need to eventually list all worktrees that we own
        // here so we can `safe_checkout` them too.
        let checkouts: Vec<Checkout> = head_selectors
            .into_iter()
            .map(|selector| Checkout::Head {
                selector,
                merge_base_override: None,
            })
            .collect();

        Ok((graph, references, checkouts))
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
