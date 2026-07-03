//! Editor-graph construction from the COMMIT graph plus the workspace projection — no segment
//! iteration. Developed beside `creation.rs` and held to canonical equality (see `canonical.rs`)
//! before it replaces the segment-graph walk.
//!
//! Sources: the [`but_graph::CommitGraph`] (commits, ordered parents, refs-on-commits, connected
//! edges) provides picks, ref chains, and parent edges; [`Workspace::stacks`] provides what the
//! commit graph cannot know — empty branches and their splice positions in each lane.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};
use but_core::RefMetadata;

use crate::graph_rebase::{
    Edge, ExtraRefMutability, GraphEditorOptions, Pick, Selector, Step, StepGraph, StepGraphIndex,
};

pub(crate) struct NativeParts {
    pub graph: StepGraph,
    pub references: Vec<gix::refs::FullName>,
    pub head_selectors: Vec<Selector>,
    pub immutable_references: HashSet<gix::refs::FullName>,
}

/// An empty-branch run spliced between `above` and `below` in a lane. The refs themselves live
/// on the `below` commit in the CommitGraph (the bridge rests empty-segment refs there); the
/// splice contributes the lane EDGE — its own parent slot and order — not the chain.
struct Splice {
    /// The commit above the run; `None` = an unanchored lane top.
    above: Option<gix::ObjectId>,
    /// The commit the run rests on, if any.
    below: Option<gix::ObjectId>,
    /// The topmost ref of the run — identifies the resting chain the lane edge lands on.
    top_ref: gix::refs::FullName,
}

pub(crate) fn create_native<M: RefMetadata>(
    workspace: &but_graph::Workspace,
    repo: &gix::Repository,
    options: &GraphEditorOptions,
    _meta: &M,
) -> Result<NativeParts> {
    let cg = but_graph::CommitGraph::from_segment_graph(&workspace.graph);
    let entrypoint = workspace.graph.entrypoint()?;
    let entrypoint_ref = entrypoint.segment.ref_name().map(|r| r.to_owned());
    let workspace_commit_id = workspace
        .graph
        .managed_entrypoint_commit(repo)?
        .map(|c| c.id);

    // ── Scope: commits reachable from the entry commits over CONNECTED parent edges ──
    let entry_commit = entrypoint.commit_and_owner.map(|(c, _)| c.id).or_else(|| {
        // Empty entrypoint segment: enter at the first commit below it.
        workspace
            .graph
            .tip_skip_empty(entrypoint.segment.id)
            .map(|c| c.id)
    });
    let mut mutable_entries: Vec<gix::ObjectId> = entry_commit.into_iter().collect();
    let mut immutable_entries: Vec<gix::ObjectId> = Vec::new();
    // A traversal ENTRY whose own segment is empty (the entrypoint, an extra ref) is on the
    // walk by definition — its resting chain is included even when nothing feeds it.
    let mut required_chain_refs: HashSet<gix::refs::FullName> = HashSet::new();
    let mut resting_immutable: Vec<gix::refs::FullName> = Vec::new();
    if entrypoint.segment.commits.is_empty()
        && let Some(rn) = entrypoint_ref.as_ref()
    {
        required_chain_refs.insert(rn.clone());
    }
    for extra_ref in &options.extra_refs {
        let Some((segment, commit)) = workspace
            .graph
            .segment_and_commit_by_ref_name(extra_ref.ref_name)
        else {
            bail!(
                "Failed to find corresponding segment for {}",
                extra_ref.ref_name
            );
        };
        if segment.commits.is_empty() {
            required_chain_refs.insert(extra_ref.ref_name.to_owned());
            if extra_ref.mutability == ExtraRefMutability::Immutable {
                resting_immutable.push(extra_ref.ref_name.to_owned());
            }
        }
        match extra_ref.mutability {
            ExtraRefMutability::Mutable => mutable_entries.push(commit.id),
            ExtraRefMutability::Immutable => immutable_entries.push(commit.id),
        }
    }

    let bfs = |entries: &[gix::ObjectId], seen: &mut HashSet<gix::ObjectId>| {
        let mut stack: Vec<_> = entries.to_vec();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            stack.extend(cg.connected_parents(id));
        }
    };
    let mut scope = HashSet::new();
    bfs(&mutable_entries, &mut scope);
    let mutable_scope = scope.clone();
    bfs(&immutable_entries, &mut scope);
    let immutable_scope: HashSet<_> = scope.difference(&mutable_scope).copied().collect();

    // ── Empty-branch splices, from the projection's lanes ──
    let mut splices: Vec<Splice> = Vec::new();
    for stack in &workspace.stacks {
        // A lane-top run anchors on the workspace commit.
        let mut above: Option<gix::ObjectId> = workspace_commit_id;
        let mut run_top: Option<gix::refs::FullName> = None;
        let mut lane_reachable = workspace_commit_id.is_some_and(|ws| scope.contains(&ws));
        for segment in &stack.segments {
            if segment.commits.is_empty() {
                if run_top.is_none() {
                    run_top = segment.ref_info.as_ref().map(|ri| ri.ref_name.clone());
                }
            } else {
                let tip = segment.commits[0].id;
                lane_reachable |= scope.contains(&tip);
                if let Some(top_ref) = run_top.take() {
                    splices.push(Splice {
                        above,
                        below: Some(tip),
                        top_ref,
                    });
                }
                above = segment.commits.last().map(|c| c.id);
            }
        }
        if let Some(top_ref) = run_top
            && lane_reachable
        {
            // Run at the lane bottom: rests on the stack base, if any.
            let below = stack.segments.last().and_then(|s| s.base);
            splices.push(Splice {
                above,
                below,
                top_ref,
            });
        }
    }
    // Only splices on reachable anchors participate.
    splices.retain(|s| {
        s.above.map(|a| scope.contains(&a)).unwrap_or(true)
            && s.below.map(|b| scope.contains(&b)).unwrap_or(true)
    });
    // One parent SLOT is consumed per splice (dup-parent lanes may route several slots to the
    // same below commit — only as many as there are splices divert through chains).
    let mut spliced_slots: HashMap<(Option<gix::ObjectId>, gix::ObjectId), usize> = HashMap::new();
    for s_ in &splices {
        if let Some(b) = s_.below {
            *spliced_slots.entry((s_.above, b)).or_default() += 1;
        }
    }

    // ── Emission ──
    let mut graph = StepGraph::new();
    let mut references = Vec::new();
    let mut head_selectors = Vec::new();
    let mut immutable_references: HashSet<gix::refs::FullName> =
        resting_immutable.into_iter().collect();
    let mut pick_of: HashMap<gix::ObjectId, StepGraphIndex> = HashMap::new();
    let mut chain_top_of: HashMap<gix::ObjectId, StepGraphIndex> = HashMap::new();

    let in_order: Vec<gix::ObjectId> = cg.commit_ids().filter(|id| scope.contains(id)).collect();
    for &id in &in_order {
        let mut prev: Option<StepGraphIndex> = None;
        for refname in cg.refs_at(id) {
            references.push(refname.clone());
            if immutable_scope.contains(&id) {
                immutable_references.insert(refname.clone());
            }
            let ix = graph.add_node(Step::Reference {
                refname: refname.clone(),
            });
            if Some(&refname) == entrypoint_ref.as_ref() {
                head_selectors.push(Selector {
                    id: ix,
                    revision: 0,
                });
            }
            if let Some(prev) = prev {
                graph.add_edge(prev, ix, Edge { order: 0 });
            }
            chain_top_of.entry(id).or_insert(ix);
            prev = Some(ix);
        }
        let pick = if workspace_commit_id == Some(id) {
            Pick::new_workspace_pick(id)
        } else {
            let mut pick = Pick::new_pick(id);
            pick.sign_commit = options.default_sign_commit;
            pick
        };
        let ix = graph.add_node(Step::Pick(pick));
        if let Some(prev) = prev {
            graph.add_edge(prev, ix, Edge { order: 0 });
        }
        pick_of.insert(id, ix);
        chain_top_of.entry(id).or_insert(ix);
    }

    // Resting chains: refs living on empty spliced segments, one Reference run per chain,
    // resting on its below commit's own chain (contraction merges them when that commit is
    // named — the old graph's Ref→Ref edge — and keeps them separate over anonymous commits).
    let mut chain_head_by_top_ref: HashMap<gix::refs::FullName, StepGraphIndex> = HashMap::new();
    for chain in cg.resting_chains() {
        let included = chain.fed || chain.refs.iter().any(|r| required_chain_refs.contains(r));
        let below_in_scope = chain.below.map(|b| scope.contains(&b)).unwrap_or(false);
        if !included || !below_in_scope || chain.refs.is_empty() {
            continue;
        }
        let mut prev: Option<StepGraphIndex> = None;
        for refname in &chain.refs {
            references.push(refname.clone());
            let ix = graph.add_node(Step::Reference {
                refname: refname.clone(),
            });
            if Some(refname) == entrypoint_ref.as_ref() {
                head_selectors.push(Selector {
                    id: ix,
                    revision: 0,
                });
            }
            if let Some(prev) = prev {
                graph.add_edge(prev, ix, Edge { order: 0 });
            }
            if prev.is_none() {
                chain_head_by_top_ref.insert(refname.clone(), ix);
            }
            prev = Some(ix);
        }
        if let (Some(prev), Some(below)) = (prev, chain.below)
            && let Some(&target) = chain_top_of.get(&below)
        {
            graph.add_edge(prev, target, Edge { order: 0 });
        }
    }

    // Parent edges for every commit; the workspace commit's spliced lanes divert below.
    for &id in &in_order {
        let parents = cg.raw_parent_ids(id);
        let is_merge = parents.len() > 1 && Some(id) != workspace_commit_id;
        let has_missing = parents.iter().any(|p| !scope.contains(p));
        // A parent link the display graph SEVERED while the commit asserts it (an
        // integrated-local kept visible, a display cut): the graph routing can't be trusted
        // for this commit — wire every parent directly to its pick, in real order. This is
        // the old builder's repair pass, encoded deliberately.
        let severed = !has_missing
            && Some(id) != workspace_commit_id
            && parents.iter().any(|p| !cg.is_connected_pair(id, *p));
        let mut empty_count = 0usize;
        for (order, parent) in parents.iter().enumerate() {
            if !scope.contains(parent) {
                continue;
            }
            if !severed
                && let Some(n) = spliced_slots
                    .get_mut(&(Some(id), *parent))
                    .filter(|n| **n > 0)
            {
                // The splice emission below carries this connection.
                *n -= 1;
                continue;
            }
            let target = if severed || (is_merge && cg.is_merge_bypass_commit(*parent)) {
                // Severed: see above. Merge: a merge rejoining a lane must bypass the empty
                // chains spliced above its parent — their refs belong to the workspace spine
                // and must stay unreachable from the merge (upstream-integration
                // classification).
                pick_of[parent]
            } else {
                chain_top_of[parent]
            };
            graph.add_edge(pick_of[&id], target, Edge { order });
        }
        // Splices under this commit: lane edges placed after the real parent slots, landing on
        // the lane's own resting chain.
        for splice in splices.iter().filter(|s| s.above == Some(id)) {
            let order = parents.len() + empty_count;
            empty_count += 1;
            let Some(&target) = chain_head_by_top_ref.get(&splice.top_ref) else {
                continue;
            };
            graph.add_edge(pick_of[&id], target, Edge { order });
        }
        if has_missing
            && !parents.is_empty()
            && let Step::Pick(p) = &mut graph[pick_of[&id]]
        {
            p.preserved_parents = Some(parents);
        }
    }

    Ok(NativeParts {
        graph,
        references,
        head_selectors,
        immutable_references,
    })
}
