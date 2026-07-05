//! The REF-PLACEMENT LEDGER: everything editor creation needs to place references, addressed
//! by COMMIT ID and REF NAME instead of arena indices — the commit-addressed distillation of
//! the segment walk. Phase A extracts it off a finished (old-path) graph, phase B builds a
//! native graph from the carried `CommitGraph` plus this ledger, and the parity assert
//! compares the two in this same canonical form.

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Context, Result, bail};

use crate::graph_rebase::{
    Checkout, Step, StepGraph,
    positions::{ref_approach, resolve_to_pick},
};

/// One reference in canonical (commit/name-addressed) form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlacedRef {
    /// The full reference name.
    pub name: gix::refs::FullName,
    /// Whether the rebase may move this reference.
    pub mutable: bool,
    /// The commit the reference sits on; `None` for unborn refs (no stored position).
    pub anchor: Option<gix::ObjectId>,
    /// The name of the reference directly underneath in the physical stack.
    pub below: Option<gix::refs::FullName>,
    /// The stored convergence signal (see `RefPosition::ambiguous`).
    pub ambiguous: bool,
    /// The approach legs as `(source commit, parent-slot)`, sorted.
    pub approach: Vec<(gix::ObjectId, usize)>,
}

/// The full ledger: refs in arena order (which IS the segment-walk insertion order — the
/// native build re-adds them in this order to preserve ref indices and render sibling order),
/// plus what creation derives alongside them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RefPlacements {
    /// All live references in ref-arena order.
    pub refs: Vec<PlacedRef>,
    /// Commits whose picks are mutable (reachable from a mutable entrypoint).
    pub mutable_commits: HashSet<gix::ObjectId>,
    /// The reference names the HEAD checkouts point at, in checkout order.
    pub head_refs: Vec<gix::refs::FullName>,
    /// The managed workspace commit's parent SLOTS — one per workspace lane, so empty lanes
    /// over one base yield duplicate entries the real commit does not have. Lane data, not
    /// commit data: creation deliberately keeps the segment wiring here (the parent-fixup
    /// pass skips the ws commit).
    pub ws_parents: Option<Vec<gix::ObjectId>>,
}

/// Extract the ledger off a finished creation graph. Only valid right after creation: every
/// reference is live and every checkout selector points at a reference.
pub(crate) fn extract(
    graph: &StepGraph,
    checkouts: &[Checkout],
    workspace_commit_id: Option<gix::ObjectId>,
) -> Result<RefPlacements> {
    let mut refs = Vec::new();
    for (node, name, mutable) in graph.references() {
        let mut anchor = None;
        let mut below = None;
        let mut ambiguous = false;
        let mut approach = Vec::new();
        if let Some(stored) = graph.position_of(node) {
            let anchor_pick = resolve_to_pick(graph, stored.anchor)
                .context("positioned ref must resolve to a pick at creation")?;
            anchor = Some(
                graph
                    .commit_id(anchor_pick)
                    .context("anchor pick must carry a commit id")?,
            );
            below = match stored.below {
                Some(b) => Some(
                    graph
                        .reference(b)
                        .map(|(name, _)| name.to_owned())
                        .context("below must name a live reference at creation")?,
                ),
                None => None,
            };
            ambiguous = stored.ambiguous;
            for (source, slot) in ref_approach(graph, node) {
                let id = graph
                    .commit_id(source)
                    .context("approach leg source must be a pick")?;
                approach.push((id, slot));
            }
            approach.sort_unstable();
        }
        refs.push(PlacedRef {
            name: name.to_owned(),
            mutable,
            anchor,
            below,
            ambiguous,
            approach,
        });
    }

    let mut mutable_commits = HashSet::new();
    for node in graph.node_indices() {
        if let Step::Pick(pick) = graph.step_view(node)
            && pick.mutable
        {
            mutable_commits.insert(pick.id);
        }
    }

    let mut head_refs = Vec::new();
    for checkout in checkouts {
        let Checkout::Head { selector, .. } = checkout;
        let Some((name, _)) = graph.reference(selector.id) else {
            bail!("creation checkout selector must point at a live reference");
        };
        head_refs.push(name.to_owned());
    }

    let mut ws_parents = None;
    if let Some(ws_id) = workspace_commit_id
        && let Some(ws_pick) = graph
            .node_indices()
            .find(|&node| graph.commit_id(node) == Some(ws_id))
    {
        let mut parents = Vec::new();
        for parent in graph.parents(ws_pick) {
            parents.push(
                graph
                    .commit_id(*parent)
                    .context("ws pick parents must be picks after creation")?,
            );
        }
        ws_parents = Some(parents);
    }

    Ok(RefPlacements {
        refs,
        mutable_commits,
        head_refs,
        ws_parents,
    })
}

/// Derive the ledger STRAIGHT from the segment graph — no step-graph intermediate. This
/// mirrors the segment walk's semantics exactly, on a throwaway IR: per-segment runs
/// (segment ref, then per commit its refs then the commit), rank-ordered inter-segment
/// edges, the parent fixup (a commit whose chain-flattened parents disagree with its raw
/// parent list is rewired directly, bypassing chains — the ws commit and partially-traversed
/// commits keep their wiring), position derivation, and the strip's slot compaction.
pub(crate) fn derive(
    workspace: &but_graph::Workspace,
    repo: &gix::Repository,
    options: &crate::graph_rebase::GraphEditorOptions,
) -> Result<RefPlacements> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum IrStep {
        /// Index into `ref_table`.
        Ref(usize),
        /// Index into `commit_table`.
        Commit(usize),
        /// The placeholder for a segment with neither name nor commits.
        None,
    }

    let graph = &workspace.graph;
    let entrypoint = graph.entrypoint()?;

    let mut mutable_entrypoints = vec![entrypoint.segment.id];
    for ref_name in &options.extra_mutable_refs {
        let Some((segment, _)) = graph.segment_and_commit_by_ref_name(ref_name.as_ref()) else {
            bail!("Failed to find corresponding segment for {ref_name}");
        };
        mutable_entrypoints.push(segment.id);
    }
    let mut mutable_segments = HashSet::new();
    for ep in mutable_entrypoints {
        graph.visit_all_segments_including_start_until(
            ep,
            but_graph::Direction::Outgoing,
            |segment| !mutable_segments.insert(segment.id),
        );
    }

    let workspace_commit_id = graph.managed_entrypoint_commit(repo)?.map(|c| c.id);

    // IR build: one run of nodes per segment, in `graph.segments()` order — which fixes the
    // ledger's ref order.
    let mut nodes: Vec<IrStep> = Vec::new();
    let mut parents: Vec<Vec<usize>> = Vec::new();
    let mut ref_table: Vec<(gix::refs::FullName, bool)> = Vec::new();
    let mut commit_table: Vec<(gix::ObjectId, Vec<gix::ObjectId>)> = Vec::new();
    let mut commit_node = HashMap::<gix::ObjectId, usize>::new();
    let mut mutable_commits = HashSet::new();
    let mut head_refs = Vec::new();
    let mut runs = Vec::new();

    for sid in graph.segments() {
        let segment = &graph[sid];
        let mutable = mutable_segments.contains(&sid);
        let mut run: Vec<usize> = vec![];
        let push = |nodes: &mut Vec<IrStep>, parents: &mut Vec<Vec<usize>>, step| {
            nodes.push(step);
            parents.push(vec![]);
            nodes.len() - 1
        };

        if let Some(reference) = segment.ref_name() {
            if Some(reference) == entrypoint.segment.ref_name() {
                head_refs.push(reference.to_owned());
            }
            ref_table.push((reference.to_owned(), mutable));
            let n = push(&mut nodes, &mut parents, IrStep::Ref(ref_table.len() - 1));
            run.push(n);
        }
        for commit in &segment.commits {
            if mutable {
                mutable_commits.insert(commit.id);
            }
            for r in &commit.refs {
                ref_table.push((r.ref_name.clone(), mutable));
                let n = push(&mut nodes, &mut parents, IrStep::Ref(ref_table.len() - 1));
                if let Some(&previous) = run.last() {
                    parents[previous].push(n);
                }
                run.push(n);
            }
            commit_table.push((commit.id, commit.parent_ids.clone()));
            let n = push(
                &mut nodes,
                &mut parents,
                IrStep::Commit(commit_table.len() - 1),
            );
            commit_node.insert(commit.id, n);
            if let Some(&previous) = run.last() {
                parents[previous].push(n);
            }
            run.push(n);
        }
        if run.is_empty() {
            run.push(push(&mut nodes, &mut parents, IrStep::None));
        }
        runs.push((sid, run));
    }

    // Rank-ordered inter-segment edges onto each run's LAST node: real parents by their index
    // in the source commit's parent array, commit-less legs after them in edge order, ranks
    // compacted by push order.
    let parents_by_commit: HashMap<gix::ObjectId, &[gix::ObjectId]> = commit_table
        .iter()
        .map(|(id, parent_ids)| (*id, parent_ids.as_slice()))
        .collect();
    let first_node_of_segment: HashMap<but_graph::SegmentIndex, usize> = runs
        .iter()
        .map(|(sid, run)| (*sid, *run.first().expect("every run has a node")))
        .collect();
    for (sid, run) in &runs {
        let source = *run.last().expect("every run has a node");
        let mut empty_branch_count = 0usize;
        let mut ranked_targets = Vec::new();
        for edge in graph.edges_directed(*sid, but_graph::Direction::Outgoing) {
            let Some(&target) = first_node_of_segment.get(&edge.target()) else {
                continue;
            };
            let edge_parents = edge
                .weight()
                .src_id()
                .and_then(|src| parents_by_commit.get(&src).copied());
            let real_parent_index = edge_parents
                .zip(edge.weight().dst_id())
                .and_then(|(parents, dst)| parents.iter().position(|p| *p == dst));
            let rank = match real_parent_index {
                Some(idx) => idx,
                None => {
                    let o = edge_parents.map_or(0, |p| p.len()) + empty_branch_count;
                    empty_branch_count += 1;
                    o
                }
            };
            ranked_targets.push((rank, target));
        }
        ranked_targets.sort_by_key(|(rank, _)| *rank);
        for (_, target) in ranked_targets {
            parents[source].push(target);
        }
    }

    // The fixup: flatten a commit's chain parents in slot order; on disagreement with the
    // RAW parent list, rewire directly to present commits (chains lose their legs). The ws
    // commit and partially-traversed commits keep their segment wiring.
    let commit_ids: HashSet<gix::ObjectId> = commit_table.iter().map(|(id, _)| *id).collect();
    let flatten = |nodes: &[IrStep], parents: &[Vec<usize>], start: usize| {
        let mut out = Vec::new();
        let mut stack: Vec<usize> = parents[start].iter().rev().copied().collect();
        while let Some(n) = stack.pop() {
            match nodes[n] {
                IrStep::Commit(c) => out.push(c),
                IrStep::Ref(_) | IrStep::None => {
                    stack.extend(parents[n].iter().rev().copied());
                }
            }
        }
        out
    };
    for (id, raw_parents) in &commit_table {
        if Some(*id) == workspace_commit_id {
            continue;
        }
        let preserved =
            !raw_parents.is_empty() && raw_parents.iter().any(|p| !commit_ids.contains(p));
        if preserved {
            continue;
        }
        let n = commit_node[id];
        let flat_ids: Vec<gix::ObjectId> = flatten(&nodes, &parents, n)
            .into_iter()
            .map(|c| commit_table[c].0)
            .collect();
        if flat_ids == *raw_parents {
            continue;
        }
        parents[n] = raw_parents
            .iter()
            .filter_map(|p| commit_node.get(p).copied())
            .collect();
    }

    // Positions from the (post-fixup, pre-strip) topology: descend first-edges for anchor and
    // below, ascend for the approach legs and the convergence signal.
    let mut incoming: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nodes.len()];
    for (child, slots) in parents.iter().enumerate() {
        for (slot, &parent) in slots.iter().enumerate() {
            incoming[parent].push((child, slot));
        }
    }
    let is_commit = |n: usize| matches!(nodes[n], IrStep::Commit(_));
    let ref_nodes: Vec<(usize, usize)> = nodes
        .iter()
        .enumerate()
        .filter_map(|(n, step)| match step {
            IrStep::Ref(r) => Some((n, *r)),
            _ => None,
        })
        .collect();
    struct DerivedPosition {
        anchor: usize,
        below: Option<usize>,
        ambiguous: bool,
        approach: Vec<(usize, usize)>,
    }
    let mut positions = HashMap::<usize, DerivedPosition>::new();
    for &(ref_node, _) in &ref_nodes {
        let mut cursor = ref_node;
        let mut anchor = None;
        let mut below = None;
        for _ in 0..10_000 {
            let Some(&next) = parents[cursor].first() else {
                break;
            };
            if is_commit(next) {
                anchor = Some(next);
                break;
            }
            if matches!(nodes[next], IrStep::Ref(_)) && below.is_none() {
                below = Some(next);
            }
            cursor = next;
        }
        let Some(anchor) = anchor else {
            continue; // unborn: no stored position
        };
        let mut cursor = ref_node;
        let mut approach = Vec::new();
        let mut ambiguous = false;
        for _ in 0..10_000 {
            let entering = &incoming[cursor];
            let picks: Vec<_> = entering
                .iter()
                .copied()
                .filter(|&(child, _)| is_commit(child))
                .collect();
            if !picks.is_empty() {
                ambiguous = entering.len() > 1;
                approach = picks;
                break;
            }
            let mut others = entering.iter().filter(|&&(child, _)| !is_commit(child));
            match (others.next(), others.next()) {
                (Some(&(child, _)), None) => cursor = child,
                _ => break,
            }
        }
        positions.insert(
            ref_node,
            DerivedPosition {
                anchor,
                below,
                ambiguous,
                approach,
            },
        );
    }

    // The strip's slot compaction: resolve each commit's parent entries to commits (dropping
    // unborn chains), record the vacated slots, and rename the captured approach legs.
    let resolve = |start: usize| -> Option<usize> {
        let mut cursor = start;
        for _ in 0..10_000 {
            if is_commit(cursor) {
                return Some(cursor);
            }
            cursor = *parents[cursor].first()?;
        }
        None
    };
    let mut dropped: Vec<(usize, usize)> = Vec::new();
    let mut final_parents = HashMap::<gix::ObjectId, Vec<gix::ObjectId>>::new();
    for (id, _) in &commit_table {
        let n = commit_node[id];
        let mut resolved = Vec::with_capacity(parents[n].len());
        for (slot, &parent) in parents[n].iter().enumerate() {
            match resolve(parent) {
                Some(pick) => {
                    let IrStep::Commit(c) = nodes[pick] else {
                        unreachable!("resolve returns commits");
                    };
                    resolved.push(commit_table[c].0);
                }
                None => dropped.push((n, slot)),
            }
        }
        final_parents.insert(*id, resolved);
    }
    for position in positions.values_mut() {
        for (leg_source, slot) in position.approach.iter_mut() {
            *slot -= dropped
                .iter()
                .filter(|(source, vacated)| source == leg_source && vacated < slot)
                .count();
        }
    }

    // Emit, in ref-table order (= the step-graph ref arena order).
    let node_of_ref: HashMap<usize, usize> = ref_nodes.iter().map(|&(n, r)| (r, n)).collect();
    let mut refs = Vec::with_capacity(ref_table.len());
    for (r, (name, mutable)) in ref_table.iter().enumerate() {
        let ref_node = node_of_ref[&r];
        let mut anchor = None;
        let mut below = None;
        let mut ambiguous = false;
        let mut approach = Vec::new();
        if let Some(position) = positions.get(&ref_node) {
            let IrStep::Commit(c) = nodes[position.anchor] else {
                unreachable!("anchors are commits");
            };
            anchor = Some(commit_table[c].0);
            below = position.below.map(|b| {
                let IrStep::Ref(br) = nodes[b] else {
                    unreachable!("below entries are refs");
                };
                ref_table[br].0.clone()
            });
            ambiguous = position.ambiguous;
            for &(child, slot) in &position.approach {
                let IrStep::Commit(c) = nodes[child] else {
                    unreachable!("approach legs come from commits");
                };
                approach.push((commit_table[c].0, slot));
            }
            approach.sort_unstable();
        }
        refs.push(PlacedRef {
            name: name.clone(),
            mutable: *mutable,
            anchor,
            below,
            ambiguous,
            approach,
        });
    }

    let ws_parents = workspace_commit_id.and_then(|id| final_parents.get(&id).cloned());

    Ok(RefPlacements {
        refs,
        mutable_commits,
        head_refs,
        ws_parents,
    })
}

/// The ledger parity oracle: compare the ledger EXTRACTED off the finished segment-walk graph
/// against the one DERIVED straight from the segment graph, and PANIC with precise diffs on
/// any divergence. Run under `BUT_REBASE_NATIVE=assert`.
pub(crate) fn assert_ledger_parity(extracted: &RefPlacements, derived: &RefPlacements) {
    if extracted == derived {
        return;
    }
    let extracted_only: Vec<_> = extracted
        .refs
        .iter()
        .filter(|r| !derived.refs.contains(r))
        .collect();
    let derived_only: Vec<_> = derived
        .refs
        .iter()
        .filter(|r| !extracted.refs.contains(r))
        .collect();
    panic!(
        "LEDGER DERIVATION DIVERGENCE\nextracted-only/differing: {extracted_only:#?}\nderived-only/differing: {derived_only:#?}\nheads: extracted {:?} derived {:?}\nws_parents: extracted {:?} derived {:?}\nmutable-commit delta: extracted-only {:?} derived-only {:?}\nref order: extracted {:?} derived {:?}",
        extracted.head_refs,
        derived.head_refs,
        extracted.ws_parents,
        derived.ws_parents,
        extracted
            .mutable_commits
            .difference(&derived.mutable_commits)
            .collect::<Vec<_>>(),
        derived
            .mutable_commits
            .difference(&extracted.mutable_commits)
            .collect::<Vec<_>>(),
        extracted
            .refs
            .iter()
            .map(|r| r.name.to_string())
            .collect::<Vec<_>>(),
        derived
            .refs
            .iter()
            .map(|r| r.name.to_string())
            .collect::<Vec<_>>(),
    );
}

/// The canonical form of the pick arena: per commit id, the ordered parent commit ids and the
/// full pick payload (as its debug form — `PickSettings` has no `PartialEq`, and the debug
/// string covers every field).
fn canonical_picks(
    graph: &StepGraph,
) -> Result<BTreeMap<gix::ObjectId, (Vec<gix::ObjectId>, String)>> {
    let mut picks = BTreeMap::new();
    for node in graph.node_indices() {
        let Step::Pick(pick) = graph.step_view(node) else {
            continue;
        };
        let mut parent_ids = Vec::new();
        for parent in graph.parents(node) {
            parent_ids.push(
                graph
                    .commit_id(*parent)
                    .context("pick parents must be picks after creation")?,
            );
        }
        if picks
            .insert(pick.id, (parent_ids, format!("{pick:?}")))
            .is_some()
        {
            bail!("duplicate pick for commit {}", pick.id);
        }
    }
    Ok(picks)
}

/// The dual-build parity oracle: compare an old-path creation against a native one in
/// canonical form and PANIC with precise diffs on any divergence. Run under
/// `BUT_REBASE_NATIVE=assert`.
pub(crate) fn assert_native_parity(
    old_graph: &StepGraph,
    old_checkouts: &[Checkout],
    old_initial_references: &[gix::refs::FullName],
    native_graph: &StepGraph,
    native_checkouts: &[Checkout],
    native_initial_references: &[gix::refs::FullName],
    workspace_commit_id: Option<gix::ObjectId>,
) -> Result<()> {
    let old_picks = canonical_picks(old_graph)?;
    let native_picks = canonical_picks(native_graph)?;
    if old_picks != native_picks {
        let old_only: Vec<_> = old_picks
            .iter()
            .filter(|(id, v)| native_picks.get(*id) != Some(v))
            .collect();
        let native_only: Vec<_> = native_picks
            .iter()
            .filter(|(id, v)| old_picks.get(*id) != Some(v))
            .collect();
        panic!(
            "NATIVE CREATION DIVERGENCE (picks)\nold-only/differing: {old_only:#?}\nnative-only/differing: {native_only:#?}"
        );
    }

    let old_ledger = extract(old_graph, old_checkouts, workspace_commit_id)?;
    let native_ledger = extract(native_graph, native_checkouts, workspace_commit_id)?;
    if old_ledger != native_ledger {
        let old_only: Vec<_> = old_ledger
            .refs
            .iter()
            .filter(|r| !native_ledger.refs.contains(r))
            .collect();
        let native_only: Vec<_> = native_ledger
            .refs
            .iter()
            .filter(|r| !old_ledger.refs.contains(r))
            .collect();
        panic!(
            "NATIVE CREATION DIVERGENCE (refs)\nold-only/differing: {old_only:#?}\nnative-only/differing: {native_only:#?}\nold heads: {:?} native heads: {:?}\nmutable-commit delta: old-only {:?} native-only {:?}\nref order: old {:?} native {:?}",
            old_ledger.head_refs,
            native_ledger.head_refs,
            old_ledger
                .mutable_commits
                .difference(&native_ledger.mutable_commits)
                .collect::<Vec<_>>(),
            native_ledger
                .mutable_commits
                .difference(&old_ledger.mutable_commits)
                .collect::<Vec<_>>(),
            old_ledger
                .refs
                .iter()
                .map(|r| r.name.to_string())
                .collect::<Vec<_>>(),
            native_ledger
                .refs
                .iter()
                .map(|r| r.name.to_string())
                .collect::<Vec<_>>(),
        );
    }

    if old_initial_references != native_initial_references {
        panic!(
            "NATIVE CREATION DIVERGENCE (initial references)\nold: {old_initial_references:?}\nnative: {native_initial_references:?}"
        );
    }

    // Ref-arena index parity: selectors and render sibling order lean on ref indices, so the
    // native build must reproduce them exactly, not just the same set.
    let old_ref_order: Vec<_> = old_graph
        .references()
        .map(|(ix, name, _)| (ix, name.to_owned()))
        .collect();
    let native_ref_order: Vec<_> = native_graph
        .references()
        .map(|(ix, name, _)| (ix, name.to_owned()))
        .collect();
    if old_ref_order != native_ref_order {
        panic!(
            "NATIVE CREATION DIVERGENCE (ref arena order)\nold: {old_ref_order:?}\nnative: {native_ref_order:?}"
        );
    }

    Ok(())
}
