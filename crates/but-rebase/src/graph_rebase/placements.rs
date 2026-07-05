//! The REF-PLACEMENT LEDGER: everything editor creation needs to place references, addressed
//! by COMMIT ID and REF NAME instead of arena indices — the commit-addressed distillation of
//! the segment walk. Phase A extracts it off a finished (old-path) graph, phase B builds a
//! native graph from the carried `CommitGraph` plus this ledger, and the parity assert
//! compares the two in this same canonical form.

use std::collections::{BTreeMap, HashSet};

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
