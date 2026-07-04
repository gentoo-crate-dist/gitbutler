//! Refs as positioned data — the collapse's ref model, derived from the step graph.
//!
//! Today a ref is a [`Step::Reference`] NODE interposed on the path into its commit; that makes
//! refs connectivity-bearing, which is the root defect behind the merge-bypass rule and
//! stack-split gluing. The functions here express every ref's true semantics as a POSITION —
//! `Above { anchor pick, via, rank }` — derived on demand from the node graph, so consumers can
//! migrate to position semantics one at a time. When the last node reader is gone, the table
//! becomes the representation and the nodes die.
//!
//! The model is corpus-validated (see the ref-anchor brief): chains are shallow (≤3 observed),
//! a chain's approaching child is unique whenever it exists, and every chain resolves downward
//! to a pick unless the graph has none below (unborn).

use crate::graph_rebase::step_graph::ViaKind;
use crate::graph_rebase::{Direction, Step, StepGraph, StepGraphIndex};

/// Where one ref sits, expressed over commits instead of node topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RefPosition {
    /// The pick this ref resolves to, reached by descending through reference and tombstone
    /// steps. `None` only when no pick exists below (an unborn or fully emptied lane).
    pub anchor: Option<StepGraphIndex>,
    /// How many reference steps sit between this ref and its anchor (0 = directly above it).
    pub rank: usize,
    /// The picks approaching this ref's position from above with their parent-slots — several
    /// when a shared chain is entered by more than one leg. Empty for chain roots.
    pub via: Vec<(StepGraphIndex, usize)>,
    /// More than one thing (legs and/or refs stacked above) converged on this position.
    pub ambiguous: bool,
}

/// The position of the reference at `ref_node`, read from stored anchors, with the anchor
/// resolved through tombstones to the current pick.
///
/// Returns `None` if `ref_node` is not a positioned reference.
pub(crate) fn ref_position(graph: &StepGraph, ref_node: StepGraphIndex) -> Option<RefPosition> {
    let stored = graph.anchor_of(ref_node)?;
    Some(RefPosition {
        anchor: resolve_to_pick(graph, stored.anchor),
        rank: stored.rank,
        via: ref_via(graph, ref_node),
        ambiguous: stored.ambiguous,
    })
}

/// Classify a fresh `via` against the picks currently feeding its anchor into a stored
/// [`ViaKind`]: empty is a chain root, the whole leg set is [`ViaKind::AllLegs`], and any proper
/// subset is a merge lane keyed by its parent-slots. Called at write time (in `set_anchor`),
/// while the `via` still matches the live edges.
pub(crate) fn classify_via(
    via: &[(StepGraphIndex, usize)],
    legs: &[(StepGraphIndex, usize)],
) -> ViaKind {
    if via.is_empty() {
        return ViaKind::Root;
    }
    let via_set: std::collections::HashSet<_> = via.iter().copied().collect();
    let legs_set: std::collections::HashSet<_> = legs.iter().copied().collect();
    if via_set == legs_set {
        return ViaKind::AllLegs;
    }
    let mut slots: Vec<usize> = via.iter().map(|(_, slot)| *slot).collect();
    slots.sort_unstable();
    slots.dedup();
    ViaKind::Lane(slots)
}

/// Reconstruct the current `via` — the `(source-pick, slot)` legs feeding a position — from its
/// [`ViaKind`] against `anchor_pick`'s live edges. The source-pick node id is read fresh here, so
/// a leg that was tombstoned or re-slotted after the kind was authored never leaks out stale.
pub(crate) fn derive_via(
    graph: &StepGraph,
    anchor_pick: StepGraphIndex,
    kind: &ViaKind,
) -> Vec<(StepGraphIndex, usize)> {
    match kind {
        ViaKind::Root => Vec::new(),
        ViaKind::AllLegs => legs_into_pick(graph, anchor_pick),
        ViaKind::Lane(slots) => legs_into_pick(graph, anchor_pick)
            .into_iter()
            .filter(|(_, slot)| slots.contains(slot))
            .collect(),
    }
}

/// The current `via` of the reference at `node`, derived from its authored [`ViaKind`] against
/// the live pick edges — the read path that replaces reading `StoredAnchor::via` directly, so a
/// stale stored leg list never reaches a consumer.
pub(crate) fn ref_via(graph: &StepGraph, node: StepGraphIndex) -> Vec<(StepGraphIndex, usize)> {
    let (Some(stored), Some(kind)) = (graph.anchor_of(node), graph.ref_kind(node)) else {
        return Vec::new();
    };
    match resolve_to_pick(graph, stored.anchor) {
        Some(pick) => derive_via(graph, pick, &kind),
        None => Vec::new(),
    }
}

/// Derive the position of the reference at `ref_node` from CHAIN TOPOLOGY — only meaningful
/// while reference edges still exist, i.e. inside the creation finalize pass.
fn derive_ref_position_from_edges(
    graph: &StepGraph,
    ref_node: StepGraphIndex,
) -> Option<RefPosition> {
    if !matches!(graph[ref_node], Step::Reference { .. }) {
        return None;
    }
    // Descend: the anchor is the first pick below, through references and tombstones.
    let mut cursor = ref_node;
    let mut rank = 0usize;
    let mut anchor = None;
    for _ in 0..10_000 {
        let Some(next) = graph
            .edges_directed(cursor, Direction::Outgoing)
            .next()
            .map(|e| e.target())
        else {
            break;
        };
        match &graph[next] {
            Step::Pick(_) => {
                anchor = Some(next);
                break;
            }
            Step::Reference { .. } => rank += 1,
            Step::None => {}
        }
        cursor = next;
    }
    // Ascend for the approaching child: a DIRECT pick edge into this position is the entry
    // point, even when further refs are stacked above (those become their own root chain —
    // nothing descends into them). Without a direct pick, follow a unique ref/tombstone edge
    // upward; anything ambiguous means no approach (a root).
    let mut cursor = ref_node;
    let mut via = Vec::new();
    let mut ambiguous = false;
    for _ in 0..10_000 {
        let incoming: Vec<_> = graph.edges_directed(cursor, Direction::Incoming).collect();
        let picks: Vec<_> = incoming
            .iter()
            .filter(|e| matches!(graph[e.source()], Step::Pick(_)))
            .map(|e| (e.source(), e.weight().order))
            .collect();
        if !picks.is_empty() {
            // Direct pick edges are the entry points, even with refs stacked above (those
            // become their own root chain). Several legs may enter a shared chain; any
            // convergence at the entry makes the position AMBIGUOUS — it belongs to its
            // anchor, not to one leg.
            ambiguous = incoming.len() > 1;
            via = picks;
            via.sort();
            break;
        }
        let mut others = incoming
            .iter()
            .filter(|e| !matches!(graph[e.source()], Step::Pick(_)));
        match (others.next(), others.next()) {
            (Some(edge), None) => cursor = edge.source(),
            _ => break,
        }
    }
    Some(RefPosition {
        anchor,
        rank,
        via,
        ambiguous,
    })
}

/// Every reference that RESOLVES to `pick` — its stored anchor, followed through tombstones,
/// ends at it. Order is unspecified (ascending node id), like the node-walking predecessor.
pub(crate) fn refs_anchored_at(graph: &StepGraph, pick: StepGraphIndex) -> Vec<StepGraphIndex> {
    graph
        .anchored_refs()
        .filter_map(|(node, stored)| {
            (resolve_to_pick(graph, stored.anchor) == Some(pick)).then_some(node)
        })
        .collect()
}

/// The standing collapse invariant: every reference in the graph has a well-formed position,
/// and positions are unique wherever order is topologically meaningful — i.e. within chains
/// approached by a child (`via = Some`). Parallel ROOT chains above one anchor are legitimate
/// unordered siblings (found by this very assert on its first corpus run): with nothing above
/// them, their relative order is not defined by topology — the collapse orders them like the
/// passive set (by name).
///
/// Wired at editor creation AND at rebase entry, so every graph shape the suite produces —
/// including post-mutation shapes — continuously validates the position model.
pub(crate) fn debug_assert_positions_total(graph: &StepGraph) {
    if !cfg!(debug_assertions) {
        return;
    }
    type OrderedPositionKey = (Option<StepGraphIndex>, Vec<(StepGraphIndex, usize)>, usize);
    let mut seen: std::collections::HashMap<OrderedPositionKey, StepGraphIndex> =
        Default::default();
    for node in graph.node_indices() {
        if !matches!(graph[node], Step::Reference { .. }) {
            continue;
        }
        let Some(pos) = ref_position(graph, node) else {
            // A reference without a stored anchor is only legitimate when the graph holds no
            // pick below it at creation (unborn); it resolves to nothing.
            continue;
        };
        if pos.via.is_empty() {
            continue;
        }
        if let Some(previous) = seen.insert((pos.anchor, pos.via.clone(), pos.rank), node) {
            debug_assert!(
                false,
                "reference nodes {previous} and {node} collide at position {pos:?}"
            );
        }
    }
}

/// The references the node-era traversal from `start` would have walked through, given the
/// PICK set it reached: a chain is entered when one of its legs was visited (the edge from
/// leg to chain top), and when `start` is itself a reference, it and its chain below count.
pub(crate) fn refs_reachable_with(
    graph: &StepGraph,
    start: StepGraphIndex,
    picks: &std::collections::HashSet<StepGraphIndex>,
) -> Vec<StepGraphIndex> {
    // Reached commits by ID as well as node: a graph can hold one commit twice (a stack lane
    // and a target lane), and the node era's shared reference nodes made reachability
    // commit-equivalent across such lanes.
    let reached_ids: std::collections::HashSet<gix::ObjectId> = picks
        .iter()
        .filter_map(|node| match &graph[*node] {
            Step::Pick(pick) => Some(pick.id),
            _ => None,
        })
        .collect();
    let mut out = Vec::new();
    for (node, stored) in graph.anchored_refs() {
        // A chain whose anchor commit is reached lies on reached history — anchor-based
        // reachability, exactly what the node-era walk through interposed reference nodes
        // computed (and the ruling the merge-bypass deletion rests on).
        let anchor_reached = resolve_to_pick(graph, stored.anchor).is_some_and(|anchor| {
            picks.contains(&anchor)
                || match &graph[anchor] {
                    Step::Pick(pick) => reached_ids.contains(&pick.id),
                    _ => false,
                }
        });
        if anchor_reached || node == start {
            out.push(node);
        }
    }
    out
}

/// A new leg enters `ref_node`'s chain at its position: the reference and its chain-mates at
/// or below its rank gain the leg in their vias. Root chains (empty via) at one anchor are
/// distinct siblings, so only the reference itself joins.
pub(crate) fn join_chain_at(
    graph: &mut StepGraph,
    ref_node: StepGraphIndex,
    leg: (StepGraphIndex, usize),
) {
    let Some(stored) = graph.anchor_of(ref_node) else {
        return;
    };
    let joiners: Vec<_> = if stored.via.is_empty() {
        vec![(ref_node, stored.clone())]
    } else {
        chain_members(graph, ref_node)
            .into_iter()
            .filter(|(_, m)| m.rank <= stored.rank)
            .collect()
    };
    for (node, mut member) in joiners {
        member.via.push(leg);
        member.via.sort();
        member.ambiguous = member.ambiguous || member.via.len() > 1;
        graph.set_anchor(node, Some(member));
    }
}

/// Re-anchor every reference resolving to `from_pick` onto `to_pick`, keeping via and rank —
/// the position-world equivalent of interposing a node between a pick and its chains.
pub(crate) fn reanchor_refs_at(
    graph: &mut StepGraph,
    from_pick: StepGraphIndex,
    to_pick: StepGraphIndex,
) {
    let moves: Vec<_> = graph
        .anchored_refs()
        .filter_map(|(node, stored)| {
            (resolve_to_pick(graph, stored.anchor) == Some(from_pick))
                .then(|| (node, stored.clone()))
        })
        .collect();
    for (node, mut stored) in moves {
        stored.anchor = to_pick;
        graph.set_anchor(node, Some(stored));
    }
}

/// Rewrite every stored via entry equal to `old` to `new` — a rewired leg keeps carrying the
/// chains it carried. A `(pick, parent-slot)` pair identifies one edge, so this is precise.
pub(crate) fn rewrite_via_entry(
    graph: &mut StepGraph,
    old: (StepGraphIndex, usize),
    new: (StepGraphIndex, usize),
) {
    let moves: Vec<_> = graph
        .anchored_refs()
        .filter_map(|(node, stored)| stored.via.contains(&old).then(|| (node, stored.clone())))
        .collect();
    for (node, mut stored) in moves {
        for entry in &mut stored.via {
            if *entry == old {
                *entry = new;
            }
        }
        graph.set_anchor(node, Some(stored));
    }
}

/// The members of `ref_node`'s chain — every reference with the same resolved anchor and the
/// same via — with their stored positions.
pub(crate) fn chain_members(
    graph: &StepGraph,
    ref_node: StepGraphIndex,
) -> Vec<(
    StepGraphIndex,
    crate::graph_rebase::step_graph::StoredAnchor,
)> {
    let Some(stored) = graph.anchor_of(ref_node) else {
        return vec![];
    };
    let anchor = resolve_to_pick(graph, stored.anchor);
    graph
        .anchored_refs()
        .filter_map(|(node, other)| {
            (other.via == stored.via && resolve_to_pick(graph, other.anchor) == anchor)
                .then(|| (node, other.clone()))
        })
        .collect()
}

/// The legs a co-located chain on `pick` is approached by: the pick edges pointing at it,
/// as `(source, parent-slot)` pairs, sorted. Every reference co-located on one pick shares
/// this via — it is the chain's single entry, replicated across members so the renderer can
/// group them by `(anchor, via)` and order them by rank.
pub(crate) fn legs_into_pick(
    graph: &StepGraph,
    pick: StepGraphIndex,
) -> Vec<(StepGraphIndex, usize)> {
    let mut legs: Vec<_> = graph
        .edges_directed(pick, Direction::Incoming)
        .filter(|e| matches!(graph[e.source()], Step::Pick(_)))
        .map(|e| (e.source(), e.weight().order))
        .collect();
    legs.sort();
    legs
}

/// Resolve `node` to the current pick it stands for: a pick resolves to itself, a tombstone
/// follows its (preserved) first edge downward, and a reference resolves via its stored
/// anchor. During the creation finalize pass references may still carry chain edges instead
/// of anchors; those resolve by descending the chain.
pub(crate) fn resolve_to_pick(graph: &StepGraph, node: StepGraphIndex) -> Option<StepGraphIndex> {
    let mut cursor = node;
    for _ in 0..10_000 {
        match &graph[cursor] {
            Step::Pick(_) => return Some(cursor),
            Step::Reference { .. } => {
                cursor = match graph.anchor_of(cursor) {
                    Some(stored) => stored.anchor,
                    None => graph
                        .edges_directed(cursor, Direction::Outgoing)
                        .next()
                        .map(|e| e.target())?,
                };
            }
            Step::None => {
                cursor = graph
                    .edges_directed(cursor, Direction::Outgoing)
                    .next()
                    .map(|e| e.target())?;
            }
        }
    }
    None
}

/// Initialize stored anchors from the freshly built node graph, then STRIP reference edges:
/// every edge whose source is a reference is deleted; every edge whose target is a reference
/// is redirected to the pick its chain resolves to (order preserved — dup-parents chains over
/// one base yield the duplicate parent edges the real workspace commit has). After this pass,
/// edges are the truth for picks and anchors the truth for references.
pub(crate) fn initialize_anchors_and_strip_ref_edges(graph: &mut StepGraph) {
    // Derive every reference's position from the chain topology while it still exists.
    let mut anchors = Vec::new();
    for node in graph.node_indices() {
        if !matches!(graph[node], Step::Reference { .. }) {
            continue;
        }
        let Some(pos) = derive_ref_position_from_edges(graph, node) else {
            continue;
        };
        let Some(anchor) = pos.anchor else {
            // A chain with no pick below (unborn) keeps no stored anchor; it resolves to
            // nothing, like today.
            continue;
        };
        anchors.push((
            node,
            crate::graph_rebase::step_graph::StoredAnchor {
                anchor,
                via: pos.via.clone(),
                rank: pos.rank,
                ambiguous: pos.ambiguous,
            },
        ));
    }
    for (node, anchor) in anchors {
        graph.set_anchor(node, Some(anchor));
    }
    // Strip: collect the full edge picture first, then rewrite.
    let mut to_remove = Vec::new();
    let mut to_add = Vec::new();
    for edge in graph.edge_references() {
        let source_is_ref = matches!(graph[edge.source()], Step::Reference { .. });
        let target_is_ref = matches!(graph[edge.target()], Step::Reference { .. });
        if source_is_ref {
            to_remove.push(edge.id());
        } else if target_is_ref {
            to_remove.push(edge.id());
            if let Some(pick) = resolve_to_pick(graph, edge.target()) {
                to_add.push((edge.source(), pick, edge.weight().clone()));
            }
        }
    }
    for id in to_remove {
        graph.remove_edge(id);
    }
    for (source, target, weight) in to_add {
        graph.add_edge(source, target, weight);
    }
    // Anchors were set above against the pre-strip topology (chain legs still targeted the ref
    // nodes); now that edges point straight at the picks, re-author every kind against the final
    // legs so `AllLegs`/`Lane` reflect the stripped graph.
    graph.reauthor_ref_kinds();
}
