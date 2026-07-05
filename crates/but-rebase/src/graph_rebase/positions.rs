//! Where each reference sits, stored as position data rather than as graph edges.
//!
//! A commit ([`Step::Pick`]) carries parent edges; a reference ([`Step::Reference`]) carries
//! NONE. Instead every reference has a [`StoredAnchor`] in a side-table (`StepGraph::anchors`)
//! that records its position:
//!
//! - `anchor` — the pick the reference points at (followed through tombstones to a live commit);
//! - `rank`   — where it sits among references stacked on that same pick (0 = closest to it);
//! - `kind`   — which of the pick's incoming child edges descend into THIS reference's position
//!   (an [`ApproachKind`]: none, all of them, or one merge lane);
//! - `ambiguous` — whether more than one thing converged here (i.e. this position is a merge).
//!
//! Keeping references out of the edge graph is deliberate: an edge running THROUGH a reference
//! node would make the reference bear connectivity it shouldn't — gluing a commit's history onto
//! whatever else the reference happens to touch. The functions here read and maintain positions.
//!
//! Vocabulary used throughout this module:
//! - **leg** — one incoming child edge of a pick, identified as `(source-pick node, parent-slot)`.
//!   A plain commit has one leg; a merge commit has several.
//! - **approach** — the legs that descend into a reference's position (see [`ref_approach`]). This is what
//!   distinguishes co-located references and picks out which merge lane a reference belongs to.
//! - **chain** — references stacked on one pick, ordered by `rank`. Chains are shallow in
//!   practice (≤3 observed).

use crate::graph_rebase::step_graph::{ApproachKind, LaneCarry, StoredAnchor};
use crate::graph_rebase::{Direction, Step, StepGraph, StepGraphIndex};

/// A reference's position, RESOLVED for reading — the counterpart to the stored [`StoredAnchor`].
///
/// [`StoredAnchor`] is what the graph keeps: a raw anchor node and an [`ApproachKind`]. `RefPosition`
/// is what a consumer wants: the anchor followed through tombstones to a live pick (hence
/// `Option`), and the `approach` legs derived from the kind against the current edges. Produced by
/// [`ref_position`]; never stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RefPosition {
    /// The pick this ref resolves to, reached by descending through reference and tombstone
    /// steps. `None` only when no pick exists below (an unborn or fully emptied lane).
    pub anchor: Option<StepGraphIndex>,
    /// How many reference steps sit between this ref and its anchor (0 = directly above it).
    pub rank: usize,
    /// The picks approaching this ref's position from above with their parent-slots — several
    /// when a shared chain is entered by more than one leg. Empty for chain roots.
    pub approach: Vec<(StepGraphIndex, usize)>,
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
        approach: ref_approach(graph, ref_node),
        ambiguous: stored.ambiguous,
    })
}

/// Classify a fresh `approach` against the picks currently feeding its anchor into a stored
/// [`ApproachKind`]: empty is a chain root, the whole leg set is [`ApproachKind::AllLegs`], and any proper
/// subset is a merge lane keyed by its parent-slots. Called at write time (in `set_anchor`),
/// while the `approach` still matches the live edges.
pub(crate) fn classify_approach(
    approach: &[(StepGraphIndex, usize)],
    legs: &[(StepGraphIndex, usize)],
) -> ApproachKind {
    if approach.is_empty() {
        return ApproachKind::Root;
    }
    let approach_set: std::collections::HashSet<_> = approach.iter().copied().collect();
    let legs_set: std::collections::HashSet<_> = legs.iter().copied().collect();
    if approach_set == legs_set {
        return ApproachKind::AllLegs;
    }
    let mut legs: Vec<(StepGraphIndex, usize)> = approach.to_vec();
    legs.sort_unstable();
    legs.dedup();
    ApproachKind::Lane(legs)
}

/// Reconstruct the current `approach` — the `(source-pick, slot)` legs feeding a position — from its
/// [`ApproachKind`] against `anchor_pick`'s live edges. The source-pick node id is read fresh here, so
/// a leg that was tombstoned or re-slotted after the kind was authored never leaks out stale.
pub(crate) fn derive_approach(
    graph: &StepGraph,
    anchor_pick: StepGraphIndex,
    kind: &ApproachKind,
) -> Vec<(StepGraphIndex, usize)> {
    match kind {
        ApproachKind::Root => Vec::new(),
        ApproachKind::AllLegs => legs_into_pick(graph, anchor_pick),
        ApproachKind::Lane(lane_legs) => legs_into_pick(graph, anchor_pick)
            .into_iter()
            .filter(|leg| lane_legs.contains(leg))
            .collect(),
    }
}

/// The current `approach` of the reference at `node` — the DIRECT lane read: the node's lane
/// carries its own leg list, kept live-exact by edge surgery (see `StepGraph::remove_edge` /
/// `move_edge` / `add_edge`), ordered and filtered by the anchor pick's live legs so a stale
/// lane leg never reaches a consumer.
pub(crate) fn ref_approach(
    graph: &StepGraph,
    node: StepGraphIndex,
) -> Vec<(StepGraphIndex, usize)> {
    let Some(stored) = graph.anchor_of(node) else {
        return Vec::new();
    };
    let lane = graph
        .lane_table()
        .get(&stored.anchor)
        .and_then(|lanes| lanes.iter().find(|lane| lane.members.contains(&node)));
    let approach = match lane {
        None => Vec::new(),
        Some(lane) => {
            let legs = match resolve_to_pick(graph, stored.anchor) {
                Some(pick) => legs_into_pick(graph, pick),
                None => Vec::new(),
            };
            match lane.carry {
                LaneCarry::None => Vec::new(),
                LaneCarry::All => legs,
                LaneCarry::Count(_) => legs
                    .into_iter()
                    .filter(|leg| lane.legs.contains(leg))
                    .collect(),
            }
        }
    };
    crate::graph_rebase::arrangement::probe_read_divergence(graph, node, &approach);
    approach
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
    let mut approach = Vec::new();
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
            approach = picks;
            approach.sort();
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
        approach,
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
/// approached by a child (`approach = Some`). Parallel ROOT chains above one anchor are legitimate
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
    crate::graph_rebase::arrangement::census_to_file(graph);
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
        if pos.approach.is_empty() {
            continue;
        }
        if let Some(previous) = seen.insert((pos.anchor, pos.approach.clone(), pos.rank), node) {
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

/// A chain about to be entered by a new leg, captured BEFORE the leg's edge exists — while
/// the store is still consistent — so [`apply_chain_join`] never reads a half-updated store.
pub(crate) struct ChainJoin {
    /// The joining members: the reference and its chain-mates at or below its rank. Root
    /// chains (empty approach) at one anchor are distinct siblings, so only the reference
    /// itself joins.
    members: Vec<(StepGraphIndex, StoredAnchor)>,
    /// The chain's shared approach at capture time.
    approach: Vec<(StepGraphIndex, usize)>,
}

/// Capture `ref_node`'s chain for a coming join — call BEFORE adding the joining leg's edge.
pub(crate) fn prepare_chain_join(graph: &StepGraph, ref_node: StepGraphIndex) -> ChainJoin {
    let Some(stored) = graph.anchor_of(ref_node) else {
        return ChainJoin {
            members: Vec::new(),
            approach: Vec::new(),
        };
    };
    let members = if matches!(stored.kind, ApproachKind::Root) {
        vec![(ref_node, stored.clone())]
    } else {
        chain_members(graph, ref_node)
            .into_iter()
            .filter(|(_, m)| m.rank <= stored.rank)
            .collect()
    };
    ChainJoin {
        members,
        approach: ref_approach(graph, ref_node),
    }
}

/// The new `leg` enters the captured chain: every member gains it in its approach, classified
/// against the anchor's now-complete legs — call right AFTER the leg's edge is added. AllLegs
/// stays AllLegs; a Lane gains the slot; a Root descends.
pub(crate) fn apply_chain_join(
    graph: &mut StepGraph,
    join: &ChainJoin,
    leg: (StepGraphIndex, usize),
) {
    for (node, member) in &join.members {
        let mut approach = join.approach.clone();
        if !approach.contains(&leg) {
            approach.push(leg);
        }
        let mut placed = StoredAnchor::place(graph, member.anchor, member.rank, &approach);
        placed.ambiguous = member.ambiguous || approach.len() > 1;
        graph.set_anchor(*node, Some(placed));
    }
}

/// Re-anchor every reference resolving to `from_pick` onto `to_pick`.
///
/// With `reclassify` false the kind is PRESERVED (an `AllLegs` chain top follows onto `to_pick`
/// and derives its legs there — the bridged leg set a deletion's re-anchor restores, robust to the
/// reconnect renumbering the leg's slot). With `reclassify` true the ref's current derived legs
/// are re-classified against `to_pick`'s legs, so a ref sliding onto a dup-parent MERGE base splits
/// into the `Lane` its leg occupies. `ambiguous` is preserved. NOTE: preserve-vs-reclassify is
/// per-situation, not cleanly per-caller — see the STAGE-B reanchor notes in graph-unify-plan.md.
pub(crate) fn reanchor_refs_at(
    graph: &mut StepGraph,
    from_pick: StepGraphIndex,
    to_pick: StepGraphIndex,
    reclassify: bool,
) {
    let moves: Vec<_> = graph
        .anchored_refs()
        .filter_map(|(node, stored)| {
            (resolve_to_pick(graph, stored.anchor) == Some(from_pick)).then_some((node, stored))
        })
        .collect();
    for (node, mut stored) in moves {
        if reclassify {
            let approach = ref_approach(graph, node);
            let ambiguous = stored.ambiguous;
            stored = StoredAnchor::place(graph, to_pick, stored.rank, &approach);
            stored.ambiguous = ambiguous;
        } else {
            stored.anchor = to_pick;
        }
        graph.set_anchor(node, Some(stored));
    }
}

/// A rewired leg (source `old.0`, its parent-slot renumbered from `old.1` to `new.1`) keeps
/// carrying the chains it carried: `Lane` kinds fed at that slot re-point at the new slot.
/// Identified by the stored `Lane` slot rather than the live `approach`, so it works even when the
/// edge has already been removed/re-added (the fan-out renumbers edges BEFORE calling this, which
/// would leave `ref_approach` empty). Scoped to the picks `old.0` currently feeds, so unrelated `Lane`
/// refs elsewhere sharing the slot number are untouched. `AllLegs`/`Root` are slot-agnostic.
pub(crate) fn rewrite_approach_leg(
    graph: &mut StepGraph,
    old: (StepGraphIndex, usize),
    new: (StepGraphIndex, usize),
) {
    let moves: Vec<_> = graph
        .anchored_refs()
        .filter(
            |(_, stored)| matches!(&stored.kind, ApproachKind::Lane(legs) if legs.contains(&old)),
        )
        .collect();
    for (node, mut stored) in moves {
        if let ApproachKind::Lane(legs) = &mut stored.kind {
            for leg in legs.iter_mut() {
                if *leg == old {
                    *leg = new;
                }
            }
            legs.sort_unstable();
            legs.dedup();
        }
        graph.set_anchor(node, Some(stored));
    }
}

/// The members of `ref_node`'s chain — every reference with the same resolved anchor and the
/// same (derived) approach — with their stored positions.
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
    let approach = ref_approach(graph, ref_node);
    graph
        .anchored_refs()
        .filter_map(|(node, other)| {
            (ref_approach(graph, node) == approach
                && resolve_to_pick(graph, other.anchor) == anchor)
                .then(|| (node, other.clone()))
        })
        .collect()
}

/// The legs a co-located chain on `pick` is approached by: the pick edges pointing at it,
/// as `(source, parent-slot)` pairs, sorted. Every reference co-located on one pick shares
/// this approach — it is the chain's single entry, replicated across members so the renderer can
/// group them by `(anchor, approach)` and order them by rank.
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
    // Derive every reference's intended position (anchor, rank, approach, ambiguous) from the chain
    // topology while it still exists. `unborn` chains (no pick below) keep no stored anchor.
    let mut positions = Vec::new();
    for node in graph.node_indices() {
        if !matches!(graph[node], Step::Reference { .. }) {
            continue;
        }
        let Some(pos) = derive_ref_position_from_edges(graph, node) else {
            continue;
        };
        let Some(anchor) = pos.anchor else {
            continue;
        };
        positions.push((node, anchor, pos.rank, pos.approach, pos.ambiguous));
    }
    // Set anchors provisionally with the correct anchor (so the strip's `resolve_to_pick` works);
    // the kind is authored below against the STRIPPED legs.
    for (node, anchor, rank, _, _) in &positions {
        graph.set_anchor(
            *node,
            Some(StoredAnchor {
                anchor: *anchor,
                rank: *rank,
                kind: ApproachKind::Root,
                ambiguous: false,
            }),
        );
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
    // Author each kind against the FINAL legs: the chain edges now target picks directly, so the
    // intended approach classifies to the right `Root`/`AllLegs`/`Lane`. `ambiguous` keeps the
    // convergence signal from the chain topology (distinct from `approach.len()`).
    for (node, anchor, rank, approach, ambiguous) in &positions {
        let mut stored = StoredAnchor::place(graph, *anchor, *rank, approach);
        stored.ambiguous = *ambiguous;
        graph.set_anchor(*node, Some(stored));
    }
}
