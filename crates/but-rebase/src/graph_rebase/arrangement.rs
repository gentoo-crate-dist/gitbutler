//! The name-keyed arrangement table — the seed of the position model's end-state.
//!
//! Everything a [`StoredAnchor`](crate::graph_rebase::step_graph::StoredAnchor) records is keyed
//! by graph coordinates (node ids, parent slots) that churn under mutation, which is why positions
//! need incremental maintenance (`rewrite_approach_leg`, `join_chain_at`, the preserve-vs-reclassify
//! flag). The intended replacement keys the same information by REF NAMES, which mutation never
//! churns: per anchor commit, an ordered list of lanes, each an ordered list of ref names — exactly
//! the shape of workspace metadata (stack order, branch order). Anchor, rank, and approach then
//! become DERIVED, projection-style, from the table + live pick edges.
//!
//! This module provides two things:
//!
//! 1. The OP API ([`place_ref`] and friends): mutation sites speak position INTENTS
//!    ([`StackSlot`]) instead of authoring `(anchor, rank, kind)` triples by hand. Implemented
//!    atop the stored anchors today; when the store swaps to the name-keyed table, only these
//!    ops' internals change.
//! 2. A corpus census (env `BUT_ARRANGE_CENSUS`, called from `debug_assert_positions_total`):
//!    extract the table from today's stored positions, re-derive every position from it, and
//!    compare. Divergences enumerate precisely where the name-keyed model needs a better rule —
//!    or where information is genuinely not order-derivable. Verdict so far: zero divergences
//!    corpus-wide.

use std::collections::HashMap;

use crate::graph_rebase::positions::{self, legs_into_pick, ref_position};
use crate::graph_rebase::step_graph::{ApproachKind, StoredAnchor};
use crate::graph_rebase::{Direction, Step, StepGraph, StepGraphIndex};

/// A position in a commit's reference stack, named by intent.
#[derive(Debug, Clone, Copy)]
pub(crate) enum StackSlot {
    /// Directly above this reference in its chain: one rank up, members above shift.
    Above(StepGraphIndex),
    /// At this reference's position, pushing it and everything above one rank up.
    Below(StepGraphIndex),
    /// The bottom of the pick's whole stack (rank 0, carrying all its legs — "the branch
    /// here"); every reference on the pick shifts up.
    Bottom(StepGraphIndex),
    /// The top of the chain the leg `(child, parent-slot)` carries into `pick`.
    LaneTop {
        /// The commit the lane's chain is anchored on.
        pick: StepGraphIndex,
        /// The child edge whose lane the reference stacks onto.
        leg: (StepGraphIndex, usize),
    },
    /// A fresh root above `pick`: nothing descends into it, no other position moves.
    Root(StepGraphIndex),
}

/// Place the reference at `node` into `slot`, shifting other positions as the slot demands.
/// The node must not currently occupy a position that should move with it (this is the FRESH
/// placement op; moving an existing reference is a different intent).
pub(crate) fn place_ref(graph: &mut StepGraph, node: StepGraphIndex, slot: StackSlot) {
    match slot {
        StackSlot::Above(target) => {
            let Some(stored) = graph.anchor_of(target) else {
                return;
            };
            let shifts: Vec<_> = positions::chain_members(graph, target)
                .into_iter()
                .filter(|(mate, m)| *mate != node && m.rank > stored.rank)
                .collect();
            for (mate, mut member) in shifts {
                member.rank += 1;
                graph.set_anchor(mate, Some(member));
            }
            graph.set_anchor(
                node,
                Some(StoredAnchor {
                    rank: stored.rank + 1,
                    ..stored
                }),
            );
        }
        StackSlot::Below(target) => {
            let Some(stored) = graph.anchor_of(target) else {
                return;
            };
            let shifts: Vec<_> = positions::chain_members(graph, target)
                .into_iter()
                .filter(|(mate, m)| *mate != node && m.rank >= stored.rank)
                .collect();
            for (mate, mut member) in shifts {
                member.rank += 1;
                graph.set_anchor(mate, Some(member));
            }
            graph.set_anchor(node, Some(stored));
        }
        StackSlot::Bottom(pick) => {
            let approach = positions::legs_into_pick(graph, pick);
            let shifts: Vec<_> = graph
                .anchored_refs()
                .filter(|(mate, stored)| {
                    *mate != node && positions::resolve_to_pick(graph, stored.anchor) == Some(pick)
                })
                .collect();
            for (mate, mut member) in shifts {
                member.rank += 1;
                graph.set_anchor(mate, Some(member));
            }
            let placed = StoredAnchor::place(graph, pick, 0, &approach);
            graph.set_anchor(node, Some(placed));
        }
        StackSlot::LaneTop { pick, leg } => {
            let approach = vec![leg];
            let rank = graph
                .anchored_refs()
                .filter(|(mate, stored)| {
                    *mate != node
                        && positions::resolve_to_pick(graph, stored.anchor) == Some(pick)
                        && positions::ref_approach(graph, *mate) == approach
                })
                .map(|(_, stored)| stored.rank + 1)
                .max()
                .unwrap_or(0);
            let placed = StoredAnchor::place(graph, pick, rank, &approach);
            graph.set_anchor(node, Some(placed));
        }
        StackSlot::Root(pick) => {
            let placed = StoredAnchor::place(graph, pick, 0, &[]);
            graph.set_anchor(node, Some(placed));
        }
    }
}

/// Move the reference at `node` into `slot`, taking its approaching legs along.
///
/// Unlike [`place_ref`] (fresh placement), the reference already holds a position: the legs
/// that approached it follow it when it is their sole carrier (their edges move onto the new
/// anchor and merge into the slot's approach), and — when moving above another reference —
/// the chain members now entered through the moved reference share the merged approach.
pub(crate) fn move_ref(graph: &mut StepGraph, node: StepGraphIndex, slot: StackSlot) {
    let Some(moving) = graph.anchor_of(node) else {
        return;
    };
    let moving_approach = positions::ref_approach(graph, node);
    // Each slot yields the new position as (anchor, rank, approach); the moving reference's
    // legs are merged into the approach below and the whole thing classified once (all leg
    // edges are moved by then, so `place` sees the complete legs).
    let (anchor, rank, mut approach) = match slot {
        StackSlot::Above(target) => {
            let Some(t_stored) = graph.anchor_of(target) else {
                return;
            };
            let shifts: Vec<_> = positions::chain_members(graph, target)
                .into_iter()
                .filter(|(mate, m)| *mate != node && m.rank > t_stored.rank)
                .collect();
            for (mate, mut member) in shifts {
                member.rank += 1;
                graph.set_anchor(mate, Some(member));
            }
            (
                t_stored.anchor,
                t_stored.rank + 1,
                positions::ref_approach(graph, target),
            )
        }
        StackSlot::Below(target) => {
            let Some(t_stored) = graph.anchor_of(target) else {
                return;
            };
            let shifts: Vec<_> = positions::chain_members(graph, target)
                .into_iter()
                .filter(|(mate, m)| *mate != node && m.rank >= t_stored.rank)
                .collect();
            for (mate, mut member) in shifts {
                member.rank += 1;
                graph.set_anchor(mate, Some(member));
            }
            (
                t_stored.anchor,
                t_stored.rank,
                positions::ref_approach(graph, target),
            )
        }
        StackSlot::Bottom(pick) => {
            // The rank-0 position at the pick; existing refs shift up. Only the moved
            // reference's own legs approach it there.
            let shifts: Vec<_> = graph
                .anchored_refs()
                .filter(|(mate, stored)| {
                    *mate != node && positions::resolve_to_pick(graph, stored.anchor) == Some(pick)
                })
                .collect();
            for (mate, mut member) in shifts {
                member.rank += 1;
                graph.set_anchor(mate, Some(member));
            }
            (pick, 0, Vec::new())
        }
        StackSlot::LaneTop { pick, leg } => {
            let approach = vec![leg];
            let rank = graph
                .anchored_refs()
                .filter(|(mate, stored)| {
                    *mate != node
                        && positions::resolve_to_pick(graph, stored.anchor) == Some(pick)
                        && positions::ref_approach(graph, *mate) == approach
                })
                .map(|(_, stored)| stored.rank + 1)
                .max()
                .unwrap_or(0);
            (pick, rank, approach)
        }
        StackSlot::Root(pick) => (pick, 0, Vec::new()),
    };
    // The legs that approached the reference follow it (node-era edges pointed at the
    // reference itself), entering the chain at its new position — but only when it was their
    // sole carrier: chain members staying behind keep their approach.
    let old_anchor_pick = positions::resolve_to_pick(graph, moving.anchor);
    let new_anchor_pick = positions::resolve_to_pick(graph, anchor);
    let sole_carrier = !positions::chain_members(graph, node)
        .into_iter()
        .any(|(mate, m)| mate != node && m.rank < moving.rank);
    if sole_carrier && let (Some(old_pick), Some(new_pick)) = (old_anchor_pick, new_anchor_pick) {
        for (leg, leg_slot) in &moving_approach {
            if old_pick != new_pick {
                let moved: Vec<_> = graph
                    .edges_directed(*leg, Direction::Outgoing)
                    .filter(|e| e.target() == old_pick && e.weight().order == *leg_slot)
                    .map(|e| (e.id(), e.weight().clone()))
                    .collect();
                for (id, weight) in moved {
                    graph.remove_edge(id);
                    graph.add_edge(*leg, new_pick, weight);
                }
            }
            if !approach.contains(&(*leg, *leg_slot)) {
                approach.push((*leg, *leg_slot));
            }
        }
        approach.sort();
        // Members below in the joined chain are now approached through the moved reference:
        // they share the merged entry set.
        if let StackSlot::Above(target) = slot
            && let Some(t_stored) = graph.anchor_of(target)
        {
            let mates: Vec<_> = positions::chain_members(graph, target)
                .into_iter()
                .filter(|(mate, m)| *mate != node && m.rank <= t_stored.rank)
                .map(|(mate, m)| (mate, m.anchor, m.rank))
                .collect();
            for (mate, m_anchor, m_rank) in mates {
                let placed = StoredAnchor::place(graph, m_anchor, m_rank, &approach);
                graph.set_anchor(mate, Some(placed));
            }
        }
    }
    let placed = StoredAnchor::place(graph, anchor, rank, &approach);
    graph.set_anchor(node, Some(placed));
}

/// Re-point the reference at `node` at the commit `new_anchor` — `git update-ref`, spoken as
/// a position move. Its approaching legs follow it (their edges move onto the new anchor),
/// chain members stacked above move with it, and members below lose their approach (they
/// become roots at the old anchor). An anchorless reference is placed as a fresh root; a
/// reference already resolving there just refreshes its stored anchor.
pub(crate) fn repoint_ref(graph: &mut StepGraph, node: StepGraphIndex, new_anchor: StepGraphIndex) {
    let Some(stored) = graph.anchor_of(node) else {
        place_ref(graph, node, StackSlot::Root(new_anchor));
        return;
    };
    match positions::resolve_to_pick(graph, stored.anchor) {
        Some(old_anchor) if old_anchor != new_anchor => {
            // Snapshot the reference's legs before moving their edges (the derived approach
            // tracks live edges), so it can be re-placed against `new_anchor`'s final legs.
            let approach = positions::ref_approach(graph, node);
            for (leg, slot) in &approach {
                let moved: Vec<_> = graph
                    .edges_directed(*leg, Direction::Outgoing)
                    .filter(|e| e.target() == old_anchor && e.weight().order == *slot)
                    .map(|e| (e.id(), e.weight().clone()))
                    .collect();
                for (id, weight) in moved {
                    graph.remove_edge(id);
                    graph.add_edge(*leg, new_anchor, weight);
                }
            }
            let mates: Vec<_> = positions::chain_members(graph, node)
                .into_iter()
                .filter(|(mate, _)| *mate != node)
                .collect();
            for (mate, mut member) in mates {
                if member.rank < stored.rank {
                    // Left behind at the old anchor without an approach.
                    member.kind = ApproachKind::Root;
                    member.ambiguous = false;
                } else {
                    // Stacked above the moved reference: it carries them along.
                    member.anchor = new_anchor;
                }
                graph.set_anchor(mate, Some(member));
            }
            // The reference's legs moved with it; re-classify its lane against `new_anchor`'s
            // final legs (its old `Lane` slot may not exist there).
            let mut placed = StoredAnchor::place(graph, new_anchor, stored.rank, &approach);
            placed.ambiguous = stored.ambiguous;
            graph.set_anchor(node, Some(placed));
        }
        _ => {
            graph.set_anchor(
                node,
                Some(StoredAnchor {
                    anchor: new_anchor,
                    ..stored
                }),
            );
        }
    }
}

/// Remove the reference at `node` from its chain: members above close the gap and the
/// reference becomes a root at its current anchor — nothing descends into it any more. With
/// `drop_legs` the pick edges that approached its position are removed outright; otherwise
/// they stay on the anchor for a follow-up reconnect to rewire.
pub(crate) fn unhook_ref(graph: &mut StepGraph, node: StepGraphIndex, drop_legs: bool) {
    let Some(unhooked) = graph.anchor_of(node) else {
        return;
    };
    let shifts: Vec<_> = positions::chain_members(graph, node)
        .into_iter()
        .filter(|(mate, m)| *mate != node && m.rank > unhooked.rank)
        .collect();
    for (mate, mut member) in shifts {
        member.rank -= 1;
        graph.set_anchor(mate, Some(member));
    }
    if drop_legs && let Some(anchor) = positions::resolve_to_pick(graph, unhooked.anchor) {
        for (leg, slot) in positions::ref_approach(graph, node) {
            let removed: Vec<_> = graph
                .edges_directed(leg, Direction::Outgoing)
                .filter(|e| e.target() == anchor && e.weight().order == slot)
                .map(|e| e.id())
                .collect();
            for id in removed {
                graph.remove_edge(id);
            }
        }
    }
    graph.set_anchor(
        node,
        Some(StoredAnchor {
            kind: ApproachKind::Root,
            ambiguous: false,
            ..unhooked
        }),
    );
}

/// Move the stack slice led by `lead_ref` — it and everything above it in its lane on
/// `source_pick` — onto `dest_anchor`: ranks rebase so the lead lands at 0, each member is
/// re-classified against its own legs at the destination (they come along), and stored
/// ambiguity is preserved.
pub(crate) fn transfer_stack(
    graph: &mut StepGraph,
    lead_ref: StepGraphIndex,
    source_pick: StepGraphIndex,
    dest_anchor: StepGraphIndex,
) {
    let Some(lead) = graph.anchor_of(lead_ref) else {
        return;
    };
    let lane = positions::ref_approach(graph, lead_ref);
    let moves: Vec<_> = graph
        .anchored_refs()
        .filter(|(node, stored)| {
            positions::resolve_to_pick(graph, stored.anchor) == Some(source_pick)
                && positions::ref_approach(graph, *node) == lane
                && stored.rank >= lead.rank
        })
        .map(|(node, _)| node)
        .collect();
    for node in moves {
        if let Some(stored) = graph.anchor_of(node) {
            let approach = positions::ref_approach(graph, node);
            let mut placed =
                StoredAnchor::place(graph, dest_anchor, stored.rank - lead.rank, &approach);
            placed.ambiguous = stored.ambiguous;
            graph.set_anchor(node, Some(placed));
        }
    }
}

/// Carry the slice of `lane` on `source_pick` strictly above `above_rank` onto `dest_anchor`
/// verbatim — same ranks, same kinds; only the anchor key changes. The delimiter position
/// below the slice stays behind. `lane`/`above_rank` are caller-captured (pre-mutation)
/// coordinates rather than live derivations.
pub(crate) fn carry_stack_above(
    graph: &mut StepGraph,
    source_pick: StepGraphIndex,
    lane: &[(StepGraphIndex, usize)],
    above_rank: usize,
    dest_anchor: StepGraphIndex,
) {
    let moves: Vec<_> = graph
        .anchored_refs()
        .filter(|(node, stored)| {
            positions::resolve_to_pick(graph, stored.anchor) == Some(source_pick)
                && positions::ref_approach(graph, *node) == lane
                && stored.rank > above_rank
        })
        .map(|(node, _)| node)
        .collect();
    for node in moves {
        if let Some(mut stored) = graph.anchor_of(node) {
            stored.anchor = dest_anchor;
            graph.set_anchor(node, Some(stored));
        }
    }
}

/// Stack every reference on `source_pick` above `top` (a reference on another pick), the
/// whole tower re-placed behind `bridge_anchor`'s full incoming leg set — the bridged legs
/// that now descend into the joined chain. Returns false (leaving the graph untouched) when
/// `top` holds no position.
pub(crate) fn land_stack_above(
    graph: &mut StepGraph,
    source_pick: StepGraphIndex,
    top: StepGraphIndex,
    bridge_anchor: StepGraphIndex,
) -> bool {
    let Some(top_stored) = graph.anchor_of(top) else {
        return false;
    };
    let bridge = positions::legs_into_pick(graph, bridge_anchor);
    let top_rank = top_stored.rank;
    let placed_top = StoredAnchor::place(graph, top_stored.anchor, top_rank, &bridge);
    graph.set_anchor(top, Some(placed_top));

    let moves: Vec<_> = graph
        .anchored_refs()
        .filter(|(_, stored)| positions::resolve_to_pick(graph, stored.anchor) == Some(source_pick))
        .map(|(node, stored)| (node, stored.rank))
        .collect();
    for (node, rank) in moves {
        let placed = StoredAnchor::place(graph, bridge_anchor, rank + top_rank + 1, &bridge);
        graph.set_anchor(node, Some(placed));
    }
    true
}

/// Re-key every reference whose anchor no longer resolves (it sat on removed picks) onto
/// `new_anchor`, positions carried verbatim — the ruled dangling semantics: the position
/// follows where the commit's place went, the approach stays.
pub(crate) fn readopt_dangling_refs(graph: &mut StepGraph, new_anchor: StepGraphIndex) {
    let dangling: Vec<_> = graph
        .anchored_refs()
        .filter(|(_, stored)| positions::resolve_to_pick(graph, stored.anchor).is_none())
        .collect();
    for (node, mut stored) in dangling {
        stored.anchor = new_anchor;
        graph.set_anchor(node, Some(stored));
    }
}

/// Which side of `at_ref` a chain split leaves with the lower part.
pub(crate) enum SplitBoundary {
    /// Members strictly above the ref move up; the ref stays with the lower part.
    Above,
    /// The ref and members above it move up; only members below stay.
    At,
}

/// The result of splitting a chain around an interposed pick.
pub(crate) struct ChainSplit {
    /// The members left behind, with their pre-split anchors — settle them with
    /// [`settle_chain_lower`] once the leg entering the lower part is known.
    pub lower: Vec<(StepGraphIndex, StoredAnchor)>,
    /// Whether any member moved onto the upper anchor. When none did, `at_ref` was the top
    /// of its chain, so the chain's carried legs belong to the caller's new pick.
    pub moved_any: bool,
}

/// Split the chain at `at_ref` around a pick interposed into it: members on the upper side
/// of `boundary` re-key onto `upper_anchor` with ranks rebased to start at 0 (approach kinds
/// carried verbatim), and the lower members are returned untouched for the caller to settle.
pub(crate) fn split_chain(
    graph: &mut StepGraph,
    at_ref: StepGraphIndex,
    boundary: SplitBoundary,
    upper_anchor: StepGraphIndex,
) -> ChainSplit {
    let Some(stored) = graph.anchor_of(at_ref) else {
        return ChainSplit {
            lower: Vec::new(),
            moved_any: false,
        };
    };
    let members = positions::chain_members(graph, at_ref);
    let (goes_up, rank_base): (fn(usize, usize) -> bool, usize) = match boundary {
        SplitBoundary::Above => (|rank, at| rank > at, stored.rank + 1),
        SplitBoundary::At => (|rank, at| rank >= at, stored.rank),
    };
    let mut lower = Vec::new();
    let mut moved_any = false;
    for (node, mut member) in members {
        if goes_up(member.rank, stored.rank) {
            member.anchor = upper_anchor;
            member.rank -= rank_base;
            graph.set_anchor(node, Some(member));
            moved_any = true;
        } else {
            lower.push((node, member));
        }
    }
    ChainSplit { lower, moved_any }
}

/// Settle the lower part of a split chain: each member keeps its anchor and rank but is now
/// approached through `leg` — the edge descending from the interposed pick.
pub(crate) fn settle_chain_lower(
    graph: &mut StepGraph,
    lower: &[(StepGraphIndex, StoredAnchor)],
    leg: (StepGraphIndex, usize),
) {
    for (node, member) in lower {
        let placed = StoredAnchor::place(graph, member.anchor, member.rank, &[leg]);
        graph.set_anchor(*node, Some(placed));
    }
}

/// How much of its anchor's incoming legs a lane carries.
#[derive(Debug, Clone, PartialEq, Eq)]
enum LaneCarry {
    /// Nothing descends into this lane (a root chain: remote above a tip, empty top).
    None,
    /// Every leg into the anchor descends through this lane (a plain chain, or a shared
    /// chain all merge lanes converge on).
    All,
    /// This lane carries exactly `n` legs — one lane of a merge. Which legs is derived by
    /// consuming the anchor's sorted legs in lane order.
    Count(usize),
}

/// One lane of a co-located group: refs bottom-up. `rank`/`ambiguous` are carried verbatim in
/// this v1 (rank is only topology-defined inside carrying chains; root-sibling order is table
/// data by design).
#[derive(Debug, Clone)]
struct Lane {
    refs: Vec<(gix::refs::FullName, usize, bool)>,
    carry: LaneCarry,
}

/// The lanes of every anchor pick that has references on it.
struct Arrangement {
    groups: HashMap<StepGraphIndex, Vec<Lane>>,
}

/// Extract the arrangement from the CURRENT stored positions, recording anomalies that the
/// name-keyed model must care about (duplicate names, unanchored refs, non-contiguous chain
/// ranks, non-consecutive lane legs).
fn extract(graph: &StepGraph, notes: &mut Vec<String>) -> Arrangement {
    let mut seen_names: HashMap<gix::refs::FullName, StepGraphIndex> = HashMap::new();
    // (anchor, approach) -> members
    type ChainKey = (StepGraphIndex, Vec<(StepGraphIndex, usize)>);
    let mut chains: HashMap<ChainKey, Vec<(gix::refs::FullName, usize, bool)>> = HashMap::new();
    for node in graph.node_indices() {
        let Step::Reference { refname, .. } = &graph[node] else {
            continue;
        };
        if let Some(previous) = seen_names.insert(refname.clone(), node) {
            notes.push(format!("DUPNAME {refname:?} nodes {previous} and {node}"));
        }
        let Some(pos) = ref_position(graph, node) else {
            continue; // no stored anchor: unborn, exempt like the standing assert
        };
        let Some(anchor) = pos.anchor else {
            notes.push(format!("UNANCHORED {refname:?}"));
            continue;
        };
        chains.entry((anchor, pos.approach)).or_default().push((
            refname.clone(),
            pos.rank,
            pos.ambiguous,
        ));
    }

    let mut groups: HashMap<StepGraphIndex, Vec<(Vec<(StepGraphIndex, usize)>, Lane)>> =
        HashMap::new();
    for ((anchor, approach), mut members) in chains {
        members.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let legs = legs_into_pick(graph, anchor);
        let carry = if approach.is_empty() {
            LaneCarry::None
        } else if approach == legs {
            LaneCarry::All
        } else {
            // The consumption model needs each lane's legs to be a consecutive run of the
            // anchor's sorted legs; measure violations instead of assuming.
            if let Some(start) = legs.iter().position(|l| Some(l) == approach.first()) {
                if legs[start..].len() < approach.len()
                    || legs[start..start + approach.len()] != approach[..]
                {
                    notes.push(format!(
                        "NONCONSECUTIVE anchor {anchor} approach {approach:?} legs {legs:?}"
                    ));
                }
            } else {
                notes.push(format!(
                    "APPROACH-NOT-IN-LEGS anchor {anchor} approach {approach:?} legs {legs:?}"
                ));
            }
            LaneCarry::Count(approach.len())
        };
        if carry != LaneCarry::None {
            let ranks: Vec<usize> = members.iter().map(|m| m.1).collect();
            if ranks.iter().copied().ne(0..members.len()) {
                notes.push(format!(
                    "RANK-GAP anchor {anchor} ranks {ranks:?} (carrying chain)"
                ));
            }
        }
        groups.entry(anchor).or_default().push((
            approach,
            Lane {
                refs: members,
                carry,
            },
        ));
    }

    // Lane order within a group: carrying lanes by their first leg (the parent-array order the
    // materialization writes), shared `All` lanes next, root lanes last by name. This is the
    // order a name-keyed table would persist.
    let groups = groups
        .into_iter()
        .map(|(anchor, mut lanes)| {
            lanes.sort_by(|(approach_a, lane_a), (approach_b, lane_b)| {
                let class = |lane: &Lane| match lane.carry {
                    LaneCarry::Count(_) => 0,
                    LaneCarry::All => 1,
                    LaneCarry::None => 2,
                };
                class(lane_a)
                    .cmp(&class(lane_b))
                    .then_with(|| approach_a.cmp(approach_b))
                    .then_with(|| lane_a.refs.cmp(&lane_b.refs))
            });
            (anchor, lanes.into_iter().map(|(_, lane)| lane).collect())
        })
        .collect();
    Arrangement { groups }
}

/// Re-derive every reference's position from the arrangement + live edges: `Count` lanes consume
/// the anchor's sorted legs in lane order, `All` lanes take them all, `None` lanes take none.
fn derive(
    graph: &StepGraph,
    arrangement: &Arrangement,
    notes: &mut Vec<String>,
) -> HashMap<gix::refs::FullName, (StepGraphIndex, usize, Vec<(StepGraphIndex, usize)>, bool)> {
    let mut out = HashMap::new();
    for (&anchor, lanes) in &arrangement.groups {
        let legs = legs_into_pick(graph, anchor);
        let mut consumed = 0usize;
        for lane in lanes {
            let approach = match lane.carry {
                LaneCarry::None => Vec::new(),
                LaneCarry::All => legs.clone(),
                LaneCarry::Count(n) => {
                    let run = legs
                        .get(consumed..consumed + n)
                        .map(<[_]>::to_vec)
                        .unwrap_or_default();
                    consumed += n;
                    run
                }
            };
            for (name, rank, ambiguous) in &lane.refs {
                out.insert(name.clone(), (anchor, *rank, approach.clone(), *ambiguous));
            }
        }
        if consumed > 0 && consumed != legs.len() {
            notes.push(format!(
                "UNCONSUMED-LEGS anchor {anchor} consumed {consumed} of {}",
                legs.len()
            ));
        }
    }
    out
}

/// Round-trip the current graph through the name-keyed arrangement and report every divergence
/// and anomaly. Empty result = this graph's positions are fully order-derivable.
fn census(graph: &StepGraph) -> Vec<String> {
    let mut notes = Vec::new();
    let arrangement = extract(graph, &mut notes);
    let derived = derive(graph, &arrangement, &mut notes);
    for node in graph.node_indices() {
        let Step::Reference { refname, .. } = &graph[node] else {
            continue;
        };
        let Some(pos) = ref_position(graph, node) else {
            continue;
        };
        let Some(anchor) = pos.anchor else {
            continue;
        };
        match derived.get(refname) {
            Some((d_anchor, d_rank, d_approach, d_ambiguous)) => {
                if (*d_anchor, *d_rank, d_approach, *d_ambiguous)
                    != (anchor, pos.rank, &pos.approach, pos.ambiguous)
                {
                    notes.push(format!(
                        "DIVERGE {refname:?} stored=({anchor},{},{:?},{}) derived=({d_anchor},{d_rank},{d_approach:?},{d_ambiguous})",
                        pos.rank, pos.approach, pos.ambiguous
                    ));
                }
            }
            None => notes.push(format!("MISSING {refname:?}")),
        }
    }
    notes
}

/// Env-gated corpus probe: when `BUT_ARRANGE_CENSUS` names a file, append this graph's census
/// findings (and a `GRAPHS` counter line) to it. Capture-proof, like the earlier census tooling.
pub(crate) fn census_to_file(graph: &StepGraph) {
    let Ok(path) = std::env::var("BUT_ARRANGE_CENSUS") else {
        return;
    };
    let notes = census(graph);
    use std::io::Write as _;
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    else {
        return;
    };
    let refs = graph
        .node_indices()
        .filter(|&n| matches!(graph[n], Step::Reference { .. }))
        .count();
    let _ = writeln!(file, "GRAPH refs={refs} findings={}", notes.len());
    for note in notes {
        let _ = writeln!(file, "{note}");
    }
}
