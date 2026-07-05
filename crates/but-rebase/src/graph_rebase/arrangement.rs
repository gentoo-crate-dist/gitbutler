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
use crate::graph_rebase::step_graph::StoredAnchor;
use crate::graph_rebase::{Step, StepGraph, StepGraphIndex};

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
