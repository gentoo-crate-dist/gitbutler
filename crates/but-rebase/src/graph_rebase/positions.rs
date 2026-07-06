//! Where each reference sits, stored as position data rather than as graph edges.
//!
//! A commit ([`Step::Pick`]) carries parent edges; a reference ([`Step::Reference`]) carries
//! NONE. Instead every reference has a [`RefPosition`] stored on its ref record
//! that records its position:
//!
//! - `on` — the node the reference points at (followed through tombstones to a live pick);
//! - `below`  — the reference directly underneath in the physical stack (`None` = on the pick);
//! - `ambiguous` — whether more than one thing converged here (i.e. this position is a merge).
//!
//! Which of the pick's incoming child edges descend into a reference's position lives in the
//! reference's LANE (`CommitGraph::lanes`): each lane records its members and a
//! [`LaneCarry`] — none of the legs, all of them, or an explicit leg list.
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
//! - **chain** — references stacked on one pick, ordered by their below-chain ([`ref_depth`]).
//!   Chains are shallow in practice (≤3 observed).

use crate::graph_rebase::commit_graph::{LaneCarry, RefPosition};
use crate::graph_rebase::{CommitGraph, CommitGraphIndex};

/// The reference's depth above its pick — the length of its below-chain (0 = directly on
/// the pick). This IS the rank: order among co-located references is adjacency, not a number.
pub(crate) fn ref_depth(graph: &CommitGraph, node: CommitGraphIndex) -> usize {
    let mut depth = 0usize;
    let mut cursor = graph.position_of(node).and_then(|s| s.below);
    while let Some(b) = cursor {
        depth += 1;
        if depth > 10_000 {
            debug_assert!(false, "below-chain cycle at ref {node}");
            return depth;
        }
        cursor = graph.position_of(b).and_then(|s| s.below);
    }
    depth
}

/// The current `approach` of the reference at `node` — the DIRECT lane read: the node's lane
/// carries its own leg list, kept aligned by the slot mutators (`CommitGraph::remove_parent` /
/// `insert_parent` / `replace_parent`), ordered and filtered by the resolved pick's live legs
/// so a stale lane leg never reaches a consumer.
pub(crate) fn ref_approach(
    graph: &CommitGraph,
    node: CommitGraphIndex,
) -> Vec<(CommitGraphIndex, usize)> {
    let Some(stored) = graph.position_of(node) else {
        return Vec::new();
    };
    let lane = graph
        .lane_table()
        .get(&stored.on)
        .and_then(|lanes| lanes.iter().find(|lane| lane.members.contains(&node)));
    match lane {
        None => Vec::new(),
        Some(lane) => {
            let legs = match resolve_to_pick(graph, stored.on) {
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
    }
}

/// Every reference that RESOLVES to `pick` — its stored `on`, followed through tombstones,
/// ends at it. Order is unspecified (ascending node id), like the node-walking predecessor.
pub(crate) fn refs_resolving_to(
    graph: &CommitGraph,
    pick: CommitGraphIndex,
) -> Vec<CommitGraphIndex> {
    graph
        .positioned_refs()
        .filter_map(|(node, stored)| {
            (resolve_to_pick(graph, stored.on) == Some(pick)).then_some(node)
        })
        .collect()
}

/// The standing collapse invariant: every reference in the graph has a well-formed position,
/// and positions are unique wherever order is topologically meaningful — i.e. within chains
/// approached by a child (`approach = Some`). Parallel ROOT chains above one pick are legitimate
/// unordered siblings (found by this very assert on its first corpus run): with nothing above
/// them, their relative order is not defined by topology — the collapse orders them like the
/// passive set (by name).
///
/// Wired at editor creation AND at rebase entry, so every graph shape the suite produces —
/// including post-mutation shapes — continuously validates the position model.
pub(crate) fn debug_assert_positions_total(graph: &CommitGraph) {
    if !cfg!(debug_assertions) {
        return;
    }
    crate::graph_rebase::arrangement::census_to_file(graph);
    census_stale_statements(graph);
    debug_assert_below_wellformed(graph);
    type OrderedPositionKey = (
        Option<CommitGraphIndex>,
        Vec<(CommitGraphIndex, usize)>,
        usize,
    );
    let mut seen: std::collections::HashMap<OrderedPositionKey, CommitGraphIndex> =
        Default::default();
    for node in graph.references().map(|(node, _, _)| node) {
        // A reference without a stored position is only legitimate when the graph holds no
        // pick below it at creation (unborn); it resolves to nothing.
        let Some(stored) = graph.position_of(node) else {
            continue;
        };
        let approach = ref_approach(graph, node);
        if approach.is_empty() {
            continue;
        }
        let pick = resolve_to_pick(graph, stored.on);
        let rank = ref_depth(graph, node);
        if let Some(previous) = seen.insert((pick, approach.clone(), rank), node) {
            debug_assert!(
                false,
                "reference nodes {previous} and {node} collide at position \
                 (pick {pick:?}, approach {approach:?}, rank {rank})"
            );
        }
    }
}

/// Stale-statement census: lane legs naming a non-live leg of their key's resolved pick.
/// Statements are read filtered against live legs, so staleness is legal mid-op — this
/// measures whether any survives to a checkpoint (a few adjudicated flows do, in the
/// upstream-integration re-parent family). Report-only, gated on `BUT_ORDER_CENSUS=<file>`
/// (one `CHECK n=<nodes-with-parents>` line per checkpoint, one `STALE` line per finding;
/// `BUT_ORDER_STALE_PANIC=1` for attribution). Parent-order density itself is structural
/// now — the store is an array.
fn census_stale_statements(graph: &CommitGraph) {
    use std::io::Write as _;
    let mut file = std::env::var("BUT_ORDER_CENSUS").ok().and_then(|path| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .ok()
    });
    let checked = graph
        .node_indices()
        .filter(|&node| graph.parent_count(node) > 0)
        .count();
    for (key, lanes) in graph.lane_table() {
        let live = match resolve_to_pick(graph, *key) {
            Some(pick) => legs_into_pick(graph, pick),
            None => Vec::new(),
        };
        for lane in lanes {
            for leg in &lane.legs {
                if !live.contains(leg) {
                    assert!(
                        std::env::var_os("BUT_ORDER_STALE_PANIC").is_none(),
                        "stale lane statement at rest: key={key} leg=({},{})",
                        leg.0,
                        leg.1
                    );
                    if let Some(file) = &mut file {
                        let _ = writeln!(
                            file,
                            "STALE key={key} leg=({},{}) members={:?}",
                            leg.0,
                            leg.1,
                            lane.members
                                .iter()
                                .map(|m| m.to_string())
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
    if let Some(file) = &mut file {
        let _ = writeln!(file, "CHECK n={checked}");
    }
}

/// Every stored `below` of a LIVE reference names a positioned reference resolving to the SAME
/// pick, and the below-chain is acyclic. Tombstoned refs keep their stored position for
/// retention reads but are spliced out of the physical stack, so only live refs are graded.
/// `BUT_BELOW_DUMP=1` prints the full position table first.
fn debug_assert_below_wellformed(graph: &CommitGraph) {
    let name = |node: CommitGraphIndex| match graph.reference(node) {
        Some((refname, _)) => refname.to_string(),
        None => format!("{:?}", graph.step_view(node)),
    };
    if std::env::var_os("BUT_BELOW_DUMP").is_some() {
        for (node, stored) in graph.positioned_refs() {
            eprintln!(
                "POS {node} ({}) on={} resolved={:?} depth={} below={:?} ambiguous={}",
                name(node),
                stored.on,
                resolve_to_pick(graph, stored.on),
                ref_depth(graph, node),
                stored.below,
                stored.ambiguous,
            );
        }
    }
    for (node, stored) in graph.positioned_refs() {
        if !graph.is_reference(node) {
            continue;
        }
        let pick = resolve_to_pick(graph, stored.on);
        let mut depth = 0usize;
        let mut cursor = stored.below;
        while let Some(b) = cursor {
            let Some(mate) = graph.position_of(b) else {
                debug_assert!(false, "ref {node}: below {b} is not a positioned reference");
                return;
            };
            debug_assert_eq!(
                resolve_to_pick(graph, mate.on),
                pick,
                "ref {node} ({}): below {b} ({}) resolves to a different pick",
                name(node),
                name(b)
            );
            depth += 1;
            debug_assert!(depth <= 10_000, "ref {node}: below-chain cycle");
            if depth > 10_000 {
                return;
            }
            cursor = mate.below;
        }
    }
}

/// The references the node-era traversal from `start` would have walked through, given the
/// PICK set it reached: a chain is entered when one of its legs was visited (the edge from
/// leg to chain top), and when `start` is itself a reference, it and its chain below count.
pub(crate) fn refs_reachable_with(
    graph: &CommitGraph,
    start: CommitGraphIndex,
    picks: &std::collections::HashSet<CommitGraphIndex>,
) -> Vec<CommitGraphIndex> {
    // Reached commits by ID as well as node: a graph can hold one commit twice (a stack lane
    // and a target lane), and the node era's shared reference nodes made reachability
    // commit-equivalent across such lanes.
    let reached_ids: std::collections::HashSet<gix::ObjectId> = picks
        .iter()
        .filter_map(|node| graph.commit_id(*node))
        .collect();
    let mut out = Vec::new();
    for (node, stored) in graph.positioned_refs() {
        // A chain whose pick is reached lies on reached history — pick-based
        // reachability, exactly what the node-era walk through interposed reference nodes
        // computed (and the ruling the merge-bypass deletion rests on).
        let pick_reached = resolve_to_pick(graph, stored.on).is_some_and(|pick| {
            picks.contains(&pick)
                || graph
                    .commit_id(pick)
                    .is_some_and(|id| reached_ids.contains(&id))
        });
        if pick_reached || node == start {
            out.push(node);
        }
    }
    out
}

/// A chain about to be entered by a new leg, captured BEFORE the leg's edge exists — while
/// the store is still consistent — so [`apply_chain_join`] never reads a half-updated store.
pub(crate) struct ChainJoin {
    /// The joining members: the reference and the chain-mates its below-chain rests on. Root
    /// chains (empty approach) at one pick are distinct siblings, so only the reference
    /// itself joins.
    members: Vec<(CommitGraphIndex, RefPosition)>,
    /// The chain's shared approach at capture time.
    approach: Vec<(CommitGraphIndex, usize)>,
}

/// Capture `ref_node`'s chain for a coming join — call BEFORE adding the joining leg's edge.
pub(crate) fn prepare_chain_join(graph: &CommitGraph, ref_node: CommitGraphIndex) -> ChainJoin {
    let Some(stored) = graph.position_of(ref_node) else {
        return ChainJoin {
            members: Vec::new(),
            approach: Vec::new(),
        };
    };
    let is_root = graph
        .lane_of(ref_node)
        .is_some_and(|lane| lane.carry == LaneCarry::None);
    let members = if is_root {
        vec![(ref_node, stored.clone())]
    } else {
        // The reference plus the chain-mates underneath it: walk the below-chain, keeping
        // members of this chain (the physical stack may pass through other lanes' refs).
        let chain = chain_members(graph, ref_node);
        let mut members = vec![(ref_node, stored.clone())];
        let mut cursor = stored.below;
        while let Some(b) = cursor {
            let Some(m) = graph.position_of(b) else {
                break;
            };
            if chain.iter().any(|(node, _)| *node == b) {
                members.push((b, m.clone()));
            }
            cursor = m.below;
        }
        members
    };
    ChainJoin {
        members,
        approach: ref_approach(graph, ref_node),
    }
}

/// The new `leg` enters the captured chain: every member gains it in its approach, classified
/// against the pick's now-complete legs — call right AFTER the leg's edge is added. AllLegs
/// stays AllLegs; a Lane gains the slot; a Root descends.
pub(crate) fn apply_chain_join(
    graph: &mut CommitGraph,
    join: &ChainJoin,
    leg: (CommitGraphIndex, usize),
) {
    for (node, member) in &join.members {
        let mut approach = join.approach.clone();
        if !approach.contains(&leg) {
            approach.push(leg);
        }
        let ambiguous = member.ambiguous || approach.len() > 1;
        graph.set_position(*node, member.on, &approach, ambiguous, member.below);
    }
}

/// Move every reference resolving to `from_pick` onto `to_pick`.
///
/// With `reclassify` false the kind is PRESERVED (an `AllLegs` chain top follows onto `to_pick`
/// and derives its legs there — the bridged leg set a deletion's re-point restores, robust to the
/// reconnect renumbering the leg's slot). With `reclassify` true the ref's current derived legs
/// are re-classified against `to_pick`'s legs, so a ref sliding onto a dup-parent MERGE base splits
/// into the `Lane` its leg occupies. `ambiguous` is preserved. NOTE: preserve-vs-reclassify is
/// per-situation, not cleanly per-caller — each call site picks based on whether the leg set
/// should survive the move or be re-derived at the destination.
pub(crate) fn reposition_refs(
    graph: &mut CommitGraph,
    from_pick: CommitGraphIndex,
    to_pick: CommitGraphIndex,
    reclassify: bool,
) {
    let moves: Vec<_> = graph
        .positioned_refs()
        .filter_map(|(node, stored)| {
            (resolve_to_pick(graph, stored.on) == Some(from_pick)).then_some((node, stored))
        })
        .collect();
    for (node, stored) in moves {
        if reclassify {
            let approach = ref_approach(graph, node);
            graph.set_position(node, to_pick, &approach, stored.ambiguous, stored.below);
        } else {
            graph.rekey_position(node, to_pick);
        }
    }
}

/// The members of `ref_node`'s chain — every reference with the same resolved pick and the
/// same (derived) approach — with their stored positions.
pub(crate) fn chain_members(
    graph: &CommitGraph,
    ref_node: CommitGraphIndex,
) -> Vec<(
    CommitGraphIndex,
    crate::graph_rebase::commit_graph::RefPosition,
)> {
    let Some(stored) = graph.position_of(ref_node) else {
        return vec![];
    };
    let pick = resolve_to_pick(graph, stored.on);
    let approach = ref_approach(graph, ref_node);
    graph
        .positioned_refs()
        .filter_map(|(node, other)| {
            (ref_approach(graph, node) == approach && resolve_to_pick(graph, other.on) == pick)
                .then(|| (node, other.clone()))
        })
        .collect()
}

/// The legs a co-located chain on `pick` is approached by: the pick edges pointing at it,
/// as `(source, parent-slot)` pairs, sorted. Every reference co-located on one pick shares
/// this approach — it is the chain's single entry, replicated across members so the renderer can
/// group them by `(pick, approach)` and order them by below-chain depth.
pub(crate) fn legs_into_pick(
    graph: &CommitGraph,
    pick: CommitGraphIndex,
) -> Vec<(CommitGraphIndex, usize)> {
    graph
        .incoming_legs(pick)
        .into_iter()
        .filter(|&(child, _)| graph.is_pick(child))
        .collect()
}

/// Resolve `node` to the current pick it stands for: a pick resolves to itself, a tombstone
/// follows its (preserved) first edge downward, and a reference resolves via its stored
/// position — dead references via their RETAINED position, the retention pointer stale
/// selectors normalize through (unborn refs carry none and resolve to nothing).
pub(crate) fn resolve_to_pick(
    graph: &CommitGraph,
    node: CommitGraphIndex,
) -> Option<CommitGraphIndex> {
    let mut cursor = node;
    for _ in 0..10_000 {
        if graph.is_pick(cursor) {
            return Some(cursor);
        }
        // A reference resolves via its stored position; a tombstone follows its preserved
        // first edge downward.
        cursor = match graph.position_of(cursor) {
            Some(stored) => stored.on,
            None => graph.parents(cursor).first().copied()?,
        };
    }
    None
}
