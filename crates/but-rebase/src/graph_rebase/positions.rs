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

use std::collections::HashSet;

use crate::graph_rebase::{Direction, Step, StepGraph, StepGraphIndex};

/// Where one ref sits, expressed over commits instead of node topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RefPosition {
    /// The pick this ref resolves to, reached by descending through reference and tombstone
    /// steps. `None` only when no pick exists below (an unborn or fully emptied lane).
    pub anchor: Option<StepGraphIndex>,
    /// How many reference steps sit between this ref and its anchor (0 = directly above it).
    pub rank: usize,
    /// The unique pick approaching this ref's chain from above, with the parent-slot (edge
    /// order) it uses — the dup-parents lane key. `None` for chain roots (nothing above).
    pub via: Option<(StepGraphIndex, usize)>,
}

/// Derive the position of the reference at `ref_node`.
///
/// Returns `None` if `ref_node` is not a [`Step::Reference`].
pub(crate) fn ref_position(graph: &StepGraph, ref_node: StepGraphIndex) -> Option<RefPosition> {
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
    // Ascend: the chain's approaching child is the first pick above, through references and
    // tombstones, along the unique incoming path.
    let mut cursor = ref_node;
    let mut via = None;
    for _ in 0..10_000 {
        let mut incoming = graph.edges_directed(cursor, Direction::Incoming);
        let Some(edge) = incoming.next() else {
            break;
        };
        if incoming.next().is_some() {
            // Multiple approaches: no unique via (censused to zero on built graphs; mutations
            // could create it, in which case the ref behaves as a root).
            break;
        }
        match &graph[edge.source()] {
            Step::Pick(_) => {
                via = Some((edge.source(), edge.weight().order));
                break;
            }
            Step::Reference { .. } | Step::None => cursor = edge.source(),
        }
    }
    Some(RefPosition { anchor, rank, via })
}

/// Every reference that RESOLVES to `pick` — i.e. whose downward chain ends at it — collected
/// by ascending all reference/tombstone paths above `pick`. Order is unspecified, like the
/// node-walking predecessor of this accessor.
pub(crate) fn refs_anchored_at(graph: &StepGraph, pick: StepGraphIndex) -> Vec<StepGraphIndex> {
    let mut refs = Vec::new();
    let mut seen = HashSet::new();
    let mut tips = vec![pick];
    while let Some(tip) = tips.pop() {
        for edge in graph.edges_directed(tip, Direction::Incoming) {
            let child = edge.source();
            if !seen.insert(child) {
                continue;
            }
            match &graph[child] {
                Step::None => tips.push(child),
                Step::Reference { .. } => {
                    refs.push(child);
                    tips.push(child);
                }
                Step::Pick(_) => {}
            }
        }
    }
    refs
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
    let mut seen: std::collections::HashMap<
        (Option<StepGraphIndex>, (StepGraphIndex, usize), usize),
        StepGraphIndex,
    > = Default::default();
    for node in graph.node_indices() {
        if !matches!(graph[node], Step::Reference { .. }) {
            continue;
        }
        let Some(pos) = ref_position(graph, node) else {
            debug_assert!(false, "reference node {node} has no derivable position");
            continue;
        };
        let Some(via) = pos.via else {
            continue;
        };
        if let Some(previous) = seen.insert((pos.anchor, via, pos.rank), node) {
            debug_assert!(
                false,
                "reference nodes {previous} and {node} collide at position {pos:?}"
            );
        }
    }
}
