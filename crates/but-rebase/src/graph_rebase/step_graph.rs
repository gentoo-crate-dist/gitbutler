//! An owned arena graph for rebase steps, replacing petgraph: nodes are never removed (a
//! removed step becomes [`Step::None`]), so node ids are stable by construction; edges live in
//! a slot arena so edge ids stay stable across removals. Iteration matches the semantics the
//! call sites were written against: `edges_directed` yields newest-first, `node_indices` and
//! `edge_references` ascend.

use crate::graph_rebase::{Edge, Step};

/// The stable identifier of a step node. Only ever grows; tombstoning is done at the
/// [`Step`] level, never by removal.
pub(crate) type StepGraphIndex = usize;

/// The stable identifier of an edge slot.
pub(crate) type StepEdgeIndex = usize;

/// The direction of edges to look at from a node's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    /// Edges from this node towards its parents.
    Outgoing,
    /// Edges from children towards this node.
    Incoming,
}

#[derive(Debug, Clone)]
struct EdgeRecord {
    source: StepGraphIndex,
    target: StepGraphIndex,
    weight: Edge,
}

/// A borrowed view of one edge, mirroring the accessors call sites used on petgraph's edge
/// references.
#[derive(Clone, Copy)]
pub(crate) struct StepEdgeRef<'graph> {
    id: StepEdgeIndex,
    source: StepGraphIndex,
    target: StepGraphIndex,
    weight: &'graph Edge,
}

impl<'graph> StepEdgeRef<'graph> {
    /// The edge's stable id, usable with [`StepGraph::remove_edge()`].
    pub(crate) fn id(&self) -> StepEdgeIndex {
        self.id
    }

    /// The node this edge points away from (the child side).
    pub(crate) fn source(&self) -> StepGraphIndex {
        self.source
    }

    /// The node this edge points at (the parent side).
    pub(crate) fn target(&self) -> StepGraphIndex {
        self.target
    }

    /// The edge payload.
    pub(crate) fn weight(&self) -> &'graph Edge {
        self.weight
    }
}

/// How a reference's approaching legs (its `via`) relate to the picks that currently feed its
/// anchor — the part of a position that plain edge topology can't recover once reference edges
/// are stripped. Derived from the fresh `via` at write time and re-resolved against the CURRENT
/// pick edges at read time, so it survives edge churn that leaves a stored `(source-pick, slot)`
/// list stale.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) enum ViaKind {
    /// Nothing descends into this position — a chain root (`via = []`): a remote ref stacked
    /// above a tip, an empty single-branch top.
    #[default]
    Root,
    /// Every leg into the anchor feeds this position — a plain chain, or a merge chain both
    /// lanes converge on (`via = legs_into_pick(anchor)`).
    AllLegs,
    /// Only the legs entering at these parent-slots feed this position — one lane of a
    /// dup-parent merge (`via = legs_into_pick(anchor)` filtered to these slots).
    Lane(Vec<usize>),
}

/// Where a reference sits, stored explicitly: references are POSITIONS, not topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoredAnchor {
    /// The node this reference resolves to (a pick, or its tombstone after deletion) — the
    /// commit the ref points at, reached lazily through tombstones at read time.
    pub anchor: StepGraphIndex,
    /// The picks approaching this reference's position from above, with their parent-slots —
    /// several when a shared chain is entered by more than one leg (a merge convergence).
    /// Empty for a chain root (nothing above).
    pub via: Vec<(StepGraphIndex, usize)>,
    /// Orders co-located references above one anchor, 0 = closest to the anchor.
    pub rank: usize,
    /// The entry into this position was ambiguous when derived — more than one thing (legs
    /// and/or refs stacked above) converged on it. Ambiguous chains belong to their anchor's
    /// lane, never to a single approaching leg.
    pub ambiguous: bool,
}

/// The rebase step graph: an arena of [`Step`]s where PICKS carry ordered parent edges and
/// REFERENCES carry explicit positions — edges are the truth for commits, anchors the truth
/// for refs, with no overlap. A reference is never part of the edge graph, so it cannot bear
/// connectivity.
#[derive(Debug, Clone, Default)]
pub(crate) struct StepGraph {
    nodes: Vec<Step>,
    edges: Vec<Option<EdgeRecord>>,
    outgoing: Vec<Vec<StepEdgeIndex>>,
    incoming: Vec<Vec<StepEdgeIndex>>,
    /// `Some` exactly for reference nodes.
    anchors: Vec<Option<StoredAnchor>>,
    /// The derived-at-write [`ViaKind`] for each reference, parallel to `anchors` and authored
    /// by [`StepGraph::set_anchor`]. Consumers read `via` through it (see `positions::ref_via`);
    /// the stored `StoredAnchor::via` is only the write-time input it is classified from.
    ref_kinds: Vec<Option<ViaKind>>,
}

impl StepGraph {
    /// An empty graph.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Add `step` and return its stable id.
    pub(crate) fn add_node(&mut self, step: Step) -> StepGraphIndex {
        self.nodes.push(step);
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
        self.anchors.push(None);
        self.ref_kinds.push(None);
        self.nodes.len() - 1
    }

    /// The stored position of the reference at `node`, if it is a positioned reference.
    pub(crate) fn anchor_of(&self, node: StepGraphIndex) -> Option<StoredAnchor> {
        self.anchors.get(node).cloned().flatten()
    }

    /// Set (or clear) the stored position of the reference at `node`, authoring its [`ViaKind`]
    /// from the fresh `via` against the CURRENT pick edges. Callers keep supplying `via` as they
    /// always did; the kind is what consumers read, so authoring here (right after the op's edge
    /// changes) captures each lane while the legs are still current.
    pub(crate) fn set_anchor(&mut self, node: StepGraphIndex, anchor: Option<StoredAnchor>) {
        let kind = anchor.as_ref().map(|a| {
            let legs = match crate::graph_rebase::positions::resolve_to_pick(self, a.anchor) {
                Some(pick) => crate::graph_rebase::positions::legs_into_pick(self, pick),
                None => Vec::new(),
            };
            crate::graph_rebase::positions::classify_via(&a.via, &legs)
        });
        self.ref_kinds[node] = kind;
        self.anchors[node] = anchor;
    }

    /// The [`ViaKind`] authored for the reference at `node`, if it is a positioned reference.
    pub(crate) fn ref_kind(&self, node: StepGraphIndex) -> Option<ViaKind> {
        self.ref_kinds.get(node).cloned().flatten()
    }

    /// Re-author every reference's [`ViaKind`] against the current pick edges — used after a
    /// construction pass that changes edges AFTER anchors were set (the creation strip), so the
    /// kinds reflect the final leg topology rather than the intermediate one.
    pub(crate) fn reauthor_ref_kinds(&mut self) {
        let refs: Vec<_> = self.anchored_refs().collect();
        for (node, stored) in refs {
            self.set_anchor(node, Some(stored));
        }
    }

    /// All positioned references, ascending by node id.
    pub(crate) fn anchored_refs(
        &self,
    ) -> impl Iterator<Item = (StepGraphIndex, StoredAnchor)> + '_ {
        self.anchors
            .iter()
            .enumerate()
            .filter_map(|(node, anchor)| anchor.clone().map(|a| (node, a)))
    }

    /// Add an edge from `source` to `target` and return its stable id.
    pub(crate) fn add_edge(
        &mut self,
        source: StepGraphIndex,
        target: StepGraphIndex,
        weight: Edge,
    ) -> StepEdgeIndex {
        let id = self.edges.len();
        self.edges.push(Some(EdgeRecord {
            source,
            target,
            weight,
        }));
        self.outgoing[source].push(id);
        self.incoming[target].push(id);
        id
    }

    /// Remove the edge with `id`, returning its payload if it was still present.
    pub(crate) fn remove_edge(&mut self, id: StepEdgeIndex) -> Option<Edge> {
        let record = self.edges.get_mut(id)?.take()?;
        self.outgoing[record.source].retain(|&e| e != id);
        self.incoming[record.target].retain(|&e| e != id);
        Some(record.weight)
    }

    /// All node ids, ascending.
    pub(crate) fn node_indices(&self) -> impl Iterator<Item = StepGraphIndex> + '_ {
        0..self.nodes.len()
    }

    /// The edges touching `node` in `direction`, newest-first.
    pub(crate) fn edges_directed(
        &self,
        node: StepGraphIndex,
        direction: Direction,
    ) -> EdgesDirected<'_> {
        let list = match direction {
            Direction::Outgoing => &self.outgoing[node],
            Direction::Incoming => &self.incoming[node],
        };
        EdgesDirected {
            graph: self,
            ids: list.iter().rev(),
        }
    }

    /// The outgoing (parent-wards) edges of `node`, newest-first.
    pub(crate) fn edges(&self, node: StepGraphIndex) -> EdgesDirected<'_> {
        self.edges_directed(node, Direction::Outgoing)
    }

    /// All live edges, in edge-id order.
    pub(crate) fn edge_references(&self) -> impl Iterator<Item = StepEdgeRef<'_>> + '_ {
        self.edges
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| slot.as_ref().map(|_| self.edge_ref(id)))
    }

    /// The nodes with no edges in `direction`, ascending.
    pub(crate) fn externals(
        &self,
        direction: Direction,
    ) -> impl Iterator<Item = StepGraphIndex> + '_ {
        let lists = match direction {
            Direction::Outgoing => &self.outgoing,
            Direction::Incoming => &self.incoming,
        };
        lists
            .iter()
            .enumerate()
            .filter_map(|(idx, edges)| edges.is_empty().then_some(idx))
    }

    fn edge_ref(&self, id: StepEdgeIndex) -> StepEdgeRef<'_> {
        let record = self.edges[id]
            .as_ref()
            .expect("BUG: adjacency lists only hold live edge ids");
        StepEdgeRef {
            id,
            source: record.source,
            target: record.target,
            weight: &record.weight,
        }
    }
}

impl std::ops::Index<StepGraphIndex> for StepGraph {
    type Output = Step;
    fn index(&self, index: StepGraphIndex) -> &Self::Output {
        &self.nodes[index]
    }
}

impl std::ops::IndexMut<StepGraphIndex> for StepGraph {
    fn index_mut(&mut self, index: StepGraphIndex) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}

/// A cloneable iterator over the edges touching one node, newest-first.
#[derive(Clone)]
pub(crate) struct EdgesDirected<'graph> {
    graph: &'graph StepGraph,
    ids: std::iter::Rev<std::slice::Iter<'graph, StepEdgeIndex>>,
}

impl<'graph> Iterator for EdgesDirected<'graph> {
    type Item = StepEdgeRef<'graph>;
    fn next(&mut self) -> Option<Self::Item> {
        self.ids.next().map(|&id| self.graph.edge_ref(id))
    }
}
