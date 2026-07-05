//! An owned arena graph for rebase steps, replacing petgraph: nodes are never removed (a
//! removed step becomes [`Step::None`]), so node ids are stable by construction; edges live in
//! a slot arena so edge ids stay stable across removals. Iteration matches the semantics the
//! call sites were written against: `edges_directed` yields newest-first, `node_indices` and
//! `edge_references` ascend.

use std::collections::HashMap;

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

/// How a reference's approaching legs (its `approach`) relate to the picks that currently feed its
/// anchor — the part of a position that plain edge topology can't recover once reference edges
/// are stripped. Derived from the fresh `approach` at write time and re-resolved against the CURRENT
/// pick edges at read time, so it survives edge churn that leaves a stored `(source-pick, slot)`
/// list stale.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) enum ApproachKind {
    /// Nothing descends into this position — a chain root (`approach = []`): a remote ref stacked
    /// above a tip, an empty single-branch top.
    #[default]
    Root,
    /// Every leg into the anchor feeds this position — a plain chain, or a merge chain both
    /// lanes converge on (`approach = legs_into_pick(anchor)`).
    AllLegs,
    /// Only these specific legs feed this position — one lane of a merge. Keyed by the full
    /// `(source-pick, parent-slot)` leg, not the slot alone: two distinct sources can feed one
    /// anchor at the same slot (and one source at two slots), so both coordinates are needed to
    /// pick the right lane. Derived approach = `legs_into_pick(anchor)` intersected with this set. The
    /// source id is remapped through `graph_mapping` on rebase and re-slotted by `rewrite_approach_leg`.
    Lane(Vec<(StepGraphIndex, usize)>),
}

/// Where a reference sits, stored explicitly: references are POSITIONS, not topology. The `approach`
/// legs are DERIVED from `kind` against the live pick edges (see `positions::ref_approach`), never
/// stored — a source-pick node id in a leg list goes stale when a later op tombstones or re-slots
/// it. `kind` is authored once (creation / fresh insert, against complete legs) and PRESERVED
/// through re-anchors; `ambiguous` is a separate stored convergence bit, NOT `approach.len() > 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoredAnchor {
    /// The node this reference resolves to (a pick, or its tombstone after deletion) — the
    /// commit the ref points at, reached lazily through tombstones at read time.
    pub anchor: StepGraphIndex,
    /// Orders co-located references above one anchor, 0 = closest to the anchor.
    pub rank: usize,
    /// How this reference's approaching legs relate to the picks feeding its anchor.
    pub kind: ApproachKind,
    /// The entry into this position converged — more than one thing (legs and/or refs stacked
    /// above) met here (a merge). A creation-time signal distinct from `approach.len() > 1` (a position
    /// can converge yet resolve to a single leg), so it is stored and PRESERVED, not re-derived.
    pub ambiguous: bool,
}

impl StoredAnchor {
    /// Author a FRESH position: classify the intended `approach` against `anchor`'s CURRENT legs into a
    /// [`ApproachKind`], with `ambiguous` from the approach convergence. Only correct when the anchor's legs
    /// are already complete — use for brand-new references, never to re-place an existing one.
    pub(crate) fn place(
        graph: &StepGraph,
        anchor: StepGraphIndex,
        rank: usize,
        approach: &[(StepGraphIndex, usize)],
    ) -> Self {
        let legs = match crate::graph_rebase::positions::resolve_to_pick(graph, anchor) {
            Some(pick) => crate::graph_rebase::positions::legs_into_pick(graph, pick),
            None => Vec::new(),
        };
        StoredAnchor {
            anchor,
            rank,
            kind: crate::graph_rebase::positions::classify_approach(approach, &legs),
            ambiguous: approach.len() > 1,
        }
    }
}

/// How much of its anchor's incoming legs a lane carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LaneCarry {
    /// Nothing descends into this lane (a root chain: remote above a tip, empty top).
    None,
    /// Every leg into the anchor descends through this lane (a plain chain, or a shared
    /// chain all merge lanes converge on).
    All,
    /// This lane carries exactly `n` legs — one lane of a merge. Which legs is derived by
    /// consuming the anchor's sorted legs in lane order.
    Count(usize),
}

/// TEMP (store-swap bridge): one lane of the shadow table — the references sharing an
/// approach above one stored anchor. Membership only: order among members stays rank's job.
#[derive(Debug, Clone)]
pub(crate) struct LaneRec {
    /// The reference nodes in this lane, unordered (sort by stored rank to read).
    pub members: Vec<StepGraphIndex>,
    /// How much of the anchor's legs this lane carries.
    pub carry: LaneCarry,
    /// Bridge-era lane identity: the legs the authored [`ApproachKind::Lane`] carried at
    /// write time. Finds the lane again on later writes and orders lanes at read time; dies
    /// with the bridge (the end-state persists lane order instead).
    pub legs: Vec<(StepGraphIndex, usize)>,
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
    /// `Some` exactly for reference nodes; carries the ref's anchor, rank, kind, and ambiguity.
    anchors: Vec<Option<StoredAnchor>>,
    /// TEMP (store-swap bridge): lane membership shadowing `anchors`, keyed by the STORED
    /// (unresolved) anchor value. Maintained in [`Self::set_anchor`], read only by the
    /// `arrangement` census to prove approach is consumption-derivable before the store swaps.
    lanes: HashMap<StepGraphIndex, Vec<LaneRec>>,
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
        self.nodes.len() - 1
    }

    /// The stored position of the reference at `node`, if it is a positioned reference.
    pub(crate) fn anchor_of(&self, node: StepGraphIndex) -> Option<StoredAnchor> {
        self.anchors.get(node).cloned().flatten()
    }

    /// Set (or clear) the stored position of the reference at `node`. The [`ApproachKind`] is authored
    /// by the caller (via [`StoredAnchor::place`] for a fresh position, or preserved on an existing
    /// anchor being re-anchored), so this just stores it.
    pub(crate) fn set_anchor(&mut self, node: StepGraphIndex, anchor: Option<StoredAnchor>) {
        if let Some(previous) = &self.anchors[node] {
            let key = previous.anchor;
            self.lane_remove(node, key);
        }
        if let Some(stored) = &anchor {
            self.lane_insert(node, stored.anchor, &stored.kind);
        }
        self.anchors[node] = anchor;
    }

    /// Author a FRESH position for `node`: `approach` is the lane intent, classified against
    /// `anchor`'s CURRENT legs for the kind oracle and opening (or joining, when a lane already
    /// carries exactly these legs) the matching lane. Only correct when the anchor's legs are
    /// already complete — never use to re-place an existing position wholesale.
    pub(crate) fn place_anchor(
        &mut self,
        node: StepGraphIndex,
        anchor: StepGraphIndex,
        rank: usize,
        approach: &[(StepGraphIndex, usize)],
        ambiguous: bool,
    ) {
        let mut stored = StoredAnchor::place(self, anchor, rank, approach);
        stored.ambiguous = ambiguous;
        self.set_anchor(node, Some(stored));
    }

    /// Join `node` into the lane CONTAINING `mate` — direct membership, not legs-equality —
    /// at `rank`, copying the mate's anchor, kind, and ambiguity.
    pub(crate) fn join_lane_of(&mut self, node: StepGraphIndex, mate: StepGraphIndex, rank: usize) {
        let Some(m) = self.anchor_of(mate) else {
            return;
        };
        if let Some(previous) = &self.anchors[node] {
            let key = previous.anchor;
            self.lane_remove(node, key);
        }
        let joined = self
            .lanes
            .entry(m.anchor)
            .or_default()
            .iter_mut()
            .find(|lane| lane.members.contains(&mate))
            .map(|lane| lane.members.push(node))
            .is_some();
        if !joined {
            self.lane_insert(node, m.anchor, &m.kind);
        }
        self.anchors[node] = Some(StoredAnchor {
            anchor: m.anchor,
            rank,
            kind: m.kind,
            ambiguous: m.ambiguous,
        });
    }

    /// Re-key `node`'s position onto `new_anchor`, carrying its CURRENT lane record — the
    /// carry and legs as maintained through edge surgery, NOT re-derived from the stored kind.
    /// Rank, kind, and ambiguity are preserved.
    pub(crate) fn rekey_anchor(&mut self, node: StepGraphIndex, new_anchor: StepGraphIndex) {
        let Some(stored) = self.anchor_of(node) else {
            return;
        };
        if stored.anchor == new_anchor {
            return;
        }
        let lane_data = self.lanes.get(&stored.anchor).and_then(|lanes| {
            lanes
                .iter()
                .find(|lane| lane.members.contains(&node))
                .map(|lane| (lane.carry.clone(), lane.legs.clone()))
        });
        self.lane_remove(node, stored.anchor);
        match lane_data {
            Some((carry, legs)) => {
                let lanes = self.lanes.entry(new_anchor).or_default();
                let existing = lanes.iter_mut().find(|lane| match carry {
                    LaneCarry::Count(_) => {
                        matches!(lane.carry, LaneCarry::Count(_)) && lane.legs == legs
                    }
                    _ => lane.carry == carry,
                });
                match existing {
                    Some(lane) => lane.members.push(node),
                    None => lanes.push(LaneRec {
                        members: vec![node],
                        carry,
                        legs,
                    }),
                }
            }
            None => self.lane_insert(node, new_anchor, &stored.kind),
        }
        if let Some(a) = self.anchors[node].as_mut() {
            a.anchor = new_anchor;
        }
    }

    /// Change `node`'s rank only — pure chain reordering. The anchor key and lane membership
    /// are untouched (no lane rebuild, unlike a full re-store).
    pub(crate) fn set_rank(&mut self, node: StepGraphIndex, rank: usize) {
        if let Some(stored) = self.anchors[node].as_mut() {
            stored.rank = rank;
        }
    }

    fn lane_remove(&mut self, node: StepGraphIndex, key: StepGraphIndex) {
        let Some(lanes) = self.lanes.get_mut(&key) else {
            return;
        };
        for lane in lanes.iter_mut() {
            lane.members.retain(|&member| member != node);
        }
        lanes.retain(|lane| !lane.members.is_empty());
        if lanes.is_empty() {
            self.lanes.remove(&key);
        }
    }

    fn lane_insert(&mut self, node: StepGraphIndex, key: StepGraphIndex, kind: &ApproachKind) {
        let (carry, legs) = match kind {
            ApproachKind::Root => (LaneCarry::None, Vec::new()),
            ApproachKind::AllLegs => (LaneCarry::All, Vec::new()),
            ApproachKind::Lane(legs) => (LaneCarry::Count(legs.len()), legs.clone()),
        };
        let lanes = self.lanes.entry(key).or_default();
        let existing = lanes.iter_mut().find(|lane| match carry {
            // `Count` lanes are identified by their bridge legs (same legs => same count).
            LaneCarry::Count(_) => matches!(lane.carry, LaneCarry::Count(_)) && lane.legs == legs,
            // One `None` and one `All` lane per key.
            _ => lane.carry == carry,
        });
        match existing {
            Some(lane) => lane.members.push(node),
            None => lanes.push(LaneRec {
                members: vec![node],
                carry,
                legs,
            }),
        }
    }

    /// TEMP (store-swap bridge): the shadow lane table, for the census derivation.
    pub(crate) fn lane_table(&self) -> &HashMap<StepGraphIndex, Vec<LaneRec>> {
        &self.lanes
    }

    /// Carry every position from `source` into this graph, node ids mapped through `mapping`
    /// (an isomorphic rebuild): the lane table wholesale — members, carry, and legs as
    /// surgery maintained them, never re-derived — and each anchor alongside. Members,
    /// anchors, and leg sources that did not survive the rebuild are dropped.
    pub(crate) fn carry_positions_mapped(
        &mut self,
        source: &StepGraph,
        mapping: &HashMap<StepGraphIndex, StepGraphIndex>,
    ) {
        for (key, lanes) in &source.lanes {
            let Some(&new_key) = mapping.get(key) else {
                continue;
            };
            let mut carried = Vec::new();
            for lane in lanes {
                let members: Vec<_> = lane
                    .members
                    .iter()
                    .filter_map(|member| mapping.get(member).copied())
                    .collect();
                if members.is_empty() {
                    continue;
                }
                let legs: Vec<_> = lane
                    .legs
                    .iter()
                    .filter_map(|(src, slot)| mapping.get(src).map(|src| (*src, *slot)))
                    .collect();
                let carry = match lane.carry {
                    LaneCarry::Count(_) => LaneCarry::Count(legs.len()),
                    ref other => other.clone(),
                };
                carried.push(LaneRec {
                    members,
                    carry,
                    legs,
                });
            }
            if !carried.is_empty() {
                self.lanes.insert(new_key, carried);
            }
        }
        for (node, stored) in source.anchored_refs() {
            let (Some(&new_node), Some(&new_anchor)) =
                (mapping.get(&node), mapping.get(&stored.anchor))
            else {
                continue;
            };
            // The kind is only the shadow oracle now; a `Lane`'s legs name old-graph nodes,
            // so remap the sources to keep it comparable.
            let kind = match &stored.kind {
                ApproachKind::Lane(legs) => ApproachKind::Lane(
                    legs.iter()
                        .filter_map(|(src, slot)| mapping.get(src).map(|src| (*src, *slot)))
                        .collect(),
                ),
                other => other.clone(),
            };
            self.anchors[new_node] = Some(StoredAnchor {
                anchor: new_anchor,
                rank: stored.rank,
                kind,
                ambiguous: stored.ambiguous,
            });
        }
    }

    /// The lane containing the reference at `node`, if it holds a position.
    pub(crate) fn lane_of(&self, node: StepGraphIndex) -> Option<&LaneRec> {
        let stored = self.anchors.get(node)?.as_ref()?;
        self.lanes
            .get(&stored.anchor)?
            .iter()
            .find(|lane| lane.members.contains(&node))
    }

    /// The [`ApproachKind`] of the reference at `node`, if it is a positioned reference.
    pub(crate) fn ref_kind(&self, node: StepGraphIndex) -> Option<ApproachKind> {
        self.anchors
            .get(node)
            .and_then(|a| a.as_ref().map(|a| a.kind.clone()))
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
    ///
    /// A new edge can REVIVE a leg: ops like disconnect/reconnect drop a leg and later
    /// re-create it at the same `(source, order)`. Lanes keep naming dropped legs (reads
    /// filter against the LIVE legs), so a revived leg re-enters its lanes by itself.
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

    /// Remove the edge with `id`, returning its payload if it was still present. Lanes that
    /// named the dead leg keep naming it: lane legs are STATEMENTS, filtered against the
    /// live legs at read time, so a stale entry is inert — and reclaims the leg by itself
    /// if a later edge revives the same `(source, order)`.
    pub(crate) fn remove_edge(&mut self, id: StepEdgeIndex) -> Option<Edge> {
        let record = self.edges.get_mut(id)?.take()?;
        self.outgoing[record.source].retain(|&e| e != id);
        self.incoming[record.target].retain(|&e| e != id);
        Some(record.weight)
    }

    /// Re-target the edge with `id` (same source). Renaming duties on an order change stay
    /// with the caller ([`Self::rename_leg`]) — a target move alone leaves the leg's
    /// `(source, order)` name intact.
    pub(crate) fn move_edge(
        &mut self,
        id: StepEdgeIndex,
        new_target: StepGraphIndex,
        new_weight: Edge,
    ) {
        let Some(record) = self.edges.get_mut(id).and_then(Option::as_mut) else {
            return;
        };
        let source = record.source;
        let old_target = record.target;
        record.target = new_target;
        record.weight = new_weight;
        // Reposition in both adjacency lists exactly like a remove+add pair would (readers
        // iterate newest-first).
        self.outgoing[source].retain(|&e| e != id);
        self.outgoing[source].push(id);
        self.incoming[old_target].retain(|&e| e != id);
        self.incoming[new_target].push(id);
    }

    /// The leg `old` is now called `new` — its edge re-slotted (or re-sourced onto another
    /// pick) by surgery: every lane and every stored kind that carried `old` carries `new`
    /// instead. Callers renaming several legs on one source must two-phase through
    /// non-colliding temporaries, exactly as with edge orders.
    pub(crate) fn rename_leg(
        &mut self,
        old: (StepGraphIndex, usize),
        new: (StepGraphIndex, usize),
    ) {
        for lanes in self.lanes.values_mut() {
            for lane in lanes.iter_mut() {
                if let Some(at) = lane.legs.iter().position(|&leg| leg == old) {
                    lane.legs[at] = new;
                    lane.legs.sort_unstable();
                    lane.legs.dedup();
                    if let LaneCarry::Count(_) = lane.carry {
                        lane.carry = LaneCarry::Count(lane.legs.len());
                    }
                }
            }
        }
        for stored in self.anchors.iter_mut().flatten() {
            if let ApproachKind::Lane(legs) = &mut stored.kind
                && let Some(at) = legs.iter().position(|&leg| leg == old)
            {
                legs[at] = new;
                legs.sort_unstable();
                legs.dedup();
            }
        }
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
