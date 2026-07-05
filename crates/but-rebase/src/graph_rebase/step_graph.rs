//! An owned arena graph for rebase steps, replacing petgraph: nodes are never removed (a
//! removed step becomes [`Step::None`]), so node ids are stable by construction; edges live in
//! a slot arena so edge ids stay stable across removals. Iteration matches the semantics the
//! call sites were written against: `edges_directed` yields newest-first, `node_indices` and
//! `edge_references` ascend.

use std::collections::{HashMap, HashSet};

use crate::graph_rebase::{Edge, Step};

/// The stable identifier of a step node. Only ever grows; tombstoning is done at the
/// [`Step`] level, never by removal.
pub(crate) type StepGraphIndex = usize;

/// The stable identifier of an edge slot.
pub(crate) type StepEdgeIndex = usize;

/// One incoming child edge of a pick, named POSITIONALLY as `(source pick, parent-slot)`.
/// Lanes state legs by this name (not by edge id) so a leg removed and re-created at the
/// same coordinates is the SAME statement — see [`LaneRec::legs`].
pub(crate) type Leg = (StepGraphIndex, usize);

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

/// Where a reference sits, stored explicitly: references are POSITIONS, not topology. The
/// approach legs live in the reference's LANE (see [`StepGraph::lane_of`]), not here.
/// Derived reads live in `positions`: `ref_depth` (rank), `ref_approach` (legs),
/// `resolve_to_pick` (anchor through tombstones).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RefPosition {
    /// The node this reference resolves to (a pick, or its tombstone after deletion) — the
    /// commit the ref points at, reached lazily through tombstones at read time.
    pub anchor: StepGraphIndex,
    /// The reference directly underneath in the physical stack (`None` = sits on the anchor).
    /// Rank is DERIVED: a reference's depth is the length of its below-chain
    /// (`positions::ref_depth`).
    pub below: Option<StepGraphIndex>,
    /// The entry into this position converged — more than one thing (legs and/or refs stacked
    /// above) met here (a merge). A creation-time signal distinct from `approach.len() > 1` (a position
    /// can converge yet resolve to a single leg), so it is stored and PRESERVED, not re-derived.
    pub ambiguous: bool,
}

/// How much of its anchor's incoming legs a lane carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LaneCarry {
    /// Nothing descends into this lane (a root chain: remote above a tip, empty top).
    None,
    /// Every leg into the anchor descends through this lane (a plain chain, or a shared
    /// chain all merge lanes converge on).
    All,
    /// This lane carries exactly the legs its [`LaneRec::legs`] statement names — one lane
    /// of a merge.
    Count(usize),
}

/// One lane above a stored anchor: the references sharing an approach at one position.
/// Membership only — order among members stays the below-chain's job.
#[derive(Debug, Clone)]
pub(crate) struct LaneRec {
    /// The reference nodes in this lane, unordered (order by below-chain depth to read).
    pub members: Vec<StepGraphIndex>,
    /// How much of the anchor's legs this lane carries.
    pub carry: LaneCarry,
    /// The legs this lane STATES it carries (`Count` lanes only). Keyed by the full
    /// `(source-pick, parent-slot)` leg: two distinct sources can feed one anchor at the
    /// same slot (and one source at two slots), so both coordinates are needed. Read
    /// filtered against the anchor's LIVE legs, so a stale entry is inert — and reclaims
    /// its leg by itself when surgery revives the same coordinates.
    pub legs: Vec<Leg>,
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
    /// `Some` exactly for reference nodes; carries the ref's anchor, below, and ambiguity.
    anchors: Vec<Option<RefPosition>>,
    /// THE approach store: lane membership per STORED (unresolved) anchor value. Which legs
    /// descend into a reference's position lives here and only here — authored by
    /// [`Self::place_anchor`]/[`Self::join_lane_of`], carried by [`Self::rekey_anchor`],
    /// renamed by [`Self::rename_legs`], read via `positions::ref_approach`.
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

    /// Add a reference and return its stable id.
    pub(crate) fn add_reference(
        &mut self,
        refname: gix::refs::FullName,
        mutable: bool,
    ) -> StepGraphIndex {
        self.add_node(Step::Reference { refname, mutable })
    }

    /// The reference payload at `node` — `Some` iff the slot holds a live (non-tombstoned)
    /// reference.
    pub(crate) fn reference(&self, node: StepGraphIndex) -> Option<(&gix::refs::FullName, bool)> {
        match self.nodes.get(node)? {
            Step::Reference { refname, mutable } => Some((refname, *mutable)),
            _ => None,
        }
    }

    /// `true` iff `node` is a live reference.
    pub(crate) fn is_reference(&self, node: StepGraphIndex) -> bool {
        self.reference(node).is_some()
    }

    /// All live references, ascending by node id.
    pub(crate) fn references(
        &self,
    ) -> impl Iterator<Item = (StepGraphIndex, &gix::refs::FullName, bool)> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, step)| match step {
                Step::Reference { refname, mutable } => Some((idx, refname, *mutable)),
                _ => None,
            })
    }

    /// The step at `node` as an owned view. This is the read every whole-step consumer must
    /// use: it is the seam that will synthesize `Step::Reference` views once references leave
    /// the arena.
    pub(crate) fn step_view(&self, node: StepGraphIndex) -> Step {
        self.nodes[node].clone()
    }

    /// The stored position of the reference at `node`, if it is a positioned reference.
    pub(crate) fn anchor_of(&self, node: StepGraphIndex) -> Option<RefPosition> {
        self.anchors.get(node).cloned().flatten()
    }

    /// Author a FRESH position for `node`: `approach` is the lane intent, classified against
    /// `anchor`'s CURRENT legs — empty opens (or joins) the root lane, the whole live set the
    /// shared `All` lane, any other set a `Count` lane stating exactly those legs. Only
    /// correct when the anchor's legs are already complete — never use to re-place an
    /// existing position wholesale.
    pub(crate) fn place_anchor(
        &mut self,
        node: StepGraphIndex,
        anchor: StepGraphIndex,
        approach: &[(StepGraphIndex, usize)],
        ambiguous: bool,
        below: Option<StepGraphIndex>,
    ) {
        let live = match crate::graph_rebase::positions::resolve_to_pick(self, anchor) {
            Some(pick) => crate::graph_rebase::positions::legs_into_pick(self, pick),
            None => Vec::new(),
        };
        let (carry, legs) = if approach.is_empty() {
            (LaneCarry::None, Vec::new())
        } else {
            let approach_set: HashSet<_> = approach.iter().copied().collect();
            let live_set: HashSet<_> = live.iter().copied().collect();
            if approach_set == live_set {
                (LaneCarry::All, Vec::new())
            } else {
                let mut legs = approach.to_vec();
                legs.sort_unstable();
                legs.dedup();
                (LaneCarry::Count(legs.len()), legs)
            }
        };
        if let Some(previous) = &self.anchors[node] {
            let key = previous.anchor;
            self.lane_remove(node, key);
        }
        self.lane_insert(node, anchor, carry, legs);
        self.anchors[node] = Some(RefPosition {
            anchor,
            ambiguous,
            below,
        });
    }

    /// Join `node` into the lane CONTAINING `mate` — direct membership, not legs-equality —
    /// sitting on `below`, copying the mate's anchor and ambiguity.
    pub(crate) fn join_lane_of(
        &mut self,
        node: StepGraphIndex,
        mate: StepGraphIndex,
        below: Option<StepGraphIndex>,
    ) {
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
        debug_assert!(joined, "positioned mate {mate} must own a lane");
        if !joined {
            self.lane_insert(node, m.anchor, LaneCarry::All, Vec::new());
        }
        self.anchors[node] = Some(RefPosition {
            anchor: m.anchor,
            ambiguous: m.ambiguous,
            below,
        });
    }

    /// Re-key `node`'s position onto `new_anchor`, carrying its CURRENT lane record — the
    /// carry and legs as maintained through edge surgery. Below and ambiguity are preserved.
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
            Some((carry, legs)) => self.lane_insert(node, new_anchor, carry, legs),
            None => {
                debug_assert!(false, "positioned node {node} must own a lane");
                self.lane_insert(node, new_anchor, LaneCarry::All, Vec::new());
            }
        }
        if let Some(a) = self.anchors[node].as_mut() {
            a.anchor = new_anchor;
        }
    }

    /// Re-hang `node` onto `below` — an adjacency statement only; anchor and lane
    /// membership are untouched.
    pub(crate) fn set_below(&mut self, node: StepGraphIndex, below: Option<StepGraphIndex>) {
        if let Some(stored) = self.anchors[node].as_mut() {
            stored.below = below;
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

    fn lane_insert(
        &mut self,
        node: StepGraphIndex,
        key: StepGraphIndex,
        carry: LaneCarry,
        legs: Vec<(StepGraphIndex, usize)>,
    ) {
        let lanes = self.lanes.entry(key).or_default();
        let existing = lanes.iter_mut().find(|lane| match carry {
            // `Count` lanes are identified by their stated legs (same legs => same lane).
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

    /// The lane table: every positioned reference's approach statement, keyed by the STORED
    /// (unresolved) anchor value.
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
            self.anchors[new_node] = Some(RefPosition {
                anchor: new_anchor,
                ambiguous: stored.ambiguous,
                below: stored.below.and_then(|b| mapping.get(&b).copied()),
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

    /// All positioned references, ascending by node id.
    pub(crate) fn anchored_refs(&self) -> impl Iterator<Item = (StepGraphIndex, RefPosition)> + '_ {
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
    /// pick) by surgery: every lane that carried `old` carries `new` instead.
    pub(crate) fn rename_leg(&mut self, old: Leg, new: Leg) {
        self.rename_legs(&[(old, new)]);
    }

    /// Apply several leg renames SIMULTANEOUSLY: every lane leg is matched against the
    /// pre-rename names once, so shifting slots in a renumber can't collide mid-flight.
    pub(crate) fn rename_legs(&mut self, renames: &[(Leg, Leg)]) {
        for lanes in self.lanes.values_mut() {
            for lane in lanes.iter_mut() {
                let mut changed = false;
                for leg in lane.legs.iter_mut() {
                    if let Some((_, new)) = renames.iter().find(|(old, _)| old == leg) {
                        *leg = *new;
                        changed = true;
                    }
                }
                if changed {
                    lane.legs.sort_unstable();
                    lane.legs.dedup();
                    if let LaneCarry::Count(_) = lane.carry {
                        lane.carry = LaneCarry::Count(lane.legs.len());
                    }
                }
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
