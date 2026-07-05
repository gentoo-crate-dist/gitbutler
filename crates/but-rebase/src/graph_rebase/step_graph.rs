//! An owned arena graph for rebase steps, replacing petgraph: picks and their tombstones live
//! in the NODE arena and bear ordered parent edges; references live in the REF table and bear
//! positions. Nothing is ever removed (a removed pick becomes [`Step::None`], a removed
//! reference goes dead in place), so ids are stable by construction; edges live in a slot
//! arena so edge ids stay stable across removals. Iteration matches the semantics the call
//! sites were written against: `edges_directed` yields newest-first, `node_indices` and
//! `edge_references` ascend.

use std::collections::{HashMap, HashSet};

use crate::graph_rebase::{Edge, Step};

/// The stable identifier of a step-graph entry. Two namespaces, one id type: `Node` points
/// into the pick arena (edges are its truth), `Ref` into the reference table (a position is
/// its truth) — so a selector can address either without knowing which.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum StepGraphIndex {
    /// A pick or its tombstone in the node arena.
    Node(usize),
    /// A reference (live or dead) in the ref table.
    Ref(usize),
}

impl std::fmt::Display for StepGraphIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepGraphIndex::Node(i) => write!(f, "n{i}"),
            StepGraphIndex::Ref(i) => write!(f, "r{i}"),
        }
    }
}

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

/// One reference: name, mutability, liveness, position. Deletion flips `live` and RETAINS
/// name and position — retention is load-bearing (stale selectors normalize through dead
/// refs, rebuilds carry them).
#[derive(Debug, Clone)]
pub(crate) struct RefRecord {
    /// The full reference name.
    pub refname: gix::refs::FullName,
    /// Whether the rebase may move this reference.
    pub mutable: bool,
    /// `false` once the reference is deleted; the record stays.
    pub live: bool,
    /// The stored position, `None` until placed (creation routes connectivity through
    /// temporary ref edges first, then converts them to a position at finalize).
    pub position: Option<RefPosition>,
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

/// The rebase step graph: an arena of [`Step`]s where PICKS carry ordered parent edges, plus
/// a table of [`RefRecord`]s where REFERENCES carry explicit positions — edges are the truth
/// for commits, positions the truth for refs, with no overlap. During CREATION a reference
/// temporarily bears edges (its own adjacency lists) until
/// `positions::initialize_positions_and_strip_ref_edges` converts them to a position; from then
/// on references are edgeless.
#[derive(Debug, Clone, Default)]
pub(crate) struct StepGraph {
    nodes: Vec<Step>,
    refs: Vec<RefRecord>,
    edges: Vec<Option<EdgeRecord>>,
    outgoing: Vec<Vec<StepEdgeIndex>>,
    incoming: Vec<Vec<StepEdgeIndex>>,
    /// Creation-phase adjacency for references; empty after the finalize strip.
    ref_outgoing: Vec<Vec<StepEdgeIndex>>,
    ref_incoming: Vec<Vec<StepEdgeIndex>>,
    /// THE approach store: lane membership per STORED (unresolved) anchor value. Which legs
    /// descend into a reference's position lives here and only here — authored by
    /// [`Self::set_position`]/[`Self::join_lane_of`], carried by [`Self::rekey_position`],
    /// renamed by [`Self::rename_legs`], read via `positions::ref_approach`.
    lanes: HashMap<StepGraphIndex, Vec<LaneRec>>,
}

impl StepGraph {
    /// An empty graph.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Add `step` to the node arena and return its stable id. References do not belong here —
    /// use [`Self::add_reference`].
    pub(crate) fn add_node(&mut self, step: Step) -> StepGraphIndex {
        debug_assert!(
            !matches!(step, Step::Reference { .. }),
            "references go through add_reference, not the step arena"
        );
        self.nodes.push(step);
        self.outgoing.push(Vec::new());
        self.incoming.push(Vec::new());
        StepGraphIndex::Node(self.nodes.len() - 1)
    }

    /// Add a reference and return its stable id.
    pub(crate) fn add_reference(
        &mut self,
        refname: gix::refs::FullName,
        mutable: bool,
    ) -> StepGraphIndex {
        self.refs.push(RefRecord {
            refname,
            mutable,
            live: true,
            position: None,
        });
        self.ref_outgoing.push(Vec::new());
        self.ref_incoming.push(Vec::new());
        StepGraphIndex::Ref(self.refs.len() - 1)
    }

    /// The reference payload at `node` — `Some` iff it names a live (non-deleted) reference.
    pub(crate) fn reference(&self, node: StepGraphIndex) -> Option<(&gix::refs::FullName, bool)> {
        let StepGraphIndex::Ref(i) = node else {
            return None;
        };
        let record = self.refs.get(i)?;
        record.live.then_some((&record.refname, record.mutable))
    }

    /// `true` iff `node` is a live reference.
    pub(crate) fn is_reference(&self, node: StepGraphIndex) -> bool {
        self.reference(node).is_some()
    }

    /// `true` iff `node` is a pick — `false` for tombstones and references.
    pub(crate) fn is_pick(&self, node: StepGraphIndex) -> bool {
        match node {
            StepGraphIndex::Node(i) => matches!(self.nodes[i], Step::Pick(_)),
            StepGraphIndex::Ref(_) => false,
        }
    }

    /// All live references, ascending by id.
    pub(crate) fn references(
        &self,
    ) -> impl Iterator<Item = (StepGraphIndex, &gix::refs::FullName, bool)> + '_ {
        self.refs.iter().enumerate().filter_map(|(i, record)| {
            record
                .live
                .then_some((StepGraphIndex::Ref(i), &record.refname, record.mutable))
        })
    }

    /// All reference ids — live AND dead — ascending. Dead references still carry their
    /// retained name and position (see [`RefRecord`]).
    pub(crate) fn ref_indices(&self) -> impl Iterator<Item = StepGraphIndex> + '_ {
        (0..self.refs.len()).map(StepGraphIndex::Ref)
    }

    /// The full record of the reference at `node`, including dead ones — rebuilds need the
    /// retained payload.
    pub(crate) fn reference_record(&self, node: StepGraphIndex) -> Option<&RefRecord> {
        match node {
            StepGraphIndex::Ref(i) => self.refs.get(i),
            StepGraphIndex::Node(_) => None,
        }
    }

    /// Rename (or resurrect) the reference at `node` in place; its position is untouched.
    pub(crate) fn set_reference(
        &mut self,
        node: StepGraphIndex,
        refname: gix::refs::FullName,
        mutable: bool,
    ) {
        let StepGraphIndex::Ref(i) = node else {
            panic!("BUG: only references can be renamed");
        };
        let record = &mut self.refs[i];
        record.refname = refname;
        record.mutable = mutable;
        record.live = true;
    }

    /// Delete the reference at `node`: it goes dead in place, RETAINING its name and
    /// position so stale selectors keep normalizing and rebuilds keep carrying it.
    pub(crate) fn tombstone_reference(&mut self, node: StepGraphIndex) {
        let StepGraphIndex::Ref(i) = node else {
            panic!("BUG: only references can be tombstoned");
        };
        self.refs[i].live = false;
    }

    /// The step at `node` as an owned view — the read for whole-step consumers. Reference
    /// entries synthesize their step: `Step::Reference` while live, `Step::None` once dead.
    pub(crate) fn step_view(&self, node: StepGraphIndex) -> Step {
        match node {
            StepGraphIndex::Node(i) => self.nodes[i].clone(),
            StepGraphIndex::Ref(i) => {
                let record = &self.refs[i];
                if record.live {
                    Step::Reference {
                        refname: record.refname.clone(),
                        mutable: record.mutable,
                    }
                } else {
                    Step::None
                }
            }
        }
    }

    /// The stored position of the reference at `node`, live or dead.
    pub(crate) fn position_of(&self, node: StepGraphIndex) -> Option<RefPosition> {
        match node {
            StepGraphIndex::Ref(i) => self.refs.get(i)?.position.clone(),
            StepGraphIndex::Node(_) => None,
        }
    }

    fn position_slot(&mut self, node: StepGraphIndex) -> &mut Option<RefPosition> {
        match node {
            StepGraphIndex::Ref(i) => &mut self.refs[i].position,
            StepGraphIndex::Node(_) => panic!("BUG: only references hold positions"),
        }
    }

    /// Author a FRESH position for `node`: `approach` is the lane intent, classified against
    /// `anchor`'s CURRENT legs — empty opens (or joins) the root lane, the whole live set the
    /// shared `All` lane, any other set a `Count` lane stating exactly those legs. Only
    /// correct when the anchor's legs are already complete — never use to re-place an
    /// existing position wholesale.
    pub(crate) fn set_position(
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
        if let Some(previous) = self.position_of(node) {
            self.lane_remove(node, previous.anchor);
        }
        self.lane_insert(node, anchor, carry, legs);
        *self.position_slot(node) = Some(RefPosition {
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
        let Some(m) = self.position_of(mate) else {
            return;
        };
        if let Some(previous) = self.position_of(node) {
            self.lane_remove(node, previous.anchor);
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
        *self.position_slot(node) = Some(RefPosition {
            anchor: m.anchor,
            ambiguous: m.ambiguous,
            below,
        });
    }

    /// Re-key `node`'s position onto `new_anchor`, carrying its CURRENT lane record — the
    /// carry and legs as maintained through edge surgery. Below and ambiguity are preserved.
    pub(crate) fn rekey_position(&mut self, node: StepGraphIndex, new_anchor: StepGraphIndex) {
        let Some(stored) = self.position_of(node) else {
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
        if let Some(a) = self.position_slot(node).as_mut() {
            a.anchor = new_anchor;
        }
    }

    /// Re-hang `node` onto `below` — an adjacency statement only; anchor and lane
    /// membership are untouched.
    pub(crate) fn set_below(&mut self, node: StepGraphIndex, below: Option<StepGraphIndex>) {
        if let Some(stored) = self.position_slot(node).as_mut() {
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

    /// Carry every position from `source` into this graph, ids mapped through `mapping`
    /// (an isomorphic rebuild): the lane table wholesale — members, carry, and legs as
    /// surgery maintained them, never re-derived — and each position alongside. Members,
    /// anchors, and leg sources that did not survive the rebuild are dropped.
    ///
    /// `leg_map` translates live `(source node, stored order)` names to the dense slots the
    /// rebuild wrote; legs it does not name (stale statements) carry verbatim — retention is
    /// deliberate, and a dead name stays equally dead under any numbering.
    pub(crate) fn carry_positions_mapped(
        &mut self,
        source: &StepGraph,
        mapping: &HashMap<StepGraphIndex, StepGraphIndex>,
        leg_map: &HashMap<(StepGraphIndex, usize), usize>,
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
                    .filter_map(|&(src, slot)| {
                        let new_src = mapping.get(&src)?;
                        let new_slot = leg_map.get(&(src, slot)).copied().unwrap_or(slot);
                        Some((*new_src, new_slot))
                    })
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
        for (node, stored) in source.positioned_refs() {
            let (Some(&new_node), Some(&new_anchor)) =
                (mapping.get(&node), mapping.get(&stored.anchor))
            else {
                continue;
            };
            *self.position_slot(new_node) = Some(RefPosition {
                anchor: new_anchor,
                ambiguous: stored.ambiguous,
                below: stored.below.and_then(|b| mapping.get(&b).copied()),
            });
        }
    }

    /// The lane containing the reference at `node`, if it holds a position.
    pub(crate) fn lane_of(&self, node: StepGraphIndex) -> Option<&LaneRec> {
        let stored = match node {
            StepGraphIndex::Ref(i) => self.refs.get(i)?.position.as_ref()?,
            StepGraphIndex::Node(_) => return None,
        };
        self.lanes
            .get(&stored.anchor)?
            .iter()
            .find(|lane| lane.members.contains(&node))
    }

    /// All positioned references — live AND dead — ascending by id.
    pub(crate) fn positioned_refs(
        &self,
    ) -> impl Iterator<Item = (StepGraphIndex, RefPosition)> + '_ {
        self.refs
            .iter()
            .enumerate()
            .filter_map(|(i, record)| record.position.clone().map(|p| (StepGraphIndex::Ref(i), p)))
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
        self.adjacency_mut(source, Direction::Outgoing).push(id);
        self.adjacency_mut(target, Direction::Incoming).push(id);
        id
    }

    /// Remove the edge with `id`, returning its payload if it was still present. Lanes that
    /// named the dead leg keep naming it: lane legs are STATEMENTS, filtered against the
    /// live legs at read time, so a stale entry is inert — and reclaims the leg by itself
    /// if a later edge revives the same `(source, order)`.
    pub(crate) fn remove_edge(&mut self, id: StepEdgeIndex) -> Option<Edge> {
        let record = self.edges.get_mut(id)?.take()?;
        self.adjacency_mut(record.source, Direction::Outgoing)
            .retain(|&e| e != id);
        self.adjacency_mut(record.target, Direction::Incoming)
            .retain(|&e| e != id);
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
        self.adjacency_mut(source, Direction::Outgoing)
            .retain(|&e| e != id);
        self.adjacency_mut(source, Direction::Outgoing).push(id);
        self.adjacency_mut(old_target, Direction::Incoming)
            .retain(|&e| e != id);
        self.adjacency_mut(new_target, Direction::Incoming).push(id);
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

    // --- Slot-based parent primitives (stage A of the parent-array swap) ---
    //
    // A node's outgoing edges, read and mutated as a DENSE ordered array of parent slots.
    // Every mutator first normalizes the child (slots renumbered to 0..n with statements
    // renamed along; statements naming a nonexistent slot purged), so after any primitive
    // touch the slot names ARE the array positions. A removed slot's statements are dropped
    // — revival is an explicit op-level re-statement, never a store-level coincidence.

    /// The parent-side edge ids of `node` in slot order: ascending stored order, ties
    /// newest-first (the order every sorted consumer reads).
    fn parent_edge_ids(&self, node: StepGraphIndex) -> Vec<StepEdgeIndex> {
        let mut ids: Vec<StepEdgeIndex> = self
            .adjacency(node, Direction::Outgoing)
            .iter()
            .rev()
            .copied()
            .collect();
        ids.sort_by_key(|&id| self.edge_ref(id).weight.order);
        ids
    }

    /// The parents of `node` in slot order.
    pub(crate) fn parents(&self, node: StepGraphIndex) -> Vec<StepGraphIndex> {
        self.parent_edge_ids(node)
            .into_iter()
            .map(|id| self.edge_ref(id).target)
            .collect()
    }

    /// How many parent slots `node` has.
    pub(crate) fn parent_count(&self, node: StepGraphIndex) -> usize {
        self.adjacency(node, Direction::Outgoing).len()
    }

    /// `(stored order, parent)` per slot, in slot order — the bridge read for consumers that
    /// must translate stored names to dense slots (dies with the store swap).
    pub(crate) fn parent_orders(&self, node: StepGraphIndex) -> Vec<(usize, StepGraphIndex)> {
        self.parent_edge_ids(node)
            .into_iter()
            .map(|id| {
                let record = self.edge_ref(id);
                (record.weight.order, record.target)
            })
            .collect()
    }

    /// Every edge into `node` as `(child, slot)` legs, sorted for determinism. Slots are the
    /// children's STORED orders — align via [`Self::normalize_parent_slots`] first when a
    /// statement-exact name is needed.
    pub(crate) fn incoming_legs(&self, node: StepGraphIndex) -> Vec<(StepGraphIndex, usize)> {
        let mut legs: Vec<_> = self
            .edges_directed(node, Direction::Incoming)
            .map(|e| (e.source(), e.weight().order))
            .collect();
        legs.sort_unstable();
        legs
    }

    /// Renumber `child`'s parent slots to dense `0..n`, renaming live statements along and
    /// purging statements that name a slot `child` does not hold. Returns the edge ids in
    /// slot order. Duplicate stored orders make slot names ambiguous and are a bug.
    ///
    /// Every slot mutator calls this itself; ops call it directly as a BRIDGE when they need
    /// slot names and statements aligned before reading (a no-op once every writer authors
    /// dense slots — it dies with the store swap).
    pub(crate) fn normalize_parent_slots(&mut self, child: StepGraphIndex) -> Vec<StepEdgeIndex> {
        let ids = self.parent_edge_ids(child);
        let live: HashSet<usize> = ids
            .iter()
            .map(|&id| self.edge_ref(id).weight.order)
            .collect();
        debug_assert_eq!(
            live.len(),
            ids.len(),
            "duplicate parent orders at {child} make slot names ambiguous"
        );
        self.retain_legs(|&(source, slot)| source != child || live.contains(&slot));
        let mut renames = Vec::new();
        for (slot, &id) in ids.iter().enumerate() {
            let record = self.edges[id]
                .as_mut()
                .expect("adjacency lists only hold live edge ids");
            if record.weight.order != slot {
                renames.push(((child, record.weight.order), (child, slot)));
                record.weight.order = slot;
            }
        }
        self.rename_legs(&renames);
        ids
    }

    /// Append `parent` as `child`'s last parent slot; returns the slot.
    pub(crate) fn push_parent(&mut self, child: StepGraphIndex, parent: StepGraphIndex) -> usize {
        self.normalize_parent_slots(child);
        let slot = self.parent_count(child);
        self.add_edge(child, parent, Edge { order: slot });
        slot
    }

    /// Insert `parent` at `slot` of `child` (clamped to the array end); later slots shift up
    /// with their statements. Returns the slot actually used.
    pub(crate) fn insert_parent(
        &mut self,
        child: StepGraphIndex,
        slot: usize,
        parent: StepGraphIndex,
    ) -> usize {
        let ids = self.normalize_parent_slots(child);
        let slot = slot.min(ids.len());
        let mut renames = Vec::new();
        for &id in &ids[slot..] {
            let record = self.edges[id]
                .as_mut()
                .expect("adjacency lists only hold live edge ids");
            let old = record.weight.order;
            record.weight.order = old + 1;
            renames.push(((child, old), (child, old + 1)));
        }
        self.rename_legs(&renames);
        self.add_edge(child, parent, Edge { order: slot });
        slot
    }

    /// Remove `child`'s parent at `slot`, returning it; later slots shift down with their
    /// statements, and statements naming the removed slot are dropped.
    pub(crate) fn remove_parent(
        &mut self,
        child: StepGraphIndex,
        slot: usize,
    ) -> Option<StepGraphIndex> {
        let ids = self.normalize_parent_slots(child);
        let &id = ids.get(slot)?;
        let target = self.edge_ref(id).target;
        self.remove_edge(id);
        self.retain_legs(|&leg| leg != (child, slot));
        let mut renames = Vec::new();
        for &id in &ids[slot + 1..] {
            let record = self.edges[id]
                .as_mut()
                .expect("adjacency lists only hold live edge ids");
            let old = record.weight.order;
            record.weight.order = old - 1;
            renames.push(((child, old), (child, old - 1)));
        }
        self.rename_legs(&renames);
        Some(target)
    }

    /// Re-point `child`'s parent at `slot` onto `new_parent`. The slot — and so the
    /// statement name — is untouched: chains stated on the leg follow it to its new target.
    pub(crate) fn replace_parent(
        &mut self,
        child: StepGraphIndex,
        slot: usize,
        new_parent: StepGraphIndex,
    ) {
        let ids = self.normalize_parent_slots(child);
        let Some(&id) = ids.get(slot) else {
            debug_assert!(false, "replace_parent: {child} has no slot {slot}");
            return;
        };
        self.move_edge(id, new_parent, Edge { order: slot });
    }

    /// Move `from`'s whole parent array onto `to` (which must have none); statements follow
    /// slot-for-slot.
    pub(crate) fn transplant_parents(&mut self, from: StepGraphIndex, to: StepGraphIndex) {
        debug_assert_eq!(
            self.parent_count(to),
            0,
            "transplant target {to} already has parents"
        );
        let ids = self.normalize_parent_slots(from);
        let mut renames = Vec::new();
        for (slot, id) in ids.into_iter().enumerate() {
            let target = self.edge_ref(id).target;
            self.remove_edge(id);
            self.add_edge(to, target, Edge { order: slot });
            renames.push(((from, slot), (to, slot)));
        }
        self.rename_legs(&renames);
    }

    /// Re-target every edge into `from` onto `to`, slots preserved — statement names are
    /// `(source, slot)`, so they stay valid untouched.
    pub(crate) fn redirect_children(&mut self, from: StepGraphIndex, to: StepGraphIndex) {
        let ids: Vec<StepEdgeIndex> = self.adjacency(from, Direction::Incoming).to_vec();
        for id in ids {
            let weight = self.edge_ref(id).weight.clone();
            self.move_edge(id, to, weight);
        }
    }

    /// Drop every lane statement `keep` rejects, keeping `Count` carries consistent.
    fn retain_legs(&mut self, keep: impl Fn(&Leg) -> bool) {
        for lanes in self.lanes.values_mut() {
            for lane in lanes.iter_mut() {
                let before = lane.legs.len();
                lane.legs.retain(&keep);
                if lane.legs.len() != before
                    && let LaneCarry::Count(_) = lane.carry
                {
                    lane.carry = LaneCarry::Count(lane.legs.len());
                }
            }
        }
    }

    /// All node-arena ids (picks and tombstones), ascending. References are NOT included —
    /// see [`Self::references`] and [`Self::ref_indices`].
    pub(crate) fn node_indices(&self) -> impl Iterator<Item = StepGraphIndex> + '_ {
        (0..self.nodes.len()).map(StepGraphIndex::Node)
    }

    fn adjacency(&self, node: StepGraphIndex, direction: Direction) -> &[StepEdgeIndex] {
        match (node, direction) {
            (StepGraphIndex::Node(i), Direction::Outgoing) => &self.outgoing[i],
            (StepGraphIndex::Node(i), Direction::Incoming) => &self.incoming[i],
            (StepGraphIndex::Ref(i), Direction::Outgoing) => &self.ref_outgoing[i],
            (StepGraphIndex::Ref(i), Direction::Incoming) => &self.ref_incoming[i],
        }
    }

    fn adjacency_mut(
        &mut self,
        node: StepGraphIndex,
        direction: Direction,
    ) -> &mut Vec<StepEdgeIndex> {
        match (node, direction) {
            (StepGraphIndex::Node(i), Direction::Outgoing) => &mut self.outgoing[i],
            (StepGraphIndex::Node(i), Direction::Incoming) => &mut self.incoming[i],
            (StepGraphIndex::Ref(i), Direction::Outgoing) => &mut self.ref_outgoing[i],
            (StepGraphIndex::Ref(i), Direction::Incoming) => &mut self.ref_incoming[i],
        }
    }

    /// The edges touching `node` in `direction`, newest-first.
    pub(crate) fn edges_directed(
        &self,
        node: StepGraphIndex,
        direction: Direction,
    ) -> EdgesDirected<'_> {
        EdgesDirected {
            graph: self,
            ids: self.adjacency(node, direction).iter().rev(),
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

    /// The ARENA nodes with no edges in `direction`, ascending. References never appear
    /// here — post-strip they are edgeless by construction, and the consumers (root/head
    /// discovery) want picks and tombstones only.
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
            .filter_map(|(idx, edges)| edges.is_empty().then_some(StepGraphIndex::Node(idx)))
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
        match index {
            StepGraphIndex::Node(i) => &self.nodes[i],
            StepGraphIndex::Ref(_) => {
                panic!("BUG: references live in the ref table, not the step arena")
            }
        }
    }
}

impl std::ops::IndexMut<StepGraphIndex> for StepGraph {
    fn index_mut(&mut self, index: StepGraphIndex) -> &mut Self::Output {
        match index {
            StepGraphIndex::Node(i) => &mut self.nodes[i],
            StepGraphIndex::Ref(_) => {
                panic!("BUG: references live in the ref table, not the step arena")
            }
        }
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
