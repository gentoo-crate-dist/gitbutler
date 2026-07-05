//! Operations for mutating the editor

use std::collections::HashSet;

use crate::graph_rebase::arrangement::{StackSlot, place_ref};
use crate::graph_rebase::step_graph::{ApproachKind, StoredAnchor};
use crate::graph_rebase::{Direction, StepGraphIndex, positions};
use anyhow::{Context as _, Result, anyhow, bail};
use but_core::RefMetadata;
use serde::{Deserialize, Serialize};

use crate::graph_rebase::{
    Edge, Editor, Pick, Selector, Step, ToCommitSelector, ToReferenceSelector, ToSelector,
};

/// Describes where relative to the selector a step should be inserted
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "export-schema", derive(schemars::JsonSchema))]
pub enum InsertSide {
    /// When inserting above, any nodes that point to the selector will now
    /// point to the inserted node instead.
    ///
    /// IE: Any child commits will become a child of what is getting inserted.
    Above,
    /// When inserting below, any nodes that the selector points to will now be
    /// pointed to by the inserted node instead.
    ///
    /// IE: Any parent commits will become a parent of what is getting inserted.
    Below,
}
#[cfg(feature = "export-schema")]
but_schemars::register_sdk_type!(InsertSide);

/// Controls where reparented insertion-location parents are ordered relative to
/// existing parents on the segment.
#[derive(Debug, Clone)]
pub enum ParentReparentingOrder {
    /// Put reparented insertion-location parents before existing segment parents.
    Prepend,
    /// Put reparented insertion-location parents after existing segment parents.
    Append,
}

/// Defines the start and end of a segment by pointing to it's parent-most and child-most nodes.
#[derive(Debug, Clone)]
pub struct SegmentDelimiter<C, P>
where
    C: ToSelector,
    P: ToSelector,
{
    /// The child-most node contained within the segment being defined.
    pub child: C,
    /// The parent-most node contained within the segment being defined.
    pub parent: P,
}

/// A set of some selectors
#[derive(Debug, Clone)]
pub struct SomeSelectors {
    selectors: Vec<AnySelector>,
}

impl SomeSelectors {
    /// Creates a set of selectors from different selector input types.
    ///
    /// Errors out if the selectors iterator is empty.
    pub fn new<T>(selectors: impl IntoIterator<Item = T>) -> Result<Self>
    where
        T: Into<AnySelector>,
    {
        let selectors: Vec<AnySelector> = selectors.into_iter().map(Into::into).collect();

        if selectors.is_empty() {
            return Err(anyhow!("Invalid selector set: This cannot be empty"));
        }

        Ok(Self { selectors })
    }

    /// Returns selectors as a slice.
    pub fn as_slice(&self) -> &[AnySelector] {
        &self.selectors
    }
}

/// A heterogeneous selector input.
#[derive(Debug, Clone)]
pub enum AnySelector {
    /// A selector that already points into the current graph revision.
    Selector(Selector),
    /// A commit id that should resolve to a pick step.
    Commit(gix::ObjectId),
    /// A reference name that should resolve to a reference step.
    Reference(gix::refs::FullName),
}

impl ToSelector for AnySelector {
    fn to_selector(&self, editor: &Editor<impl RefMetadata>) -> Result<Selector> {
        match self {
            Self::Selector(selector) => selector.to_selector(editor),
            Self::Commit(id) => editor.select_commit(*id),
            Self::Reference(reference) => editor.select_reference(reference.as_ref()),
        }
    }
}

impl From<Selector> for AnySelector {
    fn from(value: Selector) -> Self {
        Self::Selector(value)
    }
}

impl From<gix::ObjectId> for AnySelector {
    fn from(value: gix::ObjectId) -> Self {
        Self::Commit(value)
    }
}

impl From<gix::refs::FullName> for AnySelector {
    fn from(value: gix::refs::FullName) -> Self {
        Self::Reference(value)
    }
}

impl<T> TryFrom<Vec<T>> for SomeSelectors
where
    T: Into<AnySelector>,
{
    type Error = anyhow::Error;

    fn try_from(value: Vec<T>) -> std::result::Result<Self, Self::Error> {
        Self::new(value)
    }
}

/// Defines a set of node children or parents, to perform an action on.
///
/// Currently, this is used in the disconnect functionality.
#[derive(Debug, Clone, Default)]
pub enum SelectorSet {
    /// Select all of the children or parents.
    #[default]
    All,
    /// No children or parents should be selected.
    None,
    /// A subset of children or parents should be selected.
    Some(SomeSelectors),
}

/// An enum that is helpful for describing where something should be inserted
/// relative to.
#[derive(Debug, Clone)]
pub enum RelativeToRef<'a> {
    /// Relative to a commit
    Commit(gix::ObjectId),
    /// Relative to a reference
    Reference(&'a gix::refs::FullNameRef),
}

impl ToSelector for RelativeToRef<'_> {
    fn to_selector(&self, editor: &Editor<impl RefMetadata>) -> Result<Selector> {
        match self {
            Self::Commit(id) => editor.select_commit(*id),
            Self::Reference(reference) => editor.select_reference(reference),
        }
    }
}

/// Specifies a location relative to which a commit operation should occur.
/// This is the fully-owned cousin of [RelativeTo].
#[derive(Debug, Clone)]
pub enum RelativeTo {
    /// Relative to a commit.
    Commit(gix::ObjectId),
    /// Relative to a reference.
    Reference(gix::refs::FullName),
}

impl ToSelector for RelativeTo {
    fn to_selector(&self, editor: &Editor<impl RefMetadata>) -> Result<Selector> {
        match self {
            Self::Commit(commit) => editor.select_commit(*commit),
            Self::Reference(reference) => editor.select_reference(reference.as_ref()),
        }
    }
}

impl ToCommitSelector for gix::ObjectId {
    fn to_commit_selector(&self, editor: &Editor<impl RefMetadata>) -> Result<Selector> {
        editor.select_commit(*self)
    }
}

impl ToCommitSelector for gix::Id<'_> {
    fn to_commit_selector(&self, editor: &Editor<impl RefMetadata>) -> Result<Selector> {
        editor.select_commit(self.detach())
    }
}

impl ToSelector for gix::ObjectId {
    fn to_selector(&self, editor: &Editor<impl RefMetadata>) -> Result<Selector> {
        editor.select_commit(*self)
    }
}

impl ToSelector for gix::Id<'_> {
    fn to_selector(&self, editor: &Editor<impl RefMetadata>) -> Result<Selector> {
        editor.select_commit(self.detach())
    }
}

impl ToReferenceSelector for &gix::refs::FullNameRef {
    fn to_reference_selector(&self, editor: &Editor<impl RefMetadata>) -> Result<Selector> {
        editor.select_reference(self)
    }
}

impl ToReferenceSelector for gix::refs::FullName {
    fn to_reference_selector(&self, editor: &Editor<impl RefMetadata>) -> Result<Selector> {
        editor.select_reference(self.as_ref())
    }
}

impl ToSelector for &gix::refs::FullNameRef {
    fn to_selector(&self, editor: &Editor<impl RefMetadata>) -> Result<Selector> {
        editor.select_reference(self)
    }
}

impl ToSelector for gix::refs::FullName {
    fn to_selector(&self, editor: &Editor<impl RefMetadata>) -> Result<Selector> {
        editor.select_reference(self.as_ref())
    }
}

/// Operations for mutating the commit graph
impl<M: RefMetadata> Editor<'_, '_, M> {
    /// Get a selector to a particular commit in the graph
    pub fn select_commit(&self, target: gix::ObjectId) -> Result<Selector> {
        match self.try_select_commit(target) {
            Some(selector) => Ok(selector),
            None => Err(anyhow!("Failed to find commit {target} in rebase editor")),
        }
    }

    /// Get a selector to a particular reference in the graph
    pub fn select_reference(&self, target: &gix::refs::FullNameRef) -> Result<Selector> {
        match self.try_select_reference(target) {
            Some(selector) => Ok(selector),
            None => Err(anyhow!(
                "Failed to find reference {target} in rebase editor"
            )),
        }
    }

    /// Get a selector to a particular commit in the graph
    pub fn try_select_commit(&self, target: gix::ObjectId) -> Option<Selector> {
        for node_idx in self.graph.node_indices() {
            if let Step::Pick(Pick { id, .. }) = self.graph[node_idx]
                && id == target
            {
                return Some(self.new_selector(node_idx));
            }
        }

        None
    }

    /// Get a selector to a particular reference in the graph
    pub fn try_select_reference(&self, target: &gix::refs::FullNameRef) -> Option<Selector> {
        for node_idx in self.graph.node_indices() {
            if let Step::Reference { refname, .. } = &self.graph[node_idx]
                && target == refname.as_ref()
            {
                return Some(self.new_selector(node_idx));
            }
        }

        None
    }

    /// Returns all direct children of `target` together with their edge order.
    ///
    /// Children are represented as incoming edges into `target` in the step graph.
    pub fn direct_children(&self, target: impl ToSelector) -> Result<Vec<(Selector, usize)>> {
        let target = self.history.normalize_selector(target.to_selector(self)?)?;
        // A reference's children are the legs approaching its position (the node-era edges
        // into the reference).
        if self.graph.anchor_of(target.id).is_some() {
            return Ok(positions::ref_approach(&self.graph, target.id)
                .into_iter()
                .map(|(leg, slot)| (self.new_selector(leg), slot))
                .collect());
        }
        Ok(self
            .graph
            .edges_directed(target.id, Direction::Incoming)
            .map(|edge| (self.new_selector(edge.source()), edge.weight().order))
            .collect())
    }

    /// Returns all direct parents of `target` together with their edge order.
    ///
    /// Parents are represented as outgoing edges from `target` in the step graph.
    pub fn direct_parents(&self, target: impl ToSelector) -> Result<Vec<(Selector, usize)>> {
        let target = self.history.normalize_selector(target.to_selector(self)?)?;
        // A reference's one downward link is its anchor.
        if let Some(stored) = self.graph.anchor_of(target.id) {
            let anchor = positions::resolve_to_pick(&self.graph, stored.anchor)
                .context("Reference target should resolve to a commit")?;
            return Ok(vec![(self.new_selector(anchor), 0)]);
        }
        Ok(self
            .graph
            .edges_directed(target.id, Direction::Outgoing)
            .map(|edge| (self.new_selector(edge.target()), edge.weight().order))
            .collect())
    }

    /// The node-era parent view of `target`: reference chains interpose on the links into
    /// their anchors, exactly as the reference NODES used to.
    ///
    /// For a pick, each parent slot resolves to the top of the chain it carries (falling back
    /// to the pick it points at); for a reference, the next chain member below, then the
    /// anchor. Useful for renderers that interleave references with commits.
    pub fn position_parents(&self, target: impl ToSelector) -> Result<Vec<Selector>> {
        let target = self.history.normalize_selector(target.to_selector(self)?)?;
        if let Some(stored) = self.graph.anchor_of(target.id) {
            let anchor = positions::resolve_to_pick(&self.graph, stored.anchor)
                .context("Reference target should resolve to a commit")?;
            // The physical member directly below has the next lower rank, regardless of
            // which chain it belongs to (ranks count refs between a position and its
            // anchor).
            let below = self
                .graph
                .anchored_refs()
                .filter(|(node, other)| {
                    *node != target.id
                        && positions::resolve_to_pick(&self.graph, other.anchor) == Some(anchor)
                        && other.rank + 1 == stored.rank
                })
                .map(|(node, _)| node)
                .min();
            return Ok(vec![self.new_selector(below.unwrap_or(anchor))]);
        }
        let mut edges: Vec<_> = self
            .graph
            .edges_directed(target.id, Direction::Outgoing)
            .map(|e| (e.weight().order, e.target()))
            .collect();
        edges.sort();
        Ok(edges
            .into_iter()
            .map(|(slot, pick)| {
                let carried_top = self
                    .graph
                    .anchored_refs()
                    .filter(|(node, stored)| {
                        positions::ref_approach(&self.graph, *node).contains(&(target.id, slot))
                            && positions::resolve_to_pick(&self.graph, stored.anchor) == Some(pick)
                    })
                    .max_by_key(|(node, stored)| (stored.rank, *node))
                    .map(|(node, _)| node);
                self.new_selector(carried_top.unwrap_or(pick))
            })
            .collect())
    }

    /// The node-era child view of `target` — the inverse of [`Self::position_parents`].
    ///
    /// For a pick, its children are the bottom members of the chains anchored on it plus the
    /// plain legs into it; for a reference, the next chain member above, else its legs.
    pub fn position_children(&self, target: impl ToSelector) -> Result<Vec<Selector>> {
        let target = self.history.normalize_selector(target.to_selector(self)?)?;
        if let Some(stored) = self.graph.anchor_of(target.id) {
            let anchor = positions::resolve_to_pick(&self.graph, stored.anchor);
            // Everything that pointed at this reference in the node era: members one rank up
            // at the same anchor (chain-mates and root siblings stacked above), plus — when
            // this is the top of its own approach group — the legs that enter its chain.
            let mut out: Vec<Selector> = self
                .graph
                .anchored_refs()
                .filter(|(node, other)| {
                    *node != target.id
                        && other.rank == stored.rank + 1
                        && positions::resolve_to_pick(&self.graph, other.anchor) == anchor
                })
                .map(|(node, _)| self.new_selector(node))
                .collect();
            let target_approach = positions::ref_approach(&self.graph, target.id);
            let top_of_approach_group = !self.graph.anchored_refs().any(|(node, other)| {
                node != target.id
                    && positions::ref_approach(&self.graph, node) == target_approach
                    && other.rank > stored.rank
                    && positions::resolve_to_pick(&self.graph, other.anchor) == anchor
            });
            if top_of_approach_group {
                out.extend(
                    target_approach
                        .iter()
                        .map(|(leg, _)| self.new_selector(*leg)),
                );
            }
            out.sort_by_key(|s| s.id);
            out.dedup_by_key(|s| s.id);
            return Ok(out);
        }
        // Rank-0 members sit directly on the pick; other legs are plain.
        let mut out: Vec<Selector> = self
            .graph
            .anchored_refs()
            .filter(|(_, stored)| {
                stored.rank == 0
                    && positions::resolve_to_pick(&self.graph, stored.anchor) == Some(target.id)
            })
            .map(|(node, _)| self.new_selector(node))
            .collect();
        for edge in self.graph.edges_directed(target.id, Direction::Incoming) {
            let carrying = self.graph.anchored_refs().any(|(node, stored)| {
                positions::ref_approach(&self.graph, node)
                    .contains(&(edge.source(), edge.weight().order))
                    && positions::resolve_to_pick(&self.graph, stored.anchor) == Some(target.id)
            });
            if !carrying {
                out.push(self.new_selector(edge.source()));
            }
        }
        out.sort_by_key(|s| s.id);
        out.dedup_by_key(|s| s.id);
        Ok(out)
    }

    /// For a given step, find all the references that point to it.
    ///
    /// The reference selectors are provided in no particular order.
    pub fn step_references(&self, target: impl ToSelector) -> Result<Vec<Selector>> {
        let target = self.history.normalize_selector(target.to_selector(self)?)?;

        Ok(
            crate::graph_rebase::positions::refs_anchored_at(&self.graph, target.id)
                .into_iter()
                .map(|node| self.new_selector(node))
                .collect(),
        )
    }

    /// Replaces the node that the function was pointing to.
    ///
    /// Returns the replaced step.
    pub fn replace(&mut self, target: impl ToSelector, mut step: Step) -> Result<Step> {
        let target = self.history.normalize_selector(target.to_selector(self)?)?;
        std::mem::swap(&mut self.graph[target.id], &mut step);
        Ok(step)
    }

    /// Disconnect a segment from a parent segment.
    ///
    /// `target` - The segment to disconnect.
    /// `children_to_disconnect` - Child nodes to disconnect from `target.child`.
    /// If `SelectorSet::All`, all incoming children of `target.child` are disconnected.
    ///
    /// `parents_to_disconnect` - Parent nodes to disconnect from `target.parent`.
    /// If `SelectorSet::All`, all outgoing parents of `target.parent` are disconnected.
    ///
    /// `target` delimiter's child and parent can be the same node.
    /// This is the way to disconnect a single node.
    ///
    /// All disconnected children will be reconnected to all the disconnected parents unless
    /// the `skip_reconnect_step` is set to true.
    ///
    /// Returns an error when:
    /// - `parents_to_disconnect` is `SelectorSet::None` and `skip_reconnect_step` is false.
    /// - `parents_to_disconnect` contains any parent that is not a direct parent of `target.parent`.
    /// - `children_to_disconnect` contains any child that is not a direct parent of `target.child`.
    pub fn disconnect_segment_from<C, P>(
        &mut self,
        target: SegmentDelimiter<C, P>,
        children_to_disconnect: SelectorSet,
        parents_to_disconnect: SelectorSet,
        skip_reconnect_step: bool,
    ) -> Result<()>
    where
        C: ToSelector,
        P: ToSelector,
    {
        let SegmentDelimiter { child, parent } = target;
        let mut target_child = self.history.normalize_selector(child.to_selector(self)?)?;
        let mut target_parent = self.history.normalize_selector(parent.to_selector(self)?)?;
        // A single-node segment that is just a reference: the node-era op unhooked the
        // reference pending a reconnect. As a position: it leaves its chain (members above
        // close the gap) and gives up its legs — with a reconnect they stay as plain edges
        // onto the anchor (the node-era rewire), without one they are removed outright.
        if target_child.id == target_parent.id
            && let Some(unhooked) = self.graph.anchor_of(target_child.id)
        {
            let shifts: Vec<_> = positions::chain_members(&self.graph, target_child.id)
                .into_iter()
                .filter(|(node, m)| *node != target_child.id && m.rank > unhooked.rank)
                .collect();
            for (node, mut member) in shifts {
                member.rank -= 1;
                self.graph.set_anchor(node, Some(member));
            }
            if skip_reconnect_step
                && let Some(anchor) =
                    crate::graph_rebase::positions::resolve_to_pick(&self.graph, unhooked.anchor)
            {
                for (leg, slot) in positions::ref_approach(&self.graph, target_child.id) {
                    let removed: Vec<_> = self
                        .graph
                        .edges_directed(leg, Direction::Outgoing)
                        .filter(|e| e.target() == anchor && e.weight().order == slot)
                        .map(|e| e.id())
                        .collect();
                    for id in removed {
                        self.graph.remove_edge(id);
                    }
                }
            }
            self.graph.set_anchor(
                target_child.id,
                Some(StoredAnchor {
                    kind: ApproachKind::Root,
                    ambiguous: false,
                    ..unhooked
                }),
            );
            return Ok(());
        }
        // A reference delimiter stands for the pick it resolves to: edges are the truth for
        // picks, and the reference's chain rides the pick's links as position data. A
        // reference child only owns the legs approaching its own chain — plain edges into
        // its anchor belong to the pick and stay.
        let child_ref_stored = self.graph.anchor_of(target_child.id);
        let child_ref_approach = child_ref_stored
            .as_ref()
            .map(|_| positions::ref_approach(&self.graph, target_child.id));
        if let Some(pick) =
            crate::graph_rebase::positions::resolve_to_pick(&self.graph, target_child.id)
        {
            target_child = self.new_selector(pick);
        }
        if let Some(pick) =
            crate::graph_rebase::positions::resolve_to_pick(&self.graph, target_parent.id)
        {
            target_parent = self.new_selector(pick);
        }
        let children_to_disconnect = match children_to_disconnect {
            SelectorSet::All => None,
            SelectorSet::None => Some(Vec::new()),
            SelectorSet::Some(children) => Some(
                children
                    .as_slice()
                    .iter()
                    .map(|from_child| from_child.to_selector(self))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .map(|selector| self.history.normalize_selector(selector))
                    .collect::<Result<Vec<_>>>()?,
            ),
        };

        let parents_to_disconnect = match parents_to_disconnect {
            SelectorSet::All => None,
            SelectorSet::None => {
                if skip_reconnect_step {
                    Some(Vec::new())
                } else {
                    return Err(anyhow!(
                        "Invalid parents to disconnect: SelectorSet::None is not allowed"
                    ));
                }
            }
            SelectorSet::Some(parents) => Some(
                parents
                    .as_slice()
                    .iter()
                    .map(|from_parent| from_parent.to_selector(self))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .map(|selector| self.history.normalize_selector(selector))
                    .collect::<Result<Vec<_>>>()?,
            ),
        };

        // Edges to children.
        let incoming_edges = self
            .graph
            .edges_directed(target_child.id, Direction::Incoming)
            .map(|e| (e.id(), e.weight().to_owned(), e.source()))
            .filter(|(_, weight, source)| {
                child_ref_approach
                    .as_ref()
                    .is_none_or(|approach| approach.contains(&(*source, weight.order)))
            })
            .collect::<Vec<_>>();

        // Edges to parents.
        let outgoing_edges = self
            .graph
            .edges_directed(target_parent.id, Direction::Outgoing)
            .map(|e| (e.id(), e.weight().to_owned(), e.target()))
            .collect::<Vec<_>>();

        // All available parents
        let available_parents = outgoing_edges
            .iter()
            .map(|(_, _, edge_target)| *edge_target)
            .collect::<HashSet<_>>();
        let available_children = incoming_edges
            .iter()
            .map(|(_, _, edge_source)| *edge_source)
            .collect::<HashSet<_>>();

        // Requested selectors that are references stand for the links their positions
        // decorate: a parent reference maps to its anchor pick; a child reference maps to the
        // pick(s) approaching its chain.
        let parents_to_disconnect = parents_to_disconnect.map(|parents| {
            parents
                .into_iter()
                .map(|selector| {
                    match crate::graph_rebase::positions::resolve_to_pick(&self.graph, selector.id)
                    {
                        Some(pick) if pick != selector.id => self.new_selector(pick),
                        _ => selector,
                    }
                })
                .collect::<Vec<_>>()
        });
        // A requested child that is a reference is a chain member above the segment: its
        // legs are the edges to disconnect, and the member itself (with everything above it
        // in its chain) follows the disconnected parents.
        let mut moving_ref_children: Vec<StepGraphIndex> = Vec::new();
        let children_to_disconnect = children_to_disconnect.map(|children| {
            children
                .into_iter()
                .flat_map(|selector| match self.graph.anchor_of(selector.id) {
                    Some(_) => {
                        let legs = positions::ref_approach(&self.graph, selector.id)
                            .into_iter()
                            .map(|(child, _)| self.new_selector(child))
                            .collect::<Vec<_>>();
                        moving_ref_children.push(selector.id);
                        legs
                    }
                    None => vec![selector],
                })
                .collect::<Vec<_>>()
        });

        // 1. Verify that all parents and children to disconnect are directly connected to the target segment.
        if let Some(parents_to_disconnect) = parents_to_disconnect.as_ref() {
            for selector in parents_to_disconnect {
                if !available_parents.contains(&selector.id) {
                    return Err(anyhow!(
                        "Invalid parent delimitation: requested parent is not a direct parent of target.parent"
                    ));
                }
            }
        }

        if let Some(children_to_disconnect) = children_to_disconnect.as_ref() {
            for selector in children_to_disconnect {
                if !available_children.contains(&selector.id) {
                    return Err(anyhow!(
                        "Invalid parent delimitation: requested child is not a direct parent of target.child"
                    ));
                }
            }
        }

        let parent_ids_to_disconnect = parents_to_disconnect
            .as_ref()
            .map(|parents| parents.iter().map(|s| s.id).collect::<HashSet<_>>());
        let child_ids_to_disconnect = children_to_disconnect
            .as_ref()
            .map(|children| children.iter().map(|s| s.id).collect::<HashSet<_>>());

        let mut disconnected_parent_edges = Vec::new();
        let mut carried_parent_tops: Vec<StepGraphIndex> = Vec::new();
        // 2. Disconnect parents. Chains the removed legs carried lose them from their approach legs.
        for (edge_id, edge_weight, edge_target) in outgoing_edges {
            let should_disconnect = parent_ids_to_disconnect
                .as_ref()
                .is_none_or(|ids| ids.contains(&edge_target));
            if should_disconnect {
                let removed = (target_parent.id, edge_weight.order);
                // Chains this edge carried — captured BEFORE the edge is removed, since the
                // derived approach reflects the live edges. Removing the edge then drops the leg from
                // every derived approach automatically (no approach bookkeeping needed).
                let carried: Vec<_> = self
                    .graph
                    .anchored_refs()
                    .filter(|(node, _)| {
                        positions::ref_approach(&self.graph, *node).contains(&removed)
                    })
                    .collect();
                // The node-era parent this edge pointed at was the top of the chain it
                // carried — remember it so disconnected child refs can stack above it.
                if let Some((top, _)) = carried
                    .iter()
                    .filter(|(_, stored)| {
                        positions::resolve_to_pick(&self.graph, stored.anchor) == Some(edge_target)
                    })
                    .max_by_key(|(node, stored)| (stored.rank, *node))
                {
                    carried_parent_tops.push(*top);
                }
                self.graph.remove_edge(edge_id);
                disconnected_parent_edges.push((edge_weight, edge_target));
            }
        }

        // 3. Disconnect children and reconnect to the disconnected parents.
        let full_child_disconnect = child_ids_to_disconnect.is_none();
        let mut sorted_disconnected = disconnected_parent_edges.clone();
        sorted_disconnected.sort_by_key(|(weight, _)| weight.order);
        // The node era resolved a rewired reference through its first (lowest-slot) parent.
        let chain_anchor = sorted_disconnected.first().map(|(_, target)| *target);
        for (edge_id, edge_weight, edge_source) in incoming_edges {
            let should_disconnect = child_ids_to_disconnect
                .as_ref()
                .is_none_or(|ids| ids.contains(&edge_source));
            if !should_disconnect {
                continue;
            }
            let carrying = self.graph.anchored_refs().any(|(node, stored)| {
                positions::ref_approach(&self.graph, node)
                    .contains(&(edge_source, edge_weight.order))
                    && positions::resolve_to_pick(&self.graph, stored.anchor)
                        == Some(target_child.id)
            });
            // Remove the child edge. The chains this leg carried lose it from their derived approach
            // automatically — the edge is gone.
            self.graph.remove_edge(edge_id);
            if skip_reconnect_step {
                continue;
            }
            if carrying && child_ref_approach.is_none() && !sorted_disconnected.is_empty() {
                // A leg that carried the target's chain was, in the node era, an edge into
                // the chain — it never lost its parent slot. Fan it out to the disconnected
                // parents in place, renumbering the child's slots to make room.
                let mut entries: Vec<(usize, StepGraphIndex)> = self
                    .graph
                    .edges_directed(edge_source, Direction::Outgoing)
                    .map(|e| (e.weight().order, e.target()))
                    .collect();
                entries.sort_by_key(|(order, _)| *order);
                let insert_pos = entries.partition_point(|(o, _)| *o < edge_weight.order);
                let survivors = entries.clone();
                for (target, weight) in sorted_disconnected.iter().map(|(w, t)| (*t, w.order)).rev()
                {
                    entries.insert(insert_pos, (weight, target));
                }
                let edge_ids: Vec<_> = self
                    .graph
                    .edges_directed(edge_source, Direction::Outgoing)
                    .map(|e| e.id())
                    .collect();
                for id in edge_ids {
                    self.graph.remove_edge(id);
                }
                for (order, (_, target)) in entries.iter().enumerate() {
                    self.graph.add_edge(edge_source, *target, Edge { order });
                }
                // Surviving slots renumber; the carried chains follow onto the first
                // fan-out slot. Two-phase so shifting slots can't collide mid-flight.
                const TEMP: usize = usize::MAX / 2;
                let fanout_len = sorted_disconnected.len();
                let mut moves: Vec<(usize, usize)> = survivors
                    .iter()
                    .enumerate()
                    .map(|(i, (old_order, _))| {
                        let new_order = if i < insert_pos { i } else { i + fanout_len };
                        (*old_order, new_order)
                    })
                    .filter(|(old, new)| old != new)
                    .collect();
                moves.push((edge_weight.order, insert_pos));
                for (old, _) in &moves {
                    positions::rewrite_approach_leg(
                        &mut self.graph,
                        (edge_source, *old),
                        (edge_source, TEMP + *old),
                    );
                }
                for (old, new) in &moves {
                    positions::rewrite_approach_leg(
                        &mut self.graph,
                        (edge_source, TEMP + *old),
                        (edge_source, *new),
                    );
                }
            } else {
                // Reconnect the child node to all the disconnected parents.
                self.reconnect_edges_to_parents(&disconnected_parent_edges, edge_source);
            }
        }
        // The target's chains were the node-era direct children of its pick: a full child
        // disconnect rewires them onto the first disconnected parent, approach preserved. A
        // reference child delimiter means the segment INCLUDES that reference and its chain
        // at or below its rank — those stay with the segment.
        if let Some(anchor) = chain_anchor {
            for moving_node in &moving_ref_children {
                let Some(moving) = self.graph.anchor_of(*moving_node) else {
                    continue;
                };
                let moving_approach = positions::ref_approach(&self.graph, *moving_node);
                let moves: Vec<_> = self
                    .graph
                    .anchored_refs()
                    .filter(|(node, stored)| {
                        positions::resolve_to_pick(&self.graph, stored.anchor)
                            == Some(target_child.id)
                            && positions::ref_approach(&self.graph, *node) == moving_approach
                            && stored.rank >= moving.rank
                    })
                    .map(|(node, _)| node)
                    .collect::<Vec<_>>();
                // Re-anchor onto the disconnected parent, re-classifying each ref's lane there
                // (its legs come along, but the target may be a merge base with distinct lanes).
                for node in moves {
                    if let Some(stored) = self.graph.anchor_of(node) {
                        let approach = positions::ref_approach(&self.graph, node);
                        let mut placed = StoredAnchor::place(
                            &self.graph,
                            anchor,
                            stored.rank - moving.rank,
                            &approach,
                        );
                        placed.ambiguous = stored.ambiguous;
                        self.graph.set_anchor(node, Some(placed));
                    }
                }
            }
        }
        if full_child_disconnect && let Some(anchor) = chain_anchor {
            match &child_ref_stored {
                None => {
                    // When the disconnected parent edge carried a chain, the node-era parent
                    // was that chain's top ref — the child refs stack above it and follow it
                    // through later moves.
                    if let Some(top) = carried_parent_tops.first().copied()
                        && let Some(top_stored) = self.graph.anchor_of(top)
                    {
                        // The top ref's feeder was emptied in step 2; step 3's reconnect bridged
                        // fresh legs into `anchor`. Restore the top (and the moved refs that
                        // inherit its approach) to that full bridged leg set — every bridged leg,
                        // so it is AllLegs, correct in the merge case too.
                        let bridge = positions::legs_into_pick(&self.graph, anchor);
                        let top_rank = top_stored.rank;
                        let placed_top =
                            StoredAnchor::place(&self.graph, top_stored.anchor, top_rank, &bridge);
                        self.graph.set_anchor(top, Some(placed_top));

                        let moves: Vec<_> = self
                            .graph
                            .anchored_refs()
                            .filter(|(_, stored)| {
                                positions::resolve_to_pick(&self.graph, stored.anchor)
                                    == Some(target_child.id)
                            })
                            .map(|(node, stored)| (node, stored.rank))
                            .collect();
                        for (node, rank) in moves {
                            let placed = StoredAnchor::place(
                                &self.graph,
                                anchor,
                                rank + top_rank + 1,
                                &bridge,
                            );
                            self.graph.set_anchor(node, Some(placed));
                        }
                    } else {
                        positions::reanchor_refs_at(
                            &mut self.graph,
                            target_child.id,
                            anchor,
                            false,
                        );
                    }
                }
                Some(delimiter) => {
                    let delimiter_rank = delimiter.rank;
                    let delimiter_approach = child_ref_approach.clone().unwrap_or_default();
                    let moves: Vec<_> = self
                        .graph
                        .anchored_refs()
                        .filter(|(node, stored)| {
                            positions::resolve_to_pick(&self.graph, stored.anchor)
                                == Some(target_child.id)
                                && positions::ref_approach(&self.graph, *node) == delimiter_approach
                                && stored.rank > delimiter_rank
                        })
                        .map(|(node, _)| node)
                        .collect::<Vec<_>>();
                    // Re-anchor: preserve kind, just move onto the new pick.
                    for node in moves {
                        if let Some(mut stored) = self.graph.anchor_of(node) {
                            stored.anchor = anchor;
                            self.graph.set_anchor(node, Some(stored));
                        }
                    }
                }
            }
        }

        // 4. References whose anchor no longer resolves (it sat on the now-disconnected
        // segment) re-anchor to the segment's first disconnected parent — the ruled dangling
        // semantics: the position follows where the commit's place went, the approach (approach)
        // stays, so a rewired child renders its chain exactly as the edge-era rewire did.
        if let Some(new_anchor) = disconnected_parent_edges.first().map(|(_, target)| *target) {
            let dangling: Vec<_> = self
                .graph
                .anchored_refs()
                .filter_map(|(node, stored)| {
                    crate::graph_rebase::positions::resolve_to_pick(&self.graph, stored.anchor)
                        .is_none()
                        .then_some((node, stored))
                })
                .collect();
            for (node, mut stored) in dangling {
                stored.anchor = new_anchor;
                self.graph.set_anchor(node, Some(stored));
            }
        }

        Ok(())
    }

    /// The order to give a new outgoing edge from `node` so it sorts after all existing ones.
    fn next_outgoing_order(&self, node: StepGraphIndex) -> usize {
        self.graph
            .edges_directed(node, Direction::Outgoing)
            .map(|e| e.weight().order)
            .max()
            .map_or(0, |max| max + 1)
    }

    /// Remove the child edge, and reconnect to the right parents.
    fn reconnect_edges_to_parents(
        &mut self,
        disconnected_parent_edges: &[(Edge, StepGraphIndex)],
        child_node: StepGraphIndex,
    ) {
        // Reconnect the child node to all the disconnected parents. Their orders came from a
        // different parent context and can collide with `child_node`'s existing parents, so
        // renumber them after the highest existing order, preserving their relative order.
        let base_order = self.next_outgoing_order(child_node);
        let mut disconnected_parent_edges = disconnected_parent_edges.iter().collect::<Vec<_>>();
        disconnected_parent_edges.sort_by_key(|(weight, _)| weight.order);
        for (offset, (_, edge_target)) in disconnected_parent_edges.into_iter().enumerate() {
            self.graph.add_edge(
                child_node,
                *edge_target,
                Edge {
                    order: base_order + offset,
                },
            );
        }
    }

    /// Returns the parent-slot orders assigned to `new_parent_nodes`, in the given order.
    /// Existing parent edges that get renumbered carry their approach entries along.
    fn add_edges_to_parents(
        &mut self,
        child_node: StepGraphIndex,
        new_parent_nodes: impl IntoIterator<Item = StepGraphIndex>,
        parent_reparenting_order: ParentReparentingOrder,
    ) -> Vec<usize> {
        let mut existing_parent_edges = self
            .graph
            .edges_directed(child_node, Direction::Outgoing)
            .map(|edge| (edge.id(), edge.weight().order, edge.target()))
            .collect::<Vec<_>>();
        existing_parent_edges.sort_by_key(|(_, order, _)| *order);

        for (edge_id, _, _) in &existing_parent_edges {
            self.graph.remove_edge(*edge_id);
        }

        let new_parent_nodes = new_parent_nodes.into_iter().collect::<Vec<_>>();
        let mut new_orders = Vec::with_capacity(new_parent_nodes.len());
        let mut renumbered = Vec::new();
        match parent_reparenting_order {
            ParentReparentingOrder::Prepend => {
                for (order, parent_node) in new_parent_nodes.iter().enumerate() {
                    self.graph
                        .add_edge(child_node, *parent_node, Edge { order });
                    new_orders.push(order);
                }

                // Insertion-location parents define the first-parent lane. Existing parents stay
                // attached after them as merge-side parents.
                let shifted_by = new_parent_nodes.len();
                for (offset, (_, old_order, parent_node)) in
                    existing_parent_edges.into_iter().enumerate()
                {
                    let order = shifted_by + offset;
                    self.graph.add_edge(child_node, parent_node, Edge { order });
                    renumbered.push((old_order, order));
                }
            }
            ParentReparentingOrder::Append => {
                let shifted_by = existing_parent_edges.len();
                for (order, (_, old_order, parent_node)) in
                    existing_parent_edges.into_iter().enumerate()
                {
                    self.graph.add_edge(child_node, parent_node, Edge { order });
                    renumbered.push((old_order, order));
                }

                for (offset, parent_node) in new_parent_nodes.into_iter().enumerate() {
                    let order = shifted_by + offset;
                    self.graph.add_edge(child_node, parent_node, Edge { order });
                    new_orders.push(order);
                }
            }
        }
        // Two-phase approach rewrite so shifting slots can't collide mid-flight.
        const TEMP: usize = usize::MAX / 2;
        for (old, _) in renumbered.iter().filter(|(old, new)| old != new) {
            positions::rewrite_approach_leg(
                &mut self.graph,
                (child_node, *old),
                (child_node, TEMP + *old),
            );
        }
        for (old, new) in renumbered.iter().filter(|(old, new)| old != new) {
            positions::rewrite_approach_leg(
                &mut self.graph,
                (child_node, TEMP + *old),
                (child_node, *new),
            );
        }
        new_orders
    }

    /// Insert a segment relative to a selector.
    ///
    /// `target` - Selector to insert the segment relative to.
    ///
    /// `delimiter` - The segment is described by its delimiter: First (parent-most) and last (child-most) node.
    ///
    /// `side` - The relative side to do the insertion.
    ///
    /// `nodes_to_connect` - Optional set of selector to connect instead of the parents/children determined.
    ///
    /// `parent_reparenting_order` - Controls how newly connected parent edges are ordered relative to
    /// existing parent edges on the parent-most node of the inserted segment. With
    /// [`ParentReparentingOrder::Prepend`], the newly connected parents become the lowest-order parents,
    /// which makes the first inserted/reparented parent the first-parent traversal path. Existing parents
    /// remain attached after them in their previous relative order. With
    /// [`ParentReparentingOrder::Append`], existing parents keep the lowest parent orders and the newly
    /// connected parents are appended after them.
    ///
    /// If `nodes_to_connect` is None:
    ///     If inserted above, all the target selector's children will be disconnected and reconnected to the last
    ///     node of the segment. If inserted below, all the target selector's parents will be disconnected and
    ///     reconnected to the parent-most node of the segment using `parent_reparenting_order`.
    /// If `nodes_to_connect` is Some:
    ///     If inserted above, connect the given nodes as children. If inserted below, connect the given nodes as parents
    ///     using `parent_reparenting_order`.
    ///
    pub fn insert_segment_into<C, P>(
        &mut self,
        target: impl ToSelector,
        delimiter: SegmentDelimiter<C, P>,
        side: InsertSide,
        nodes_to_connect: Option<SomeSelectors>,
        parent_reparenting_order: ParentReparentingOrder,
    ) -> Result<()>
    where
        C: ToSelector,
        P: ToSelector,
    {
        let SegmentDelimiter { child, parent } = delimiter;
        let target = self.history.normalize_selector(target.to_selector(self)?)?;
        let child = self.history.normalize_selector(child.to_selector(self)?)?;
        let parent = self.history.normalize_selector(parent.to_selector(self)?)?;

        // An empty segment — a lone reference — is pure position data: it slots into the
        // target's chain, and any nodes to connect become its approaching legs.
        if child.id == parent.id
            && let Some(moving) = self.graph.anchor_of(child.id)
        {
            let moving_approach = positions::ref_approach(&self.graph, child.id);
            // Each arm yields the new position as (anchor, rank, approach); the moving reference's
            // legs are merged into the approach below and the whole thing classified once (all leg
            // edges are moved by then, so `place` sees the complete legs).
            let (anchor, rank, mut approach) = match (side, self.graph.anchor_of(target.id)) {
                (InsertSide::Above, Some(t_stored)) => {
                    let shifts: Vec<_> = positions::chain_members(&self.graph, target.id)
                        .into_iter()
                        .filter(|(node, m)| *node != child.id && m.rank > t_stored.rank)
                        .collect();
                    for (node, mut member) in shifts {
                        member.rank += 1;
                        self.graph.set_anchor(node, Some(member));
                    }
                    (
                        t_stored.anchor,
                        t_stored.rank + 1,
                        positions::ref_approach(&self.graph, target.id),
                    )
                }
                (InsertSide::Below, Some(t_stored)) => {
                    let shifts: Vec<_> = positions::chain_members(&self.graph, target.id)
                        .into_iter()
                        .filter(|(node, m)| *node != child.id && m.rank >= t_stored.rank)
                        .collect();
                    for (node, mut member) in shifts {
                        member.rank += 1;
                        self.graph.set_anchor(node, Some(member));
                    }
                    (
                        t_stored.anchor,
                        t_stored.rank,
                        positions::ref_approach(&self.graph, target.id),
                    )
                }
                (InsertSide::Above, None) => {
                    // The rank-0 position at the target pick; existing refs shift up.
                    let shifts: Vec<_> = self
                        .graph
                        .anchored_refs()
                        .filter_map(|(node, stored)| {
                            (node != child.id
                                && positions::resolve_to_pick(&self.graph, stored.anchor)
                                    == Some(target.id))
                            .then(|| (node, stored.clone()))
                        })
                        .collect();
                    for (node, mut member) in shifts {
                        member.rank += 1;
                        self.graph.set_anchor(node, Some(member));
                    }
                    (target.id, 0, vec![])
                }
                (InsertSide::Below, None) => {
                    // On top of the chain the target pick's first-parent leg approaches.
                    let first_parent = self
                        .graph
                        .edges_directed(target.id, Direction::Outgoing)
                        .min_by_key(|e| e.weight().order)
                        .map(|e| (e.target(), e.weight().order))
                        .context("Cannot insert a reference below a parentless commit")?;
                    let approach = vec![(target.id, first_parent.1)];
                    let rank = self
                        .graph
                        .anchored_refs()
                        .filter(|(node, _)| {
                            *node != child.id
                                && positions::ref_approach(&self.graph, *node) == approach
                        })
                        .map(|(_, stored)| stored.rank + 1)
                        .max()
                        .unwrap_or(0);
                    (first_parent.0, rank, approach)
                }
            };
            // The legs that approached the reference follow it (node-era edges pointed at
            // the reference itself), entering the chain at its new position — but only when
            // it was their sole carrier: chain members staying behind keep their approach.
            let old_anchor_pick = positions::resolve_to_pick(&self.graph, moving.anchor);
            let new_anchor_pick = positions::resolve_to_pick(&self.graph, anchor);
            let sole_carrier = !positions::chain_members(&self.graph, child.id)
                .into_iter()
                .any(|(node, m)| node != child.id && m.rank < moving.rank);
            if sole_carrier
                && let (Some(old_pick), Some(new_pick)) = (old_anchor_pick, new_anchor_pick)
            {
                for (leg, slot) in &moving_approach {
                    if old_pick != new_pick {
                        let moved: Vec<_> = self
                            .graph
                            .edges_directed(*leg, Direction::Outgoing)
                            .filter(|e| e.target() == old_pick && e.weight().order == *slot)
                            .map(|e| (e.id(), e.weight().clone()))
                            .collect();
                        for (id, weight) in moved {
                            self.graph.remove_edge(id);
                            self.graph.add_edge(*leg, new_pick, weight);
                        }
                    }
                    if !approach.contains(&(*leg, *slot)) {
                        approach.push((*leg, *slot));
                    }
                }
                approach.sort();
                // Members below in the joined chain are now approached through the moved
                // reference: they share the merged entry set.
                if matches!(side, InsertSide::Above)
                    && let Some(t_stored) = self.graph.anchor_of(target.id)
                {
                    let mates: Vec<_> = positions::chain_members(&self.graph, target.id)
                        .into_iter()
                        .filter(|(node, m)| *node != child.id && m.rank <= t_stored.rank)
                        .map(|(node, m)| (node, m.anchor, m.rank))
                        .collect();
                    for (node, m_anchor, m_rank) in mates {
                        let placed = StoredAnchor::place(&self.graph, m_anchor, m_rank, &approach);
                        self.graph.set_anchor(node, Some(placed));
                    }
                }
            }
            let placed = StoredAnchor::place(&self.graph, anchor, rank, &approach);
            self.graph.set_anchor(child.id, Some(placed));
            if let Some(nodes_to_connect) = nodes_to_connect {
                for any_selector in nodes_to_connect.as_slice() {
                    let selector = any_selector.to_selector(self)?;
                    let node = self.history.normalize_selector(selector)?;
                    let order = self.next_outgoing_order(node.id);
                    self.add_edge(node, child, order)?;
                }
            }
            return Ok(());
        }

        match side {
            InsertSide::Above => {
                // Find the child node of the highest order from the child-most node in the segment being inserted.
                let highest_order_child = self
                    .graph
                    .edges_directed(child.id, Direction::Incoming)
                    .map(|e| (e.id(), e.weight().to_owned(), e.source()))
                    .max_by_key(|(_, weight, _)| weight.order);

                if let Some(nodes_to_connect) = nodes_to_connect {
                    // If there were nodes to connect defined, create edges from them into the child node of the segment
                    // being inserted. `add_edge` gives the edge `node`'s next parent order and
                    // handles a reference child (the leg joins its chain).
                    for any_selector in nodes_to_connect.as_slice() {
                        let selector = any_selector.to_selector(self)?;
                        let node = self.history.normalize_selector(selector)?;
                        let order = self.next_outgoing_order(node.id);
                        self.add_edge(node, child, order)?;
                    }
                } else if let Some(stored) = self.graph.anchor_of(target.id) {
                    // Above a reference: split the chain there. Members above move onto the
                    // segment's child-most pick; the reference and members below are now
                    // approached by its parent-most pick.
                    let anchor_pick = positions::resolve_to_pick(&self.graph, stored.anchor)
                        .context("Reference target should resolve to a commit")?;
                    let child_pick = positions::resolve_to_pick(&self.graph, child.id)
                        .context("Segment child should resolve to a commit")?;
                    let parent_pick = positions::resolve_to_pick(&self.graph, parent.id)
                        .context("Segment parent should resolve to a commit")?;
                    let target_approach = positions::ref_approach(&self.graph, target.id);
                    let members = positions::chain_members(&self.graph, target.id);
                    let is_top = !members.iter().any(|(_, m)| m.rank > stored.rank);
                    let mut lower_members = Vec::new();
                    for (node, mut member) in members {
                        if member.rank > stored.rank {
                            member.anchor = child_pick;
                            member.rank -= stored.rank + 1;
                            self.graph.set_anchor(node, Some(member));
                        } else {
                            lower_members.push((node, member));
                        }
                    }
                    if is_top {
                        // The chain's legs now enter through the segment's child-most pick.
                        let legs: Vec<_> = self
                            .graph
                            .edge_references()
                            .filter(|e| {
                                e.target() == anchor_pick
                                    && target_approach.contains(&(e.source(), e.weight().order))
                            })
                            .map(|e| (e.id(), e.source(), e.weight().clone()))
                            .collect();
                        for (edge_id, source, weight) in legs {
                            self.graph.remove_edge(edge_id);
                            let new_weight =
                                if let Some((_, child_weight, _)) = highest_order_child.as_ref() {
                                    Edge {
                                        order: weight.order + child_weight.order + 1,
                                    }
                                } else {
                                    weight.clone()
                                };
                            let new_order = new_weight.order;
                            self.graph.add_edge(source, child_pick, new_weight);
                            if new_order != weight.order {
                                positions::rewrite_approach_leg(
                                    &mut self.graph,
                                    (source, weight.order),
                                    (source, new_order),
                                );
                            }
                        }
                    }
                    // Connect the parent-most node to the reference's anchor; a reference
                    // parent-most (an empty segment) re-anchors instead of gaining edges. The
                    // target reference and members below are now approached through that leg.
                    let entry_slot = if self.graph.anchor_of(parent.id).is_some() {
                        let anchor_selector = self.new_selector(anchor_pick);
                        self.add_edge(parent, anchor_selector, 0)?;
                        // The segment is positioned data: the split-off lower chain is
                        // approached through the segment's chain, which ends at the anchor.
                        0
                    } else {
                        let orders = self.add_edges_to_parents(
                            parent.id,
                            [anchor_pick],
                            parent_reparenting_order,
                        );
                        orders.first().copied().unwrap_or(0)
                    };
                    for (node, member) in lower_members {
                        let placed = StoredAnchor::place(
                            &self.graph,
                            member.anchor,
                            member.rank,
                            &[(parent_pick, entry_slot)],
                        );
                        self.graph.set_anchor(node, Some(placed));
                    }
                    return Ok(());
                } else {
                    let edges = self
                        .graph
                        .edges_directed(target.id, Direction::Incoming)
                        .map(|e| (e.id(), e.weight().to_owned(), e.source()))
                        .collect::<Vec<_>>();

                    // Connect all target's children with the child-most node in the given segment.
                    for (edge_id, edge_weight, edge_source) in edges {
                        self.graph.remove_edge(edge_id);
                        // Avoid weight collision by adding the order value of the highest order child plus one,
                        // accommodating for order 0.
                        let new_weight =
                            if let Some((_, child_weight, _)) = highest_order_child.as_ref() {
                                Edge {
                                    order: edge_weight.order + child_weight.order + 1,
                                }
                            } else {
                                edge_weight.clone()
                            };
                        let new_order = new_weight.order;
                        self.graph.add_edge(edge_source, child.id, new_weight);
                        if new_order != edge_weight.order {
                            positions::rewrite_approach_leg(
                                &mut self.graph,
                                (edge_source, edge_weight.order),
                                (edge_source, new_order),
                            );
                        }
                    }
                    // The target's chains slide under the segment: refs anchored on the target
                    // move up onto the segment's child-most pick.
                    if let Some(child_pick) = positions::resolve_to_pick(&self.graph, child.id) {
                        positions::reanchor_refs_at(&mut self.graph, target.id, child_pick, true);
                    }
                }

                // Connect the target to the parent-most node in the given segment according to
                // the requested parent ordering policy. A reference target stands for its
                // anchor, with the new leg entering its chain; a reference parent-most (an
                // empty segment) has no edges — it re-anchors onto the target instead.
                if self.graph.anchor_of(parent.id).is_some() {
                    self.add_edge(parent, target, 0)?;
                } else {
                    let connect_to = match self.graph.anchor_of(target.id) {
                        Some(stored) => positions::resolve_to_pick(&self.graph, stored.anchor)
                            .context("Reference target should resolve to a commit")?,
                        None => target.id,
                    };
                    let orders = self.add_edges_to_parents(
                        parent.id,
                        [connect_to],
                        parent_reparenting_order,
                    );
                    if connect_to != target.id
                        && let Some(order) = orders.first()
                    {
                        positions::join_chain_at(&mut self.graph, target.id, (parent.id, *order));
                    }
                }
            }
            InsertSide::Below => {
                let mut moved_leg_orders = Vec::new();
                let mut ref_parents: Vec<(usize, StepGraphIndex)> = Vec::new();
                let parents_to_add = if let Some(nodes_to_connect) = nodes_to_connect {
                    let mut nodes = Vec::new();
                    for any_selector in nodes_to_connect.as_slice() {
                        let selector = any_selector.to_selector(self)?;
                        let node = self.history.normalize_selector(selector)?;
                        // A reference parent: the pick edge goes to its anchor and the leg
                        // joins its chain once the final slot is known.
                        if self.graph.anchor_of(node.id).is_some() {
                            let anchor = positions::resolve_to_pick(&self.graph, node.id)
                                .context("Reference target should resolve to a commit")?;
                            ref_parents.push((nodes.len(), node.id));
                            nodes.push(anchor);
                        } else {
                            nodes.push(node.id);
                        }
                    }
                    nodes
                } else if let Some(t_stored) = self.graph.anchor_of(target.id) {
                    // A reference target's one downward link is its anchor: the segment goes
                    // between the reference and it. The reference's own re-anchoring onto the
                    // segment happens in the connect step below.
                    vec![
                        positions::resolve_to_pick(&self.graph, t_stored.anchor)
                            .context("Reference target should resolve to a commit")?,
                    ]
                } else {
                    let mut edges = self
                        .graph
                        .edges_directed(target.id, Direction::Outgoing)
                        .map(|e| (e.id(), e.weight().order, e.target()))
                        .collect::<Vec<_>>();
                    edges.sort_by_key(|(_, order, _)| *order);

                    let mut nodes = Vec::with_capacity(edges.len());
                    for (edge_id, order, edge_target) in edges {
                        self.graph.remove_edge(edge_id);
                        moved_leg_orders.push(order);
                        nodes.push(edge_target);
                    }
                    nodes
                };

                // Connect the target to the child-most node in the given segment FIRST — before the
                // segment gains its own downward parent edge below. A reference target re-anchors
                // onto the child-most, dragging its approaching legs along; if the segment's fresh
                // parent leg already existed (an `AllLegs` ref sees every leg into its anchor), that
                // leg would be dragged too and self-loop the segment. `parents_to_add` is captured
                // up front, so it still names the pre-re-anchor target position.
                // Find the child node of the highest order from the child-most node in the segment being inserted.
                let highest_order_child = self
                    .graph
                    .edges_directed(child.id, Direction::Incoming)
                    .map(|e| (e.id(), e.weight().to_owned(), e.source()))
                    .max_by_key(|(_, weight, _)| weight.order);

                let new_weight = if let Some((_, child_weight, _)) = highest_order_child.as_ref() {
                    Edge {
                        order: child_weight.order + 1,
                    }
                } else {
                    Edge { order: 0 }
                };
                // A reference child-most stands for its anchor, with the leg entering its chain.
                self.add_edge(target, child, new_weight.order)?;

                // A reference parent-most (an empty segment) has no edges — it re-anchors
                // onto its first new parent instead of gaining edges.
                if self.graph.anchor_of(parent.id).is_some() {
                    if let Some(first) = parents_to_add.first() {
                        let first = self.new_selector(*first);
                        self.add_edge(parent, first, 0)?;
                    }
                } else {
                    let new_orders = self.add_edges_to_parents(
                        parent.id,
                        parents_to_add,
                        parent_reparenting_order,
                    );
                    for (k, ref_node) in &ref_parents {
                        if let Some(order) = new_orders.get(*k) {
                            positions::join_chain_at(
                                &mut self.graph,
                                *ref_node,
                                (parent.id, *order),
                            );
                        }
                    }
                    // Chains those legs carried are now approached through the segment's
                    // parent-most pick.
                    for (old_order, new_order) in moved_leg_orders.iter().zip(new_orders) {
                        positions::rewrite_approach_leg(
                            &mut self.graph,
                            (target.id, *old_order),
                            (parent.id, new_order),
                        );
                    }
                }
            }
        }

        Ok(())
    }
    /// Insert a segment relative to a selector.
    ///
    /// The segment is described by its delimiter: First (parent-most) and last (child-most) node.
    ///
    /// If inserted above, all the target selector's children will be disconnected and reconnected to the last
    /// node of the segment.
    /// If inserted below, all the target selector's parents will be disconnected and reconnected to the
    /// parent-most node of the segment.
    ///
    /// Reparented parents are prepended by default: newly connected parents receive the lowest parent orders,
    /// so the first inserted/reparented parent becomes the first-parent traversal path and existing parents
    /// remain attached after them in their previous relative order. Use [`Self::insert_segment_into`] with
    /// [`ParentReparentingOrder::Append`] when existing parents should keep the lowest parent orders instead.
    ///
    pub fn insert_segment<C, P>(
        &mut self,
        target: impl ToSelector,
        delimiter: SegmentDelimiter<C, P>,
        side: InsertSide,
    ) -> Result<()>
    where
        C: ToSelector,
        P: ToSelector,
    {
        self.insert_segment_into(
            target,
            delimiter,
            side,
            None,
            ParentReparentingOrder::Prepend,
        )
    }

    /// Add a step node to the graph.
    ///
    /// Almost always you really want to use `insert` function instead.
    pub fn add_step(&mut self, step: Step) -> Result<Selector> {
        let new_idx = self.graph.add_node(step);
        Ok(self.new_selector(new_idx))
    }

    /// Inserts a new node relative to a selector
    ///
    /// When inserting above, any nodes that point to the selector will now
    /// point to the inserted node instead. When inserting below, any nodes
    /// that the selector points to will now be pointed to by the inserted node
    /// instead.
    ///
    /// Returns a selector to the inserted step
    pub fn insert(
        &mut self,
        target: impl ToSelector,
        step: Step,
        side: InsertSide,
    ) -> Result<Selector> {
        let target = self.history.normalize_selector(target.to_selector(self)?)?;
        let inserting_reference = matches!(step, Step::Reference { .. });
        let target_anchor = self.graph.anchor_of(target.id);
        match (side, target_anchor) {
            (InsertSide::Above, None) if !inserting_reference => {
                // Above a pick: the interposed node slides under the pick's chains — its
                // children rewire to the new node and every ref anchored on it moves up
                // (weights are preserved, so stored approach legs stay valid).
                let edges = self
                    .graph
                    .edges_directed(target.id, Direction::Incoming)
                    .map(|e| (e.id(), e.weight().to_owned(), e.source()))
                    .collect::<Vec<_>>();

                let new_idx = self.graph.add_node(step);
                self.graph.add_edge(new_idx, target.id, Edge { order: 0 });

                for (edge_id, edge_weight, edge_source) in edges {
                    self.graph.remove_edge(edge_id);
                    self.graph.add_edge(edge_source, new_idx, edge_weight);
                }
                positions::reanchor_refs_at(&mut self.graph, target.id, new_idx, false);

                Ok(self.new_selector(new_idx))
            }
            (InsertSide::Above, None) => {
                // A reference above a pick becomes the bottom of the pick's stack.
                let new_idx = self.graph.add_node(step);
                place_ref(&mut self.graph, new_idx, StackSlot::Bottom(target.id));
                Ok(self.new_selector(new_idx))
            }
            (InsertSide::Above, Some(stored)) if !inserting_reference => {
                // A pick above a reference splits the chain at that reference: members above
                // move onto the new pick, the reference and members below are now approached
                // by it.
                let anchor_pick = positions::resolve_to_pick(&self.graph, stored.anchor)
                    .context("Reference target should resolve to a commit")?;
                // Capture the target's approaching legs BEFORE adding the new pick's edge, so the
                // fresh edge does not leak into `target_approach` (which `is_top` would then re-move
                // onto itself, a self-loop).
                let target_approach = positions::ref_approach(&self.graph, target.id);
                let new_idx = self.graph.add_node(step);
                self.graph.add_edge(new_idx, anchor_pick, Edge { order: 0 });
                let members = positions::chain_members(&self.graph, target.id);
                let is_top = !members.iter().any(|(_, m)| m.rank > stored.rank);
                for (node, member) in members {
                    if member.rank > stored.rank {
                        let mut member = member;
                        member.anchor = new_idx;
                        member.rank -= stored.rank + 1;
                        self.graph.set_anchor(node, Some(member));
                    } else {
                        let placed = StoredAnchor::place(
                            &self.graph,
                            member.anchor,
                            member.rank,
                            &[(new_idx, 0)],
                        );
                        self.graph.set_anchor(node, Some(placed));
                    }
                }
                if is_top {
                    // The chain's legs now enter through the new pick.
                    let legs: Vec<_> = self
                        .graph
                        .edge_references()
                        .filter(|e| {
                            e.target() == anchor_pick
                                && target_approach.contains(&(e.source(), e.weight().order))
                        })
                        .map(|e| (e.id(), e.source(), e.weight().clone()))
                        .collect();
                    for (edge_id, source, weight) in legs {
                        self.graph.remove_edge(edge_id);
                        self.graph.add_edge(source, new_idx, weight);
                    }
                }
                Ok(self.new_selector(new_idx))
            }
            (InsertSide::Above, Some(_)) => {
                // A reference above a reference joins its chain one rank up.
                let new_idx = self.graph.add_node(step);
                place_ref(&mut self.graph, new_idx, StackSlot::Above(target.id));
                Ok(self.new_selector(new_idx))
            }
            (InsertSide::Below, None) if !inserting_reference => {
                // Below a pick: parents rewire to the new node with preserved weights, so
                // chains carried by those legs follow approach rewrites.
                let edges = self
                    .graph
                    .edges_directed(target.id, Direction::Outgoing)
                    .map(|e| (e.id(), e.weight().to_owned(), e.target()))
                    .collect::<Vec<_>>();

                let new_idx = self.graph.add_node(step);
                self.graph.add_edge(target.id, new_idx, Edge { order: 0 });

                for (edge_id, edge_weight, edge_target) in edges {
                    self.graph.remove_edge(edge_id);
                    let order = edge_weight.order;
                    self.graph.add_edge(new_idx, edge_target, edge_weight);
                    positions::rewrite_approach_leg(
                        &mut self.graph,
                        (target.id, order),
                        (new_idx, order),
                    );
                }

                Ok(self.new_selector(new_idx))
            }
            (InsertSide::Below, None) => {
                // A reference below a pick sits on top of the chain the pick's first-parent
                // leg approaches (or starts one).
                let first_parent = self
                    .graph
                    .edges_directed(target.id, Direction::Outgoing)
                    .min_by_key(|e| e.weight().order)
                    .map(|e| (e.target(), e.weight().order));
                let new_idx = self.graph.add_node(step);
                if let Some((parent_pick, slot)) = first_parent {
                    place_ref(
                        &mut self.graph,
                        new_idx,
                        StackSlot::LaneTop {
                            pick: parent_pick,
                            leg: (target.id, slot),
                        },
                    );
                }
                Ok(self.new_selector(new_idx))
            }
            (InsertSide::Below, Some(stored)) if !inserting_reference => {
                // A pick below a reference splits the chain there: the reference and members
                // above re-anchor onto the new pick, members below are approached by it.
                let anchor_pick = positions::resolve_to_pick(&self.graph, stored.anchor)
                    .context("Reference target should resolve to a commit")?;
                // Capture the target's approaching legs BEFORE the new pick's edge is added.
                let target_approach = positions::ref_approach(&self.graph, target.id);
                let new_idx = self.graph.add_node(step);
                self.graph.add_edge(new_idx, anchor_pick, Edge { order: 0 });
                let members = positions::chain_members(&self.graph, target.id);
                for (node, member) in members {
                    if member.rank >= stored.rank {
                        let mut member = member;
                        member.anchor = new_idx;
                        member.rank -= stored.rank;
                        self.graph.set_anchor(node, Some(member));
                    } else {
                        let placed = StoredAnchor::place(
                            &self.graph,
                            member.anchor,
                            member.rank,
                            &[(new_idx, 0)],
                        );
                        self.graph.set_anchor(node, Some(placed));
                    }
                }
                // The legs enter the (moved) upper part of the chain, which now rests on the
                // new pick.
                let legs: Vec<_> = self
                    .graph
                    .edge_references()
                    .filter(|e| {
                        e.target() == anchor_pick
                            && target_approach.contains(&(e.source(), e.weight().order))
                    })
                    .map(|e| (e.id(), e.source(), e.weight().clone()))
                    .collect();
                for (edge_id, source, weight) in legs {
                    self.graph.remove_edge(edge_id);
                    self.graph.add_edge(source, new_idx, weight);
                }
                Ok(self.new_selector(new_idx))
            }
            (InsertSide::Below, Some(_)) => {
                // A reference below a reference takes its position; it and everything above
                // shift up.
                let new_idx = self.graph.add_node(step);
                place_ref(&mut self.graph, new_idx, StackSlot::Below(target.id));
                Ok(self.new_selector(new_idx))
            }
        }
    }

    /// Add an edge to the graph with a desired order.
    ///
    /// Bails if there is already an edge from the child to the parent with the
    /// same order.
    pub fn add_edge(
        &mut self,
        child: impl ToSelector,
        parent: impl ToSelector,
        desired_order: usize,
    ) -> Result<()> {
        let child = self.history.normalize_selector(child.to_selector(self)?)?;
        let parent = self.history.normalize_selector(parent.to_selector(self)?)?;

        if cfg!(debug_assertions) {
            let mut seen = HashSet::from([parent.id]);
            let mut tips = vec![parent.id];

            while let Some(tip) = tips.pop() {
                for parent in self
                    .graph
                    .edges_directed(tip, Direction::Outgoing)
                    .map(|e| e.target())
                {
                    if seen.insert(parent) {
                        tips.push(parent);
                    }
                }
            }

            if seen.contains(&child.id) {
                bail!("BUG: Add edge introduces a cycle");
            }
        }

        if self
            .graph
            .edges_directed(child.id, Direction::Outgoing)
            .any(|edge| edge.weight().order == desired_order)
        {
            bail!("An edge with desired order {desired_order} already exists");
        }

        // An edge FROM a reference is its downward link: it POSITIONS the reference at the
        // parent, never a raw edge (references are positions). Only a LIVE reference does so; a
        // tombstone carrying a stale anchor (which upstream-integration retention still reads) is
        // not a reference and must not be treated as one, or the re-anchor cascades the stale
        // position through the graph.
        if matches!(self.graph[child.id], Step::Reference { .. }) {
            let new_anchor = match self.graph.anchor_of(parent.id) {
                Some(parent_stored) => {
                    positions::resolve_to_pick(&self.graph, parent_stored.anchor)
                        .context("Reference target should resolve to a commit")?
                }
                None => parent.id,
            };
            let Some(stored) = self.graph.anchor_of(child.id) else {
                // A fresh, anchorless reference gaining its first downward link (the transaction's
                // `add_step` + `add_edge` branch-creation path): position it at the target as a
                // root chain — nothing descends into it yet.
                place_ref(&mut self.graph, child.id, StackSlot::Root(new_anchor));
                return Ok(());
            };
            // An existing reference re-anchors onto the parent. The legs approaching it (node-era
            // edges into the reference) follow it; chain members left behind at the old anchor lose
            // their approach and become root chains.
            let old_anchor = positions::resolve_to_pick(&self.graph, stored.anchor);
            if let Some(old_anchor) = old_anchor
                && old_anchor != new_anchor
            {
                // Snapshot the reference's legs before moving their edges (derived approach tracks
                // live edges), so it can be re-placed against `new_anchor`'s final legs below.
                let child_approach = positions::ref_approach(&self.graph, child.id);
                for (leg, slot) in &child_approach {
                    let moved: Vec<_> = self
                        .graph
                        .edges_directed(*leg, Direction::Outgoing)
                        .filter(|e| e.target() == old_anchor && e.weight().order == *slot)
                        .map(|e| (e.id(), e.weight().clone()))
                        .collect();
                    for (id, weight) in moved {
                        self.graph.remove_edge(id);
                        self.graph.add_edge(*leg, new_anchor, weight);
                    }
                }
                let mates: Vec<_> = positions::chain_members(&self.graph, child.id)
                    .into_iter()
                    .filter(|(node, _)| *node != child.id)
                    .collect();
                for (node, mut member) in mates {
                    if member.rank < stored.rank {
                        // Left behind at the old anchor without an approach.
                        member.kind = ApproachKind::Root;
                        member.ambiguous = false;
                    } else {
                        // Stacked above the moved reference: it carries them along.
                        member.anchor = new_anchor;
                    }
                    self.graph.set_anchor(node, Some(member));
                }
                // The reference's legs moved with it; re-classify its lane against `new_anchor`'s
                // final legs (its old `Lane` slot may not exist there).
                let mut placed =
                    StoredAnchor::place(&self.graph, new_anchor, stored.rank, &child_approach);
                placed.ambiguous = stored.ambiguous;
                self.graph.set_anchor(child.id, Some(placed));
                return Ok(());
            }
            self.graph.set_anchor(
                child.id,
                Some(StoredAnchor {
                    anchor: new_anchor,
                    ..stored
                }),
            );
            return Ok(());
        }
        // An edge into a reference enters its chain: the pick edge goes to the anchor and the
        // reference (with members below it) gains the new leg. Add the pick edge FIRST so that
        // `join_chain_at`'s `set_anchor` authors the chain's `ApproachKind` against the final legs
        // (the new leg included), not the pre-edge topology.
        let parent_ref = self.graph.anchor_of(parent.id);
        let parent_pick = match &parent_ref {
            Some(stored) => positions::resolve_to_pick(&self.graph, stored.anchor)
                .context("Reference target should resolve to a commit")?,
            None => parent.id,
        };
        self.graph.add_edge(
            child.id,
            parent_pick,
            Edge {
                order: desired_order,
            },
        );
        if parent_ref.is_some() {
            positions::join_chain_at(&mut self.graph, parent.id, (child.id, desired_order));
        }

        Ok(())
    }

    /// Removes all edges between a child and parent, returning the orders of the removed edges.
    pub fn remove_edges(
        &mut self,
        child: impl ToSelector,
        parent: impl ToSelector,
    ) -> Result<Vec<usize>> {
        let child = self.history.normalize_selector(child.to_selector(self)?)?;
        let parent = self.history.normalize_selector(parent.to_selector(self)?)?;

        // A reference child holds one conceptual downward edge (order 0) — its anchor. It is
        // reported but not cleared; a follow-up add_edge re-anchors, and a position without a
        // resolving anchor is not representable.
        if let Some(stored) = self.graph.anchor_of(child.id) {
            let resolves_to_parent = positions::resolve_to_pick(&self.graph, stored.anchor)
                == positions::resolve_to_pick(&self.graph, parent.id);
            return Ok(if resolves_to_parent { vec![0] } else { vec![] });
        }
        let edges = match self.graph.anchor_of(parent.id) {
            // Disconnecting from a reference removes the legs carrying its chain — the
            // node-era edge into the reference node.
            Some(stored) => {
                let target_pick = positions::resolve_to_pick(&self.graph, stored.anchor)
                    .context("Reference target should resolve to a commit")?;
                let parent_approach = positions::ref_approach(&self.graph, parent.id);
                self.graph
                    .edges_directed(child.id, Direction::Outgoing)
                    .filter_map(|e| {
                        (e.target() == target_pick
                            && parent_approach.contains(&(child.id, e.weight().order)))
                        .then_some(e.id())
                    })
                    .collect::<Vec<_>>()
            }
            // Disconnecting from a pick removes its edges; chains riding a removed leg lose
            // it from their approach legs below.
            None => self
                .graph
                .edges_directed(child.id, Direction::Outgoing)
                .filter_map(|e| (e.target() == parent.id).then_some(e.id()))
                .collect::<Vec<_>>(),
        };

        let mut orders = vec![];
        for edge in edges {
            let weight = self
                .graph
                .remove_edge(edge)
                .context("BUG: Failed to remove edge")?;

            orders.push(weight.order);
        }
        // Chains that rode the removed legs lose them from their derived approach automatically — the
        // edges are gone, so no approach bookkeeping is needed.

        Ok(orders)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn empty_selector_set_creation_fails() {
        let empty_parent_set = SomeSelectors::new(Vec::<gix::ObjectId>::new())
            .expect_err("expected empty selector set creation to fail");
        assert!(
            empty_parent_set
                .to_string()
                .contains("Invalid selector set: This cannot be empty"),
            "unexpected error: {empty_parent_set:#}"
        );
    }
}
