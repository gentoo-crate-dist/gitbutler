//! A commit-first graph flattened out of the raw traversal — the substrate every
//! [`Graph`](crate::Graph) is built from (see `commit_graph_to_segment_graph`).
//!
//! # Why
//!
//! Today the pipeline is `gix traversal → SegmentGraph (segments own commit ranges) → projection`,
//! and but-rebase builds its `StepGraph` from that same segment graph. The segment layer turned out
//! to be an *artifact of incremental construction*, not something either consumer fundamentally
//! needs:
//!
//! * **StepGraph** is already commit/ref-granular — its nodes are `Pick(commit)` / `Reference(ref)`
//!   and its edges carry only the parent-array `order`. It re-derives parent order from
//!   `commit.parent_ids` and even *corrects* but-graph when they disagree. It needs: commit id,
//!   parent ids (first-parent at `[0]`), the refs on each commit, an entrypoint, and parent-walk
//!   reachability. No segment boundaries, no segment ids.
//!
//! * **Projection** emits segment-shaped output (`Stack`/`StackSegment`), but the segmentation is
//!   recomputable: a segment is a maximal first-parent run, split where a local-branch ref appears,
//!   at branch/merge points, and at the projection's own stops (entrypoint, merge-base, target).
//!   `generation`, merge-base, and remote-reachability are commit-level. `sibling_segment_id` /
//!   `remote_tracking_branch_segment_id` are just cached pointers — recomputable by ref-name match.
//!
//! So both can build straight from the commit DAG, and segments become a *view* produced during
//! projection rather than a stored graph.
//!
//! # The model
//!
//! A node is a commit (we reuse [`crate::Commit`]) plus its topological `generation`. An edge is
//! simply `commit → parent`, taken from `parent_ids` (first-parent at index 0) — there is no
//! `src`/`dst` within-segment payload to carry, because there are no segments to index into.
//!
//! A nice consequence: the **workspace commit's `parent_ids` array is the stack order**, so the
//! order-stacks machinery ([`crate::Graph`] post-pass) disappears — the order is read straight off
//! the merge commit's parents.
//!
//! Historically this was a standalone spike toward deleting the segment graph outright; today the
//! production builders source it from the REAL traversal ([`CommitGraph::from_walk`]) and rebuild
//! the full segment graph on top, so downstream consumers are unchanged. The commit-first model
//! remains the intended shape for the eventual but-graph/but-rebase unification.

use std::collections::{HashMap, HashSet};

use crate::{Commit, CommitFlags};

/// An index into a [`CommitGraph`]'s node arena.
pub type CommitIdx = usize;

/// A node in the commit graph: a commit, plus where it sits topologically.
#[derive(Debug, Clone)]
pub struct CommitNode {
    /// The commit itself — `id`, `parent_ids` (first-parent at `[0]`), `flags`, and the `refs`
    /// pointing at it. This is exactly the data a segment used to hold per-commit.
    pub commit: Commit,
    /// Distance from a root (a commit with no parents in the graph). Higher means deeper in history.
    /// Used where the projection picks "the lowest of several tips"; cheap to compute during build.
    pub generation: u32,
}

/// A commit-first graph: an arena of commits keyed by id, with `commit → parent` edges read from
/// each node's `parent_ids` and the reverse (`parent → child`) adjacency derived for downward walks.
#[derive(Debug, Clone, Default)]
pub struct CommitGraph {
    nodes: Vec<CommitNode>,
    by_id: HashMap<gix::ObjectId, CommitIdx>,
    /// `parent → children` adjacency, derived at build time so we can detect branch points and walk
    /// downward (the projection walks from the workspace tip toward the base).
    children: Vec<Vec<CommitIdx>>,
    /// Where traversal/HEAD started; the projection uses it as a focus boundary.
    entrypoint: Option<gix::ObjectId>,
    /// The ref the entrypoint was checked out as, if any. When set, it names the entrypoint segment
    /// (overriding disambiguation), mirroring `from_commit_traversal(id, Some(ref))`.
    entrypoint_ref: Option<gix::refs::FullName>,
    /// Commits whose message marks them as a GitButler-managed workspace commit. Kept out of
    /// [`CommitFlags`](crate::CommitFlags) so it neither perturbs the walk's goal bits nor the
    /// segment fingerprint; used to tell a real managed merge from a ws ref advanced past it.
    managed_ws_commits: HashSet<gix::ObjectId>,
    /// `(child, parent)` pairs the traversal actually CONNECTED, when built
    /// [from the walk](Self::from_walk). A commit's raw `parent_ids` can point past a traversal
    /// cut (limit, integrated stop-early); connectivity accessors must not rejoin what the walk
    /// severed. `None` for graphs built directly from commits (all raw parents count).
    connected: Option<HashSet<(gix::ObjectId, gix::ObjectId)>>,
    /// When built [from the walk](Self::from_walk): whether the traversal stopped queueing after
    /// hitting the hard limit. Derived graphs must carry it onto the final `Graph`.
    pub(crate) hard_limit_hit: bool,
    /// When built [from the walk](Self::from_walk): the traversal's normalized seed tips. Graphs
    /// built from EXPLICIT tips must carry them onto the final `Graph` — the projection reads tip
    /// roles (e.g. integrated tips) for such graphs.
    pub(crate) traversal_tips: Vec<crate::init::Tip>,
    /// Built from EXPLICIT tips ([`Self::from_walk_tips`]): every tip must start (or get) its own
    /// segment. Workspace-discovered builds must NOT carve boundaries at their normalized tips —
    /// the walk merges tip-seeded segments back in post-processing.
    pub(crate) explicit_tips: bool,
}

impl CommitGraph {
    /// Build from a set of commits (as produced by the gix traversal). Commits whose parents are
    /// outside the set are simply roots of this subgraph (a partial graph), mirroring how the
    /// StepGraph handles missing parents via `preserved_parents`.
    pub fn from_commits(
        commits: impl IntoIterator<Item = Commit>,
        entrypoint: Option<gix::ObjectId>,
    ) -> Self {
        let nodes: Vec<CommitNode> = commits
            .into_iter()
            .map(|commit| CommitNode {
                commit,
                generation: 0,
            })
            .collect();
        let by_id: HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(idx, n)| (n.commit.id, idx))
            .collect();

        // Reverse adjacency: for each node, record it as a child of every parent that is present.
        let mut children = vec![Vec::new(); nodes.len()];
        for (idx, n) in nodes.iter().enumerate() {
            for parent in &n.commit.parent_ids {
                if let Some(&pidx) = by_id.get(parent) {
                    children[pidx].push(idx);
                }
            }
        }

        let mut graph = CommitGraph {
            nodes,
            by_id,
            children,
            entrypoint,
            entrypoint_ref: None,
            managed_ws_commits: HashSet::new(),
            connected: None,
            hard_limit_hit: false,
            traversal_tips: Vec::new(),
            explicit_tips: false,
        };
        graph.recompute_generations();
        graph
    }

    /// Bridge: build a commit graph from the existing segment graph, so the StepGraph and
    /// projection builders can be exercised against it without first rewriting traversal. Every
    /// segment's commits become nodes; their `parent_ids` are the edges, and the entrypoint commit
    /// carries over.
    pub fn from_segment_graph(graph: &crate::Graph) -> Self {
        let ep = graph.entrypoint().ok();
        let entrypoint = ep
            .as_ref()
            .and_then(|ep| ep.commit_and_owner.map(|(c, _)| c.id));
        // The entrypoint segment's own ref names it (e.g. a checkout of a specific branch inside a
        // stack); the owner is the segment holding the entrypoint commit.
        let entrypoint_ref = ep
            .as_ref()
            .and_then(|ep| ep.commit_and_owner)
            .and_then(|(_, owner)| owner.ref_info.as_ref().map(|ri| ri.ref_name.clone()));
        let mut commits = Vec::new();
        for s in graph.node_weights() {
            for (i, c) in s.commits.iter().enumerate() {
                let mut c = c.clone();
                // The segment graph hoists the tip ref onto `segment.ref_info`; in a commit graph a
                // ref belongs on the commit it points at — the segment's first (tip) commit.
                if i == 0
                    && let Some(ri) = &s.ref_info
                    && !c.refs.iter().any(|r| r.ref_name == ri.ref_name)
                {
                    c.refs.insert(0, ri.clone());
                }
                commits.push(c);
            }
        }
        // The traversal's ACTUAL connectivity: consecutive commits within a segment, plus each
        // connection's `src → dst` commits (resolved through empty segments). Raw `parent_ids`
        // reach past traversal cuts (limits, integrated stop-early) — those stay severed.
        let mut connected: HashSet<(gix::ObjectId, gix::ObjectId)> = HashSet::new();
        for s in graph.node_weights() {
            for w in s.commits.windows(2) {
                connected.insert((w[0].id, w[1].id));
            }
            let Some(last) = s.commits.last().map(|c| c.id) else {
                continue;
            };
            for conn in &s.connections {
                let src = conn.src_id.unwrap_or(last);
                // Resolve the target commit through empty segments (e.g. an empty named segment
                // spliced between commit-carrying ones).
                let mut target = conn.target;
                let mut dst = conn.dst_id;
                for _ in 0..graph.num_segments() {
                    if dst.is_some() {
                        break;
                    }
                    let t = &graph[target];
                    match t.commits.first() {
                        Some(c) => dst = Some(c.id),
                        None => {
                            let Some(next) = t.connections.first() else {
                                break;
                            };
                            dst = next.dst_id;
                            target = next.target;
                        }
                    }
                }
                if let Some(dst) = dst {
                    connected.insert((src, dst));
                }
            }
        }
        let mut cg = CommitGraph::from_commits(commits, entrypoint);
        cg.entrypoint_ref = entrypoint_ref;
        cg.set_connected(connected);
        cg.hard_limit_hit = graph.hard_limit_hit();
        cg.traversal_tips = graph.traversal_tips.clone();
        cg
    }

    /// Restrict connectivity to the given `(child, parent)` pairs and rebuild the child adjacency
    /// accordingly. See the `connected` field.
    fn set_connected(&mut self, connected: HashSet<(gix::ObjectId, gix::ObjectId)>) {
        for children in &mut self.children {
            children.clear();
        }
        for idx in 0..self.nodes.len() {
            let id = self.nodes[idx].commit.id;
            for pos in 0..self.nodes[idx].commit.parent_ids.len() {
                let parent = self.nodes[idx].commit.parent_ids[pos];
                if connected.contains(&(id, parent))
                    && let Some(&pidx) = self.by_id.get(&parent)
                {
                    self.children[pidx].push(idx);
                }
            }
        }
        self.connected = Some(connected);
        self.recompute_generations();
    }

    /// Is the `child → parent` link one the traversal actually followed?
    fn is_connected(&self, child: gix::ObjectId, parent: gix::ObjectId) -> bool {
        self.connected
            .as_ref()
            .is_none_or(|c| c.contains(&(child, parent)))
    }

    /// Compare against `other` field-by-field, returning one human-readable line per
    /// difference. The S1 native-walker oracle: the traversal's direct accumulation must equal
    /// the segment-graph flattening exactly, including per-commit ref ORDER (it surfaces in
    /// snapshots) and flags (goal bits included).
    pub fn diff_against(&self, other: &CommitGraph) -> Vec<String> {
        let mut out = Vec::new();
        let ids: std::collections::BTreeSet<_> = self
            .nodes
            .iter()
            .map(|n| n.commit.id)
            .chain(other.nodes.iter().map(|n| n.commit.id))
            .collect();
        for id in ids {
            match (self.node(id), other.node(id)) {
                (Some(_), None) => out.push(format!("{id}: only in SELF")),
                (None, Some(_)) => out.push(format!("{id}: only in OTHER")),
                (Some(a), Some(b)) => {
                    let (a, b) = (&a.commit, &b.commit);
                    if a.parent_ids != b.parent_ids {
                        out.push(format!(
                            "{id}: parents {:?} != {:?}",
                            a.parent_ids, b.parent_ids
                        ));
                    }
                    if a.flags != b.flags {
                        out.push(format!(
                            "{id}: flags {} != {}",
                            a.flags.debug_string(None),
                            b.flags.debug_string(None)
                        ));
                    }
                    let (mut ra, mut rb): (Vec<_>, Vec<_>) = (
                        a.refs.iter().map(|r| r.ref_name.to_string()).collect(),
                        b.refs.iter().map(|r| r.ref_name.to_string()).collect(),
                    );
                    // Ref ORDER is canonicalized by the native walker; compare as sets.
                    ra.sort();
                    rb.sort();
                    if ra != rb {
                        out.push(format!("{id}: refs {ra:?} != {rb:?}"));
                    }
                }
                (None, None) => unreachable!(),
            }
        }
        if self.entrypoint != other.entrypoint {
            out.push(format!(
                "entrypoint {:?} != {:?}",
                self.entrypoint, other.entrypoint
            ));
        }
        if self.entrypoint_ref != other.entrypoint_ref {
            out.push(format!(
                "entrypoint_ref {:?} != {:?}",
                self.entrypoint_ref.as_ref().map(|r| r.as_bstr()),
                other.entrypoint_ref.as_ref().map(|r| r.as_bstr())
            ));
        }
        if self.connected != other.connected {
            let (a, b) = (
                self.connected.clone().unwrap_or_default(),
                other.connected.clone().unwrap_or_default(),
            );
            for pair in a.difference(&b) {
                out.push(format!("connected only in SELF: {pair:?}"));
            }
            for pair in b.difference(&a) {
                out.push(format!("connected only in OTHER: {pair:?}"));
            }
        }
        if self.hard_limit_hit != other.hard_limit_hit {
            out.push(format!(
                "hard_limit_hit {} != {}",
                self.hard_limit_hit, other.hard_limit_hit
            ));
        }
        if format!("{:?}", self.traversal_tips) != format!("{:?}", other.traversal_tips) {
            out.push(format!(
                "traversal_tips {:?} != {:?}",
                self.traversal_tips, other.traversal_tips
            ));
        }
        if self.explicit_tips != other.explicit_tips {
            out.push(format!(
                "explicit_tips {} != {}",
                self.explicit_tips, other.explicit_tips
            ));
        }
        if self.managed_ws_commits != other.managed_ws_commits {
            out.push(format!(
                "managed_ws_commits {:?} != {:?}",
                self.managed_ws_commits, other.managed_ws_commits
            ));
        }
        out
    }

    /// Assemble from the NATIVE traversal outcome (see `init::native_walk`).
    pub(crate) fn from_native_outcome(o: crate::init::native_walk::NativeOutcome) -> Self {
        let mut cg = CommitGraph::from_commits(o.commits, o.entrypoint);
        cg.entrypoint_ref = o.entrypoint_ref;
        cg.set_connected(o.connected);
        cg.hard_limit_hit = o.hard_limit_hit;
        cg.traversal_tips = o.tips;
        cg
    }

    /// Build by running the WALK's real traversal (queue, goals, limits, flag propagation) with
    /// post-processing skipped, flattening the raw traversal segments into commits. This keeps the
    /// battle-tested traversal semantics — extents (limit cuts, integrated stop-early) and flags are
    /// exactly the walk's — while segments remain a derived view built on top.
    pub fn from_walk<T: but_core::RefMetadata>(
        repo: &gix::Repository,
        meta: &T,
        tip: gix::ObjectId,
        ref_name: Option<gix::refs::FullName>,
        project_meta: but_core::ref_metadata::ProjectMeta,
        options: crate::init::Options,
        overlay: crate::init::Overlay,
    ) -> anyhow::Result<Self> {
        let native =
            Self::from_native_outcome(crate::Graph::native_from_commit_traversal_with_overlay(
                repo,
                tip,
                ref_name.clone(),
                meta,
                project_meta.clone(),
                crate::init::Options {
                    raw_traversal: true,
                    ..options.clone()
                },
                overlay.clone(),
            )?);
        // Transitional oracle: BUT_GRAPH_NATIVE=assert also runs the legacy raw walk and panics
        // with precise diffs if its flattening disagrees with the native walker.
        if std::env::var("BUT_GRAPH_NATIVE").ok().as_deref() == Some("assert") {
            let raw = crate::Graph::from_commit_traversal_with_overlay(
                repo,
                tip,
                ref_name,
                meta,
                project_meta,
                crate::init::Options {
                    raw_traversal: true,
                    ..options
                },
                overlay,
            )?;
            let diffs = native.diff_against(&Self::from_segment_graph(&raw));
            if !diffs.is_empty() {
                panic!(
                    "NATIVE_WALK_DIVERGENCE ({} lines):\n{}",
                    diffs.len(),
                    diffs.join("\n")
                );
            }
        }
        Ok(native)
    }

    /// Like [`Self::from_walk`], but seeded from explicit `tips` — the REAL
    /// [`Graph::from_commit_traversal_tips`](crate::Graph::from_commit_traversal_tips) traversal
    /// with `raw_traversal`, flattened.
    pub fn from_walk_tips<T: but_core::RefMetadata>(
        repo: &gix::Repository,
        meta: &T,
        tips: Vec<crate::init::Tip>,
        project_meta: but_core::ref_metadata::ProjectMeta,
        options: crate::init::Options,
        overlay: crate::init::Overlay,
    ) -> anyhow::Result<Self> {
        let mut native = Self::from_native_outcome(
            crate::Graph::native_from_commit_traversal_tips_with_overlay(
                repo,
                tips.clone(),
                meta,
                project_meta.clone(),
                crate::init::Options {
                    raw_traversal: true,
                    ..options.clone()
                },
                overlay.clone(),
            )?,
        );
        native.explicit_tips = true;
        // Transitional oracle, like `from_walk`.
        if std::env::var("BUT_GRAPH_NATIVE").ok().as_deref() == Some("assert") {
            let raw = crate::Graph::from_commit_traversal_tips_with_overlay(
                repo,
                tips,
                meta,
                project_meta,
                crate::init::Options {
                    raw_traversal: true,
                    ..options
                },
                overlay,
            )?;
            let mut cg = Self::from_segment_graph(&raw);
            cg.explicit_tips = true;
            let diffs = native.diff_against(&cg);
            if !diffs.is_empty() {
                panic!(
                    "NATIVE_WALK_DIVERGENCE tips ({} lines):\n{}",
                    diffs.len(),
                    diffs.join("\n")
                );
            }
        }
        Ok(native)
    }

    /// Mark `id` as a GitButler-managed workspace commit when its message says so.
    pub fn mark_managed_ws_commit_by_message(&mut self, repo: &gix::Repository, id: gix::ObjectId) {
        if let Ok(commit) = repo.find_commit(id)
            && let Ok(message) = commit.message_raw()
            && crate::workspace::commit::is_managed_workspace_by_message(message)
        {
            self.managed_ws_commits.insert(id);
        }
    }

    /// Where traversal/HEAD started (a checkout inside a stack), if any. The projection forces a
    /// segment boundary here — there is always a segment starting at the entrypoint.
    pub fn entrypoint(&self) -> Option<gix::ObjectId> {
        self.entrypoint
    }

    /// The ref the entrypoint was checked out as, if any — it names the entrypoint segment.
    pub fn entrypoint_ref(&self) -> Option<&gix::refs::FullName> {
        self.entrypoint_ref.as_ref()
    }

    /// Whether `id` is a GitButler-managed workspace commit (recognised by its message).
    pub fn is_managed_ws_commit(&self, id: gix::ObjectId) -> bool {
        self.managed_ws_commits.contains(&id)
    }

    /// The node at `id`, if present.
    pub fn node(&self, id: gix::ObjectId) -> Option<&CommitNode> {
        self.by_id.get(&id).map(|&idx| &self.nodes[idx])
    }

    /// Every commit id in the graph, in node order.
    pub fn commit_ids(&self) -> impl Iterator<Item = gix::ObjectId> + '_ {
        self.nodes.iter().map(|n| n.commit.id)
    }

    /// The commit's full parent list, first-parent first, INCLUDING parents not present in this
    /// graph (a partial traversal) — callers preserve those rather than re-pointing them.
    pub fn all_parent_ids(&self, id: gix::ObjectId) -> Vec<gix::ObjectId> {
        self.node(id)
            .map(|n| {
                n.commit
                    .parent_ids
                    .iter()
                    .copied()
                    .filter(|p| self.is_connected(id, *p))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The commit that `ref_name` points at, if present in the graph.
    pub fn commit_by_ref(&self, ref_name: &gix::refs::FullNameRef) -> Option<gix::ObjectId> {
        self.nodes
            .iter()
            .find(|n| {
                n.commit
                    .refs
                    .iter()
                    .any(|r| r.ref_name.as_ref() == ref_name)
            })
            .map(|n| n.commit.id)
    }

    /// The reference names pointing at `id`.
    pub fn refs_at(&self, id: gix::ObjectId) -> Vec<gix::refs::FullName> {
        self.node(id)
            .map(|n| n.commit.refs.iter().map(|r| r.ref_name.clone()).collect())
            .unwrap_or_default()
    }

    /// The parents of `id` that are present in this graph, first-parent first.
    pub fn parents(&self, id: gix::ObjectId) -> impl Iterator<Item = gix::ObjectId> + '_ {
        self.node(id)
            .into_iter()
            .flat_map(|n| n.commit.parent_ids.iter().copied())
            .filter(|p| self.by_id.contains_key(p))
    }

    /// The first parent of `id` (the next commit walking down first-parent), if present.
    pub fn first_parent(&self, id: gix::ObjectId) -> Option<gix::ObjectId> {
        let n = self.node(id)?;
        n.commit
            .parent_ids
            .first()
            .copied()
            .filter(|p| self.by_id.contains_key(p) && self.is_connected(id, *p))
    }

    /// The children of `id` (commits that list `id` as a parent). More than one means a branch point.
    pub fn children(&self, id: gix::ObjectId) -> impl Iterator<Item = gix::ObjectId> + '_ {
        self.by_id
            .get(&id)
            .into_iter()
            .flat_map(move |&idx| self.children[idx].iter().map(|&c| self.nodes[c].commit.id))
    }

    /// Whether walking first-parent should *stop* before entering `id` — i.e. `id` begins a new
    /// segment. Structural boundaries only (the projection layers its own: entrypoint, merge-base,
    /// target). A new segment begins where:
    /// * a local branch ref points at the commit (a named segment starts), or
    /// * the commit is a merge (more than one parent), or
    /// * the commit is a branch point (more than one child) — the paths are distinct segments.
    pub fn is_segment_boundary(&self, id: gix::ObjectId) -> bool {
        let Some(n) = self.node(id) else {
            return false;
        };
        let has_local_branch_ref = n
            .commit
            .ref_name_iter()
            .any(|rn| rn.category() == Some(gix::reference::Category::LocalBranch));
        let is_merge = n.commit.parent_ids.len() > 1;
        let is_branch_point = self.children(id).take(2).count() > 1;
        has_local_branch_ref || is_merge || is_branch_point
    }

    /// Derive one segment's commits: the maximal first-parent run starting at `start` and continuing
    /// while the next first-parent commit is not itself a boundary. This is the grouping the segment
    /// graph used to store, recomputed on demand — the proof that segments are a *view*.
    pub fn first_parent_run(&self, start: gix::ObjectId) -> Vec<gix::ObjectId> {
        let mut run = Vec::new();
        let mut cur = Some(start);
        while let Some(id) = cur {
            run.push(id);
            match self.first_parent(id) {
                Some(next) if !self.is_segment_boundary(next) => cur = Some(next),
                _ => break,
            }
        }
        run
    }

    /// Recompute `generation` for every node (longest path from a root, by Kahn order). Cheap; the
    /// graph is small.
    fn recompute_generations(&mut self) {
        // Process in topological order (parents before children) so a child's generation is the max
        // over its present parents + 1.
        let order = self.toposort_parents_first();
        for id in order {
            let idx = self.by_id[&id];
            let generation = self.nodes[idx]
                .commit
                .parent_ids
                .iter()
                .filter_map(|p| self.by_id.get(p))
                .map(|&pidx| self.nodes[pidx].generation + 1)
                .max()
                .unwrap_or(0);
            self.nodes[idx].generation = generation;
        }
    }

    /// Topological order with parents before children (history order).
    fn toposort_parents_first(&self) -> Vec<gix::ObjectId> {
        let mut indegree = vec![0usize; self.nodes.len()];
        for (idx, n) in self.nodes.iter().enumerate() {
            indegree[idx] = n
                .commit
                .parent_ids
                .iter()
                .filter(|p| self.by_id.contains_key(*p))
                .count();
        }
        let mut queue: std::collections::VecDeque<CommitIdx> = (0..self.nodes.len())
            .filter(|&i| indegree[i] == 0)
            .collect();
        let mut out = Vec::with_capacity(self.nodes.len());
        while let Some(idx) = queue.pop_front() {
            out.push(self.nodes[idx].commit.id);
            for &child in &self.children[idx] {
                indegree[child] -= 1;
                if indegree[child] == 0 {
                    queue.push_back(child);
                }
            }
        }
        out
    }

    /// Commits carrying the in-workspace flag — a stand-in for the kind of flag-based query both
    /// consumers do instead of asking "which segment owns this".
    pub fn in_workspace(&self) -> impl Iterator<Item = gix::ObjectId> + '_ {
        self.nodes
            .iter()
            .filter(|n| n.commit.flags.contains(CommitFlags::InWorkspace))
            .map(|n| n.commit.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CommitFlags;

    fn id(b: u8) -> gix::ObjectId {
        let mut bytes = [0u8; 20];
        bytes[0] = b;
        gix::ObjectId::from_bytes_or_panic(&bytes)
    }

    fn commit(b: u8, parents: &[u8]) -> Commit {
        Commit {
            id: id(b),
            parent_ids: parents.iter().map(|&p| id(p)).collect(),
            flags: CommitFlags::empty(),
            refs: Vec::new(),
        }
    }

    #[test]
    fn children_generation_and_first_parent_walk() {
        // Linear: 3 -> 2 -> 1 (child -> parent).
        let g = CommitGraph::from_commits(
            [commit(3, &[2]), commit(2, &[1]), commit(1, &[])],
            Some(id(3)),
        );
        assert_eq!(g.first_parent(id(3)), Some(id(2)));
        assert_eq!(g.first_parent(id(1)), None);
        assert_eq!(g.children(id(1)).collect::<Vec<_>>(), vec![id(2)]);
        // Generation increases with history depth.
        assert_eq!(g.node(id(1)).unwrap().generation, 0);
        assert_eq!(g.node(id(3)).unwrap().generation, 2);
        // No boundaries on a plain linear chain → the whole thing is one run.
        assert_eq!(g.first_parent_run(id(3)), vec![id(3), id(2), id(1)]);
    }

    #[test]
    fn bridge_from_segment_graph_captures_commits_and_parents() {
        // Build a tiny real segment graph: segment A (a2 -> a1) on base segment B (b0).
        let mut graph = crate::Graph::default();
        let a = graph.insert_segment_set_entrypoint(crate::Segment {
            commits: vec![commit(0xA2, &[0xA1]), commit(0xA1, &[0xB0])],
            ..Default::default()
        });
        graph.connect_new_segment(
            a,
            1, // from a1 (A's second commit)
            crate::Segment {
                commits: vec![commit(0xB0, &[])],
                ..Default::default()
            },
            0,
            id(0xB0),
        );

        let cg = CommitGraph::from_segment_graph(&graph);
        // All three commits made it across, with their parent edges intact.
        assert!(
            cg.node(id(0xA2)).is_some()
                && cg.node(id(0xA1)).is_some()
                && cg.node(id(0xB0)).is_some()
        );
        assert_eq!(cg.first_parent(id(0xA2)), Some(id(0xA1)));
        assert_eq!(cg.first_parent(id(0xA1)), Some(id(0xB0)));
        assert_eq!(cg.first_parent(id(0xB0)), None);
        // Reverse adjacency derived correctly.
        assert_eq!(cg.children(id(0xB0)).collect::<Vec<_>>(), vec![id(0xA1)]);
        // Entrypoint commit carried over (A is the entrypoint segment; its tip is a2).
        assert_eq!(cg.entrypoint, Some(id(0xA2)));
    }

    #[test]
    fn merge_is_a_segment_boundary_so_the_run_stops() {
        // 4 is a merge of 2 and 3; both descend from 1.
        //   4 -> [2, 3] ; 2 -> 1 ; 3 -> 1
        let g = CommitGraph::from_commits(
            [
                commit(4, &[2, 3]),
                commit(2, &[1]),
                commit(3, &[1]),
                commit(1, &[]),
            ],
            Some(id(4)),
        );
        assert!(
            g.is_segment_boundary(id(4)),
            "merge commit starts its own segment"
        );
        assert!(
            g.is_segment_boundary(id(1)),
            "1 has two children (2 and 3) → branch point, a boundary"
        );
        // First-parent run from the merge: 4, then first-parent 2, then stop before boundary 1.
        assert_eq!(g.first_parent_run(id(4)), vec![id(4), id(2)]);
    }
}
