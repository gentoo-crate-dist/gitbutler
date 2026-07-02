//! The remains of the legacy walk post-processing: every graph is now assembled from a
//! [`CommitGraph`](crate::CommitGraph) (see `commit_graph_to_segment_graph`), and the raw
//! traversal only ever runs with `dangerously_skip_postprocessing_for_debugging` — as the
//! builders' substrate ([`CommitGraph::from_walk`](crate::CommitGraph::from_walk)) or for raw
//! debugging graphs. What is left is the minimal finishing the raw form needs.

use but_core::RefMetadata;
use gix::ObjectId;
use tracing::instrument;

use crate::{
    EntryPointCommit, Graph, SegmentIndex,
    init::{
        overlay::OverlayMetadata,
        walk::{RefsById, WorktreeByBranch},
    },
};

pub(super) struct Context {
    pub inserted_proxy_segments: Vec<SegmentIndex>,
    pub refs_by_id: RefsById,
    pub hard_limit: bool,
    pub detach_entrypoint: bool,
    pub dangerously_skip_postprocessing_for_debugging: bool,
    pub worktree_by_branch: WorktreeByBranch,
}

/// Processing
impl Graph {
    /// Finish the raw traversal graph. The structural post-processing passes that used to live
    /// here were replaced by the CommitGraph-derived builders; a raw graph only records the hard
    /// limit, re-points the entrypoint commit, and detaches when asked to.
    #[instrument(level = "trace", skip_all, fields(tip), err(Debug))]
    pub(super) fn post_processed<T: RefMetadata>(
        mut self,
        _meta: &OverlayMetadata<'_, T>,
        tip: ObjectId,
        Context {
            hard_limit,
            detach_entrypoint,
            dangerously_skip_postprocessing_for_debugging,
            ..
        }: Context,
    ) -> anyhow::Result<Self> {
        debug_assert!(
            dangerously_skip_postprocessing_for_debugging,
            "the raw traversal only runs as the builders' substrate or for raw debugging graphs"
        );
        self.hard_limit_hit = hard_limit;

        // Keep the original traversal tip available even if the entrypoint moved to a segment
        // that doesn't contain it.
        self.update_entrypoint_commit_id(tip);

        if detach_entrypoint {
            self.detach_entrypoint_segment()?;
        }
        Ok(self)
    }

    /// Ensure the entrypoint commit-id is updated to match the actual tip commit.
    fn update_entrypoint_commit_id(&mut self, tip: ObjectId) {
        if let Some((_segment, ep_commit)) = self.entrypoint.as_mut() {
            *ep_commit = EntryPointCommit::AtCommit(tip);
        }
    }
}
