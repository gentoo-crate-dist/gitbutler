//! Functions for materializing a rebase
use anyhow::{Context, Result, bail};
use but_core::{
    ObjectStorageExt as _, RefMetadata,
    worktree::{checkout::Options, safe_checkout_from_head},
};
use gix::refs::{
    Target,
    transaction::{Change, LogChange, PreviousValue, RefEdit, RefLog},
};

use crate::graph_rebase::{Checkout, MaterializeOutcome, Pick, Step, SuccessfulRebase};

impl<'ws, 'graph, M: RefMetadata> SuccessfulRebase<'ws, 'graph, M> {
    /// Materializes a history rewrite
    pub fn materialize(mut self) -> Result<MaterializeOutcome<'ws, 'graph, M>> {
        let repo = self.repo.clone();
        if let Some(memory) = self.repo.objects.take_object_memory() {
            memory.persist(self.repo)?;
        }

        let mut head_reference_update = None;
        for checkout in self.checkouts {
            match checkout {
                Checkout::Head {
                    selector,
                    merge_base_override,
                } => {
                    let step = self.graph.step_view(selector.id);

                    let (new_head, new_head_refname) = match step {
                        Step::None => bail!("Checkout selector is pointing to none"),
                        Step::Pick(Pick { id, .. }) => (id, None),
                        Step::Reference { refname, .. } => {
                            let parent_step_id = crate::graph_rebase::positions::resolve_to_pick(
                                &self.graph,
                                selector.id,
                            )
                            .context("No commit to reference")?;
                            let Some(id) = self.graph.commit_id(parent_step_id) else {
                                bail!("resolve_to_pick should always return a commit pick");
                            };
                            (id, Some(refname))
                        }
                    };
                    head_reference_update = new_head_refname;

                    // If the head has changed (which means it's in the
                    // commit mapping), perform a safe checkout.
                    safe_checkout_from_head(
                        new_head,
                        &repo,
                        Options {
                            skip_head_update: true,
                            merge_base_override,
                            allow_conflicted_commit_checkout: true,
                        },
                    )?;
                }
            }
        }

        let mut ref_edits = self.ref_edits.clone();
        if let Some(refname) = head_reference_update
            && repo.head_name()?.as_ref() != Some(&refname)
        {
            let ref_short_name = refname.shorten().to_owned();
            ref_edits.push(RefEdit {
                change: Change::Update {
                    log: LogChange {
                        mode: RefLog::AndReference,
                        force_create_reflog: false,
                        message: gix::reference::log::message(
                            "safe checkout",
                            ref_short_name.as_ref(),
                            0,
                        ),
                    },
                    expected: PreviousValue::Any,
                    new: Target::Symbolic(refname),
                },
                name: "HEAD".try_into().expect("root refs are always valid"),
                deref: false,
            });
        }
        repo.edit_references(ref_edits)?;

        let project_meta = self.workspace.graph.project_meta.clone();
        self.workspace
            .refresh_from_head(&repo, &*self.meta, project_meta)?;
        assert_write_through_parity(&self.graph, self.workspace, &repo, &*self.meta)?;

        Ok(MaterializeOutcome {
            graph: self.graph,
            history: self.history,
            workspace: self.workspace,
            meta: self.meta,
        })
    }

    /// Materializes a rebase without performing a checkout.
    ///
    /// For the vast majority of operations you want to use
    /// [`Self::materialize`]. This is intended to be used in niche cases like
    /// `uncommit`.
    ///
    /// This has means that we don't "cherry pick" the uncommitted changes from
    /// the old head onto the new one.
    ///
    /// If I dropped a commit from the history,
    /// [`Self::materialize_without_checkout`] will now see those changes in
    /// your working directory.
    ///
    /// If I instead called [`Self::materialize`], the changes would instead be
    /// gone from disk.
    pub fn materialize_without_checkout(mut self) -> Result<MaterializeOutcome<'ws, 'graph, M>> {
        let repo = self.repo.clone();
        if let Some(memory) = self.repo.objects.take_object_memory() {
            memory.persist(self.repo)?;
        }

        repo.edit_references(self.ref_edits.clone())?;

        let project_meta = self.workspace.graph.project_meta.clone();
        self.workspace
            .refresh_from_head(&repo, &*self.meta, project_meta)?;
        assert_write_through_parity(&self.graph, self.workspace, &repo, &*self.meta)?;

        Ok(MaterializeOutcome {
            graph: self.graph,
            history: self.history,
            workspace: self.workspace,
            meta: self.meta,
        })
    }
}

/// THE WRITE-THROUGH ORACLE (`BUT_REBASE_WRITE_THROUGH=assert`): projecting the editor's
/// MUTATED arena must equal the rewalk's projection — the dissolve's parity obligation
/// (mutate-then-project == rewalk-then-project). Compared on an index-free fingerprint of
/// the stack shape, since segment indices differ between independently built graphs.
fn assert_write_through_parity<M: RefMetadata>(
    graph: &crate::graph_rebase::StepGraph,
    rewalked: &but_graph::Workspace,
    repo: &gix::Repository,
    meta: &M,
) -> anyhow::Result<()> {
    if std::env::var_os("BUT_REBASE_WRITE_THROUGH").is_none_or(|v| v != "assert") {
        return Ok(());
    }
    let Some(mutated) = but_graph::workspace_from_commit_graph(
        graph.arena().clone(),
        repo,
        meta,
        rewalked.graph.project_meta.clone(),
        rewalked.graph.options.clone(),
    )?
    else {
        // Nothing the seam can project: HEAD is unborn (e.g. its referent was deleted
        // without a repoint) or points outside the editor's graph.
        return Ok(());
    };
    let (mutated_fp, rewalked_fp) = (
        projection_fingerprint(&mutated),
        projection_fingerprint(rewalked),
    );
    if mutated_fp != rewalked_fp {
        bail!(
            "WRITE-THROUGH DIVERGENCE\n--- mutate-then-project\n{mutated_fp}\n--- rewalk-then-project\n{rewalked_fp}"
        );
    }
    Ok(())
}

/// The parity view: stack ids, segment names, per-segment commit ids and bases — everything
/// the rebase is obliged to preserve, nothing graph-index-dependent.
fn projection_fingerprint(ws: &but_graph::Workspace) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for stack in &ws.stacks {
        writeln!(out, "stack {:?}", stack.id).ok();
        for segment in &stack.segments {
            writeln!(
                out,
                "  {} base={:?} commits=[{}]",
                segment
                    .ref_name()
                    .map_or_else(|| "<anon>".to_string(), |n| n.as_bstr().to_string()),
                segment.base,
                segment
                    .commits
                    .iter()
                    .map(|c| c.id.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            )
            .ok();
        }
    }
    out
}
