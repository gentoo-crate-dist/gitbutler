//! The C2 parity corpus: mutate-then-project must equal rewalk-then-project.
//!
//! Every test drives one representative mutation through `rebase()`, then compares the
//! projection of the mutated editor graph against the projection of a fresh editor created
//! from the materialized, re-walked repository (via
//! [`but_rebase::graph_rebase::testing::rewalk_parity_report`]). Divergence here is exactly
//! the gap that stops editor sessions from living directly on the walked graph — the corpus
//! is the collapse's worklist, so new mutation kinds should gain a scenario when they land.

use anyhow::Result;
use but_core::ref_metadata::ProjectMeta;
use but_graph::Graph;
use but_rebase::graph_rebase::{
    Editor, Step, mutate, mutate::InsertSide, testing::rewalk_parity_report,
};

use crate::utils::{fixture_writable, standard_options};

/// Assert both projections agree, with a readable dump on divergence.
fn assert_parity(mutated: &str, rewalked: &str) {
    assert!(
        mutated == rewalked,
        "mutate-then-project != rewalk-then-project\n\n--- mutated editor graph ---\n{mutated}\n\n--- rewalked repository ---\n{rewalked}\n"
    );
}

/// Build a workspace editor for `fixture` with no target.
macro_rules! editor {
    ($fixture:literal, $repo:ident, $tmp:ident, $meta:ident, $ws:ident) => {
        let ($repo, $tmp, mut $meta) = fixture_writable($fixture)?;
        let graph = Graph::from_head(&$repo, &*$meta, ProjectMeta::default(), standard_options())?
            .validated()?;
        let mut $ws = graph.into_workspace()?;
    };
}

/// The identity mutation: a plain rebase must round-trip through materialize + rewalk.
#[test]
fn noop_rebase() -> Result<()> {
    editor!("workspace-signed", repo, _tmp, meta, ws);
    let editor = Editor::create(&mut ws, &mut *meta, &repo)?;
    let rebase = editor.rebase()?;
    let (mutated, rewalked) = rewalk_parity_report(rebase, &repo)?;
    assert_parity(&mutated, &rewalked);
    Ok(())
}

/// A new commit inserted above a mid-stack commit.
#[test]
fn insert_pick_above_commit() -> Result<()> {
    editor!("workspace-signed", repo, _tmp, meta, ws);
    let mut editor = Editor::create(&mut ws, &mut *meta, &repo)?;

    let b = repo.rev_parse_single("b")?.detach();
    let mut new_commit = but_core::Commit::from_id(repo.rev_parse_single("b")?)?;
    new_commit.message = "inserted above b".into();
    new_commit.parents = vec![].into();
    let new_id = repo.write_object(new_commit.inner)?.detach();

    let selector = editor.select_commit(b)?;
    editor.insert(selector, Step::new_pick(new_id), InsertSide::Above)?;

    let rebase = editor.rebase()?;
    let (mutated, rewalked) = rewalk_parity_report(rebase, &repo)?;
    assert_parity(&mutated, &rewalked);
    Ok(())
}

/// A new reference created above a mid-stack commit (branch creation).
#[test]
fn insert_reference_above_commit() -> Result<()> {
    editor!("workspace-signed", repo, _tmp, meta, ws);
    let mut editor = Editor::create(&mut ws, &mut *meta, &repo)?;

    let b = repo.rev_parse_single("b")?.detach();
    let selector = editor.select_commit(b)?;
    editor.insert(
        selector,
        Step::new_reference("refs/heads/created-here".try_into()?),
        InsertSide::Above,
    )?;

    let rebase = editor.rebase()?;
    let (mutated, rewalked) = rewalk_parity_report(rebase, &repo)?;
    assert_parity(&mutated, &rewalked);
    Ok(())
}

/// A new commit inserted below a mid-stack commit (exercises the Below-pick parent rewire).
#[test]
fn insert_pick_below_commit() -> Result<()> {
    editor!("workspace-signed", repo, _tmp, meta, ws);
    let mut editor = Editor::create(&mut ws, &mut *meta, &repo)?;

    let b = repo.rev_parse_single("b")?.detach();
    let mut new_commit = but_core::Commit::from_id(repo.rev_parse_single("base")?)?;
    new_commit.message = "inserted below b".into();
    new_commit.parents = vec![].into();
    let new_id = repo.write_object(new_commit.inner)?.detach();

    let selector = editor.select_commit(b)?;
    editor.insert(selector, Step::new_pick(new_id), InsertSide::Below)?;

    let rebase = editor.rebase()?;
    let (mutated, rewalked) = rewalk_parity_report(rebase, &repo)?;
    assert_parity(&mutated, &rewalked);
    Ok(())
}

/// A mid-stack commit disconnected and tombstoned (commit deletion).
///
/// IGNORED — an open C2 gap. When a commit is deleted, its co-located branch ref (and the ref
/// on the surviving parent) must re-anchor onto that parent with the surviving chain's legs as
/// their via. The disconnect surgery empties those vias as it rewires and nothing restores the
/// new legs, so the re-anchored refs render as rootless (dropped from the projection) while a
/// fresh walk shows them co-located on the parent. Restoring the legs is subtle: a blanket
/// `legs_into_pick` restore collides ranks in the dup-parent / multi-leg merge case (it must
/// preserve the descended-group vs root-group split and rank structure), so it belongs with a
/// dedicated pass rather than an ad-hoc patch in `disconnect_segment_from`.
#[test]
#[ignore = "C2 gap: deletion re-anchor leaves co-located refs with empty vias (dropped from projection); needs a group/rank-preserving leg restore"]
fn disconnect_and_remove_commit() -> Result<()> {
    editor!("workspace-signed", repo, _tmp, meta, ws);
    let mut editor = Editor::create(&mut ws, &mut *meta, &repo)?;

    let b = repo.rev_parse_single("b")?.detach();
    let selector = editor.select_commit(b)?;
    editor.disconnect_segment_from(
        mutate::SegmentDelimiter {
            child: selector,
            parent: selector,
        },
        mutate::SelectorSet::All,
        mutate::SelectorSet::All,
        false,
    )?;
    editor.replace(selector, Step::None)?;

    let rebase = editor.rebase()?;
    let (mutated, rewalked) = rewalk_parity_report(rebase, &repo)?;
    assert_parity(&mutated, &rewalked);
    Ok(())
}

/// An empty stack round-trips unchanged.
#[test]
fn noop_rebase_with_empty_stack() -> Result<()> {
    editor!("workspace-with-empty-stack", repo, _tmp, meta, ws);
    let editor = Editor::create(&mut ws, &mut *meta, &repo)?;
    let rebase = editor.rebase()?;
    let (mutated, rewalked) = rewalk_parity_report(rebase, &repo)?;
    assert_parity(&mutated, &rewalked);
    Ok(())
}
