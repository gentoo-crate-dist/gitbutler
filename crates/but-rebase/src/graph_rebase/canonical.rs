//! A node-id-independent canonical form of a [`StepGraph`] — the equality oracle for swapping
//! the editor-graph construction from segment-graph iteration to the CommitGraph.
//!
//! Reference nodes are contracted: every maximal group of `Step::Reference` nodes connected by
//! edges collapses into one `Refs{set}` node. Within-chain ref ORDER is not semantic — an edge
//! reaching the top of a chain reaches every ref in it (what upstream-integration reachability
//! reads), and materialization writes all of a chain's refs to the same commit — but chain
//! MEMBERSHIP and where chains sit between picks are load-bearing, and stay exact.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::graph_rebase::{Step, StepGraph, StepGraphIndex};

/// A canonical node label.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Label {
    /// A `Step::Pick`, by commit id.
    Pick(gix::ObjectId),
    /// A contracted group of `Step::Reference` nodes, by their ref-name set.
    Refs(BTreeSet<String>),
    /// A `Step::None` placeholder (disambiguated by an arbitrary but stable ordinal).
    None(usize),
}

/// What matters about a Pick beyond its id: every behavior field, rendered.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PickInfo {
    pub flavor: String,
    pub preserved_parents: Option<Vec<gix::ObjectId>>,
}

/// The canonical form: labeled nodes + labeled ordered edges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CanonicalStepGraph {
    pub picks: BTreeMap<gix::ObjectId, PickInfo>,
    pub chains: BTreeSet<BTreeSet<String>>,
    pub edges: BTreeSet<(Label, Label, usize)>,
}

impl CanonicalStepGraph {
    pub fn diff_against(&self, other: &Self) -> Vec<String> {
        let mut out = Vec::new();
        for (id, info) in &self.picks {
            match other.picks.get(id) {
                None => out.push(format!("pick {id} only in SELF")),
                Some(o) if o != info => {
                    out.push(format!("pick {id}: {info:?} != {o:?}"));
                }
                _ => {}
            }
        }
        for id in other.picks.keys() {
            if !self.picks.contains_key(id) {
                out.push(format!("pick {id} only in OTHER"));
            }
        }
        for c in self.chains.difference(&other.chains) {
            out.push(format!("chain only in SELF: {c:?}"));
        }
        for c in other.chains.difference(&self.chains) {
            out.push(format!("chain only in OTHER: {c:?}"));
        }
        for e in self.edges.difference(&other.edges) {
            out.push(format!("edge only in SELF: {e:?}"));
        }
        for e in other.edges.difference(&self.edges) {
            out.push(format!("edge only in OTHER: {e:?}"));
        }
        out
    }
}

/// Contract `graph` into its canonical form.
pub(crate) fn canonical_form(graph: &StepGraph) -> CanonicalStepGraph {
    // Union Reference nodes across Reference→Reference edges.
    let mut group_of: HashMap<StepGraphIndex, usize> = HashMap::new();
    let mut groups: Vec<BTreeSet<String>> = Vec::new();
    let mut none_ordinal = 0usize;
    let mut none_of: HashMap<StepGraphIndex, usize> = HashMap::new();

    fn find(parents: &mut [usize], mut g: usize) -> usize {
        while parents[g] != g {
            parents[g] = parents[parents[g]];
            g = parents[g];
        }
        g
    }
    let mut parents: Vec<usize> = Vec::new();

    for ix in graph.node_indices() {
        match &graph[ix] {
            Step::Reference { .. } => {
                let g = groups.len();
                groups.push(BTreeSet::new());
                parents.push(g);
                group_of.insert(ix, g);
            }
            Step::None => {
                none_of.insert(ix, none_ordinal);
                none_ordinal += 1;
            }
            Step::Pick(_) => {}
        }
    }
    for ix in graph.node_indices() {
        if let Step::Reference { refname } = &graph[ix] {
            let g = find(&mut parents, group_of[&ix]);
            groups[g].insert(refname.to_string());
        }
    }
    for eix in graph.edge_indices() {
        let Some((src, dst)) = graph.edge_endpoints(eix) else {
            continue;
        };
        if let (Step::Reference { .. }, Step::Reference { .. }) = (&graph[src], &graph[dst]) {
            let (a, b) = (
                find(&mut parents, group_of[&src]),
                find(&mut parents, group_of[&dst]),
            );
            if a != b {
                // Merge b into a (names travel with the root).
                let names = std::mem::take(&mut groups[b]);
                groups[a].extend(names);
                parents[b] = a;
            }
        }
    }

    let label_of =
        |ix: StepGraphIndex, parents: &mut Vec<usize>, groups: &Vec<BTreeSet<String>>| -> Label {
            match &graph[ix] {
                Step::Pick(p) => Label::Pick(p.id),
                Step::Reference { .. } => {
                    let g = find(parents, group_of[&ix]);
                    Label::Refs(groups[g].clone())
                }
                Step::None => Label::None(none_of[&ix]),
            }
        };

    let mut picks = BTreeMap::new();
    for ix in graph.node_indices() {
        if let Step::Pick(p) = &graph[ix] {
            picks.insert(
                p.id,
                PickInfo {
                    flavor: format!(
                        "{:?}/{:?}/{}/{}/{:?}",
                        p.pick_mode,
                        p.sign_commit,
                        p.exclude_from_tracking,
                        p.conflictable,
                        p.tree_merge_mode
                    ),
                    preserved_parents: p.preserved_parents.clone(),
                },
            );
        }
    }

    let mut edges = BTreeSet::new();
    for eix in graph.edge_indices() {
        let Some((src, dst)) = graph.edge_endpoints(eix) else {
            continue;
        };
        if let (Step::Reference { .. }, Step::Reference { .. }) = (&graph[src], &graph[dst]) {
            let (a, b) = (
                find(&mut parents, group_of[&src]),
                find(&mut parents, group_of[&dst]),
            );
            if a == b {
                continue; // intra-chain
            }
        }
        let a = label_of(src, &mut parents, &groups);
        let b = label_of(dst, &mut parents, &groups);
        let order = graph.edge_weight(eix).map(|w| w.order).unwrap_or_default();
        edges.insert((a, b, order));
    }

    let mut chains = BTreeSet::new();
    for (g, names) in groups.iter().enumerate() {
        if find(&mut parents, g) == g && !names.is_empty() {
            chains.insert(names.clone());
        }
    }

    CanonicalStepGraph {
        picks,
        chains,
        edges,
    }
}
