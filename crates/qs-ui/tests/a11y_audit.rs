//! T050 / SC-008 — the automated accessibility audit.
//!
//! From the quickstart:
//!
//! > Expected: zero focusable nodes without a role, name, and state; and the list reports
//! > `set_size == 1_000_000` with correct `index_in_set` -- **not** the ~60 recycled rows.
//! > That specific assertion is the one worth reading the test for; it is the canonical
//! > virtualized-list accessibility bug.

// Integration tests assert by panicking; `unwrap`/`expect`/`panic!` are the
// vocabulary of a test, not a hazard in one. The workspace lints deny them for
// production code, so each test binary opts out at its root.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
use qs_ui::a11y::{LIST_ID, SemanticTree, WINDOW_ID, audit, row_node_id};
use qs_ui::density::Density;
use qs_ui::fenwick::Heights;
use qs_ui::recycler::{Recycler, ViewportLayout};
use qs_ui::row::Interaction;
use qs_ui::row_source::{RowBuf, StubbornSource};

const CORPUS: u64 = 1_000_000;

fn frame_at(scroll: f64, rows: u64) -> (RowBuf, ViewportLayout) {
    let source = StubbornSource { count: rows };
    let heights = Heights::Uniform(28);
    let mut recycler = Recycler::new();
    let layout = recycler.layout(
        &source,
        &heights,
        1920,
        1080,
        1.0,
        1.0,
        Density::Default,
        28,
        scroll,
    );
    recycler.fill(&source, &layout);
    (recycler.rows().clone(), layout)
}

#[test]
fn the_list_reports_the_corpus_size_not_the_recycled_row_count() {
    let (buf, layout) = frame_at(0.0, CORPUS);
    let tree = SemanticTree::build(&buf, &layout, Interaction::default());

    let list = tree
        .nodes
        .iter()
        .find(|n| n.id == LIST_ID)
        .expect("a List node must be published");

    assert_eq!(
        list.set_size,
        Some(CORPUS as usize),
        "the list must announce 1,000,000 items, not the ~40 rows on screen"
    );

    // And the rows on screen really are only ~40, so the assertion above is meaningful.
    let published = tree.nodes.iter().filter(|n| n.focusable).count();
    assert!(
        published < 60,
        "{published} rows published; virtualization is not working"
    );
}

#[test]
fn index_in_set_is_the_corpus_ordinal_at_a_deep_scroll_position() {
    // Scroll to row 4,311 (0-based), which announces as "item 4,312".
    let scroll = 4311.0 * 28.0;
    let (buf, layout) = frame_at(scroll, CORPUS);
    let tree = SemanticTree::build(&buf, &layout, Interaction::default());

    let first = tree
        .nodes
        .iter()
        .find(|n| n.focusable)
        .expect("at least one row must be published");

    assert_eq!(first.index_in_set, Some(4312), "1-based corpus ordinal");
    assert_eq!(first.set_size, Some(CORPUS as usize));
    assert_eq!(first.id, row_node_id(4311));
}

#[test]
fn zero_focusable_nodes_lack_a_role_name_or_bounds() {
    for scroll in [0.0, 1.0, 27.5, 120_736.0, 27_998_919.0] {
        let (buf, layout) = frame_at(scroll, CORPUS);
        let tree = SemanticTree::build(&buf, &layout, Interaction::default());

        for node in &tree.nodes {
            if !node.focusable {
                continue;
            }
            assert!(
                !node.label.trim().is_empty(),
                "focusable node {:?} has no name (FR-028)",
                node.id
            );
            assert!(
                node.bounds.is_some(),
                "focusable node {:?} has no bounds",
                node.id
            );
        }

        let findings = audit(&tree, layout.row_count);
        assert!(findings.is_empty(), "scroll {scroll}: {findings:#?}");
    }
}

#[test]
fn published_indices_are_contiguous_and_match_the_visible_range() {
    let (buf, layout) = frame_at(500_000.0, CORPUS);
    let tree = SemanticTree::build(&buf, &layout, Interaction::default());

    let indices: Vec<usize> = tree
        .nodes
        .iter()
        .filter(|n| n.focusable)
        .filter_map(|n| n.index_in_set)
        .collect();

    assert!(!indices.is_empty());
    assert_eq!(
        indices.first().copied(),
        Some(layout.visible.first as usize + 1)
    );
    for pair in indices.windows(2) {
        assert_eq!(pair[1], pair[0] + 1, "a gap in the published indices");
    }
}

#[test]
fn the_audit_can_actually_fail() {
    // A test that only ever passes is decoration. Inject the canonical bug -- publish the
    // recycled-row count as `set_size` -- and confirm the audit names it.
    let (buf, layout) = frame_at(120_736.0, CORPUS);
    let mut tree = SemanticTree::build(&buf, &layout, Interaction::default());

    let visible = buf.len();
    for node in &mut tree.nodes {
        if node.focusable {
            node.set_size = Some(visible);
        }
    }

    let findings = audit(&tree, layout.row_count);
    assert!(!findings.is_empty(), "the audit failed to catch the bug");
    assert!(
        findings[0].problem.contains("recycled-row count"),
        "the finding must explain what went wrong: {}",
        findings[0].problem
    );
}

#[test]
fn a_nameless_row_is_caught() {
    let (buf, layout) = frame_at(0.0, CORPUS);
    let mut tree = SemanticTree::build(&buf, &layout, Interaction::default());
    if let Some(node) = tree.nodes.iter_mut().find(|n| n.focusable) {
        node.label = "   ".into();
    }
    let findings = audit(&tree, layout.row_count);
    assert!(findings.iter().any(|f| f.problem.contains("no name")));
}

#[test]
fn the_accesskit_update_is_well_formed() {
    let (buf, layout) = frame_at(120_736.0, CORPUS);
    let tree = SemanticTree::build(&buf, &layout, Interaction::default());
    let update = tree.to_update(Some(layout.visible.first));

    // Focus must name a node in the update, or AccessKit rejects the whole thing.
    assert!(update.nodes.iter().any(|(id, _)| *id == update.focus));
    // The root must be present and must be the tree's declared root.
    assert!(update.nodes.iter().any(|(id, _)| *id == WINDOW_ID));
    assert!(update.tree.is_some());
    // Every published row appears exactly once.
    let ids: Vec<_> = update.nodes.iter().map(|(id, _)| *id).collect();
    let mut sorted = ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "duplicate node ids in the update");
}

#[test]
fn an_empty_corpus_still_publishes_a_usable_tree() {
    // A window with nothing in it must still announce a list, not an empty tree that a
    // screen reader reports as a blank window.
    let (buf, layout) = frame_at(0.0, 0);
    let tree = SemanticTree::build(&buf, &layout, Interaction::default());

    assert!(tree.nodes.iter().any(|n| n.id == LIST_ID));
    assert!(audit(&tree, 0).is_empty());
    assert_eq!(tree.to_update(None).focus, LIST_ID);
}
