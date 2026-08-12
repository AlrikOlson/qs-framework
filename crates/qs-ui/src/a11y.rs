//! The accessibility tree.
//!
//! # The one invariant this file exists for
//!
//! > `set_size` is the corpus length and `index_in_set` is the row's corpus ordinal. A
//! > screen reader announcing "item 4 of 60" instead of "item 4,312 of 1,204,883" is the
//! > canonical virtualized-list accessibility bug, and it is a build failure here, not a
//! > polish item.
//!
//! It is easy to get wrong because the natural thing to publish is what you just rendered,
//! and what you just rendered is sixty recycled rows. [`SemanticTree::build`] takes the
//! logical index explicitly and `tests/a11y_audit.rs` asserts the result against a
//! million-row corpus.
//!
//! # Why this is here at M0 rather than at M4
//!
//! Research R12: the virtualized-list-to-semantic-tree mapping is the single hardest
//! accessibility problem in this product, and discovering at M4 that the recycler's
//! architecture cannot express it would be catastrophic. M4 is where accessibility is
//! *audited*; M0 is where it is proven possible.

use accesskit::{Live, Node, NodeId, Rect, Role, Tree, TreeUpdate};

use crate::recycler::ViewportLayout;
use crate::row::Interaction;
use crate::row_source::{LoadState, RowBuf, RowFlags, RowView};

/// Node id of the window. Fixed, because it never changes.
pub const WINDOW_ID: NodeId = NodeId(0);
/// Node id of the list container.
pub const LIST_ID: NodeId = NodeId(1);
/// Node id of the polite live region that announces the selection count.
pub const SELECTION_STATUS_ID: NodeId = NodeId(2);
/// Row node ids start here, offset by the row's **logical** index.
const ROW_ID_BASE: u64 = 16;

pub fn row_node_id(logical_index: u64) -> NodeId {
    NodeId(ROW_ID_BASE + logical_index)
}

/// A published semantic node, in the form the audit test inspects.
///
/// AccessKit's `Node` is write-only from our side, so the tree is also recorded in this
/// plain shape. That is what makes the invariant above assertable without a platform
/// adapter and a running screen reader.
#[derive(Clone, PartialEq, Debug)]
pub struct SemanticNode {
    pub id: NodeId,
    pub role: Role,
    pub label: String,
    /// **Logical** position in the corpus, 1-based as assistive technology expects.
    pub index_in_set: Option<usize>,
    /// **Logical** corpus length -- not the count of recycled rows.
    pub set_size: Option<usize>,
    pub bounds: Option<(f64, f64, f64, f64)>,
    pub selected: bool,
    pub focusable: bool,
    /// Announced when its label changes, without taking focus. Only the selection status
    /// node sets this.
    pub live: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SemanticTree {
    pub nodes: Vec<SemanticNode>,
}

impl SemanticTree {
    /// Build the tree for one frame.
    ///
    /// Selection comes from `interaction`, not from the row flags: selection is view state,
    /// so a `RowSource` never sets `IS_SELECTED` and a tree built from the buffer alone
    /// would report a screen-reader user's selection as permanently empty.
    pub fn build(buf: &RowBuf, layout: &ViewportLayout, interaction: Interaction<'_>) -> Self {
        let mut nodes = Vec::with_capacity(buf.len() + 3);

        nodes.push(SemanticNode {
            id: WINDOW_ID,
            role: Role::Window,
            label: "Quicksilver".to_string(),
            index_in_set: None,
            set_size: None,
            bounds: None,
            selected: false,
            focusable: false,
            live: false,
        });

        nodes.push(SemanticNode {
            id: LIST_ID,
            role: Role::List,
            label: "Files".to_string(),
            index_in_set: None,
            // The corpus length. Not `buf.len()`.
            set_size: Some(layout.row_count as usize),
            bounds: Some((0.0, 0.0, f64::from(layout.width), f64::from(layout.height))),
            selected: false,
            focusable: false,
            live: false,
        });

        // A polite live region rather than a focus move: selecting forty files by dragging
        // must tell the user how many they have without stealing the focus they are dragging
        // from, and without an announcement per row on the way.
        nodes.push(SemanticNode {
            id: SELECTION_STATUS_ID,
            role: Role::Status,
            label: selection_announcement(interaction.selection.len()),
            index_in_set: None,
            set_size: None,
            bounds: None,
            selected: false,
            focusable: false,
            live: true,
        });

        for (slot, row) in buf.rows().iter().enumerate() {
            let logical = layout.visible.first + slot as u64;
            let top = f64::from(layout.row_top(slot as u32));
            nodes.push(SemanticNode {
                id: row_node_id(logical),
                role: Role::ListItem,
                label: describe(buf, row),
                // 1-based: assistive technology announces "item 4,312", not "item 4,311".
                index_in_set: Some(logical as usize + 1),
                set_size: Some(layout.row_count as usize),
                bounds: Some((
                    0.0,
                    top,
                    f64::from(layout.width),
                    top + f64::from(layout.row_height),
                )),
                selected: row.flags.contains(RowFlags::IS_SELECTED)
                    || interaction.selection.contains(logical),
                focusable: true,
                live: false,
            });
        }

        Self { nodes }
    }

    /// Convert to an AccessKit update.
    pub fn to_update(&self, focused: Option<u64>) -> TreeUpdate {
        let mut nodes = Vec::with_capacity(self.nodes.len());

        let row_ids: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|n| n.role == Role::ListItem)
            .map(|n| n.id)
            .collect();

        for node in &self.nodes {
            let mut built = Node::new(node.role);
            if !node.label.is_empty() {
                built.set_label(node.label.clone());
            }
            if let Some(index) = node.index_in_set {
                built.set_position_in_set(index);
            }
            if let Some(size) = node.set_size {
                built.set_size_of_set(size);
            }
            if let Some((x0, y0, x1, y1)) = node.bounds {
                built.set_bounds(Rect { x0, y0, x1, y1 });
            }
            if node.selected {
                built.set_selected(true);
            }
            if node.live {
                built.set_live(Live::Polite);
            }
            match node.id {
                WINDOW_ID => built.set_children(vec![LIST_ID, SELECTION_STATUS_ID]),
                LIST_ID => built.set_children(row_ids.clone()),
                _ => {}
            }
            nodes.push((node.id, built));
        }

        let focus = focused
            .map(row_node_id)
            .filter(|id| self.nodes.iter().any(|n| n.id == *id))
            // Focus must always name a node that exists in the tree; a dangling focus id is
            // rejected by AccessKit and takes the whole update with it.
            .unwrap_or(LIST_ID);

        TreeUpdate {
            nodes,
            tree: Some(Tree::new(WINDOW_ID)),
            // M0 publishes one window with one list; there are no subtrees to graft.
            tree_id: accesskit::TreeId::ROOT,
            focus,
        }
    }
}

/// What the live region says about a selection of `count`.
///
/// Empty at zero rather than "0 selected": a live region whose label goes from "3 selected"
/// to "0 selected" announces the number nobody asked for, where an empty one simply stops
/// talking.
fn selection_announcement(count: u64) -> String {
    match count {
        0 => String::new(),
        1 => "1 selected".to_string(),
        n => format!("{n} selected"),
    }
}

/// The string a screen reader announces for a row.
///
/// Never empty for a focusable node (FR-028). A row whose name is empty -- which happens
/// for a stub that has not yet learned its name -- announces "Loading" rather than nothing,
/// because a focusable node with no name is a node the user cannot identify.
fn describe(buf: &RowBuf, row: &RowView) -> String {
    let name = buf.name(row);
    let name = name.trim();
    if name.is_empty() {
        return "Loading".to_string();
    }

    let mut description = name.to_string();
    if row.flags.contains(RowFlags::IS_DIR) {
        description.push_str(", folder");
    }
    if row.flags.contains(RowFlags::IS_SYMLINK) {
        description.push_str(", link");
    }
    if row.flags.contains(RowFlags::IS_HIDDEN) {
        description.push_str(", hidden");
    }

    // Everything `crate::substance` puts into the surface is said here too, and that is what
    // makes the encoding legal rather than decorative. Constitution IX refuses an effect that
    // is the sole carrier of a fact; a roughness nobody can hear is exactly that. This is the
    // same granularity as the encoding -- one row, its own attributes -- and it costs no I/O,
    // because the row being described is a row the fill window already covers.
    //
    // The original plan was a sort per fact. That cannot be built: `FillWindow` holds
    // attributes for the visible window only, so ordering a million entries by size needs a
    // million stats. See roadmap chunk `row-sort-order`.
    //
    // Nothing is said for a stub. `size` and `mtime` are undefined there, not stale, and a
    // screen reader announcing "0 bytes, 1970" about a row that is still loading is worse
    // than one that says only its name -- it is a confident wrong answer where silence was
    // available.
    if row.state != LoadState::Stub {
        if !row.flags.contains(RowFlags::IS_DIR) {
            description.push_str(", ");
            description.push_str(&crate::format_size(row.size));
        }
        description.push_str(", modified ");
        description.push_str(&crate::format_mtime(row.mtime));
    }
    // Said whatever the load state, because the flag is only ever set from a positive
    // answer -- see `RowFlags::IS_READONLY`. Its absence is no claim, and no claim is
    // correctly announced as nothing at all.
    if row.flags.contains(RowFlags::IS_READONLY) {
        description.push_str(", read-only");
    }
    description
}

#[cfg(test)]
mod describe_tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )]

    use super::*;
    use crate::row_source::{KindId, RowId};

    fn buf_with(row: RowView, name: &[u8]) -> (RowBuf, RowView) {
        let mut buf = RowBuf::new();
        buf.push(row.clone(), name);
        let stored = buf.rows()[0].clone();
        (buf, stored)
    }

    fn file(state: LoadState, flags: RowFlags) -> RowView {
        RowView {
            id: RowId(0),
            name: 0..0,
            size: 4_194_304,
            mtime: 1_786_060_800,
            kind: KindId(0),
            flags,
            state,
            depth: 0,
        }
    }

    #[test]
    fn every_fact_the_surface_encodes_is_also_announced() {
        // Constitution IX: an effect may not be the sole carrier of a fact. `substance` puts
        // age into roughness, size into the bevel and permission into the environment, and
        // this is the route that does not require eyes. If an encoding is ever added there
        // without a clause here, the surface is carrying something alone.
        let (buf, row) = buf_with(file(LoadState::Basic, RowFlags::IS_READONLY), b"notes.txt");
        let said = describe(&buf, &row);

        assert!(said.starts_with("notes.txt"), "{said}");
        assert!(said.contains("4.0 MB"), "size must be announced: {said}");
        assert!(
            said.contains("modified 2026-08-07"),
            "age must be announced, and as a real date rather than the epoch: {said}"
        );
        assert!(
            said.contains("read-only"),
            "permission must be announced: {said}"
        );
    }

    #[test]
    fn a_stub_announces_its_name_and_makes_no_claims() {
        // `size` and `mtime` are undefined on a stub, not stale. "0 bytes, modified
        // 1970-01-01" is a confident wrong answer where silence was available, and a screen
        // reader user has no way to tell it from a real answer.
        let (buf, row) = buf_with(file(LoadState::Stub, RowFlags::EMPTY), b"waiting.bin");
        let said = describe(&buf, &row);

        assert_eq!(said, "waiting.bin");
        assert!(!said.contains("1970"), "{said}");
        assert!(!said.contains("modified"), "{said}");
    }

    #[test]
    fn a_folder_is_not_given_a_size() {
        // A directory's `size` is whatever the filesystem puts in the inode -- 4096 on ext4,
        // zero on NTFS -- and none of it is the number a user means by "how big is this
        // folder". Announcing it would be announcing an implementation detail as a fact.
        // `recursive-size-rollup` is the chunk that would make this answerable.
        let (buf, row) = buf_with(file(LoadState::Basic, RowFlags::IS_DIR), b"src");
        let said = describe(&buf, &row);

        assert!(said.contains("folder"), "{said}");
        assert!(!said.contains("MB"), "{said}");
        assert!(
            said.contains("modified"),
            "a folder still has a date: {said}"
        );
    }
}

/// One audit finding.
#[derive(Clone, PartialEq, Debug)]
pub struct AuditFinding {
    pub node: NodeId,
    pub problem: String,
}

/// Check the tree against the rules SC-008 gates on.
///
/// Returns findings rather than a bool so a failure names the node and the rule, which is
/// the difference between a test that fails and a test that tells you what to fix.
pub fn audit(tree: &SemanticTree, expected_set_size: u64) -> Vec<AuditFinding> {
    let mut findings = Vec::new();

    for node in &tree.nodes {
        if node.focusable && node.label.trim().is_empty() {
            findings.push(AuditFinding {
                node: node.id,
                problem: "focusable node has no name (FR-028)".to_string(),
            });
        }

        if node.role == Role::ListItem {
            match node.set_size {
                Some(size) if size as u64 == expected_set_size => {}
                Some(size) => findings.push(AuditFinding {
                    node: node.id,
                    problem: format!(
                        "set_size is {size}, expected {expected_set_size} -- this is the \
                         recycled-row count leaking into the semantic tree (FR-027)"
                    ),
                }),
                None => findings.push(AuditFinding {
                    node: node.id,
                    problem: "list item has no set_size".to_string(),
                }),
            }

            match node.index_in_set {
                Some(index) if index >= 1 && index as u64 <= expected_set_size => {}
                Some(index) => findings.push(AuditFinding {
                    node: node.id,
                    problem: format!("index_in_set {index} is outside 1..={expected_set_size}"),
                }),
                None => findings.push(AuditFinding {
                    node: node.id,
                    problem: "list item has no index_in_set".to_string(),
                }),
            }

            if node.bounds.is_none() {
                findings.push(AuditFinding {
                    node: node.id,
                    problem: "list item has no bounds".to_string(),
                });
            }
        }
    }

    if !tree.nodes.iter().any(|n| n.role == Role::List) {
        findings.push(AuditFinding {
            node: LIST_ID,
            problem: "no List node was published".to_string(),
        });
    }

    findings
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;
    use crate::density::Density;
    use crate::fenwick::Heights;
    use crate::recycler::Recycler;
    use crate::row_source::{RowSource, StubbornSource};

    fn million_row_frame() -> (RowBuf, ViewportLayout) {
        let source = StubbornSource { count: 1_204_883 };
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
            120_736.0,
        );
        recycler.fill(&source, &layout);
        (recycler.rows().clone(), layout)
    }

    #[test]
    fn the_tree_reports_the_corpus_size_not_the_recycled_row_count() {
        // "item 4,312 of 1,204,883", never "item 4 of 60".
        let (buf, layout) = million_row_frame();
        let tree = SemanticTree::build(&buf, &layout, Interaction::default());

        let items: Vec<_> = tree
            .nodes
            .iter()
            .filter(|n| n.role == Role::ListItem)
            .collect();

        assert!(items.len() < 60, "only the visible rows are published");
        assert!(!items.is_empty());

        for item in &items {
            assert_eq!(
                item.set_size,
                Some(1_204_883),
                "set_size leaked the recycled row count"
            );
        }
        assert!(
            items[0].index_in_set.unwrap() > 4000,
            "index_in_set is a slot index, not a corpus ordinal"
        );
    }

    #[test]
    fn index_in_set_is_one_based_and_contiguous() {
        let (buf, layout) = million_row_frame();
        let tree = SemanticTree::build(&buf, &layout, Interaction::default());
        let indices: Vec<usize> = tree
            .nodes
            .iter()
            .filter(|n| n.role == Role::ListItem)
            .filter_map(|n| n.index_in_set)
            .collect();

        assert_eq!(indices[0], layout.visible.first as usize + 1);
        for pair in indices.windows(2) {
            assert_eq!(pair[1], pair[0] + 1, "the published indices have a gap");
        }
    }

    #[test]
    fn the_audit_passes_on_a_correct_tree() {
        let (buf, layout) = million_row_frame();
        let tree = SemanticTree::build(&buf, &layout, Interaction::default());
        let findings = audit(&tree, layout.row_count);
        assert!(findings.is_empty(), "{findings:#?}");
    }

    #[test]
    fn the_audit_catches_the_canonical_virtualized_list_bug() {
        // The test has to be able to fail, or it is decoration. Publish the recycled-row
        // count as `set_size` -- the exact mistake -- and confirm the audit names it.
        let (buf, layout) = million_row_frame();
        let mut tree = SemanticTree::build(&buf, &layout, Interaction::default());
        let visible = buf.len();
        for node in &mut tree.nodes {
            if node.role == Role::ListItem {
                node.set_size = Some(visible);
            }
        }

        let findings = audit(&tree, layout.row_count);
        assert!(!findings.is_empty());
        assert!(findings[0].problem.contains("recycled-row count"));
    }

    #[test]
    fn a_focusable_node_is_never_nameless() {
        // FR-028. A stub row has no name yet, and must still announce something.
        let source = StubbornSource { count: 10 };
        let heights = Heights::Uniform(28);
        let mut recycler = Recycler::new();
        let layout = recycler.layout(
            &source,
            &heights,
            800,
            400,
            1.0,
            1.0,
            Density::Default,
            28,
            0.0,
        );
        recycler.fill(&source, &layout);

        let tree = SemanticTree::build(recycler.rows(), &layout, Interaction::default());
        for node in &tree.nodes {
            if node.focusable {
                assert!(!node.label.trim().is_empty(), "{node:?}");
            }
        }
        assert!(audit(&tree, source.len()).is_empty());
    }

    #[test]
    fn an_accesskit_update_names_a_focus_node_that_exists() {
        // A dangling focus id is rejected by AccessKit and takes the whole update with it.
        let (buf, layout) = million_row_frame();
        let tree = SemanticTree::build(&buf, &layout, Interaction::default());

        let update = tree.to_update(Some(layout.visible.first));
        assert!(update.nodes.iter().any(|(id, _)| *id == update.focus));

        // A focus request for a row that is not published falls back to the list.
        let update = tree.to_update(Some(999_999_999));
        assert_eq!(update.focus, LIST_ID);

        // And with no focus at all.
        let update = tree.to_update(None);
        assert_eq!(update.focus, LIST_ID);
    }

    #[test]
    fn row_node_ids_do_not_collide_with_the_fixed_ids() {
        assert_ne!(row_node_id(0), WINDOW_ID);
        assert_ne!(row_node_id(0), LIST_ID);
        assert_ne!(row_node_id(0), row_node_id(1));
    }

    #[test]
    fn descriptions_carry_the_attributes_a_reader_needs() {
        let mut buf = RowBuf::new();
        buf.push(
            RowView {
                flags: RowFlags::IS_DIR | RowFlags::IS_HIDDEN,
                ..Default::default()
            },
            b".config",
        );
        assert_eq!(describe(&buf, &buf.rows()[0]), ".config, folder, hidden");
    }
}
