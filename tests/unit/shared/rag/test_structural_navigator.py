from __future__ import annotations

from app.shared.rag.structural_navigator import navigate_tree


def _serialized_docling_tree() -> dict[str, object]:
    return {
        "name": "Supply Agreement",
        "body": {
            "self_ref": "#/body",
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/texts/2"},
                {"$ref": "#/texts/3"},
                {"$ref": "#/texts/4"},
            ],
        },
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "section_header",
                "level": 1,
                "text": "Payment Terms",
            },
            {
                "self_ref": "#/texts/1",
                "label": "text",
                "text": "Invoices are payable within thirty days.",
            },
            {
                "self_ref": "#/texts/2",
                "label": "section_header",
                "level": 2,
                "text": "Late Fees",
            },
            {
                "self_ref": "#/texts/3",
                "label": "text",
                "text": "Late payment accrues interest at one percent per month.",
            },
            {
                "self_ref": "#/texts/4",
                "label": "section_header",
                "level": 1,
                "text": "Termination",
            },
        ],
    }


def test_navigator_returns_the_expected_docling_section_path() -> None:
    tree = _serialized_docling_tree()

    paths = navigate_tree(tree, "late payment interest")

    assert paths == [("Supply Agreement", "Payment Terms", "Late Fees")]


def test_navigator_is_deterministic_and_honours_the_result_limit() -> None:
    tree = _serialized_docling_tree()

    first = navigate_tree(tree, "payment", limit=2)
    second = navigate_tree(tree, "payment", limit=2)

    assert first == second
    assert first == [
        ("Supply Agreement", "Payment Terms"),
        ("Supply Agreement", "Payment Terms", "Late Fees"),
    ]


def test_navigator_accepts_nested_structural_nodes_without_docling_runtime_types() -> None:
    tree: dict[str, object] = {
        "name": "Policy Manual",
        "body": {
            "children": [
                {
                    "label": "section",
                    "name": "Security",
                    "children": [{"label": "text", "text": "Rotate signing keys every quarter."}],
                }
            ]
        },
    }

    assert navigate_tree(tree, "signing keys") == [("Policy Manual", "Security")]
    assert navigate_tree(tree, "   ") == []
