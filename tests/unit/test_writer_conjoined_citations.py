"""A citation must never be folded into a prose phrase naming two authors.

The hosted writer never fused ``Mokyr and Kuznets (1973) [E0081] [E0089]``,
but when the model omitted the bare year its renderer produced
``Mokyr and Kuznets (1973, p.1)`` directly. These tests cover the renderer
guard and the pre-ship repair.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rrr import writer  # noqa: E402
from rrr.render import _build_display_lookup, parse_citations  # noqa: E402

EVIDENCE = {
    "E0081": {"doc_id": "Kuznets_1973", "page": 1},
    "E0089": {"doc_id": "Mokyr_2005", "page": 49},
    "E0001": {"doc_id": "North&Weingast_1989", "page": 2},
}
LOOKUP = _build_display_lookup({entry["doc_id"] for entry in EVIDENCE.values()})


def rendered(text):
    text, stats = writer._render_evidence_id_citations(text, EVIDENCE)
    text, _ = writer._collapse_double_parens(text)
    text, _ = writer._merge_adjacent_paren_cites(text, display_lookup=LOOKUP)
    return text, stats


def resolved_docs(text):
    return [c.get("doc_id") for c in parse_citations(text, display_lookup=LOOKUP)]


class ConjoinedAuthorCitationTests(unittest.TestCase):
    def test_conjoined_subject_renders_parenthetically(self):
        text, stats = rendered("Mokyr and Kuznets [E0081] [E0089] argue the point.")
        self.assertEqual(
            "Mokyr and Kuznets (Kuznets 1973, p.1; Mokyr 2005, p.49) argue the point.",
            text,
        )
        self.assertEqual(1, stats["conjoined_subject_parentheticals"])
        self.assertEqual(["Kuznets_1973", "Mokyr_2005"], resolved_docs(text))

    def test_legitimate_narrative_forms_keep_the_bare_year(self):
        cases = {
            "North and Weingast [E0001] show that commitment mattered.":
                "North and Weingast (1989, p.2) show that commitment mattered.",
            "Kuznets [E0081] notes the pattern.": "Kuznets (1973, p.1) notes the pattern.",
            "Both Mokyr [E0089] and Kuznets [E0081] argue the point.":
                "Both Mokyr (2005, p.49) and Kuznets (1973, p.1) argue the point.",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                text, _ = rendered(source)
                self.assertEqual(expected, text)
                self.assertNotIn(None, resolved_docs(text))

    def test_fused_label_is_rebound_before_shipping(self):
        text = "Mokyr and Kuznets (1973, p.1) (Mokyr 2005, p.49) argue the point."
        repaired, count = writer._repair_unresolved_display_citations(text, LOOKUP)
        self.assertEqual(1, count)
        self.assertEqual(
            "Mokyr and Kuznets (Kuznets 1973, p.1) (Mokyr 2005, p.49) argue the point.",
            repaired,
        )
        self.assertNotIn(None, resolved_docs(repaired))

    def test_unrepairable_label_stays_unresolved(self):
        text = "Smith and Jones (1973, p.1) argue the point."
        repaired, count = writer._repair_unresolved_display_citations(text, LOOKUP)
        self.assertEqual(0, count)
        self.assertIn(None, resolved_docs(repaired))


if __name__ == "__main__":
    unittest.main()
