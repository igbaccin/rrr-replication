import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "rescore_paper_outputs.py"
SPEC = importlib.util.spec_from_file_location("rescore_paper_outputs", SCRIPT_PATH)
RESCORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESCORE)


CHECK_RESULT = {
    "e1": 0,
    "e2": 0,
    "e3": 0,
    "e4": 0,
    "e5": 0,
    "e1_details": [],
    "e2_details": [],
    "e3_details": [],
    "e4_details": [],
    "e5_details": [],
    "e1_loose_advisory_count": 0,
    "quotes_checked": 1,
    "quotes_verified": 1,
    "n_citations": 1,
    "n_docs_cited": 1,
    "docs_cited": ["Doc_2000"],
    "word_count": 10,
    "refusal": False,
}


class FakeChecker:
    @staticmethod
    def check_file(*_args, **_kwargs):
        return copy.deepcopy(CHECK_RESULT)


class RescorePaperOutputsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def make_scored_target(self, nested=True):
        phase_dir = self.root / "phase_a_main"
        result_file = phase_dir / "phase_a_results.json"
        old_result = copy.deepcopy(CHECK_RESULT)
        old_result["e4_details"] = [{"old_association": True}]
        row = {"run": 1, **old_result}
        self.write_json(
            result_file,
            {"summary": {"phase": "phase_a"}, "per_run": [row]},
        )
        reviews_dir = phase_dir / "runs" if nested else phase_dir
        run_dir = reviews_dir / "run_001"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "review_composed.md").write_text(
            "A review (Doc_2000: p.1).\n",
            encoding="utf-8",
        )
        self.write_json(run_dir / "citations.json", old_result)
        return phase_dir, result_file, run_dir

    def test_phase_allowlist_rejects_future_and_gibberish_phases(self):
        self.assertTrue(RESCORE.is_page_phase("phase_a_main"))
        self.assertTrue(RESCORE.is_page_phase("phase_g_qwen3_4b"))
        self.assertTrue(
            RESCORE.is_page_phase("phase_g_qwen3_4b_refusal_near"))
        self.assertFalse(RESCORE.is_page_phase("phase_c2_prompt_constrained"))
        self.assertFalse(RESCORE.is_page_phase("phase_g_refusal_gibberish"))
        self.assertFalse(RESCORE.is_page_phase("phase_g_future_99b"))

    def test_checker_hash_is_independent_of_platform_line_endings(self):
        lf_path = self.root / "lf.py"
        crlf_path = self.root / "crlf.py"
        lf_path.write_bytes(b"one\ntwo\n")
        crlf_path.write_bytes(b"one\r\ntwo\r\n")

        expected = hashlib.sha256(b"one\ntwo\n").hexdigest()
        self.assertEqual(RESCORE.sha256_lf_text(lf_path), expected)
        self.assertEqual(RESCORE.sha256_lf_text(crlf_path), expected)

    def test_dry_run_is_read_only_and_tracks_detail_only_change(self):
        phase_dir, result_file, run_dir = self.make_scored_target(nested=True)
        result_before = result_file.read_bytes()
        citations_before = (run_dir / "citations.json").read_bytes()

        record = RESCORE.rescore_target(
            FakeChecker,
            phase_dir,
            result_file,
            True,
            self.root / "metadata.csv",
            self.root / "data",
            self.root,
            False,
        )

        self.assertEqual(result_file.read_bytes(), result_before)
        self.assertEqual((run_dir / "citations.json").read_bytes(), citations_before)
        self.assertEqual(record["changed_run_count"], 1)
        self.assertEqual(record["changed_runs"][0]["changed_fields"], ["e4_details"])
        self.assertTrue(record["result_file_changed"])
        self.assertEqual(record["citation_files_changed"], 1)
        self.assertFalse(record["result_file_written"])
        self.assertEqual(record["citation_files_written"], 0)

    def test_zero_scored_target_is_not_rewritten(self):
        phase_dir = self.root / "phase_g_gemma3_270m_refusal_far"
        result_file = phase_dir / "phase_g_results.json"
        payload = {
            "summary": {"phase": "phase_g_test", "custom": "preserve"},
            "per_run": [{"run": 1, "refusal": True}],
        }
        self.write_json(result_file, payload)
        before = result_file.read_bytes()

        record = RESCORE.rescore_target(
            FakeChecker,
            phase_dir,
            result_file,
            False,
            self.root / "metadata.csv",
            self.root / "data",
            self.root,
            True,
        )

        self.assertEqual(result_file.read_bytes(), before)
        self.assertFalse(record["result_file_changed"])
        self.assertFalse(record["result_file_written"])
        self.assertEqual(record["changed_run_count"], 0)

    def test_discovery_and_subset_metadata_use_declared_paths(self):
        battery = self.root / "battery"
        allowed = battery / "phase_g_qwen3_4b_refusal_near"
        self.write_json(
            allowed / "phase_g_results.json",
            {"summary": {}, "per_run": []},
        )
        (battery / "phase_g_future_99b").mkdir(parents=True)
        claude_result = self.root / "claude" / "phase_h_results.json"
        self.write_json(claude_result, {"summary": {}, "per_run": []})
        manifest = {
            "source_directories": {
                "primary_battery": "battery",
                "additional_batteries": [],
            },
            "selected_claude_results": {"haiku": "claude/phase_h_results.json"},
        }

        targets = RESCORE.discover_targets(manifest, self.root)
        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[0], (allowed, allowed / "phase_g_results.json", False))
        self.assertEqual(targets[1], (claude_result.parent, claude_result, True))

        subset = battery / "subsample_ws_10" / "metadata.csv"
        subset.parent.mkdir(parents=True)
        subset.write_text("doc_id\nDoc_2000\n", encoding="utf-8")
        selected = RESCORE.subset_metadata(
            battery / "phase_s10_docs", self.root / "metadata.csv")
        self.assertEqual(selected, subset)


if __name__ == "__main__":
    unittest.main()
