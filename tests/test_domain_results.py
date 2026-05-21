import csv
import json
import tempfile
import unittest
from pathlib import Path

import summarize_domain_results as summary


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_judge(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "judge_model": "gpt-4.1",
        "details": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class DomainResultsTest(unittest.TestCase):
    def test_builds_domain_and_culture_tables_from_matching_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(
                root / "model_outputs_gujarati" / "nonexistent" / "en" / "gpt-5.jsonl",
                [
                    {
                        "id": 1,
                        "question": "fake food?",
                        "gold_answer": "No Information Available",
                        "model_answer": "made up food",
                        "domain": "Food & Cuisine",
                        "model_tag": "gpt-5",
                        "dataset_type": "nonexistent",
                        "lang": "en",
                    },
                    {
                        "id": 3,
                        "question": "another fake food?",
                        "gold_answer": "No Information Available",
                        "model_answer": "No Information Available",
                        "domain": "Food & Cuisine",
                        "model_tag": "gpt-5",
                        "dataset_type": "nonexistent",
                        "lang": "en",
                    },
                    {
                        "id": 2,
                        "question": "fake place?",
                        "gold_answer": "No Information Available",
                        "model_answer": "No Information Available",
                        "domain": "Geography",
                        "model_tag": "gpt-5",
                        "dataset_type": "nonexistent",
                        "lang": "en",
                    },
                ],
            )
            write_judge(
                root / "judge_results" / "guj" / "en" / "results_gpt-5_gpt-4.1.json",
                [
                    {
                        "id": 1,
                        "question": "fake food?",
                        "gold_answer": "No Information Available",
                        "model_answer": "made up food",
                        "is_abstaining": False,
                        "is_hallucinated": True,
                    },
                    {
                        "id": 3,
                        "question": "another fake food?",
                        "gold_answer": "No Information Available",
                        "model_answer": "No Information Available",
                        "is_abstaining": False,
                        "is_hallucinated": False,
                    },
                    {
                        "id": 2,
                        "question": "fake place?",
                        "gold_answer": "No Information Available",
                        "model_answer": "No Information Available",
                        "is_abstaining": True,
                        "is_hallucinated": True,
                    },
                ],
            )

            result = summary.build_tables(root)

            domain_rows = result.domain_rows
            self.assertEqual(len(domain_rows), 2)
            food = next(row for row in domain_rows if row["domain"] == "Food & Cuisine")
            self.assertEqual(food["culture"], "gujarati")
            self.assertEqual(food["dataset_type"], "nonexistent")
            self.assertEqual(food["language"], "en")
            self.assertEqual(food["model"], "gpt-5")
            self.assertEqual(food["total"], 2)
            self.assertEqual(food["refusal_rate"], 0.0)
            self.assertEqual(food["hallucination_rate_not_refused"], 0.5)
            self.assertEqual(food["hallucination_rate_all"], 0.5)

            combined = result.culture_rows
            self.assertEqual(len(combined), 2)
            food_combined = next(
                row for row in combined if row["canonical_domain"] == "Food & Cuisine"
            )
            self.assertEqual(food_combined["culture"], "gujarati")
            self.assertEqual(food_combined["dataset_type"], "nonexistent")
            self.assertEqual(food_combined["total"], 2)
            self.assertEqual(food_combined["refusal_rate"], 0.0)
            self.assertEqual(food_combined["hallucination_rate_not_refused"], 0.5)
            self.assertEqual(food_combined["hallucination_rate_all"], 0.5)

    def test_culture_table_combines_translated_parallel_domains(self):
        rows = [
            {
                "culture": "gujarati",
                "dataset_type": "precise",
                "domain": "Food & Cuisine",
                "is_abstaining": False,
                "is_hallucinated": False,
            },
            {
                "culture": "gujarati",
                "dataset_type": "precise",
                "domain": "ખોરાક અને ભોજન",
                "is_abstaining": False,
                "is_hallucinated": True,
            },
            {
                "culture": "gujarati",
                "dataset_type": "precise",
                "domain": "भोजन एवं व्यंजन",
                "is_abstaining": True,
                "is_hallucinated": True,
            },
        ]

        culture_rows = summary.build_culture_rows(rows)

        self.assertEqual(len(culture_rows), 1)
        self.assertEqual(culture_rows[0]["canonical_domain"], "Food & Cuisine")
        self.assertEqual(culture_rows[0]["total"], 3)
        self.assertEqual(culture_rows[0]["refusals"], 1)

    def test_skips_outputs_without_a_matching_judge_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(
                root / "model_outputs_tamil" / "precise_ta" / "ta" / "qwen-mt-plus.jsonl",
                [
                    {
                        "id": 1,
                        "question": "known food?",
                        "gold_answer": "Idli",
                        "model_answer": "Idli",
                        "domain": "Food & Cuisine",
                        "model_tag": "qwen-mt-plus",
                        "dataset_type": "precise",
                        "lang": "ta",
                    }
                ],
            )

            result = summary.build_tables(root)

            self.assertEqual(result.domain_rows, [])
            self.assertEqual(result.culture_rows, [])
            self.assertEqual(len(result.skipped_pairs), 1)
            self.assertIn("no matching judge file", result.skipped_pairs[0]["reason"])

    def test_writes_csv_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "tables"
            rows = [
                {
                    "culture": "gujarati",
                    "dataset_type": "precise",
                    "language": "en",
                    "domain": "Food & Cuisine",
                    "model": "gpt-5",
                    "total": 1,
                    "refusals": 0,
                    "non_refusals": 1,
                    "hallucinations": 0,
                    "hallucinations_not_refused": 0,
                    "correct": 1,
                    "refusal_rate": 0.0,
                    "hallucination_rate_not_refused": 0.0,
                    "hallucination_rate_all": 0.0,
                    "correct_rate": 1.0,
                }
            ]

            summary.write_csv(out_dir / "domain.csv", rows, summary.DOMAIN_COLUMNS)

            with (out_dir / "domain.csv").open(encoding="utf-8", newline="") as f:
                written = list(csv.DictReader(f))
            self.assertEqual(written[0]["culture"], "gujarati")
            self.assertEqual(written[0]["correct_rate"], "1.0000")


if __name__ == "__main__":
    unittest.main()
