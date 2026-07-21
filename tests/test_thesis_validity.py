from __future__ import annotations

import unittest

from embodied_gap.analysis.thesis_validity import (
    audit_rag_overlap,
    audit_rq3_identifiability,
    family_clustered_paired_analysis,
)


def _task(
    task_id: str,
    instruction: str,
    family: str,
    plan: list[str],
) -> dict[str, object]:
    return {
        "id": task_id,
        "instruction": instruction,
        "slots": {"task_family": family},
        "gold_plan": plan,
    }


def _metric(task_id: str, family: str, success: bool) -> dict[str, object]:
    return {
        "task_id": task_id,
        "task_success": success,
        "metadata": {"task_family": family},
    }


class ThesisValidityTests(unittest.TestCase):
    def test_rag_overlap_separates_id_family_instruction_and_plan_overlap(self) -> None:
        training = [
            _task("train-a", "Drink", "Drink", ["walk()", "grab()"]),
            _task("train-b", "Read", "Read", ["walk()", "read()"]),
        ]
        heldout = [
            _task("test-a", " drink ", "Drink", ["walk()", "grab()"]),
            _task("test-c", "Work", "Work", ["type()"]),
        ]
        runs = [
            {
                "task_id": "test-a",
                "planner_name": "P1_rag",
                "harness_mode": "H0_open_loop",
                "initial_plan": {
                    "metadata": {"retrieved": "train-a", "retrieval_score": 1.0}
                },
            },
            {
                "task_id": "test-c",
                "planner_name": "P1_rag",
                "harness_mode": "H0_open_loop",
                "initial_plan": {
                    "metadata": {"retrieved": "train-b", "retrieval_score": 0.25}
                },
            },
        ]

        report = audit_rag_overlap(training, heldout, runs)

        self.assertEqual(report["task_id_overlap"]["count"], 0)
        self.assertEqual(report["normalized_instruction_seen"]["count"], 1)
        self.assertEqual(report["task_family_seen"]["count"], 1)
        self.assertEqual(report["gold_plan_exactly_seen_anywhere"]["count"], 1)
        self.assertEqual(
            report["selected_demo_audit"]["selected_same_family"]["count"], 1
        )
        self.assertEqual(
            report["selected_demo_audit"]["selected_exact_gold_plan"]["count"], 1
        )

    def test_family_clustered_analysis_resamples_whole_families(self) -> None:
        left = [
            _metric("a1", "A", False),
            _metric("a2", "A", False),
            _metric("b1", "B", False),
            _metric("b2", "B", False),
        ]
        right = [
            _metric("a1", "A", True),
            _metric("a2", "A", True),
            _metric("b1", "B", True),
            _metric("b2", "B", False),
        ]

        report = family_clustered_paired_analysis(
            left, right, samples=200, seed=7
        )

        self.assertEqual(report["paired_task_count"], 4)
        self.assertEqual(report["task_family_count"], 2)
        self.assertAlmostEqual(report["task_weighted_uplift"], 0.75)
        self.assertAlmostEqual(report["equal_family_uplift"], 0.75)
        self.assertAlmostEqual(report["exact_family_sign_flip_p_value"], 0.5)
        self.assertEqual(report["bootstrap"]["unit"], "task_family")

    def test_rq3_audit_refuses_to_infer_interaction_without_fourth_cell(self) -> None:
        report = audit_rq3_identifiability(
            {
                "P0_engineered_prompt__H0_open_loop",
                "P1_rag__H0_open_loop",
                "P1_rag__H2_llm_reflection",
            }
        )

        self.assertFalse(report["full_factorial_interaction_identifiable"])
        self.assertEqual(
            report["missing_cells"],
            ["P0_engineered_prompt__H2_llm_reflection"],
        )
        self.assertIn("conditional", report["claim_policy"].lower())

if __name__ == "__main__":
    unittest.main()
