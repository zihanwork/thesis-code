from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from embodied_gap.experiments.final_protocol import verify_final_protocol


class FinalProtocolTests(unittest.TestCase):
    def test_verifies_hashes_and_refuses_an_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "artifact.txt"
            artifact.write_text("frozen", encoding="utf-8")
            output_root = root / "final-output"
            protocol_path = root / "protocol.json"
            protocol_path.write_text(
                json.dumps(
                    {
                        "protocol_id": "unit-final",
                        "required_git_tag": "unit-missing-tag",
                        "artifacts": [
                            {
                                "role": "unit",
                                "path": str(artifact),
                                "sha256": hashlib.sha256(b"frozen").hexdigest(),
                            }
                        ],
                        "experiments": [
                            {
                                "id": "unit",
                                "output_root": str(output_root),
                            }
                        ],
                        "totals": {"worst_case_llm_calls": 0},
                    }
                ),
                encoding="utf-8",
            )

            first = verify_final_protocol(
                protocol_path,
                repo_root=Path.cwd(),
                require_git_tag=False,
                require_clean_worktree=False,
            )
            self.assertTrue(first["valid"])
            self.assertTrue(first["artifacts_valid"])
            self.assertTrue(first["outputs_unrun"])

            output_root.mkdir()
            (output_root / "run_index.jsonl").write_text("{}\n", encoding="utf-8")
            second = verify_final_protocol(
                protocol_path,
                repo_root=Path.cwd(),
                require_git_tag=False,
                require_clean_worktree=False,
            )
            self.assertFalse(second["valid"])
            self.assertFalse(second["outputs_unrun"])


if __name__ == "__main__":
    unittest.main()
