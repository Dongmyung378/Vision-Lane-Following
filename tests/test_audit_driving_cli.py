"""Source/output ownership and interruption regressions for native QA CLI."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from scripts import audit_driving


class AuditDrivingCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.native = self.root / "native"
        self.driving = self.native / "driving"
        self.driving.mkdir(parents=True)
        self.source = self.driving / "steps.jsonl"
        self.source.write_bytes(b"preserved native source")
        self.output = self.root / "report.json"

    def test_report_cannot_overwrite_source_or_prior_evidence(self):
        for path, existing in ((self.source, b"preserved native source"), (self.output, b'{"status":"PASS"}')):
            path.write_bytes(existing)
            with patch("scripts.audit_driving.run_audit") as run, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                audit_driving.main(["--native", str(self.native), "--output", str(path)])
            run.assert_not_called()
            self.assertEqual(path.read_bytes(), existing)

    def test_interruption_publishes_fail_after_running_without_raw_exception(self):
        def interrupted(native):
            self.assertEqual(json.loads(self.output.read_text())["status"], "RUNNING")
            raise KeyboardInterrupt("private raw contents")
        stdout = io.StringIO()
        with patch("scripts.audit_driving.run_audit", side_effect=interrupted), redirect_stdout(stdout):
            self.assertEqual(audit_driving.main(["--native", str(self.native), "--output", str(self.output)]), 1)
        self.assertEqual(json.loads(self.output.read_text())["status"], "FAIL")
        self.assertNotIn("private raw contents", stdout.getvalue())
        self.assertEqual(self.source.read_bytes(), b"preserved native source")

    def test_report_changed_by_another_writer_is_preserved(self):
        replacement = b"other writer's evidence"
        def changed(native):
            self.output.write_bytes(replacement)
            return {"status": "PASS", "errors": []}
        with patch("scripts.audit_driving.run_audit", side_effect=changed), redirect_stdout(io.StringIO()):
            self.assertEqual(audit_driving.main(["--native", str(self.native), "--output", str(self.output)]), 1)
        self.assertEqual(self.output.read_bytes(), replacement)

    def test_positive_gate_failure_does_not_modify_native_sources(self):
        with patch("scripts.audit_driving.validate_driving_manifest", return_value={"status": "FAIL", "errors": [{"code": "incomplete"}]}):
            report = audit_driving.run_audit(self.native)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(report["sources_preserved"])
        self.assertEqual(self.source.read_bytes(), b"preserved native source")


if __name__ == "__main__":
    unittest.main()
