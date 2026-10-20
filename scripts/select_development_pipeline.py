"""Apply frozen development selection to saved offline and native reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_tools.contracts import strict_json_loads
from carla_tools.development_selection import LEARNED_ARMS, select_development_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in LEARNED_ARMS:
        parser.add_argument("--" + arm.replace("_", "-") + "-offline", type=Path, required=True)
    for name in ("nominal", "robustness", "nominal-protocol", "robustness-protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = {arm + "_offline": getattr(args, arm + "_offline") for arm in LEARNED_ARMS}
    inputs.update({name: getattr(args, name) for name in
                   ("nominal", "robustness", "nominal_protocol", "robustness_protocol")})
    values, hashes = {}, {}
    for name, path in inputs.items():
        # Hash and parse the same bytes; later file replacement cannot change
        # the identity attributed to the decision's input.
        payload = path.read_bytes()
        values[name] = strict_json_loads(payload.decode("utf-8"))
        hashes[name] = hashlib.sha256(payload).hexdigest()
    decision = select_development_pipeline(
        {arm: values[arm + "_offline"] for arm in LEARNED_ARMS},
        values["nominal"], values["robustness"],
        values["nominal_protocol"], values["robustness_protocol"],
    )
    report = {"status": "PASS", "decision": decision, "input_sha256": hashes,
              "pointer_updated": False, "local_gpu_used": False}
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "selected_arm": decision["selected_arm"],
        "eligible_improvements": decision["eligible_improvements"],
        "balanced_error_score_m": decision["balanced_error_score_m"],
        "pointer_updated": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
