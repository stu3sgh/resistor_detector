#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def infer_truth_label(image_path: Path, positive_suffix: str, negative_suffix: str) -> str | None:
    stem = image_path.stem
    if stem.endswith(positive_suffix):
        return "OK"
    if stem.endswith(negative_suffix):
        return "NG"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate detector outputs using filename suffix labels such as -1=OK and -2=NG."
    )
    parser.add_argument("--results", required=True, help="Path to results JSON produced by the detector")
    parser.add_argument(
        "--positive-suffix",
        default="-1",
        help="Filename suffix that means resistor present / truth OK",
    )
    parser.add_argument(
        "--negative-suffix",
        default="-2",
        help="Filename suffix that means resistor absent / truth NG",
    )
    parser.add_argument("--output-json", help="Optional path to save the evaluation summary as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_path = Path(args.results).resolve()
    rows = json.loads(results_path.read_text(encoding="utf-8"))

    confusion = Counter({"TP": 0, "TN": 0, "FP": 0, "FN": 0})
    skipped_images: list[str] = []
    mismatches: list[dict[str, Any]] = []
    reason_counter = Counter()

    evaluated_count = 0
    for row in rows:
        image_path = Path(row["image"])
        truth = infer_truth_label(image_path, args.positive_suffix, args.negative_suffix)
        if truth is None:
            skipped_images.append(image_path.name)
            continue

        evaluated_count += 1
        prediction = row.get("result", "UNKNOWN")
        reason_counter[row.get("reason", "unknown")] += 1

        if prediction == "OK" and truth == "OK":
            confusion["TP"] += 1
        elif prediction == "NG" and truth == "NG":
            confusion["TN"] += 1
        elif prediction == "OK" and truth == "NG":
            confusion["FP"] += 1
        elif prediction == "NG" and truth == "OK":
            confusion["FN"] += 1
        else:
            mismatches.append(
                {
                    "image": image_path.name,
                    "prediction": prediction,
                    "truth": truth,
                    "reason": row.get("reason"),
                    "decision_reason": row.get("decision_reason"),
                }
            )
            continue

        if prediction != truth:
            mismatches.append(
                {
                    "image": image_path.name,
                    "prediction": prediction,
                    "truth": truth,
                    "reason": row.get("reason"),
                    "decision_reason": row.get("decision_reason"),
                    "inlier_count": row.get("inlier_count"),
                    "selected_config": row.get("selected_config"),
                }
            )

    accuracy = safe_divide(confusion["TP"] + confusion["TN"], evaluated_count)
    precision_ok = safe_divide(confusion["TP"], confusion["TP"] + confusion["FP"])
    recall_ok = safe_divide(confusion["TP"], confusion["TP"] + confusion["FN"])
    specificity_ng = safe_divide(confusion["TN"], confusion["TN"] + confusion["FP"])
    f1_ok = safe_divide(2 * precision_ok * recall_ok, precision_ok + recall_ok)

    summary = {
        "results_path": str(results_path),
        "evaluated_count": evaluated_count,
        "skipped_count": len(skipped_images),
        "skipped_images": skipped_images,
        "confusion": dict(confusion),
        "metrics": {
            "accuracy": round(accuracy, 6),
            "precision_ok": round(precision_ok, 6),
            "recall_ok": round(recall_ok, 6),
            "specificity_ng": round(specificity_ng, 6),
            "f1_ok": round(f1_ok, 6),
        },
        "reason_counts": dict(reason_counter),
        "mismatches": mismatches,
    }

    print(f"evaluated={evaluated_count} skipped={len(skipped_images)}")
    print(
        "confusion: TP={TP} FP={FP} FN={FN} TN={TN}".format(
            TP=confusion["TP"], FP=confusion["FP"], FN=confusion["FN"], TN=confusion["TN"]
        )
    )
    print(
        "metrics: accuracy={accuracy:.4f} precision_ok={precision_ok:.4f} recall_ok={recall_ok:.4f} specificity_ng={specificity_ng:.4f} f1_ok={f1_ok:.4f}".format(
            **summary["metrics"]
        )
    )
    if mismatches:
        print("mismatches:")
        for item in mismatches:
            print(json.dumps(item, ensure_ascii=False))
    else:
        print("mismatches: []")

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())