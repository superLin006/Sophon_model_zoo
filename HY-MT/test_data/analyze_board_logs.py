#!/usr/bin/env python3
"""Compare board regression logs with the native greedy baseline."""

import argparse
import difflib
import json
import re
import statistics
from pathlib import Path


CASE_RE = re.compile(r"^===== (\S+?)(?: repeat=\d+/\d+)? =====$")
PERF_RE = re.compile(r"^Prefill: ([\d.]+) ms, decode: ([\d.]+) token/s$")
ALIASES = {"terminology_short": "terminology", "formatting_short": "formatting"}


def parse_log(path: Path) -> dict:
    cases = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CASE_RE.match(line)
        if match:
            current = match.group(1)
            cases.setdefault(current, {"outputs": [], "prefill_ms": [], "decode_tps": []})
        elif current and line.startswith("Translation: "):
            cases[current]["outputs"].append(line.removeprefix("Translation: "))
        elif current and (match := PERF_RE.match(line)):
            cases[current]["prefill_ms"].append(float(match.group(1)))
            cases[current]["decode_tps"].append(float(match.group(2)))
    return cases


def summarize(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--w8-log", required=True, type=Path)
    parser.add_argument("--w4-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline = {
        case["id"]: case["output"]
        for case in json.loads(args.baseline.read_text(encoding="utf-8"))["cases"]
    }
    variants = {"w8bf16": parse_log(args.w8_log), "w4bf16_g64": parse_log(args.w4_log)}
    report = {"variants": {}}
    for variant, cases in variants.items():
        comparisons = []
        prefill = []
        decode = []
        for board_id, result in cases.items():
            baseline_id = ALIASES.get(board_id, board_id)
            output = result["outputs"][0]
            reference = baseline[baseline_id]
            comparisons.append({
                "id": baseline_id,
                "exact_match": output == reference,
                "character_similarity": difflib.SequenceMatcher(None, reference, output).ratio(),
                "reference": reference,
                "output": output,
            })
            prefill.extend(result["prefill_ms"])
            decode.extend(result["decode_tps"])
        report["variants"][variant] = {
            "exact_matches": sum(item["exact_match"] for item in comparisons),
            "case_count": len(comparisons),
            "mean_character_similarity": statistics.fmean(
                item["character_similarity"] for item in comparisons
            ),
            "prefill_ms": summarize(prefill),
            "decode_tokens_per_second": summarize(decode),
            "cases": comparisons,
        }

    w8_decode = report["variants"]["w8bf16"]["decode_tokens_per_second"]["mean"]
    w4_decode = report["variants"]["w4bf16_g64"]["decode_tokens_per_second"]["mean"]
    report["w4_decode_speedup_percent"] = (w4_decode / w8_decode - 1.0) * 100.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {
            "exact_matches": data["exact_matches"],
            "mean_character_similarity": data["mean_character_similarity"],
            "prefill_ms_mean": data["prefill_ms"]["mean"],
            "decode_tps_mean": data["decode_tokens_per_second"]["mean"],
        }
        for name, data in report["variants"].items()
    }, ensure_ascii=False, indent=2))
    print(f"W4 decode speedup: {report['w4_decode_speedup_percent']:.2f}%")


if __name__ == "__main__":
    main()
