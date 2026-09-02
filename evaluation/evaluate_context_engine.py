import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.context_engine import Candidate, ContextEngine


def average(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Taiwan Context Engine")
    parser.add_argument("router_json", help="ASR router result JSON")
    parser.add_argument(
        "--glossary",
        default=str(PROJECT_ROOT / "datasets" / "taiwan_context" / "glossary.json"),
    )
    parser.add_argument("--output", default="context_engine_results.json")
    args = parser.parse_args()

    with open(args.router_json, "r", encoding="utf-8-sig") as handle:
        router = json.load(handle)

    engine = ContextEngine(glossary_path=args.glossary)
    details = []
    all_cers = []
    automatic_cers = []
    confirmation_cers = []

    for item in router["detailed_results"]:
        baseline = Candidate(
            source="baseline",
            text=item["baseline"]["text"],
            confidence=item["baseline"]["confidence"],
        )
        prompt = Candidate(
            source="prompt",
            text=item["prompt"]["text"],
            confidence=item["prompt"]["confidence"],
        )
        decision = engine.decide(baseline, prompt)
        cer_key = "prompt_cer" if decision.selected_source == "prompt" else "baseline_cer"
        selected_cer = item.get(cer_key)
        cer_value = selected_cer.get("cer") if selected_cer else None
        if cer_value is not None:
            all_cers.append(cer_value)
            target = confirmation_cers if decision.needs_confirmation else automatic_cers
            target.append(cer_value)

        details.append(
            {
                "audio_file": item["audio_file"],
                "decision": decision.to_dict(),
                "selected_cer": selected_cer,
            }
        )

    confirmation_count = sum(
        1 for item in details if item["decision"]["needs_confirmation"]
    )
    result = {
        "summary": {
            "total_files": len(details),
            "average_selected_cer": average(all_cers),
            "automatic_count": len(details) - confirmation_count,
            "confirmation_count": confirmation_count,
            "automatic_coverage": (len(details) - confirmation_count) / len(details),
            "automatic_average_cer": average(automatic_cers),
            "confirmation_group_average_cer": average(confirmation_cers),
        },
        "detailed_results": details,
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print("=== Context Engine評測完成 ===")
    for key, value in result["summary"].items():
        print(f"{key}: {value}")
    print(f"結果已儲存至: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
