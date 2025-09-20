import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
from src.harness import Harness, Scenario, build_env, ToolCall
from src.scoring import check_acceptance, partial_credit, compute_weighted_score, check_technical_success

ROOT = Path(__file__).parent


def load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, data: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


log = logging.getLogger("eval")


def run_ground_truth(h: Harness, scenarios: List[Scenario], limit: int) -> Dict[str, Any]:
    gt = load_json(ROOT / "data" / "ground_truth.json")
    results = {"runs": []}
    for sc in scenarios[:limit]:
        env = build_env(sc.initial_state)
        seq = [ToolCall(**c) for c in gt.get(str(sc.id), [])]
        trace = h.run_sequence(env, seq)
        acc = check_acceptance(env, sc.acceptance_criteria)
        technical = check_technical_success(env, sc.acceptance_criteria)
        partial = partial_credit(trace.model_dump())
        weighted = compute_weighted_score(acc, partial, technical)
        log.info(f"[ground-truth] scenario={sc.id} valid={acc['valid']} technical={technical['valid']} score={weighted['total_score']}% reasons={acc['reasons']}")
        results["runs"].append({
            "scenario_id": sc.id,
            "valid": acc["valid"],
            "technical_success": technical["valid"],
            "reasons": acc["reasons"],
            "trace": trace.model_dump(),
            "partial": partial,
            "weighted_score": weighted,
        })
    return results


def run_llm(h: Harness, scenarios: List[Scenario], limit: int, variant: str) -> Dict[str, Any]:
    results = {"runs": []}
    for sc in scenarios[:limit]:
        env = build_env(sc.initial_state)
        trace = h.run_llm(env, sc, variant=variant)
        acc = check_acceptance(env, sc.acceptance_criteria)
        technical = check_technical_success(env, sc.acceptance_criteria)
        partial = partial_credit(trace.model_dump())
        weighted = compute_weighted_score(acc, partial, technical)
        log.info(f"[{variant}] scenario={sc.id} valid={acc['valid']} technical={technical['valid']} score={weighted['total_score']}% reasons={acc['reasons']}")
        results["runs"].append({
            "scenario_id": sc.id,
            "valid": acc["valid"],
            "technical_success": technical["valid"],
            "reasons": acc["reasons"],
            "trace": trace.model_dump(),
            "partial": partial,
            "weighted_score": weighted,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openrouter/auto")
    ap.add_argument("--variant", choices=["baseline", "improved", "both", "ground-truth"], default="ground-truth")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = ap.parse_args()

    # Logging setup
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # File log
    log_dir = ROOT / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    scenarios_raw = load_json(ROOT / "data" / "scenarios.json")
    scenarios = [Scenario(**s) for s in scenarios_raw]

    h = Harness(model=args.model, project_root=ROOT)

    all_results: Dict[str, Any] = {}
    if args.variant == "ground-truth":
        all_results["ground_truth"] = run_ground_truth(h, scenarios, args.limit)
    elif args.variant == "both":
        all_results["baseline"] = run_llm(h, scenarios, args.limit, variant="baseline")
        all_results["improved"] = run_llm(h, scenarios, args.limit, variant="improved")
    else:
        all_results[args.variant] = run_llm(h, scenarios, args.limit, variant=args.variant)

    results_path = ROOT / "results" / "results.json"
    save_json(results_path, all_results)
    print("Wrote results/results.json")

    # write a minimal Markdown summary
    md_lines = ["# Evaluation Results", ""]
    for variant, payload in all_results.items():
        runs = payload.get("runs", [])
        passed = sum(1 for r in runs if r.get("valid"))
        technical_passed = sum(1 for r in runs if r.get("technical_success", False))
        total = len(runs)
        avg_score = sum(r.get("weighted_score", {}).get("total_score", 0) for r in runs) / max(total, 1)
        
        md_lines.append(f"## {variant}")
        md_lines.append(f"**Full Pass Rate**: {passed}/{total} ({100*passed//max(total,1)}%)")
        md_lines.append(f"**Technical Success Rate**: {technical_passed}/{total} ({100*technical_passed//max(total,1)}%)")
        md_lines.append(f"**Average Score**: {avg_score:.1f}/100.0")
        md_lines.append("")
        
        if total:
            pc = {
                "tool_syntax_ok": sum(1 for r in runs if r.get("partial", {}).get("tool_syntax_ok")),
                "verification_performed": sum(1 for r in runs if r.get("partial", {}).get("verification_performed")),
                "incident_logged": sum(1 for r in runs if r.get("partial", {}).get("incident_logged")),
                "ticket_updated": sum(1 for r in runs if r.get("partial", {}).get("ticket_updated")),
            }
            md_lines.append("### Component Performance:")
            md_lines.append(f"- tool_syntax_ok: {pc['tool_syntax_ok']}/{total} ({100*pc['tool_syntax_ok']//total}%)")
            md_lines.append(f"- verification_performed: {pc['verification_performed']}/{total} ({100*pc['verification_performed']//total}%)")
            md_lines.append(f"- incident_logged: {pc['incident_logged']}/{total} ({100*pc['incident_logged']//total}%)")
            md_lines.append(f"- ticket_updated: {pc['ticket_updated']}/{total} ({100*pc['ticket_updated']//total}%)")
            
            # Detailed scoring breakdown - average across all scenarios
            if runs:
                avg_breakdown = {}
                for component in ["technical_task", "ticket_updated", "incident_logged", "verification", "tool_syntax"]:
                    total_component_score = sum(r.get("weighted_score", {}).get("breakdown", {}).get(component, 0) for r in runs)
                    avg_breakdown[component] = total_component_score / max(total, 1)
                
                if avg_breakdown:
                    md_lines.append("")
                    md_lines.append("### Average Score Breakdown:")
                    for component, score in avg_breakdown.items():
                        md_lines.append(f"- {component}: {score:.1f}%")
        md_lines.append("")
    (ROOT / "results" / "results.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("Wrote results/results.md")


if __name__ == "__main__":
    main()
