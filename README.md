# SRE Tool-Calling Evaluation

Evaluates LLM performance on realistic SRE scenarios using simulated infrastructure tools. Demonstrates how systematic prompting strategies can dramatically improve tool-calling success rates.

## Quick Start

```bash
pip install -r requirements.txt
echo "OPENROUTER_API_KEY=sk-or-..." > .env

# Test ground truth
python run_eval.py --variant ground-truth --limit 1

# Compare prompting strategies
python run_eval.py --variant both --limit 3 --model openrouter/auto --verbose
```

## What We Built

**Evaluation Framework**: 3 SRE scenarios (error spikes, latency issues, staging problems) with realistic acceptance criteria.

**Simulated Tools**: `metrics_query`, `k8s_scale`, `incident_log`, `ticket_update` - behave like real infrastructure APIs but deterministic for evaluation.

**Smart Scoring**: Separates technical success (did they fix the actual problem?) from administrative tasks (did they update the ticket correctly?).

## Key Innovation: Enhanced Tool Feedback

Instead of returning raw metrics, tools provide actionable guidance:

```json
{
  "value": 0.044,
  "status": "concerning",
  "recommendation": "Error rate still high, consider further scaling"
}
```

This teaches LLMs to interpret results and iterate toward solutions.

## Prompting Strategy Results

**Baseline** (minimal instructions)
**Improved** (systematic process + feedback awareness): improved technical succes

The improved prompt teaches:

- Step-by-step SRE methodology
- How to interpret and act on tool feedback
- Iterative scaling until metrics show "good" status
- Proper incident closure procedures

## Why This Matters

Most tool-calling evaluations focus on syntax correctness. We measure actual problem-solving effectiveness. The systematic prompting approach works because it:

1. Breaks complex workflows into clear steps
2. Teaches feedback interpretation
3. Encourages iterative refinement

Results show that LLMs can learn sophisticated tool orchestration through carefully designed prompts and enhanced tool feedback (no fine-tuning necessary).
