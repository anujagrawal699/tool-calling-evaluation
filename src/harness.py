from __future__ import annotations
from typing import Dict, Any, List, Optional
import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from .tools import EnvState, Deployment, TOOL_REGISTRY, ToolResult
import logging

log = logging.getLogger("harness")


class Scenario(BaseModel):
    id: int
    user_prompt: str
    initial_state: Dict[str, Any]
    acceptance_criteria: Dict[str, Any]


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class ExecutionTrace(BaseModel):
    tool_calls: List[Dict[str, Any]]
    final: Dict[str, Any]


def build_env(initial_state: Dict[str, Any]) -> EnvState:
    env = EnvState()
    # Deployments
    for item in initial_state.get("deployments", []):
        key = (item["service"], item["namespace"])
        env.deployments[key] = Deployment(service=item["service"], namespace=item["namespace"], replicas=item.get("replicas", 1))
    # Feature flags
    for k, v in initial_state.get("feature_flags", {}).items():
        env.feature_flags[k] = bool(v)
    # Ticket
    if "ticket" in initial_state:
        env.ticket.status = initial_state["ticket"].get("status", env.ticket.status)
        env.ticket.note = initial_state["ticket"].get("note", env.ticket.note)
    return env


def exec_tool(env: EnvState, call: ToolCall) -> Dict[str, Any]:
    fn = TOOL_REGISTRY.get(call.name)
    if not fn:
        return {"ok": False, "error": f"unknown tool: {call.name}"}
    try:
        res: ToolResult = fn(env=env, **call.arguments)
        return res.model_dump()
    except TypeError as e:
        return {"ok": False, "error": f"argument error: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"runtime error: {e}"}


class Harness:
    def __init__(self, model: str | None = None, project_root: Optional[Path] = None):
        self.model = model
        self.root = project_root or Path(__file__).resolve().parents[1]
        # Load .env from project root
        try:
            load_dotenv(self.root / ".env")
        except Exception:
            pass
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.prompts_dir = self.root / "prompts"

    def _tool_schema_text(self) -> str:
        # Human-readable schema for the model
        parts = [
            "TOOLS AVAILABLE (invoke one at a time):",
            "- metrics_query(service: string, metric: 'latency_p95'|'error_rate'|'qps', minutes: int[1..120], namespace: string)",
            "  Returns: value, status ('good'|'acceptable'|'concerning'), recommendation",
            "- k8s_scale(service: string, replicas: int[1..100], namespace: string)",
            "- k8s_restart(service: string, namespace: string)",
            "- feature_flag_set(flag: string, enabled: boolean)",
            "- incident_log(message: string, severity: 'info'|'warning'|'critical')",
            "- ticket_update(status: 'open'|'investigating'|'mitigated'|'resolved', note: string)",
            "\nRESPONSE FORMAT (strict JSON, no extra text):",
            '{"tool_call": {"name": "<tool_name>", "arguments": { /* tool args */ }}}',
            'OR {"final_answer": "<concise result/summary>"}',
        ]
        return "\n".join(parts)

    def _load_prompt(self, variant: str) -> str:
        fname = "baseline.txt" if variant == "baseline" else "improved.txt"
        p = self.prompts_dir / fname
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8")

    def _openrouter_chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 512) -> str:
        if not self.api_key:
            # If no key, return an empty final answer to allow offline smoke test
            return json.dumps({"final_answer": "no_api_key"})
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model or "openrouter/auto",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # For OpenAI-compatible models to enforce JSON outputs
        try:
            if (self.model or "").startswith("openai/"):
                payload["response_format"] = {"type": "json_object"}
        except Exception:
            pass
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content

    def run_sequence(self, env: EnvState, sequence: List[ToolCall], max_steps: int = 8) -> ExecutionTrace:
        trace: List[Dict[str, Any]] = []
        for i, call in enumerate(sequence):
            if i >= max_steps:
                break
            result = exec_tool(env, call)
            trace.append({"step": i + 1, "call": call.model_dump(), "result": result})
        final = {
            "deployments": {f"{svc}:{ns}": dep.replicas for (svc, ns), dep in env.deployments.items()},
            "ticket": env.ticket.model_dump(),
            "feature_flags": env.feature_flags,
            "incident_log": env.incident_log,
        }
        return ExecutionTrace(tool_calls=trace, final=final)

    def run_llm(self, env: EnvState, scenario: Scenario, variant: str = "baseline", max_steps: int = 8) -> ExecutionTrace:
        sys_preamble = self._tool_schema_text()
        variant_prompt = self._load_prompt(variant)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": sys_preamble},
            {"role": "system", "content": variant_prompt},
            {"role": "user", "content": scenario.user_prompt},
        ]

        trace: List[Dict[str, Any]] = []

        def parse_response(txt: str) -> Dict[str, Any]:
            # Expect a single JSON object
            txt = txt.strip()
            # Try to find first JSON object if model adds wrappers
            start = txt.find("{")
            end = txt.rfind("}")
            if start >= 0 and end >= start:
                txt = txt[start:end + 1]
            return json.loads(txt)

        for step in range(1, max_steps + 1):
            try:
                raw = self._openrouter_chat(messages)
            except Exception as e:
                # Communication failure -> stop
                log.error(f"OpenRouter error: {e}")
                messages.append({"role": "system", "content": f"ERROR: {e}"})
                break

            # Try to parse into JSON
            parsed: Dict[str, Any]
            for attempt in range(2):
                try:
                    parsed = parse_response(raw)
                    break
                except Exception:
                    # Ask the model to reformat strictly as JSON
                    log.warning("Model response not valid JSON; requesting reformat")
                    messages.append({"role": "user", "content": "Your last response was not valid JSON. Reply with JSON only as specified."})
                    raw = self._openrouter_chat(messages)
            else:
                # Could not parse
                break

            if "final_answer" in parsed:
                final = {
                    "deployments": {f"{svc}:{ns}": dep.replicas for (svc, ns), dep in env.deployments.items()},
                    "ticket": env.ticket.model_dump(),
                    "feature_flags": env.feature_flags,
                    "incident_log": env.incident_log,
                }
                return ExecutionTrace(tool_calls=trace, final=final)

            call = parsed.get("tool_call")
            if not call or not isinstance(call, dict):
                # Ask to provide a tool_call or final_answer
                log.warning("Missing tool_call and final_answer; prompting model to follow schema")
                messages.append({"role": "user", "content": "You must respond with either a tool_call or a final_answer in JSON."})
                continue

            # Validate call shape
            name = call.get("name")
            args = call.get("arguments") or {}
            if not isinstance(name, str) or not isinstance(args, dict):
                messages.append({"role": "user", "content": "Malformed tool_call. Provide name (string) and arguments (object)."})
                continue

            # Execute tool
            res = exec_tool(env, ToolCall(name=name, arguments=args))
            log.info(f"tool_call step={step} name={name} ok={res.get('ok')} err={res.get('error')}")
            trace.append({"step": step, "call": {"name": name, "arguments": args}, "result": res})

            # Provide tool result back to the model
            messages.append({"role": "assistant", "content": json.dumps({"tool_result": res})})

        # Step limit reached
        final = {
            "deployments": {f"{svc}:{ns}": dep.replicas for (svc, ns), dep in env.deployments.items()},
            "ticket": env.ticket.model_dump(),
            "feature_flags": env.feature_flags,
            "incident_log": env.incident_log,
        }
        return ExecutionTrace(tool_calls=trace, final=final)
