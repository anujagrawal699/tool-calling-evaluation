from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
import time
import random


class Deployment(BaseModel):
    service: str
    namespace: str
    replicas: int = 1
    restarted_at: Optional[float] = None


class Ticket(BaseModel):
    status: str = "open"  # open|investigating|mitigated|resolved
    note: str = ""


class EnvState(BaseModel):
    deployments: Dict[Tuple[str, str], Deployment] = Field(default_factory=dict)
    feature_flags: Dict[str, bool] = Field(default_factory=dict)
    ticket: Ticket = Field(default_factory=Ticket)
    incident_log: List[Dict[str, Any]] = Field(default_factory=list)
    seed: int = 42


class ToolResult(BaseModel):
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _rng(env: EnvState, key: str) -> random.Random:
    # Deterministic RNG for reproducibility
    h = hash(key) ^ env.seed
    return random.Random(h)


def metrics_query(env: EnvState, service: str, metric: str, minutes: int, namespace: str) -> ToolResult:
    if metric not in {"latency_p95", "error_rate", "qps"}:
        return ToolResult(ok=False, error=f"invalid metric: {metric}")
    if minutes <= 0 or minutes > 120:
        return ToolResult(ok=False, error=f"minutes out of range: {minutes}")
    key = (service, namespace)
    if key not in env.deployments:
        return ToolResult(ok=False, error=f"unknown deployment: {service} in {namespace}")
    dep = env.deployments[key]

    # Simple synthetic model: more replicas -> lower error_rate & latency; higher qps
    rnd = _rng(env, f"{service}:{namespace}:{metric}:{minutes}")
    base = 0.0
    status = "unknown"
    recommendation = ""
    
    if metric == "error_rate":
        base = max(0.005, 0.06 - 0.006 * dep.replicas)  # 6% down to ~<1%
        noise = rnd.uniform(-0.003, 0.003)
        value = max(0.0, base + noise)
        
        # status assessment for error rate
        if value < 0.015:    # < 1.5%
            status = "good"
            recommendation = "Error rate is acceptable"
        elif value < 0.025:  # < 2.5%  
            status = "acceptable"
            recommendation = "Error rate improved but could be better"
        else:                # >= 2.5%
            status = "concerning"
            recommendation = "Error rate still high, consider further scaling"
            
    elif metric == "latency_p95":
        base = max(80.0, 300.0 - 20.0 * dep.replicas)
        noise = rnd.uniform(-10.0, 10.0)
        value = max(0.0, base + noise)
        
        # status assessment for latency
        if value < 120.0:    # < 120ms
            status = "good"
            recommendation = "Latency is acceptable"
        elif value < 200.0:  # < 200ms
            status = "acceptable" 
            recommendation = "Latency improved but could be better"
        else:                # >= 200ms
            status = "concerning"
            recommendation = "Latency still high, consider further scaling"
            
    else:  # qps
        base = 50.0 * dep.replicas
        noise = rnd.uniform(-10.0, 10.0)
        value = max(0.0, base + noise)
        status = "info"
        recommendation = f"Current throughput with {dep.replicas} replicas"

    return ToolResult(ok=True, data={
        "metric": metric, 
        "minutes": minutes, 
        "value": value,
        "status": status,
        "recommendation": recommendation
    })


def k8s_scale(env: EnvState, service: str, replicas: int, namespace: str) -> ToolResult:
    if replicas < 1 or replicas > 100:
        return ToolResult(ok=False, error=f"replicas out of range: {replicas}")
    key = (service, namespace)
    if key not in env.deployments:
        return ToolResult(ok=False, error=f"unknown deployment: {service} in {namespace}")
    env.deployments[key].replicas = replicas
    return ToolResult(ok=True, data={"service": service, "namespace": namespace, "replicas": replicas})


def k8s_restart(env: EnvState, service: str, namespace: str) -> ToolResult:
    key = (service, namespace)
    if key not in env.deployments:
        return ToolResult(ok=False, error=f"unknown deployment: {service} in {namespace}")
    env.deployments[key].restarted_at = time.time()
    return ToolResult(ok=True, data={"service": service, "namespace": namespace, "restarted_at": env.deployments[key].restarted_at})


def feature_flag_set(env: EnvState, flag: str, enabled: bool) -> ToolResult:
    env.feature_flags[flag] = enabled
    return ToolResult(ok=True, data={"flag": flag, "enabled": enabled})


def incident_log(env: EnvState, message: str, severity: str) -> ToolResult:
    if severity not in {"info", "warning", "critical"}:
        return ToolResult(ok=False, error=f"invalid severity: {severity}")
    entry = {"ts": time.time(), "severity": severity, "message": message}
    env.incident_log.append(entry)
    return ToolResult(ok=True, data=entry)


def ticket_update(env: EnvState, status: str, note: str) -> ToolResult:
    if status not in {"open", "investigating", "mitigated", "resolved"}:
        return ToolResult(ok=False, error=f"invalid status: {status}")
    env.ticket.status = status
    env.ticket.note = note
    return ToolResult(ok=True, data={"status": status, "note": note})


TOOL_REGISTRY = {
    "metrics_query": metrics_query,
    "k8s_scale": k8s_scale,
    "k8s_restart": k8s_restart,
    "feature_flag_set": feature_flag_set,
    "incident_log": incident_log,
    "ticket_update": ticket_update,
}
