from __future__ import annotations
from typing import Dict, Any, List
from .tools import EnvState


def check_acceptance(env: EnvState, criteria: Dict[str, Any]) -> Dict[str, Any]:
    ok = True
    reasons: List[str] = []

    dep_req = criteria.get("deployment")
    if dep_req:
        svc = dep_req.get("service"); ns = dep_req.get("namespace"); gte = dep_req.get("replicas_gte")
        key = (svc, ns)
        if key not in env.deployments:
            ok = False; reasons.append(f"missing deployment {svc}:{ns}")
        else:
            reps = env.deployments[key].replicas
            if gte is not None and reps < gte:
                ok = False; reasons.append(f"replicas {reps} < {gte}")

    log_contains = criteria.get("incident_log_contains")
    if log_contains:
        if not any(log_contains in e.get("message", "") for e in env.incident_log):
            ok = False; reasons.append("incident log missing required message")

    ticket_status = criteria.get("ticket_status")
    if ticket_status:
        if env.ticket.status != ticket_status:
            ok = False; reasons.append(f"ticket status {env.ticket.status} != {ticket_status}")

    return {"valid": ok, "reasons": reasons}


def check_technical_success(env: EnvState, criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Check only technical success criteria, excluding administrative tasks like ticket status."""
    ok = True
    reasons: List[str] = []

    # Check deployment scaling (main technical task)
    dep_req = criteria.get("deployment")
    if dep_req:
        svc = dep_req.get("service"); ns = dep_req.get("namespace"); gte = dep_req.get("replicas_gte")
        key = (svc, ns)
        if key not in env.deployments:
            ok = False; reasons.append(f"missing deployment {svc}:{ns}")
        else:
            reps = env.deployments[key].replicas
            if gte is not None and reps < gte:
                ok = False; reasons.append(f"replicas {reps} < {gte}")

    # Check incident logging (documentation requirement)
    log_contains = criteria.get("incident_log_contains")
    if log_contains:
        if not any(log_contains in e.get("message", "") for e in env.incident_log):
            ok = False; reasons.append("incident log missing required message")

    # Info: Ticket status is NOT included in technical success - it's scored separately

    return {"valid": ok, "reasons": reasons}


def partial_credit(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Compute simple partial credit metrics from the execution trace.

    Metrics:
      - tool_syntax_ok: no tool result with ok=false due to argument error
      - verification_performed: whether a metrics_query was called after an action
      - incident_logged: whether incident_log was called
      - ticket_updated: whether ticket_update was called
    """
    calls = trace.get("tool_calls", [])
    any_arg_error = any((not c.get("result", {}).get("ok", False)) and "argument error" in str(c.get("result", {}).get("error", "")) for c in calls)
    verification = False
    did_action = False
    incident_logged = any(c.get("call", {}).get("name") == "incident_log" for c in calls)
    ticket_updated = any(c.get("call", {}).get("name") == "ticket_update" for c in calls)
    for c in calls:
        name = c.get("call", {}).get("name")
        if name in ("k8s_scale", "k8s_restart", "feature_flag_set"):
            did_action = True
        if did_action and name == "metrics_query":
            verification = True
    return {
        "tool_syntax_ok": not any_arg_error,
        "verification_performed": verification,
        "incident_logged": incident_logged,
        "ticket_updated": ticket_updated,
    }


def compute_weighted_score(acceptance: Dict[str, Any], partial: Dict[str, Any], technical: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a weighted score combining technical success and partial credit.
    
    Scoring weights:
    - Technical task completion (deployment + logging): 60%
    - Ticket updating: 15%
    - Incident logging: 15%  
    - Verification: 5%
    - Tool syntax: 5%
    """
    weights = {
        "technical_task": 0.60,       # Core technical success (deployment + incident log)
        "ticket_updated": 0.15,       # Administrative task completion
        "incident_logged": 0.15,      # Process compliance
        "verification": 0.05,         # Best practices
        "tool_syntax": 0.05,          # Basic competency
    }
    
    # Technical task score (deployment + incident logging)
    technical_score = 1.0 if technical.get("valid", False) else 0.0
    
    # Partial credit scores (0 or 1 for each)
    ticket_score = 1.0 if partial.get("ticket_updated", False) else 0.0
    incident_score = 1.0 if partial.get("incident_logged", False) else 0.0
    verification_score = 1.0 if partial.get("verification_performed", False) else 0.0
    syntax_score = 1.0 if partial.get("tool_syntax_ok", False) else 0.0
    
    # Weighted total
    total_score = (
        technical_score * weights["technical_task"] +
        ticket_score * weights["ticket_updated"] +
        incident_score * weights["incident_logged"] +
        verification_score * weights["verification"] +
        syntax_score * weights["tool_syntax"]
    )
    
    return {
        "total_score": round(total_score * 100, 1),  # percentage
        "max_score": 100.0,
        "breakdown": {
            "technical_task": round(technical_score * weights["technical_task"] * 100, 1),
            "ticket_updated": round(ticket_score * weights["ticket_updated"] * 100, 1),
            "incident_logged": round(incident_score * weights["incident_logged"] * 100, 1),
            "verification": round(verification_score * weights["verification"] * 100, 1),
            "tool_syntax": round(syntax_score * weights["tool_syntax"] * 100, 1),
        },
        "weights": weights
    }
