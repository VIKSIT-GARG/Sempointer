"""Parsing of run_benchmark.py `--systems` specs (pure: no model loading).

Grammar (comma-separated list; commas inside [...] do not split):

    full
    sempointer                          default full system (pipeline defaults)
    sempointer[key=val;key=val]         kwargs forwarded to SemPointerPipeline
    sempointer:key=val;key=val          same (use quotes in the shell: ';')
    rag | bm25 | hybrid                 top-m with --rag-m (a list expands), or
                                        token-budget mode if --token-budget set
    rag:4 | bm25:2 | hybrid:8           explicit top-m
    rag_budget:1500 | bm25_budget | hybrid_budget:N
                                        fill top-ranked blocks up to N block tokens
                                        (no value -> --token-budget)
    rag[m=4;rrf_k=30] ...               extra constructor kwargs (validated)
    kvpress:<press>:<ratio>

Inside brackets ';' or ',' separate key=val pairs. Values are coerced
int -> float -> bool (true/false) -> None (none/null) -> str.

Unknown constructor kwargs raise ValueError (validate_kwargs), so specs keep
working as SemPointerPipeline grows options and typos fail loudly.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

RETRIEVAL_KINDS = ("rag", "bm25", "hybrid")
_SPEC_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?::([^\[\]]*))?(?:\[(.*)\])?$")


@dataclass
class SystemSpec:
    name: str                      # unique result name (predictions.jsonl "system")
    kind: str                      # sempointer | full | rag | bm25 | hybrid | kvpress
    kwargs: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""


def split_specs(s: str) -> List[str]:
    """Split on commas that are not inside [...]."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced ']' in --systems {s!r}")
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if depth != 0:
        raise ValueError(f"unbalanced '[' in --systems {s!r}")
    out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def coerce(v: str) -> Any:
    v = v.strip()
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for fn in (int, float):
        try:
            return fn(v)
        except ValueError:
            pass
    return v


def parse_kv(body: str, raw: str) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    for part in re.split(r"[;,]", body or ""):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"system spec {raw!r}: expected key=val, got {part!r}")
        k, v = part.split("=", 1)
        k = k.strip()
        if not re.fullmatch(r"[A-Za-z_]\w*", k):
            raise ValueError(f"system spec {raw!r}: invalid kwarg name {k!r}")
        if k in kw:
            raise ValueError(f"system spec {raw!r}: duplicate kwarg {k!r}")
        kw[k] = coerce(v)
    return kw


def _fmt(v: Any) -> str:
    return re.sub(r"[^\w.+-]", "_", str(v))


def _suffix(kw: Dict[str, Any]) -> str:
    return "".join(f"__{k}-{_fmt(kw[k])}" for k in sorted(kw))


def parse_system_specs(systems: str, rag_m: Sequence[int] = (1,),
                       token_budget: Optional[int] = None) -> List[SystemSpec]:
    rag_m = [int(m) for m in rag_m] or [1]
    specs: List[SystemSpec] = []
    for raw in split_specs(systems):
        if raw.startswith("kvpress:"):
            parts = raw.split(":")
            if len(parts) < 2 or not parts[1]:
                raise ValueError(f"kvpress spec needs a press name: {raw!r}")
            ratio = float(parts[2]) if len(parts) > 2 else 0.3
            specs.append(SystemSpec(f"kvpress_{parts[1]}_{ratio}", "kvpress",
                                    {"press_name": parts[1], "compression_ratio": ratio}, raw))
            continue
        mt = _SPEC_RE.match(raw)
        if not mt:
            raise ValueError(f"Unknown system spec: {raw!r}")
        base, colon, bracket = mt.group(1), mt.group(2), mt.group(3)
        if base == "full":
            if colon or bracket:
                raise ValueError(f"'full' takes no arguments: {raw!r}")
            specs.append(SystemSpec("full", "full", {}, raw))
        elif base == "sempointer":
            if colon is not None and colon.strip() and "=" not in colon:
                raise ValueError(f"sempointer spec expects key=val pairs: {raw!r}")
            kw = parse_kv(colon or "", raw)
            for k, v in parse_kv(bracket or "", raw).items():
                if k in kw:
                    raise ValueError(f"system spec {raw!r}: duplicate kwarg {k!r}")
                kw[k] = v
            specs.append(SystemSpec("sempointer" + _suffix(kw), "sempointer", kw, raw))
        elif base in RETRIEVAL_KINDS or base in tuple(f"{k}_budget" for k in RETRIEVAL_KINDS):
            kind = base.replace("_budget", "")
            extra = parse_kv(bracket or "", raw)
            m = extra.pop("m", None)
            budget = extra.pop("token_budget", None)
            if base.endswith("_budget"):
                if colon:
                    budget = int(colon)
                if budget is None:
                    budget = token_budget
                if budget is None:
                    raise ValueError(f"{raw!r} needs a budget (e.g. {base}:1500) or --token-budget")
            elif colon:
                try:
                    m = int(colon)
                except ValueError:
                    raise ValueError(f"{raw!r}: expected integer m after ':'") from None
            if budget is not None:
                if m is not None:
                    raise ValueError(f"{raw!r}: give either m or a token budget, not both")
                specs.append(SystemSpec(f"{kind}_b{int(budget)}" + _suffix(extra), kind,
                                        {"m": None, "token_budget": int(budget), **extra}, raw))
            elif m is not None:
                specs.append(SystemSpec(f"{kind}_m{int(m)}" + _suffix(extra), kind,
                                        {"m": int(m), "token_budget": None, **extra}, raw))
            elif token_budget is not None:
                specs.append(SystemSpec(f"{kind}_b{int(token_budget)}" + _suffix(extra), kind,
                                        {"m": None, "token_budget": int(token_budget), **extra}, raw))
            elif len(rag_m) == 1 and not extra:
                # back-compat: plain `rag` keeps its historical result name
                specs.append(SystemSpec(kind, kind, {"m": rag_m[0], "token_budget": None}, raw))
            else:
                for mm in rag_m:
                    specs.append(SystemSpec(f"{kind}_m{mm}" + _suffix(extra), kind,
                                            {"m": mm, "token_budget": None, **extra}, raw))
        else:
            raise ValueError(f"Unknown system spec: {raw!r}")
    names = [s.name for s in specs]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(f"duplicate system names from --systems: {dups}")
    return specs


def accepted_params(cls) -> Optional[set]:
    """Constructor parameter names, or None if it accepts **kwargs."""
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return {n for n in params if n != "self"}


def validate_kwargs(cls, kwargs: Dict[str, Any], raw: str = "") -> None:
    acc = accepted_params(cls)
    if acc is None:
        return
    unknown = sorted(set(kwargs) - acc)
    if unknown:
        raise ValueError(f"system spec {raw!r}: unknown {cls.__name__} kwarg(s) {unknown}; "
                         f"accepted: {sorted(acc - {'tokenizer'})}")


def merge_kwargs(cls, defaults: Dict[str, Any], spec_kwargs: Dict[str, Any], raw: str = "") -> Dict[str, Any]:
    """Validate spec kwargs, keep only defaults the constructor accepts, spec wins."""
    validate_kwargs(cls, spec_kwargs, raw)
    acc = accepted_params(cls)
    base = dict(defaults) if acc is None else {k: v for k, v in defaults.items() if k in acc}
    base.update(spec_kwargs)
    return base
