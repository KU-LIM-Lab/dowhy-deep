#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple
import re

# -------- DOT 파서 --------
_EDGE_RE = re.compile(r'([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\s*;')
_NODE_RE = re.compile(r'([A-Za-z0-9_]+)\s*\[label="([^"]*)"\]\s*;')

def parse_edges_from_dot(graph_txt: str) -> List[Tuple[str, str]]:
    return [(u.strip(), v.strip()) for (u, v) in _EDGE_RE.findall(graph_txt)]

def parse_nodes_from_dot(graph_txt: str) -> Dict[str, str]:
    """node_id -> label"""
    return {nid: label for nid, label in _NODE_RE.findall(graph_txt)}

# -------- 그래프 유틸 --------
def build_graph(edges: List[Tuple[str, str]]):
    G_out, G_in, nodes = defaultdict(set), defaultdict(set), set()
    for u, v in edges:
        G_out[u].add(v)
        G_in[v].add(u)
        nodes.add(u); nodes.add(v)
        if u not in G_in:  G_in[u]  = G_in[u]
        if v not in G_out: G_out[v] = G_out[v]
    return G_out, G_in, nodes

def has_path(G_out: Dict[str, Set[str]], src: str, dst: str) -> bool:
    if src == dst: return True
    q, seen = deque([src]), {src}
    while q:
        u = q.popleft()
        for w in G_out.get(u, ()):
            if w == dst: return True
            if w not in seen:
                seen.add(w); q.append(w)
    return False

def ancestors(G_in: Dict[str, Set[str]], node: str) -> Set[str]:
    res, stack = set(), list(G_in.get(node, []))
    while stack:
        u = stack.pop()
        if u in res: continue
        res.add(u); stack.extend(G_in.get(u, []))
    return res

def descendants(G_out: Dict[str, Set[str]], node: str) -> Set[str]:
    res, stack = set(), list(G_out.get(node, []))
    while stack:
        u = stack.pop()
        if u in res: continue
        res.add(u); stack.extend(G_out.get(u, []))
    return res

# -------- 역할 추출 --------
_TAG_TRTS = ("(trt", "(treat", "(treatment")
_TAG_OUTS = ("(outcome",)
_TAG_MEDS = ("(med",)
_TAG_ZTRT = ("(Z: trt-only", "(z: trt-only")   # 👈 추가됨

def _lower(s: str) -> str:
    return s.strip().lower()

def extract_roles_general(graph_txt: str, outcome: str) -> Dict[str, List[str]]:
    """
    DOT 라벨 태그 + 구조 규칙으로 Treatment/Mediator/Confounder 추출
    - Mediator = Desc(T) ∩ Anc(Y) − {T,Y}
    - Confounder = Anc(T) ∩ Anc(Y) − {T,Y} − Mediator − Z_trt_only
    """
    nodes_labels = parse_nodes_from_dot(graph_txt)
    edges = parse_edges_from_dot(graph_txt)
    G_out, G_in, nodes = build_graph(edges)

    # 0) outcome 확정: 인자 우선, 없으면 라벨에서 추론
    if not outcome:
        outs = [n for n, l in nodes_labels.items() if any(t in _lower(l) for t in _TAG_OUTS)]
        if len(outs) != 1:
            raise ValueError("Outcome not provided and cannot be uniquely inferred from labels.")
        outcome = outs[0]
    if outcome not in nodes:
        raise ValueError(f"Outcome '{outcome}' not found in DAG.")

    # 1) 라벨 기반 treatment/mediator 우선
    trt_tagged = [n for n, l in nodes_labels.items() if any(t in _lower(l) for t in _TAG_TRTS)]
    meds_tagged = set(n for n, l in nodes_labels.items() if any(t in _lower(l) for t in _TAG_MEDS))

    # 👇 (Z: trt-only) 라벨을 가진 노드들도 미리 추출
    ztrt_only = set(n for n, l in nodes_labels.items() if any(t in _lower(l) for t in _TAG_ZTRT))

    if len(trt_tagged) >= 1:
        treatment = trt_tagged[0]
    else:
        parents_y = set(G_in.get(outcome, set()))
        candidates = []
        def mediator_count(t):
            return len(descendants(G_out, t) & ancestors(G_in, outcome) - {outcome})
        pool = sorted(nodes - {outcome})
        pool1 = [t for t in pool if t not in parents_y and has_path(G_out, t, outcome)]
        if not pool1:
            pool1 = [t for t in pool if has_path(G_out, t, outcome)]
        if not pool1:
            raise RuntimeError("No viable treatment candidate found.")
        treatment = max(pool1, key=mediator_count)

    # 2) Mediator/Confounder 구조 규칙
    anc_y = ancestors(G_in, outcome)
    meds_struct = (descendants(G_out, treatment) & anc_y) - {treatment, outcome}
    mediators = sorted(meds_tagged | meds_struct)

    # 👇 confounder 계산 시 (Z: trt-only) 노드 제거 추가됨
    confounders = sorted((ancestors(G_in, treatment) & anc_y)
                         - {treatment, outcome}
                         - set(mediators)
                         - ztrt_only)

    return {
        "treatment": treatment,
        "mediators": mediators,
        "confounders": confounders,
    }

if __name__ == "__main__":
    from pathlib import Path
    dag_dir = Path("./kubig_experiments/dags/output")
    dag_files = sorted(dag_dir.glob("dag_*.txt"))
    for f in dag_files:
        txt = f.read_text(encoding="utf-8")
        roles = extract_roles_general(txt, outcome="ACQ_180_YN")
        print(f"\n[{f.name}]")
        print("  X (treatment):", roles["treatment"])
        print("  M (mediators):", roles["mediators"])
        print("  C (confounders):", roles["confounders"])

