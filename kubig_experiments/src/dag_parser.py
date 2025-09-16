#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple
import re

# parse "A->B;" edges
_EDGE_RE = re.compile(r'([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\s*;')

def parse_edges_from_dot(graph_txt: str) -> List[Tuple[str, str]]:
    return [(u.strip(), v.strip()) for (u, v) in _EDGE_RE.findall(graph_txt)]

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

def extract_roles_general(graph_txt: str, outcome: str) -> Dict[str, List[str]]:
    """
    일반화된 규칙으로 Treatment/Mediator/Confounder 추출
    """
    edges = parse_edges_from_dot(graph_txt)
    G_out, G_in, nodes = build_graph(edges)
    if outcome not in nodes:
        raise ValueError(f"Outcome '{outcome}' not found in DAG.")

    parents_y = set(G_in.get(outcome, set()))

    candidates = []
    for t in sorted(nodes):
        if t == outcome: continue
        if t in parents_y: continue
        if not has_path(G_out, t, outcome): continue
        meds = sorted(m for m in descendants(G_out, t) if m != outcome and has_path(G_out, m, outcome))
        score = (len(meds), )
        candidates.append((score, t, meds))

    if not candidates:
        for t in sorted(parents_y):
            meds = sorted(m for m in descendants(G_out, t) if m != outcome and has_path(G_out, m, outcome))
            score = (len(meds), )
            candidates.append((score, t, meds))

    if not candidates:
        raise RuntimeError("No viable treatment candidate found.")

    _, treatment, mediators = max(candidates, key=lambda x: x[0])
    confounders = sorted(set(G_in.get(treatment, set())) & parents_y)

    return {
        "treatment": treatment,
        "mediators": mediators,
        "confounders": confounders,
    }
