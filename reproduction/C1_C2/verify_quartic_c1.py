"""Replay unit quartic reduction and coarse rigidity with exact integer cuts."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
from random import Random

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from cut_rank import PRIME, insert, tight_pairs, tight_sa_echelon


PAIRINGS = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))


def incidences(edges):
    adjacent = defaultdict(list)
    for a, b in edges:
        adjacent[a].append(b)
        adjacent[b].append(a)
    return adjacent


def suppress_degree_two(edges, colors):
    edges = [(a, b) for a, b in edges if a != b]
    while True:
        selected = next(((v, ns) for v, ns in incidences(edges).items()
                         if v not in colors and len(ns) == 2), None)
        if selected is None:
            return edges
        v, (a, b) = selected
        edges = [(i, j) for i, j in edges if v != i and v != j]
        if a != b:
            edges.append((a, b))


def unit_graph(weighted, terminal_colors):
    labels = {v: j for j, v in enumerate(sorted({v for a,b,_ in weighted for v in (a,b)}))}
    degree = Counter()
    for a, b, w in weighted:
        if a != b:
            degree[a] += w
            degree[b] += w
    factor = 2 if any(d % 2 for v, d in degree.items() if v not in terminal_colors) else 1
    next_vertex = len(labels)
    colors = {}
    edges = []
    for a, b, w in weighted:
        if a == b:
            continue
        for _ in range(factor * w):
            endpoints = []
            for v in (a, b):
                if v in terminal_colors:
                    colors[next_vertex] = terminal_colors[v]
                    endpoints.append(next_vertex)
                    next_vertex += 1
                else:
                    endpoints.append(labels[v])
            edges.append(tuple(endpoints))
    return edges, colors, factor


def quartic_graph(edges, colors):
    ports = defaultdict(list)
    for i, edge in enumerate(edges):
        for side, v in enumerate(edge):
            if v not in colors:
                ports[v].append((i, side))
    next_vertex = 1 + max(v for edge in edges for v in edge)
    incidence_vertex = {}
    internal = []
    for v, incidences_at_v in ports.items():
        if len(incidences_at_v) % 2:
            raise ValueError('Odd bulk degree')
        if len(incidences_at_v) <= 4:
            incidence_vertex.update((key, v) for key in incidences_at_v)
            continue
        m = len(incidences_at_v) // 2
        crossings = {}
        for i, j in combinations(range(m), 2):
            crossings[i, j] = crossings[j, i] = next_vertex
            next_vertex += 1
        for i in range(m):
            wire = [crossings[i, j] for j in range(m) if j != i]
            incidence_vertex[incidences_at_v[2*i]] = wire[0]
            incidence_vertex[incidences_at_v[2*i+1]] = wire[-1]
            internal.extend(zip(wire, wire[1:]))
    result = [tuple(incidence_vertex.get((i, side), v) for side, v in enumerate(edge))
              for i, edge in enumerate(edges)]
    return suppress_degree_two(result + internal, colors)


def split(edges, colors, v, ns, pairing):
    i, j, k, l = pairing
    return suppress_degree_two([(a, b) for a, b in edges if v not in (a, b)]
                               + [(ns[i], ns[j]), (ns[k], ns[l])], colors)


def exhaustive_cuts(edges, colors, palette):
    # Direct enumeration over bulk spins, independent of the max-flow code.
    bulk = sorted(set(incidences(edges)) - colors.keys())
    keys = {v: palette.index(c) for v, c in colors.items()}
    keys.update((v, i + len(palette)) for i, v in enumerate(bulk))
    es = [(keys[a], keys[b]) for a, b in edges]
    choices = []
    for x in range(1 << (len(palette)-1)):
        best = len(edges) + 1
        opt = []
        for y in range(1 << len(bulk)):
            shore = x | (y << len(palette))
            cost = sum(((shore >> i) ^ (shore >> j)) & 1 for i, j in es)
            if cost < best:
                best, opt = cost, [shore]
            elif cost == best:
                opt.append(shore)
        choices.append((best, opt))
    return choices, keys


def profile(edges, colors, palette, find_witnesses):
    vertices = sorted(incidences(edges))
    index = {v: i for i, v in enumerate(vertices)}
    adjacent = incidences(edges)
    n = len(vertices)
    source, sink = n, n+1
    original = np.zeros((n+2, n+2), dtype=np.int64)
    for a, b in edges:
        original[index[a], index[b]] += 1
        original[index[b], index[a]] += 1
    pins = [(index[v], palette.index(color)) for v, color in colors.items()]
    bulk = {v: tuple(index[u] for u in ns) for v, ns in adjacent.items() if v not in colors}
    found = {v: {} for v in bulk}
    values = []
    for mask in range(1 << (len(palette)-1)):
        caps = original.copy()
        for i, color in pins:
            if mask >> color & 1:
                caps[source, i] = len(edges)+1
            else:
                caps[i, sink] = len(edges)+1
        answer = maximum_flow(csr_matrix(caps), source, sink)
        value = int(answer.flow_value)
        values.append(value)
        if not find_witnesses:
            continue
        arcs = [np.flatnonzero(row) for row in caps-answer.flow.toarray() > 0]
        cache = {}

        def closure(v):
            if v not in cache:
                reached = {v}
                queue = [v]
                for u in queue:
                    for t in arcs[u]:
                        if t not in reached:
                            reached.add(t)
                            queue.append(t)
                cache[v] = reached
            return cache[v]

        for v, ns in bulk.items():
            for number, (i, j, k, l) in enumerate(PAIRINGS):
                if number in found[v]:
                    continue
                for (a, b), (c, d) in (((i, j), (k, l)), ((k, l), (i, j))):
                    reached = closure(source) | closure(ns[a]) | closure(ns[b])
                    if any(t in reached for t in (sink, ns[c], ns[d])):
                        continue
                    shore = {vertices[u] for u in reached if u < n}
                    if any((v in shore) != bool(mask >> palette.index(color) & 1)
                           for v, color in colors.items()):
                        raise ValueError('Invalid witnessed boundary mask')
                    toggled = shore ^ {v}
                    if any(sum((a in cut) != (b in cut) for a, b in edges) != value
                           for cut in (shore, toggled)):
                        raise ValueError('Invalid optimum tie')
                    found[v][number] = (mask, sorted(shore))
                    break
    return values, found


def reduce_quartic(edges, colors, palette):
    before, witnesses = profile(edges, colors, palette, True)
    steps = 0
    while True:
        move = next(((v, p) for v in sorted(witnesses) for p in range(3)
                     if p not in witnesses[v]), None)
        if move is None:
            return edges, witnesses, steps
        v, p = move
        edges = split(edges, colors, v, incidences(edges)[v], PAIRINGS[p])
        current, witnesses = profile(edges, colors, palette, True)
        if current != before:
            raise ValueError('A split changed a coarse entropy')
        steps += 1


def components(edges):
    adjacent = incidences(edges)
    remaining = set(adjacent)
    while remaining:
        reached = {min(remaining)}
        queue = list(reached)
        for v in queue:
            for u in adjacent[v]:
                if u not in reached:
                    reached.add(u)
                    queue.append(u)
        yield [(a, b) for a, b in edges if a in reached], reached
        remaining -= reached


def independent_c1_certificate(edges, colors, witnesses, coarse_values):
    if len({tuple(sorted(edge)) for edge in edges}) != len(edges):
        raise ValueError('Parallel edge in irreducible graph')
    adjacent = incidences(edges)
    if any(len(ns) != (1 if v in colors else 4) for v, ns in adjacent.items()):
        raise ValueError('Incorrect terminal or bulk degree')
    if any(set(witnesses[v]) != {0, 1, 2} for v in adjacent if v not in colors):
        raise ValueError('Coarse weight rigidity is missing a tie')
    pivots = {}
    for v in adjacent.keys() - colors.keys():
        for mask, shore in witnesses[v].values():
            left = set(shore)
            right = left ^ {v}
            a = [int((u in left) != (w in left)) for u, w in edges]
            b = [int((u in right) != (w in right)) for u, w in edges]
            if sum(a) != coarse_values[mask] or sum(b) != coarse_values[mask]:
                raise ValueError('Stored tie fails on the selected component')
            insert(pivots, {i: (x-y) % PRIME for i, (x,y) in enumerate(zip(a, b)) if x != y}, 0)
    if len(pivots) != len(edges)-1:
        raise ValueError('Insufficient exact coarse tie rank')
    n = len(colors)
    if n <= 14 and len(adjacent)-n <= 10:
        relabel = {v: i for i, v in enumerate(colors)}
        half, _ = exhaustive_cuts(edges, relabel, list(range(n)))
        values = [s for s, _ in half]
        values += values[::-1]
        dim = (len(values)-2)//2
        tight = tight_sa_echelon(tight_pairs(values), len(values)-1)
        if len(tight) != dim-1 or min(values[1:-1]) <= 0:
            raise ValueError('Independent direct fine PMI certificate failed')
        return f'coarse tie rank {len(pivots)}/{len(edges)}, direct fine PMI rank {len(tight)}/{dim}'
    return f'coarse tie rank {len(pivots)}/{len(edges)}, fine PMI by even-degree theorem'


def verify_input(name, weighted, labels):
    palette = sorted(set(labels.values()))
    unit, colors, factor = unit_graph(weighted, labels)
    coarse, _ = profile(unit, colors, palette, False)
    base = [(a, b) for a, b, w in weighted if a != b for _ in range(w)]
    base, ids = compress_names(base)
    original, _ = exhaustive_cuts(base, {ids[v]: c for v, c in labels.items()}, palette)
    if [factor*s for s, _ in original] != coarse:
        raise ValueError('Unitization changed a coarse entropy')
    quartic = quartic_graph(unit, colors)
    quartic_values, _ = profile(quartic, colors, palette, False)
    if coarse != quartic_values:
        raise ValueError('Four-valent replacement changed a coarse entropy')
    final, witnesses, steps = reduce_quartic(quartic, colors, palette)
    for part, vertices in components(final):
        pcolors = {v: c for v, c in colors.items() if v in vertices}
        values, _ = profile(part, pcolors, palette, False)
        if not any(values):
            continue
        direct, _ = exhaustive_cuts(part, pcolors, palette)
        if [s for s, _ in direct] != values:
            raise ValueError('Direct cuts disagree on the selected component')
        if any(a*coarse[-1] != b*values[-1] for a, b in zip(values, coarse, strict=True)):
            raise ValueError('Component is not on the stated extreme ray')
        detail = independent_c1_certificate(part, pcolors, witnesses, values)
        bulk = len(vertices)-len(pcolors)
        print(f'{name}: {steps} splits, {len(pcolors)} fine leaves, {bulk} bulk, {len(part)} edges; {detail}', flush=True)
        return
    raise ValueError('No nonzero component')


def compress_names(edges):
    ids = {v: i for i, v in enumerate(sorted({v for a, b in edges for v in (a, b)}))}
    return [(ids[a], ids[b]) for a, b in edges], ids


def check_small_operations():
    rng = Random(6249)
    trials = safe = unsafe = 0
    for n in range(1, 6):
        for _ in range(90):
            stubs = [v for v in range(n) for _ in range(4)]
            rng.shuffle(stubs)
            core_size = rng.randrange(2*n)
            core = [(a, b) for a, b in zip(stubs[:2*core_size:2], stubs[1:2*core_size:2]) if a != b]
            edges = core + [(v, n+j) for j, v in enumerate(stubs[2*core_size:])]
            colors = {v: rng.randrange(4) for v in range(n, n+len(stubs[2*core_size:]))}
            edges = suppress_degree_two(edges, colors)
            palette = list(range(4))
            before, keys = exhaustive_cuts(edges, colors, palette)
            maxflow_values, witnesses = profile(edges, colors, palette, True)
            if maxflow_values != [v for v, _ in before]:
                raise ValueError('Direct and flow cuts differ')
            for v, ns in incidences(edges).items():
                if v in colors:
                    continue
                for p, (i,j,k,l) in enumerate(PAIRINGS):
                    def present(shore):
                        a, b, c, d = ((shore >> keys[ns[t]]) & 1 for t in (i,j,k,l))
                        return a == b and c == d and a != c
                    occurs = any(present(x) for _, opts in before for x in opts)
                    if occurs != (p in witnesses[v]):
                        raise ValueError('Residual and direct optimum witnesses differ')
                    after, _ = exhaustive_cuts(split(edges, colors, v, ns, PAIRINGS[p]), colors, palette)
                    changed = [a-b for (a,_), (b,_) in zip(before, after, strict=True)]
                    if any(d not in (0,2) for d in changed) or occurs != any(changed):
                        raise ValueError('Split criterion failed')
                    trials += 1
                    safe += not occurs
                    unsafe += occurs
    print(f'Direct local checks: {trials} operations, {safe} preserve all cuts, {unsafe} lower at least one minimum by two', flush=True)
    edges = [(0,2), (0,3), (0,4), (0,1), (1,5), (1,6)]
    colors = {2: 0, 3: 0, 4: 1, 5: 0, 6: 1}
    before, keys = exhaustive_cuts(edges, colors, [0, 1])
    if any(((shore >> keys[1]) & 1) == 0 for shore in before[1][1]):
        raise ValueError('Odd-degree control unexpectedly has the critical partition')
    after, _ = exhaustive_cuts([(2,3), (4,1), (1,5), (1,6)], colors, [0, 1])
    if (before[1][0], after[1][0]) != (2, 1):
        raise ValueError('Odd-degree control unexpectedly preserves the cut value')
    print('Odd-degree control: missing balanced partition, but minimum drops from 2 to 1', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--small-only', action='store_true')
    parser.add_argument('--catalog', type=Path, help='path to the optional IRSI graph catalogue JSON')
    args = parser.parse_args()
    check_small_operations()
    if args.small_only:
        return
    bundled_path = Path('work/c3_audit/irsi_catalog_graphs.json')
    catalogue = args.catalog
    if catalogue is None and bundled_path.is_file():
        catalogue = bundled_path
    if catalogue is not None:
        data = json.loads(catalogue.read_text(encoding='utf-8'))
        for i in (5, 55, 294, 532):
            g = data[i-1]
            weighted = [(a, b, w) for (a, b), w in zip(g['edges'], g['weights'], strict=True)]
            colors = {v: v for a, b, _ in weighted for v in (a, b) if v in 'ABCDEFO'}
            verify_input(f'Catalogue {i}', weighted, colors)
    fletcher = Path('fletcher_graphs.json')
    if not fletcher.is_file():
        fletcher = Path('work/c1_c2/replay/fletcher_graphs.json')
    data = json.loads(fletcher.read_text(encoding='utf-8'))
    for i, g in data['rays'].items():
        verify_input(f'Fletcher {i}', g['edges'], {v: v for v in data['boundary_vertices']})


if __name__ == '__main__':
    main()
