"""Verify six-party ray 110 and the restricted-signature obstruction.

Run with Python 3.10+ and the standard library. The restricted check is
supplemented by the all-size certificate in verify_c3.py and C3_PROOF.md.
"""

from collections import deque
from itertools import combinations


LABELS = 'ABCDEFO'
FULL = (1 << len(LABELS)) - 1
COORDINATES = [sum(1 << k for k in inds) for size in range(1, 7)
               for inds in combinations(range(6), size)]
VALUES = [1, 1, 2, 2, 2, 2,
          2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
          4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 6, 6,
          6, 4, 4, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 4,
          4, 4, 4, 4, 3, 3,
          2]
# The graph is Fletcher's archived witness, DOI 10.5281/zenodo.22020364.
EDGE_TYPES = [('A', 'X0'), ('B', 'X0'), ('C', 'X1'), ('C', 'X4'),
              ('D', 'X4'), ('D', 'X5'), ('E', 'X2'), ('E', 'X4'),
              ('F', 'X1'), ('F', 'X5'), ('O', 'X2'), ('O', 'X5'),
              ('X0', 'X1'), ('X0', 'X2'), ('X1', 'X3'), ('X2', 'X3'),
              ('X3', 'X4'), ('X3', 'X5')]
BULK = [f'X{k}' for k in range(6)]


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def extend(values):
    check(len(values) == len(COORDINATES), 'Wrong number of entropy coordinates')
    S = [0] * (FULL + 1)
    for mask, value in zip(COORDINATES, values, strict=True):
        check(type(value) is int and value > 0, 'Entropy must be a positive integer')
        S[mask] = S[FULL ^ mask] = value
    return S


def check_graph(S, edges):
    check(all(type(w) is int and w > 0 and a != b for a, b, w in edges),
          'Graph has an invalid capacity or loop')
    vertices = set(LABELS) | {v for a, b, _ in edges for v in (a, b)}
    bulk = sorted(vertices - set(LABELS))
    names = {v: i for i, v in enumerate([*LABELS, *bulk])}
    integer_edges = [(names[a], names[b], w) for a, b, w in edges]

    profiles = {v: [] for v in names}
    for mask in range(FULL + 1):
        cuts = (mask | (assign << len(LABELS)) for assign in range(1 << len(bulk)))
        minimum = 1 + sum(w for _, _, w in integer_edges)
        minimal_cut = 0
        for cut in cuts:
            cost = sum(w for a, b, w in integer_edges if ((cut >> a) ^ (cut >> b)) & 1)
            if cost < minimum:
                minimum, minimal_cut = cost, cut
            elif cost == minimum:
                minimal_cut &= cut
        check(minimum == S[mask], f'Graph enumeration disagrees at {mask}')
        check(graph_max_flow(integer_edges, len(names), mask) == S[mask],
              f'Integral max-flow disagrees at {mask}')
        check(sum(w for a, b, w in integer_edges if ((minimal_cut >> a) ^ (minimal_cut >> b)) & 1)
              == minimum, 'Intersection of all optimal cuts failed the cost check')
        for v, index in names.items():
            profiles[v].append((minimal_cut >> index) & 1)
    return profiles


def graph_max_flow(edges, count, mask):
    source, sink = count, count + 1
    residual = [[0] * (count + 2) for _ in range(count + 2)]
    dominating = 1 + sum(w for _, _, w in edges)
    for a, b, w in edges:
        residual[a][b] += w
        residual[b][a] += w
    for terminal in range(len(LABELS)):
        if (mask >> terminal) & 1:
            residual[source][terminal] = dominating
        else:
            residual[terminal][sink] = dominating
    total = 0
    while True:
        prev = {source: source}
        queue = deque([source])
        while queue and sink not in prev:
            a = queue.popleft()
            for b, capacity in enumerate(residual[a]):
                if capacity and b not in prev:
                    prev[b] = a
                    queue.append(b)
        if sink not in prev:
            return total
        path = []
        current = sink
        while current != source:
            path.append((prev[current], current))
            current = prev[current]
        gain = min(residual[a][b] for a, b in path)
        for a, b in path:
            residual[a][b] -= gain
            residual[b][a] += gain
        total += gain


def tight_subadditivity_rows(S):
    column = {mask: j for j, mask in enumerate(COORDINATES)}
    rows = []
    for a, b in combinations(range(1, FULL), 2):
        if a & b or a | b == FULL or S[a] + S[b] != S[a | b]:
            continue
        row = [0] * len(COORDINATES)
        for mask, coefficient in ((a, 1), (b, 1), (a | b, -1)):
            row[column[mask if mask < 64 else FULL ^ mask]] += coefficient
        check(sum(row[j] * S[mask] for j, mask in enumerate(COORDINATES)) == 0,
              'Extremality row does not annihilate the target')
        rows.append(row)
    return rows


def rank_mod_prime(rows, prime):
    matrix = [[entry % prime for entry in row] for row in rows]
    pivots = 0
    for col in range(len(matrix[0])):
        chosen = next((r for r in range(pivots, len(matrix)) if matrix[r][col]), None)
        if chosen is None:
            continue
        matrix[pivots], matrix[chosen] = matrix[chosen], matrix[pivots]
        pivot = matrix[pivots]
        inverse = pow(pivot[col], -1, prime)
        for r in range(pivots + 1, len(matrix)):
            row = matrix[r]
            if row[col]:
                factor = row[col] * inverse % prime
                for j in range(col, len(row)):
                    row[j] = (row[j] - factor * pivot[j]) % prime
        pivots += 1
    return pivots


def allowed_on_reference_profiles(S, profiles):
    tight_submod, tight_posimod = [], []
    for a, b in combinations(range(FULL + 1), 2):
        if S[a] + S[b] == S[a | b] + S[a & b]:
            tight_submod.append((a, b))
        if S[a] + S[b] == S[a & ~b] + S[b & ~a]:
            tight_posimod.append((a, b))

    check(len({tuple(bits) for bits in profiles.values()}) == len(profiles),
          'Reference profiles are not distinct')
    allowed = set()
    for u, v in combinations(profiles, 2):
        p, q = profiles[u], profiles[v]
        submod_ok = all((p[a] ^ q[a]) + (p[b] ^ q[b])
                        == ((p[a] | p[b]) ^ (q[a] | q[b])) + ((p[a] & p[b]) ^ (q[a] & q[b]))
                        for a, b in tight_submod)
        posimod_ok = all((p[a] ^ q[a]) + (p[b] ^ q[b])
                         == ((p[a] & (1 - p[b])) ^ (q[a] & (1 - q[b])))
                         + (((1 - p[a]) & p[b]) ^ ((1 - q[a]) & q[b]))
                         for a, b in tight_posimod)
        if submod_ok and posimod_ok:
            allowed.add(frozenset((u, v)))
    check(allowed == {frozenset(edge) for edge in EDGE_TYPES}, 'Unexpected allowed edge types')


def flip_rows(profiles, group):
    return [tuple(1 - 2 * (profiles[u][mask] ^ profiles[v][mask])
                  if (u in group) != (v in group) else 0 for u, v in EDGE_TYPES)
            for mask in range(FULL + 1)]


def check_restricted_obstruction(S, profiles):
    allowed_on_reference_profiles(S, profiles)
    incidence = [[profiles[u][mask] ^ profiles[v][mask] for u, v in EDGE_TYPES]
                 for mask in COORDINATES]
    check(all(sum(row) == S[m] for row, m in zip(incidence, COORDINATES, strict=True)),
          'Unit edge capacities do not solve the aggregate equations')
    check(rank_mod_prime(incidence, 1009) == 15, 'Unexpected incidence rank')
    balanced = []
    for v in BULK:
        rows = set(flip_rows(profiles, {v}))
        balanced.extend(row for row in rows if tuple(-x for x in row) in rows)
    check(all(sum(row) == 0 for row in balanced), 'An opposite pair contradicts unit capacities')
    check(rank_mod_prime(incidence + balanced, 1009) == 18, 'Aggregate capacities not determined')

    for v in ('X0', 'X1', 'X2', 'X4', 'X5'):
        cols = [j for j, e in enumerate(EDGE_TYPES) if v in e]
        tied = [[row[j] for j in cols] for row in flip_rows(profiles, {v}) if sum(row) == 0]
        check(len(cols) == 4 and rank_mod_prime(tied, 1009) == 3,
              f'Incident capacities not balanced at {v}')

    group = set(BULK) - {'X0'}
    variables = ['X1', 'X2', 'X4', 'X5']
    check(all('X3' not in edge for edge in EDGE_TYPES if (edge[0] in group) != (edge[1] in group)),
          'Unexpected X3 edge crossing the group boundary')
    tied = [[sum(a for a, edge in zip(row, EDGE_TYPES, strict=True) if v in edge)
             for v in variables] for row in flip_rows(profiles, group) if sum(row) == 0]
    check(rank_mod_prime(tied, 1009) == 3, 'Component masses not balanced')
    print('PASS restricted obstruction: 13 profiles, 18 edge types, unique aggregate weights, '
          'vertex and component ranks 3')


def check_unrolled_family(S):
    name = {v: i for i, v in enumerate(BULK)}
    seam = ('X0', 'X1')
    base_count = len(BULK)
    for sheets in (1, 2, 3, 7):
        count = len(LABELS) + sheets * base_count
        bulk_edges, all_edges = [], []
        for sheet in range(sheets):
            def index(v):
                if v in LABELS:
                    return LABELS.index(v)
                return len(LABELS) + sheet * base_count + name[v]

            for u, v in EDGE_TYPES:
                if (u, v) == seam:
                    if sheet == sheets - 1:
                        continue
                    a, b = index(u), index(v) + base_count
                else:
                    a, b = index(u), index(v)
                all_edges.append((a, b, 1))
                if u in BULK and v in BULK:
                    bulk_edges.append((a, b))
        seen = {len(LABELS)}
        queue = deque(seen)
        while queue:
            a = queue.popleft()
            for u, v in bulk_edges:
                b = v if a == u else u if a == v else a
                if b not in seen:
                    seen.add(b)
                    queue.append(b)
        check(len(seen) == count - len(LABELS) and len(bulk_edges) == len(seen) - 1,
              'Unrolled bulk graph is not a tree')
        for mask in COORDINATES:
            value = graph_max_flow(all_edges, count, mask)
            check(sheets * S[mask] - 1 <= value <= sheets * S[mask],
                  f'Unrolling error bound failed for {sheets} sheets and mask {mask}')
    print('PASS cover bound: exact integer max-flow for 1, 2, 3 and 7 sheets')


def main():
    S = extend(VALUES)
    profiles = check_graph(S, [(u, v, 1) for u, v in EDGE_TYPES])
    print('PASS graph witness: exact cut enumeration and max-flow, all 128 masks')
    rows = tight_subadditivity_rows(S)
    check(rank_mod_prime(rows, 1009) == 62, 'Wrong tight SA rank')
    print(f'PASS extremality: {len(rows)} tight SA rows, rank 62 modulo 1009')
    check_restricted_obstruction(S, profiles)
    check_unrolled_family(S)


if __name__ == '__main__':
    main()
