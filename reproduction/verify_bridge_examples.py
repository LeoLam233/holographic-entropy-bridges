"""Check the manuscript's two-block and marked-port examples with integer cuts."""

from fractions import Fraction
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from C1_C2.cut_rank import insert, tight_pairs, tight_sa_echelon
from C3.verify_ray110 import EDGE_TYPES, LABELS, VALUES, check, extend


def minimum_cut_profile(edges, terminals):
    """Enumerate every bulk assignment and compare with a separate max flow."""
    bulk = sorted({v for a, b, _ in edges for v in (a, b)} - set(terminals))
    index = {v: i for i, v in enumerate((*terminals, *bulk))}
    left = np.array([index[a] for a, _, _ in edges], dtype=np.int64)
    right = np.array([index[b] for _, b, _ in edges], dtype=np.int64)
    weights = np.array([w for _, _, w in edges], dtype=np.int64)
    bulk_shores = np.arange(1 << len(bulk), dtype=np.int64) << len(terminals)
    source = len(index)
    sink = source + 1
    capacity = np.zeros((sink + 1, sink + 1), dtype=np.int64)
    for a, b, weight in zip(left, right, weights, strict=True):
        capacity[a, b] += weight
        capacity[b, a] += weight
    pin_capacity = int(weights.sum()) + 1

    values = []
    tie_rows = set()
    for mask in range(1 << len(terminals)):
        shores = (bulk_shores | mask)[:, None]
        incidence = ((shores >> left) ^ (shores >> right)) & 1
        costs = incidence @ weights
        best = int(costs.min())
        optimal_rows = incidence[costs == best]
        tie_rows.update(map(tuple, optimal_rows[1:] - optimal_rows[0]))

        pinned = capacity.copy()
        for i in range(len(terminals)):
            if mask & (1 << i):
                pinned[source, i] = pin_capacity
            else:
                pinned[i, sink] = pin_capacity
        result = maximum_flow(csr_matrix(pinned), source, sink)
        check(best == result.flow_value, f'Enumeration and max flow disagree at mask {mask}')
        values.append(best)
    return values, tie_rows


def rational_rank(rows):
    pivots = {}
    for row in rows:
        reduced = [Fraction(int(value)) for value in row]
        for column, pivot in pivots.items():
            coefficient = reduced[column]
            if coefficient:
                reduced = [a - coefficient * b for a, b in zip(reduced, pivot, strict=True)]
        for column, value in enumerate(reduced):
            if value:
                pivots[column] = [x / value for x in reduced]
                break
    return len(pivots)


def purity_row(width, terms):
    row = [0] * width
    for full, mask, offset, coefficient in terms:
        column = min(mask, full ^ mask)
        if column:
            row[offset + column - 1] += coefficient
    return row


def factor_rows(values, width, offset):
    full = len(values) - 1
    for a, b in tight_pairs(values):
        yield purity_row(width, [(full, a, offset, 1), (full, b, offset, 1),
                                (full, a | b, offset, -1)])


def verify_marked_port(seed_values):
    left = [('A', 'L', 1), ('B', 'L', 1), ('p', 'L', 2)]
    right = [(a, b, 1) for a, b in EDGE_TYPES if a not in ('A', 'B')]
    right.append(('q', 'X0', 2))
    splice = [edge for edge in (*left, *right) if edge[0] not in ('p', 'q')]
    splice.append(('L', 'X0', 2))

    s, s_ties = minimum_cut_profile(left, 'ABp')
    t, t_ties = minimum_cut_profile(right, 'CDEFOq')
    combined, combined_ties = minimum_cut_profile(splice, LABELS)
    check(combined == seed_values, 'Marked-port splice did not recover ray 110')
    cell_dims = tuple(len(edges) - rational_rank(rows) for edges, rows in
                      ((left, s_ties), (right, t_ties), (splice, combined_ties)))
    check(cell_dims == (2, 1, 1), f'Wrong marked-port weight-cell dimensions {cell_dims}')

    s_full, t_full = len(s) - 1, len(t) - 1
    s_dim, t_dim = s_full // 2, t_full // 2
    width = s_dim + t_dim
    s_rows = list(factor_rows(s, width, 0))
    t_rows = list(factor_rows(t, width, s_dim))
    dims = s_dim - rational_rank(s_rows), t_dim - rational_rank(t_rows)
    check(dims == (2, 5), f'Wrong marked-port PMI dimensions {dims}')
    s_port, t_port = len(s) // 2, len(t) // 2
    singleton_match = purity_row(width, [(s_full, s_port, 0, 1),
                                         (t_full, t_port, s_dim, -1)])
    matched = [*s_rows, *t_rows, singleton_match]
    matched_rank = rational_rank(matched)
    check(width - matched_rank == 6, 'Wrong dimension after matching the port entropy')

    tied_branches = []
    for x in range(s_port):
        for y in range(t_port):
            a, b = s[x] + t[y], s[x | s_port] + t[y | t_port]
            check(min(a, b) == combined[x | (y << 2)], 'Marked-port branch formula failed')
            if a == b:
                tied_branches.append(purity_row(width, [(s_full, x, 0, 1),
                    (s_full, x | s_port, 0, -1), (t_full, y, s_dim, 1),
                    (t_full, y | t_port, s_dim, -1)]))
    added_rank = rational_rank([*matched, *tied_branches]) - matched_rank
    check((len(tied_branches), added_rank) == (40, 5),
          f'Wrong tied branch count or rank: {len(tied_branches)}, {added_rank}')
    print('PASS marked port: PMI dimensions 2 and 5, 40 ties of rank 5 after matching, '
          'weight-cell dimensions 2, 1 and 1', flush=True)


def verify_two_blocks():
    left = [(f'L_{a}', f'L_{b}', 1) for a, b in EDGE_TYPES if a != 'B']
    right = [(f'R_{a}', f'R_{b}', 1) for a, b in EDGE_TYPES if a != 'A']
    terminals = [f'L_{c}' for c in LABELS if c != 'B']
    terminals += [f'R_{c}' for c in LABELS if c != 'A']
    edges = [*left, *right, ('L_X0', 'R_X0', 1)]
    values, tied_cuts = minimum_cut_profile(edges, terminals)
    check(len(values) == 4096 and min(values[1:-1]) > 0, 'Wrong two-block entropy profile')
    pivots = {}
    for row in tied_cuts:
        insert(pivots, {i: int(v) for i, v in enumerate(row) if v}, 0)
    sa_rank = len(tight_sa_echelon(tight_pairs(values), len(values) - 1))
    check((len(pivots), len(edges), sa_rank) == (34, 35, 2046), 'Wrong two-block ranks')
    print('PASS two blocks: 4,096 masks, tied-cut rank 34/35, tight-SA rank 2,046/2,047',
          flush=True)


def main():
    seed = extend(VALUES)
    for a, b in combinations(range(len(LABELS)), 2):
        x, y = 1 << a, 1 << b
        check(seed[x] + seed[y] == seed[x | y], 'Ray 110 has positive singleton mutual information')
    print('PASS ray 110: singleton mutual information vanishes for all 21 pairs', flush=True)
    verify_marked_port(seed)
    verify_two_blocks()


if __name__ == '__main__':
    main()
