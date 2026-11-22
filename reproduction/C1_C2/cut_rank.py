"""Exact modular rank helpers for tight-cut and subadditivity checks."""

PRIME = 1009


def tight_pairs(values):
    full = len(values) - 1
    for a in range(1, full):
        free = full ^ a
        b = free
        while b > a:
            slack = values[a] + values[b] - values[a | b]
            if slack == 0:
                yield a, b
            elif slack < 0:
                raise ValueError('Subadditivity violated')
            b = (b - 1) & free


def insert(pivots, row, rhs):
    while row:
        column = max(row)
        coefficient = row[column]
        if column not in pivots:
            inverse = pow(coefficient, -1, PRIME)
            pivots[column] = ({i: x * inverse % PRIME for i, x in row.items()},
                              rhs * inverse % PRIME)
            return
        pivot, target = pivots[column]
        for i, x in pivot.items():
            value = (row.get(i, 0) - coefficient * x) % PRIME
            if value:
                row[i] = value
            else:
                row.pop(i, None)
        rhs = (rhs - coefficient * target) % PRIME
    if rhs:
        raise ValueError('Inconsistent modular system')


def tight_sa_echelon(pairs, full):
    pivots = {}
    for a, b in pairs:
        row = {}
        for mask, sign in ((a, 1), (b, 1), (a | b, -1)):
            column = min(mask, full ^ mask)
            if column:
                row[column] = (row.get(column, 0) + sign) % PRIME
        insert(pivots, {i: x for i, x in row.items() if x}, 0)
    return pivots
