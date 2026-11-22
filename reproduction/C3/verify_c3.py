"""Integer/rational verifier for the C3 counterexample. Python 3.10+, no solver.

Run `python verify_c3.py` with verify_ray110.py and c3_certificate.json beside it.
The analytic link between the checks and the conjecture is in C3_PROOF.md.
"""

from fractions import Fraction
from itertools import combinations, combinations_with_replacement
from pathlib import Path
import json

from verify_ray110 import (
    COORDINATES, EDGE_TYPES, FULL, LABELS, VALUES, check, check_graph, extend,
    rank_mod_prime, tight_subadditivity_rows,
)


MASKS = sorted(range(1, FULL), key=lambda mask: (mask.bit_count(), mask))
ALL_REACH = (1 << (FULL - 1)) - 1


def tight_pairs(S):
    submod, posimod = [], []
    for a, b in combinations(range(1, FULL), 2):
        if S[a] + S[b] == S[a | b] + S[a & b] and (a & b) not in (a, b):
            submod.append((a, b))
        if S[a] + S[b] == S[a & ~b] + S[b & ~a] and a & b:
            posimod.append((a, b))
    return submod, posimod


def signature_types(submod):
    # In cardinality order, every proper-subset dependency is already assigned.
    unions = {mask: [] for mask in [*MASKS, FULL]}
    for a, b in submod:
        unions[a | b].append((1 << a) | (1 << b))
    below = {mask: sum(1 << (mask ^ (1 << i)) for i in range(len(LABELS))
                       if (mask >> i) & 1) for mask in MASKS}

    def enumerate_bits(k, signature):
        if k == len(MASKS):
            if all(signature & pair for pair in unions[FULL]):
                yield signature | (1 << FULL)
            return
        m = MASKS[k]
        if not (signature & below[m]):
            yield from enumerate_bits(k + 1, signature)
        if not (signature >> (FULL ^ m) & 1) and all(signature & pair for pair in unions[m]):
            yield from enumerate_bits(k + 1, signature | (1 << m))

    terminals = {label: sum(1 << m for m in range(FULL + 1) if (m >> i) & 1)
                 for i, label in enumerate(LABELS)}
    values = set(terminals.values())
    all_signatures = list(enumerate_bits(0, 0))
    kept = [s for s in all_signatures if s in values or all(s & ~t for t in values)]
    kept.sort(key=lambda s: tuple(m for m in range(FULL + 1) if (s >> m) & 1))
    terminal_ids = {name: kept.index(s) for name, s in terminals.items()}
    print(f'PASS signature enumeration: {len(all_signatures)} before pinning, {len(kept)} after')
    return kept, terminal_ids


def edge_candidates(signatures, submod, posimod):
    tests = [(a, b, True) for a, b in submod] + [(a, b, False) for a, b in posimod]
    bits = [[(s >> m) & 1 for m in range(FULL + 1)] for s in signatures]
    reflected = [sum(1 << (FULL ^ m) for m, b in enumerate(row) if b) for row in bits]
    neighbors = [set() for _ in signatures]
    for i, j in combinations(range(len(signatures)), 2):
        if not (signatures[i] & reflected[j] or signatures[j] & reflected[i]):
            continue
        p, q = bits[i], bits[j]
        for a, b, use_union in tests:
            lhs = (p[a] ^ q[a]) + (p[b] ^ q[b])
            if use_union:
                rhs = ((p[a] | p[b]) ^ (q[a] | q[b])) + ((p[a] & p[b]) ^ (q[a] & q[b]))
            else:
                rhs = ((p[a] & (1 - p[b])) ^ (q[a] & (1 - q[b])))
                rhs += (((1 - p[a]) & p[b]) ^ ((1 - q[a]) & q[b]))
            if lhs != rhs:
                break
        else:
            neighbors[i].add(j)
            neighbors[j].add(i)
    return neighbors


def prove_reach_closure(data, signatures, neighbors, terminal_ids):
    check(len(data['reach']) == len(signatures), 'Wrong number of reach records')
    reach = [{j: int(v, 16) for j, v in records} for records in data['reach']]
    terminals = set(terminal_ids.values())
    for i, record in enumerate(reach):
        check(len(record) == len(data['reach'][i]) and set(record) <= neighbors[i],
              'Invalid or duplicate reach direction')
        check(all(0 <= r <= ALL_REACH for r in record.values()), 'Invalid reach bitmask')
        if i in terminals:
            check(record == {j: ALL_REACH for j in neighbors[i]}, 'Terminal base case missing')

    omitted = {(i, j): terms for i, j, terms in data['local_witnesses']}
    check(len(omitted) == len(data['local_witnesses']), 'Duplicate local certificate')
    used = set()
    mismatches = 0
    for i, options in enumerate(neighbors):
        if i in terminals:
            continue
        # Vertex signatures use bit m. Reach records use bit m-1.
        mismatch = {j: ((signatures[i] ^ signatures[j]) >> 1) & ALL_REACH for j in options}
        children = {k: reach[k][i] & ~mismatch[k] for k in options if i in reach[k]}
        upper = 0
        for supply in children.values():
            upper |= supply
        for parent in options:
            if parent in reach[i]:
                check(upper & ~reach[i][parent] == 0, 'Reach upper bound not closed')
                continue
            if mismatch[parent] & ~upper:
                mismatches += 1
                continue
            key = (i, parent)
            check(key in omitted, f'Missing direction certificate {key}')
            used.add(key)
            witness = [(m, Fraction(n, d)) for m, n, d in omitted[key]]
            check(all(0 <= m < FULL and weight > 0 for m, weight in witness), 'Invalid multiplier')
            check(any(m == 0 or (signatures[i] >> m) & 1 for m, _ in witness), 'No strict row')
            for k in set(children) | {parent}:
                supply = (ALL_REACH ^ mismatch[k]) if k == parent else children[k]
                total = Fraction(0)
                for m, weight in witness:
                    entry = int(k == parent) if m == 0 else ((supply >> (m - 1)) & 1) - (
                        (mismatch[k] >> (m - 1)) & 1)
                    total += weight * entry
                check(total <= 0, f'Invalid local dual at {(i, parent, k)}')
    check(used == set(omitted), 'Unused local witness')
    live = {(i, j) for i, record in enumerate(reach) for j in record if i < j and i in reach[j]}
    print(f'PASS reach closure: {sum(map(len, neighbors)) // 2} allowed edge types, '
          f'{mismatches} reach exclusions, {len(used)} rational exclusions, {len(live)} surviving')
    return live


def replay_reach_without_witnesses(signatures, neighbors, terminal_ids):
    terminals = set(terminal_ids.values())
    reach = [{j: ALL_REACH for j in adj} if i in terminals else {}
             for i, adj in enumerate(neighbors)]
    while True:
        new = [row.copy() for row in reach]
        for i, adj in enumerate(neighbors):
            if i in terminals:
                continue
            mismatch = {j: ((signatures[i] ^ signatures[j]) >> 1) & ALL_REACH for j in adj}
            child = {j: reach[j][i] & ~mismatch[j] for j in adj if i in reach[j]}
            upper = 0
            for r in child.values():
                upper |= r
            for parent in adj:
                if mismatch[parent] & ~upper or reach[i].get(parent) == upper:
                    continue
                columns = [parent, *sorted(set(child) - {parent})]
                supply = [ALL_REACH ^ mismatch[parent], *(child[j] for j in columns[1:])]
                rows = {}
                for m in range(1, FULL):
                    coeffs = tuple(((s >> (m-1)) & 1) - ((mismatch[j] >> (m-1)) & 1)
                                   for j, s in zip(columns, supply))
                    rows[coeffs] = rows.get(coeffs, 0) | ((signatures[i] >> m) & 1)
                # Two rows may contradict strict positivity, or the parent column
                # may need +2 to expose the strictly positive parent capacity.
                forbidden = any((sa or sb or a[0] + b[0] == -2)
                                and all(x + y <= 0 for x, y in zip(a, b))
                                for (a, sa), (b, sb) in combinations_with_replacement(rows.items(), 2))
                if not forbidden:
                    new[i][parent] = upper
        if new == reach:
            break
        check(all(all(k in b and v & ~b[k] == 0 for k, v in a.items())
                  for a, b in zip(reach, new)), 'Non-monotone reach replay')
        reach = new
    return {(i, j) for i, row in enumerate(reach) for j in row if i < j and i in reach[j]}


def swap_ef(mask):
    return mask ^ 48 if ((mask >> 4) ^ (mask >> 5)) & 1 else mask


def named_profiles(S, reference, signatures, terminal_ids):
    check(all(S[m] == S[swap_ef(m)] for m in range(FULL + 1)), 'E/F symmetry is missing')
    output = {name: sum(b << m for m, b in enumerate(p)) for name, p in reference.items()}
    output.update({f'Y{k}': sum(reference[f'X{k}'][swap_ef(m)] << m
                                for m in range(FULL + 1)) for k in range(1, 6)})
    ids = {name: signatures.index(s) for name, s in output.items()}
    check(len(set(ids.values())) == len(ids) and all(ids[v] == i for v, i in terminal_ids.items()),
          'Named profiles ambiguous or terminal numbering changed')
    return ids, output


def prove_aggregate(data, S, live, ids, profiles):
    layer_x = {tuple(sorted(e)) for e in EDGE_TYPES}
    recolor = {'E': 'F', 'F': 'E', **{f'X{k}': f'Y{k}' for k in range(1, 6)}}
    layer_y = {tuple(sorted(recolor.get(v, v) for v in e)) for e in EDGE_TYPES}
    edges = sorted(layer_x | layer_y)
    check(edges == [tuple(e) for e in data['edges']], 'Aggregate edge ordering changed')
    check(live == {tuple(sorted((ids[a], ids[b]))) for a, b in edges}, 'Wrong surviving topology')
    bulk = sorted(set(profiles) - set(LABELS))
    check(data['bulk'] == bulk, 'Bulk group indexing changed')
    crossed = [profiles[a] ^ profiles[b] for a, b in edges]

    def incidence(mask):
        return [(s >> mask) & 1 for s in crossed]

    def flip(mask, code):
        check(0 < mask < FULL and 0 < code < (1 << len(bulk)), 'Invalid flip move')
        group = {v for k, v in enumerate(bulk) if (code >> k) & 1}
        return [int((a in group) != (b in group)) * (1 - 2 * ((s >> mask) & 1))
                for (a, b), s in zip(edges, crossed, strict=True)]

    basis = [flip(m, g) for m, g in data['basis']]
    layer_weights = [[int(e in layer) for e in edges] for layer in (layer_x, layer_y)]
    for w in layer_weights:
        check(all(sum(a*b for a, b in zip(incidence(m), w)) == S[m] for m in COORDINATES),
              'A reference layer does not satisfy the target equations')
        check(all(sum(a*b for a, b in zip(row, w)) == 0 for row in basis),
              'A reference layer does not satisfy the proposed equality rows')

    combined = [Fraction(0)] * len(edges)
    constant = Fraction(0)
    for m, n, d in data['equal_dual']:
        check(m in COORDINATES, 'Invalid equality row in aggregate certificate')
        z = Fraction(n, d)
        constant += z * S[m]
        combined = [a + z*b for a, b in zip(combined, incidence(m))]
    for m, g, n, d in data['flip_dual']:
        y = Fraction(n, d)
        check(y > 0, 'Negative inequality multiplier in aggregate certificate')
        combined = [a + y*b for a, b in zip(combined, flip(m, g))]
    check(constant == 0 and combined == [-sum(col) for col in zip(*basis)],
          'Aggregate Farkas identity does not hold')
    check(rank_mod_prime([incidence(m) for m in COORDINATES] + basis, 1009) == len(edges) - 1,
          'Aggregate capacities have more than one free dimension')
    print('PASS aggregate certificate: rank 33, affine line between the two layers')
    return edges, layer_x, layer_y, crossed


def prove_degree_obstruction(edges, layer_x, layer_y, crossed):
    def flip_rows(group, layer):
        cols = [j for j, (a, b) in enumerate(edges) if edges[j] in layer and (a in group) != (b in group)]
        rows = [[1 - 2*((crossed[j] >> m) & 1) for j in cols] for m in range(1, FULL)]
        return cols, [row for row in rows if sum(row) == 0]

    for prefix, layer in (('X', layer_x), ('Y', layer_y)):
        group = {f'{prefix}{k}' for k in range(1, 6)}
        for k in (1, 2, 4, 5):
            cols, tied = flip_rows({f'{prefix}{k}'}, layer)
            check(len(cols) == 4 and rank_mod_prime(tied, 1009) == 3, 'Vertex balance failed')
        cols, tied = flip_rows(group, layer)
        variables = [f'{prefix}{k}' for k in (1, 2, 4, 5)]
        check(all(f'{prefix}3' not in edges[j] for j in cols), 'Central node crosses group boundary')
        compressed = [[sum(a for a, j in zip(row, cols) if v in edges[j]) for v in variables]
                      for row in tied]
        check(rank_mod_prime(compressed, 1009) == 3, 'Connected-component balance failed')
    a, b = edges.index(('A', 'X0')), edges.index(('B', 'X0'))
    bulk_neighbors = [j for j, e in enumerate(edges) if 'X0' in e and j not in (a, b)]
    check(len(bulk_neighbors) == 4, 'Unexpected number of bulk neighbors of X0')
    for j in bulk_neighbors:
        signs = {tuple(1-2*((crossed[k] >> m) & 1) for k in (a, b, j)) for m in range(1, FULL)}
        check({(1, -1, -1), (-1, 1, -1)} <= signs, 'X0 leaf obstruction failed')
    print('PASS degree obstruction: every vertex in the nonempty contracted forest has degree >= 2')


def main():
    data = json.loads(Path(__file__).with_name('c3_certificate.json').read_text(encoding='utf-8'))
    S = extend(VALUES)
    ref = check_graph(S, [(a, b, 1) for a, b in EDGE_TYPES])
    check(rank_mod_prime(tight_subadditivity_rows(S), 1009) == 62, 'Ray not verified extreme')
    print('PASS graph realizability and extremality')
    submod, posimod = tight_pairs(S)
    signatures, terminal_ids = signature_types(submod)
    neighbors = edge_candidates(signatures, submod, posimod)
    live = prove_reach_closure(data, signatures, neighbors, terminal_ids)
    check(replay_reach_without_witnesses(signatures, neighbors, terminal_ids) == live,
          'Independent integer replay returned different edge types')
    print('PASS independent integer replay without the 176 witnesses')
    ids, profiles = named_profiles(S, ref, signatures, terminal_ids)
    edges, layer_x, layer_y, crossed = prove_aggregate(data['aggregate'], S, live, ids, profiles)
    prove_degree_obstruction(edges, layer_x, layer_y, crossed)


if __name__ == '__main__':
    main()
