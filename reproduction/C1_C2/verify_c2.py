"""Independent exact cut/PMI-rank checks for the C2 construction."""
from itertools import combinations
from pathlib import Path
import json


def graph_entropies(edges, leaf_count):
    size = max(max(u, v) for u, v, _ in edges) + 1
    return [min(sum(w for u, v, w in edges if ((mask >> u) ^ (mask >> v)) & 1)
                for mask in range(x, 1 << size, 1 << leaf_count))
            for x in range(1 << leaf_count)]


def equality_rank(entropies, prime=1009):
    all_terminals = len(entropies) - 1
    dimension = all_terminals // 2
    pivots = {}
    for x in range(1, all_terminals):
        free = all_terminals ^ x
        y = free
        while y:
            if x < y:
                deficit = entropies[x] + entropies[y] - entropies[x | y]
                if deficit < 0:
                    raise ValueError('Subadditivity failed')
                if deficit == 0:
                    row = {}
                    for coord, value in ((x, 1), (y, 1), (x | y, -1)):
                        coord = min(coord, all_terminals ^ coord)
                        if coord:
                            row[coord] = (row.get(coord, 0) + value) % prime
                            if not row[coord]:
                                del row[coord]
                    while row:
                        col = max(row)
                        if col not in pivots:
                            inverse = pow(row[col], -1, prime)
                            pivots[col] = {i: value * inverse % prime for i, value in row.items()}
                            break
                        multiplier = row[col]
                        for i, value in pivots[col].items():
                            value = (row.get(i, 0) - multiplier * value) % prime
                            if value:
                                row[i] = value
                            elif i in row:
                                del row[i]
                    if len(pivots) == dimension - 1:
                        return len(pivots), dimension
            y = (y - 1) & free
    return len(pivots), dimension


def split_leaves(coarse, leaf_count):
    colors = [u for u, v, w in coarse for _ in range(2*w) if u < leaf_count]
    fine = [(u-leaf_count+len(colors), v-leaf_count+len(colors), 2*w)
            for u, v, w in coarse if u >= leaf_count]
    index = 0
    for u, v, w in coarse:
        if u < leaf_count:
            for _ in range(2*w):
                fine.append((index, v-leaf_count+len(colors), 1))
                index += 1
    return fine, colors


def verify_example(name, coarse, colors_count):
    fine, colors = split_leaves(coarse, colors_count)
    base = graph_entropies(coarse, colors_count)
    entropies = graph_entropies(fine, len(colors))
    projection = [entropies[sum(1 << i for i, color in enumerate(colors) if (mask >> color) & 1)]
                  for mask in range(1 << colors_count)]
    if projection != [2*s for s in base]:
        raise ValueError(f'{name}: bad coarse projection')
    rank, dimension = equality_rank(entropies)
    if rank != dimension - 1:
        raise ValueError(f'{name}: rank {rank} in dimension {dimension}')
    print(f'{name}: {len(colors)} fine terminals; SA rank {rank}/{dimension}; exact coarse projection')


def verify_fletcher(source):
    catalog = json.loads(source.read_text())
    letters = catalog['boundary_vertices']
    coords = [sum(1 << i for i in subset)
              for k in range(1, len(letters)) for subset in combinations(range(len(letters)-1), k)]
    for ray in ('110', '145', '168'):
        r = catalog['rays'][ray]
        labels = {v: i for i, v in enumerate(letters + r['bulk_vertices'])}
        edges = [(labels[u], labels[v], w) for u, v, w in r['edges']]
        entropies = graph_entropies(edges, len(letters))
        if [entropies[i] for i in coords] != [x for group in r['target_groups'] for x in group]:
            raise ValueError(f'{ray}: mismatch with original entropy coordinates')
        rank, dimension = equality_rank(entropies)
        if rank != dimension-1:
            raise ValueError(f'{ray}: coarse ray is not SA extreme')
        print(f'Fletcher {ray}: graph reconstructed, rank {rank}/{dimension}')


if __name__ == '__main__':
    import sys

    verify_example('connected path, internal cuts used', [(0, 3, 2), (1, 4, 2), (2, 5, 2), (3, 4, 1), (4, 5, 1)], 3)
    verify_example('connected K4 core', [(i, i+4, 1) for i in range(4)] + [(u, v, 1) for u, v in combinations(range(4,8),2)], 4)
    verify_example('cycle with internal cuts', [(0, 3, 2), (1, 4, 2), (2, 5, 2), (3, 4, 1), (4, 5, 1), (3, 5, 1)], 3)
    verify_example('sparse terminals, one leafless bulk vertex', [(0, 3, 2), (1, 4, 2), (2, 5, 2), (3, 6, 1), (4, 6, 1), (5, 6, 1)], 3)
    eulerian = graph_entropies([(i, 6 + i // 3, 1) for i in range(6)] + [(6, 7, 1)], 6)
    rank, dimension = equality_rank(eulerian)
    if (rank, dimension) != (30, 31):
        raise ValueError('Eulerian unit graph rank check')
    print(f'even-degree network with odd bulk capacity: SA rank {rank}/{dimension}')
    odd_star = graph_entropies([(u, 5, 1) for u in range(5)], 5)
    rank, dimension = equality_rank(odd_star)
    if (rank, dimension) != (10, 15):
        raise ValueError('Odd star negative control')
    print(f'odd star negative control: SA rank {rank}/{dimension}')
    if len(sys.argv) == 2:
        verify_fletcher(Path(sys.argv[1]))
