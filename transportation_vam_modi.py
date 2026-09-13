# Transportation Problem
# Initial BFS by Vogel's Approximation Method (VAM)
# Optimality test and improvement by MODI method

import numpy as np

def vogel_approximation(cost, supply, demand):
    cost = np.array(cost, dtype=float)
    supply = supply.copy()
    demand = demand.copy()

    m, n = cost.shape
    allocation = np.zeros((m, n))

    active_rows = [True] * m
    active_cols = [True] * n

    while any(s > 0 for s in supply):
        row_penalty = []

        for i in range(m):
            values = [
                cost[i, j]
                for j in range(n)
                if active_rows[i] and active_cols[j]
            ]
            values.sort()

            if len(values) >= 2:
                row_penalty.append(values[1] - values[0])
            elif len(values) == 1:
                row_penalty.append(values[0])
            else:
                row_penalty.append(-1)

        col_penalty = []

        for j in range(n):
            values = [
                cost[i, j]
                for i in range(m)
                if active_rows[i] and active_cols[j]
            ]
            values.sort()

            if len(values) >= 2:
                col_penalty.append(values[1] - values[0])
            elif len(values) == 1:
                col_penalty.append(values[0])
            else:
                col_penalty.append(-1)

        maximum_penalty = max(row_penalty + col_penalty)

        candidates = []

        for i, p in enumerate(row_penalty):
            if p == maximum_penalty:
                min_cost = min(
                    cost[i, j]
                    for j in range(n)
                    if active_cols[j]
                )
                candidates.append((min_cost, "row", i))

        for j, p in enumerate(col_penalty):
            if p == maximum_penalty:
                min_cost = min(
                    cost[i, j]
                    for i in range(m)
                    if active_rows[i]
                )
                candidates.append((min_cost, "col", j))

        _, choice_type, index = min(candidates)

        if choice_type == "row":
            i = index
            j = min(
                (j for j in range(n) if active_cols[j]),
                key=lambda j: cost[i, j]
            )
        else:
            j = index
            i = min(
                (i for i in range(m) if active_rows[i]),
                key=lambda i: cost[i, j]
            )

        quantity = min(supply[i], demand[j])
        allocation[i, j] += quantity

        supply[i] -= quantity
        demand[j] -= quantity

        if supply[i] == 0:
            active_rows[i] = False
        if demand[j] == 0:
            active_cols[j] = False

    return allocation


def find_cycle(basis, start, m, n):
    """Find a closed row-column cycle for the MODI improvement."""

    def dfs(path):
        current = path[-1]

        if len(path) >= 4 and current == start:
            return path

        for cell in basis:
            if cell == current:
                continue

            same_row = cell[0] == current[0]
            same_col = cell[1] == current[1]

            if not (same_row or same_col):
                continue

            if cell != start and cell in path:
                continue

            # Row/column direction must alternate.
            if len(path) >= 2:
                previous = path[-2]
                if same_row and previous[0] == current[0]:
                    continue
                if same_col and previous[1] == current[1]:
                    continue

            result = dfs(path + [cell])
            if result is not None:
                return result

        return None

    return dfs([start])


def modi(cost, allocation):
    cost = np.array(cost, dtype=float)
    allocation = allocation.copy()

    m, n = cost.shape

    basis = {
        (i, j)
        for i in range(m)
        for j in range(n)
        if allocation[i, j] > 1e-9
    }

    history = []

    while True:
        # Find u and v from c_ij = u_i + v_j for occupied cells.
        u = [None] * m
        v = [None] * n
        u[0] = 0

        changed = True
        while changed:
            changed = False

            for i, j in basis:
                if u[i] is not None and v[j] is None:
                    v[j] = cost[i, j] - u[i]
                    changed = True

                elif v[j] is not None and u[i] is None:
                    u[i] = cost[i, j] - v[j]
                    changed = True

        delta = np.full((m, n), np.nan)

        # Delta = cost - (u + v)
        # For minimization, all deltas >= 0 means optimal.
        for i in range(m):
            for j in range(n):
                if (i, j) not in basis:
                    delta[i, j] = cost[i, j] - u[i] - v[j]

        history.append(
            (allocation.copy(), u.copy(), v.copy(), delta.copy())
        )

        negative_cells = np.argwhere(delta < -1e-9)

        if len(negative_cells) == 0:
            break

        # Choose the most negative delta as entering cell.
        entering = min(
            negative_cells,
            key=lambda x: delta[x[0], x[1]]
        )
        entering = (int(entering[0]), int(entering[1]))

        basis.add(entering)

        cycle = find_cycle(basis, entering, m, n)

        if cycle is None:
            raise ValueError("Unable to form a MODI cycle.")

        # + - + - ... around the cycle
        minus_cells = cycle[1:-1:2]
        theta = min(allocation[cell] for cell in minus_cells)

        for k, cell in enumerate(cycle[:-1]):
            if k % 2 == 0:
                allocation[cell] += theta
            else:
                allocation[cell] -= theta

        # Remove one zero allocation from the negative positions.
        for cell in minus_cells:
            if abs(allocation[cell]) < 1e-9:
                basis.remove(cell)
                break

    return allocation, history


# ---------------- CASE STUDY ----------------
# Sources: S1, S2, S3
# Destinations: D1, D2, D3, D4

cost = np.array([
    [1,  4, 15, 6],
    [7, 20, 18, 2],
    [10, 6,  1, 1]
], dtype=float)

supply = [20, 30, 25]
demand = [10, 25, 15, 25]

initial = vogel_approximation(cost, supply, demand)

optimal, history = modi(cost, initial)

print("\nTRANSPORTATION PROBLEM")
print("-" * 40)

print("\nCost matrix:")
print(cost.astype(int))

print("\nInitial BFS by VAM:")
print(initial.astype(int))
print("Initial transportation cost =", int(np.sum(initial * cost)))

for k, (alloc, u, v, delta) in enumerate(history):
    print(f"\nMODI iteration {k}")
    print("Allocation:")
    print(alloc.astype(int))
    print("u =", [round(x, 2) for x in u])
    print("v =", [round(x, 2) for x in v])
    print("Delta (c_ij - u_i - v_j):")
    print(delta)

print("\nOptimal allocation:")
print(optimal.astype(int))

print("\nMinimum total transportation cost =",
      int(np.sum(optimal * cost)))
