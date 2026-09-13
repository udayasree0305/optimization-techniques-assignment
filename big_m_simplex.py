# Big-M Simplex Method
# Case study: Product-mix planning with a minimum production requirement

import numpy as np

def big_m_simplex(c, A, b, signs, M=100000):
    """
    Maximization form.
    signs contains '<=', '>=', or '=' for each constraint.
    Big-M is used for artificial variables.
    """

    m = len(b)
    n = len(c)

    names = [f"x{i+1}" for i in range(n)]
    extra_columns = []

    # Add slack/surplus/artificial variables
    for i, sign in enumerate(signs):
        if sign == "<=":
            col = np.zeros(m)
            col[i] = 1
            extra_columns.append(col)
            names.append(f"s{i+1}")

        elif sign == ">=":
            col = np.zeros(m)
            col[i] = -1
            extra_columns.append(col)
            names.append(f"s{i+1}")

            col = np.zeros(m)
            col[i] = 1
            extra_columns.append(col)
            names.append(f"a{i+1}")

        elif sign == "=":
            col = np.zeros(m)
            col[i] = 1
            extra_columns.append(col)
            names.append(f"a{i+1}")

        else:
            raise ValueError("Constraint sign must be <=, >=, or =")

    A_std = np.column_stack([np.array(A, dtype=float)] + extra_columns)

    # Objective coefficients:
    # artificial variables get -M in a maximization problem
    cj = []
    for name in names:
        if name.startswith("x"):
            cj.append(cj_original[int(name[1:]) - 1])
        elif name.startswith("a"):
            cj.append(-M)
        else:
            cj.append(0)

    cj = np.array(cj, dtype=float)

    # Initial basic variables
    basis = []
    for i, sign in enumerate(signs):
        if sign == "<=":
            basis.append(names.index(f"s{i+1}"))
        else:
            basis.append(names.index(f"a{i+1}"))

    tableau = np.column_stack([A_std, np.array(b, dtype=float)])

    # Convert the tableau to canonical form
    for r, col in enumerate(basis):
        pivot = tableau[r, col]
        tableau[r, :] /= pivot
        for rr in range(m):
            if rr != r:
                tableau[rr, :] -= tableau[rr, col] * tableau[r, :]

    iteration = 0

    while True:
        cb = cj[basis]
        zj = cb @ tableau[:, :-1]
        cj_minus_zj = cj - zj

        entering_candidates = [
            j for j in range(len(cj)) if cj_minus_zj[j] > 1e-9
        ]

        if not entering_candidates:
            break

        entering = max(
            entering_candidates,
            key=lambda j: cj_minus_zj[j]
        )

        ratios = []
        for r in range(m):
            if tableau[r, entering] > 1e-10:
                ratios.append(tableau[r, -1] / tableau[r, entering])
            else:
                ratios.append(np.inf)

        leaving = min(range(m), key=lambda r: ratios[r])

        if ratios[leaving] == np.inf:
            raise ValueError("The LPP is unbounded.")

        pivot = tableau[leaving, entering]
        tableau[leaving, :] /= pivot

        for r in range(m):
            if r != leaving:
                tableau[r, :] -= (
                    tableau[r, entering] * tableau[leaving, :]
                )

        basis[leaving] = entering
        iteration += 1

        if iteration > 100:
            raise ValueError("Too many iterations.")

    solution = np.zeros(len(cj))
    for r, col in enumerate(basis):
        solution[col] = tableau[r, -1]

    # Feasibility check for artificial variables
    for j, name in enumerate(names):
        if name.startswith("a") and solution[j] > 1e-7:
            raise ValueError("Problem is infeasible.")

    objective_value = np.dot(cj, solution)

    return names, basis, tableau, solution, objective_value, iteration


# ---------------- CASE STUDY ----------------
# x1 = quantity of Product A
# x2 = quantity of Product B
#
# Maximize Z = 40x1 + 30x2
#
# Subject to:
# x1 + x2 <= 40
# 2x1 + x2 >= 50
# x1 + 2x2 <= 60
# x1, x2 >= 0

cj_original = [40, 30]

A = [
    [1, 1],
    [2, 1],
    [1, 2]
]

b = [40, 50, 60]
signs = ["<=", ">=", "<="]

names, basis, tableau, solution, Z, iterations = big_m_simplex(
    cj_original, A, b, signs
)

print("\nBIG-M SIMPLEX METHOD")
print("-" * 40)

for name, value in zip(names, solution):
    print(f"{name:4s} = {value:.2f}")

print(f"\nOptimal objective value = {Z:.2f}")
print(f"Simplex iterations = {iterations}")

print("\nVerification:")
print(f"x1 + x2       = {solution[0] + solution[1]:.2f} <= 40")
print(f"2x1 + x2      = {2*solution[0] + solution[1]:.2f} >= 50")
print(f"x1 + 2x2      = {solution[0] + 2*solution[1]:.2f} <= 60")
