import numpy as np
import numpy.typing as npt
from pyqubo import Array, Constraint
from exps.create_study import load_study
from utils import sample_integer_matrix, encode_one_hot
from surrogates import BayesianLinearRegressor
from itertools import combinations
from openjij import SASampler

low = 0
high = 3
λ = 100
n_vars = 4
range_vars = high - low + 1
study = load_study("miqp", f'{n_vars}.json')
Q = study['Q'][0]


def objective(X: npt.NDArray) -> npt.NDArray:
    return np.diag(X @ Q @ X.T)


X = sample_integer_matrix(10, low, high, n_vars)
y = objective(X)
X = encode_one_hot(low, high, n_vars, X)
blr = BayesianLinearRegressor(range_vars * n_vars, 2)
blr.fit(X, y)

# print(blr.to_qubo())
# print()
# print("=" * 50)

qubo_c = {}

for i in range(n_vars * range_vars):
    qubo_c[(i, i)] = -1 * λ
for i in range(n_vars):
    for c in list(combinations(list(range(i * range_vars, (i + 1) * range_vars)), 2)):
        qubo_c[(c[0], c[1])] = 2 * λ


# print(qubo_c)
# print(len(qubo_c))
# print()

QUBO = blr.to_qubo()
qubo_q = {}
for i in range(n_vars * range_vars):
    qubo_q[(i, i)] = QUBO[i, i]
for i in range(n_vars * range_vars):
    for j in range(i + 1, n_vars * range_vars):
        qubo_q[(i, j)] = 2 * QUBO[i, j]

# print(qubo_q)
# print()


qubo = {}
for i, j in qubo_q.keys():
    qubo[(i, j)] = -1 * qubo_q[(i, j)]
    if (i, j) in qubo_c.keys():
        qubo[(i, j)] += qubo_c[(i, j)]

# print(qubo)
# print(len(qubo))
# print()
# print("=" * 50)

x = Array.create('x', shape=(n_vars * range_vars, ), vartype='BINARY')
H_A = Constraint(sum(λ * (1 - sum(x[j * range_vars + i]
                                  for i in range(range_vars))) ** 2 for j in range(n_vars)), label='HA')
model = H_A.compile()
qubo, _ = model.to_qubo()

# print(qubo)
# print(len(qubo))
# print()

Q = Array(blr.to_qubo())
H_B = x @ Q @ x.T
model = H_B.compile()
qubo, _ = model.to_qubo()

# print(qubo)
# print()

H = H_A - H_B
model = H.compile()
qubo, _ = model.to_qubo()
# print(qubo)
# print(len(qubo))
# print()

Q = blr.to_qubo()
# ----- Constraint -----
constraint = {}
for i in range(n_vars * range_vars):
    constraint[(i, i)] = -1 * λ
for i in range(n_vars):
    for c in list(combinations(list(range(i * range_vars, (i + 1) * range_vars)), 2)):
        constraint[(c[0], c[1])] = 2 * λ

# ----- QUBO -----
QUBO = {}
for i in range(n_vars * range_vars):
    QUBO[(i, i)] = Q[i, i]
for i in range(n_vars * range_vars):
    for j in range(i + 1, n_vars * range_vars):
        QUBO[(i, j)] = 2 * Q[i, j]

qubo = {}
for i, j in QUBO.keys():
    qubo[(i, j)] = -1 * QUBO[(i, j)]
    if (i, j) in constraint.keys():
        qubo[(i, j)] += constraint[(i, j)]

sampler = SASampler(
    num_sweeps=1000,
    num_reads=10)
res = sampler.sample_qubo(Q=qubo)
print(res.record)
print(sorted(res.record, key=lambda x: x[1]))
