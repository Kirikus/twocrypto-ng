# flake8: noqa
import sys, os
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from itertools import permutations

sys.stdout = sys.stderr

# IMPORTANT: keep number of workers equal to number of N_CASES since module variables do not reinitialize.
N_CASES = 32
N_COINS = 2
MAX_SAMPLES = 10000 // N_CASES  # Increase for fuzzing

A_MUL = 10000 * 2 ** 2
MIN_A = int(0.01 * A_MUL)
MAX_A = 1000 * A_MUL

# gamma from 1e-8 up to 0.05
MIN_GAMMA = 10 ** 10
MAX_GAMMA = 5 * 10 ** 16

METHODS = ["ORIGINAL", "SUM", "A", "B", "C", "C1", "S_ANCHORED", "S_ANCHORED_EXACT", "P_ANCHORED"]

pytest.progress = 0
pytest.passed_cases = 0
pytest.bad_cases = 0
pytest.t_start = time.time()
pytest.file = open("resN2-pure.csv", "a")

if os.fstat(pytest.file.fileno()).st_size == 0:
    pytest.file.write(",".join(map(str, ["A", "gamma", "x", "y", "yx", "perm", *[f"gas_{m},guess_{m},result_{m},steps_{m}" for m in METHODS]])) + "\n")
    pytest.file.flush()
pytest.buf = []

class Results:
    def __init__(self, gas, guess, result, steps):
        self.gas = gas
        self.guess = guess
        self.result = result
        self.steps = steps
    def __str__(self):
        return f"{self.gas},{self.guess},{self.result},{self.steps}"

@pytest.mark.parametrize(
    "_tmp", range(N_CASES)
)
@given(
    A=st.integers(min_value=MIN_A, max_value=MAX_A),
    x=st.integers(min_value=10 ** 18, max_value=10 ** 8 * 10 ** 18),  # 1 USD to 100M USD
    yx=st.integers(min_value=int(1.001e18), max_value=int(0.999e20)),  # <- ratio 1e18 * y/x, typically 1e18 * 1
    perm=st.integers(min_value=0, max_value=1),  # Permutation
    gamma=st.integers(min_value=MIN_GAMMA, max_value=MAX_GAMMA)
)
@settings(max_examples=MAX_SAMPLES, deadline=None)
def test_newton_D(math_optimized, math_unoptimized, A, x, yx, perm, gamma, _tmp):
    i, j = list(permutations(range(2)))[perm]
    X = [x, x * yx // 10 ** 18]
    X = [X[i], X[j]]

    if X[0] == X[1]:
        return

    pytest.progress += 1
    if pytest.progress % 10 == 0:
        print(
            f"Worker {_tmp}, {pytest.progress} | {pytest.passed_cases} cases processed in {time.time() - pytest.t_start:.1f} seconds. {pytest.bad_cases} bad cases.")
        if len(pytest.buf) >= 10:
            pytest.file.write("\n".join(pytest.buf) + "\n")
            pytest.file.flush()
            pytest.buf.clear()

    try:
        result_contract_obj = math_optimized.newton_D(A, gamma, X, 0, int(math_optimized.Method_ORIGINAL()), True)
        result_contract = result_contract_obj.D
    except:
        pytest.bad_cases += 1
        return  # original fails so we move on

    #results = {}
    results = {
        "ORIGINAL": Results(
            guess=result_contract_obj.D0,
            gas=math_optimized._computation.get_gas_used(),
            result=result_contract_obj.D,
            steps=result_contract_obj.steps
        )
    }
   
    for _method in METHODS[1:]:
    #for _method in METHODS:
        method = f"Method_{_method}"
        guess = None
        result = None
        gas = None
        steps = None
        try:
            guess = math_optimized.newton_D(A, gamma, X, 0, int(getattr(math_optimized, method)()), False).D0
            math_optimized._computation.get_gas_used()
        except:
            results[_method] = Results(guess=None, gas=None, result=None, steps=None)
            continue
            #return
        try:
            result_obj = math_optimized.newton_D(A, gamma, X, 0, getattr(math_optimized,method)(), True)
            result = result_obj.D
            gas = math_optimized._computation.get_gas_used()
            steps = result_obj.steps
        except:
            result = gas = steps = None
            #return

        results[_method] = Results(guess=guess, gas=gas, result=result, steps=steps)

    #method = "Method_S_ANCHORED"
    #guess = math_optimized.newton_D(A, gamma, X, 0, int(getattr(math_optimized, method)()), False).D0
    #result_obj = math_optimized.newton_D(A, gamma, X, 0, getattr(math_optimized,method)(), True)

    line = ",".join(map(str, [A, gamma, X[0], X[1], yx, perm, *[results[m] for m in METHODS]]))

    pytest.buf.append(line)
    pytest.passed_cases += 1

