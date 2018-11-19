import math
import numpy as np

NCAR = 20  # number of cars in zone
MCAR = 5  # maximum number of moved cars
GAMMA = 0.9

LAM_RENT1 = 4
LAM_RET1 = 3

LAM_RENT2 = 2
LAM_RET2 = 3

RENT_PRICE = 10
MOVE_PRICE = 2

diff_table = {}


def poisson(lam, n):
    return math.exp(-lam) * math.pow(lam, n) / math.factorial(n)


def poisson_diff(d, lam1, lam2):
    global diff_table
    key = (d, lam1, lam2)
    if key in diff_table:
        p = diff_table[key]
    else:
        p = 0
        for v1 in range(0, NCAR + MCAR):
            v2 = v1 + d
            if v2 >= 0:
                p += poisson(lam1, v1) * poisson(lam2, v2)
        diff_table[key] = p
    return p


def p_transfer(a1, b1, a2, b2, m):
    """ Probability of transferring from (a1,b1) to (a2,b2) if m cars are moved overnignt """
    assert a1 <= NCAR
    assert b1 <= NCAR
    assert a2 <= NCAR
    assert b2 <= NCAR

    if a1 < m or b1 < -m:
        return 0

    if a2 == 0:
        a2s = (-10, 1)
    elif a2 == NCAR:
        a2s = (NCAR, 10+NCAR)
    else:
        a2s = (a2, a2+1)

    if b2 == 0:
        b2s = (-10, 1)
    elif b2 == NCAR:
        b2s = (NCAR, 10+NCAR)
    else:
        b2s = (b2, b2+1)

    p = 0
    for a2 in range(*a2s):
        for b2 in range(*b2s):
            d1 = a2 - a1 - m
            d2 = b2 - b1 + m

            p += poisson_diff(d1, LAM_RENT1, LAM_RET1) * poisson_diff(d2, LAM_RENT2, LAM_RET2)

    return p


def reward(a1, b1, m):
    r = 0
    for n in range(NCAR+1):
        r += poisson(LAM_RENT1, n) * min(n, a1) * RENT_PRICE
    for n in range(NCAR+1):
        r += poisson(LAM_RENT2, n) * min(n, b1) * RENT_PRICE

    r -= MOVE_PRICE * abs(m)
    return r


def value(values, a1, b1, m):
    v = reward(a1, b1, m)
    for a2 in range(NCAR + 1):
        for b2 in range(NCAR + 1):
            p_trans = p_transfer(a1, b1, a2, b2, m)
            v += GAMMA * p_trans * values[a2, b2]
    return v


def evaluate_policy(policy, values):
    values2 = np.zeros_like(values)
    for a1 in range(NCAR + 1):
        for b1 in range(NCAR + 1):
            m = policy[a1, b1]
            values2[a1, b1] = value(values, a1, b1, m)
    return values2


def greedy_policy(values):
    policy = np.zeros_like(values)
    for a1 in range(NCAR + 1):
        for b1 in range(NCAR + 1):
            best_m = None
            best_v = None
            for m in range(-MCAR, MCAR+1):
                v = value(values, a1, b1, m)
                if best_v is None or v > best_v:
                    best_v = v
                    best_m = m
            policy[a1, b1] = best_m

    return policy


def get_default_policy():
    return np.zeros([NCAR+1, NCAR+1], dtype=np.int32)


def get_default_values():
    return np.zeros([NCAR+1, NCAR+1])
