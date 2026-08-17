# -*- coding: utf-8 -*-
"""
A题 多无人机协同巡检路径优化 —— 核心求解器
问题1(理论最小N + min-makespan mTSP) 与 问题2(负载均衡)
"""
import numpy as np
import openpyxl, json, os, math
from collections import Counter
from numpy.random import default_rng
from scipy.sparse.csgraph import minimum_spanning_tree

BASE = r'D:/HuaweiMoveData/Users/17669/Desktop/A 题'
OUT = r'D:/A题_work'
os.makedirs(OUT, exist_ok=True)

SCALE = 0.1
SPEED = 55.0
SERVICE = 5.0 / 60.0
LVL = {'I': 3, 'II': 2, 'III': 1}
START_H = 8.0  # problem3 departure time (clock hours)

def load_case(sheet):
    wb = openpyxl.load_workbook(os.path.join(BASE, '附件1.xlsx'), data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    pts = []
    for r in rows[1:]:
        if r[0] is None:
            continue
        pts.append((int(r[0]), float(r[1]), float(r[2]), LVL[str(r[3]).strip()]))
    return pts

def dist_matrix(pts):
    coords = np.array([[0.0, 0.0]] + [[p[1], p[2]] for p in pts])
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    return d * SCALE / SPEED  # hours

def mst_time(D):
    """MST of depot+points, edge weights = travel time (hours)."""
    T = minimum_spanning_tree(D)
    return T.sum()

# ---------------- route cost ----------------
def route_travel(route, D):
    if not route:
        return 0.0
    t = D[0, route[0]]
    for a, b in zip(route, route[1:]):
        t += D[a, b]
    return t + D[route[-1], 0]

def route_total(route, D):
    return route_travel(route, D) + len(route) * SERVICE

def eval_sol(sol, D):
    loads = [route_total(r, D) for r in sol]
    return (max(loads), max(loads) - min(loads))

# ---------------- 2-opt ----------------
def two_opt(route, D):
    seq = [0] + list(route) + [0]
    improved = True
    while improved:
        improved = False
        best = 0.0
        bi = bj = -1
        m = len(seq)
        for i in range(1, m - 2):
            for j in range(i + 1, m - 1):
                old = D[seq[i - 1], seq[i]] + D[seq[j], seq[j + 1]]
                new = D[seq[i - 1], seq[j]] + D[seq[i], seq[j + 1]]
                delta = new - old
                if delta < best - 1e-12:
                    best = delta
                    bi, bj = i, j
        if bi != -1:
            seq[bi:bj + 1] = seq[bi:bj + 1][::-1]
            improved = True
    return seq[1:-1]

# ---------------- nearest neighbor tour ----------------
def nn_tour(visits, coords, D, rng):
    """Greedy nearest-neighbor tour over visit list starting from depot."""
    M = len(visits)
    if M == 0:
        return []
    order = []
    unvisited = set(range(M))
    cur = 0  # depot
    while unvisited:
        # nearest unvisited to cur
        best = None
        bestd = np.inf
        for u in unvisited:
            p = int(visits[u])
            dd = D[cur, p]
            if dd < bestd:
                bestd = dd
                best = u
        order.append(best)
        cur = int(visits[best])
        unvisited.remove(best)
    return [int(visits[u]) for u in order]

def giant_tour(visits, coords, D, rng):
    route = nn_tour(visits, coords, D, rng)
    return two_opt(route, D)

# ---------------- split giant tour into N routes (min-max DP) ----------------
def split_tour(gtour, D, N):
    M = len(gtour)
    if M == 0:
        return [[] for _ in range(N)]
    # cost[a][b] inclusive
    cost = np.zeros((M, M))
    for a in range(M):
        t = D[0, gtour[a]]
        for b in range(a, M):
            if b > a:
                t += D[gtour[b - 1], gtour[b]]
            cost[a, b] = t + D[gtour[b], 0] + (b - a + 1) * SERVICE
    INF = 1e18
    dp = np.full((N, M), INF)
    par = np.zeros((N, M), dtype=int)
    for b in range(M):
        dp[0, b] = cost[0, b]
        par[0, b] = -1
    for k in range(1, N):
        for b in range(M):
            best = INF
            bestj = -1
            for j in range(0, b):
                v = max(dp[k - 1, j], cost[j + 1, b])
                if v < best:
                    best = v
                    bestj = j
            dp[k, b] = best
            par[k, b] = bestj
    # reconstruct
    sol = []
    b = M - 1
    k = N - 1
    while k >= 0:
        j = par[k, b]
        if k == 0:
            seg = gtour[0:b + 1]
        else:
            seg = gtour[j + 1:b + 1]
        sol.append(list(seg))
        b = j
        k -= 1
    sol.reverse()
    return sol

# ---------------- local search (min-max relocate/swap) ----------------
def local_search(sol, D, objective='makespan', max_iter=400, rng=None):
    N = len(sol)
    cur = [list(r) for r in sol]
    cur_obj = eval_sol(cur, D)

    def better(a, b):
        if objective == 'balance':
            return a < b
        return a[0] < b[0] - 1e-12 or (abs(a[0] - b[0]) < 1e-12 and a[1] < b[1] - 1e-12)

    it = 0
    while it < max_iter:
        it += 1
        improved = False
        # 2-opt all routes
        for i in range(N):
            cur[i] = two_opt(cur[i], D)
        cur_obj = eval_sol(cur, D)
        loads = [route_total(r, D) for r in cur]
        h = int(np.argmax(loads))
        # best improving move
        best_move = None
        best_obj = cur_obj
        for vi in range(len(cur[h])):
            v = cur[h][vi]
            newh = cur[h][:vi] + cur[h][vi + 1:]
            lh = route_total(newh, D)
            for l in range(N):
                if l == h:
                    continue
                # relocate into l
                for pos in range(len(cur[l]) + 1):
                    newl = cur[l][:pos] + [v] + cur[l][pos:]
                    ll = route_total(newl, D)
                    # new solution loads: replace h,l
                    newloads = list(loads)
                    newloads[h] = lh
                    newloads[l] = ll
                    mx = max(newloads)
                    dlt = mx - min(newloads)
                    cand = (mx, dlt)
                    if better(cand, best_obj):
                        best_obj = cand
                        best_move = ('relocate', vi, l, pos)
                # swap v with w in l
                for wi in range(len(cur[l])):
                    w = cur[l][wi]
                    newh2 = cur[h][:vi] + [w] + cur[h][vi + 1:]
                    newl2 = cur[l][:wi] + [v] + cur[l][wi + 1:]
                    lh2 = route_total(newh2, D)
                    ll2 = route_total(newl2, D)
                    newloads = list(loads)
                    newloads[h] = lh2
                    newloads[l] = ll2
                    mx = max(newloads)
                    dlt = mx - min(newloads)
                    cand = (mx, dlt)
                    if better(cand, best_obj):
                        best_obj = cand
                        best_move = ('swap', vi, l, wi)
        if best_move is not None:
            if best_move[0] == 'relocate':
                _, vi, l, pos = best_move
                v = cur[h].pop(vi)
                cur[l].insert(pos, v)
            else:
                _, vi, l, wi = best_move
                v = cur[h][vi]
                cur[h][vi] = cur[l][wi]
                cur[l][wi] = v
            cur_obj = eval_sol(cur, D)
            improved = True
        if not improved:
            break
    return cur, cur_obj

def solve(pts, N, objective='makespan', n_restart=6, max_iter=400, seed=0, init_sol=None):
    D = dist_matrix(pts)
    coords = np.array([[0.0, 0.0]] + [[p[1], p[2]] for p in pts])
    visits = []
    for idx, p in enumerate(pts):
        visits += [idx + 1] * p[3]
    visits = np.array(visits, dtype=int)
    rng = default_rng(seed)
    best_sol, best_obj = None, (1e18, 1e18)

    inits = []
    if init_sol is not None:
        inits.append([list(r) for r in init_sol])
    for r in range(n_restart):
        gt = giant_tour(visits, coords, D, rng)
        s0 = split_tour(gt, D, N)
        inits.append(s0)

    for s0 in inits:
        sol, obj = local_search(s0, D, objective, max_iter, rng)
        if obj < best_obj or (objective == 'makespan' and obj[0] < best_obj[0] - 1e-9):
            if objective == 'makespan':
                if obj[0] < best_obj[0] - 1e-9:
                    best_sol, best_obj = sol, obj
            else:
                if obj < best_obj:
                    best_sol, best_obj = sol, obj
    return best_sol, best_obj, D

if __name__ == '__main__':
    results = {}
    for c in ['Case1', 'Case2', 'Case3', 'Case4']:
        pts = load_case(c)
        D = dist_matrix(pts)
        M = sum(p[3] for p in pts)
        S = M * SERVICE
        T_mst = mst_time(D)
        n_lb = math.ceil((S + T_mst) / 9.0)
        print(f'== {c}: visits={M}, S={S:.3f}h, MST={T_mst:.3f}h, Nmin_bound=ceil((S+MST)/9)={n_lb}')
        results[c] = dict(visits=M, S_h=S, MST_h=T_mst, n_lb=n_lb)
    json.dump(results, open(os.path.join(OUT, 'stats.json'), 'w'), indent=1)
