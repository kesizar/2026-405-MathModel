# -*- coding: utf-8 -*-
"""
A题 求解器 v2 —— 加强版 min-max mTSP
init: iterated-LS giant tour + DP split  /  k-means++ 聚类
local search: relocate / swap / intra 2-opt / double-bridge perturbation
"""
import numpy as np
import openpyxl, os, math
from numpy.random import default_rng
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.cluster.vq import kmeans2

BASE = r'D:/HuaweiMoveData/Users/17669/Desktop/A 题'
OUT = r'D:/A题_work'
SCALE = 0.1
SPEED = 55.0
SERVICE = 5.0 / 60.0
LVL = {'I': 3, 'II': 2, 'III': 1}

def load_case(sheet):
    wb = openpyxl.load_workbook(os.path.join(BASE, '附件1.xlsx'), data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    return [(int(r[0]), float(r[1]), float(r[2]), LVL[str(r[3]).strip()])
            for r in rows[1:] if r[0] is not None]

def dist_matrix(pts):
    coords = np.array([[0.0, 0.0]] + [[p[1], p[2]] for p in pts])
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    return d * SCALE / SPEED

def mst_time(D):
    return minimum_spanning_tree(D).sum()

# ---------- route cost ----------
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

# ---------- 2-opt ----------
def two_opt(route, D):
    route = list(route)
    seq = [0] + route + [0]
    improved = True
    while improved:
        improved = False
        best = 0.0
        bi = bj = -1
        m = len(seq)
        for i in range(1, m - 2):
            si = seq[i - 1]
            for j in range(i + 1, m - 1):
                old = D[si, seq[i]] + D[seq[j], seq[j + 1]]
                new = D[si, seq[j]] + D[seq[i], seq[j + 1]]
                if new - old < best - 1e-12:
                    best = new - old
                    bi, bj = i, j
        if bi != -1:
            seq[bi:bj + 1] = seq[bi:bj + 1][::-1]
            improved = True
    return seq[1:-1]

# ---------- nearest neighbor ----------
def nn_tour(visits, D, rng, start=0):
    M = len(visits)
    if M == 0:
        return []
    unvisited = list(range(M))
    order = []
    cur = start  # point index
    while unvisited:
        best_u = -1
        bestd = np.inf
        for u in unvisited:
            dd = D[cur, int(visits[u])]
            if dd < bestd:
                bestd = dd
                best_u = u
        order.append(best_u)
        cur = int(visits[best_u])
        unvisited.remove(best_u)
    return [int(visits[u]) for u in order]

def double_bridge(route, rng):
    """random double-bridge move; route: list of point indices (no depot)."""
    n = len(route)
    if n < 8:
        return route
    cuts = sorted(rng.choice(np.arange(1, n), size=3, replace=False))
    a, b, c = cuts
    # segments: [0:a][a:b][b:c][c:n]
    seg1 = route[:a]
    seg2 = route[a:b]
    seg3 = route[b:c]
    seg4 = route[c:]
    return seg1 + seg3 + seg2 + seg4

def giant_tour(visits, D, rng):
    """iterated local search TSP over visits."""
    best = two_opt(nn_tour(visits, D, rng, 0), D)
    bestlen = route_travel(best, D)
    for _ in range(40):
        cand = two_opt(double_bridge(best, rng), D)
        cl = route_travel(cand, D)
        if cl < bestlen - 1e-12:
            best, bestlen = cand, cl
    return best

def split_tour(gtour, D, N):
    M = len(gtour)
    if M == 0:
        return [[] for _ in range(N)]
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
    for k in range(1, N):
        for b in range(M):
            best = INF
            bestj = -1
            for j in range(b):
                v = max(dp[k - 1, j], cost[j + 1, b])
                if v < best:
                    best = v
                    bestj = j
            dp[k, b] = best
            par[k, b] = bestj
    sol = []
    b = M - 1
    for k in range(N - 1, -1, -1):
        j = par[k, b]
        seg = gtour[0:b + 1] if k == 0 else gtour[j + 1:b + 1]
        sol.append(list(seg))
        b = j
    sol.reverse()
    return sol

def kmeans_init(visits, coords, D, N, rng, seed):
    vcoords = coords[visits]
    centroids, labels = kmeans2(vcoords, N, minit='++', seed=seed)
    clusters = [[] for _ in range(N)]
    for u, lab in zip(visits, labels):
        clusters[int(lab)].append(int(u))
    sol = []
    for c in clusters:
        if c:
            sol.append(two_opt(nn_tour(np.array(c), D, rng, 0), D))
        else:
            sol.append([])
    return sol

# ---------- local search ----------
def local_search(sol, D, objective='makespan', max_iter=800, rng=None):
    N = len(sol)
    cur = [list(r) for r in sol]
    cur_obj = eval_sol(cur, D)

    def better(a, b):
        if objective == 'makespan':
            return a[0] < b[0] - 1e-9
        return a < b

    for _ in range(max_iter):
        improved = False
        for i in range(N):
            cur[i] = two_opt(cur[i], D)
        cur_obj = eval_sol(cur, D)
        loads = [route_total(r, D) for r in cur]
        h = int(np.argmax(loads))
        best_obj = cur_obj
        best_move = None
        for vi in range(len(cur[h])):
            v = cur[h][vi]
            newh = cur[h][:vi] + cur[h][vi + 1:]
            lh = route_total(newh, D)
            for l in range(N):
                if l == h:
                    continue
                rl = cur[l]
                for pos in range(len(rl) + 1):
                    newl = rl[:pos] + [v] + rl[pos:]
                    ll = route_total(newl, D)
                    nl = list(loads)
                    nl[h] = lh
                    nl[l] = ll
                    cand = (max(nl), max(nl) - min(nl))
                    if better(cand, best_obj):
                        best_obj = cand
                        best_move = ('relocate', vi, l, pos)
                for wi in range(len(rl)):
                    w = rl[wi]
                    lh2 = route_total(cur[h][:vi] + [w] + cur[h][vi + 1:], D)
                    ll2 = route_total(rl[:wi] + [v] + rl[wi + 1:], D)
                    nl = list(loads)
                    nl[h] = lh2
                    nl[l] = ll2
                    cand = (max(nl), max(nl) - min(nl))
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
                cur[h][vi], cur[l][wi] = cur[l][wi], cur[h][vi]
            cur_obj = eval_sol(cur, D)
        else:
            break
    return cur, cur_obj

def solve(pts, N, objective='makespan', n_restart=8, max_iter=800, seed=0, init_sol=None):
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
        if r % 2 == 0:
            gt = giant_tour(visits, D, rng)
            inits.append(split_tour(gt, D, N))
        else:
            inits.append(kmeans_init(visits, coords, D, N, rng, seed + r))

    for s0 in inits:
        sol, obj = local_search(s0, D, objective, max_iter, rng)
        if objective == 'makespan':
            if obj[0] < best_obj[0] - 1e-9:
                best_sol, best_obj = sol, obj
        else:
            if obj < best_obj:
                best_sol, best_obj = sol, obj
    return best_sol, best_obj, D

if __name__ == '__main__':
    import time
    for c in ['Case1']:
        pts = load_case(c)
        for N in [2, 3]:
            t0 = time.time()
            sol, obj, D = solve(pts, N, 'makespan', n_restart=8, max_iter=800, seed=3)
            loads = [route_total(r, D) for r in sol]
            print(f'{c} N={N}: makespan={obj[0]:.4f} delta={obj[1]:.4f} '
                  f'loads={[round(x,2) for x in loads]} sizes={[len(r) for r in sol]} '
                  f't={time.time()-t0:.1f}s')
