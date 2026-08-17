# -*- coding: utf-8 -*-
"""
问题3 v2(轻量)：时间窗感知调度
1) 空间mTSP(绕行距离) -> 访问集合
2) 每架无人机：贪心"最早可服务"重排 + 2-opt(带等待)
3) 轻量再均衡：仅移动max路线的末访问点到min路线
"""
import sys, os, json
sys.path.insert(0, r'D:/A题_work')
import numpy as np
from solve3 import load_case, dist_matrix, solve, SERVICE
from zones import load_zones, seg_intersects_circle, detour_extra

SCALE = 0.1
SPEED = 55.0
START_H = 8.0

def travel_time(u, v, t_dep, zones, D0, coords):
    straight = D0[u, v]
    t_arr = t_dep + straight
    extra_u = 0.0
    for z in zones:
        if z['t0'] < t_arr and z['t1'] > t_dep:
            if seg_intersects_circle(coords[u], coords[v], z['c'], z['r']):
                extra_u += detour_extra(coords[u], coords[v], z['c'], z['r'])
    return straight + extra_u * SCALE / SPEED

def ensure_accessible(p, t, zones, coords, dur=5.0 / 60.0):
    pos = np.array(coords[p], float)
    t_cur = t
    for _ in range(400):
        bz = None
        for z in zones:
            if z['t0'] < t_cur + dur and z['t1'] > t_cur:
                if np.linalg.norm(pos - np.array(z['c'], float)) < z['r']:
                    bz = z
                    break
        if bz is None:
            return t_cur
        t_cur = max(t_cur, bz['t1'])
    return t_cur

def route_finish(order, zones, coords, D0, start_h=START_H):
    t = start_h
    prev = 0
    tl = []
    for p in order:
        t += travel_time(prev, p, t, zones, D0, coords)
        t = ensure_accessible(p, t, zones, coords)
        tl.append((p, round(t, 4)))
        t += SERVICE
        prev = p
    t += D0[prev, 0]
    return t, tl

def greedy_order(route, zones, coords, D0, start_h=START_H):
    remaining = list(route)
    order = []
    cur = 0
    t = start_h
    while remaining:
        best = None
        best_t = np.inf
        for v in remaining:
            arr = t + travel_time(cur, v, t, zones, D0, coords)
            ts = ensure_accessible(v, arr, zones, coords)
            if ts < best_t - 1e-9:
                best_t = ts
                best = v
        order.append(best)
        cur = best
        t = best_t + SERVICE
        remaining.remove(best)
    return order

def two_opt_order(order, zones, coords, D0, start_h=START_H, max_pass=5):
    order = list(order)
    m = len(order)
    cur = route_finish(order, zones, coords, D0, start_h)[0]
    for _ in range(max_pass):
        improved = False
        for i in range(m - 1):
            for j in range(i + 1, m):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                cf = route_finish(cand, zones, coords, D0, start_h)[0]
                if cf < cur - 1e-9:
                    order, cur = cand, cf
                    improved = True
        if not improved:
            break
    return order, cur

def solve_p3(case, N, seed=31):
    pts = load_case(case)
    zones = load_zones(case)
    D0 = dist_matrix(pts)
    coords = [(0.0, 0.0)] + [(p[1], p[2]) for p in pts]

    sol, obj, D = solve(pts, N, n_restart=6, max_iter=1200, seed=seed, objective='balance')
    D_eff = D0.copy()
    prev_block = None
    for it in range(6):
        block = {}
        for r in sol:
            t = START_H
            prev = 0
            for p in r:
                tt = travel_time(prev, p, t, zones, D0, coords)
                if tt > D0[prev, p] + 1e-9:
                    block[(prev, p)] = tt
                    block[(p, prev)] = tt
                t += tt + SERVICE
                prev = p
        new_D = D0.copy()
        for (u, v), tt in block.items():
            new_D[u, v] = tt
        D_eff = new_D
        sol, obj, _ = solve(pts, N, n_restart=3, max_iter=800, seed=seed + it + 1,
                            objective='balance', D_override=D_eff, init_sol=sol)
        if prev_block is not None and set(block.keys()) == prev_block:
            break
        prev_block = set(block.keys())

    orders = []
    for r in sol:
        o, fin = two_opt_order(greedy_order(r, zones, coords, D0), zones, coords, D0)
        orders.append(o)

    # 轻量再均衡：直接评估单点迁移(不重排)
    for _ in range(20):
        fts = [route_finish(o, zones, coords, D0)[0] for o in orders]
        h = int(np.argmax(fts))
        l = int(np.argmin(fts))
        best_cand = None
        for vi in range(len(orders[h])):
            v = orders[h][vi]
            newh = orders[h][:vi] + orders[h][vi + 1:]
            fh = route_finish(newh, zones, coords, D0)[0]
            for pos in range(len(orders[l]) + 1):
                newl = orders[l][:pos] + [v] + orders[l][pos:]
                fl = route_finish(newl, zones, coords, D0)[0]
                mx = max(fh, fl)
                dl = mx - min(fh, fl)
                cand = (mx, dl, vi, pos)
                if best_cand is None or (mx, dl) < (best_cand[0], best_cand[1]):
                    best_cand = cand
        mx, dl, vi, pos = best_cand
        if mx < max(fts) - 1e-6:
            v = orders[h].pop(vi)
            orders[l].insert(pos, v)
        else:
            break

    fts = [route_finish(o, zones, coords, D0)[0] for o in orders]
    timelines = [route_finish(o, zones, coords, D0)[1] for o in orders]
    total_wait = 0.0
    for o in orders:
        t = START_H
        prev = 0
        for p in o:
            t += travel_time(prev, p, t, zones, D0, coords)
            t2 = ensure_accessible(p, t, zones, coords)
            total_wait += t2 - t
            t = t2 + SERVICE
            prev = p
    route_times = [f - START_H for f in fts]
    return orders, route_times, total_wait, timelines, D0, zones, coords, pts

if __name__ == '__main__':
    import time
    summary = {}
    for c in ['Case1', 'Case2', 'Case3', 'Case4']:
        N = {'Case1': 4, 'Case2': 2, 'Case3': 4, 'Case4': 4}[c]
        t0 = time.time()
        orders, rts, wait, tl, D0, zones, coords, pts = solve_p3(c, N)
        tmax = max(rts); tmin = min(rts); delta = tmax - tmin
        summary[c] = dict(N=N, Tmax=round(float(tmax), 4), Tmin=round(float(tmin), 4),
                          delta=round(float(delta), 4),
                          route_times=[round(float(x), 4) for x in rts],
                          wait_h=round(float(wait), 4))
        print(f'{c}: Tmax={tmax:.3f} Tmin={tmin:.3f} delta={delta:.3f} wait={wait:.3f} '
              f'times={[round(x,2) for x in rts]} t={time.time()-t0:.0f}s', flush=True)
    json.dump(summary, open(r'D:/A题_work/summary_p3b.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
