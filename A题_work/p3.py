# -*- coding: utf-8 -*-
"""
问题3：含时间窗禁飞区的多无人机巡检
模型：8:00起飞；航段穿越"活动"禁飞区时绕圆飞行(绕行)；点位于活动禁飞区内时等待至解禁。
迭代：D_eff 更新(绕行) -> 重求解 -> 仿真(含等待) -> 直到绕行集合稳定。
"""
import sys, os, json, math
sys.path.insert(0, r'D:/A题_work')
import numpy as np
from solve3 import load_case, dist_matrix, solve, route_total, SERVICE
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

def simulate_full(sol, pts, zones, D0, coords, start_h=START_H):
    block = {}
    route_times = []
    timeline = []
    total_wait = 0.0
    for r in sol:
        t = start_h
        prev = 0
        for p in r:
            tt = travel_time(prev, p, t, zones, D0, coords)
            if tt > D0[prev, p] + 1e-9:
                block[(prev, p)] = tt
                block[(p, prev)] = tt
            t += tt
            t2 = ensure_accessible(p, t, zones, coords)
            total_wait += t2 - t
            t = t2
            timeline.append((p, round(t, 4)))
            t += SERVICE
            prev = p
        t += D0[prev, 0]
        route_times.append(t - start_h)
    return max(route_times), block, total_wait, timeline, route_times

def solve_p3(case, N, seed=31):
    pts = load_case(case)
    zones = load_zones(case)
    D0 = dist_matrix(pts)
    coords = [(0.0, 0.0)] + [(p[1], p[2]) for p in pts]

    sol, obj, D = solve(pts, N, n_restart=6, max_iter=1200, seed=seed, objective='balance')
    D_eff = D0.copy()
    prev_block = None
    for it in range(8):
        ms, block, wait, tl, rts = simulate_full(sol, pts, zones, D0, coords)
        new_D = D0.copy()
        for (u, v), tt in block.items():
            new_D[u, v] = tt
        D_eff = new_D
        sol, obj, _ = solve(pts, N, n_restart=4, max_iter=1000, seed=seed + it + 1,
                            objective='balance', D_override=D_eff, init_sol=sol)
        if prev_block is not None and set(block.keys()) == prev_block:
            break
        prev_block = set(block.keys())

    ms, block, wait, tl, rts = simulate_full(sol, pts, zones, D0, coords)
    return sol, ms, wait, block, tl, rts, D0, zones, coords

if __name__ == '__main__':
    summary = {}
    for c in ['Case1', 'Case2', 'Case3', 'Case4']:
        N = {'Case1': 4, 'Case2': 2, 'Case3': 4, 'Case4': 4}[c]
        sol, ms, wait, block, tl, rts, D0, zones, coords = solve_p3(c, N)
        tmax = max(rts)
        tmin = min(rts)
        delta = tmax - tmin
        summary[c] = dict(N=N, Tmax=round(float(tmax), 4), Tmin=round(float(tmin), 4),
                          delta=round(float(delta), 4),
                          route_times=[round(float(x), 4) for x in rts],
                          wait_h=round(float(wait), 4),
                          n_blocked=len(block) // 2)
        print(f'{c}: N={N} Tmax={tmax:.3f} Tmin={tmin:.3f} delta={delta:.3f} '
              f'wait={wait:.3f}h blocked_segments={len(block)//2} times={[round(x,2) for x in rts]}')
    json.dump(summary, open(r'D:/A题_work/summary_p3.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
