# -*- coding: utf-8 -*-
"""
问题3：含圆形临时飞行管制区（禁飞区）的路径规划
- 线段与圆相交判定
- 绕圆最短路径（切线+圆弧）
- 时间相关仿真（管制区仅在活动时间窗内禁飞）
- 迭代不动点：D_eff 更新 + 重求解
"""
import numpy as np, openpyxl, os

BASE = r'D:/HuaweiMoveData/Users/17669/Desktop/A 题'
SCALE = 0.1
SPEED = 55.0

def load_zones(sheet):
    wb = openpyxl.load_workbook(os.path.join(BASE, '附件2.xlsx'), data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    zones = []
    for r in rows[1:]:
        if r[0] is None:
            continue
        zid = str(r[0]).strip()
        cx, cy, rad = float(r[1]), float(r[2]), float(r[3])
        st = str(r[4]).strip()
        en = str(r[5]).strip()
        h, m = st.split(':')
        t0 = int(h) + int(m) / 60.0
        h, m = en.split(':')
        t1 = int(h) + int(m) / 60.0
        zones.append(dict(id=zid, c=(cx, cy), r=rad, t0=t0, t1=t1))
    return zones

def seg_intersects_circle(a, b, c, r):
    a = np.asarray(a, float); b = np.asarray(b, float); c = np.asarray(c, float)
    ab = b - a
    L2 = ab.dot(ab)
    if L2 == 0:
        return np.linalg.norm(a - c) < r
    t = np.clip((c - a).dot(ab) / L2, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(closest - c) < r - 1e-9

def detour_extra(a, b, c, r):
    """绕圆最短路径相对直线段的额外长度(单位)。a,b在圆外。"""
    a = np.asarray(a, float); b = np.asarray(b, float); c = np.asarray(c, float)
    du = np.linalg.norm(a - c)
    dv = np.linalg.norm(b - c)
    straight = np.linalg.norm(a - b)
    if du <= r or dv <= r:
        return 0.0
    cosang = np.clip((a - c).dot(b - c) / (du * dv), -1.0, 1.0)
    theta = np.arccos(cosang)
    alpha = np.arccos(r / du)
    beta = np.arccos(r / dv)
    phi = theta - alpha - beta
    if phi <= 0:
        return 0.0
    L = np.sqrt(du * du - r * r) + np.sqrt(dv * dv - r * r) + r * phi
    return max(0.0, L - straight)

def segment_travel_time(a, b, zones, t_dep, D_straight_ab):
    """给定出发时刻t_dep，返回该航段实际飞行时间(h)，考虑活动管制区绕行。"""
    straight = D_straight_ab
    t_arr = t_dep + straight
    total_extra = 0.0
    for z in zones:
        # 管制区活动窗口与该航段穿越窗口是否有重叠
        if z['t0'] < t_arr and z['t1'] > t_dep:  # 时间重叠
            if seg_intersects_circle(a, b, z['c'], z['r']):
                extra_u = detour_extra(a, b, z['c'], z['r'])
                total_extra += extra_u
    return straight + total_extra * SCALE / SPEED

def simulate(sol, pts, zones, D0, start_h=8.0):
    """仿真：返回 (makespan_h, actual_segment_times_dict, timeline)。
    actual_segment_times: {(u,v): time_h} 用于更新D_eff。"""
    coords = [None] + [(p[1], p[2]) for p in pts]
    block = {}
    makespan = 0.0
    timeline = []
    for r in sol:
        t = start_h
        prev = 0
        for p in r:
            a = coords[prev]
            b = coords[p]
            tt = segment_travel_time(a, b, zones, t, D0[prev, p])
            if tt > D0[prev, p] + 1e-9:
                block[(prev, p)] = tt
                block[(p, prev)] = tt
            t += tt
            timeline.append((p, t))
            t += 5.0 / 60.0
            prev = p
        t += D0[prev, 0]  # 回基地(近似直线，管制区对回程影响已在block中考虑则略)
        makespan = max(makespan, t)
    return makespan, block, timeline

if __name__ == '__main__':
    # 检查点是否落入禁飞区、以及简单相交统计
    from solve3 import load_case, dist_matrix
    for c in ['Case1', 'Case2', 'Case3', 'Case4']:
        pts = load_case(c)
        zones = load_zones(c)
        coords = [(0, 0)] + [(p[1], p[2]) for p in pts]
        inside = []
        for z in zones:
            for i, p in enumerate(pts):
                if np.linalg.norm(np.array([p[1], p[2]]) - np.array(z['c'])) < z['r']:
                    inside.append((p[0], z['id']))
        # 统计多少条基地-点、点-点直线穿过禁飞区
        D = dist_matrix(pts)
        n_seg = 0
        n_cross = 0
        from itertools import combinations
        idxs = list(range(len(pts) + 1))
        for (u, v) in combinations(idxs, 2):
            n_seg += 1
            for z in zones:
                if seg_intersects_circle(coords[u], coords[v], z['c'], z['r']):
                    n_cross += 1
                    break
        print(f'{c}: zones={len(zones)}, points_inside_zones={inside}, '
              f'crossing_segments={n_cross}/{n_seg}')
