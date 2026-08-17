# -*- coding: utf-8 -*-
"""
最终运行：问题1 & 问题2
- 计算理论下界与最小可行无人机数 Nmin
- P1: min makespan  /  P2: 负载均衡
- 输出 result1.xlsx / result2.xlsx + JSON + 图
"""
import sys, os, json, math, time
sys.path.insert(0, r'D:/A题_work')
import numpy as np
from solve3 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体
for f in ['Microsoft YaHei', 'SimHei', 'SimSun']:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [f]
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

OUT = r'D:/A题_work'
RES_OUT = r'D:/A题_work/结果文件'
FIG_OUT = r'D:/A题_work/figures'
for d in [RES_OUT, FIG_OUT]:
    os.makedirs(d, exist_ok=True)

def save_routes_xlsx(sol, pts, D, path, start_h=0.0, zones=None):
    """按模板格式保存调度方案：UAV ID | 1th Inspection Point | ...，另附时间表sheet。"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Sheet1'
    maxlen = max((len(r) for r in sol), default=0)
    header = ['UAV ID'] + [f'{k}th Inspection Point' for k in range(1, maxlen + 1)]
    ws.append(header)
    for i, r in enumerate(sol):
        ws.append([i + 1] + [int(x) for x in r])
    # 时间表
    ws2 = wb.create_sheet('TimeTable')
    ws2.append(['UAV ID', 'Seq', 'Point_ID', 'Arrive_Time(h)', 'Leave_Time(h)'])
    for i, r in enumerate(sol):
        t = start_h
        prev = 0
        for k, p in enumerate(r):
            t += D[prev, p]
            ws2.append([i + 1, k + 1, int(p), round(t, 4), round(t + SERVICE, 4)])
            t += SERVICE
            prev = p
    wb.save(path)

def write_results(result_files, table_rows):
    pass

if __name__ == '__main__':
    cases = ['Case1', 'Case2', 'Case3', 'Case4']
    summary = {}
    all_fig = {}
    for c in cases:
        pts = load_case(c)
        D = dist_matrix(pts)
        M = sum(p[3] for p in pts)
        S = M * SERVICE
        T_mst = mst_time(D)
        lb1 = math.ceil(S / 9.0)
        lb2 = math.ceil((S + T_mst) / 9.0)

        # 确定最小可行 N（makespan <= 9h）
        Nmin = None
        for N in range(1, 8):
            sol, obj, _ = solve(pts, N, n_restart=4, max_iter=1000, seed=13, objective='makespan')
            if obj[0] <= 9.0 + 1e-6:
                Nmin = N
                break
        assert Nmin is not None
        print(f'{c}: S={S:.3f} MST={T_mst:.3f} lb1={lb1} lb2={lb2} Nmin={Nmin}')

        # P1: min makespan
        sol1, obj1, D = solve(pts, Nmin, n_restart=8, max_iter=1500, seed=21, objective='makespan')
        loads1 = [route_total(r, D) for r in sol1]
        # P2: balance (lexicographic), init from P1
        sol2, obj2, D = solve(pts, Nmin, n_restart=6, max_iter=1500, seed=21,
                              init_sol=sol1, objective='balance')
        loads2 = [route_total(r, D) for r in sol2]
        # 若P2 makespan略差则回退P1
        if obj2[0] > obj1[0] + 1e-6:
            sol2, obj2, loads2 = sol1, obj1, loads1

        summary[c] = dict(
            n_points=len(pts), visits=M, S_h=round(S, 4), MST_h=round(T_mst, 4),
            lb1=lb1, lb2=lb2, Nmin=Nmin,
            P1=dict(N=Nmin, Tmax=round(float(obj1[0]), 4), Tmin=round(float(min(loads1)), 4),
                    loads=[round(float(x), 4) for x in loads1]),
            P2=dict(N=Nmin, Tmax=round(float(obj2[0]), 4), Tmin=round(float(min(loads2)), 4),
                    delta=round(float(obj2[1]), 4),
                    loads=[round(float(x), 4) for x in loads2]),
        )
        save_routes_xlsx(sol1, pts, D, os.path.join(RES_OUT, f'result1_{c}.xlsx'))
        save_routes_xlsx(sol2, pts, D, os.path.join(RES_OUT, f'result2_{c}.xlsx'))
        all_fig[c] = (pts, D, sol1, sol2, loads1, loads2)

    json.dump(summary, open(os.path.join(OUT, 'summary_p1p2.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    # ------- 绘图 -------
    for c in cases:
        pts, D, sol1, sol2, loads1, loads2 = all_fig[c]
        coords = np.array([[p[1], p[2]] for p in pts])
        ids = [p[0] for p in pts]
        lvls = [p[3] for p in pts]
        # 路线图（P1）
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(sol1)))
        for i, r in enumerate(sol1):
            seq = [0] + list(r) + [0]
            xy = np.array([[0, 0]] + [coords[p - 1] for p in r] + [[0, 0]])
            ax.plot(xy[:, 0], xy[:, 1], '-o', color=colors[i], lw=1.3, ms=2.5,
                    label=f'UAV{i+1} ({len(r)}点)')
        ax.scatter(0, 0, marker='*', c='k', s=180, zorder=5, label='基地')
        ax.set_title(f'{c} 问题一 路径规划 (N={len(sol1)}, Tmax={round(float(obj1[0]),2)}h)')
        ax.set_xlabel('X (单位=100m)')
        ax.set_ylabel('Y (单位=100m)')
        ax.legend(fontsize=7, loc='best')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_OUT, f'{c}_P1_routes.png'), dpi=130)
        plt.close()

        # 负载条形图（P1 vs P2）
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = np.arange(len(loads2))
        w = 0.38
        ax.bar(x - w / 2, loads1, w, label='问题一', color='#4C72B0')
        ax.bar(x + w / 2, loads2, w, label='问题二(均衡)', color='#DD8452')
        ax.axhline(9, color='r', ls='--', lw=1, label='9小时')
        ax.set_xticks(x)
        ax.set_xticklabels([f'UAV{i+1}' for i in x])
        ax.set_ylabel('工作时间 (h)')
        ax.set_title(f'{c} 无人机负载对比')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_OUT, f'{c}_loads.png'), dpi=130)
        plt.close()
    print('FIGURES DONE')
