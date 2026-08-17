# -*- coding: utf-8 -*-
import sys, os, json, time
sys.path.insert(0, r'D:/A题_work')
from solve import *

if __name__ == '__main__':
    t0 = time.time()
    case = sys.argv[1] if len(sys.argv) > 1 else 'Case1'
    N = int(sys.argv[2]) if len(sys.argv) > 2 else None
    pts = load_case(case)
    if N is None:
        M = sum(p[3] for p in pts)
        S = M * SERVICE
        D = dist_matrix(pts)
        N = math.ceil((S + mst_time(D)) / 9.0)
    print(f'[{case}] solving P1 with N={N} ...')
    sol, obj, D = solve(pts, N, objective='makespan', n_restart=4, max_iter=300, seed=7)
    makespan, delta = obj
    loads = [route_total(r, D) for r in sol]
    print(f'  makespan={makespan:.4f}h  Tmax={max(loads):.4f}  Tmin={min(loads):.4f}  delta={delta:.4f}')
    print(f'  loads(h) =', [round(x,3) for x in loads])
    print(f'  within 9h? {makespan <= 9.0 + 1e-9}')
    print(f'  elapsed {time.time()-t0:.1f}s')
    for i, r in enumerate(sol):
        print(f'  UAV{i+1}: {len(r)} visits -> {r[:15]}{"..." if len(r)>15 else ""}')
