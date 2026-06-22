import numpy as np

data = {
    'baseline': {'reward': [25.611, 25.867, 26.294], 'ep_len': [988.4, 987.3, 990.9]},
    'v1':       {'reward': [30.959, 29.334, 28.887], 'ep_len': [983.3, 983.9, 980.3]},
    'v4':       {'reward': [30.411, 26.509, 29.405], 'ep_len': [992.3, 975.5, 981.9]},
    'v5':       {'reward': [27.295, 27.145, 28.555], 'ep_len': [979.4, 974.7, 977.3]},
    'v6':       {'reward': [23.984, 26.114, 28.190], 'ep_len': [970.5, 959.5, 972.8]},
}

print("=" * 75)
print("ROUND 2 — Cassie (seeds 4,5,6)")
print("=" * 75)
print(f"{'Config':<12s} {'Reward':>20s} {'Ep Len':>18s} {'CV':>8s}")
print("-" * 75)

base_mean = np.mean(data['baseline']['reward'])
for group in ['baseline', 'v1', 'v4', 'v5', 'v6']:
    rs = data[group]['reward']
    es = data[group]['ep_len']
    cv = np.std(rs, ddof=0) / np.mean(rs) * 100
    print(f"{group:<12s} {np.mean(rs):>10.3f} ± {np.std(rs,ddof=0):<7.3f} {np.mean(es):>10.1f} ± {np.std(es,ddof=0):<6.1f} {cv:>7.1f}%")

print()
print("=" * 75)
print("DELTA vs BASELINE")
print("=" * 75)
for group in ['v1', 'v4', 'v5', 'v6']:
    rs = data[group]['reward']
    delta = np.mean(rs) - base_mean
    pct = delta / base_mean * 100
    print(f"  {group}: Δ = {delta:+.3f}  ({pct:+.1f}%)")

print()
print("=" * 75)
print("PER-SEED")
print("=" * 75)
for group in ['baseline', 'v1', 'v4', 'v5', 'v6']:
    rs = data[group]['reward']
    print(f"  {group}: seeds=[{', '.join(f'{v:.3f}' for v in rs)}]  range=[{min(rs):.3f}, {max(rs):.3f}]")

print()
print("=" * 75)
print("ALL 5 SEEDS COMBINED (round 1: seed1,seed42 + round 2: seed4,5,6)")
print("=" * 75)
all_data = {
    'baseline': [25.020, 25.437, 25.611, 25.867, 26.294],
    'v1':       [29.327, 26.352, 30.959, 29.334, 28.887],
    'v4':       [27.298, 30.675, 30.411, 26.509, 29.405],
}
for group in ['baseline', 'v1', 'v4']:
    rs = all_data[group]
    cv = np.std(rs, ddof=0) / np.mean(rs) * 100
    delta = np.mean(rs) - np.mean(all_data['baseline'])
    pct = delta / np.mean(all_data['baseline']) * 100
    print(f"  {group}: {np.mean(rs):.3f} ± {np.std(rs,ddof=0):.3f}  CV={cv:.1f}%  Δ={delta:+.3f} ({pct:+.1f}%)")
