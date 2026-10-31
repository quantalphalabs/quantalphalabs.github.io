#!/usr/bin/env python3
"""Fit the sim's settings and calibration on one season, and test them on the next.

    python tune.py --data DIR --train 2025 --test 2026 --sims 200

Every game is simulated using only data from before its week, as on game day.
Settings are chosen on the training season by coordinate search (one setting at
a time, kept only if the training Brier score improves), the calibration is fitted
on the training season's results, and only then is the test season scored. The
test season is never used to choose anything.

Writes calibration.json (read by sim.py on import) and a report to --out.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

import sim

IDENTITY = {"cal_a": 0.0, "cal_b": 1.0, "cal_c": 0.0}

# One setting at a time, in this order. The first value of each is the default.
SEARCH = [
    ("effect_pow", [1.0, 1.3, 1.6, 2.0]),
    ("hfa", [0.0, 1.0, 2.0]),
    ("own_w", [0.6, 0.4, 0.8]),
    ("prior_w", [0.35, 0.15, 0.6]),
    ("half_life", [4.0, 2.5, 8.0]),
    ("k_def_pass", [120, 60, 240]),
    ("k_def_run", [150, 75, 300]),
]

_STATE = {}


def _init(data, season, cfg):
    sim.SEASON = season
    sim.CFG.update(cfg)
    sim.CFG.update(IDENTITY)
    _STATE["D"] = sim.load(data)


def _week(args):
    season, wk, sims = args
    D = _STATE["D"]
    g = D["games"]
    g = g[(g["season"] == season) & (g["game_type"] == "REG") & (g["week"] == wk)
          & g["result"].notna() & g["spread_line"].notna()]
    if not len(g):
        return []
    L = sim.League(D, wk)
    rows = []
    for _, row in g.iterrows():
        o = sim.run_game(L, D, row, sims, keep_typical=False)
        s = np.array(o["scores"]).reshape(-1, 2)
        rows.append({
            "game_id": row["game_id"], "season": season, "week": wk,
            "margins": (s[:, 0] - s[:, 1]).tolist(), "totals": s.sum(1).tolist(),
            "mkt_margin": float(row["spread_line"]), "mkt_total": float(row["total_line"]),
            "mkt_home_win": o["market"]["home_win"],
            "margin": float(row["result"]), "total": float(row["total"]),
        })
    return rows


def backtest(data, season, weeks, cfg, sims, procs):
    with mp.Pool(procs, initializer=_init, initargs=(data, season, cfg)) as pool:
        out = pool.map(_week, [(season, w, sims) for w in weeks], chunksize=1)
    return [r for week in out for r in week]


def fit_calibration(rows):
    mu = np.array([np.mean(r["margins"]) for r in rows])
    y = np.array([r["margin"] for r in rows])
    b, a = np.polyfit(mu, y, 1)
    c = float(np.mean([r["total"] - np.mean(r["totals"]) for r in rows]))
    return {"cal_a": float(a), "cal_b": float(b), "cal_c": c}


def evaluate(rows, cal=IDENTITY, thresholds=(0.0, 1.0, 2.0, 3.0)):
    """Scores for a set of simulated games, after applying a calibration."""
    pw, mm, tt = [], [], []
    for r in rows:
        m = np.array(r["margins"], float)
        m = m + cal["cal_a"] + (cal["cal_b"] - 1) * m.mean()
        pw.append((m > 0).mean() + 0.5 * (m == 0).mean())
        mm.append(m.mean())
        tt.append(np.mean(r["totals"]) + cal["cal_c"])
    pw, mm, tt = map(np.array, (pw, mm, tt))
    y = np.array([r["margin"] for r in rows])
    tot = np.array([r["total"] for r in rows])
    mk = np.array([r["mkt_margin"] for r in rows])
    mkt_t = np.array([r["mkt_total"] for r in rows])
    mkp = np.array([r["mkt_home_win"] if r["mkt_home_win"] is not None else np.nan for r in rows])
    win = (y > 0) + 0.5 * (y == 0)
    pc = np.clip(pw, 0.02, 0.98)
    out = {
        "games": len(rows),
        "brier": float(np.mean((pw - win) ** 2)),
        "brier_market": float(np.nanmean((mkp - win) ** 2)),
        "logloss": float(-np.mean(win * np.log(pc) + (1 - win) * np.log(1 - pc))),
        "winners": int(((pw > 0.5) == (y > 0))[y != 0].sum()),
        "winners_market": int(((mk > 0) == (y > 0))[(y != 0) & (mk != 0)].sum()),
        "margin_mae": float(np.mean(np.abs(mm - y))),
        "margin_mae_market": float(np.mean(np.abs(mk - y))),
        "margin_sd": float(mm.std()), "market_sd": float(mk.std()),
        "slope": float(np.polyfit(mm, y, 1)[0]),
        "total_mae": float(np.mean(np.abs(tt - tot))),
        "total_mae_market": float(np.mean(np.abs(mkt_t - tot))),
        "total_bias": float(np.mean(tt - tot)),
        "ats": {}, "ou": {},
    }
    edge, res = mm - mk, y - mk
    for t in thresholds:
        k = (np.abs(edge) >= t) & (res != 0)
        hit = np.sign(edge[k]) == np.sign(res[k])
        dog = np.mean(np.sign(edge[k]) != np.sign(mk[k])) if k.sum() else 0.0
        out["ats"][str(t)] = {"w": int(hit.sum()), "l": int((~hit).sum()), "dog_share": float(dog)}
        te, tr = tt - mkt_t, tot - mkt_t
        k2 = (np.abs(te) >= t) & (tr != 0)
        h2 = np.sign(te[k2]) == np.sign(tr[k2])
        out["ou"][str(t)] = {"w": int(h2.sum()), "l": int((~h2).sum())}
    return out


def line(tag, e):
    a = e["ats"]
    return (f"{tag:<34} brier {e['brier']:.4f} (mkt {e['brier_market']:.4f})  "
            f"margin MAE {e['margin_mae']:.2f} (mkt {e['margin_mae_market']:.2f})  "
            f"sd {e['margin_sd']:.1f}/{e['market_sd']:.1f}  total bias {e['total_bias']:+.1f}  "
            f"ATS all {a['0.0']['w']}-{a['0.0']['l']} dogs {a['0.0']['dog_share']:.0%}  "
            f">=2 {a['2.0']['w']}-{a['2.0']['l']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--train", type=int, default=2025)
    ap.add_argument("--test", type=int, default=2026)
    ap.add_argument("--train-weeks", default="3-18")
    ap.add_argument("--test-weeks", default="3-18")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--procs", type=int, default=mp.cpu_count())
    ap.add_argument("--out", default="tune_out")
    ap.add_argument("--no-search", action="store_true", help="only score the current settings")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    tw = range(*[int(x) + i for i, x in enumerate(args.train_weeks.split("-"))])
    vw = range(*[int(x) + i for i, x in enumerate(args.test_weeks.split("-"))])
    base = {k: sim.CFG[k] for k in sim.CFG if not k.startswith("cal_")}
    log = []

    def score(cfg, tag):
        rows = backtest(args.data, args.train, tw, cfg, args.sims, args.procs)
        e = evaluate(rows)
        print(line(tag, e), flush=True)
        log.append({"tag": tag, "cfg": dict(cfg), "train": e})
        return rows, e

    best = dict(base)
    best_rows, best_e = score(best, "baseline")
    if not args.no_search:
        for key, values in SEARCH:
            for v in values[1:]:
                trial = dict(best, **{key: v})
                rows, e = score(trial, f"{key}={v}")
                if e["brier"] < best_e["brier"] - 0.0005:
                    best, best_rows, best_e = trial, rows, e
            print(f"  -> keep {key}={best[key]}", flush=True)

    cal = fit_calibration(best_rows)
    train_cal = evaluate(best_rows, cal)
    print(line(f"train {args.train}, calibrated", train_cal))

    # Only now look at the test season, once, with everything fixed.
    test_rows = backtest(args.data, args.test, vw, best, args.sims, args.procs)
    test_base = backtest(args.data, args.test, vw, base, args.sims, args.procs)
    results = {
        "train_season": args.train, "test_season": args.test, "sims": args.sims,
        "chosen": best, "calibration": cal,
        "train": {"baseline": log[0]["train"], "tuned": best_e, "tuned_calibrated": train_cal},
        "test": {"baseline": evaluate(test_base), "tuned": evaluate(test_rows),
                 "tuned_calibrated": evaluate(test_rows, cal)},
        "search": log,
    }
    for k, e in results["test"].items():
        print(line(f"test {args.test}, {k}", e))
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "test_rows.json").write_text(json.dumps(test_rows))
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
