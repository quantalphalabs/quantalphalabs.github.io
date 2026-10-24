#!/usr/bin/env python3
"""Runs the snap sim on the NFL calendar. Meant to be called every 15 minutes
(see .github/workflows/snap-sim.yml); each call does only what is due:

  night   8pm ET the night before each game day: every game that day.
  pregame 2 hours before each kickoff: that kickoff's games again, with the
          latest injury news, and a record of what changed since the night run.
  grade   once a final score is in: the last run before kickoff is scored
          against the result and the closing market, into record.json.

Everything it writes lives under --site (sim/ on the website). It never commits;
the workflow does. Exit status is 0 whether or not anything was due.
"""

import argparse
import datetime as dt
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import sim

ET = ZoneInfo("America/New_York")
NIGHT_BEFORE_HOUR = 20   # 8pm ET the day before a game day
PREGAME_HOURS = 2.0      # official inactives come out at 90 minutes; this runs before them
BACKTEST_FROM_WEEK = 3   # weeks 1-2 have too little current-season data to say anything
BACKTEST_SIMS = 400
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
ESPN_TEAM = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX"}  # ESPN abbreviation -> nflverse
OUT_STATUSES = {"out", "doubtful", "injured reserve", "suspension", "physically unable to perform",
                "pup", "ir", "non-football injury"}


def season_for(now):
    return now.year if now.month >= 3 else now.year - 1


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": "quantalphalabs-snapsim"})
    with urllib.request.urlopen(req, timeout=120) as r:
        path.write_bytes(r.read())


def fetch_data(data_dir, season, schedule_only=False):
    data_dir.mkdir(parents=True, exist_ok=True)
    files = ["schedules/games.csv"]
    if not schedule_only:
        files += [f"pbp/play_by_play_{s}.parquet" for s in (season - 1, season)]
        files += [f"ftn_charting/ftn_charting_{s}.parquet" for s in (season - 1, season)]
        files += [f"rosters/roster_{s}.parquet" for s in (season - 1, season)]
        files += [f"injuries/injuries_{season}.parquet"]
    for f in files:
        download(f"{NFLVERSE}/{f}", data_dir / Path(f).name)


def kickoff(row):
    t = str(row["gametime"]) if pd.notna(row["gametime"]) else "13:00"
    return dt.datetime.combine(dt.date.fromisoformat(row["gameday"]), dt.time.fromisoformat(t), ET)


def schedule(data_dir, season):
    g = pd.read_csv(data_dir / "games.csv")
    g = g[g["season"] == season].copy()
    g["ko"] = [kickoff(r) for _, r in g.iterrows()]
    g["date"] = [k.date() for k in g["ko"]]
    return g


def due_jobs(g, manifest, now):
    """What should run at `now`: a list of (stage, key, game_ids)."""
    jobs = []
    open_games = g[g["result"].isna()]
    for day, sub in open_games.groupby("date"):
        key = f"night:{day}"
        start = dt.datetime.combine(day - dt.timedelta(days=1), dt.time(NIGHT_BEFORE_HOUR), ET)
        if key not in manifest and start <= now < sub["ko"].min():
            jobs.append(("night", key, list(sub["game_id"])))
    for ko, sub in open_games.groupby("ko"):
        key = f"pregame:{ko.isoformat()}"
        if key not in manifest and ko - dt.timedelta(hours=PREGAME_HOURS) <= now < ko:
            jobs.append(("pregame", key, list(sub["game_id"])))
    return jobs


# --------------------------------------------------------------------------- injuries

def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def espn_out(D):
    """Players ESPN currently lists as Out/Doubtful/IR, as {team: {gsis_id}}.
    ESPN's injury feed is updated through the weekend, after nflverse's weekly
    file. Any failure falls back to nflverse alone; the caller records which."""
    req = urllib.request.Request(ESPN_INJURIES, headers={"User-Agent": "quantalphalabs-snapsim"})
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.load(r)

    found = []

    def walk(x, team=None):
        if isinstance(x, dict):
            t = x.get("team") if isinstance(x.get("team"), dict) else None
            if t and t.get("abbreviation"):
                team = t["abbreviation"]
            ath = x.get("athlete")
            status = x.get("status")
            if isinstance(ath, dict) and status:
                at = ath.get("team") if isinstance(ath.get("team"), dict) else None
                found.append((ath.get("displayName"), (at or {}).get("abbreviation") or team,
                              status if isinstance(status, str) else status.get("name", "")))
            for v in x.values():
                walk(v, team)
        elif isinstance(x, list):
            for v in x:
                walk(v, team)

    walk(payload)
    roster = D["roster"]
    by_name = {}
    for gsis, row in roster.iterrows():
        by_name.setdefault((row["team"], norm_name(row["full_name"])), gsis)
    out, n_status = {}, 0
    for name, team, status in found:
        if not name or not team:
            continue
        n_status += 1
        if str(status).strip().lower() not in OUT_STATUSES:
            continue
        team = ESPN_TEAM.get(team, team)
        gsis = by_name.get((team, norm_name(name)))
        if gsis:
            out.setdefault(team, set()).add(gsis)
    if n_status == 0:
        raise ValueError("ESPN injury feed had no recognizable entries")
    return out


# --------------------------------------------------------------------------- runs

def summary(o):
    return {"stage": o["stage"], "run_at": o["run_at"], "home_win": o["home_win"],
            "margin": o["margin_mean"], "total": o["total_mean"],
            "missing": o["missing"], "qb": o["qb"], "injuries": o["injuries"]}


def run_games(D, g, game_ids, stage, sims, games_dir, now, inj_source):
    leagues = {}
    rows = g[g["game_id"].isin(game_ids)]
    for _, row in rows.iterrows():
        wk = int(row["week"])
        if wk not in leagues:
            leagues[wk] = sim.League(D, wk)
        o = sim.run_game(leagues[wk], D, row, sims, use_roster=True)
        o.update({"stage": stage, "run_at": now.astimezone(dt.timezone.utc).isoformat(timespec="seconds"),
                  "kickoff_at": row["ko"].isoformat(), "injuries": inj_source,
                  "data_through_week": wk - 1})
        path = games_dir / f"{row['game_id']}.json"
        prev = json.loads(path.read_text()) if path.exists() else None
        o["runs"] = (prev or {}).get("runs", []) + [summary(o)]
        if prev and stage == "pregame":
            before = prev["runs"][-1]
            o["check"] = {
                "since": before["stage"], "since_at": before["run_at"],
                "home_win": [before["home_win"], o["home_win"]],
                "total": [before["total"], o["total_mean"]],
                "newly_out": {s: sorted(set(o["missing"][s]) - set(before["missing"][s])) for s in ("home", "away")},
                "back_in": {s: sorted(set(before["missing"][s]) - set(o["missing"][s])) for s in ("home", "away")},
                "qb_changed": before["qb"] != o["qb"],
            }
        path.write_text(json.dumps(o, default=float, separators=(",", ":")))
        print(f"  {stage:7} {o['away']:>3} @ {o['home']:<3}  home win {o['home_win']:.0%}"
              f"  total {o['total_mean']:.1f}", flush=True)


def grade(g, games_dir, record_path):
    record = json.loads(record_path.read_text()) if record_path.exists() else []
    done = {r["game_id"] for r in record}
    finals = g[g["result"].notna()].set_index("game_id")
    changed = False
    for path in sorted(games_dir.glob("*.json")):
        o = json.loads(path.read_text())
        gid = o["game_id"]
        if gid not in finals.index or gid in done:
            continue
        f = finals.loc[gid]
        ko = dt.datetime.fromisoformat(o["kickoff_at"])
        locked = [r for r in o["runs"] if dt.datetime.fromisoformat(r["run_at"]) < ko]
        if not locked:
            continue  # never simulated before kickoff: nothing honest to grade
        r = locked[-1]
        m = o.get("market", {})
        record.append({
            "game_id": gid, "week": o["week"], "home": o["home"], "away": o["away"],
            "kickoff_at": o["kickoff_at"], "stage": r["stage"], "run_at": r["run_at"],
            "sim_home_win": r["home_win"], "sim_margin": r["margin"], "sim_total": r["total"],
            "mkt_home_win": m.get("home_win"), "mkt_margin": m.get("spread_home"), "mkt_total": m.get("total"),
            "margin": float(f["result"]), "total": float(f["total"]),
            "home_score": int(f["home_score"]), "away_score": int(f["away_score"]),
        })
        o["final"] = {"home": int(f["home_score"]), "away": int(f["away_score"])}
        path.write_text(json.dumps(o, default=float, separators=(",", ":")))
        changed = True
        print(f"  graded  {o['away']:>3} @ {o['home']:<3}  final {int(f['away_score'])}-{int(f['home_score'])}")
    if changed:
        record.sort(key=lambda r: (r["kickoff_at"], r["game_id"]))
        record_path.write_text(json.dumps(record, indent=1))
    return changed, record


def record_summary(record):
    """Forward record: every row was simulated and committed before its kickoff."""
    rows = [r for r in record if r["mkt_margin"] is not None]
    if not rows:
        return None
    n = len(rows)
    win = [1.0 if r["margin"] > 0 else 0.5 if r["margin"] == 0 else 0.0 for r in rows]
    ats = [0, 0, 0]
    ou = [0, 0, 0]
    for r in rows:
        side = (r["sim_margin"] > r["mkt_margin"]) - (r["sim_margin"] < r["mkt_margin"])
        res = (r["margin"] > r["mkt_margin"]) - (r["margin"] < r["mkt_margin"])
        ats[2 if res == 0 else 0 if side == res else 1] += 1
        side = (r["sim_total"] > r["mkt_total"]) - (r["sim_total"] < r["mkt_total"])
        res = (r["total"] > r["mkt_total"]) - (r["total"] < r["mkt_total"])
        ou[2 if res == 0 else 0 if side == res else 1] += 1
    mk = [r for r in rows if r["mkt_home_win"] is not None]
    return {
        "games": n,
        "winners": sum((r["sim_home_win"] > 0.5) == (r["margin"] > 0) for r in rows if r["margin"] != 0),
        "brier_sim": sum((r["sim_home_win"] - w) ** 2 for r, w in zip(rows, win)) / n,
        "brier_market": (sum((r["mkt_home_win"] - (1.0 if r["margin"] > 0 else 0.5 if r["margin"] == 0 else 0.0)) ** 2
                             for r in mk) / len(mk)) if mk else None,
        "margin_mae_sim": sum(abs(r["sim_margin"] - r["margin"]) for r in rows) / n,
        "margin_mae_market": sum(abs(r["mkt_margin"] - r["margin"]) for r in rows) / n,
        "ats_w_l_p": ats, "ou_w_l_p": ou,
    }


def completed_week(g):
    """Latest regular-season week whose games are all final, or 0."""
    reg = g[g["game_type"] == "REG"]
    done = [int(w) for w, sub in reg.groupby("week") if sub["result"].notna().all()]
    return max(done) if done else 0


def backtest_due(g, site):
    path = site / "data" / "backtest.json"
    through = json.loads(path.read_text()).get("weeks", [0, 0])[1] if path.exists() else 0
    return completed_week(g) >= BACKTEST_FROM_WEEK and completed_week(g) > through


def refresh_backtest(D, g, site):
    """Re-runs the out-of-sample backtest through the latest finished week. Each
    week is simulated with only the data from before it, as on game day."""
    through = completed_week(g)
    games = g[g["game_type"] == "REG"]
    bt = sim.backtest(D, games, range(BACKTEST_FROM_WEEK, through + 1), BACKTEST_SIMS, quiet=True)
    out = sim.summarize_backtest(bt, as_dict=True)
    out["sims_per_game"] = BACKTEST_SIMS
    (site / "data" / "backtest.json").write_text(json.dumps(out, indent=1))
    bt.to_csv(site / "data" / "backtest.csv", index=False)
    print(f"backtest refreshed: weeks {BACKTEST_FROM_WEEK}-{through}, {out['games']} games")


def bundle(g, site, games_dir, record, now, sims):
    """Writes data/sim-data.js for the page: the current week's games plus the record."""
    open_games = g[g["result"].isna()]
    week = int(open_games["week"].min()) if len(open_games) else int(g["week"].max())
    wk = g[g["week"] == week].sort_values(["ko", "game_id"])
    games = []
    for _, row in wk.iterrows():
        path = games_dir / f"{row['game_id']}.json"
        if path.exists():
            o = json.loads(path.read_text())
            o["players"] = sorted(o["players"], key=sim.involvement, reverse=True)[:16]
        else:
            night = dt.datetime.combine(row["date"] - dt.timedelta(days=1), dt.time(NIGHT_BEFORE_HOUR), ET)
            o = {"game_id": row["game_id"], "home": row["home_team"], "away": row["away_team"],
                 "week": week, "kickoff_at": row["ko"].isoformat(), "pending": True,
                 "next_run_at": night.isoformat()}
        if pd.notna(row["result"]) and "final" not in o:
            o["final"] = {"home": int(row["home_score"]), "away": int(row["away_score"])}
        games.append(o)
    bt = site / "data" / "backtest.json"
    if not bt.exists():
        bt = Path(__file__).with_name("backtest.json")
    data = {"season": int(g["season"].iloc[0]), "week": week, "sims": sims,
            "updated_at": now.astimezone(dt.timezone.utc).isoformat(timespec="seconds"),
            "night_hour_et": NIGHT_BEFORE_HOUR, "pregame_hours": PREGAME_HOURS,
            "backtest": json.loads(bt.read_text()) if bt.exists() else None,
            "record": record, "record_summary": record_summary(record), "games": games}
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "sim-data.js").write_text(
        "window.SIM = " + json.dumps(data, default=float, separators=(",", ":")) + ";\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="scratch directory for nflverse downloads")
    ap.add_argument("--site", default="sim", help="website folder to write into")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--now", help="pretend it is this time (ISO, for testing)")
    ap.add_argument("--force", choices=["night", "pregame"], help="run this stage now for --games")
    ap.add_argument("--games", help="comma-separated game_ids for --force")
    ap.add_argument("--no-download", action="store_true", help="use files already in --data")
    args = ap.parse_args()

    now = dt.datetime.fromisoformat(args.now).astimezone(ET) if args.now else dt.datetime.now(ET)
    season = season_for(now)
    sim.SEASON = season
    data_dir, site = Path(args.data), Path(args.site)
    games_dir = site / "data" / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = site / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if not args.no_download:
        fetch_data(data_dir, season, schedule_only=True)
    g = schedule(data_dir, season)

    if args.force:
        ids = args.games.split(",") if args.games else list(g[g["result"].isna() & (g["week"] == g[g["result"].isna()]["week"].min())]["game_id"])
        jobs = [(args.force, f"{args.force}:manual:{now.isoformat(timespec='minutes')}", ids)]
    else:
        jobs = due_jobs(g, manifest, now)

    changed = False
    bt_due = backtest_due(g, site)
    if jobs or bt_due:
        if not args.no_download:
            fetch_data(data_dir, season)
        D = sim.load(data_dir)
    if bt_due:
        refresh_backtest(D, g, site)
        changed = True
    if jobs:
        try:
            D["live_out"] = espn_out(D)
            inj = "nflverse report + ESPN live feed"
        except Exception as e:  # noqa: BLE001 - any feed problem means fall back
            inj = f"nflverse report only (ESPN feed unavailable: {type(e).__name__})"
        print(f"injuries: {inj}")
        for stage, key, ids in jobs:
            print(f"{key}: {len(ids)} games")
            run_games(D, g, ids, stage, args.sims, games_dir, now, inj)
            manifest[key] = now.astimezone(dt.timezone.utc).isoformat(timespec="seconds")
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        changed = True

    graded, record = grade(g, games_dir, site / "data" / "record.json")
    if changed or graded or not (site / "data" / "sim-data.js").exists():
        bundle(g, site, games_dir, record, now, args.sims)
        print("site data updated")
    else:
        print("nothing due")


if __name__ == "__main__":
    sys.exit(main())
