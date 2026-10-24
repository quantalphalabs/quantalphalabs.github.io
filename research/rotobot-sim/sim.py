#!/usr/bin/env python3
"""Snap-level Monte Carlo NFL game simulator, rebuilt from RotoBot's public description.

RotoBot's posts say: "We simulated <game> N times. Every snap is built from real
2026 play-by-play: the coverage each defense actually plays, and how each player
actually performs against it. Here's the most typical game." This file is an
independent reconstruction of that idea from public nflverse data. It does not
use any RotoBot code, data or API.

How a snap is built
-------------------
1. Play call. Pass/run comes from a league logistic model on down, distance,
   field position, score and clock, shifted by the offense's own pass rate over
   expected. Fourth downs use the league's go/kick/punt frequencies for that
   field position and distance, with late-game overrides.
2. Defensive look. The defense blitzes at its own observed rate (FTN charting).
   Man/zone coverage is not public for 2026, so blitz is the scheme variable.
3. Dropback. Sack / scramble / throw-away / target probabilities combine league
   rates for that look with the QB's and the defense's own rates (odds-ratio).
4. Outcome. A real play is resampled from a mixture of pools: the ball
   carrier's own plays, the QB's (split by blitz), the plays this defense
   allowed to that position, and the league, each weighted by its sample size.
   The resampled play supplies yards, completion, turnover and clock run-off.

Everything is weighted toward recent 2026 games, with 2025 as a light prior.
Run `python sim.py --help` for usage.
"""

import argparse
import bisect
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

PRIOR_SEASON_WEIGHT = 0.35   # a 2025 play counts this much against a current one
HALF_LIFE_WEEKS = 4.0        # recency decay inside the current season
SHARE_HALF_LIFE_GAMES = 2.0  # target and carry shares move faster than efficiency
SEASON = 2026

PBP_COLS = [
    "game_id", "play_id", "season_type", "week", "posteam", "defteam", "home_team", "away_team",
    "play_type", "qtr", "down", "ydstogo", "yardline_100", "game_seconds_remaining",
    "half_seconds_remaining", "score_differential", "yards_gained", "air_yards",
    "complete_pass", "interception", "sack", "qb_scramble", "qb_dropback", "pass_attempt",
    "fumble_lost", "receiver_player_id", "receiver_player_name", "rusher_player_id",
    "rusher_player_name", "passer_player_id", "passer_player_name", "penalty", "penalty_yards",
    "penalty_team", "first_down_penalty", "field_goal_result", "kick_distance", "return_yards",
    "touchback", "extra_point_result", "two_point_conv_result", "qb_kneel", "qb_spike", "pass_oe",
    "td_team", "first_down", "fixed_drive", "touchdown",
]


# --------------------------------------------------------------------------- data

def load(data_dir):
    d = Path(data_dir)
    frames = []
    for season in (SEASON - 1, SEASON):
        p = pd.read_parquet(d / f"play_by_play_{season}.parquet", columns=PBP_COLS)
        p["season"] = season
        frames.append(p)
    pbp = pd.concat(frames, ignore_index=True)
    pbp = pbp.sort_values(["game_id", "play_id"], kind="stable").reset_index(drop=True)

    ftn = pd.concat([pd.read_parquet(d / f"ftn_charting_{s}.parquet",
                                     columns=["nflverse_game_id", "nflverse_play_id", "n_blitzers"])
                     for s in (SEASON - 1, SEASON)], ignore_index=True)
    ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    ftn["play_id"] = ftn["play_id"].astype(float)
    pbp = pbp.merge(ftn.drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left")
    pbp["blitz"] = (pbp["n_blitzers"].fillna(0) > 0).astype(int)

    # Game clock that ran before the next snap, from the next row in the same quarter.
    g = pbp.groupby("game_id", sort=False)
    nxt_clock = g["game_seconds_remaining"].shift(-1)
    nxt_qtr = g["qtr"].shift(-1)
    el = pbp["game_seconds_remaining"] - nxt_clock
    pbp["elapsed"] = el.where((nxt_qtr == pbp["qtr"]) & (el >= 0) & (el <= 50))

    # Field position the receiving team starts at after a kickoff.
    scrim = pbp["play_type"].isin(["pass", "run", "no_play", "punt", "field_goal", "qb_kneel"])
    nxt_yl = pbp["yardline_100"].where(scrim).groupby(pbp["game_id"]).transform(
        lambda s: s.shift(-1).bfill())
    nxt_pos = pbp["posteam"].where(scrim).groupby(pbp["game_id"]).transform(
        lambda s: s.shift(-1).bfill())
    pbp["ko_start"] = nxt_yl.where((pbp["play_type"] == "kickoff") & (nxt_pos == pbp["posteam"]))

    rosters = pd.concat([pd.read_parquet(d / f"roster_{s}.parquet") for s in (SEASON - 1, SEASON)])
    rosters = rosters.sort_values(["season", "week"])
    pos = rosters.dropna(subset=["gsis_id"]).groupby("gsis_id")["position"].last().to_dict()
    cur = rosters[rosters["season"] == SEASON].dropna(subset=["gsis_id"]).groupby("gsis_id").last()
    games = pd.read_csv(d / "games.csv")
    injuries = pd.read_parquet(d / f"injuries_{SEASON}.parquet")
    return {"pbp": pbp, "pos": pos, "roster": cur, "games": games, "injuries": injuries}


ZONES = (20, 12, 5)  # red zone split three ways; plays near the goal line behave differently


def zone(yl):
    return 0 if yl > 20 else 1 if yl > 12 else 2 if yl > 5 else 3


def sit(down, togo):
    """Coarse down-and-distance group, for player pools: 1st, 2nd, late-short, late-long."""
    return 0 if down == 1 else 1 if down == 2 else 2 if togo < 5 else 3


LATE_TOGO = (1, 2, 4, 6, 9)  # 3rd/4th down: 1, 2, 3-4, 5-6, 7-9, 10+


def fsit(down, togo):
    """Fine down-and-distance group, for league pools. On third down what matters is
    the distance to the sticks: offenses throw to the marker, so a 3rd-and-6 play
    can't be resampled from 3rd-and-15 without losing the conversion rate."""
    if down == 1:
        return 0
    if down == 2:
        return 1 if togo <= 3 else 2 if togo <= 7 else 3
    return 4 + bisect.bisect_left(LATE_TOGO, togo)


SITS = 4
FSITS = 10


def fsit_to_sit(fs):
    return 0 if fs == 0 else 1 if fs <= 3 else 2 if fs <= 6 else 3


NBINS = 6


def target_bin(df):
    y = df["yards_gained"].fillna(0)
    return np.select([df["interception"] == 1, df["complete_pass"] != 1, y < 5, y < 10, y < 20],
                     [0, 1, 2, 3, 4], 5)


def run_bin(df):
    y = df["yards_gained"].fillna(0)
    return np.select([df["fumble_lost"] == 1, y <= 0, y <= 3, y <= 6, y <= 12], [0, 1, 2, 3, 4], 5)


def bin_dist(df):
    v = df.groupby("bin")["w"].sum().reindex(range(NBINS), fill_value=0).to_numpy()
    return v / v.sum() if v.sum() else np.full(NBINS, 1 / NBINS)


def ratio(df, base, k):
    """How this unit shifts each outcome bin relative to the league, shrunk toward 1."""
    v = df.groupby("bin")["w"].sum().reindex(range(NBINS), fill_value=0).to_numpy()
    return ((v + k * base) / (v.sum() + k)) / np.maximum(base, 1e-9)


def pos_group(pos):
    if pos in ("RB", "FB"):
        return "RB"
    if pos in ("TE", "QB"):
        return pos
    return "WR"


def logit_fit(X, y, w, iters=30, l2=1e-2):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = w * p * (1 - p)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(len(b))
        step = np.linalg.solve(H, X.T @ (w * (y - p)) - l2 * b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def pass_features(down, togo, yl, sd, qtr, clock):
    half = clock if qtr in (2, 4, 5) else clock + 900
    sdc = max(-21.0, min(21.0, sd)) / 7
    late = 1.0 if qtr >= 4 else 0.0
    two = 1.0 if half <= 120 else 0.0
    return (1.0, float(down == 2), float(down == 3), float(down == 4), math.log(max(togo, 1)),
            yl / 100, sdc, late * max(-sdc, 0), late * max(sdc, 0), two, two * float(sd < 0))


def pass_features_frame(df):
    half = np.where(df["qtr"].isin([2, 4]), df["clock"], df["clock"] + 900)
    sdc = df["score_differential"].clip(-21, 21).to_numpy() / 7
    late = (df["qtr"] >= 4).to_numpy().astype(float)
    two = (half <= 120).astype(float)
    down = df["down"].to_numpy()
    return np.column_stack([
        np.ones(len(df)), down == 2, down == 3, down == 4, np.log(df["ydstogo"].clip(lower=1)),
        df["yardline_100"] / 100, sdc, late * np.maximum(-sdc, 0), late * np.maximum(sdc, 0),
        two, two * (sdc < 0)]).astype(float)


class Pool:
    """Weighted set of rows in one event table."""
    __slots__ = ("rows", "cum", "n")

    def __init__(self, rows, weights):
        self.rows = list(rows)
        self.cum = list(np.cumsum(weights)) if len(rows) else []
        self.n = float(self.cum[-1]) if self.cum else 0.0

    def draw(self, r):
        return self.rows[bisect.bisect_left(self.cum, r.random() * self.n)]


class Mix:
    """Mixture of pools. Each component is trusted in proportion to its sample size."""
    __slots__ = ("pools", "cum", "parts")

    def __init__(self, parts, fallback):
        # parts: [(pool, base_weight, k)]; whatever weight they don't claim goes to fallback
        pools, ws, self.parts = [], [], []
        for pool, base, k in parts:
            if pool is None or pool.n <= 0:
                continue
            w = base * pool.n / (pool.n + k)
            pools.append(pool)
            ws.append(w)
        pools.append(fallback)
        ws.append(max(0.0, 1.0 - sum(ws)))
        self.pools = pools
        self.cum = list(np.cumsum(ws) / sum(ws))
        self.parts = ws

    def draw(self, r):
        u = r.random()
        i = bisect.bisect_left(self.cum, u)
        return self.pools[min(i, len(self.pools) - 1)].draw(r)


def pool_by(df, keys, weight="w"):
    out = {}
    if len(df) == 0:
        return out
    for key, sub in df.groupby(keys, sort=False):
        out[key] = Pool(sub.index.to_numpy(), sub[weight].to_numpy())
    return out


def table(df, cols):
    return {c: df[c].tolist() for c in cols}


# --------------------------------------------------------------------------- model build

DROPBACK_CLASSES = ("sack", "scramble", "throwaway", "target")
YL_BUCKETS = [0, 5, 15, 25, 35, 45, 55, 65, 100]
TOGO_BUCKETS = [0, 1, 2, 3, 5, 9, 100]


def bucket(edges, v):
    return min(len(edges) - 2, max(0, bisect.bisect_left(edges, v) - 1))


class League:
    """Everything that depends only on the data cutoff, shared by every game that week."""

    def __init__(self, D, week):
        pbp = D["pbp"]
        hist = pbp[(pbp["season"] < SEASON) | (pbp["week"] < week)].copy()
        cur = hist["season"] == SEASON
        hist["w"] = np.where(cur, 0.5 ** ((week - 1 - hist["week"]) / HALF_LIFE_WEEKS),
                             PRIOR_SEASON_WEIGHT)
        self.week = week
        self.hist = hist
        self.pos = D["pos"]

        scrim = hist[hist["play_type"].isin(["pass", "run"]) & (hist["qb_kneel"] != 1)
                     & (hist["qb_spike"] != 1) & hist["down"].notna()].copy()
        scrim["z"] = [zone(v) for v in scrim["yardline_100"]]
        scrim["rz"] = (scrim["z"] > 0).astype(int)
        scrim["sit"] = [sit(d, t) for d, t in zip(scrim["down"], scrim["ydstogo"])]
        scrim["fsit"] = [fsit(d, t) for d, t in zip(scrim["down"], scrim["ydstogo"])]

        # ---- pass/run call (a scramble was a called pass)
        calls = scrim.copy()
        calls["clock"] = np.where(calls["qtr"].isin([2, 4]), calls["half_seconds_remaining"],
                                  calls["half_seconds_remaining"] - 900).clip(0)
        X = pass_features_frame(calls)
        self.pass_beta = logit_fit(X, (calls["qb_dropback"] == 1).to_numpy().astype(float),
                                   calls["w"].to_numpy())
        oe = scrim.dropna(subset=["pass_oe"])
        sw = oe.groupby("posteam")["w"].sum()
        s_oe = (oe["pass_oe"] * oe["w"]).groupby(oe["posteam"]).sum()
        self.proe_offset = ((s_oe / (sw + 150)) / 100 / 0.235).to_dict()

        # ---- dropbacks
        db = scrim[scrim["qb_dropback"] == 1].copy()
        db["qb"] = db["passer_player_id"].fillna(db["rusher_player_id"])
        db["cls"] = np.select(
            [db["sack"] == 1, db["qb_scramble"] == 1, db["receiver_player_id"].isna()],
            ["sack", "scramble", "throwaway"], "target")
        self.db = db
        lg = db.groupby(["blitz", "z", "fsit", "cls"])["w"].sum().unstack(fill_value=0)
        self.lg_cls = {k: (lg.loc[k] / lg.loc[k].sum()).reindex(DROPBACK_CLASSES, fill_value=0)
                       for k in lg.index}
        self.lg_cls_all = db.groupby("cls")["w"].sum().reindex(DROPBACK_CLASSES, fill_value=0)
        self.lg_cls_all = self.lg_cls_all / self.lg_cls_all.sum()
        lg_blitz = (db["blitz"] * db["w"]).sum() / db["w"].sum()
        n_def = db.groupby("defteam")["w"].sum()
        n_blitz = (db["blitz"] * db["w"]).groupby(db["defteam"]).sum()
        self.blitz_rate = ((n_blitz + 100 * lg_blitz) / (n_def + 100)).to_dict()
        self.lg_blitz = lg_blitz

        sacks = db[db["cls"] == "sack"]
        self.sack_pool = Pool(sacks.index, sacks["w"])
        scr = db[db["cls"] == "scramble"]
        self.scr_lg = Pool(scr.index, scr["w"])
        self.scr_qb = pool_by(scr, "qb")

        # ---- targets
        tg = db[db["cls"] == "target"].copy()
        tg["pg"] = [pos_group(self.pos.get(p)) for p in tg["receiver_player_id"]]
        tg["yards"] = np.where(tg["complete_pass"] == 1, tg["yards_gained"], 0)
        self.tg = tg
        self.tg_lg = pool_by(tg, ["pg", "z", "fsit", "blitz"])
        self.tg_lg_any = pool_by(tg, ["z", "fsit"])
        self.tg_lg_sit = pool_by(tg, "fsit")
        self.tg_rec = pool_by(tg, ["receiver_player_id", "sit"])

        # ---- designed runs
        rn = scrim[(scrim["play_type"] == "run") & (scrim["qb_scramble"] != 1)
                   & scrim["rusher_player_id"].notna()].copy()
        rn["pg"] = [pos_group(self.pos.get(p)) for p in rn["rusher_player_id"]]
        self.rn = rn
        self.rn_lg = pool_by(rn, ["pg", "z", "fsit"])
        self.rn_lg_any = pool_by(rn, ["z", "fsit"])
        self.rn_lg_sit = pool_by(rn, "fsit")
        self.rn_rush = pool_by(rn, ["rusher_player_id", "sit"])

        # Outcome bins, for combining an offense with a defense (see Offense).
        tg["bin"] = target_bin(tg)
        rn["bin"] = run_bin(rn)
        self.T_bin = dict(zip(tg.index, tg["bin"]))
        self.R_bin = dict(zip(rn.index, rn["bin"]))
        self.tg_dist = {k: bin_dist(sub) for k, sub in tg.groupby("pg")}
        self.tg_dist_b = {k: bin_dist(sub) for k, sub in tg.groupby("blitz")}
        self.rn_dist = {k: bin_dist(sub) for k, sub in rn.groupby("pg")}
        self.rn_dist_all = bin_dist(rn)

        # Event tables: row -> (yards, complete, interception, fumble lost, air yards,
        # offensive TD, field zone it was run from). Plain tuples for a fast inner loop.
        for name, df in (("T", tg), ("R", rn), ("S", db)):
            td = ((df["touchdown"] == 1) & (df["td_team"] == df["posteam"])).astype(int)
            td = td * 10 + df["z"]
            setattr(self, name, {i: v for i, v in zip(df.index, zip(
                df["yards_gained"].fillna(0), df["complete_pass"].fillna(0),
                df["interception"].fillna(0), df["fumble_lost"].fillna(0),
                df["air_yards"].fillna(8), td))})

        # Clock run-off depends on the game state more than on the play: an offense
        # racing the clock snaps fast, a team protecting a lead uses all of it.
        # So the clock comes from (state, result) pools rather than the resampled play.
        ck = scrim.dropna(subset=["elapsed"])
        state = [clock_state(q, h, sd) for q, h, sd in zip(
            ck["qtr"], ck["half_seconds_remaining"], ck["score_differential"])]
        kind = np.select([ck["sack"] == 1, ck["complete_pass"] == 1, ck["play_type"] == "pass"],
                         ["sack", "comp", "inc"], "run")
        el = pd.DataFrame({"s": state, "k": kind, "e": ck["elapsed"].to_numpy()})
        self.clock = {key: sub["e"].tolist() for key, sub in el.groupby(["s", "k"])}

        # ---- penalties (accepted, play wiped out)
        pen = hist[(hist["play_type"] == "no_play") & (hist["penalty"] == 1)
                   & hist["penalty_team"].notna() & hist["down"].notna()]
        self.p_pen = pen["w"].sum() / (scrim["w"].sum() + pen["w"].sum())
        self.pen = list(zip((pen["penalty_team"] == pen["posteam"]).astype(int),
                            pen["penalty_yards"].fillna(5), pen["first_down_penalty"].fillna(0),
                            pen["elapsed"].fillna(0)))

        # Penalties enforced on top of a play that counts (roughness, face mask...).
        live = scrim[(scrim["penalty"] == 1) & scrim["penalty_team"].notna()]
        self.p_live_pen = live["w"].sum() / scrim["w"].sum()
        self.live_pen = list(zip((live["penalty_team"] == live["posteam"]).astype(int),
                                 live["penalty_yards"].fillna(10), live["first_down_penalty"].fillna(0)))

        # Non-offensive touchdowns, which no resampled snap can produce.
        cur_td = hist["td_team"].notna() & (hist["td_team"] != hist["posteam"])
        ints = hist[hist["interception"] == 1]
        fums = hist[hist["fumble_lost"] == 1]
        self.p_int_td = float(cur_td[ints.index].mean()) if len(ints) else 0.0
        self.p_fum_td = float(cur_td[fums.index].mean()) if len(fums) else 0.0
        pr = hist[hist["play_type"] == "punt"]
        self.p_punt_td = float(cur_td[pr.index].mean()) if len(pr) else 0.0

        # ---- special teams
        kicks = hist[(hist["play_type"] == "field_goal") & hist["kick_distance"].notna()]
        Xf = np.column_stack([np.ones(len(kicks)), kicks["kick_distance"] / 10])
        self.fg_beta = logit_fit(Xf, (kicks["field_goal_result"] == "made").to_numpy().astype(float),
                                 kicks["w"].to_numpy())
        xp = hist[hist["play_type"] == "extra_point"]
        self.p_xp = float((xp["extra_point_result"] == "good").mean())
        tp = hist[hist["two_point_conv_result"].notna()]
        self.p_2pt = float((tp["two_point_conv_result"] == "success").mean()) if len(tp) else 0.48
        punts = hist[(hist["play_type"] == "punt") & hist["kick_distance"].notna()]
        self.punts = {}
        for yb, sub in punts.groupby((punts["yardline_100"] // 5).astype(int)):
            self.punts[yb] = list(zip(sub["kick_distance"] - sub["return_yards"].fillna(0),
                                      sub["touchback"].fillna(0)))
        ko = hist[(hist["season"] == SEASON) & hist["ko_start"].notna()]
        if len(ko) < 50:
            ko = hist[hist["ko_start"].notna()]
        self.ko_start = ko["ko_start"].clip(1, 99).tolist()

        # ---- fourth down
        fd = hist[(hist["down"] == 4) & hist["play_type"].isin(["pass", "run", "punt", "field_goal"])
                  & ~((hist["qtr"] >= 4) & (hist["game_seconds_remaining"] <= 300))]
        dec = np.select([fd["play_type"] == "punt", fd["play_type"] == "field_goal"],
                        ["punt", "fg"], "go")
        self.fourth = {}
        tb = [bucket(TOGO_BUCKETS, v) for v in fd["ydstogo"]]
        yb = [bucket(YL_BUCKETS, v) for v in fd["yardline_100"]]
        # Smooth each cell toward its field-position row, not toward a flat prior.
        row = {}
        for y, dd, w in zip(yb, dec, fd["w"]):
            row.setdefault(y, {"go": 0.0, "fg": 0.0, "punt": 0.0})[dd] += w
        cnt = {}
        for t, y, dd, w in zip(tb, yb, dec, fd["w"]):
            if (t, y) not in cnt:
                tot = sum(row[y].values())
                cnt[(t, y)] = {k: 2.0 * v / tot for k, v in row[y].items()}
            cnt[(t, y)][dd] += w
        for k, v in cnt.items():
            s = sum(v.values())
            self.fourth[k] = (v["go"] / s, v["fg"] / s)

    def p_pass(self, team, down, togo, yl, sd, qtr, clock):
        f = pass_features(down, togo, yl, sd, qtr, clock)
        z = sum(a * b for a, b in zip(f, self.pass_beta)) + self.proe_offset.get(team, 0.0)
        return 1 / (1 + math.exp(-z))

    def p_fg(self, yl):
        dist = yl + 18
        if dist > 66:
            return 0.0
        z = self.fg_beta[0] + self.fg_beta[1] * dist / 10
        return 1 / (1 + math.exp(-z))


class Offense:
    """One team's personnel and pools against one specific defense."""

    def __init__(self, L, D, team, opp, week, use_roster):
        self.team, self.opp = team, opp
        h = L.hist
        cur = h[(h["season"] == SEASON) & (h["posteam"] == team)]
        game_order = {g: i for i, g in enumerate(cur.sort_values("week")["game_id"].unique())}
        last_games = set(list(game_order)[-3:])
        inj = D["injuries"]
        out = set(inj[(inj["week"] == week) & (inj["team"] == team)
                      & inj["report_status"].isin(["Out", "Doubtful"])]["gsis_id"])
        if use_roster:
            r = D["roster"]
            out |= set(r[(r["team"] == team) & (r["status"] != "ACT")].index)
        # Players who have since moved to another team.
        latest_team = h[h["season"] == SEASON].groupby("receiver_player_id")["posteam"].last().to_dict()
        latest_team.update(h[h["season"] == SEASON].groupby("rusher_player_id")["posteam"].last().to_dict())
        self.missing = set()  # recent contributors ruled out this week

        def share(df, col):
            df = df[df[col].notna() & df["game_id"].isin(last_games)]
            if not len(df):
                return {}
            n = len(game_order)
            gw = df["game_id"].map(lambda g: 0.5 ** ((n - 1 - game_order[g]) / SHARE_HALF_LIFE_GAMES))
            s = gw.groupby(df[col]).sum()
            keep = [p not in out and latest_team.get(p, team) == team for p in s.index]
            self.missing |= {p for p, k in zip(s.index, keep) if not k and p in out}
            s = s[keep]
            return (s / s.sum()).to_dict() if s.sum() > 0 else {}

        tg = L.tg[L.tg["posteam"] == team]
        rn = L.rn[L.rn["posteam"] == team]
        self.tshare = share(tg, "receiver_player_id")
        self.rshare = share(rn, "rusher_player_id")

        dbs = L.db[(L.db["posteam"] == team) & (L.db["season"] == SEASON)]
        qb = None
        if len(dbs):
            last = dbs[dbs["game_id"] == list(game_order)[-1]] if game_order else dbs
            for cand in list(last["qb"].value_counts().index) + list(dbs["qb"].value_counts().index):
                if cand not in out:
                    qb = cand
                    break
            if len(last) and last["qb"].value_counts().index[0] != qb:
                self.missing.add(last["qb"].value_counts().index[0])
        self.qb = qb
        names = {}
        for idcol, ncol in (("receiver_player_id", "receiver_player_name"),
                            ("rusher_player_id", "rusher_player_name"),
                            ("passer_player_id", "passer_player_name")):
            sub = h[[idcol, ncol]].dropna().drop_duplicates(idcol, keep="last")
            names.update(dict(zip(sub[idcol], sub[ncol])))
        self.names = names

        # Receiver and rusher picks, as cumulative lists.
        self.t_ids = list(self.tshare)
        self.t_cum = list(np.cumsum([self.tshare[p] for p in self.t_ids]))
        self.r_ids = list(self.rshare)
        self.r_cum = list(np.cumsum([self.rshare[p] for p in self.r_ids]))

        self.blitz = L.blitz_rate.get(opp, L.lg_blitz)

        # Dropback class probabilities per (blitz, rz): league x QB factor x defense factor.
        qdb = L.db[L.db["qb"] == qb]
        ddb = L.db[L.db["defteam"] == opp]

        def rates(df, prior, k):
            c = df.groupby("cls")["w"].sum().reindex(DROPBACK_CLASSES, fill_value=0)
            return (c + k * prior) / (c.sum() + k)

        q_all = rates(qdb, L.lg_cls_all, 150)
        d_all = rates(ddb, L.lg_cls_all, 250)
        self.cls = {}
        for b in (0, 1):
            q = rates(qdb[qdb["blitz"] == b], q_all, 60)
            for z in range(len(ZONES) + 1):
                for fs in range(FSITS):
                    lgc = L.lg_cls.get((b, z, fs), L.lg_cls_all)
                    p = lgc * (q / L.lg_cls_all) * (d_all / L.lg_cls_all)
                    p = (p / p.sum()).to_numpy()
                    self.cls[(b, z, fs)] = list(np.cumsum(p))

        # Outcomes. The play itself is drawn from the ball carrier's own real plays
        # (shrunk toward league plays from the same field zone and down-distance).
        # The QB, offensive line and this defense then re-weight it: a drawn play is
        # kept with probability proportional to how much each of them makes that kind
        # of outcome more or less likely than the league does. Averaging pools instead
        # would halve every team effect; this multiplies them, as log5 does.
        tq = L.tg[L.tg["qb"] == qb]
        td = L.tg[L.tg["defteam"] == opp]
        q_ratio = {bb: ratio(tq[tq["blitz"] == bb], L.tg_dist_b[bb], 80) for bb in (0, 1)}
        d_ratio = {pg: ratio(td[td["pg"] == pg], L.tg_dist[pg], 120) for pg in L.tg_dist}
        self.tmix = {}
        for pid in self.t_ids:
            pg = pos_group(L.pos.get(pid))
            for st in range(SITS):
                own = L.tg_rec.get((pid, st))
                for z in range(len(ZONES) + 1):
                    for fs in range(FSITS):
                        if fsit_to_sit(fs) != st:
                            continue
                        lgp = (L.tg_lg.get((pg, z, fs, 0)) or L.tg_lg_any.get((z, fs))
                               or L.tg_lg_sit[fs])
                        for b in (0, 1):
                            lgb = (L.tg_lg.get((pg, z, fs, b)) or lgp)
                            m = Mix([(own, 0.6, 20)], lgb)
                            # The receiver's own plays already carry his usual QB.
                            acc = (q_ratio[b] ** (1 - m.parts[0] if len(m.parts) > 1 else 1)
                                   * d_ratio.get(pg, 1))
                            self.tmix[(pid, z, fs, b)] = (m, list(acc / acc.max()))
        ro = L.rn[L.rn["posteam"] == team]
        rd = L.rn[L.rn["defteam"] == opp]
        self.rmix = {}
        for pid in self.r_ids:
            pg = pos_group(L.pos.get(pid))
            base = L.rn_dist.get(pg, L.rn_dist_all)
            o_ratio = ratio(ro[ro["pg"] == pg], base, 150)
            dr = ratio(rd[rd["pg"] == pg], base, 150)
            for st in range(SITS):
                own = L.rn_rush.get((pid, st))
                for z in range(len(ZONES) + 1):
                    for fs in range(FSITS):
                        if fsit_to_sit(fs) != st:
                            continue
                        lgp = L.rn_lg.get((pg, z, fs)) or L.rn_lg_any.get((z, fs)) or L.rn_lg_sit[fs]
                        m = Mix([(own, 0.6, 25)], lgp)
                        acc = o_ratio ** (1 - m.parts[0] if len(m.parts) > 1 else 1) * dr
                        self.rmix[(pid, z, fs)] = (m, list(acc / acc.max()))
        self.smix = Mix([(L.scr_qb.get(qb), 0.6, 15)], L.scr_lg)

    def pick(self, ids, cum, r):
        return ids[min(bisect.bisect_left(cum, r.random() * cum[-1]), len(ids) - 1)]


# --------------------------------------------------------------------------- the game

STAT = ("pass_att", "cmp", "pass_yds", "pass_td", "int", "sacked", "rush_att", "rush_yds",
        "rush_td", "tgt", "rec", "rec_yds", "rec_td")
SI = {k: i for i, k in enumerate(STAT)}


def sim_game(L, offs, seed, log=None):
    """offs = (home Offense, away Offense). Returns the box score of one simulated game."""
    r = random.Random(seed)
    score = [0, 0]
    tstat = [[0.0, 0.0, 0, 0], [0.0, 0.0, 0, 0]]  # pass yds (net of sacks), rush yds, turnovers, plays
    pstat = {}

    def ps(pid):
        a = pstat.get(pid)
        if a is None:
            a = pstat[pid] = [0] * len(STAT)
        return a

    def kickoff_yl():
        nonlocal clock
        clock -= 5
        return L.ko_start[r.randrange(len(L.ko_start))]

    first_recv = r.randrange(2)
    qtr, clock = 1, 900.0
    pos, yl = first_recv, kickoff_yl()
    down, togo = 1, min(10.0, yl)
    drive_no = 0
    ot_poss = [0, 0]

    def note(kind, text, yards=0.0):
        if log is not None:
            log.append({"q": qtr, "t": max(0, round(clock)), "pos": pos, "down": down,
                        "togo": round(togo), "yl": round(yl), "kind": kind, "text": text,
                        "yds": round(yards), "score": list(score), "drive": drive_no})

    while True:
        # ---- quarters
        if clock <= 0:
            if qtr in (1, 3):
                qtr, clock = qtr + 1, 900.0
                continue
            if qtr == 2 or (qtr == 4 and score[0] == score[1]):
                qtr, clock = (3, 900.0) if qtr == 2 else (5, 600.0)
                pos = 1 - first_recv if qtr == 3 else r.randrange(2)
                yl = kickoff_yl()
                down, togo, drive_no = 1, min(10.0, yl), drive_no + 1
                if qtr == 5:
                    ot_poss[pos] += 1
                continue
            break

        off = offs[pos]
        sd = score[pos] - score[1 - pos]
        late = qtr >= 4
        cstate = clock_state(qtr, clock if qtr in (2, 4, 5) else clock + 900, sd)

        if (late and sd > 0 and down < 4 and clock <= 40 * (4 - down) + 2) or \
           (qtr == 2 and clock <= 25 and yl >= 55):
            action = "kneel"
        elif down == 4:
            action = fourth_down(L, r, yl, togo, sd, qtr, clock)
        elif clock <= 8 and yl <= 40 and (qtr == 2 or (late and -3 <= sd <= 0)):
            action = "fg"
        else:
            action = "snap"

        new_pos = None      # (team, yardline) when possession changes
        if action == "kneel":
            note("kneel", "kneel")
            clock -= 40 if late else clock
            yl += 1
            down += 1
            togo += 1
            continue

        if action == "fg":
            made = r.random() < L.p_fg(yl)
            note("fg", f"{int(yl + 18)}-yd FG {'good' if made else 'no good'}")
            clock -= 4
            if made:
                score[pos] += 3
                new_pos = (1 - pos, kickoff_yl())
            else:
                new_pos = (1 - pos, 80.0 if yl + 7 <= 20 else 100.0 - (yl + 7))

        elif action == "punt":
            opts = L.punts.get(int(yl // 5)) or L.punts[max(L.punts)]
            net, tb = opts[r.randrange(len(opts))]
            note("punt", f"punt, net {int(net)}")
            clock -= 9
            if r.random() < L.p_punt_td:
                score[1 - pos] += 7
                note("td", "punt returned for a TOUCHDOWN")
                new_pos = (pos, kickoff_yl())
            else:
                new_pos = (1 - pos, 80.0 if tb else max(1.0, min(99.0, 100 - yl + net)))

        elif r.random() < L.p_pen:
            # Accepted penalty, the play does not count.
            on_off, pyds, auto, el = L.pen[r.randrange(len(L.pen))]
            clock -= el
            if on_off:
                pyds = (100 - yl) / 2 if yl + pyds > 99 else pyds
                note("pen", f"offensive penalty, {int(pyds)} yds", -pyds)
                yl += pyds
                togo += pyds
            else:
                pyds = yl / 2 if pyds >= yl else pyds
                note("pen", f"defensive penalty, {int(pyds)} yds", pyds)
                yl -= pyds
                togo -= pyds
                if auto or togo <= 0:
                    down, togo = 1, min(10.0, yl)
            continue

        else:
            # ---- a scrimmage snap
            tstat[pos][3] += 1
            z, st = zone(yl), fsit(down, togo)
            gained, fumble, kind, ball = 0.0, False, "inc", None
            q = ps(off.qb) if off.qb else None
            if r.random() < L.p_pass(off.team, down, togo, yl, sd, qtr, clock):
                b = 1 if r.random() < off.blitz else 0
                c = DROPBACK_CLASSES[min(bisect.bisect_left(off.cls[(b, z, st)], r.random()), 3)]
                qb_name = off.names.get(off.qb, "QB")
                if c == "sack":
                    y, _, _, f, _, _ = L.S[L.sack_pool.draw(r)]
                    gained, fumble, kind = y, bool(f), "sack"
                    if q:
                        q[SI["sacked"]] += 1
                    tstat[pos][0] += y
                    note("sack", f"{qb_name} sacked" + (" on a blitz" if b else ""), y)
                elif c == "scramble":
                    y, _, _, f, _, td = L.S[off.smix.draw(r)]
                    gained, fumble, kind, ball = reach(y, td, yl, z), bool(f), "run", off.qb
                    if q:
                        q[SI["rush_att"]] += 1
                        q[SI["rush_yds"]] += gained
                    tstat[pos][1] += gained
                    note("scramble", f"{qb_name} scrambles", gained)
                elif c == "throwaway" or not off.t_ids:
                    if q:
                        q[SI["pass_att"]] += 1
                    note("inc", f"{qb_name} throws it away")
                else:
                    pid = off.pick(off.t_ids, off.t_cum, r)
                    y, comp, it, f, air, td = L.T[draw(off.tmix[(pid, z, st, b)], L.T_bin, r)]
                    rec = ps(pid)
                    rec[SI["tgt"]] += 1
                    if q:
                        q[SI["pass_att"]] += 1
                    nm = off.names.get(pid, "?")
                    tag = " vs the blitz" if b else ""
                    if it:
                        if q:
                            q[SI["int"]] += 1
                        tstat[pos][2] += 1
                        note("int", f"{qb_name} intercepted, intended for {nm}{tag}")
                        spot = yl - air
                        if r.random() < L.p_int_td:
                            score[1 - pos] += 7
                            note("td", "pick-six")
                            new_pos = (pos, kickoff_yl())
                        else:
                            new_pos = (1 - pos, 80.0 if spot <= 0 else max(1.0, 100.0 - spot))
                    elif comp:
                        gained, fumble, kind, ball = reach(y, td, yl, z), bool(f), "comp", pid
                        rec[SI["rec"]] += 1
                        rec[SI["rec_yds"]] += gained
                        if q:
                            q[SI["cmp"]] += 1
                            q[SI["pass_yds"]] += gained
                        tstat[pos][0] += gained
                        note("comp", f"{qb_name} to {nm}{tag}", gained)
                    else:
                        note("inc", f"{qb_name} incomplete to {nm}{tag}")
            elif off.r_ids:
                pid = off.pick(off.r_ids, off.r_cum, r)
                y, _, _, f, _, td = L.R[draw(off.rmix[(pid, z, st)], L.R_bin, r)]
                gained, fumble, kind, ball = reach(y, td, yl, z), bool(f), "run", pid
                a = ps(pid)
                a[SI["rush_att"]] += 1
                a[SI["rush_yds"]] += gained
                tstat[pos][1] += gained
                note("run", f"{off.names.get(pid, '?')} run", gained)

            pool = L.clock.get((cstate, kind)) or L.clock[("normal", kind)]
            clock -= pool[r.randrange(len(pool))]

            if new_pos is None:
                if fumble and gained < yl:
                    tstat[pos][2] += 1
                    note("fum", "fumble, lost")
                    if r.random() < L.p_fum_td:
                        score[1 - pos] += 7
                        note("td", "fumble returned for a TOUCHDOWN")
                        new_pos = (pos, kickoff_yl())
                    else:
                        new_pos = (1 - pos, max(1.0, min(99.0, 100 - (yl - gained))))
                else:
                    yl -= gained
                    if yl <= 0:
                        if kind == "comp":
                            ps(ball)[SI["rec_td"]] += 1
                            if q:
                                q[SI["pass_td"]] += 1
                        elif ball:
                            ps(ball)[SI["rush_td"]] += 1
                        go2 = late and (sd + 6) in (-2, -5, -10, 1, 5) and clock < 600
                        pts = 6 + ((2 if r.random() < L.p_2pt else 0) if go2
                                   else (1 if r.random() < L.p_xp else 0))
                        score[pos] += pts
                        note("td", f"TOUCHDOWN (+{pts})")
                        new_pos = (1 - pos, kickoff_yl())
                    elif yl >= 100:
                        score[1 - pos] += 2
                        note("safety", "safety")
                        new_pos = (1 - pos, 60.0)
                    else:
                        if gained >= togo:
                            down, togo = 1, min(10.0, yl)
                        else:
                            down += 1
                            togo -= gained
                        if r.random() < L.p_live_pen:
                            on_off, pyds, auto = L.live_pen[r.randrange(len(L.live_pen))]
                            if on_off:
                                pyds = (100 - yl) / 2 if yl + pyds > 99 else pyds
                                yl += pyds
                                togo += pyds
                            else:
                                pyds = yl / 2 if pyds >= yl else pyds
                                yl -= pyds
                                togo -= pyds
                                if auto or togo <= 0:
                                    down, togo = 1, min(10.0, yl)
                            note("pen", f"{'offensive' if on_off else 'defensive'} penalty after "
                                        f"the play, {int(pyds)} yds", pyds * (-1 if on_off else 1))
                        if down > 4:
                            note("downs", "turnover on downs")
                            new_pos = (1 - pos, 100.0 - yl)

        if new_pos is not None:
            # Overtime: once both teams have had the ball, the first lead wins.
            if qtr == 5 and min(ot_poss) >= 1 and score[0] != score[1]:
                break
            pos, yl = new_pos
            down, togo, drive_no = 1, min(10.0, yl), drive_no + 1
            if qtr == 5:
                ot_poss[pos] += 1

    return score, tstat, pstat


def draw(mix_acc, bins, r):
    mix, acc = mix_acc
    for _ in range(50):
        i = mix.draw(r)
        if r.random() < acc[bins[i]]:
            return i
    return i


def reach(y, tdz, yl, z):
    """Yards a resampled play gains from here. tdz packs (scored, zone it came from).
    When the play came from the same part of the red zone, whether it scored carries
    over: a goal-line TD from the 1 is still a TD from the 4, and a stop short of the
    line stays short. A play from elsewhere on the field just gains its yards."""
    if z and tdz % 10 == z:
        return yl if tdz >= 10 else min(y, yl - 1)
    return min(y, yl)


def clock_state(qtr, half_secs, sd):
    if half_secs <= 120 or (qtr >= 4 and half_secs <= 300 and sd < 0):
        return "hurry"
    if qtr >= 4 and half_secs <= 480 and sd > 0:
        return "milk"
    return "normal"


def fourth_down(L, r, yl, togo, sd, qtr, clock):
    fg_ok = yl <= 38
    if qtr >= 4 and clock <= 300:
        if sd < 0:
            if sd >= -3 and fg_ok and (clock <= 120 or togo > 3):
                return "fg"
            if clock <= 120 or sd < -8 or togo <= 2:
                return "go"
        elif fg_ok:
            return "fg"
    go, fg = L.fourth.get((bucket(TOGO_BUCKETS, togo), bucket(YL_BUCKETS, yl)), (0.1, 0.0))
    if not fg_ok:
        fg = 0.0 if yl > 45 else fg
    u = r.random()
    if u < go:
        return "go"
    if u < go + fg:
        return "fg"
    return "punt"


# --------------------------------------------------------------------------- many games

def run_game(L, D, game, n, seed=7, use_roster=False, keep_typical=True):
    home, away = game["home_team"], game["away_team"]
    offs = (Offense(L, D, home, away, L.week, use_roster),
            Offense(L, D, away, home, L.week, use_roster))
    players = sorted({p for o in offs for p in (o.t_ids + o.r_ids + ([o.qb] if o.qb else []))})
    pidx = {p: i for i, p in enumerate(players)}
    pts = np.zeros((n, 2))
    team = np.zeros((n, 2, 4))
    pl = np.zeros((n, len(players), len(STAT)))
    for i in range(n):
        s, t, p = sim_game(L, offs, seed + i)
        pts[i] = s
        team[i] = t
        for pid, a in p.items():
            j = pidx.get(pid)
            if j is not None:
                pl[i, j] = a

    margin = pts[:, 0] - pts[:, 1]
    total = pts.sum(1)
    feats = np.column_stack([pts * 2, team[:, :, 0], team[:, :, 1], team[:, :, 2] * 30])
    med = np.median(feats, 0)
    sdv = feats.std(0) + 1e-9
    typical = int(np.argmin((((feats - med) / sdv) ** 2).sum(1)))

    out = {
        "game_id": game["game_id"], "home": home, "away": away, "sims": n, "week": L.week,
        "kickoff": f"{game.get('weekday', '')} {game.get('gametime', '')}".strip(),
        "scores": pts.astype(int).ravel().tolist(),  # home, away, home, away, ...
        "home_win": float((margin > 0).mean() + 0.5 * (margin == 0).mean()),
        "home_pts": float(pts[:, 0].mean()), "away_pts": float(pts[:, 1].mean()),
        "margin_mean": float(margin.mean()), "margin_median": float(np.median(margin)),
        "total_mean": float(total.mean()), "total_median": float(np.median(total)),
        "qb": {"home": offs[0].names.get(offs[0].qb), "away": offs[1].names.get(offs[1].qb)},
        "missing": {side: sorted(o.names.get(p, p) for p in o.missing)
                    for side, o in (("home", offs[0]), ("away", offs[1]))},
    }
    if pd.notna(game.get("spread_line")):
        # nflverse spread_line is the expected home margin.
        out["market"] = {"spread_home": float(game["spread_line"]), "total": float(game["total_line"]),
                         "home_ml": game.get("home_moneyline"), "away_ml": game.get("away_moneyline")}
        out["p_home_cover"] = float((margin > game["spread_line"]).mean())
        out["p_over"] = float((total > game["total_line"]).mean())
        out["market"]["home_win"] = devig(game.get("home_moneyline"), game.get("away_moneyline"))
    props = []
    for p in players:
        j = pidx[p]
        a = pl[:, j]
        name = offs[0].names.get(p) or offs[1].names.get(p) or p
        tm = home if (p in offs[0].tshare or p in offs[0].rshare or p == offs[0].qb) else away
        row = {"id": p, "name": name, "team": tm, "pos": L.pos.get(p)}
        for k in ("pass_yds", "rush_yds", "rec_yds", "rec", "pass_td"):
            v = a[:, SI[k]]
            if v.max() > 0:
                row[k] = [float(np.percentile(v, q)) for q in (10, 50, 90)] + [float(v.mean())]
        row["any_td"] = float(((a[:, SI["rush_td"]] + a[:, SI["rec_td"]]) > 0).mean())
        props.append(row)
    out["players"] = props

    if keep_typical:
        log = []
        s, t, _ = sim_game(L, offs, seed + typical, log=log)
        out["typical"] = {"sim": typical, "score": {"home": s[0], "away": s[1]},
                          "plays": log, "team_stats": {"home": t[0], "away": t[1]}}
    return out


def devig(h, a):
    def imp(m):
        m = float(m)
        return 100 / (m + 100) if m > 0 else -m / (-m + 100)
    try:
        x, y = imp(h), imp(a)
        return x / (x + y)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="directory with the nflverse files")
    ap.add_argument("--week", type=int, help="2026 week to simulate")
    ap.add_argument("--game", help="one game_id, e.g. 2026_07_GB_DET (default: the week's unplayed games)")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--backtest", help="weeks to backtest on finished games, e.g. 3-7")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    D = load(args.data)
    games = D["games"][(D["games"]["season"] == SEASON) & (D["games"]["game_type"] == "REG")]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.backtest:
        lo, hi = (int(x) for x in args.backtest.split("-"))
        rows = []
        for wk in range(lo, hi + 1):
            L = League(D, wk)
            for _, g in games[(games["week"] == wk) & games["result"].notna()].iterrows():
                o = run_game(L, D, g, args.sims, keep_typical=False)
                rows.append({"game_id": g["game_id"], "week": wk, "sim_margin": o["margin_mean"],
                             "sim_total": o["total_mean"], "sim_home_win": o["home_win"],
                             "mkt_margin": g["spread_line"], "mkt_total": g["total_line"],
                             "mkt_home_win": o["market"]["home_win"], "margin": g["result"],
                             "total": g["total"]})
                print(f"wk{wk} {g['game_id']:>18}  sim {o['margin_mean']:+5.1f} / {o['total_mean']:5.1f}"
                      f"  mkt {g['spread_line']:+5.1f} / {g['total_line']:5.1f}"
                      f"  final {g['result']:+4.0f} / {g['total']:3.0f}", flush=True)
        bt = pd.DataFrame(rows)
        bt.to_csv(outdir / "backtest.csv", index=False)
        print(summarize_backtest(bt))
        (outdir / "backtest.json").write_text(json.dumps(summarize_backtest(bt, as_dict=True), indent=1))
        return

    L = League(D, args.week)
    wk = games[games["week"] == args.week]
    if args.game:
        wk = wk[wk["game_id"] == args.game]
    else:
        wk = wk[wk["result"].isna()]
    index = []
    for _, g in wk.iterrows():
        o = run_game(L, D, g, args.sims, use_roster=True)
        (outdir / f"{g['game_id']}.json").write_text(json.dumps(o, default=float))
        m = o.get("market", {})
        index.append({k: o.get(k) for k in ("game_id", "home", "away", "home_win", "home_pts",
                                             "away_pts", "margin_mean", "total_mean", "p_home_cover",
                                             "p_over")} | {"market": m,
                     "typical": o["typical"]["score"]})
        print(f"{o['away']:>3} @ {o['home']:<3}  {o['away_pts']:4.1f}-{o['home_pts']:4.1f}  "
              f"home win {o['home_win']:.0%} (mkt {m.get('home_win') or 0:.0%})  "
              f"total {o['total_mean']:.1f} (mkt {m.get('total')})  "
              f"typical {o['typical']['score']['away']}-{o['typical']['score']['home']}", flush=True)
    (outdir / "index.json").write_text(json.dumps(index, default=float, indent=1))

    # One script file the viewer page loads, so it works from GitHub Pages or a disk.
    bt = Path(__file__).with_name("backtest.json")
    bundle = {"week": args.week, "season": SEASON, "sims": args.sims,
              "data_through_week": args.week - 1,
              "backtest": json.loads(bt.read_text()) if bt.exists() else None,
              "games": [json.loads((outdir / f"{g['game_id']}.json").read_text())
                        for _, g in wk.iterrows()]}
    for g in bundle["games"]:
        g["players"] = sorted(g["players"], key=involvement, reverse=True)[:16]
    (outdir / "sim-data.js").write_text(
        "window.SIM = " + json.dumps(bundle, default=float, separators=(",", ":")) + ";\n")


def involvement(p):
    return sum(p.get(k, [0, 0, 0, 0])[3] for k in ("pass_yds", "rush_yds", "rec_yds"))


def summarize_backtest(bt, as_dict=False):
    bt = bt.dropna(subset=["mkt_home_win"])
    win = (bt["margin"] > 0).astype(float) + 0.5 * (bt["margin"] == 0)
    d = {
        "games": int(len(bt)),
        "brier_sim": float(((bt["sim_home_win"] - win) ** 2).mean()),
        "brier_market": float(((bt["mkt_home_win"] - win) ** 2).mean()),
        "margin_mae_sim": float((bt["sim_margin"] - bt["margin"]).abs().mean()),
        "margin_mae_market": float((bt["mkt_margin"] - bt["margin"]).abs().mean()),
        "total_mae_sim": float((bt["sim_total"] - bt["total"]).abs().mean()),
        "total_mae_market": float((bt["mkt_total"] - bt["total"]).abs().mean()),
        "total_bias_sim": float((bt["sim_total"] - bt["total"]).mean()),
        "total_bias_market": float((bt["mkt_total"] - bt["total"]).mean()),
        "sim_vs_market_margin_corr": float(np.corrcoef(bt["sim_margin"], bt["mkt_margin"])[0, 1]),
    }
    side = np.sign(bt["sim_margin"] - bt["mkt_margin"])
    res = np.sign(bt["margin"] - bt["mkt_margin"])
    d["ats_w_l_p"] = [int(((side == res) & (res != 0)).sum()), int(((side != res) & (res != 0)).sum()),
                      int((res == 0).sum())]
    ou = np.sign(bt["sim_total"] - bt["mkt_total"])
    ro = np.sign(bt["total"] - bt["mkt_total"])
    d["ou_w_l_p"] = [int(((ou == ro) & (ro != 0)).sum()), int(((ou != ro) & (ro != 0)).sum()),
                     int((ro == 0).sum())]
    if as_dict:
        return d
    return "\n".join(f"{k:>28}: {v}" for k, v in d.items())


if __name__ == "__main__":
    main()
