# Snap sim: RotoBot's game simulator, reverse engineered

RotoBot (@rotobot_ai) posts NFL and NBA game simulations in a fixed format:

> We simulated Thursday Night's Seahawks @ Broncos game 2,000 times 🏈 Every snap is
> built from real 2026 play-by-play: the coverage each defense actually plays, and how
> each player actually performs against it. Here's the most typical game 👇

The NFL posts come with a video replay of one game. One also admits a bias: "Our sims
run light on scoring, so read the under as a lean."

This folder rebuilds that product from public data. It has no RotoBot code, data or API
in it.

## What could be reverse engineered, and how

RotoBot's simulator runs on its servers, and the app only shows the results. Decompiling
the iOS or Android app (for example with [REA](https://github.com/morluto/rea), which
handles APKs through JADX) would show API endpoints and screens, not the model. It would
also mean taking apart a commercial app against its terms. So this is a behavioural
reconstruction instead. It takes every claim the posts make, builds the simplest engine
that satisfies all of them, and then checks it against real games.

| RotoBot's claim | How it is implemented here |
| --- | --- |
| "Every snap is built from real 2026 play-by-play" | Each snap resamples a real play from nflverse play-by-play: 2026 weighted by recency, 2025 as a light prior. |
| "how each player actually performs" | The ball carrier's own real plays from a similar down and distance are the base draw. |
| "the coverage each defense actually plays" | Man/zone coverage is not public for 2026. FTN charting's blitz flag is the scheme variable instead: each defense blitzes at its own rate, and QBs are rated separately against blitz and no blitz. |
| "simulated 2,000 / 5,000 times" | `--sims N` independent games, each with its own seed. |
| "the most typical game" | The one simulation closest to the median of all of them on both scores, passing and rushing yards, and turnovers. It is replayed with the same seed to get the play-by-play. |

## How a snap is built (`sim.py`)

1. **Play call.** A logistic model of down, distance, field position, score and clock,
   fitted on every 2025–26 dropback, plus each offense's pass rate over expected. Fourth
   downs use the league's go/kick/punt frequencies by field position and distance, with
   overrides for the last five minutes.
2. **Defensive look.** Blitz or not, at the defense's shrunk blitz rate.
3. **Dropback.** Sack / scramble / throw-away / target. League rates for that look, field
   zone and down-distance, multiplied by the QB's and the defense's odds ratios.
4. **Outcome.** A real play drawn from the receiver's or rusher's own plays in the same
   down-distance group, shrunk toward league plays from the same field zone and exact
   distance band. The QB, the offensive line and the defense then re-weight it by
   rejection sampling. A drawn play is kept in proportion to how much each unit makes
   that kind of result (incomplete, short, medium, long, turnover) more or less likely
   than the league does.
5. **Everything else.** Clock run-off is drawn from real snaps in the same game state
   (normal, hurry-up, protecting a lead). Penalties come before and after the snap at
   real rates. Punts, kickoffs and field goals come from real distributions, plus return
   TDs and pick-sixes. Overtime follows the both-teams-possess rule.

Starters and target/carry shares come from the last three games, decayed by recency.
Players ruled Out or Doubtful on the week's injury report, or not active on the roster,
are removed.

## Calibration: what had to be fixed

The first version scored 40 points a game against 46 in real 2026 games, and its spreads
were half as wide as the market's. Finding out why is most of what reverse engineering a
simulator involves:

- **Averaging pools halves every team effect.** A 50/50 mix of "plays by this offense"
  and "plays against this defense" gives half of each team's edge. The sim's margins had
  a standard deviation of 1.9 points (the market's is 5.2), and actual results moved 1.9
  points per sim point. Switching to multiplicative re-weighting (log5 style) moved the
  standard deviation to 3.4 and the slope to 0.88.
- **Third downs need the distance to the sticks.** A 3rd-and-6 snap can't resample a
  3rd-and-15 play. League pools are split by exact distance band.
- **Red-zone plays need their scoring outcome.** A goal-line TD from the 1, replayed at
  the 4, used to come up short. Plays from the same part of the red zone keep whether
  they scored.
- **Clock.** Using each resampled play's own clock run-off double-counted hurry-up
  snaps, so the sim ran 136 plays a game against 124 real. The clock now comes from the
  game state.
- **Penalties on live plays** (roughness, face mask) were missing and cost about one
  touchdown every three games.

After those fixes, on a sample of 16 games from weeks 1–6, the sim averaged 124 plays a
game (real 124) and 4.92 offensive TD drives (real 4.95). Its red-zone TD rate was 63.5%
(real 63.4%) and third-down conversion 38% (real 40%).

## Backtest

Weeks 3–7 of 2026: 62 finished games, 400 sims each. Each week used only data from before
it (`python sim.py --backtest 3-7`, results in `backtest.csv` and `backtest.json`).

| | Sim | Closing market |
| --- | --- | --- |
| Win probability, Brier score | 0.232 | 0.235 |
| Margin, mean absolute miss | 7.7 | 8.0 |
| Total, mean absolute miss | 10.8 | 10.2 |
| Total, average bias | −3.0 | −1.8 |

Against the closing spread the sim's side went 35–22–5, and on totals 27–35. That many
games cannot separate skill from luck. The sim still runs about 3 points light on totals,
the same bias RotoBot admits to, so its unders are not a signal. It has no home-field
term, no weather, no explicit timeouts and no coverage data.

## What tuning found (October 31, 2026)

`tune.py` fits the sim's hand-set numbers on the 2025 season (240 games, weeks 3-18)
and scores the result once on 2026 (76 games, weeks 3-8). Each game is simulated
with only the data from before its week. Details: `validation.json`,
`tuning_results.json`.

| Season | Brier, sim (market) | Margin miss, sim (market) | ATS | Spread picks on the underdog |
| --- | --- | --- | --- | --- |
| 2025 | 0.235 (0.215) | 10.9 (10.1) | 119-120 | 85% |
| 2026 | 0.227 (0.234) | 8.1 (8.4) | 45-26 | 82% |

- **The sim does not beat the closing line.** When it disagreed with the market, the
  result moved against it in 2025 (-0.38 points per point of disagreement, t = -1.9)
  and with it in 2026 (+0.56, t = +1.4). Neither is significant and they point opposite
  ways. A blend fitted on 2025 puts no weight on the sim.
- **Its ATS record mostly measures underdogs.** Its margins are flatter than the
  market's, so its side is the underdog about 83% of the time. Underdogs covered in
  2026 and split in 2025, and so did the sim.
- **Tuning did not transfer.** Wider team effects, less prior-season weight, slower
  recency decay and less defensive shrinkage improved 2025 Brier from 0.235 to 0.224
  and made 2026 worse (0.227 to 0.236). A margin calibration, a home-field weight and
  the FTN play-action / stacked-box matchups also failed on 2026. None was adopted.
- **Kept:** timeouts and the two-minute warning, and a +1.68-point totals correction
  fitted on 2025 (`calibration.json`). On 2026 it cut the scoring bias from -2.0 to
  -0.3 and the totals miss from 10.56 to 10.39.

A larger test needs earlier seasons: `tune.py --train 2024` with the 2023 files.

## On the website, automatically

The page lives at [`/sim/`](../../sim/) on the site and is linked from the main
navigation. `.github/workflows/snap-sim.yml` runs `autorun.py`, commits what it wrote, and then stays up sleeping until the next check if that opens within about 5.5 hours. GitHub's own schedule is best effort: asked for every 15 minutes, it started this workflow every 3-7 hours in practice, which made early pregame checks late and missed two night games. Fixed game-day start times keep a run waiting before each window. The calendar:

| When | What runs |
| --- | --- |
| 8pm ET the night before each game day | Every game that day, 2,000 sims each, with data through the previous week. |
| 2 hours before each kickoff | That kickoff's games again, with the latest injury news. The page shows what changed since the night run: players newly ruled out, a starting-QB change, and how far the win probability and total moved. |
| Once every game in a week is final | The backtest re-runs through that week (each week simulated with only earlier data) and replaces the one on the page. |
| Once a final score is in | The last run from before kickoff is graded into `sim/data/record.json` and the season record on the page. |

Each simulation keeps its seed between runs, so a pregame run on unchanged information
reproduces the night run exactly. Any difference comes from new information.

Injuries come from nflverse's weekly report and roster status, plus ESPN's live injury
feed when it answers. Each game's page says which sources it used. Official inactives
are posted 90 minutes before kickoff, after the 2-hour check, so a late scratch announced
only with the inactives is not reflected. Change `PREGAME_HOURS` in `autorun.py` to move
the check (for example 1.25 to run after inactives).

To run a stage by hand: Actions → Snap sim → Run workflow, choose `night` or `pregame`.

## Run it locally

```sh
pip install -r requirements.txt
python autorun.py --data /tmp/nflverse --site ../../sim               # whatever is due now
python autorun.py --data /tmp/nflverse --site ../../sim --force night  # this week's open games now
python sim.py --data /tmp/nflverse --backtest 3-7 --sims 400 --out bt  # backtest
./fetch_data.sh /tmp/nflverse                                          # data for sim.py on its own
```

`autorun.py --now 2026-10-25T11:00-04:00` pretends it is a given time, which is how the
schedule is tested. Don't use it for runs that will be published: grading trusts the run
timestamp.

NFL data via [nflverse](https://nflverse.com), CC-BY 4.0. Not affiliated with RotoBot.
Not betting advice.
