# Calibrated Striker Shortlist

**📊 [Live cross-league report →](https://sharia143.github.io/striker-preselect/outputs/comparison_report.html)**

Football clubs increasingly make expensive recruitment decisions on model outputs, and the industry's own named barrier is explainability/trust. This project builds a deliberately simple striker-shortlist model whose headline feature is **calibrated confidence**: it outputs `recommend` / `review` / `insufficient-evidence` instead of false certainty.

## Data

- **Source**: StatsBomb open data via `statsbombpy`
- **Competition**: La Liga 2015/16 (380 matches — full season)
- **Why this season**: It is one of only three *complete* Big-5 league seasons in StatsBomb open data (La Liga, Premier League and Serie A 2015/16 — each 380 matches, all 20 teams). Every "newer" free season is a **single featured club's fixtures**, not a full league (see [Cross-league comparison](#cross-league-comparison--latest-data-spotlight)), so a league-wide shortlist can only be built on 2015/16.
- **Scope**: 140 forwards/wingers identified by primary position (Center Forward, Left/Right Center Forward, Left/Right Wing).

### Player-season metrics (per 90 minutes)

| Metric | Description |
|--------|-------------|
| shots_p90 | Total shots |
| npxg_p90 | Non-penalty expected goals |
| touches_in_box_p90 | Touches in the opposition penalty area |
| key_passes_p90 | Passes directly leading to a shot |
| pressures_p90 | Defensive pressures applied |
| npga_p90 | Non-penalty goals + assists (target basis) |

### How each raw stat is derived from events

Event-level aggregation (`ingest.py`), computed per player per match, then rolled up to per-90 (above):

| Field | Definition (from StatsBomb events) |
|-------|------------------------------------|
| shots | count of `type == Shot` |
| xg | sum of `shot_statsbomb_xg` |
| goals | shots with `shot_outcome == Goal` |
| np_goals | goals excluding `shot_type == Penalty` |
| npxg | sum of `shot_statsbomb_xg` excluding penalties |
| touches_in_box | count of events with pitch location **`x >= 102` and `18 <= y <= 62`** (opposition box on the 120×80 pitch) |
| key_passes | count of `pass_shot_assist == True` — passes leading to a shot; goal assists are counted separately, not double-counted |
| assists | count of `pass_goal_assist == True` |
| pressures | count of `type == Pressure` |
| np_goals_assists | np_goals + assists |
| minutes | derived from the Starting XI event + Substitution events + match end (must read the *full* event stream — Starting XI / Half End rows carry no `player_id`) |
| primary_position | season-level **mode** of the player's `position` across all events (one label per player) |

This aggregation reproduces the cached La Liga tables **exactly** — all 2,946 player-match and 140 player-season rows, every column — verified by `validate_ingest.py`.

## Method

### SQL step
Player-match records are loaded into SQLite (`data/players.db`), and the player-season table is produced via the aggregation query in `sql/aggregate.sql`. `players.db` is gitignored; `build_db.py` rebuilds it from the committed `data/player_match_stats.csv` so the repo is clone-and-run.

### Model
- **Labelled set**: 109 forwards with >= 450 minutes (approximately 5 full matches).
- **Target**: top-quartile npGA/90 (threshold: 0.559 per 90).
- **Features**: shots_p90, npxg_p90, touches_in_box_p90, key_passes_p90, pressures_p90.
- **Models compared**: Logistic Regression vs Gradient Boosting (100 trees, depth 3).
- **Evaluation**: 5-fold stratified cross-validation, Brier score.

### Results

| Model | Brier Score |
|-------|-------------|
| Logistic Regression | 0.137 |
| Gradient Boosting | 0.168 |

**Logistic Regression wins** — better calibrated, simpler, more interpretable. This is expected: with 109 samples and 5 features, a linear model generalises better than a flexible one.

### Calibration & Abstention Band

The model outputs a probability of being top-quartile. The final decision column uses three bands:

| Decision | Rule | Justification |
|----------|------|---------------|
| **recommend** | P >= 0.60 | High confidence the player is elite. 0.60 chosen because the calibration curve shows good reliability above this threshold. |
| **review** | 0.25 <= P < 0.60 | Model is uncertain — scout should investigate further. This band captures players where the model probability sits in the steepest part of the calibration curve. |
| **not recommended** | P < 0.25 | Low probability of top-quartile performance. |
| **insufficient evidence** | < 450 minutes | Small sample makes per-90 rates unstable. Wide confidence intervals mean any prediction would be unreliable. We abstain rather than guess. |

### Decision distribution

- Recommend: 14 players
- Review: 24 players
- Not recommended: 71 players
- Insufficient evidence: 31 players

## Cross-league comparison & latest-data spotlight

The single-league model was extended to **all three complete 2015/16 leagues** and to the
**freshest data StatsBomb releases for free**. The ingestion layer (`ingest.py`) reproduces the
original cached La Liga tables *exactly* (verified in `validate_ingest.py`, all 2,946 player-match
and 140 player-season rows), so every league is aggregated by identical logic. `compare.py`
produces the comparison; findings were independently recomputed and adversarially stress-tested.

### Data-coverage reality

| Dataset | Coverage | Use |
|---------|----------|-----|
| La Liga 2015/16 | 380 matches, 20 teams | Full league |
| Premier League 2015/16 | 380 matches, 20 teams | Full league |
| Serie A 2015/16 | 380 matches, 20 teams | Full league |
| Bundesliga **2023/24** | **Bayer Leverkusen only** (34 fixtures) | Spotlight |
| Ligue 1 **2022/23** | **PSG only** (32 fixtures) | Spotlight |

The "latest" free seasons contain only one featured club's fixtures (with opponents), so they
support a *modern-stars spotlight*, not a league shortlist. (The `data/player_season_leverkusen_2324.csv`
/ `_psg_2223.csv` files therefore contain all teams from that club's match set; `compare.py` filters
to the club at scoring time.)

### 1 — The recipe replicates across leagues

| League | forwards | qualified (≥450') | top-quartile bar (npGA/90) | Brier LR | Brier GB | AUC (LR) |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| La Liga | 140 | 109 | 0.559 | **0.137** | 0.168 | 0.775 |
| Premier League | 149 | 98 | 0.504 | **0.118** | 0.223 | 0.852 |
| Serie A | 127 | 95 | 0.505 | **0.170** | 0.181 | 0.736 |

Logistic regression beats gradient boosting on calibration in **all three** leagues (robust across
50 CV seeds), confirming the "simple model" choice generalises. *Caveat:* GBM is a weak comparator at
n≈100 — "LR wins" argues for simplicity, not that any model is strongly calibrated (Serie A's Brier is
poor in absolute terms). The top-quartile bar is highest in La Liga, but with ~100 players/league the
bars' confidence intervals overlap heavily — it is a property of these samples (the Messi/Suárez/Ronaldo
cluster pulling La Liga's top up), not proof that La Liga was a harder league. PL vs Serie A (0.504 vs
0.505) is a dead heat.

### 2 — The ranking transfers; the decision threshold does not

Train on La Liga only, then score another league (La Liga's scaler + LR applied to that league's forwards):

| Test league | in-league AUC | transfer AUC | recall of that league's top-quartile at P ≥ 0.60 |
|-------------|:---:|:---:|:---:|
| Premier League | 0.852 | 0.837 | **32%** (8/25) |
| Serie A | 0.736 | 0.776 | **8%** (2/24) |

A La-Liga-only model **ranks** Premier League strikers almost as well as a PL-native model. (For Serie A
the transferred and native models are statistically indistinguishable — both weak and noisy at n=95 — so
this is *not* evidence transfer is "better".) But at the fixed 0.60 "recommend" cutoff the La Liga model
flags only 8% of Serie A's own top-quartile strikers. **The ordering travels; the calibrated cutoff must be
re-fit per league** — a concrete, quantified version of the "calibrated to one season/league only"
limitation.

*Why it breaks:* the model is npxG-dominated (coef +1.37) and slightly **penalises** raw shot volume
(−0.20). Serie A's top-quartile scorers actually take *more* shots and get *more* box touches than La
Liga's, but post lower npxG (0.33 vs 0.40) — they finish above expectation. An npxG-driven model therefore
under-rates them. (The 0.60 band is also conservative *everywhere* — it flags only 46% of La Liga's *own*
top-quartile — so it is precision-over-recall by design.)

### 3 — "Elite" has a different fingerprint per league

Standardised LR coefficients (bigger = more predictive of top-quartile output):

| League | shots | npxG | touches-in-box | key passes | pressures |
|--------|:---:|:---:|:---:|:---:|:---:|
| La Liga | −0.20 | **1.37** | 0.30 | 0.47 | −0.17 |
| Premier League | 0.67 | 0.76 | 0.50 | **0.97** | −0.34 |
| Serie A | 0.24 | **0.83** | 0.14 | 0.19 | −0.09 |

npxG predicts elite output everywhere. Beyond that: La Liga rewards chance *quality*, the Premier League
uniquely rewards *creation* (key passes — its largest coefficient anywhere), Serie A rewards *efficient
finishing*. Pressures are negative everywhere (elite scorers press less). Same small-sample caveat applies.

### 4 — Modern-stars spotlight (illustrative)

The 2015/16 La Liga model, applied to modern single-club squads:

| Player | Squad | Model P | Decision | actual npGA/90 |
|--------|-------|:---:|:---:|:---:|
| Kylian Mbappé | PSG 2022/23 | 0.98 | recommend | 1.13 |
| Victor Boniface | Leverkusen 2023/24 | 0.95 | recommend | 1.02 |
| Lionel Messi | PSG 2022/23 | 0.78 | recommend | 0.92 |
| Patrik Schick | Leverkusen 2023/24 | 0.45 | review | 0.49 |
| Hugo Ekitike | PSG 2022/23 | 0.14 | not recommended | 0.66 |

The model still cleanly identifies Mbappé and Boniface as elite and (fairly) rejects PSG misfit Ekitike.
*Illustrative, not evaluative:* single-club squads yield ≤3 qualifying forwards each, and the
striker-position filter drops players whose season-modal position falls outside the 5 forward buckets —
e.g. Florian Wirtz (an attacking midfielder) and Neymar (logged across wing/attacking-mid/midfield roles
that season).

### Reproduce

```bash
.\venv\Scripts\python.exe validate_ingest.py                          # prove ingestion is faithful
.\venv\Scripts\python.exe ingest.py --comp 2 --season 27 --label premier_league_1516 --matchcsv
.\venv\Scripts\python.exe ingest.py --comp 12 --season 27 --label serie_a_1516 --matchcsv
.\venv\Scripts\python.exe ingest.py --comp 9  --season 281 --label leverkusen_2324 --matchcsv
.\venv\Scripts\python.exe ingest.py --comp 7  --season 235 --label psg_2223 --matchcsv
.\venv\Scripts\python.exe compare.py                                  # build the comparison
```

## Outputs

- `outputs/shortlist.csv` — full ranked list with probabilities and decisions
- `outputs/powerbi_export.csv` — same data, tidy format for Power BI
- `outputs/calibration_curve.png` — reliability diagram comparing both models
- `outputs/top_shortlist.png` — horizontal bar chart of recommended players

**Cross-league comparison** (`compare.py`):
- `outputs/comparison_summary.csv` — per-league metrics, bars and decision counts
- `outputs/cross_league_transfer.csv` — La Liga model scored on the other leagues
- `outputs/league_coefficients.csv` — standardised LR coefficients per league
- `outputs/shortlist_{laliga,premier_league,serie_a}.csv` — per-league shortlists
- `outputs/spotlight_modern.csv` — modern stars scored by the La Liga model
- `outputs/comparison_metrics.png`, `comparison_decisions.png`, `league_coefficients.png`, `transfer_calibration.png`, `spotlight_modern.png`

## How to run

```bash
# Requires Python 3.10+ with venv already set up
.\venv\Scripts\python.exe build_db.py   # build data/players.db from committed CSVs (first run / fresh clone)
.\venv\Scripts\python.exe model.py      # single-league model + shortlist
```

`build_db.py` rebuilds the SQLite database from the version-controlled player-match CSV (no StatsBomb download needed). The raw events pickle and `players.db` are gitignored; everything the model needs is regenerated from the committed CSVs.

## Limitations

0. **Descriptive, not predictive (the important one)**: the target (`npga_p90`) and a core feature
   (`npxg_p90`) are both same-season output measures and correlate ≈0.72 — npGA is essentially realised
   npxG. The model therefore partly re-describes a cousin of its own target, which inflates in-league
   AUC/Brier and explains why npxG dominates the coefficients. It characterises the *profile* of a
   high-output striker within a season; it does **not** forecast future or transfer-market value. Read it
   as description, not talent identification.

1. **Small samples**: 95–109 qualifying forwards and only 24–28 "elite" per league, so every statistic
   (AUC, Brier, coefficients, the quartile bars, and especially the cross-league gaps) carries wide
   confidence intervals. Most cross-league differences are within sampling/CV noise — treat them as
   directional, not definitive. The cross-league *comparison* mitigates the old "single season" concern
   but does not remove the small-n one.

2. **Low minutes floor**: the 450-minute threshold lets small-sample per-90 leaders (e.g. Gabbiadini, 621')
   outrank higher-volume proven scorers (Icardi, Immobile). The model rewards per-90 *rate* over volume.

3. **Weak baseline**: gradient boosting is a straw comparator at n≈100, so "logistic regression wins"
   argues for simplicity, not for strong absolute performance.

4. **Proxy target**: "Top-quartile npGA/90" is a within-sample label, not a ground-truth recruitment outcome. A player can be top-quartile in a weak league or season without being worth signing.

3. **Survivorship of minutes**: The 450-minute threshold filters out injured players and youth prospects who might be excellent but lack data. The model literally cannot evaluate them — hence the "insufficient evidence" label.

4. **Open-data scope**: StatsBomb open data is free but limited. Event tagging quality and coverage may differ from their commercial product. Some metrics (e.g., progressive carries, xA) are unavailable in this dataset version.

5. **No contextual features**: Team strength, opponent quality, league competitiveness, and age are not included. A striker scoring 0.8 npGA/90 for a relegation side is more impressive than the same rate for Barcelona.

6. **Position classification**: Players are assigned a single primary position based on mode across all events. A player who shifts between wing and striker mid-season gets only one label.

## Project structure

```
├── data/
│   ├── events_laliga_1516.pkl        # Cached raw events (La Liga)
│   ├── player_match_stats.csv        # La Liga player-match aggregates
│   ├── player_season_stats.csv       # La Liga player-season aggregates
│   ├── player_season_*.csv           # Other leagues / spotlights (via ingest.py)
│   └── players.db                    # SQLite database
├── sql/
│   └── aggregate.sql                 # Season-level aggregation query
├── outputs/                          # shortlists, comparison CSVs + charts
├── build_db.py                       # Rebuild data/players.db from committed CSVs (clone-and-run)
├── ingest.py                         # StatsBomb events → player-match → season (any comp/season)
├── validate_ingest.py                # Proves ingest.py reproduces the cached La Liga tables
├── model.py                          # Single-league model pipeline (La Liga)
├── compare.py                        # Cross-league comparison + modern spotlight
└── README.md
```
