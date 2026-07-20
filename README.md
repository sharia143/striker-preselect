# Calibrated Striker Shortlist

Football clubs increasingly make expensive recruitment decisions on model outputs, and the industry's own named barrier is explainability/trust. This project builds a deliberately simple striker-shortlist model whose headline feature is **calibrated confidence**: it outputs `recommend` / `review` / `insufficient-evidence` instead of false certainty.

## Data

- **Source**: StatsBomb open data via `statsbombpy`
- **Competition**: La Liga 2015/16 (380 matches — full season)
- **Why this season**: It is the largest complete Big-5 league season in StatsBomb open data (Premier League and Serie A 2015/16 are also complete at 380 matches, but La Liga has the highest event-data quality due to StatsBomb's original coverage focus).
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

## Method

### SQL step
Player-match records are loaded into SQLite (`data/players.db`). The player-season table is produced via the aggregation query in `sql/aggregate.sql`.

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

## Outputs

- `outputs/shortlist.csv` — full ranked list with probabilities and decisions
- `outputs/powerbi_export.csv` — same data, tidy format for Power BI
- `outputs/calibration_curve.png` — reliability diagram comparing both models
- `outputs/top_shortlist.png` — horizontal bar chart of recommended players

## How to run

```bash
# Requires Python 3.10+ with venv already set up
.\venv\Scripts\python.exe model.py
```

Data is cached in `data/` after first pull — the model script reads from SQLite, so StatsBomb re-download is not needed for re-runs.

## Limitations

1. **Single season**: One season of La Liga provides ~109 qualifying forwards. This is too few for robust out-of-sample validation of calibration. Results would be more trustworthy pooled across multiple seasons.

2. **Proxy target**: "Top-quartile npGA/90" is a within-sample label, not a ground-truth recruitment outcome. A player can be top-quartile in a weak league or season without being worth signing.

3. **Survivorship of minutes**: The 450-minute threshold filters out injured players and youth prospects who might be excellent but lack data. The model literally cannot evaluate them — hence the "insufficient evidence" label.

4. **Open-data scope**: StatsBomb open data is free but limited. Event tagging quality and coverage may differ from their commercial product. Some metrics (e.g., progressive carries, xA) are unavailable in this dataset version.

5. **No contextual features**: Team strength, opponent quality, league competitiveness, and age are not included. A striker scoring 0.8 npGA/90 for a relegation side is more impressive than the same rate for Barcelona.

6. **Position classification**: Players are assigned a single primary position based on mode across all events. A player who shifts between wing and striker mid-season gets only one label.

## Project structure

```
├── data/
│   ├── events_laliga_1516.pkl    # Cached raw events
│   ├── matches_laliga_1516.csv   # Match metadata
│   ├── player_match_stats.csv    # Player-match aggregates
│   ├── player_season_stats.csv   # Player-season aggregates
│   └── players.db                # SQLite database
├── sql/
│   └── aggregate.sql             # Season-level aggregation query
├── outputs/
│   ├── shortlist.csv             # Final shortlist with decisions
│   ├── powerbi_export.csv        # Power BI-ready export
│   ├── calibration_curve.png     # Reliability diagram
│   └── top_shortlist.png         # Top recommendations chart
├── model.py                      # Main model pipeline
└── README.md
```
