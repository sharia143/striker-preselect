"""
Build data/players.db from the committed CSVs so the repo is clone-and-run.

`players.db` (and the raw events pickle) are gitignored, and model.py / compare.py
both read `player_season` from it. This script rebuilds the database from the
version-controlled player-match CSV: it loads player_match_stats.csv into the
`player_match` table, then (re)derives the `player_season` table by running the
documented aggregation in sql/aggregate.sql — so the SQL is exercised, not bypassed.

Usage:
    python build_db.py
"""
import pathlib
import sqlite3

import pandas as pd

DB = "data/players.db"
PLAYER_MATCH_CSV = "data/player_match_stats.csv"
PLAYER_SEASON_CSV = "data/player_season_stats.csv"  # reference, for the self-check
AGG_SQL = "sql/aggregate.sql"


def main():
    pm = pd.read_csv(PLAYER_MATCH_CSV)
    conn = sqlite3.connect(DB)

    # 1. load player-match rows
    pm.to_sql("player_match", conn, if_exists="replace", index=False)

    # 2. derive player-season via the actual aggregation query
    season = pd.read_sql_query(pathlib.Path(AGG_SQL).read_text(), conn)
    season.to_sql("player_season", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    print(f"Built {DB}")
    print(f"  player_match  rows: {len(pm)}")
    print(f"  player_season rows: {len(season)}")

    # 3. self-check against the committed reference CSV, if present
    ref_path = pathlib.Path(PLAYER_SEASON_CSV)
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        num_cols = ref.select_dtypes("number").columns
        merged = ref.merge(season, on="player_id", suffixes=("_ref", "_built"))
        ok = len(merged) == len(ref)
        for c in num_cols:
            if c == "player_id":
                continue
            ok &= bool(
                (merged[f"{c}_ref"] - merged[f"{c}_built"]).abs().max() < 1e-6
            )
        print(f"  self-check vs {PLAYER_SEASON_CSV}: {'PASS' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
