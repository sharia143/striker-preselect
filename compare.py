"""
Cross-league striker-shortlist comparison.

Extends the single-league model (model.py) to three FULL 2015/16 leagues
(La Liga, Premier League, Serie A) plus a 'latest data' spotlight on single-team
modern squads (Bayer Leverkusen 2023/24, PSG 2022/23).

Three questions:
  1. Per-league — does the same recipe (LR vs GB, Brier-selected, banded) behave
     consistently across leagues? How does the scoring environment differ?
  2. Transfer — train on La Liga, score the OTHER leagues. Does an elite-striker
     model generalise across leagues, or does it need recalibration? (Directly
     tests the README limitation "calibrated to this season only".)
  3. Spotlight — where do modern stars land on the 2015/16-calibrated scale?

All config mirrors model.py so results are directly comparable.
"""
import sqlite3
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Config (identical to model.py) ──────────────────────────────────────────
MINUTES_THRESHOLD = 450
RECOMMEND_THRESHOLD = 0.60
REVIEW_LOWER = 0.25
FEATURES = ["shots_p90", "npxg_p90", "touches_in_box_p90", "key_passes_p90", "pressures_p90"]
RANDOM_STATE = 42

# ─── League sources ──────────────────────────────────────────────────────────
FULL_LEAGUES = {
    "La Liga 2015/16": ("db", None),
    "Premier League 2015/16": ("csv", "data/player_season_premier_league_1516.csv"),
    "Serie A 2015/16": ("csv", "data/player_season_serie_a_1516.csv"),
}
# Single-team open-data releases also contain opponents (only the featured club's
# players exceed the minutes threshold, but filter by team to be unambiguous).
SPOTLIGHTS = {
    "Leverkusen 2023/24": ("data/player_season_leverkusen_2324.csv", "Bayer Leverkusen"),
    "PSG 2022/23": ("data/player_season_psg_2223.csv", "Paris Saint-Germain"),
}


def load_league(source):
    kind, path = source
    if kind == "db":
        conn = sqlite3.connect("data/players.db")
        df = pd.read_sql_query("SELECT * FROM player_season", conn)
        conn.close()
        return df
    return pd.read_csv(path)


def prep(df):
    """Return qualified modelling frame + X, y, quartile threshold."""
    m = df[df["total_minutes"] >= MINUTES_THRESHOLD].copy()
    q75 = m["npga_p90"].quantile(0.75)
    m["target"] = (m["npga_p90"] >= q75).astype(int)
    return m, m[FEATURES].values, m["target"].values, q75


def cv_probs(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def band(p, minutes):
    if minutes < MINUTES_THRESHOLD:
        return "insufficient evidence"
    if p >= RECOMMEND_THRESHOLD:
        return "recommend"
    if p >= REVIEW_LOWER:
        return "review"
    return "not recommended"


def fit_league(name, df):
    """Full per-league fit mirroring model.py; returns a result bundle."""
    m, X, y, q75 = prep(df)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=RANDOM_STATE,
        learning_rate=0.1, min_samples_leaf=5,
    )
    lr_probs = cv_probs(lr, Xs, y)
    gb_probs = cv_probs(gb, Xs, y)
    lr_brier = brier_score_loss(y, lr_probs)
    gb_brier = brier_score_loss(y, gb_probs)
    lr_auc = roc_auc_score(y, lr_probs)
    gb_auc = roc_auc_score(y, gb_probs)

    chosen = "Logistic Regression" if lr_brier <= gb_brier else "Gradient Boosting"
    chosen_model = lr if chosen == "Logistic Regression" else gb
    chosen_probs = lr_probs if chosen == "Logistic Regression" else gb_probs

    # fit chosen on all league data for scoring / coefficients
    chosen_model.fit(Xs, y)
    lr_fit = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000).fit(Xs, y)

    m = m.copy()
    m["probability"] = chosen_probs  # honest out-of-fold probs for decisions
    m["decision"] = [band(p, mn) for p, mn in zip(m["probability"], m["total_minutes"])]

    # decisions across the FULL squad (incl. < 450 min)
    full = df.copy()
    below = full[full["total_minutes"] < MINUTES_THRESHOLD]

    dec_counts = m["decision"].value_counts().to_dict()
    dec_counts["insufficient evidence"] = int(len(below))

    return {
        "name": name,
        "frame": m,
        "scaler": scaler,
        "lr_fit": lr_fit,
        "chosen": chosen,
        "q75": q75,
        "n_forwards": len(full),
        "n_qualified": len(m),
        "n_positive": int(y.sum()),
        "lr_brier": lr_brier, "gb_brier": gb_brier,
        "lr_auc": lr_auc, "gb_auc": gb_auc,
        "decisions": dec_counts,
        "coefs": dict(zip(FEATURES, lr_fit.coef_[0])),
    }


def main():
    print("Loading leagues ...")
    results = {}
    for name, src in FULL_LEAGUES.items():
        df = load_league(src)
        results[name] = fit_league(name, df)
        r = results[name]
        print(f"  {name:24s} forwards={r['n_forwards']:3d} qualified={r['n_qualified']:3d} "
              f"q75={r['q75']:.3f} chosen={r['chosen']} Brier(LR)={r['lr_brier']:.3f}")

    # ── 1. comparison summary ──
    rows = []
    for name, r in results.items():
        d = r["decisions"]
        rows.append({
            "league": name,
            "forwards": r["n_forwards"],
            "qualified_>=450min": r["n_qualified"],
            "top_quartile_npga_p90": round(r["q75"], 3),
            "brier_lr": round(r["lr_brier"], 4),
            "brier_gb": round(r["gb_brier"], 4),
            "auc_lr": round(r["lr_auc"], 3),
            "auc_gb": round(r["gb_auc"], 3),
            "chosen_model": r["chosen"],
            "recommend": d.get("recommend", 0),
            "review": d.get("review", 0),
            "not_recommended": d.get("not recommended", 0),
            "insufficient_evidence": d.get("insufficient evidence", 0),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv("outputs/comparison_summary.csv", index=False, encoding="utf-8")
    print("\nSaved outputs/comparison_summary.csv")
    print(summary.to_string(index=False))

    # ── league LR coefficients ──
    coef_df = pd.DataFrame({name: r["coefs"] for name, r in results.items()}).T
    coef_df.index.name = "league"
    coef_df.round(3).to_csv("outputs/league_coefficients.csv", encoding="utf-8")

    # ── per-league shortlists ──
    for name, r in results.items():
        slug = name.split()[0].lower() if "La" not in name else "laliga"
        slug = {"La Liga 2015/16": "laliga", "Premier League 2015/16": "premier_league",
                "Serie A 2015/16": "serie_a"}[name]
        sl = r["frame"].sort_values("probability", ascending=False)
        cols = ["player", "team", "primary_position", "total_minutes", "probability",
                "decision", "npga_p90", "shots_p90", "npxg_p90", "touches_in_box_p90",
                "key_passes_p90", "pressures_p90"]
        sl[cols].round(4).to_csv(f"outputs/shortlist_{slug}.csv", index=False, encoding="utf-8")

    # ── 2. cross-league transfer (train La Liga → score others) ──
    base = results["La Liga 2015/16"]
    base_scaler, base_model = base["scaler"], base["lr_fit"]
    transfer_rows = []
    for name in ["Premier League 2015/16", "Serie A 2015/16"]:
        r = results[name]
        m = r["frame"]
        Xs_other = base_scaler.transform(m[FEATURES].values)
        p_transfer = base_model.predict_proba(Xs_other)[:, 1]
        y_other = m["target"].values

        auc_transfer = roc_auc_score(y_other, p_transfer)
        brier_transfer = brier_score_loss(y_other, p_transfer)
        # recall of that league's actual top-quartile at the recommend threshold
        top = y_other == 1
        recall = float((p_transfer[top] >= RECOMMEND_THRESHOLD).mean())
        transfer_rows.append({
            "test_league": name,
            "auc_inleague": round(r["lr_auc"], 3),
            "auc_transfer_from_laliga": round(auc_transfer, 3),
            "brier_inleague": round(r["lr_brier"], 4),
            "brier_transfer_from_laliga": round(brier_transfer, 4),
            "recall_top_quartile_at_0.60": round(recall, 3),
        })
        m2 = m.copy()
        m2["laliga_model_prob"] = p_transfer
        m2["laliga_model_decision"] = [band(p, mn) for p, mn in zip(p_transfer, m2["total_minutes"])]
    transfer = pd.DataFrame(transfer_rows)
    transfer.to_csv("outputs/cross_league_transfer.csv", index=False, encoding="utf-8")
    print("\nSaved outputs/cross_league_transfer.csv")
    print(transfer.to_string(index=False))

    # ── 3. modern-stars spotlight (La Liga model → single-team squads) ──
    spot_rows = []
    for name, (path, club) in SPOTLIGHTS.items():
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"  ! spotlight source missing: {path}")
            continue
        df = df[df["team"] == club]
        q = df[df["total_minutes"] >= MINUTES_THRESHOLD].copy()
        Xs = base_scaler.transform(q[FEATURES].values)
        q["laliga_model_prob"] = base_model.predict_proba(Xs)[:, 1]
        q["decision"] = [band(p, mn) for p, mn in zip(q["laliga_model_prob"], q["total_minutes"])]
        q["squad"] = name
        spot_rows.append(q)
    if spot_rows:
        spot = pd.concat(spot_rows, ignore_index=True)
        cols = ["squad", "player", "primary_position", "total_minutes", "laliga_model_prob",
                "decision", "npga_p90", "shots_p90", "npxg_p90", "touches_in_box_p90",
                "key_passes_p90", "pressures_p90"]
        spot = spot.sort_values("laliga_model_prob", ascending=False)
        spot[cols].round(4).to_csv("outputs/spotlight_modern.csv", index=False, encoding="utf-8")
        print("\nSaved outputs/spotlight_modern.csv")
        print(spot[["squad", "player", "laliga_model_prob", "decision", "npga_p90"]].head(12).to_string(index=False))

    # ── charts ──
    make_charts(results, summary, transfer, base, base_scaler, base_model)
    print("\nDone.")


def make_charts(results, summary, transfer, base, base_scaler, base_model):
    names = list(results.keys())
    short = {"La Liga 2015/16": "La Liga", "Premier League 2015/16": "Prem", "Serie A 2015/16": "Serie A"}

    # (a) Brier + AUC by league
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(names)); w = 0.38
    axes[0].bar(x - w / 2, [results[n]["lr_brier"] for n in names], w, label="Logistic Reg", color="#3498db")
    axes[0].bar(x + w / 2, [results[n]["gb_brier"] for n in names], w, label="Grad Boost", color="#e67e22")
    axes[0].set_xticks(x); axes[0].set_xticklabels([short[n] for n in names])
    axes[0].set_ylabel("Brier score (lower = better)"); axes[0].set_title("Calibration by league"); axes[0].legend()
    axes[1].bar(x - w / 2, [results[n]["lr_auc"] for n in names], w, label="Logistic Reg", color="#3498db")
    axes[1].bar(x + w / 2, [results[n]["gb_auc"] for n in names], w, label="Grad Boost", color="#e67e22")
    axes[1].set_xticks(x); axes[1].set_xticklabels([short[n] for n in names])
    axes[1].set_ylabel("ROC-AUC (higher = better)"); axes[1].set_title("Ranking power by league"); axes[1].set_ylim(0.5, 1.0); axes[1].legend()
    fig.tight_layout(); fig.savefig("outputs/comparison_metrics.png", dpi=150); plt.close(fig)

    # (b) decision distribution stacked
    fig, ax = plt.subplots(figsize=(9, 5))
    cats = ["recommend", "review", "not_recommended", "insufficient_evidence"]
    colors = ["#2ecc71", "#f39c12", "#95a5a6", "#bdc3c7"]
    bottom = np.zeros(len(names))
    for c, col in zip(cats, colors):
        vals = summary.set_index("league").loc[names, c].values
        ax.bar([short[n] for n in names], vals, bottom=bottom, label=c.replace("_", " "), color=col)
        bottom += vals
    ax.set_ylabel("Players"); ax.set_title("Shortlist decision distribution by league"); ax.legend()
    fig.tight_layout(); fig.savefig("outputs/comparison_decisions.png", dpi=150); plt.close(fig)

    # (c) LR coefficients by league
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(FEATURES)); w = 0.25
    for i, n in enumerate(names):
        ax.bar(x + (i - 1) * w, [results[n]["coefs"][f] for f in FEATURES], w, label=short[n])
    ax.set_xticks(x); ax.set_xticklabels([f.replace("_p90", "") for f in FEATURES], rotation=20)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Standardised LR coefficient"); ax.set_title("What predicts an elite striker — by league"); ax.legend()
    fig.tight_layout(); fig.savefig("outputs/league_coefficients.png", dpi=150); plt.close(fig)

    # (d) transfer calibration: La Liga model applied to each league
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for name in names:
        m = results[name]["frame"]
        Xs = base_scaler.transform(m[FEATURES].values)
        p = base_model.predict_proba(Xs)[:, 1]
        y = m["target"].values
        try:
            pt, pp = calibration_curve(y, p, n_bins=5, strategy="quantile")
            ax.plot(pp, pt, "s-", label=f"{short[name]} (Brier={brier_score_loss(y, p):.3f})")
        except Exception:
            pass
    ax.axvspan(REVIEW_LOWER, RECOMMEND_THRESHOLD, alpha=0.1, color="orange")
    ax.set_xlabel("La Liga-model predicted probability"); ax.set_ylabel("Actual fraction top-quartile")
    ax.set_title("Transfer calibration: La Liga model on each league"); ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig("outputs/transfer_calibration.png", dpi=150); plt.close(fig)

    # (e) modern-stars spotlight
    try:
        spot = pd.read_csv("outputs/spotlight_modern.csv").head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ["#2ecc71" if d == "recommend" else "#f39c12" if d == "review" else "#95a5a6"
                  for d in spot["decision"]]
        ax.barh(range(len(spot)), spot["laliga_model_prob"], color=colors)
        ax.set_yticks(range(len(spot)))
        ax.set_yticklabels([f"{r.player} ({r.squad})" for r in spot.itertuples()], fontsize=8)
        ax.axvline(RECOMMEND_THRESHOLD, color="green", ls="--", alpha=0.5)
        ax.axvline(REVIEW_LOWER, color="orange", ls="--", alpha=0.5)
        ax.set_xlabel("La Liga 2015/16-calibrated model probability")
        ax.set_title("Modern stars scored by the 2015/16 La Liga model")
        fig.tight_layout(); fig.savefig("outputs/spotlight_modern.png", dpi=150); plt.close(fig)
    except FileNotFoundError:
        pass
    print("Saved comparison charts to outputs/")


if __name__ == "__main__":
    main()
