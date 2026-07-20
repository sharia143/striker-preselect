"""
Striker shortlist model with calibrated confidence and abstention band.
"""
import pandas as pd
import numpy as np
import sqlite3
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ─── Config ──────────────────────────────────────────────────────────────────
MINUTES_THRESHOLD = 450  # ~5 full matches — below this, abstain as "insufficient evidence"
RECOMMEND_THRESHOLD = 0.60  # probability above this → "recommend"
REVIEW_LOWER = 0.25        # probability below this → not recommended (implicit)
# Between REVIEW_LOWER and RECOMMEND_THRESHOLD → "review"

FEATURES = ['shots_p90', 'npxg_p90', 'touches_in_box_p90', 'key_passes_p90', 'pressures_p90']

# ─── Load data ───────────────────────────────────────────────────────────────
conn = sqlite3.connect('data/players.db')
df = pd.read_sql_query("SELECT * FROM player_season", conn)
conn.close()

# Full dataset (for final output)
all_players = df.copy()

# Modelling subset: players above minutes threshold
model_df = df[df['total_minutes'] >= MINUTES_THRESHOLD].copy()
print(f"Players above {MINUTES_THRESHOLD} min threshold: {len(model_df)} / {len(df)}")

# Target: top-quartile npGA/90
q75 = model_df['npga_p90'].quantile(0.75)
model_df['target'] = (model_df['npga_p90'] >= q75).astype(int)
print(f"Top-quartile threshold (npGA/90): {q75:.3f}")
print(f"Positive class: {model_df['target'].sum()} / {len(model_df)}")

X = model_df[FEATURES].values
y = model_df['target'].values

# ─── Models ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr_probs = cross_val_predict(lr, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
lr_brier = brier_score_loss(y, lr_probs)

# Gradient Boosted Trees
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42,
                                 learning_rate=0.1, min_samples_leaf=5)
gb_probs = cross_val_predict(gb, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
gb_brier = brier_score_loss(y, gb_probs)

print(f"\nBrier scores (lower = better):")
print(f"  Logistic Regression: {lr_brier:.4f}")
print(f"  Gradient Boosting:   {gb_brier:.4f}")

# Choose the better-calibrated model
if lr_brier <= gb_brier:
    chosen_name = "Logistic Regression"
    chosen_probs = lr_probs
    chosen_model = lr
else:
    chosen_name = "Gradient Boosting"
    chosen_probs = gb_probs
    chosen_model = gb

print(f"\nChosen model: {chosen_name} (lower Brier score)")

# ─── Calibration plot ────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7, 6))

for name, probs in [('Logistic Regression', lr_probs), ('Gradient Boosting', gb_probs)]:
    prob_true, prob_pred = calibration_curve(y, probs, n_bins=7, strategy='quantile')
    brier = brier_score_loss(y, probs)
    ax.plot(prob_pred, prob_true, 's-', label=f'{name} (Brier={brier:.3f})')

ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
ax.axvspan(REVIEW_LOWER, RECOMMEND_THRESHOLD, alpha=0.1, color='orange', label='Review band')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('Calibration Curve — Striker Shortlist Model')
ax.legend(loc='lower right')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig('outputs/calibration_curve.png', dpi=150)
print("Saved outputs/calibration_curve.png")

# ─── Fit final model on all data and produce scores ─────────────────────────
chosen_model.fit(X_scaled, y)

# Score all players above minutes threshold
model_df['probability'] = chosen_model.predict_proba(X_scaled)[:, 1]

# Decision column
def decision(row):
    if row['total_minutes'] < MINUTES_THRESHOLD:
        return 'insufficient evidence'
    if row['probability'] >= RECOMMEND_THRESHOLD:
        return 'recommend'
    if row['probability'] >= REVIEW_LOWER:
        return 'review'
    return 'not recommended'

model_df['decision'] = model_df.apply(decision, axis=1)

# Handle below-threshold players
below = all_players[all_players['total_minutes'] < MINUTES_THRESHOLD].copy()
below['probability'] = np.nan
below['target'] = np.nan
below['decision'] = 'insufficient evidence'

# Combine
output = pd.concat([model_df, below], ignore_index=True)
output['score'] = output['probability'].rank(pct=True, na_option='bottom')
output = output.sort_values('probability', ascending=False, na_position='last')

# ─── Shortlist CSV ───────────────────────────────────────────────────────────
shortlist = output[['player', 'team', 'primary_position', 'total_minutes',
                     'score', 'probability', 'decision',
                     'npga_p90', 'shots_p90', 'npxg_p90', 'touches_in_box_p90',
                     'key_passes_p90', 'pressures_p90']].copy()
shortlist.columns = ['player', 'team', 'position', 'minutes', 'score', 'probability',
                      'decision', 'npga_p90', 'shots_p90', 'npxg_p90',
                      'touches_in_box_p90', 'key_passes_p90', 'pressures_p90']
shortlist = shortlist.round(4)
shortlist.to_csv('outputs/shortlist.csv', index=False, encoding='utf-8')
print(f"Saved outputs/shortlist.csv ({len(shortlist)} players)")

# ─── Power BI export ─────────────────────────────────────────────────────────
powerbi = shortlist.copy()
powerbi.to_csv('outputs/powerbi_export.csv', index=False, encoding='utf-8')
print("Saved outputs/powerbi_export.csv")

# ─── Top shortlist chart ─────────────────────────────────────────────────────
top = shortlist[shortlist['decision'] == 'recommend'].head(15)

fig2, ax2 = plt.subplots(figsize=(9, 6))
colors = ['#2ecc71' if d == 'recommend' else '#f39c12' for d in top['decision']]
bars = ax2.barh(range(len(top)), top['probability'], color=colors)
ax2.set_yticks(range(len(top)))
ax2.set_yticklabels([f"{r['player']} ({r['team']})" for _, r in top.iterrows()], fontsize=9)
ax2.set_xlabel('Model probability (top-quartile npGA/90)')
ax2.set_title('Top Recommended Strikers — La Liga 2015/16')
ax2.axvline(RECOMMEND_THRESHOLD, color='green', linestyle='--', alpha=0.5, label=f'Recommend threshold ({RECOMMEND_THRESHOLD})')
ax2.axvline(REVIEW_LOWER, color='orange', linestyle='--', alpha=0.5, label=f'Review threshold ({REVIEW_LOWER})')
ax2.legend(loc='lower right', fontsize=8)
ax2.invert_yaxis()
fig2.tight_layout()
fig2.savefig('outputs/top_shortlist.png', dpi=150)
print("Saved outputs/top_shortlist.png")

# ─── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"DECISION SUMMARY")
print(f"{'='*50}")
print(output['decision'].value_counts().to_string())
print(f"\nTop 10 recommended:")
rec = shortlist[shortlist['decision'] == 'recommend'].head(10)
print(rec[['player', 'team', 'minutes', 'probability', 'npga_p90']].to_string(index=False))
