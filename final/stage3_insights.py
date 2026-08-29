"""
=====================================================================
 STAGE 3 — ACTIONABLE INSIGHTS  (the business-decision layer)
 Machine Learning-Based Sentiment Analysis of Amazon Fine Food Reviews
 Module: ITC 4378 - Big Data Analytics
---------------------------------------------------------------------
 Stage 2 told us WHAT each review is positive or negative about.
 Stage 3 turns that into decisions a seller or product manager can
 act on. It answers four questions:

   1. Which food aspects drive the most NEGATIVE sentiment overall?
   2. Which complaints matter most once we weight by how HELPFUL
      other customers found the review? (a complaint 40 people
      agreed with outweighs one nobody voted on)
   3. How is each aspect TRENDING over time — getting better or worse?
   4. For a given PRODUCT, what is the aspect breakdown, so a manager
      knows whether to fix the recipe, the packaging, or the shipping?

 Inputs : aspect_sentiment.parquet  (from Stage 2)
 Outputs: printed insight tables + CSVs + PNG charts for the report.
=====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless: save charts, don't display
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
#  CONFIGURATION
# ------------------------------------------------------------------ #
PARQUET_IN = "aspect_sentiment.parquet"   # from Stage 2
ASPECTS = ["taste", "quality", "freshness", "packaging",
           "price", "delivery", "health", "portion"]

# ================================================================== #
#  1. LOAD
# ================================================================== #
df = pd.read_parquet(PARQUET_IN)
print(f"[LOAD] reviews: {len(df):,}")

has_help = {"HelpfulnessNumerator", "HelpfulnessDenominator"}.issubset(df.columns)
has_time = "Time" in df.columns
has_prod = "ProductId" in df.columns

# helpfulness weight: how many people found the review helpful, +1 so
# that a review nobody voted on still counts once (never zero-weight).
if has_help:
    df["help_weight"] = df["HelpfulnessNumerator"].fillna(0).astype(float) + 1.0
else:
    df["help_weight"] = 1.0
    print("[WARN] no helpfulness columns — using unweighted counts")

# ================================================================== #
#  2. ASPECT NEGATIVITY  (unweighted vs helpfulness-weighted)
# ------------------------------------------------------------------ #
#  For each aspect we compute, among reviews that MENTION it:
#    - % negative  (plain count)
#    - % negative WEIGHTED by helpfulness (what customers endorse)
#  The gap between the two is itself interesting: if weighted
#  negativity is higher, the complaints are the ones people agree with.
# ================================================================== #
rows = []
for a in ASPECTS:
    sent = df[f"{a}_sent"]
    mentioned = df[sent != "not_mentioned"]
    if len(mentioned) == 0:
        continue
    n = len(mentioned)
    neg_mask = mentioned[f"{a}_sent"] == "negative"
    pos_mask = mentioned[f"{a}_sent"] == "positive"

    pct_neg = neg_mask.mean()
    pct_pos = pos_mask.mean()

    w = mentioned["help_weight"]
    w_neg = w[neg_mask].sum() / w.sum()      # helpfulness-weighted % neg

    rows.append({
        "aspect": a,
        "mentions": n,
        "mention_rate": n / len(df),
        "pct_negative": pct_neg,
        "pct_positive": pct_pos,
        "weighted_pct_negative": w_neg,
    })

insight = pd.DataFrame(rows).sort_values("weighted_pct_negative",
                                         ascending=False)

print("\n" + "=" * 74)
print("ASPECT PAIN-POINT RANKING  (what to fix first)")
print("=" * 74)
print(f"{'aspect':<11}{'mentions':>9}{'ment.rate':>11}"
      f"{'%neg':>7}{'%pos':>7}{'wt.%neg':>9}")
print("-" * 74)
for _, r in insight.iterrows():
    print(f"{r['aspect']:<11}{int(r['mentions']):>9}"
          f"{r['mention_rate']:>10.1%}"
          f"{r['pct_negative']:>7.0%}{r['pct_positive']:>7.0%}"
          f"{r['weighted_pct_negative']:>8.0%}")
print("=" * 74)
print("Read: aspects at the top have the most negative sentiment that\n"
      "customers actually agreed was helpful -> highest-priority fixes.")

insight.to_csv("insight_aspect_ranking.csv", index=False)

# ---- chart: weighted vs unweighted negativity --------------------- #
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(insight))
ax.bar(x - 0.2, insight["pct_negative"], 0.4, label="% negative (count)")
ax.bar(x + 0.2, insight["weighted_pct_negative"], 0.4,
       label="% negative (helpfulness-weighted)")
ax.set_xticks(x); ax.set_xticklabels(insight["aspect"], rotation=30)
ax.set_ylabel("share of mentions that are negative")
ax.set_title("Aspect pain points: count vs helpfulness-weighted")
ax.legend()
fig.tight_layout(); fig.savefig("chart_aspect_negativity.png", dpi=130)
plt.close(fig)

# ================================================================== #
#  3. TREND OVER TIME  (is each aspect getting better or worse?)
# ------------------------------------------------------------------ #
#  Time in the dataset is a unix timestamp. We bucket by YEAR and
#  track the % negative per aspect. A rising line = worsening aspect.
# ================================================================== #
if has_time:
    df["year"] = pd.to_datetime(df["Time"], unit="s", errors="coerce").dt.year
    trend_rows = []
    for a in ASPECTS:
        sub = df[df[f"{a}_sent"] != "not_mentioned"].copy()
        if sub.empty:
            continue
        sub["is_neg"] = (sub[f"{a}_sent"] == "negative").astype(int)
        by_year = sub.groupby("year")["is_neg"].mean()
        for yr, val in by_year.items():
            if pd.notna(yr):
                trend_rows.append({"aspect": a, "year": int(yr),
                                   "pct_negative": val})
    trend = pd.DataFrame(trend_rows)
    trend.to_csv("insight_aspect_trend.csv", index=False)

    # chart: negativity trend for the 4 most-negative aspects
    top4 = insight["aspect"].head(4).tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    for a in top4:
        t = trend[trend["aspect"] == a].sort_values("year")
        if len(t) > 1:
            ax.plot(t["year"], t["pct_negative"], marker="o", label=a)
    ax.set_ylabel("% of mentions that are negative")
    ax.set_xlabel("year")
    ax.set_title("Negativity trend over time (top pain-point aspects)")
    ax.legend()
    fig.tight_layout(); fig.savefig("chart_aspect_trend.png", dpi=130)
    plt.close(fig)
    print("\n[TREND] yearly negativity trend saved "
          "(insight_aspect_trend.csv, chart_aspect_trend.png)")
else:
    print("\n[TREND] skipped — no Time column")

# ================================================================== #
#  4. PER-PRODUCT DRILL-DOWN
# ------------------------------------------------------------------ #
#  For the products with the most reviews, show the aspect-level
#  negative rate so a manager can see WHERE a product is failing.
#  This is the "fix shipping, not the recipe" table.
# ================================================================== #
if has_prod:
    top_products = (df["ProductId"].value_counts()
                    .head(5).index.tolist())
    prod_rows = []
    for pid in top_products:
        sub = df[df["ProductId"] == pid]
        row = {"ProductId": pid, "n_reviews": len(sub)}
        for a in ASPECTS:
            col = sub[f"{a}_sent"]
            among = col[col != "not_mentioned"]
            row[a] = (among == "negative").mean() if len(among) else np.nan
        prod_rows.append(row)
    prod = pd.DataFrame(prod_rows)
    prod.to_csv("insight_product_breakdown.csv", index=False)

    print("\n" + "=" * 74)
    print("PER-PRODUCT NEGATIVE-RATE BY ASPECT  (top 5 most-reviewed)")
    print("=" * 74)
    show_cols = ["ProductId", "n_reviews"] + ASPECTS
    with pd.option_context("display.float_format", lambda v: f"{v:.0%}"
                           if pd.notna(v) else "-"):
        print(prod[show_cols].to_string(index=False))
    print("=" * 74)
    print("Read across a row: the highest-% aspect is that product's\n"
          "weakest point — the thing to fix for THAT product.")
else:
    print("\n[PRODUCT] skipped — no ProductId column")

# ================================================================== #
#  5. HEADLINE TAKEAWAYS  (auto-generated, drop into your report)
# ================================================================== #
print("\n" + "=" * 74)
print("HEADLINE INSIGHTS")
print("=" * 74)
worst = insight.iloc[0]
best_pos = insight.sort_values("pct_positive", ascending=False).iloc[0]
print(f"* Biggest pain point: '{worst['aspect']}' — "
      f"{worst['weighted_pct_negative']:.0%} of helpfulness-weighted "
      f"mentions are negative.")
print(f"* Strongest positive: '{best_pos['aspect']}' — "
      f"{best_pos['pct_positive']:.0%} of mentions are positive.")
most_disc = insight.assign(
    gap=(insight["weighted_pct_negative"] - insight["pct_negative"]).abs()
).sort_values("gap", ascending=False).iloc[0]
print(f"* Complaint most endorsed by customers: '{most_disc['aspect']}' "
      f"(helpfulness-weighted negativity differs most from raw count).")
print("=" * 74)

print("\n[SAVE] insight CSVs + charts written:")
print("  insight_aspect_ranking.csv     chart_aspect_negativity.png")
if has_time: print("  insight_aspect_trend.csv       chart_aspect_trend.png")
if has_prod: print("  insight_product_breakdown.csv")
