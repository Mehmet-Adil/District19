# Datasets: what we found and how to use it for the test-shop forecast

This is a summary of the exploration in [set_a_b_test_analysis.ipynb](set_a_b_test_analysis.ipynb) and [set_c_d_analysis.ipynb](set_c_d_analysis.ipynb). Numbers come from those notebooks. For file layouts and licences, see [README.md](README.md).

**Contents**
1. [At a glance](#1-at-a-glance)
2. [The task: test shop](#2-the-task-test-shop)
3. [Set A](#3-set-a--weekly-store--article-real)
4. [Set B](#4-set-b--transactions-synthetic)
5. [Set C](#5-set-c--daily-store--article)
6. [Set D](#6-set-d--sales-lines--stock-intervals)
7. [How the sets compare to the test shop](#7-how-the-sets-compare-to-the-test-shop)
8. [Playbook: using all of this for better test predictions](#8-playbook-using-all-of-this-for-better-test-predictions)
9. [Open questions](#9-open-questions)

---

## 1. At a glance

| | Test shop | Set A | Set B | Set C | Set D |
|---|---|---|---|---|---|
| **What** | the shop we are scored on | weekly sales + in-stock flag | grocery transactions, promos, stock | daily sales, discounts, stock, recency flags | daily sales lines + stock history |
| **Real / synthetic** | real | real | **synthetic** | real | real |
| **Grain** | article × week | store × article × week | basket line (timestamp) | store × article × day | store × article × day × sales type |
| **Period** | 2020-07-27 → 2023-04-17 (143 wk) | 2021-04-12 → 2024-06-03 (165 wk) | 2025-01-01 → 2025-11-01 (~44 wk) | 2023-12-31 → 2026-04-20 (~120 wk) | 2025-06-01 → 2026-04-24 (~47 wk) |
| **Size** | 2,904 articles | 67 stores, 297 articles, 599 pairs | 5 stores, 1,500 articles, 168k baskets | 29 stores, 338 articles, 851 pairs | 40 stores, 2,326 articles, 50k pairs |
| **Zero weeks** ¹ | 58% | 39% | 23% | 74% | 92% (store × article) / 50% (article) |
| **Yearly seasonality learnable?** | yes (2.7 years) | yes (3+ years) | no (< 1 year) | yes (2.3 years) | no (< 1 year) |
| **Usefulness for the test shop** | — | **high** | **low** | **medium** | **medium** (only aggregated to article × week) |

¹ Share of weeks with no sale, counted from each series' first sale to the end of the data (section 7).

**The one-line version:** the test shop is a sparse, intermittent, long-tail weekly series scored with an unweighted MASE. Robust *median-style* forecasts with zeros for dead articles already beat the site's baselines. A and C (and D at article level) are the extra data most like it, and they are the ones to use for a global model. B is synthetic and too smooth to teach anything about intermittent demand.

---

## 2. The task: test shop

### What has to be predicted
- One real shop, not part of A–D. `test-shop/history.parquet` has 4 columns: `article_id, week_start, units, on_hand`.
- Forecast **13 weeks, 2023-04-24 → 2023-07-17 (w1–w13), for all 2,904 articles**: 37,752 numbers.
- Upload a CSV with the header `article_id, w1, …, w13`, one row per article. Decimals are allowed.
- **Score = (1 − MASE) × 100**, higher is better. For each article: mean absolute error over the 13 weeks ÷ the article's mean absolute week-to-week change in its history. The score is then averaged **unweighted** over articles.
- Site reference points: naive (last value) ≈ −64.5, AutoARIMA ≈ −33.7. 0 means "off by one typical weekly move", 100 is perfect.
- The live leaderboard scores only w1–w7. **The final ranking uses the hidden w8–w13**, so don't tune to the leaderboard.

### Data findings
- **Complete grid**: 2,904 articles × 143 Mondays = 415,272 rows, with no gaps and no duplicates. 0 means no sale.
- **No other features**: no store, category, price or promo data. Everything has to come from the series themselves (and from other sets).
- **Dates are shifted by whole weeks.** Holiday calendars (Easter, Christmas, Black Friday) don't line up with the real dates. Learn seasonality from **52-week lags**, not from a calendar.
- **Very sparse**: 65% of all rows are 0. By demand class, ~66% of articles are intermittent, 14% lumpy and only 16% smooth.
- **Very concentrated**: the top 1% of articles sell 32% of units and the top 10% sell 84%. The metric weights every article equally, so **the long tail matters as much as the best sellers**.
- **Lifecycle**:
  - 1,028 articles sold in the last week.
  - **1,073 haven't sold for 8+ weeks**, and 532 not for over a year. They are probably discontinued, and forecasting 0 for them is almost free.
  - 142 articles are new (first sale in the last 26 weeks), so their history is short.
- **`on_hand` is noisy**:
  - Missing on 14% of rows (before an article's first ledger entry).
  - 7 negative values, outliers up to ~200k.
  - The ledger matches its own movements on only ~56% of rows.
  - Use it at most as a soft stock-out hint, never as truth.

### What the metric rewards
- MASE uses **absolute error**, and the median minimises absolute error, so medians beat means.
- Many slow movers have a **tiny denominator** (median 0.80, 10% of articles below 0.04, 535 below 0.1). For those, forecasting 1 when the truth is 0 gives MASE > 10. **Over-forecasting slow sellers is the biggest risk.**
- Backtest (fit on history, score on the last 13 weeks, same MASE):

  | Model | Last 13 weeks | 13 weeks before that |
  |---|---|---|
  | median last 26 w | **−4.9** | −30.3 |
  | median last 13 w | −6.7 | −30.5 |
  | median last 8 w | −8.4 | −31.4 |
  | zero everywhere | −23.4 | −49.3 |
  | mean last 13 w | −36.0 | −52.7 |
  | seasonal naive (52 w ago) | −40.9 | −71.0 |
  | naive (last week) | −58.3 | −97.6 |

  - Our naive (−58) is close to the site's naive (−64.5), so the offline setup mirrors the real scoring.
  - **Scores move ~25 points between windows**, but the ranking stays the same: medians > zero > mean > seasonal naive > naive. Always average several rolling windows.
  - The worst 5% of articles cause ~47% of the total MASE. They are mostly low-volume articles with a non-zero forecast and a tiny denominator.
- Seasonality check: compare the horizon weeks of earlier years with the 13 weeks before them. The notebook has this as a ratio per year (section 8.1).

---

## 3. Set A: weekly store × article (real)

### Files
| File | Rows | Content |
|---|---|---|
| `weekly` | 98,835 | 599 pairs × 165 weeks, full grid. Columns `units`, `in_stock` |
| `articles` | 599 | product hierarchy + store format per pair |
| `stock_snapshot` | 599 | a single week (2024-04-08) |

### Findings
- **Closest in format to the test shop**: weekly, Monday weeks, full grid, units in selling units.
- **Hierarchy** is nested numeric codes: department_group (6) > department (26) > division (47) > product_group (111) > article (297). For example, product group 301202 → division 3012 → department 30.
- **Stores are very unequal**:
  - `store_format` and `format` are the same attribute coded twice (1↔13, 2↔216, 5↔521). Keep one.
  - Format 2 is 5 stores with 86% of pairs (12–224 articles each). Format 1 is 60 tiny stores with 1–3 articles each.
  - **Store 61 alone is ~40% of rows.** Plain averages mostly reflect store 61, so use store-normalised indices to check patterns.
- **Demand** is low and skewed: mean 2.9 units/week, median 1–2, 43% of rows are 0.
- **`in_stock` is valuable**:
  - `in_stock = True` with units = 0 is real zero demand.
  - `in_stock = False` means sales were capped (censored), so treat those rows as missing, not 0.
  - The in-stock rate rises from ~65% to 100% over the period, so **early weeks understate demand**.
- **Seasonality** (ISO weeks, consistent across years):
  1. **Christmas, KW50–51**: the strongest peak (KW51 ≈ 8 units vs a normal 2.5–3). KW52 drops back because of the 25/26 Dec closures. KW1 is still elevated.
  2. **November promotion**: the week starting on the 2nd Monday of November (KW45/46) is ~2.5× a normal week, followed by a smaller tail week. It is a recurring retailer promo (no promo data to confirm). Mainly product group 240201 (articles 124/125); e.g. store 61 / article 124 sells ~490 vs a normal ~90.
  3. **May holidays, KW17–19**: a moderate lift of 1.2–1.4×.
  4. No visible effect from Easter or summer/autumn school holidays. The KW24 bump is noise.
- **Outliers**:
  - The largest rows (up to 494) are real promo and Christmas peaks, so don't cap them.
  - 15 rows have a robust z-score > 20, 12 of them in department 30. They look like sudden bulk orders and are still to be checked.
- **`stock_snapshot`**:
  - Only `on_hand_end` (mean 3.7) and `in_transit_2w` (non-zero for ~71% of pairs) are filled; the other columns are all 0.
  - It looks like the starting state of an ordering simulation with a 2-week lead time. Not needed for the test-shop forecast.
- 2024 looks lower only because the data ends in June (no Nov/Dec peaks). Compare years on matching weeks only.

### Use for the test shop
- **Best extra training data** for a global weekly model:
  - Same grain, 3+ years of history (so 52-week lags work), and a similar mix of demand classes (section 7).
  - Its `in_stock` flag shows how stock-outs depress sales, which also helps interpret the test shop's noisy `on_hand`.
- Use store × article series as additional "articles". Mask `in_stock = False` weeks out of the loss, or treat them as missing.
- Seasonality shapes like "Christmas peak, then a KW52 dip, then an elevated KW1" show how retail peaks look. In the test shop they appear at *shifted* dates, so only lag-based features can pick them up.

---

## 4. Set B: transactions (synthetic)

### Files
| File | Rows | Content |
|---|---|---|
| `baskets` | 167,823 | one per transaction (5 stores, 5,000 customers) |
| `sales_lines` | 504,000 | basket lines, 1–3 units each |
| `articles` | 1,500 | category, unit size, price, cost |
| `promotions` (+ `promotion_articles`, `promotion_stores`, `basket_promotions`) | 264 promos | mechanics BOGO, BUNDLE, COUPON, MANAGER_SPECIAL |
| `stock`, `stock_events` | 7,970 / 26,215 | stock level and reorder point; replenish / stockout events |

### Findings
- **It is generated data**:
  - Baskets are spread almost exactly evenly over the hours of the day.
  - **Exactly 6 promotions are active every day.**
  - Every article sells every week at chain level (0.1% zero weeks, 100% "smooth").
  - Don't read real-world patterns into it.
- Revenue is 4.88M. The mean basket is 29.06 (median 26.12). 10.5% of lines carry a discount and **32.8% of baskets include a promotion**. Coupons give the largest total savings.
- **Data problems**:
  - `stock.store_id` is text (`"S0001"`), while every other table uses integers (`1`). Convert it before joining.
  - `stock` has 1,998 article IDs, and **500 of them are missing from `articles`**.
  - `promotions` has 30% missing cells: mechanic-specific columns such as `percent_off` and `buy_qty`.
  - `stock_events` has 181 fully duplicated rows.
- Stock: 11% of rows are below the reorder point and 35% have a price override. REPLENISH events add 30–100 units. STOCKOUT events have delta 0.

### Use for the test shop
- **Low.** Its demand is smooth and dense (median 3.2 units per store × article week, 50% smooth), the opposite of the test shop. It also has less than a year of history, so no yearly seasonality.
- Adding it to a global model would pull forecasts toward "always sells", which is exactly what MASE punishes on slow movers. **Leave it out**, or at most use it to prototype promo or basket features that the test shop doesn't have anyway.

---

## 5. Set C: daily store × article

### File
`daily`: 491,857 rows. One row per store × article × day with units, sales, discount, stock, receipt and sale recency, and a phasing-out flag. 29 stores, 338 articles, 851 pairs.

### Data quality and meaning of columns
- **Numbers are stored as `Decimal` objects and dates as `date` objects.** Convert them to float and datetime before use.
- **Not a full grid**: only 69% of possible pair-days exist.
  - Listed pairs fall from ~800 to ~350 over the period, and only 43% are present on the last day.
  - Gaps of weeks (median 31 days) usually start in the last week of a month and end mid-month.
  - **A missing row ≠ zero sales**. Fill zeros only inside listed periods.
- **Code columns**:
  - `is_return`: −1 = no transaction (filler row, units always 0), 0 = sale, 1 = return.
  - `sales_type`: R (regular) / C (clearance) / PROMO / MD (markdown), each with a `*_RETURN` variant. On filler rows it equals `stock_status`.
  - PROMO and MD sales only happen on clearance-status stock.
- **Hierarchy**: 1 division > 13 categories > 30 subcategories > 171 segments > 338 articles. Almost a tree: SCAT-004 and SCAT-023 sit under two categories.
- **Stores** are unequal: the two largest (STR-016, STR-010) hold 29% of pairs, and the smallest has one article.

### Demand
- **94% of rows are 0.** A typical pair sells one unit every ~17 days (0.06 units/day). Returns are 0.15% of rows.
- **Main peak in June (KW24–26, ~2–2.5× normal in both 2024 and 2025)**, a smaller Christmas peak (KW51), and spring above autumn. Fri/Sat are ~1.3–1.6× Mon–Thu.
- Demand per listed pair is lower in 2025/26 than in 2024.
- **Outlier**: PROD-0000202 causes the 9 largest rows. In STR-022 it is listed for only 53 days at price 0.91 (5.97 elsewhere) and sells 574 units, including **292 in one day**. Probably a bulk sale or a pack-size error. Cap or drop it.

### Prices, stock and flags
- `cogs = units × unit_cost` holds on 98% of rows and `discount_pct` is consistent. `sales_amount = units × price − discount` only holds on 84%.
- 91% of articles change price over time. There are glitches: 8,141 rows at price 0, and spikes to ~1,000 for a ~30 article.
- PROMO rows carry ~28% discount and MD ~24%. 12% of regular sales are also discounted.
- **`on_hand` doesn't track sales**: it fell by the units sold on only 4% of sale days, and the median pair holds ~2,200 days of demand. It probably isn't shelf stock, so it is weak as a stock-out signal.
- `is_audit_anomaly` only occurs when `on_hand = 0` (about half of those rows). `latest_on_hand` is a single constant per pair (sometimes negative), so it is useless over time.
- **`is_phasing_out` leaks the target**: it is never true on a sale day and only appears after ≥ 31 days without a sale. Don't use it, or same-day `days_since_sale`, as features. Lagged versions are fine.

### Use for the test shop
- **Medium.** Store × article weekly series are sparser than the test shop (74% zero weeks, 85% intermittent), which resembles its **slow tail**. That is exactly where most of the MASE error comes from.
- Over 2 years of history, so 52-week lags work.
- Useful extra training data for the tail, after:
  - aggregating to Monday weeks,
  - keeping only listed periods,
  - dropping the leaky columns,
  - removing the PROD-0000202 burst.
- The phasing-out pattern (rows in the table but no sales, then delisting) is a real-world example of how articles die. It supports the "dead → forecast 0" rule.

---

## 6. Set D: sales lines + stock intervals

### Files
| File | Rows | Content |
|---|---|---|
| `sales` | 125,751 | one row per date × store × article × sales type × return flag; transactions only |
| `stock` | 284,755 | validity intervals: `on_hand` and price status from `start_date` to `end_date` |

40 stores, 2,326 articles, 30 suppliers, 2025-06-01 → 2026-04-24.

### Data quality and meaning of columns
- `sales_channel`, `store_type` and `return_reason` are constant or empty, so drop them. There are no duplicate keys.
- **Hierarchy** is a clean tree: 3 divisions > 27 categories > 51 subcategories > 600 segments > 2,326 articles. Each article has exactly one supplier, and the top 3 suppliers sell 69% of units.
- **`stock` is a history table**:
  - A new interval starts when stock or price status changes (4.7 per pair, median 8 days). `9999-12-31` marks the current row (22,701 open intervals).
  - 96% of intervals are contiguous, with no overlaps. `unit_price = 0` exactly when `on_hand = 0`.
  - 1,380 intervals have negative stock.
- **Stock is incomplete**:
  - Chain stock rises from ~5k units on 1 Jun to ~74k in November, so the table likely lacks an opening balance.
  - Only 74% of sales fall inside a stock interval, and 12% of those have `on_hand ≤ 0`.

### Demand
- **Only transactions are stored, with no zero rows.** A store × article pair sells on a median of **2 days in 11 months** (40% only once), so 92% of store × article weeks are zero.
- Share of periods with a sale at different aggregation levels:

  | Level | Periods with a sale |
  |---|---|
  | store × article × day | 1% |
  | store × article × week | 8% |
  | store × segment × week | 18% |
  | **article × week (chain)** | **50%** |
  | store × category × week | 66% |

- **Clear retail calendar**:
  - Closed on 27 Nov and 25 Dec (Thanksgiving and Christmas, likely a US retailer).
  - Peaks on Black Friday (28 Nov), in the December run-up (**24 Dec is the biggest day, ~7× an average day**), on Valentine's Day, and on late-August weekends.
  - Saturday sells ~2.5× Monday.
- **High churn**: 55% of articles first sell after June 2025, and 17% have not sold in the last 8 weeks.

### Sales types, prices and returns
| Sales type | Units | Revenue | Median price | Median margin |
|---|---|---|---|---|
| FULL_PRICE | 75% | 85% | 100 | 49% |
| PROMO | 19% | 12.5% | 70 | 30% |
| MARKDOWN_TIER_1 | 3% | 2% | 50 | 14% |
| MARKDOWN_TIER_2 | 2% | 1% | 40 | **−10%** |
| CLEARANCE | 1% | 0.2% | 10 | **−54%** |

- The promo share comes in waves (25–45% of units in some weeks). Clearance appears only in a few windows.
- Return rate is ~6% at full price and ~2% on clearance.
- `sales_type` matches the `stock_status` valid on the sale date. PROMO sales happen on FULL_PRICE stock, so promo is a sales-time event.

### Use for the test shop
- **Medium, only at article × week (chain) level.** At that level it is the closest match to the test shop's demand profile (50% vs 58% zero weeks; 60% vs 66% intermittent).
- Store × article is unusable on its own.
- Less than a year of history, so it **can't teach yearly seasonality**. It can still add examples of short-horizon dynamics: new-article ramp-up, decay to zero, and promo or holiday spikes.
- Its high churn (many new and dying articles) is a good training ground for the test shop's 142 new and 1,073 dormant articles.

---

## 7. How the sets compare to the test shop

Weekly demand per series, counted from its first sale week to the end of the data (Set C: to the pair's last row). Demand classes follow Syntetos–Boylan (ADI cut-off 1.32, CV² cut-off 0.49).

| Series | Count | Zero weeks | Median units in a sale week | Smooth | Intermittent | Erratic | Lumpy |
|---|---|---|---|---|---|---|---|
| **Test shop: article** | 2,904 | **58%** | 2.1 | **16%** | **66%** | **4%** | **14%** |
| A: store × article | 599 | 39% | 2.0 | 11% | 50% | 8% | 31% |
| A: article (chain) | 297 | 32% | 2.3 | 22% | 36% | 16% | 26% |
| B: store × article | 7,500 | 23% | 3.2 | 50% | 35% | 8% | 7% |
| B: article (chain) | 1,500 | 0% | 12.7 | 100% | 0% | 0% | 0% |
| C: store × article | 851 | 74% | 1.4 | 0% | 85% | 0% | 15% |
| D: store × article | 49,860 | 92% | 1.0 | 0% | 99% | 0% | 1% |
| D: article (chain) | 2,326 | 50% | 1.9 | 7% | 60% | 8% | 26% |

**Reading it**
- The test shop sits between A (denser, lumpier) and C (sparser, almost all intermittent).
- D at article level is close.
- B is in a different world.
- A training pool of **A (store × article) + C (store × article) + D (article)** covers the test shop's range from best sellers to the dead tail.

---

## 8. Playbook: using all of this for better test predictions

### 8.1 Baseline to beat
1. Forecast **median of the last 13–26 weeks**, flat for all 13 weeks. This scores about −5 to −7 on the last window, far above AutoARIMA's −33.7.
2. **Forecast 0 for dead articles** (no sale in the last ~8 weeks, 1,073 articles).
3. Clip forecasts at ≥ 0, and make sure every one of the 2,904 articles has a row. The notebook's `make_submission` handles both.

### 8.2 Validation (most important habit)
- Hold out 13 weeks, fit on the rest, and score with the same MASE (`mase_score` in the test-shop notebook).
- Use **several rolling windows**, because scores move ~25 points between windows.
- Report w1–w7 and w8–w13 separately. The final score uses w8–w13, so don't trust the leaderboard (w1–w7) on its own.

### 8.3 Where the points are
- The worst 5% of articles cause ~half of the error: slow movers with a tiny MASE denominator and a non-zero forecast.
- **Being conservative on slow sellers is worth more than being accurate on best sellers.** Ideas:
  - Shrink forecasts toward 0 for articles whose denominator is < 0.1.
  - Use quantiles below the median for intermittent series.
  - Tune the dead-article rule (how many quiet weeks before forecasting 0).
- Best sellers carry the volume, but every article weighs the same in the score.

### 8.4 Global model with extra data
- Train one model across all series: LightGBM with **L1 / quantile-0.5 loss**, because MASE is absolute error. Use only lag-based features:
  - lags 1–13 and 52,
  - rolling medians and means over 4 / 8 / 13 / 26 / 52 weeks,
  - weeks since last sale and since first sale (age),
  - share of zero weeks,
  - ADI / CV² class,
  - forecast horizon step (w1…w13).
- **Data to add**, each prepared as below:

  | Source | Level | Preparation |
  |---|---|---|
  | Test shop | article × week | as is |
  | Set A | store × article × week | mask `in_stock = False` weeks; keep promo/Christmas peaks |
  | Set C | store × article × week | Monday weeks; only listed periods; drop `is_phasing_out` / `latest_on_hand` / raw `on_hand`; remove the PROD-0000202 burst at STR-022 |
  | Set D | article × week (chain) | Monday weeks; net returns; only weeks after the first sale |
  | Set B | — | **leave out** (synthetic, too smooth) |

- **Normalise each series** (e.g. divide by its own mean, or by its MASE denominator) so that a 494-unit week in Set A doesn't dominate the 0/1 series. Add a "source" feature so the model can tell the shops apart.
- Weight the test shop's own series higher, or fine-tune on them at the end.

### 8.5 Seasonality
- The test shop's dates are shifted, so **don't use holiday calendars**. Use the 52-week lag and the horizon-vs-previous-13-weeks ratio of earlier years.
- A and C show that retail peaks are short and sharp (1–2 weeks): Christmas, November promo, June. That is why seasonal naive scores badly: it copies last year's noise.
- Blend seasonality in mildly: a ratio adjustment on a median, not raw seasonal naive.

### 8.6 Specialist models for intermittent series
- **Croston / SBA / TSB** for intermittent articles; TSB also handles decay to zero.
- Compare them per demand class against the median baseline, and keep whichever wins per class.

### 8.7 Stock signal
- The test shop's `on_hand` is too noisy to trust. Treat a week with 0 units **and** `on_hand ≤ 0` at the start as "possibly censored" rather than true zero demand, similar to Set A's `in_stock = False`. Test whether masking those weeks improves the backtest.

---

## 9. Open questions
- **Set A**: inspect the 15 extreme rows in department 30 before training.
- **Set A**: confirm with the brief what `stock_snapshot` is for (an ordering simulation with a 2-week lead time?).
- **Set C**: what does `on_hand` measure, if not store stock?
- **Set C**: why do pairs drop out from about the 24th–28th of a month until mid-month?
- **Set D**: is the stock build-up from June to November real, or a missing opening balance?
- **Test shop**: does masking possibly censored weeks (0 units, no stock) help in backtests?
- **Test shop**: which aggregation of the extra sets actually improves the rolling-window backtest? Test each source separately before combining them.
