---
license: cc-by-4.0
pretty_name: Hackathon retail sets
task_categories:
- time-series-forecasting
tags:
- retail
- demand-forecasting
- inventory
configs:
- config_name: set-a
  data_files:
  - split: articles
    path: set-a/articles.parquet
  - split: stock_snapshot
    path: set-a/stock_snapshot.parquet
  - split: weekly
    path: set-a/weekly.parquet
- config_name: set-b
  data_files:
  - split: articles
    path: set-b/articles.parquet
  - split: basket_promotions
    path: set-b/basket_promotions.parquet
  - split: baskets
    path: set-b/baskets.parquet
  - split: promotion_articles
    path: set-b/promotion_articles.parquet
  - split: promotion_stores
    path: set-b/promotion_stores.parquet
  - split: promotions
    path: set-b/promotions.parquet
  - split: sales_lines
    path: set-b/sales_lines.parquet
  - split: stock
    path: set-b/stock.parquet
  - split: stock_events
    path: set-b/stock_events.parquet
- config_name: set-c
  data_files:
  - split: daily
    path: set-c/daily.parquet
- config_name: set-d
  data_files:
  - split: sales
    path: set-d/sales.parquet
  - split: stock
    path: set-d/stock.parquet
- config_name: test-shop
  data_files:
  - split: history
    path: test-shop/history.parquet
---

# Hackathon retail sets

Four retail datasets for a forecasting hackathon: Set A, Set B, Set C and Set D. Each set is one shop chain; they share no stores or articles.
Values are as published by the sources; LemeBase renamed columns to shared plain names and dropped name columns (changes per set below).
Files in a set have different columns, so load one file at a time, e.g.
`load_dataset("LemeBase/hackathon-retail-sets", "set-a", split="weekly")`.

**Everything in one file:** [all-sets.zip](https://huggingface.co/datasets/LemeBase/hackathon-retail-sets/resolve/main/all-sets.zip) (23 MB): set-a to set-d and test-shop exactly as published here, with ATTRIBUTION.md and LICENSE.txt. Or: `huggingface-cli download LemeBase/hackathon-retail-sets --repo-type dataset`.

## Set A (`set-a`)

Weekly units sold and an in-stock flag per store × article, the article and store hierarchy (codes only), and stock on hand for one week.

| File | Rows | Columns |
|---|---|---|
| `articles.parquet` | 599 | store_id, article_id, product_group, division, department, department_group, store_format, format |
| `stock_snapshot.parquet` | 599 | store_id, article_id, week_start, on_hand_start, units, missed_units, on_hand_end, in_transit_1w, in_transit_2w |
| `weekly.parquet` | 98,835 | store_id, article_id, week_start, units, in_stock |

Licence: CC BY 4.0. Changed by LemeBase: Reshaped from one column per week to one row per week; columns renamed; cost columns of the stock snapshot dropped.

## Set B (`set-b`)

**Synthetic** grocery data: baskets, sales lines, articles, stock levels and stock events, promotions.

| File | Rows | Columns |
|---|---|---|
| `articles.parquet` | 1,500 | article_id, category, subcategory, unit_size, price, unit_cost |
| `basket_promotions.parquet` | 68,573 | transaction_id, line_no, promotion_id, mechanic, savings |
| `baskets.parquet` | 167,823 | transaction_id, customer_id, store_id, timestamp, subtotal, tax, total, items, payment_type, weekday |
| `promotion_articles.parquet` | 3,099 | promotion_id, article_id |
| `promotion_stores.parquet` | 149 | promotion_id, store_id |
| `promotions.parquet` | 264 | bundle_qty, buy_qty, can_stack, category, end_date, get_qty, get_type, mechanic, percent_off, priority, promotion_id, scope_type, start_date |
| `sales_lines.parquet` | 504,000 | transaction_id, line_no, article_id, units, unit_price, line_subtotal, line_discount, line_total |
| `stock.parquet` | 7,970 | article_id, store_id, on_hand, reorder_point, price_override_cents, active |
| `stock_events.parquet` | 26,215 | timestamp, store_id, article_id, delta, reason |

Licence: CC BY 4.0. Changed by LemeBase: Columns renamed; product names, brands, store names and addresses, the customer table and derived tables dropped; text categories, unit sizes and weekdays replaced by codes.

## Set C (`set-c`)

One row per store × article × day: units sold, sales, discount, stock on hand, receipt and sale recency, phasing-out flag.

| File | Rows | Columns |
|---|---|---|
| `daily.parquet` | 491,857 | date, store_id, article_id, division, category, subcategory, segment, is_return, units, sales_amount, cogs, discount_amount, discount_pct, on_hand, is_audit_anomaly, unit_cost, unit_price, latest_on_hand, stock_status, sales_type, last_receipt_date, last_sold_date, days_since_receipt, days_since_sale, is_phasing_out |

Licence: CC BY 4.0. Changed by LemeBase: Columns renamed; text categories replaced by codes.

## Set D (`set-d`)

Daily sales lines per store × article with sales type and returns, and stock on hand per store × article over date ranges.

| File | Rows | Columns |
|---|---|---|
| `sales.parquet` | 125,751 | date, sales_type, is_return, return_reason, supplier_id, article_id, division, category, subcategory, segment, store_id, sales_channel, units, sales_amount, cogs, transactions |
| `stock.parquet` | 284,755 | start_date, end_date, stock_status, supplier_id, article_id, division, category, subcategory, segment, store_id, store_type, sales_channel, on_hand, stock_value_at_price, stock_value_at_cost, unit_price, unit_cost |

Licence: CC BY 4.0. Changed by LemeBase: Columns renamed; product descriptions dropped; product hierarchy, supplier, sales channel, store type, sales type and stock status text replaced by codes.

## Test shop (`test-shop`)

One real shop, not one of the four sets: the shop the hackathon scores your forecast on. This folder holds its weekly history only; the 13 weeks that follow are hidden.

| File | Rows | Columns |
|---|---|---|
| `history.parquet` | 415,272 | article_id, week_start, units, on_hand |

- One row per article and week (week_start is a Monday). Every article has every week; units 0 = no sale.
- `units`: units sold that week, in the shop's own selling unit (a pack, a bottle, a piece).
- `on_hand`: stock at the end of the week as the shop's stock ledger recorded it, carried forward through weeks without a movement; empty before the article's first ledger entry. The ledger does not add up (opening stock plus receipts minus issues matches the recorded stock on only about 56% of rows) and some values are negative, so treat it as a noisy signal.
- Dates are moved by a fixed number of whole weeks (weekdays kept); article ids are random; names and categories are removed.
- To forecast: the 13 Mondays after the last history week (2023-04-24 to 2023-07-17), the same for every article. Upload format and scoring: the Submit tab, https://lemebase.com/hackathon/submit.

Licence: CC BY 4.0. Changed by LemeBase: aggregated to weeks, re-keyed, dates moved, columns dropped.

## Shared column names

| Column | Meaning |
|---|---|
| `date` | day |
| `week_start` | Monday of the week |
| `timestamp` | time of the event |
| `store_id` | store |
| `article_id` | article (SKU) |
| `units` | units sold (negative = return where is_return is set) |
| `on_hand` | units in stock |
| `in_stock` | article was in stock that week |
| `division` | product hierarchy level 1 (code) |
| `category` | product hierarchy level 2 (code) |
| `subcategory` | product hierarchy level 3 (code) |
| `segment` | product hierarchy level 4 (code) |
| `unit_price` | selling price per unit |
| `unit_cost` | cost per unit |
| `sales_amount` | sales value |
| `cogs` | cost of goods sold |
| `supplier_id` | supplier (pseudonymous id) |
| `stock_status` | price status of the stock (code) |
| `sales_type` | sales type code, e.g. FULL_PRICE, PROMO, MARKDOWN_TIER_1, CLEARANCE (codes vary by set) |

Other columns keep their meaning from the source and a plain snake_case name. Text values are codes (e.g. DIV-1, CAT-11, SCAT-011); within a set the same value always has the same code, and each code sits under the same parent as in the source.

---

<sub>[Sources and licences](ATTRIBUTION.md)</sub>
