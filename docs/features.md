# Feature dictionary

One row per customer per month-end snapshot. Every feature uses only transactions dated
before the snapshot date. Money is in USD (ZiG converted at `fx.zig_per_usd` in settings.yaml).
Ratios 'versus baseline' compare the latest month with the customer's own average over the
6 months before it; 1 means no change. Missing values are filled with 0 for amounts and
shares, and with 1 for change ratios.

## Identifiers and metadata (not model inputs)

| column | meaning |
| --- | --- |
| customer_id | raw customer id (pseudonymised before any dashboard or log) |
| snapshot_date | last day of the month the features describe |
| month_idx | 0-based month number from the data start |
| segment | customer segment from the customers table |
| branch | home branch from the customers table |

## Income

| feature | meaning |
| --- | --- |
| salary_detected | 1 if a credit of similar size arrived in at least 5 of the last 12 months (or 4 of the last 6) |
| salary_amount_usd | typical size of that regular credit over the last 12 months, USD (0 if none) |
| months_since_salary | months since the last credit that looked like the salary (0 = this month, capped at 12) |
| salary_change_1m | this month's largest credit divided by the typical salary (1 = unchanged, 0 = nothing arrived) |
| salary_trend_3m | average largest credit of the last 3 months divided by the 3 months before that |
| n_salary_months_6m | how many of the last 6 months had a salary-sized credit |
| payer_changed | 1 if the most recent salary came from a different payer than the one before |
| income_1m_usd | all money in this month, excluding transfers from the customer's own accounts, USD |
| income_3m_usd | all money in over the last 3 months, USD |
| income_change | this month's income divided by the monthly average of the 6 months before it (1 = unchanged) |
| top_credit_1m_usd | largest single credit this month, USD |

## Balance

| feature | meaning |
| --- | --- |
| bal_end_usd | month-end balance across all accounts, USD |
| bal_avg_3m_usd | average month-end balance over the last 3 months, USD |
| bal_avg_6m_usd | average month-end balance over the last 6 months, USD |
| bal_slope_6m | monthly change in balance over the last 6 months as a share of the 6-month average (negative = falling) |
| bal_change_vs_6m | current balance divided by the average of the 6 months before (1 = unchanged) |
| bal_drop_from_3m_peak | share of the 3-month peak balance that has gone (0 = at peak, 0.8 = 80% gone) |
| bal_min_3m_ratio | lowest month-end balance of the last 3 months divided by the 6-month average |
| net_flow_1m_usd | money in minus money out this month, USD |
| net_flow_3m_usd | money in minus money out over the last 3 months, USD |
| usd_share | share of the balance held in USD rather than ZiG (0 to 1) |
| usd_share_change_6m | change in the USD share versus 6 months ago (positive = moving into USD) |

## Leakage

| feature | meaning |
| --- | --- |
| wallet_share_1m | share of this month's outflows sent to mobile wallets (EcoCash and similar) |
| wallet_share_3m | share of the last 3 months' outflows sent to mobile wallets |
| wallet_share_trend | wallet share of the last 3 months minus the 3 months before (positive = rising) |
| obank_share_1m | share of this month's outflows sent to other banks |
| obank_share_3m | share of the last 3 months' outflows sent to other banks |
| obank_share_trend | other-bank share of the last 3 months minus the 3 months before |
| leak_share_3m | share of the last 3 months' outflows that left to wallets or other banks combined |
| leak_share_6m | the same share over the last 6 months |
| leak_share_trend | combined leak share, last 3 months minus the 3 months before |
| leak_amount_3m_usd | money sent to wallets and other banks in the last 3 months, USD |
| cash_share_3m | share of the last 3 months' outflows taken as cash |

## Engagement

| feature | meaning |
| --- | --- |
| active_days_1m | days with at least one transaction this month |
| active_days_3m_avg | average active days per month over the last 3 months |
| active_days_change | this month's active days divided by the monthly average of the 6 months before (1 = unchanged) |
| txn_count_1m | transactions this month |
| txn_count_3m | transactions over the last 3 months |
| txn_count_6m_avg | average transactions per month over the last 6 months |
| txn_count_change | this month's transactions divided by the monthly average of the 6 months before |
| txn_3m_vs_prior_3m | transactions in the last 3 months divided by the 3 months before |
| n_credits_1m | credits this month |
| n_debits_1m | debits this month |
| digital_share_3m | share of the last 3 months' transactions made on mobile app, USSD or ZIPIT |
| digital_share_change_6m | digital share now minus 6 months ago |
| branch_share_3m | share of the last 3 months' transactions made in a branch |
| atm_share_3m | share of the last 3 months' transactions at an ATM |
| pos_share_3m | share of the last 3 months' transactions at a card terminal |
| months_since_last_txn | months since the last month with any transaction (0 = this month, capped at 12) |
| tenure_months | months since the customer opened their first account |

## Friction

| feature | meaning |
| --- | --- |
| fees_1m_usd | bank fees charged this month, USD |
| fees_3m_usd | bank fees charged over the last 3 months, USD |
| fee_share_3m | fees as a share of all money out over the last 3 months |
| fee_change | this month's fees divided by the monthly average of the 6 months before (1 = unchanged) |
| reversals_3m | reversed transactions in the last 3 months |
| reversals_6m | reversed transactions in the last 6 months |

## Life Events

| feature | meaning |
| --- | --- |
| remit_months_6m | months in the last 6 with a remittance credit |
| remit_regular | 1 if remittances arrived in at least 4 of the last 6 months |
| remit_amount_3m_usd | remittances received in the last 3 months, USD |
| remit_change | remittances of the last 3 months divided by the 3 months before (1 = unchanged) |
| school_terms_12m | months in the last 12 with a school-fee payment |
| school_amount_12m_usd | school fees paid in the last 12 months, USD |
| school_recent | 1 if school fees were paid in the last 4 months |
| supplier_months_6m | months in the last 6 with a supplier payment |
| supplier_regular | 1 if suppliers were paid in at least 4 of the last 6 months |
| supplier_amount_3m_usd | supplier payments in the last 3 months, USD |
| supplier_share_3m | supplier payments as a share of the last 3 months' outflows |
| salary_growth_6m | typical salary-sized credit of the last 3 months divided by that of 6 to 8 months ago |
| salary_rising | 1 if the salary-sized credit grew by more than 10% over 6 months |

## Target

| column | meaning |
| --- | --- |
| churn_90d | 1 if the customer closed or went silent in the 90 days after the snapshot (empty when the 90 days are not yet over) |
| churn_type | hard (account closed), silent (activity and balance collapsed), or none |
| days_to_churn | days from the snapshot to the closure or to the end of the first silent month; empty if no churn |
| labelled | 1 if a full 90 days of data follow the snapshot, so the label is known |
