# Scenario 3: Customer Segmentation with SQL

## Problem and my prompting strategy

The objective was to find recently active customers in the top 20% of spenders and examine their communication preferences for a product rollout. I supplied the precise three-table schema, eligibility window, ranking cohort and output fields. I first tested the generated SQL against these requirements, then prompted for a fixed as-of date, SQLite compatibility, deterministic ranking, safer joins and query optimization. Since no customer database was supplied, I tested the final query on a synthetic in-memory SQLite dataset.

## Initial AI prompt

> **Role:** Act as a senior product analytics engineer and SQL developer. Using the exact tables and columns above, write SQL to identify users active during the last 30 days, rank customers by total purchase amount, select the highest-spending 20% of those active users, and report their communication, interface and notification preferences. Explain how the date window and percentile selection work. Use CTEs for readability and propose a visualization-friendly output.

## Initially generated code

```sql
SELECT a.user_id, SUM(t.amount) AS total_spend,
       p.communication_preference, p.interface_theme
FROM user_activity a
JOIN transactions t ON a.user_id = t.user_id
JOIN user_preferences p ON p.user_id = a.user_id
WHERE a.last_login_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY a.user_id, p.communication_preference, p.interface_theme
ORDER BY total_spend DESC;
```

## Follow-up prompts and refinements

**Follow-up A:** “Rewrite for SQLite so it can run in a standalone notebook using `sqlite3`. Parameterize an `as_of_date`; use ISO-8601 date strings and **exactly 30 calendar dates including the as-of date**. First filter active users, then aggregate their completed transaction amounts for the same window, and only then rank users. Select `ceil(0.20 × active_user_count)` using a deterministic tie-break on `user_id`. Keep preference-free customers by a `LEFT JOIN`.”

**Follow-up B:** “Explain the CTEs, correct the risks of duplicate transaction counting, include a separate query for preference distributions, and add performance-oriented indexes. Keep filters sargable (avoid applying functions to indexed date columns) and test against synthetic data with one inactive big spender and an out-of-window transaction.”

## Final working solution

```python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

AS_OF_DATE = '2026-11-25'  # fixed for reproducible demonstrations
con = sqlite3.connect(':memory:')
con.executescript("""
CREATE TABLE user_activity (
    user_id INTEGER PRIMARY KEY,
    last_login_date TEXT NOT NULL,
    feature_usage_count INTEGER NOT NULL,
    account_type TEXT NOT NULL
);
CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    transaction_date TEXT NOT NULL,
    amount REAL NOT NULL,
    platform TEXT NOT NULL
);
CREATE TABLE user_preferences (
    user_id INTEGER PRIMARY KEY,
    communication_preference TEXT,
    interface_theme TEXT,
    notification_settings TEXT
);
CREATE INDEX ix_user_login ON user_activity(last_login_date);
CREATE INDEX ix_txn_user_date ON transactions(user_id, transaction_date);
""")

activity = [
 (1,'2026-11-25',21,'premium'), (2,'2026-11-23',18,'free'),
 (3,'2026-11-19',14,'premium'), (4,'2026-11-21',4,'free'),
 (5,'2026-11-22',8,'free'), (6,'2026-11-18',2,'free'),
 (7,'2026-11-24',11,'premium'), (8,'2026-11-02',6,'free'),
 (9,'2026-11-12',5,'free'), (10,'2026-11-25',8,'free'),
 (11,'2026-11-04',3,'free'),
 (12,'2026-08-20',50,'premium'),  # deliberately inactive large spender
]
transactions = [
 (100,1,'2026-11-20',900,'web'),(101,1,'2026-11-23',700,'app'),
 (102,2,'2026-11-18',500,'web'),(103,2,'2026-11-22',300,'app'),
 (104,3,'2026-11-20',1000,'app'),
 (105,4,'2026-11-23',100,'web'),(106,4,'2026-11-24',200,'app'),
 (107,5,'2026-11-22',50,'web'),
 (108,7,'2026-11-20',750,'app'),(109,8,'2026-11-12',230,'web'),
 (110,9,'2026-11-11',120,'app'),(111,10,'2026-11-10',60,'web'),
 (112,11,'2026-11-04',30,'app'),
 (113,12,'2026-11-22',2400,'web'),  # exclude inactive user
 (114,2,'2026-10-01',3000,'app'),  # exclude old spending
 (115,7,'2026-12-01',10000,'app'), # exclude future spending
]
prefs = [
 (1,'email','dark','on'),(2,'sms','light','off'),
 # user 3 has no row: exercise LEFT JOIN and Unspecified
 (4,'email','light','on'),(5,'push','dark','on'),
 (6,'sms','light','off'),(7,'push','dark','on'),
 (8,'email','light','off'),(9,'push','light','on'),
 (10,'sms','light','on'),(11,'email','dark','on'),
 (12,'email','dark','on'),
]
con.executemany('INSERT INTO user_activity VALUES (?,?,?,?)', activity)
con.executemany('INSERT INTO transactions VALUES (?,?,?,?,?)', transactions)
con.executemany('INSERT INTO user_preferences VALUES (?,?,?,?)', prefs)
con.commit()
print('Synthetic SQLite demo database loaded')

cte_sql = """
WITH active_users AS (
    SELECT user_id, account_type, feature_usage_count, last_login_date
    FROM user_activity
    WHERE last_login_date >= date(:as_of_date, '-29 days')
      AND last_login_date < date(:as_of_date, '+1 day')
),
spend_30d AS (
    SELECT t.user_id, SUM(t.amount) AS spend_30d,
           COUNT(DISTINCT t.transaction_id) AS transactions_30d
    FROM transactions AS t
    INNER JOIN active_users AS a ON a.user_id = t.user_id
    WHERE t.transaction_date >= date(:as_of_date, '-29 days')
      AND t.transaction_date < date(:as_of_date, '+1 day')
    GROUP BY t.user_id
),
ranked AS (
    SELECT a.user_id, a.account_type, a.feature_usage_count,
           a.last_login_date,
           COALESCE(s.spend_30d, 0) AS spend_30d,
           COALESCE(s.transactions_30d, 0) AS transactions_30d,
           ROW_NUMBER() OVER (
               ORDER BY COALESCE(s.spend_30d, 0) DESC, a.user_id ASC
           ) AS spend_rank,
           COUNT(*) OVER () AS active_user_count
    FROM active_users AS a
    LEFT JOIN spend_30d AS s ON s.user_id = a.user_id
),
high_value AS (
    SELECT * FROM ranked
    WHERE spend_rank <= CAST((active_user_count + 4) / 5 AS INTEGER)
      AND spend_30d > 0
)
"""

sql_customers = cte_sql + """
SELECT h.user_id, h.account_type, h.feature_usage_count,
       h.last_login_date, h.spend_30d, h.transactions_30d,
       h.spend_rank, h.active_user_count,
       COALESCE(p.communication_preference, 'Unspecified') AS communication_preference,
       COALESCE(p.interface_theme, 'Unspecified') AS interface_theme,
       COALESCE(p.notification_settings, 'Unspecified') AS notification_settings
FROM high_value AS h
LEFT JOIN user_preferences AS p ON p.user_id = h.user_id
ORDER BY h.spend_rank, h.user_id
"""

sql_preferences = cte_sql + """
SELECT COALESCE(p.communication_preference, 'Unspecified') AS communication_preference,
       COALESCE(p.interface_theme, 'Unspecified') AS interface_theme,
       COALESCE(p.notification_settings, 'Unspecified') AS notification_settings,
       COUNT(*) AS high_value_customers,
       ROUND(AVG(h.spend_30d), 2) AS avg_customer_spend_30d
FROM high_value AS h
LEFT JOIN user_preferences AS p ON p.user_id = h.user_id
GROUP BY COALESCE(p.communication_preference, 'Unspecified'),
         COALESCE(p.interface_theme, 'Unspecified'),
         COALESCE(p.notification_settings, 'Unspecified')
ORDER BY high_value_customers DESC, communication_preference
"""

customers = pd.read_sql_query(sql_customers, con, params={'as_of_date':AS_OF_DATE})
preference_trends = pd.read_sql_query(sql_preferences, con, params={'as_of_date':AS_OF_DATE})
print('Top spending active customers:')
print(customers.to_string(index=False))
print('\nPreference trends:')
print(preference_trends.to_string(index=False))

# Validate cohort, 30-day window, IDs, non-duplication, and rank logic.
assert list(customers['user_id']) == [1,3,2], customers['user_id'].tolist()
assert customers['user_id'].is_unique
assert len(customers) == 3   # ceil(0.2 * 11 active users) = 3
assert customers.loc[customers['user_id'].eq(2),'spend_30d'].item() == 800.0
assert 12 not in customers['user_id'].tolist()  # inactive large spender excluded
assert 7 not in customers['user_id'].tolist()   # future transaction excluded
assert customers.loc[customers['user_id'].eq(3),'communication_preference'].item() == 'Unspecified'
assert preference_trends['high_value_customers'].sum() == len(customers)
print('SQL cohort and metric tests passed')

print('\nIndexed access / query plan (SQLite):')
for row in con.execute('EXPLAIN QUERY PLAN ' + sql_customers,
                       {'as_of_date':AS_OF_DATE}).fetchall()[-10:]:
    print(row[3])

# Visualization-friendly trend summaries for selected top spenders.
if not customers.empty:
    ax = customers.sort_values('spend_30d').plot.barh(
        x='user_id', y='spend_30d', legend=False, figsize=(8,4))
    ax.set_title('High-value active customers: 30-day spend (synthetic data)')
    ax.set_xlabel('Spend (example currency units)')
    ax.set_ylabel('User ID')
    plt.tight_layout(); plt.show()

    pref_counts = customers['communication_preference'].value_counts()
    ax = pref_counts.plot.bar(figsize=(8,4))
    ax.set_title('Communication preferences among selected customers')
    ax.set_xlabel('Preference')
    ax.set_ylabel('Number of customers')
    plt.xticks(rotation=0)
    plt.tight_layout(); plt.show()
```

## Results, validation and reflection

The initial query sorted customers by spending but **never enforced the top-20% cutoff**. It also filtered logins without restricting transaction dates, so old purchases could distort a 30-day targeting decision. The refinement first identifies the active cohort, aggregates spending in the same 30-calendar-date interval, ranks those active users using `ROW_NUMBER() OVER (ORDER BY spend_30d DESC, user_id ASC)`, and selects the top `ceil(0.20 × active_user_count)` users. This means 20% of **eligible customers by headcount**, not 20% of revenue. The user-ID tie-break makes results reproducible.

The SQLite implementation uses a parameterized **2026-11-25 as-of date**, with lower-inclusive and upper-exclusive boundaries. It structures the logic into CTEs for active users, 30-day spending, ranking and cohort selection. I aggregate transactions **before** joining preference data to avoid inflating spending totals, use `LEFT JOIN` to preserve users without recorded preferences, and create indexes supporting the date filters and joins. A query-plan inspection helps evaluate the actual access paths; a CTE alone does not guarantee a performance improvement.

On the synthetic database, there were **11 active users** and **3 selected users** after rounding up 20% of 11. Their 30-day spending was **1,600**, **1,000** and **800**, respectively. Edge-case assertions verified that an inactive high spender, old transactions and future transactions did not enter the cohort, and that the results had unique customer IDs. The two charts summarize selected-customer spending and communication preferences.

The major learning was that SQL prompting requires precise definitions of the cohort, ranking threshold, date horizon, join cardinality and database dialect. A SQL statement can be syntactically valid yet answer the wrong business question. For rollout design, would you preserve an exact cohort size with deterministic tie-breaking, or admit all customers tied at the cutoff?
