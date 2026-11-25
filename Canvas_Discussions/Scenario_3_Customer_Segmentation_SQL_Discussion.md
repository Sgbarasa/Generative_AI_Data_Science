# Scenario 3: Customer Segmentation with SQL

For this scenario, I found that giving AI the full table structure was important. Otherwise, it could write valid SQL that did not answer the customer-targeting question.

**Initial prompt:**
> Using `user_activity`, `transactions` and `user_preferences`, write SQL to find customers active in the last 30 days, rank them by spending, select the highest-spending 20% and summarize their communication preferences. Use CTEs and explain the query.

The first AI query grouped transactions and sorted the results:

```sql
SELECT a.user_id, SUM(t.amount) AS total_spend
FROM user_activity a
JOIN transactions t ON a.user_id = t.user_id
WHERE a.last_login_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY a.user_id
ORDER BY total_spend DESC;
```

It missed two important requirements: **the top-20% cutoff** and the transaction-date filter.

**Follow-up prompt:**
> Use SQLite and a fixed analysis date. Filter logins and purchases to the same 30-day window, rank active users by spending, select the top 20% by customer count, keep missing preferences and test the edge cases. Explain the CTEs and any query optimizations.

I ended up with separate CTEs for the active users, spending totals, ranking and final selection. The query aggregates spending before joining preferences to avoid duplicated totals. It also uses indexes and checks the query plan.

On the synthetic data, **11 customers were active and 3 were selected**. Tests confirmed that an inactive big spender and out-of-window purchases did not affect the list. The hardest decision was defining “top 20%” clearly. Would you include ties at the cutoff or stick to a fixed cohort size?
