# Scenario 1: Retail Inventory Analysis

I used a role-based prompt because I wanted the AI to approach the problem as a retail analyst, not just write a few calculations.

**Initial prompt:**
> Act as a senior retail data analyst. Using `product_id`, `category`, `stock_level`, `last_restock_date`, `sales_last_30_days`, `supplier_lead_time` and `unit_cost`, write pandas/NumPy code to estimate stock movement, identify slow-moving products and possible stockouts, and plot the results with Matplotlib. Explain the assumptions.

The first code it produced included:

```python
df['turnover'] = df['sales_last_30_days'] / df['stock_level']
df['days_to_stockout'] = df['stock_level'] / (df['sales_last_30_days'] / 30)
df['days_since_restock'] = (pd.Timestamp.today() - df['last_restock_date']).dt.days
```

The problem was that it divided by zero when stock or sales were zero, and it called a rough movement measure “turnover” without historical inventory data.

**Follow-up prompt:**
> Use a fixed date, validate quantities and restock dates, handle zero values safely and add supplier lead-time checks. Call the turnover calculation a proxy and include tests and three charts.

The revised code validates inputs, estimates days of supply and flags products that need reordering. I tested a negative lead time to confirm the validation works. The charts show movement by category, reorder status and stock value tied up in slow sellers.

My biggest takeaway was that AI can generate working code while still making a misleading business assumption. I'd want more sales history before relying on the stockout estimates. Would you use seasonality to improve the reorder rules?
