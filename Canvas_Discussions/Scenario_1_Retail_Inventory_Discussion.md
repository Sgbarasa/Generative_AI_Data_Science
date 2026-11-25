# Scenario 1: Retail Inventory Analysis

## Problem and my prompting strategy

The scenario was to identify inventory movement, slow-moving products and replenishment risk for a retail chain. I approached this as an analytics engineering task rather than asking AI to generate a generic report: I supplied the complete schema, the business questions, the required Python libraries and expected visualizations, and assigned the model a senior retail data scientist role. The first response was a useful starting point, but I tested the assumptions before accepting it. The example dataset is synthetic because no retailer data was supplied.

## Initial AI prompt

> **Role:** Act as a senior retail analytics data scientist and Python engineer. **Task:** Develop a Jupyter-ready inventory analysis from a pandas DataFrame containing `product_id`, `category`, `stock_level`, `last_restock_date`, `sales_last_30_days`, `supplier_lead_time`, `unit_cost`. Calculate 30-day inventory movement/turnover, flag slow-moving SKUs, forecast days until a potential stockout using recent sales, and visualize actionable findings by category and risk. Use pandas, NumPy, and Matplotlib. Explain assumptions and show the output as readable tables and charts. Include explicit date conversions and comments.

## Initially generated code

```python
import pandas as pd

def first_draft_inventory(df):
    df = df.copy()
    df['last_restock_date'] = pd.to_datetime(df['last_restock_date'])
    df['days_since_restock'] = (pd.Timestamp.today() - df['last_restock_date']).dt.days
    df['turnover'] = df['sales_last_30_days'] / df['stock_level']
    df['days_to_stockout'] = df['stock_level'] / (df['sales_last_30_days'] / 30)
    df['slow_moving'] = df['sales_last_30_days'] < 3
    return df
```

## Follow-up prompts and refinements

**Follow-up A (correctness):** “Refactor the draft to accept a fixed `as_of_date`, reject duplicates, invalid or future dates, negative stock, negative sales, invalid costs and lead times, and handle zero sales/zero stock without division errors. Define any turnover *proxy* honestly; do not call it an audited turnover rate unless historical average inventory is available.”

**Follow-up B (decision support):** “Add an explicit `days_of_supply` estimate using recent units sold per day, a `REORDER_NOW` classification when coverage is no more than supplier lead time, and a transparent slow-moving threshold. Produce three business-friendly charts with axes and titles; add executable assertions and docstrings.”

## Final working solution

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AS_OF = pd.Timestamp('2026-11-25', tz='UTC')
source_path = Path('retail_inventory.csv')

if source_path.exists():
    inventory = pd.read_csv(source_path)
    print(f'Real file loaded: {source_path}')
else:
    inventory = pd.DataFrame([
        ('P100','Electronics',12,'2026-10-02',65,14,120.00),
        ('P101','Electronics',3,'2026-10-15',40,21,75.00),
        ('P102','Electronics',68,'2026-09-30',1,10,40.00),
        ('P103','Home',140,'2026-10-05',4,8,11.00),
        ('P104','Home',28,'2026-11-10',35,7,18.00),
        ('P105','Home',0,'2026-10-02',12,8,9.00),
        ('P106','Beauty',25,'2026-10-04',82,10,22.00),
        ('P107','Beauty',75,'2026-09-15',0,14,13.00),
        ('P108','Beauty',40,'2026-11-01',18,30,16.00),
        ('P109','Grocery',20,'2026-11-14',120,5,5.00),
        ('P110','Grocery',125,'2026-10-01',6,4,3.50),
        ('P111','Grocery',44,'2026-10-03',15,12,4.00),
        ('P112','Sports',45,'2026-09-01',3,20,28.00),
        ('P113','Sports',18,'2026-09-18',33,21,35.00),
    ], columns=['product_id','category','stock_level','last_restock_date',
                'sales_last_30_days','supplier_lead_time','unit_cost'])
    print('Synthetic example data loaded — not actual retailer data')

print(inventory.head(5).to_string(index=False))

def analyze_inventory(df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Validate SKU snapshot and compute labeled replenishment estimates.

    Important: the turnover *proxy* assumes no receipts during the 30-day
    interval. Actual inventory turnover requires historic average inventory.
    """
    required = ['product_id','category','stock_level','last_restock_date',
                'sales_last_30_days','supplier_lead_time','unit_cost']
    missing = [col for col in required if col not in df]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    result = df[required].copy()
    result['product_id'] = result['product_id'].astype('string').str.strip()
    result['category'] = result['category'].astype('string').str.strip()
    if result['product_id'].isna().any() or result['product_id'].eq('').any():
        raise ValueError('Product ID cannot be missing or blank')
    if result['product_id'].duplicated().any():
        raise ValueError('Expected one snapshot row per unique product_id')
    if result['category'].isna().any() or result['category'].eq('').any():
        raise ValueError('Category cannot be missing or blank')

    for col in ['stock_level','sales_last_30_days','supplier_lead_time','unit_cost']:
        result[col] = pd.to_numeric(result[col], errors='coerce')
        if result[col].isna().any() or not np.isfinite(result[col].to_numpy()).all():
            raise ValueError(f'{col} must contain finite numeric values')
        if (result[col] < 0).any():
            raise ValueError(f'{col} cannot be negative')
    for col in ['stock_level','sales_last_30_days','supplier_lead_time']:
        if not np.equal(result[col], np.floor(result[col])).all():
            raise ValueError(f'{col} must contain non-negative integers')

    as_of = pd.Timestamp(as_of_date)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize('UTC')
    else:
        as_of = as_of.tz_convert('UTC')
    result['last_restock_date'] = pd.to_datetime(
        result['last_restock_date'], utc=True, errors='coerce')
    if result['last_restock_date'].isna().any():
        raise ValueError('last_restock_date must contain valid dates')
    if (result['last_restock_date'] > as_of).any():
        raise ValueError('Restock dates cannot be in the future')

    result['days_since_restock'] = (as_of - result['last_restock_date']).dt.days
    result['daily_units_sold_30d'] = result['sales_last_30_days'] / 30.0
    velocity = result['daily_units_sold_30d'].to_numpy(dtype=float)
    stock = result['stock_level'].to_numpy(dtype=float)
    result['days_of_supply'] = np.divide(stock, velocity,
        out=np.full(len(result), np.inf), where=velocity > 0)
    # Zero inventory always means out-of-stock, even if sales are also zero.
    result.loc[result['stock_level'].eq(0), 'days_of_supply'] = 0.0
    result['slow_moving'] = (result['stock_level'] > 0) & (
        (result['sales_last_30_days'] <= 2) | (result['days_of_supply'] >= 180))
    result['reorder_units_14d_buffer'] = np.maximum(0, np.ceil(
        result['daily_units_sold_30d'] * (result['supplier_lead_time'] + 14)
        - result['stock_level'])).astype(int)
    result['inventory_value'] = result['stock_level'] * result['unit_cost']
    result['replenishment_status'] = np.select([
        result['stock_level'].eq(0),
        result['days_of_supply'].le(result['supplier_lead_time']),
        result['days_of_supply'].le(result['supplier_lead_time'] + 7),
        result['slow_moving'],
    ], ['OUT_OF_STOCK','REORDER_NOW','WATCH','SLOW_MOVING'], default='OK')

    # Illustrative *estimate* only: assumes opening stock = ending stock + sales,
    # i.e., no receipts, returns, or losses during the last 30 days.
    estimated_avg_units = result['stock_level'] + result['sales_last_30_days'] / 2
    result['estimated_turnover_30d_proxy'] = np.divide(
        result['sales_last_30_days'], estimated_avg_units,
        out=np.zeros(len(result), dtype=float), where=estimated_avg_units > 0)
    result['restocked_within_30d'] = result['days_since_restock'] < 30
    result['proxy_assumption_violated_or_unknown'] = result['restocked_within_30d']
    return result

analysis = analyze_inventory(inventory, AS_OF)
print(analysis[['product_id','category','estimated_turnover_30d_proxy',
  'days_of_supply','replenishment_status','slow_moving']].to_string(index=False))

# Assertions test successful calculations and failure cases.
assert set(['OUT_OF_STOCK','REORDER_NOW']).issubset(set(analysis['replenishment_status']))
assert (analysis['estimated_turnover_30d_proxy'] >= 0).all()
assert analysis.loc[analysis.product_id.eq('P105'), 'days_of_supply'].item() == 0
assert np.isinf(analysis.loc[analysis.product_id.eq('P107'), 'days_of_supply'].item())
assert analysis['reorder_units_14d_buffer'].ge(0).all()
try:
    invalid = inventory.copy()
    invalid.loc[0, 'supplier_lead_time'] = -1
    analyze_inventory(invalid, AS_OF)
    raise AssertionError('Validation should have rejected negative lead time')
except ValueError:
    pass
print('Inventory analysis tests passed')

# Three management-ready visual summaries.
category = analysis.groupby('category', as_index=False).agg(
    avg_proxy_turnover=('estimated_turnover_30d_proxy','mean'),
    inventory_value=('inventory_value','sum'))
ax = category.sort_values('avg_proxy_turnover').plot.barh(
    x='category', y='avg_proxy_turnover', legend=False, figsize=(9,4))
ax.set_title('Illustrative 30-day turnover proxy by category')
ax.set_xlabel('Units sold / estimated average units (proxy)')
ax.set_ylabel('Category')
plt.tight_layout(); plt.show()

status = analysis['replenishment_status'].value_counts().sort_values()
ax = status.plot.barh(figsize=(9,4))
ax.set_title('SKU count by replenishment status')
ax.set_xlabel('Number of products')
ax.set_ylabel('Status')
plt.tight_layout(); plt.show()

slow = analysis[analysis['slow_moving']].nlargest(8, 'inventory_value')
if not slow.empty:
    ax = slow.sort_values('inventory_value').plot.barh(
        x='product_id', y='inventory_value', legend=False, figsize=(9,4))
    ax.set_title('Slow-moving SKUs: capital tied up at unit cost')
    ax.set_xlabel('Inventory carrying value (dataset currency)')
    ax.set_ylabel('Product ID')
    plt.tight_layout(); plt.show()
```

## Results, validation and reflection

The initial code calculated `sales_last_30_days / stock_level` and called it "turnover." That could be misleading: audited turnover requires **cost of goods sold divided by average inventory value** over a defined period. I changed the metric name to `estimated_turnover_30d_proxy`, labeled the calculation's assumptions, and added days-of-supply and supplier lead-time comparisons. The analysis flags zero-stock items as `OUT_OF_STOCK`, items whose remaining coverage is less than or equal to supplier lead time as `REORDER_NOW`, and slow-moving items using an explicit sales threshold.

The first draft also relied on the machine's current date, which would make runs inconsistent, and could divide by zero for unsold or out-of-stock items. The final version uses an explicit **2026-11-25 UTC as-of date**, validates product IDs and input types, rejects negative values and future restock dates, and handles zero-stock/zero-sales cases. The tests include invalid inputs and calculations. On the synthetic inventory, the implementation correctly identifies urgent replenishment items and slow-moving SKUs; the executable assertions passed.

I used three different visualizations, each with one decision purpose: a category-level turnover-proxy comparison, replenishment-status counts, and slow-moving inventory value by product. This is better than combining several metrics into an unreadable chart. My main lesson was that stronger prompts are not just more specific: they force the AI to state assumptions, handle edge cases and distinguish a heuristic from a defensible business measure.

For forecasting beyond these heuristics, I would request longer historical sales, receipts, beginning and ending inventory, lead-time variability and seasonal demand. What historical data would you prioritize to move from simple days-of-supply flags to a calibrated stockout probability?
