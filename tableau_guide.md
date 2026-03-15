# 📊 Tableau Dashboard Setup Guide

This guide explains how to build the interactive Tableau dashboard using
the CSV exports produced by the Jupyter notebook.

---

## Step 1 – Export Data from Notebook

Add the following cell at the end of the notebook to export CSVs:

```python
# Export aggregated data for Tableau
monthly.to_csv('../visualizations/tableau_monthly.csv', index=False)
cust_df.to_csv('../visualizations/tableau_customers.csv', index=False)
df[['category','sent_pred','total_amount','rating']].to_csv(
    '../visualizations/tableau_sentiment.csv', index=False)
df.groupby('category')['total_amount'].sum().reset_index().to_csv(
    '../visualizations/tableau_products.csv', index=False)
print("Tableau CSVs exported.")
```

---

## Step 2 – Connect Data Sources in Tableau

1. Open **Tableau Desktop** (or Tableau Public – free).
2. Click **Connect → Text File** and import each CSV:
   - `tableau_monthly.csv`
   - `tableau_customers.csv`
   - `tableau_sentiment.csv`
   - `tableau_products.csv`

---

## Step 3 – Build the Four Worksheets

### Sheet 1 · Sales Trends
- **Columns**: `period` (set to Month granularity)
- **Rows**: `SUM(revenue)`
- **Mark type**: Line
- Add a **Trend Line** (Analytics pane → Trend Line)
- Dual axis: add `SUM(orders)` as bars on secondary axis

### Sheet 2 · Customer Segments
- Data source: `tableau_customers.csv`
- **Columns**: `total_spend`
- **Rows**: `num_orders`
- **Color**: `cluster` (discrete)
- **Size**: `avg_order_val`
- Mark type: Circle (Scatter plot)

### Sheet 3 · Sentiment Distribution
- Data source: `tableau_sentiment.csv`
- **Rows**: `sent_pred`
- **Columns**: `COUNT(sent_pred)`
- Mark type: Bar, colored by `sent_pred`
- Add a secondary sheet as a Pie chart using `CNT` measure

### Sheet 4 · Product Performance
- Data source: `tableau_products.csv`
- **Rows**: `category`
- **Columns**: `total_amount`
- Mark type: Bar (sorted descending)
- Color by `total_amount` (continuous gradient)

---

## Step 4 – Assemble the Dashboard

1. Click **New Dashboard** (bottom tabs).
2. Drag and drop all four sheets into the layout grid.
3. Suggested layout:
   ```
   ┌─────────────────────┬───────────────────┐
   │   Sales Trends      │ Customer Segments │
   ├─────────────────────┼───────────────────┤
   │ Sentiment Dist.     │ Product Perf.     │
   └─────────────────────┴───────────────────┘
   ```
4. Add a **Dashboard Title**: "E-Commerce Customer Behavior Dashboard"
5. Add **Filters** (drag sheet filters to dashboard):
   - Category filter (affects all sheets via "Apply to all worksheets")
   - Date range filter on Sales Trends

---

## Step 5 – Publish (Optional)

- **Tableau Public**: File → Save to Tableau Public
- **Tableau Server / Cloud**: Server → Publish Workbook

---

## Alternative: Plotly Dash

If you prefer a fully Python-based dashboard, replace this guide with
`plotly` + `dash`:

```bash
pip install dash plotly
```

Then run:

```python
# In a new file: dashboard_app.py
import dash
from dash import dcc, html
import plotly.express as px
# ... (extend as needed)
```
