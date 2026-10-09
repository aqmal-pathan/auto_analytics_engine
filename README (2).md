# Assignment 4 — Automated Insight Generation

A general-purpose **Auto-Analytics Engine** built with Python, pandas and Streamlit.
It loads a district-level healthcare CSV and automatically produces **trend, outlier,
correlation and threshold-breach insights** as structured rows plus plain-English explanations.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads `data.csv` from the same folder. You can also upload any CSV with the same shape
from the sidebar (`month`, `district`, then any number of numeric indicator columns).

## Files

| File | Purpose |
|---|---|
| `app.py` | Full app: loading/validation, detectors, insight generator, UI |
| `data.csv` | Sample dataset (6 districts × 2 months) |
| `requirements.txt` | Python dependencies |
| `sample_insights.json` | Example output with default settings |

## How it works

| Part | Method |
|---|---|
| A. Validation | `head()`, `info()` and missing-value counts are printed to the console and shown in the UI. Bad rows and duplicates are reported. |
| B. Trend | `pct_change = (current - previous) / previous * 100` per (district, indicator). Flagged when `abs(pct_change) >= threshold` (default 10%). |
| C. Outlier | **IQR rule** (default, multiplier k=1.5) or **Z-score** (default threshold 3), selectable in the UI. |
| D. Correlation | Pearson matrix; pairs with `abs(r) >= 0.70` are flagged. |
| E. Insights | Fields: `insight_id, type, indicator, entity, period, value, prev_value, change_pct, severity, explanation`. Types: `trend`, `outlier`, `correlation`, `threshold_breach`. |
| F. UI | Live filters (district, month, indicator), threshold sliders, insight table, severity bar chart, correlation heatmap, per-district line chart, CSV/JSON downloads. |

Explanations are templates, but **every number comes from the data**. No district or indicator
name is hardcoded; indicators are auto-detected from the CSV's numeric columns.

### Severity (derived from the configurable thresholds)

* **Trend / threshold breach:** `ratio = |change| / trend_threshold`.
* **Outlier:** IQR: how many IQRs the value sits beyond the nearest quartile divided by k. Z-score: `|z| / z_threshold`.
* Low/Medium/High cut-offs are multiples of the threshold (defaults 1.25x and 1.75x) and are adjustable in the sidebar.
* **Correlation:** how far `|r|` sits between the threshold and 1 (thirds: Low / Medium / High).

## Notes and limitations

* **Correlations are fragile on this dataset.** The sample has only 2 months x 6 districts = 12 rows.
  Pearson correlations on so few points can be driven by a single district (here, Mehsana), so the
  matrix is shown as computed and should be read as indicative only. The brief recommends >= 10 districts and >= 3 months.
* **Z-score vs IQR on small samples:** with n = 12, a Z-score cannot easily reach 3, so the Mehsana ANC
  value (z about 2.8) is not flagged at the default threshold of 3. IQR is therefore the default method;
  Z-score remains available (try 2.5).
* **Small counts inflate % change:** `high_risk_cases` is a raw count, so small absolute moves
  look large in percentage terms.
* Trends compare each month with the previous available month in the filtered data.
  A zero baseline has no defined % change and is skipped.
