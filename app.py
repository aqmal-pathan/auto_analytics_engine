"""
Healthcare Auto-Analytics Engine
--------------------------------
Ingests a district-level healthcare CSV and automatically produces
trend, outlier, correlation and threshold-breach insights.

Run:  streamlit run app.py

Design rules :
  * General purpose: no district names, indicator names or numbers are hardcoded.
    Every number in an explanation comes from the data.
  * Every threshold is a UI slider; severity is derived from those thresholds.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Auto-Analytics Engine", layout="wide")

DATA_PATH = Path(__file__).parent / "data.csv"
ID_COLS = ["month", "district"]
SEVERITY_ORDER = ["Low", "Medium", "High"]
SEVERITY_COLORS = {"Low": "#2ca02c", "Medium": "#ff9f1c", "High": "#d62728"}
TYPE_ORDER = ["trend", "outlier", "correlation", "threshold_breach"]
INSIGHT_COLUMNS = [
    "insight_id", "type", "indicator", "entity", "period",
    "value", "prev_value", "change_pct", "severity", "explanation",
]


# ==========================================================================
# PART A: Data loading & validation
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_data(source):
    """Load + validate the CSV. `source` is a file path (str) or raw bytes."""
    raw = pd.read_csv(io.BytesIO(source) if isinstance(source, bytes) else source)

    missing_cols = [c for c in ID_COLS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing_cols)}")

    warnings = []
    df = raw.copy()
    df["month_dt"] = pd.to_datetime(df["month"], errors="coerce")
    df["district"] = (
        df["district"].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    )

    bad = df["month_dt"].isna() | df["district"].isna()
    if bad.any():
        warnings.append(f"Dropped {int(bad.sum())} row(s) with an unreadable month or district.")
        df = df[~bad]
    df["month"] = df["month_dt"].dt.strftime("%Y-%m")

    # Every other column is treated as an indicator if it is numeric.
    indicators = []
    for col in df.columns:
        if col in ID_COLS or col == "month_dt":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().any():
            indicators.append(col)
    if not indicators:
        raise ValueError("No numeric indicator columns found in the CSV.")

    dups = df.duplicated(["district", "month"], keep="last")
    if dups.any():
        warnings.append(
            f"Found {int(dups.sum())} duplicate (district, month) row(s); kept the last of each."
        )
        df = df[~dups]

    df = df.sort_values(["district", "month_dt"]).reset_index(drop=True)

    clean = df[ID_COLS + indicators]
    buf = io.StringIO()
    clean.info(buf=buf)
    missing = clean.isnull().sum().rename("missing_values").to_frame()

    # Console report (assignment Part A)
    print("--- head() ---\n", clean.head())
    print("--- info() ---\n", buf.getvalue())
    print("--- missing values per column ---\n", missing["missing_values"])

    report = {
        "head": clean.head(),
        "info": buf.getvalue(),
        "missing": missing,
        "warnings": warnings,
        "shape": clean.shape,
    }
    return df, indicators, report


# ==========================================================================
# Severity helpers: always derived from the user-configurable thresholds
# ==========================================================================
def severity_from_ratio(ratio, medium_x, high_x):
    """ratio = how many multiples of the threshold the finding reaches."""
    if ratio >= high_x:
        return "High"
    if ratio >= medium_x:
        return "Medium"
    return "Low"


def severity_from_strength(strength):
    """strength in [0, 1] = how far a finding sits between threshold and maximum."""
    if strength >= 2 / 3:
        return "High"
    if strength >= 1 / 3:
        return "Medium"
    return "Low"


# ==========================================================================
# PART B: Trend detection (month-over-month % change)
# ==========================================================================
def detect_trends(df, indicators, threshold, medium_x, high_x):
    rows = []
    for dist, g in df.groupby("district"):
        g = g.sort_values("month_dt")
        prev_month = g["month"].shift(1)
        for ind in indicators:
            prev = g[ind].shift(1)
            # A zero baseline has no defined % change, so it is skipped.
            pct = (g[ind] - prev) / prev.where(prev != 0) * 100
            for i, change in pct[pct.abs() >= threshold].items():
                direction = "increased" if change > 0 else "dropped"
                rows.append({
                    "type": "trend", "indicator": ind, "entity": dist,
                    "period": g.at[i, "month"],
                    "value": g.at[i, ind], "prev_value": prev[i],
                    "change_pct": round(change, 1),
                    "severity": severity_from_ratio(abs(change) / threshold, medium_x, high_x),
                    "explanation": (
                        f"{ind} in {dist} {direction} by {abs(change):.1f}% compared to the "
                        f"previous month ({prev_month[i]}), exceeding the {threshold:g}% "
                        f"significant-change threshold."
                    ),
                })
    return rows


# ==========================================================================
# PART C: Outlier detection (IQR rule or Z-score)
# ==========================================================================
def detect_outliers(df, indicators, method, z_thr, iqr_k, medium_x, high_x):
    rows = []
    for ind in indicators:
        s = df[ind].dropna()
        if len(s) < 4:
            continue

        if method == "IQR":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
            ref = s.median()
            for idx, val in s[(s < lo) | (s > hi)].items():
                below = val < lo
                # How many IQRs the value sits beyond the nearest quartile,
                # relative to the multiplier k: ratio >= 1 means "outlier".
                extremity = ((q1 - val) if below else (val - q3)) / iqr
                explanation = (
                    f"{df.at[idx, 'district']}'s {ind} of {val:g} is "
                    f"{'below the lower' if below else 'above the upper'} outlier fence "
                    f"({lo if below else hi:.1f}); the typical (median) value is {ref:g} "
                    f"(Q1={q1:.1f}, Q3={q3:.1f}), flagging for review."
                )
                rows.append(_outlier_row(df, idx, ind, val, ref, extremity / iqr_k,
                                         medium_x, high_x, explanation))
        else:  # Z-score
            mean, std = s.mean(), s.std()
            if not std or np.isnan(std):
                continue
            z = (s - mean) / std
            for idx, zv in z[z.abs() >= z_thr].items():
                val = s[idx]
                explanation = (
                    f"{df.at[idx, 'district']}'s {ind} of {val:g} is {abs(zv):.2f}σ "
                    f"{'below' if zv < 0 else 'above'} the mean ({mean:.1f}), flagging for review."
                )
                rows.append(_outlier_row(df, idx, ind, val, mean, abs(zv) / z_thr,
                                         medium_x, high_x, explanation))
    return rows


def _outlier_row(df, idx, ind, val, ref, ratio, medium_x, high_x, explanation):
    return {
        "type": "outlier", "indicator": ind, "entity": df.at[idx, "district"],
        "period": df.at[idx, "month"], "value": val,
        "prev_value": round(float(ref), 1),  # reference level (mean or median)
        "change_pct": round((val - ref) / ref * 100, 1) if ref else np.nan,
        "severity": severity_from_ratio(ratio, medium_x, high_x),
        "explanation": explanation,
    }


# ==========================================================================
# PART D: Correlation detection (Pearson)
# ==========================================================================
def detect_correlations(df, indicators, threshold):
    data = df[indicators]
    corr = data.corr(method="pearson")
    n_rows, n_dist, n_months = len(data), df["district"].nunique(), df["month"].nunique()
    period = f"{df['month'].min()}..{df['month'].max()}"

    rows = []
    for i in range(len(indicators)):
        for j in range(i + 1, len(indicators)):
            r = corr.iloc[i, j]
            if pd.isna(r) or abs(r) < threshold:
                continue
            strength = (abs(r) - threshold) / max(1 - threshold, 1e-9)
            caution = (
                f" Based on only {n_rows} rows ({n_dist} districts × {n_months} months), "
                f"so treat as indicative, not conclusive."
                if n_dist < 10 else ""
            )
            rows.append({
                "type": "correlation",
                "indicator": f"{indicators[i]}:{indicators[j]}",
                "entity": "All districts", "period": period,
                "value": round(r, 2), "prev_value": np.nan, "change_pct": np.nan,
                "severity": severity_from_strength(strength),
                "explanation": (
                    f"{indicators[i]} and {indicators[j]} show a strong "
                    f"{'negative' if r < 0 else 'positive'} correlation (r={r:.2f})."
                    + caution
                ),
            })
    return rows, corr, (n_rows, n_dist, n_months)


# ==========================================================================
# Threshold breach (optional absolute limits set in the UI)
# ==========================================================================
def detect_breaches(df, limits, trend_thr, medium_x, high_x):
    rows = []
    for ind, (lo, hi) in limits.items():
        for idx, val in df[ind].dropna().items():
            for limit, is_min in ((lo, True), (hi, False)):
                if limit is None:
                    continue
                if (is_min and val < limit) or (not is_min and val > limit):
                    pct = (val - limit) / abs(limit) * 100 if limit else np.nan
                    ratio = abs(pct) / trend_thr if not np.isnan(pct) else 0
                    rows.append({
                        "type": "threshold_breach", "indicator": ind,
                        "entity": df.at[idx, "district"], "period": df.at[idx, "month"],
                        "value": val, "prev_value": limit,
                        "change_pct": round(pct, 1) if not np.isnan(pct) else np.nan,
                        "severity": severity_from_ratio(ratio, medium_x, high_x),
                        "explanation": (
                            f"{df.at[idx, 'district']}'s {ind} of {val:g} is "
                            f"{'below the minimum' if is_min else 'above the maximum'} limit of "
                            f"{limit:g}" + (f" ({abs(pct):.1f}% beyond the limit)." if not np.isnan(pct) else ".")
                        ),
                    })
    return rows


# ==========================================================================
# PART F: UI
# ==========================================================================
st.title("Healthcare Auto-Analytics Engine")
st.caption("Upload or load a district-level CSV and get trends, outliers, correlations and "
           "plain-English insights automatically.")

# ---- Data source ---------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type="csv")
if uploaded is not None:
    source = uploaded.getvalue()
elif DATA_PATH.exists():
    source = str(DATA_PATH)
else:
    st.error("data.csv not found next to app.py. Upload a CSV from the sidebar.")
    st.stop()

try:
    df, indicators, report = load_data(source)
except Exception as exc:  # surface bad files in the UI instead of a traceback
    st.error(f"Could not load the data: {exc}")
    st.stop()

with st.expander("Data validation report (Part A)", expanded=False):
    for w in report["warnings"]:
        st.warning(w)
    st.write(f"Shape: {report['shape'][0]} rows × {report['shape'][1]} columns")
    st.markdown("**head()**")
    st.dataframe(report["head"])
    st.markdown("**info()**")
    st.code(report["info"])
    st.markdown("**Missing values per column**")
    st.dataframe(report["missing"])

# ---- Sidebar: filters ----------------------------------------------------
st.sidebar.header("Data filters")
all_districts = sorted(df["district"].unique())
all_months = sorted(df["month"].unique())
sel_districts = st.sidebar.multiselect("District", all_districts, default=all_districts)
sel_months = st.sidebar.multiselect("Month", all_months, default=all_months)
sel_indicators = st.sidebar.multiselect("Indicator", indicators, default=indicators)

# ---- Sidebar: thresholds -------------------------------------------------
st.sidebar.header("Detection thresholds")
trend_thr = st.sidebar.slider("Trend threshold |% change|", 1.0, 100.0, 10.0, 1.0)
method = st.sidebar.radio("Outlier method", ["IQR", "Z-score"], horizontal=True)
if method == "IQR":
    iqr_k = st.sidebar.slider("IQR multiplier (k)", 0.5, 3.0, 1.5, 0.1)
    z_thr = 3.0
else:
    z_thr = st.sidebar.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1)
    iqr_k = 1.5
corr_thr = st.sidebar.slider("Correlation threshold |r|", 0.1, 1.0, 0.70, 0.05)

with st.sidebar.expander("Severity cut-offs"):
    st.caption("Severity = how many multiples of the threshold a finding reaches. "
               "For correlations, how far |r| sits between the threshold and 1.")
    medium_x = st.slider("Medium from (× threshold)", 1.0, 3.0, 1.25, 0.05)
    high_x = max(st.slider("High from (× threshold)", 1.0, 5.0, 1.75, 0.05), medium_x)

limits = {}
with st.sidebar.expander("Absolute limits (optional)"):
    st.caption("Set a min and/or max acceptable value to create threshold_breach insights.")
    for ind in sel_indicators:
        c1, c2 = st.columns(2)
        lo = c1.number_input(f"{ind} min", value=None, placeholder="none", key=f"lo_{ind}")
        hi = c2.number_input(f"{ind} max", value=None, placeholder="none", key=f"hi_{ind}")
        if lo is not None or hi is not None:
            limits[ind] = (lo, hi)

# ---- Apply filters -------------------------------------------------------
filtered = df[df["district"].isin(sel_districts) & df["month"].isin(sel_months)]
if filtered.empty or not sel_indicators:
    st.warning("No data for the current filters. Adjust the sidebar selections.")
    st.stop()

# ---- Run detectors (Parts B, C, D) ---------------------------------------
trend_rows = detect_trends(filtered, sel_indicators, trend_thr, medium_x, high_x)
outlier_rows = detect_outliers(filtered, sel_indicators, method, z_thr, iqr_k, medium_x, high_x)
breach_rows = detect_breaches(filtered, limits, trend_thr, medium_x, high_x)

corr_matrix = filtered[sel_indicators].corr(method="pearson")
corr_rows, corr_stats = [], (len(filtered), filtered["district"].nunique(), filtered["month"].nunique())
if len(sel_indicators) >= 2 and len(filtered) >= 3:
    corr_rows, corr_matrix, corr_stats = detect_correlations(filtered, sel_indicators, corr_thr)

# ---- Part E: assemble insights ------------------------------------------
all_rows = trend_rows + outlier_rows + corr_rows + breach_rows
if all_rows:
    insights = pd.DataFrame(all_rows, columns=[c for c in INSIGHT_COLUMNS if c != "insight_id"])
    insights["_sev"] = insights["severity"].map({s: i for i, s in enumerate(SEVERITY_ORDER)})
    insights["_typ"] = insights["type"].map({t: i for i, t in enumerate(TYPE_ORDER)})
    insights = (insights.sort_values(["_sev", "_typ"], ascending=[False, True], kind="stable")
                .drop(columns=["_sev", "_typ"]).reset_index(drop=True))
    insights.insert(0, "insight_id", [f"INS-{i + 1:04d}" for i in range(len(insights))])
else:
    insights = pd.DataFrame(columns=INSIGHT_COLUMNS)

# ---- Summary metrics -----------------------------------------------------
counts = insights["severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total insights", len(insights))
m2.metric("High", int(counts["High"]))
m3.metric("Medium", int(counts["Medium"]))
m4.metric("Low", int(counts["Low"]))

# ---- Insight table + downloads ------------------------------------------
st.subheader("Generated insights")
if insights.empty:
    st.info("No insights at the current thresholds and filters. Try lowering a threshold.")
else:
    f1, f2 = st.columns(2)
    shown_types = f1.multiselect("Insight types", TYPE_ORDER, default=TYPE_ORDER)
    shown_sev = f2.multiselect("Severity", SEVERITY_ORDER, default=SEVERITY_ORDER)
    view = insights[insights["type"].isin(shown_types) & insights["severity"].isin(shown_sev)]
    st.dataframe(
        view, hide_index=True,
        column_config={"explanation": st.column_config.TextColumn(width="large")},
    )
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("Download insights CSV", insights.to_csv(index=False, na_rep="n/a"),
                       "insights.csv", "text/csv")
    d2.download_button("Download insights JSON", insights.to_json(orient="records", indent=2),
                       "insights.json", "application/json")

# ---- Visualisations (always shown, even with zero insights) --------------
st.subheader("Visualisations")
v1, v2 = st.columns(2)

with v1:
    st.markdown("**Insight severity counts**")
    sev_df = counts.rename_axis("severity").reset_index(name="count")
    fig = px.bar(sev_df, x="severity", y="count", color="severity", text="count",
                 color_discrete_map=SEVERITY_COLORS,
                 category_orders={"severity": SEVERITY_ORDER})
    fig.update_layout(showlegend=False, yaxis_title="Insights", xaxis_title=None)
    st.plotly_chart(fig)

with v2:
    st.markdown("**Correlation heatmap (Pearson)**")
    if len(sel_indicators) >= 2:
        fig = px.imshow(corr_matrix, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                        color_continuous_scale="RdBu_r")
        st.plotly_chart(fig)
        n_rows, n_dist, n_months = corr_stats
        if n_dist < 10:
            st.warning(
                f"Limitation: computed on {n_rows} district-month rows ({n_dist} districts × "
                f"{n_months} months). With this little data, correlations are fragile and can be "
                f"driven by a single point; treat them as indicative only."
            )
        else:
            st.caption(f"Computed on {n_rows} district-month rows ({n_dist} districts × {n_months} months).")
        st.download_button("Download correlation matrix CSV", corr_matrix.to_csv(),
                           "correlation_matrix.csv", "text/csv")
    else:
        st.info("Select at least two indicators to see correlations.")

st.markdown("**Per-district trend**")
line_ind = st.selectbox("Indicator", sel_indicators)
line_data = filtered.pivot_table(index="month", columns="district", values=line_ind, aggfunc="mean")
fig = px.line(line_data, markers=True, labels={"value": line_ind, "month": "Month", "district": "District"})
st.plotly_chart(fig)