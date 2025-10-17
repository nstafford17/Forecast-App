# app.py — Colby Forecast (Quarters or Months) with Product Filters
# -------------------------------------------------------------
# Fresh scaffold with:
#  - CSV upload
#  - Column selectors (date/value)
#  - Cascading Product Filters (Butts/Epic/T-10X) that operate on Part Number
#  - Customer / Part multiselect filters
#  - Date range filter
#  - Colby method on **quarters or months** (selectable)
#  - Math/debug tables
#  - Plot + forecast table
#
# How to run:
#   1) pip install streamlit pandas numpy plotly
#   2) streamlit run app.py
# -------------------------------------------------------------

from __future__ import annotations
import io
import re
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Forecast → Colby", layout="wide")


col1, col2 = st.columns([1, 4])

with col1:
    st.image("winthrop_inverted.png", width=600)  # replace with your logo file

with col2:
    st.markdown(
        "<h1 style='color:black; font-family:sans-serif; margin-bottom:0;'>"
        "Forecasting Dashboard"
        "</h1>",
        unsafe_allow_html=True
    )

st.write("Upload your shipping CSV and explore the forecast.")

# -------------------------- Utilities --------------------------

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df

# ----------------- Product Filters (UI + logic) ----------------
# Exclusions always win (exactly like your other app)
EXCLUDE_PRIORITY = ["AC ", "CART ", " SET", "SFC ", "SA ", "SP ", "SC "]

def render_product_filters() -> Dict[str, Any]:
    """
    Renders the cascading product filters UI and returns a 'selections' dict.
    Applies to the Part Number column only.
    """
    with st.expander("Product Filters (Part Number patterns)", expanded=True):
        st.markdown("**Product Type**")
        sel_butts = st.checkbox("Butts", value=False, key="pt_butts")
        sel_guides = st.checkbox("Guides", value=False, key="pt_guides")
        sel_tops = st.checkbox("Tops", value=False, key="pt_tops")

        selections: Dict[str, Any] = {
            "Product Type": [],
            "Butt Type": None,
            "Ferrule Size": None,
            "Colour": None,
            "Handle Length": None,
            "Gimbal Size": None,
        }

        butt_type = None
        show_next = False
        ferrule_opts = []
        next_title = ""
        next_opts = []
        next_key: Optional[str] = None

        if sel_butts:
            selections["Product Type"].append("Butts")

            butt_type = st.radio("Butt Type", ["Terminator", "Epic", "T-10X"], key="butt_type")
            selections["Butt Type"] = butt_type

            # Branching
            if butt_type == "Terminator":
                ferrule_opts = ["#2", "#4"]
                show_next = True
                next_title = "Handle Length"
                next_opts = ["Short", "Long"]
                next_key = "Handle Length"
                selections["Gimbal Size"] = None
            elif butt_type == "Epic":
                ferrule_opts = ["#1", "#2"]
                show_next = True
                next_title = "Gimbal Size"
                next_opts = ["Short", "Long"]
                next_key = "Gimbal Size"
                selections["Handle Length"] = None
            else:  # T-10X
                ferrule_opts = ["#4", "#6"]
                show_next = False
                selections["Handle Length"] = None
                selections["Gimbal Size"] = None

            ferrule = st.radio("Ferrule Size", ferrule_opts, key="ferrule_size")
            selections["Ferrule Size"] = ferrule

            colour = st.radio("Colour", ["Black", "Silver", "Blue", "Custom"], key="butt_colour")
            selections["Colour"] = colour

            if show_next and next_key:
                next_sel = st.radio(next_title, next_opts, key="butt_next")
                selections[next_key] = next_sel

        if sel_guides:
            selections["Product Type"].append("Guides")
            st.caption("Subfilters for Guides can be added later.")

        if sel_tops:
            selections["Product Type"].append("Tops")
            st.caption("Subfilters for Tops can be added later.")

    return selections

def filter_by_partnumber_patterns(df: pd.DataFrame, selections: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply includes/excludes to df using ONLY the Part Number column, based on
    sidebar selections. Exclusions always win.

    Rules:
      - First, drop rows whose Part Number contains any EXCLUDE_PRIORITY token
      - If "Butts" isn't selected, we return after exclusions
      - If Butts is selected, include rows that match the butt type + ferrule rules
      - Epic #1 is treated as “no explicit number present” → exclude rows that mention " #2 "
    """
    if "Part Number" not in df.columns:
        return df  # silently no-op if the column doesn't exist

    d = df.copy()
    pn = d["Part Number"].astype(str).str.lower()

    # Exclusions
    exclude_mask = pd.Series(False, index=d.index)
    for token in EXCLUDE_PRIORITY:
        t = token.lower()
        exclude_mask |= pn.str.contains(re.escape(t), na=False)
    d = d[~exclude_mask]

    if "Butts" not in selections.get("Product Type", []):
        return d

    pn = d["Part Number"].astype(str).str.lower()  # refresh after exclusion

    butt_type = (selections.get("Butt Type") or "").lower()
    ferrule   = (selections.get("Ferrule Size") or "").lower()

    # Base token by butt type
    if butt_type == "terminator":
        base_token = "trmtr "
    elif butt_type == "epic":
        base_token = "epic "
    elif butt_type == "t-10x":
        base_token = "t-10x "
    else:
        return d

    base_mask = pn.str.contains(re.escape(base_token), na=False)

    # Ferrule refinement
    if butt_type == "terminator":
        if ferrule == "#2":
            ferr_mask = pn.str.contains(re.escape(" #2 "), na=False)
            include_mask = base_mask & ferr_mask
        elif ferrule == "#4":
            ferr_mask = pn.str.contains(re.escape(" #4 "), na=False)
            include_mask = base_mask & ferr_mask
        else:
            include_mask = base_mask

    elif butt_type == "t-10x":
        if ferrule == "#4":
            ferr_mask = pn.str.contains(re.escape(" #4 "), na=False)
            include_mask = base_mask & ferr_mask
        elif ferrule == "#6":
            ferr_mask = pn.str.contains(re.escape(" #6 "), na=False)
            include_mask = base_mask & ferr_mask
        else:
            include_mask = base_mask

    else:  # Epic
        if ferrule == "#2":
            ferr_mask = pn.str.contains(re.escape(" #2 "), na=False)
            include_mask = base_mask & ferr_mask
        else:
            # Epic #1 = no explicit number present → exclude rows with " #2 "
            not_2_mask = ~pn.str.contains(re.escape(" #2 "), na=False)
            include_mask = base_mask & not_2_mask

    return d[include_mask]

# --------------------- Colby (generic implementation) ---------------------

def colby_forecast_from_periods(pseries: pd.Series, freq: str = 'Q', horizon: int = 4, return_debug: bool = False):
    """
    Colby method — per-bucket growth (quarters or months):
      • Work on totals at the requested frequency (freq='Q' or 'M').
      • Exclude the period containing the latest date; call the cutoff C = latest - 1 period.
      • Define buckets:
          - If quarterly → buckets = {1,2,3,4} with attribute .quarter
          - If monthly   → buckets = {1..12} with attribute .month
      • For each bucket b:
          - L_b = value from the most recent year ≤ C that has bucket b (prefer the year of C if available).
          - E_b = value from the earliest available year that has bucket b.
          - factor_b = 1 + (L_b - E_b) / E_b  (if E_b ≤ 0 or missing → factor_b = 1).
          - Base_b = L_b.
      • Forecast the next `horizon` periods after C. For each future period p (with bucket b),
        output V_p = Base_b * factor_b.

    Returns a Timestamp-indexed Series, or `(series, debug)` if `return_debug=True`.
    """
    if pseries.empty:
        return pd.Series(dtype=float)

    if not isinstance(pseries.index, pd.PeriodIndex):
        raise ValueError("colby_forecast_from_periods expects a PeriodIndex series.")

    freq = freq.upper()
    if freq not in {"Q", "M"}:
        raise ValueError("freq must be 'Q' or 'M'")

    if pseries.index.freqstr[0] != freq:
        raise ValueError("Series frequency and 'freq' argument disagree.")

    # Setup bucket function and size
    if freq == 'Q':
        bucket_size = 4
        get_bucket = lambda per: per.quarter
        bucket_name = "Quarter"
    else:
        bucket_size = 12
        get_bucket = lambda per: per.month
        bucket_name = "Month"

    latest_period = pseries.index.max()
    cutoff = latest_period - 1
    if cutoff not in pseries.index:
        cutoff = latest_period

    # Earliest per bucket
    per_bucket_earliest = {}
    for b in range(1, bucket_size + 1):
        candidates = [p for p in pseries.index if get_bucket(p) == b]
        candidates.sort()
        per_bucket_earliest[b] = candidates[0] if candidates else None

    # Latest ≤ cutoff per bucket (prefer same year as cutoff)
    per_bucket_latest = {}
    for b in range(1, bucket_size + 1):
        if freq == 'Q':
            same_year = pd.Period(year=cutoff.year, quarter=b, freq='Q')
        else:
            same_year = pd.Period(year=cutoff.year, month=b, freq='M')
        if same_year in pseries.index and same_year <= cutoff:
            per_bucket_latest[b] = same_year
        else:
            cands = [p for p in pseries.index if get_bucket(p) == b and p <= cutoff]
            per_bucket_latest[b] = max(cands) if cands else None

    # Compute Base and factor by bucket
    base = {}
    factor = {}
    for b in range(1, bucket_size + 1):
        Lp = per_bucket_latest[b]
        Ep = per_bucket_earliest[b]
        L = float(pseries.get(Lp, np.nan)) if Lp is not None else np.nan
        E = float(pseries.get(Ep, np.nan)) if Ep is not None else np.nan
        base[b] = 0.0 if np.isnan(L) else L
        if np.isnan(E) or E == 0.0:
            factor[b] = 1.0
        else:
            # annualize by years between Ep and Lp
            year_diff = abs(Lp.year - Ep.year) if (Lp is not None and Ep is not None) else 1
            year_diff = max(1, year_diff)
            rate = (base[b] - E) / E
            factor[b] = 1.0 + rate / year_diff

    # Build future index
    fut_periods = pd.period_range(start=cutoff + 1, periods=horizon, freq=freq)
    fut_index = fut_periods.to_timestamp(how='end')

    # Values + per-step rows
    values = []
    rows = []
    for p in fut_periods:
        b = get_bucket(p)
        val = base[b] * factor[b]
        values.append(val)
        rows.append({
            "Future Period": p.to_timestamp(how='end'),
            bucket_name: b,
            f"Base L_{bucket_name[0].lower()}": base[b],
            f"E_{bucket_name[0].lower()}": (float(pseries.get(per_bucket_earliest[b], np.nan))
                                if per_bucket_earliest[b] is not None else np.nan),
            "factor": factor[b],
            "Value": val,
        })

    series_out = pd.Series(values, index=fut_index, name="Colby Forecast")

    if not return_debug:
        return series_out

    # Per-bucket debug table
    perb_rows = []
    for b in range(1, bucket_size + 1):
        Ep = per_bucket_earliest[b]
        Lp = per_bucket_latest[b]
        perb_rows.append({
            bucket_name: b,
            "Earliest Period": Ep.to_timestamp(how='end') if Ep is not None else pd.NaT,
            f"E_{bucket_name[0].lower()}": (float(pseries.get(Ep, np.nan)) if Ep is not None else np.nan),
            "Latest≤Cutoff": Lp.to_timestamp(how='end') if Lp is not None else pd.NaT,
            f"L_{bucket_name[0].lower()}": (float(pseries.get(Lp, np.nan)) if Lp is not None else np.nan),
            "factor": factor[b],
        })

    debug = {
        "freq": freq,
        "cutoff_period": cutoff,
        "cutoff_end": cutoff.to_timestamp(how='end') if isinstance(cutoff, pd.Period) else pd.NaT,
        "per_bucket_table": pd.DataFrame(perb_rows),
        "future_table": pd.DataFrame(rows),
        "notes": "factor = 1 + (L - E) / E; if E ≤ 0 or missing → factor = 1; Base = latest≤cutoff for that bucket",
    }
    return series_out, debug

# --------------------------- UI -------------------------------

st.title("Forecast")
st.caption("Single-method app with cascading Product Filters.")
f = st.file_uploader("Upload CSV", type=["csv"])
with st.sidebar:
    st.header("1) Data")
    st.markdown(
        """
        *Expected columns (flexible):* `Customer`, `Part Number`, `Qty Due`, `Extended Amount`, `Due Date`.
        You can choose actual date/value columns below.
        """
    )

main = st.container()

if f is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# Load
try:
    df = load_csv(f)
except Exception as e:
    st.error(f"Couldn't read that CSV: {e}")
    st.stop()

if df.empty:
    st.warning("Your file seems empty.")
    st.stop()

# Column selectors
with st.sidebar:
    st.header("2) Columns")
    preferred_date = "Due Date"
    preferred_value = "Qty Due"
    preferred_customer = "Customer"
    preferred_part = "Part Number"

    # Date column options — only show columns with "date" in the name (case-insensitive)
    date_cols = [c for c in df.columns if "date" in c.lower()]

    # If nothing matches, fall back to all columns just in case
    if not date_cols:
        date_cols = list(df.columns)

    date_default = date_cols.index(preferred_date) if preferred_date in date_cols else 0
    date_col = st.selectbox("Date column", options=date_cols, index=date_default)


    # Value column options (numeric or convertible)
    num_cols = []
    for c in df.columns:
        s = df[c]
        if _is_numeric(s):
            num_cols.append(c)
        else:
            # Try to coerce text columns like "$1,234.56" to numeric
            s_coerced = pd.to_numeric(
                s.astype(str).str.replace(r"[^\d.\-]", "", regex=True), 
                errors="coerce"
            )
            if s_coerced.notna().sum() > 0:  # some numeric values found
                num_cols.append(c)

    if not num_cols:
        st.error("No numeric or convertible columns detected. Please include at least one numeric field (e.g., Qty Due or Extended Amount).")
        st.stop()

    # Prefer "Qty Due" or "Extended Amount"
    preferred_candidates = ["Qty Due", "Extended Amount"]
    preferred_value = next((c for c in preferred_candidates if c in num_cols), num_cols[0])

    value_default = num_cols.index(preferred_value)
    value_col = st.selectbox("Value column", options=num_cols, index=value_default)


# --------- Product Filters (operate strictly on Part Number) ----------
with st.sidebar:
    st.header("3) Product Filters")
    selections = render_product_filters()
    if "Part Number" not in df.columns:
        st.caption("ℹ️ These filters require a **Part Number** column; skipping if unavailable.")

# Apply pattern-based Product Filters first (if Part Number exists)
df_pf = filter_by_partnumber_patterns(df, selections)

# Optional category filters (user-chosen columns)
with st.sidebar:
    st.header("4) Attribute Filters")
    cust_col = st.selectbox("Customer column (optional)", ["(none)"] + df.columns.tolist(),
                            index=(df.columns.tolist().index(preferred_customer) + 1
                                   if preferred_customer in df.columns else 0))
    part_col = st.selectbox("Part column (optional)", ["(none)"] + df.columns.tolist(),
                            index=(df.columns.tolist().index(preferred_part) + 1
                                   if preferred_part in df.columns else 0))

# Parse dates safely
try:
    df_pf[date_col] = pd.to_datetime(df_pf[date_col], errors="coerce")
except Exception:
    pass

# Build working frame with chosen columns
keep_cols = [date_col, value_col]
if cust_col != "(none)":
    keep_cols.append(cust_col)
if part_col != "(none)":
    keep_cols.append(part_col)

work = df_pf[keep_cols].copy()
work = work.dropna(subset=[date_col, value_col])

if work.empty:
    st.error("No usable rows after parsing dates/values.")
    st.stop()

# User-facing attribute filters
with st.sidebar:
    if cust_col != "(none)":
        customers = sorted(work[cust_col].dropna().unique().tolist())
        sel_customers = st.multiselect("Customers", customers)
        if sel_customers:
            work = work[work[cust_col].isin(sel_customers)]
    if part_col != "(none)":
        parts = sorted(work[part_col].dropna().unique().tolist())
        sel_parts = st.multiselect("Parts", parts)
        if sel_parts:
            work = work[work[part_col].isin(sel_parts)]

# Date range (typed inputs instead of slider)
min_d, max_d = work[date_col].min(), work[date_col].max()
with st.sidebar:
    st.header("5) Date Range")
    if pd.isna(min_d) or pd.isna(max_d):
        st.warning("Could not infer date range; using all rows.")
        d_start, d_end = None, None
    else:
        c1, c2 = st.columns(2)
        with c1:
            start_input = st.date_input(
                "Start",
                value=min_d.date(),
                min_value=min_d.date(),
                max_value=max_d.date(),
                key="typed_start",
            )
        with c2:
            end_input = st.date_input(
                "End",
                value=max_d.date(),
                min_value=min_d.date(),
                max_value=max_d.date(),
                key="typed_end",
            )

        # Convert to pandas Timestamps
        d_start = pd.to_datetime(start_input)
        # include the full end day
        d_end = pd.to_datetime(end_input) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        if d_start > d_end:
            st.error("Start date must be on or before end date.")
            st.stop()

# Apply the typed date range
if d_start is not None and d_end is not None:
    work = work[(work[date_col] >= d_start) & (work[date_col] <= d_end)]


# Frequency & horizon (Colby now supports months or quarters)
with st.sidebar:
    st.header("6) Forecast Settings")
    period_choice = st.selectbox("Periodicity", ["Quarterly", "Monthly"], index=0)
    if period_choice == "Quarterly":
        freq = 'Q'
        horizon = st.number_input("Forecast horizon (quarters)", min_value=1, max_value=12, value=4, step=1)
        st.caption("This method operates on **quarterly** sums.")
    else:
        freq = 'M'
        horizon = st.number_input("Forecast horizon (months)", min_value=1, max_value=36, value=12, step=1)
        st.caption("This method operates on **monthly** sums.")

# ----------------- Build period series (sum) -----------------

work = work.sort_values(date_col)
resample_rule = 'Q' if freq == 'Q' else 'M'
series = (
    work.set_index(date_col)[value_col]
    .resample(resample_rule)
    .sum()
    .asfreq(resample_rule)
    .fillna(0.0)
)

# History for plotting
plot_series = series.copy()
if isinstance(plot_series.index, pd.PeriodIndex):
    plot_series.index = plot_series.index.to_timestamp(how='end')

pseries = series.copy()
pseries.index = pseries.index.to_period(resample_rule)

# ------------------------- Forecast ----------------------------

with st.container():
    st.success(f"Colby Method is active using {('quarters' if freq=='Q' else 'months')} as periods.")

result = colby_forecast_from_periods(pseries, freq=freq, horizon=int(horizon), return_debug=True)
if isinstance(result, tuple):
    forecast, debug = result
else:
    forecast, debug = result, None

# ---------------------- Math / Steps ---------------------------

with st.expander("Show math / steps"):
    st.markdown(
        r"""
        **Colby per-period math**  
        For each period bucket *(quarter=1..4 or month=1..12)*:  
        $\mathrm{factor} = 1 + \frac{L - E}{E}$  
        (if $E \le 0$ or missing → factor = 1)  
        Future periods after cutoff use $V = L \times \mathrm{factor}$.
        """
    )
    if 'debug' in locals() and debug is not None:
        st.write("**Cutoff (last usable period end):**", debug["cutoff_end"])
        st.markdown("**Per-bucket summary (earliest vs latest ≤ cutoff):**")
        st.dataframe(debug["per_bucket_table"], use_container_width=True)
        st.markdown("**Future periods — step-by-step values:**")
        st.dataframe(debug["future_table"], use_container_width=True)
    else:
        st.info("Debug info unavailable. (Run with return_debug=True.)")

# -------------------------- Plot -------------------------------

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=series.index, y=series.values, mode="lines+markers", name="History")
)
if not forecast.empty:
    fig.add_trace(
        go.Scatter(x=forecast.index, y=forecast.values, mode="lines+markers", name="Sales Forecast")
    )
fig.update_layout(title="History + Forecast", height=480, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

# ------------------------ Forecast table -----------------------

if not forecast.empty:
    st.subheader("Forecast table")

    # Make friendly period labels from the forecast index.
    # We subtract 1 day so that indexes like 2025-09-01 map to "August 2025".
    idx = forecast.index

    if freq == 'M':
        labels = (idx - pd.Timedelta(days=1)).to_period('M').strftime('%B %Y')
    else:  # 'Q'
        q_periods = (idx - pd.Timedelta(days=1)).to_period('Q')
        labels = [f"Q{p.quarter} {p.year % 100:02d}" for p in q_periods]

    out = pd.DataFrame({
        "Period": labels,
        "Forecast": forecast.values
    })

    st.dataframe(out, use_container_width=True)
else:
    st.info("No forecast produced (need at least some history; if earliest same-bucket is zero or missing, forecast equals last usable period for that bucket).")

# -------------------- Quick summary counters -------------------

with st.expander("Data summary after filters"):
    st.write(f"**Rows in file:** {len(df):,}")
    st.write(f"**Rows after Product Filters:** {len(df_pf):,}")
    if "Part Number" in df.columns:
        st.write(f"**Unique Part Numbers after Product Filters:** {df_pf['Part Number'].nunique():,}")
