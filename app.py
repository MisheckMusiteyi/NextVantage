# -*- coding: utf-8 -*-
# =============================================================================
#  AFRICAN ACTUARIAL CONSULTANTS — COMPREHENSIVE ACTUARIAL TOOLKIT
#  Main App with Multi-Page Navigation
#  Run: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import date, datetime
import re
from scipy import interpolate
from scipy import stats as scipy_stats

# =============================================================================
#  ROBUST PATH & IMPORT SYSTEM (with diagnostic error capture)
# =============================================================================
import sys
import os
import importlib.util
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

_IMPORT_ERRORS = {}

def import_file_glob(relative_pattern):
    search_pattern = os.path.join(BASE_DIR, relative_pattern.replace('/', os.sep))
    matched_files = glob.glob(search_pattern)
    if not matched_files:
        search_pattern = os.path.join(BASE_DIR, relative_pattern.replace('/', os.sep).replace(' ', '?'))
        matched_files = glob.glob(search_pattern)
    if not matched_files:
        _IMPORT_ERRORS[relative_pattern] = "File not found"
        return None
    abs_path = matched_files[0]
    module_name = os.path.splitext(os.path.basename(abs_path))[0]
    try:
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if spec is None:
            _IMPORT_ERRORS[relative_pattern] = "spec_from_file_location returned None"
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _IMPORT_ERRORS.pop(relative_pattern, None)
        return module
    except Exception as e:
        import traceback
        _IMPORT_ERRORS[relative_pattern] = traceback.format_exc()
        return None

# --- Load all engine modules ---
upr_engine = import_file_glob("LRC_Calculators/upr_engine.py")
loss_comp_engine = import_file_glob("LRC_Calculators/loss_component_engine.py")
ocr_engine = import_file_glob("LIC_Calculators/FCF_Calculators/OCR_Calculators/ocr_engine.py")
ibnr_pct = import_file_glob("LIC_Calculators/FCF_Calculators/IBNR_Calculators/percentage_ibnr.py")
ibnr_bcl = import_file_glob("LIC_Calculators/FCF_Calculators/IBNR_Calculators/bcl_ibnr.py")
ibnr_cc = import_file_glob("LIC_Calculators/FCF_Calculators/IBNR_Calculators/cape_cod_ibnr.py")
ibnr_bf = import_file_glob("LIC_Calculators/FCF_Calculators/IBNR_Calculators/bf_ibnr.py")
ulae_engine = import_file_glob("LIC_Calculators/FCF_Calculators/ULAE_Calculators/ulae_engine.py")
npr_engine = import_file_glob("LIC_Calculators/FCF_Calculators/NPR_Calculators/npr_engine.py")
mack_engine = import_file_glob("LIC_Calculators/RA_Calculators/mack_ra.py")
bootstrap_engine = import_file_glob("LIC_Calculators/RA_Calculators/bootstrap_ra.py")
engine_utils = import_file_glob("utils/actuarial_engine_utils.py")

# --- Load Full Valuation engine ---
full_engine = None
for pattern in ["Full_Valuation/full_lrc_ifrs17.py", "Full_Valuation/full_LRC_IFRS17.py", "Full_Valuation/*.py"]:
    full_engine = import_file_glob(pattern)
    if full_engine is not None:
        break
if full_engine is None:
    fv_path = os.path.join(BASE_DIR, "Full_Valuation", "full_lrc_ifrs17.py")
    if os.path.exists(fv_path):
        try:
            spec = importlib.util.spec_from_file_location("full_lrc_ifrs17", fv_path)
            if spec:
                full_engine = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(full_engine)
        except Exception:
            pass
if full_engine is not None:
    keys_to_pop = [k for k in _IMPORT_ERRORS if "Full_Valuation" in k]
    for k in keys_to_pop:
        _IMPORT_ERRORS.pop(k, None)

# =============================================================================
#  MODULE STATUS CHECK
# =============================================================================
module_status = {
    "UPR Engine": upr_engine, "Loss Component Engine": loss_comp_engine, "OCR Engine": ocr_engine,
    "Percentage IBNR": ibnr_pct, "BCL IBNR": ibnr_bcl, "Cape Cod IBNR": ibnr_cc,
    "BF IBNR": ibnr_bf, "ULAE Engine": ulae_engine, "NPR Engine": npr_engine,
    "Mack RA": mack_engine, "Bootstrap RA": bootstrap_engine,
    "Engine Utils": engine_utils, "Full Valuation": full_engine,
}
critical_modules = ["UPR Engine", "OCR Engine", "Engine Utils"]
missing_critical = [name for name in critical_modules if module_status[name] is None]
if missing_critical:
    st.error("Critical Error: Essential modules could not be loaded.")
    for mod in missing_critical:
        st.error(f"  - {mod}")
    st.stop()

missing_optional = [name for name, mod in module_status.items() if mod is None and name not in critical_modules]
if missing_optional or _IMPORT_ERRORS:
    with st.sidebar.expander("Module Status", expanded=False):
        if missing_optional:
            st.warning("Some optional modules could not be loaded:")
            for mod in missing_optional:
                st.write(f"  - {mod}")
        if _IMPORT_ERRORS:
            st.error("Import errors detected:")
            for pattern, err in _IMPORT_ERRORS.items():
                with st.expander(f"Error: {pattern}"):
                    st.code(err)

# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================
def _date_filter(df, col, from_date, to_date):
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors='coerce')
    fd = pd.Timestamp(from_date); td = pd.Timestamp(to_date)
    return df[(df[col] >= fd) & (df[col] <= td)]

def periods_per_year(grain):
    return {"Y": 1, "Q": 4, "M": 12}[grain]

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', '', name).strip() or "Client"

def show_error(e):
    """Display a clear, human-readable error."""
    msg = str(e)
    col_match = re.search(r"column ([^\s]+(?: [A-Za-z]+)?) with type", msg)
    col_name = col_match.group(1) if col_match else None
    col_text = f" in the **'{col_name}'** column" if col_name else ""

    bad_value_match = re.search(r"[Cc]ould not convert '([^']+)' with type str", msg)
    also_generic_float_match = re.search(r"could not convert string to float: '([^']+)'", msg)

    if bad_value_match or also_generic_float_match:
        bad_value = bad_value_match.group(1) if bad_value_match else also_generic_float_match.group(1)
        st.error(
            f"**Non-numeric value found{col_text}: '{bad_value}'**\n\n"
            f"A column expected to contain only numbers has at least one value that isn't a valid number. "
            f"This commonly happens because of a stray text entry, formatting issue, or currency symbol.\n\n"
            f"Please fix the offending value(s) and re-upload."
        )
        return

    type_conversion_match = re.search(r"cannot be converted to (int|float|double)", msg)
    if type_conversion_match:
        looks_like_date_col = bool(col_name) and re.search(r"date", col_name, re.IGNORECASE)
        if looks_like_date_col:
            st.error(f"**Invalid date value found{col_text}**\n\nPlease fix or remove the offending value(s) and re-upload.")
        else:
            st.error(f"**Invalid value found{col_text}**\n\nA column expected to contain only numbers or dates has at least one value that couldn't be interpreted.")
        return

    st.error(f"Error: {msg}")

def _check_duplicate_columns(df, filename=None):
    cols = pd.Series(df.columns.astype(str))
    dupes = cols[cols.duplicated()].unique().tolist()
    if dupes:
        where = f" in '{filename}'" if filename else ""
        dupe_list = ", ".join(f"'{d}'" for d in dupes)
        raise ValueError(
            f"Duplicate column name(s) found{where}: {dupe_list}. "
            f"Please rename or remove the repeated column(s) and re-upload."
        )

def map_columns(df, required_fields, prefix):
    cols = df.columns.tolist()
    mapping = {}
    st.markdown(f"**Map columns:**")
    for field in required_fields:
        default_val = field if field in cols else (cols[0] if cols else "")
        default_idx = cols.index(default_val) if default_val in cols else 0
        mapping[field] = st.selectbox(f"{field}", cols, index=default_idx, key=f"{prefix}_{field}")
    return mapping

def _optimize_dtypes(df, category_threshold=0.5, category_max_unique=10000):
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_float_dtype(dtype): df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_integer_dtype(dtype): df[col] = pd.to_numeric(df[col], downcast='integer')
        elif dtype == object:
            n_total = len(df[col])
            if n_total > 0:
                n_unique = df[col].nunique(dropna=True)
                if (n_unique / n_total) < category_threshold and n_unique < category_max_unique:
                    df[col] = df[col].astype('category')
    return df

@st.cache_data(show_spinner=False)
def read_uploaded_file(uploaded_file, optimize=True):
    name = uploaded_file.name
    df = pd.read_csv(uploaded_file) if name.endswith('.csv') else pd.read_excel(uploaded_file)
    unnamed = [c for c in df.columns if str(c).startswith('Unnamed:')]
    if unnamed: df = df.drop(columns=unnamed)
    df.columns = df.columns.astype(str).str.strip()
    _check_duplicate_columns(df, filename=name)
    if optimize: df = _optimize_dtypes(df)
    return df

def build_download_payload(sheets: dict, row_threshold: int = 100_000):
    total_rows = sum(len(df) for df in sheets.values())
    if total_rows <= row_threshold:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
        output.seek(0)
        return output, "xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        import zipfile
        output = BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
            for sheet_name, df in sheets.items():
                zf.writestr(f"{sheet_name}.csv", df.to_csv(index=False))
        output.seek(0)
        return output, "zip", "application/zip"

def load_inflation_data_ui(grain_code, ppy, page_key):
    st.markdown("**Inflation Adjustment**")
    inf_file = st.file_uploader("Upload Inflation Curve (CSV/Excel: Period, Rate %)", type=["csv","xlsx","xls"], key=f"inf_{page_key}")
    cum_inflation = None; per_period_rates = None
    if inf_file:
        try:
            inf_df = read_uploaded_file(inf_file)
            inf_df.columns = inf_df.columns.astype(str).str.strip()
            c1, c2 = st.columns(2)
            with c1: p_col = st.selectbox("Period Column", inf_df.columns, key=f"inf_p_{page_key}")
            with c2: r_col = st.selectbox("Rate Column (%)", inf_df.columns, key=f"inf_r_{page_key}")
            inf_df = inf_df[[p_col, r_col]].dropna()
            inf_df[r_col] = pd.to_numeric(inf_df[r_col], errors='coerce') / 100.0
            rates_inf = inf_df[r_col].values
            ratio = ppy / periods_per_year(grain_code)
            x_inf = np.arange(len(rates_inf)) * ratio
            x_tgt = np.arange(int(x_inf[-1]) + 1) if len(x_inf) > 0 else np.array([0])
            if len(x_inf) >= 4: f_interp = interpolate.CubicSpline(x_inf, rates_inf, extrapolate=True)
            else: f_interp = interpolate.interp1d(x_inf, rates_inf, kind='linear', fill_value='extrapolate')
            annual_rates_tgt = np.clip(f_interp(x_tgt), -0.5, 2.0)
            per_period_rates = (1 + annual_rates_tgt) ** (1 / ppy) - 1
            cum_inflation = np.cumprod(1 + per_period_rates)
            st.success("Inflation curve loaded and interpolated.")
        except Exception as e: show_error(e)
    return cum_inflation, per_period_rates

def _apply_flexible_date_filter(df, date_col, filter_type, date1=None, date2=None):
    if not date_col or filter_type == "No Filter":
        return pd.Series(True, index=df.index)
    s = pd.to_datetime(df[date_col], errors='coerce')
    end_of_day = pd.Timedelta(hours=23, minutes=59, seconds=59)
    if filter_type == "On":
        d1 = pd.Timestamp(date1)
        return (s >= d1) & (s <= d1 + end_of_day)
    if filter_type == "On or Before":
        return s <= pd.Timestamp(date1) + end_of_day
    if filter_type == "On or After":
        return s >= pd.Timestamp(date1)
    if filter_type == "Before":
        return s < pd.Timestamp(date1)
    if filter_type == "After":
        return s > pd.Timestamp(date1) + end_of_day
    if filter_type == "Between":
        return (s >= pd.Timestamp(date1)) & (s <= pd.Timestamp(date2) + end_of_day)
    return pd.Series(True, index=df.index)

def _render_single_date_filter_ui(cols, label, key_prefix):
    st.markdown(f"**{label}**")
    c1, c2 = st.columns(2)
    with c1:
        date_col = st.selectbox(f"{label} Column", cols, key=f"{key_prefix}_col")
    with c2:
        filter_type = st.selectbox(
            "Filter Type",
            ["No Filter", "On", "On or Before", "On or After", "Before", "After", "Between"],
            key=f"{key_prefix}_type"
        )
    date1 = date2 = None
    if filter_type == "Between":
        cc1, cc2 = st.columns(2)
        with cc1: date1 = st.date_input(f"{label} From", date(2020, 1, 1), key=f"{key_prefix}_d1")
        with cc2: date2 = st.date_input(f"{label} To", date(2025, 12, 31), key=f"{key_prefix}_d2")
    elif filter_type != "No Filter":
        date1 = st.date_input(f"{label} Date", date(2025, 1, 1), key=f"{key_prefix}_d1")
    return date_col, filter_type, date1, date2

def load_discounting_data_ui(grain_code, ppy, page_key):
    st.markdown("**Discounting**")
    disc_method = st.radio("Discounting Method", ["None", "Single Flat Rate", "Yield Curve"], key=f"disc_m_{page_key}", horizontal=True)
    spot_rates = None; flat_rate = None
    if disc_method == "Yield Curve":
        yc_file = st.file_uploader("Upload Yield Curve (CSV/Excel: Duration_Years, Spot_Rate %)", type=["csv","xlsx","xls"], key=f"yc_{page_key}")
        if yc_file:
            try:
                yc_df = read_uploaded_file(yc_file); yc_df.columns = yc_df.columns.astype(str).str.strip()
                c1, c2 = st.columns(2)
                with c1: m_col = st.selectbox("Duration Column", yc_df.columns, key=f"yc_m_{page_key}")
                with c2: r_col = st.selectbox("Spot Rate Column (%)", yc_df.columns, key=f"yc_r_{page_key}")
                yc_df = yc_df[[m_col, r_col]].dropna()
                yc_df[m_col] = pd.to_numeric(yc_df[m_col], errors='coerce')
                yc_df[r_col] = pd.to_numeric(yc_df[r_col], errors='coerce') / 100.0
                maturities = yc_df[m_col].values; rates = yc_df[r_col].values
                if len(maturities) >= 4: f_interp = interpolate.CubicSpline(maturities, rates, extrapolate=True)
                else: f_interp = interpolate.interp1d(maturities, rates, kind='linear', fill_value='extrapolate')
                period_maturities = np.arange(1, 61) / ppy
                spot_rates = np.clip(f_interp(period_maturities), 0, 1.0)
                st.success("Yield curve loaded and interpolated.")
            except Exception as e: show_error(e)
    elif disc_method == "Single Flat Rate":
        flat_rate = st.number_input("Annual Discount Rate (%)", 0.0, 50.0, 5.0, 0.5, key=f"flat_{page_key}") / 100.0
    return spot_rates, flat_rate


# =============================================================================
#  STREAMLIT CONFIGURATION
# =============================================================================
st.set_page_config(page_title="African Actuarial Consultants Actuarial Toolkit", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    *, *::before, *::after { box-sizing: border-box !important; }
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    [data-testid="stIconMaterial"], span[class*="material-symbols"], span[class*="material-icons"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    }
    .stApp { background-color: #FFFFFF; color: #000000; font-family: 'Calisto MT', 'Georgia', serif; font-size: 11pt; line-height: 1.5; }
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        font-family: 'Calisto MT', 'Georgia', serif !important;
        line-height: 1.45; overflow-wrap: break-word; word-break: break-word;
    }
    .hero { background: linear-gradient(135deg, #000000 0%, #1a1a2e 100%); color: #FFFFFF; padding: 2.5rem 2rem; text-align: center; border-bottom: 3px solid #4A90D9; margin-bottom: 2rem; }
    .hero h1 { color: #4A90D9; font-size: 2.5rem; margin: 0 0 0.5rem 0; line-height: 1.2; }
    .hero p { font-size: 1.1rem; max-width: 800px; margin: 0 auto; line-height: 1.5; }
    div[data-testid="stHorizontalBlock"] { align-items: stretch !important; }
    div[data-testid="column"] { display: flex !important; flex-direction: column !important; }
    div[data-testid="column"] > div[data-testid="stVerticalBlock"] { display: flex; flex-direction: column; flex: 1 1 auto; height: 100%; }
    .card {
        background-color: #F9F9F9; border: 2px solid #4A90D9; border-radius: 10px; padding: 1.25rem 1.25rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem; text-align: center;
        min-height: 150px; flex: 1 1 auto; width: 100%;
        display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 0.4rem; overflow: hidden;
    }
    .card h3 { color: #4A90D9; margin: 0; font-size: 1.1rem; line-height: 1.3; width: 100%; }
    .card p { font-size: 0.9rem; color: #555; margin: 0; line-height: 1.4; width: 100%; }
    .status-badge {
        display: inline-block; margin-top: 0.35rem; padding: 0.15rem 0.7rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: bold; line-height: 1.6;
    }
    .status-badge.available { background-color: #DFF3E3; color: #1E7B34; }
    .status-badge.unavailable { background-color: #F5DADA; color: #A32626; }
    .breadcrumb { background-color: #F0F0F0; padding: 0.5rem 1rem; border-radius: 5px; margin-bottom: 1rem; font-size: 0.85rem; border-left: 4px solid #4A90D9; line-height: 1.6; }
    .breadcrumb span { color: #4A90D9; font-weight: bold; }
    .stButton > button {
        background-color: #4A90D9 !important; color: #FFFFFF !important; border: none !important; border-radius: 6px !important;
        font-weight: bold !important; padding: 0.6rem 1rem !important; width: 100% !important; min-height: 46px;
        line-height: 1.3 !important; white-space: normal !important; font-family: 'Calisto MT', 'Georgia', serif !important;
    }
    .stButton > button:hover { background-color: #357ABD !important; color: #FFFFFF !important; }
    .stButton > button:disabled { background-color: #CCCCCC !important; color: #888888 !important; }
    .stButton { margin-top: auto; }
    .section-container { background-color: #F9F9F9; border: 2px solid #4A90D9; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; }
    .section-container h3 { color: #4A90D9; margin-top: 0; }
    .stFileUploader { border: 2px dashed #4A90D9 !important; border-radius: 10px !important; padding: 1rem !important; }
    .report-meta { background-color: #F0F4F8; border: 2px solid #4A90D9; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; font-size: 0.85rem; }
    .report-meta td { padding: 4px 8px; line-height: 1.4; }
    .footer { background-color: #000000; color: #FFFFFF; text-align: center; padding: 1.5rem; border-top: 3px solid #4A90D9; margin-top: 3rem; font-size: 0.9rem; line-height: 1.4; }
    div[data-testid="stMetric"] { padding: 0.5rem 0.75rem; }
    div[data-testid="stMetricLabel"] { white-space: normal !important; line-height: 1.3 !important; }
    div[data-testid="stMetricValue"] { line-height: 1.3 !important; overflow-wrap: break-word; font-size: 1.6rem !important; }
    div[data-testid="stMetricDelta"] { line-height: 1.3 !important; }
    .stSelectbox label, .stMultiSelect label, .stTextInput label,
    .stNumberInput label, .stDateInput label, .stRadio label,
    .stCheckbox label, .stFileUploader label {
        line-height: 1.4 !important; overflow-wrap: break-word; margin-bottom: 0.25rem !important;
    }
    div[data-baseweb="select"] { line-height: 1.4 !important; }
    div[data-baseweb="select"] > div { min-height: 42px; }
    div[data-testid="stDataFrame"] * { line-height: 1.4 !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  SESSION STATE & NAVIGATION
# =============================================================================
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'breadcrumb' not in st.session_state: st.session_state.breadcrumb = ['Home']
if 'report_metadata' not in st.session_state: st.session_state.report_metadata = {}

def navigate_to(page, breadcrumb_label=None):
    st.session_state.page = page
    if breadcrumb_label: st.session_state.breadcrumb = breadcrumb_label
    st.rerun()

def show_breadcrumb():
    if len(st.session_state.breadcrumb) > 1 or st.session_state.breadcrumb[0] != 'Home':
        bc = " > ".join([f"<span>{b}</span>" for b in st.session_state.breadcrumb])
        st.markdown(f'<div class="breadcrumb">{bc}</div>', unsafe_allow_html=True)

def back_button(target_page, target_breadcrumb):
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Back", key=f"back_{st.session_state.page}_to_{target_page}"):
        navigate_to(target_page, target_breadcrumb)


# =============================================================================
#  NAVIGATION PAGES
# =============================================================================
def render_home():
    st.markdown('<div class="hero"><h1>African Actuarial Consultants</h1><p>Comprehensive Actuarial Reserving Toolkit - IFRS 17 Compliant<br></p></div>', unsafe_allow_html=True)
    st.markdown("### Select a Module")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><h3>Full IFRS 17 Valuation</h3><p>Complete PAA valuation with Income Statement & Liability Rollforward</p></div>', unsafe_allow_html=True)
        disabled = full_engine is None
        if st.button("Open Full Valuation", key="nav_home_full", disabled=disabled, use_container_width=True):
            navigate_to('full_valuation', ['Home', 'Full Valuation'])
    with col2:
        st.markdown('<div class="card"><h3>LRC Calculators</h3><p>UPR (365th/24th/8th) & Loss Component (Onerous Contracts)</p></div>', unsafe_allow_html=True)
        if st.button("Open LRC Calculators", key="nav_home_lrc", use_container_width=True):
            navigate_to('lrc', ['Home', 'LRC Calculators'])
    with col3:
        st.markdown('<div class="card"><h3>LIC Calculators</h3><p>IBNR - OCR - ULAE - NPR - Risk Adjustment</p></div>', unsafe_allow_html=True)
        if st.button("Open LIC Calculators", key="nav_home_lic", use_container_width=True):
            navigate_to('lic', ['Home', 'LIC Calculators'])
    st.markdown('<div class="footer">2026 African Actuarial Consultants. All rights reserved.</div>', unsafe_allow_html=True)

def render_lrc():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>LRC Calculators</h1><p>Liability for Remaining Coverage - IFRS 17 PAA</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>UPR Calculator</h3><p>Unearned Premium Reserve using 365th, 24th, or 8th methods</p></div>', unsafe_allow_html=True)
        if st.button("Open UPR Calculator", key="nav_lrc_upr", use_container_width=True):
            navigate_to('upr_calculator', ['Home', 'LRC Calculators', 'UPR Calculator'])
    with col2:
        st.markdown('<div class="card"><h3>Loss Component</h3><p>Onerous contract identification</p></div>', unsafe_allow_html=True)
        disabled = loss_comp_engine is None
        if st.button("Open Loss Component", key="nav_lrc_loss", disabled=disabled, use_container_width=True):
            navigate_to('loss_component', ['Home', 'LRC Calculators', 'Loss Component'])
    back_button('home', ['Home'])

def render_lic():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>LIC Calculators</h1><p>Liability for Incurred Claims</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>Fulfilment Cashflows</h3><p>OCR - IBNR (4 methods) - ULAE - NPR</p></div>', unsafe_allow_html=True)
        if st.button("Open Fulfilment Cashflows", key="nav_lic_fcf", use_container_width=True):
            navigate_to('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows'])
    with col2:
        st.markdown('<div class="card"><h3>Risk Adjustment</h3><p>Mack Chain Ladder - ODP Bootstrap</p></div>', unsafe_allow_html=True)
        if st.button("Open Risk Adjustment", key="nav_lic_ra", use_container_width=True):
            navigate_to('risk_adjustment', ['Home', 'LIC Calculators', 'Risk Adjustment'])
    back_button('home', ['Home'])

def render_fulfilment_cashflows():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Fulfilment Cashflows</h1><p>Components of LIC</p></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    items = [("OCR", "ocr_calculator", ocr_engine), ("IBNR", "ibnr_menu", True),
             ("ULAE", "ulae_calculator", ulae_engine), ("NPR", "npr_calculator", npr_engine)]
    for i, (title, page, module) in enumerate(items):
        with cols[i]:
            available = module is not None
            st.markdown(f'<div class="card"><h3>{title}</h3><p>{"Available" if available else "Unavailable"}</p></div>', unsafe_allow_html=True)
            if st.button(f"Open {title}", key=f"nav_fcf_{page}", disabled=not available, use_container_width=True):
                navigate_to(page, ['Home', 'LIC Calculators', 'Fulfilment Cashflows', title])
    back_button('lic', ['Home', 'LIC Calculators'])

def render_ibnr_menu():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>IBNR Methods</h1><p>Select a calculation method</p></div>', unsafe_allow_html=True)
    methods = [("Percentage", "percentage_calculator", ibnr_pct), ("BCL", "bcl_calculator", ibnr_bcl),
               ("Cape Cod", "capecod_calculator", ibnr_cc), ("Bornhuetter-Ferguson", "bf_calculator", ibnr_bf)]
    for i in range(0, len(methods), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(methods):
                name, page, module = methods[i + j]
                with cols[j]:
                    available = module is not None
                    st.markdown(f'<div class="card"><h3>{name}</h3><p>{"Available" if available else "Unavailable"}</p></div>', unsafe_allow_html=True)
                    if st.button(f"Open {name}", key=f"nav_ibnr_{page}", disabled=not available, use_container_width=True):
                        navigate_to(page, ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods', name])
    back_button('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows'])

def render_risk_adjustment():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Risk Adjustment</h1><p>RA Methods for IFRS 17</p></div>', unsafe_allow_html=True)
    cols = st.columns(2)
    methods = [("Mack", "mack_calculator", mack_engine), ("Bootstrap", "bootstrap_calculator", bootstrap_engine)]
    for i, (name, page, module) in enumerate(methods):
        with cols[i]:
            available = module is not None
            st.markdown(f'<div class="card"><h3>{name}</h3><p>{"Available" if available else "Unavailable"}</p></div>', unsafe_allow_html=True)
            if st.button(f"Open {name}", key=f"nav_ra_{page}", disabled=not available, use_container_width=True):
                navigate_to(page, ['Home', 'LIC Calculators', 'Risk Adjustment', name])
    back_button('lic', ['Home', 'LIC Calculators'])


# =============================================================================
#  INDIVIDUAL CALCULATORS
# =============================================================================

# ---------- UPR ----------
def render_upr_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>UPR Calculator</h1><p>Unearned Premium Reserve - Pro-rata Methods</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: valuation_date = st.date_input("Valuation Date", value=date(2025, 12, 31), key="upr_vd")
    with c2: client_name = st.text_input("Client Name", value="Client", key="upr_cn").strip()
    with c3: method = st.selectbox("UPR Method", ["365th", "24th", "8th"], key="upr_mt")
    valuation_date_ts = pd.Timestamp(str(valuation_date))
    uploaded_file = st.file_uploader("Upload Premium Register (CSV or Excel)", type=["csv", "xlsx", "xls"], key="upr_f")
    if uploaded_file is not None:
        try:
            original_filename = uploaded_file.name
            base_filename = re.sub(r'\.[^.]*$', '', original_filename)
            df = read_uploaded_file(uploaded_file)
            unnamed = [c for c in df.columns if str(c).startswith('Unnamed:')]
            if unnamed: df = df.drop(columns=unnamed)
            df.columns = df.columns.astype(str).str.strip()
            st.dataframe(df.head(5), use_container_width=True)
            all_columns = df.columns.tolist()
            c1, c2 = st.columns(2)
            with c1: start_date_col = st.selectbox("Start Date Column", [""] + all_columns, key="upr_sd")
            with c2: end_date_col = st.selectbox("End Date Column", [""] + all_columns, key="upr_ed")
            if not start_date_col or not end_date_col: st.info("Please select Start and End Date columns."); return
            grouping_options = [c for c in all_columns if c not in [start_date_col, end_date_col]]
            grouping_cols = st.multiselect("Group By Columns", options=grouping_options, default=[grouping_options[0]] if grouping_options else [], key="upr_gc")
            if not grouping_cols: st.info("Please select at least one Group By column."); return
            numeric_columns = [c for c in all_columns if c not in [start_date_col, end_date_col] + grouping_cols and pd.api.types.is_numeric_dtype(df[c])]
            selected_value_cols = st.multiselect("Amount Columns", options=numeric_columns, default=numeric_columns[:min(4, len(numeric_columns))], key="upr_vc")
            if not selected_value_cols: st.info("Please select at least one Amount column."); return

            if st.button("Calculate UPR", key="upr_calc", use_container_width=True):
                if upr_engine is None:
                    st.error("UPR engine not available.")
                    return
                with st.spinner("Calculating UPR..."):
                    result = upr_engine.calculate_upr(
                        df=df,
                        start_date_col=start_date_col,
                        end_date_col=end_date_col,
                        value_cols=selected_value_cols,
                        grouping_cols=grouping_cols,
                        valuation_date=valuation_date_ts,
                        method=method
                    )

                st.markdown("### UPR Results")
                disp = result.copy()
                for c in selected_value_cols: disp[c] = disp[c].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
                st.dataframe(disp, use_container_width=True, hide_index=True)
                total_upr = sum(result[c].sum() for c in selected_value_cols)
                st.metric("Total UPR", f"{total_upr:,.2f}")

                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as w:
                    result.to_excel(w, index=False, sheet_name='UPR_Summary')
                output.seek(0)
                sc = sanitize_filename(client_name); so = sanitize_filename(base_filename)
                st.download_button("Download UPR Results", data=output, file_name=f"{sc}_{so}_UPR_{method}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="upr_dl")
        except Exception as e: show_error(e)
    back_button('lrc', ['Home', 'LRC Calculators'])


# ---------- Loss Component ----------
def render_loss_component():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Loss Component Calculator</h1><p>Onerous Contract Identification - IFRS 17 PAA</p></div>', unsafe_allow_html=True)
    client_name = st.text_input("Client Name", value="Client", key="lc_cn").strip()
    uploaded_file = st.file_uploader("Upload Data File (CSV or Excel)", type=["csv", "xlsx", "xls"], key="lc_f")
    if uploaded_file is not None:
        try:
            df = read_uploaded_file(uploaded_file)
            df.columns = df.columns.astype(str).str.strip()
            st.dataframe(df.head(5), use_container_width=True)
            st.markdown("### Column Mapping")
            cols = df.columns.tolist()
            c1, c2, c3 = st.columns(3)
            with c1:
                lob_col = st.selectbox("Line of Business", cols, key="lc_lob")
                opening_ocr_col = st.selectbox("Opening OCR", cols, key="lc_oocr")
                opening_ibnr_col = st.selectbox("Opening IBNR", cols, key="lc_oibnr")
            with c2:
                wp_col = st.selectbox("Written Premium", cols, key="lc_wp")
                closing_ocr_col = st.selectbox("Closing OCR", cols, key="lc_cocr")
                closing_ibnr_col = st.selectbox("Closing IBNR", cols, key="lc_cibnr")
            with c3:
                commission_col = st.selectbox("Commission Paid", cols, key="lc_comm")
                paid_claims_col = st.selectbox("Paid Claims", cols, key="lc_pc")
                ra_col = st.selectbox("Risk Adjustment", cols, key="lc_ra")
            c1, c2 = st.columns(2)
            with c1: expenses_col = st.selectbox("Expenses", cols, key="lc_exp")
            with c2: opening_upr_col = st.selectbox("Opening UPR", cols, key="lc_oupr")
            closing_upr_col = st.selectbox("Closing UPR", cols, key="lc_cupr")
            if st.button("Calculate Loss Component", key="lc_run", use_container_width=True):
                if loss_comp_engine is None: st.error("Loss Component engine not available."); return
                with st.spinner("Calculating..."):
                    result = loss_comp_engine.calculate_loss_component(
                        df=df, lob_col=lob_col, written_premium_col=wp_col,
                        expenses_col=expenses_col, commission_col=commission_col,
                        paid_claims_col=paid_claims_col, opening_ocr_col=opening_ocr_col,
                        closing_ocr_col=closing_ocr_col, opening_ibnr_col=opening_ibnr_col,
                        closing_ibnr_col=closing_ibnr_col, opening_upr_col=opening_upr_col,
                        closing_upr_col=closing_upr_col, risk_adjustment_col=ra_col
                    )
                st.markdown("### Loss Component Results")
                disp = result.copy()
                for c in disp.columns:
                    if c != lob_col and pd.api.types.is_numeric_dtype(disp[c]):
                        disp[c] = disp[c].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
                st.dataframe(disp, use_container_width=True, hide_index=True)
                total_lc = result['Loss_Component'].sum()
                st.metric("Total Loss Component", f"{total_lc:,.2f}")
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as w: result.to_excel(w, index=False, sheet_name='Loss_Component')
                output.seek(0)
                sc = sanitize_filename(client_name)
                st.download_button("Download Results", data=output, file_name=f"{sc}_Loss_Component.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="lc_dl")
        except Exception as e: show_error(e)
    back_button('lrc', ['Home', 'LRC Calculators'])


# ---------- OCR ----------
def render_ocr_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>OCR Calculator</h1><p>Outstanding Claims Reserve - Group & Sum</p></div>', unsafe_allow_html=True)
    client_name = st.text_input("Client Name", value="Client", key="ocr_cn").strip()
    uploaded_file = st.file_uploader("Upload Case Estimates File (CSV or Excel)", type=["csv", "xlsx", "xls"], key="ocr_f")
    if uploaded_file is not None:
        try:
            original_filename = uploaded_file.name; base_filename = re.sub(r'\.[^.]*$', '', original_filename)
            df = read_uploaded_file(uploaded_file)
            unnamed = [c for c in df.columns if str(c).startswith('Unnamed:')]
            if unnamed: df = df.drop(columns=unnamed)
            df.columns = df.columns.astype(str).str.strip()
            st.dataframe(df.head(5), use_container_width=True)
            all_columns = df.columns.tolist()

            st.markdown("### Date Filters")
            include_date_filters = st.checkbox(
                "Include Date Filters",
                value=False,
                key="ocr_use_date_filters",
                help="Turn on if your file has a Loss Date and/or Notification Date and you want to restrict "
                     "the OCR calculation to specific periods on either date."
            )
            loss_date_col = notif_date_col = None
            loss_filter_type = notif_filter_type = "No Filter"
            loss_d1 = loss_d2 = notif_d1 = notif_d2 = None
            if include_date_filters:
                st.caption("Filter independently on the Loss Date and/or the Notification (Report) Date.")
                fc1, fc2 = st.columns(2)
                with fc1:
                    loss_date_col, loss_filter_type, loss_d1, loss_d2 = _render_single_date_filter_ui(all_columns, "Loss Date", "ocr_lossdt")
                with fc2:
                    notif_date_col, notif_filter_type, notif_d1, notif_d2 = _render_single_date_filter_ui(all_columns, "Notification Date", "ocr_notifdt")

                mask = _apply_flexible_date_filter(df, loss_date_col, loss_filter_type, loss_d1, loss_d2)
                mask &= _apply_flexible_date_filter(df, notif_date_col, notif_filter_type, notif_d1, notif_d2)
                df = df[mask].copy()
                st.info(f"{len(df):,} record(s) remain after date filtering.")
                if df.empty:
                    st.warning("No records remain after applying the date filters.")
                    back_button('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows'])
                    return

            grouping_cols = st.multiselect("Group By Columns", options=all_columns, default=[all_columns[0]] if all_columns else [], key="ocr_gc")
            if not grouping_cols: st.info("Please select at least one Group By column."); return
            numeric_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c not in grouping_cols]
            selected_value_cols = st.multiselect("Amount Columns", options=numeric_columns, default=numeric_columns[:min(5, len(numeric_columns))], key="ocr_vc")
            if not selected_value_cols: st.info("Please select at least one Amount column."); return

            if st.button("Calculate OCR", key="ocr_calc", use_container_width=True):
                if ocr_engine is None:
                    st.error("OCR engine not available.")
                    return
                with st.spinner("Calculating OCR..."):
                    grouped, cleaning_report, grand_total = ocr_engine.calculate_ocr(
                        df=df,
                        grouping_cols=grouping_cols,
                        value_cols=selected_value_cols,
                        clean_data=True
                    )

                st.markdown("### OCR Summary")
                if cleaning_report.get('duplicates_removed', 0) > 0:
                    st.info(f"Cleaning: {cleaning_report['duplicates_removed']} duplicate rows removed.")
                if cleaning_report.get('conversion_issues'):
                    for issue in cleaning_report['conversion_issues']:
                        st.warning(f"Column '{issue['column']}': {issue['failed_count']} non-numeric values converted to 0.")

                disp = grouped.copy()
                for c in selected_value_cols:
                    disp[c] = disp[c].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
                st.dataframe(disp, use_container_width=True, hide_index=True)
                st.metric("Grand Total OCR", f"{grand_total:,.2f}")

                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as w: grouped.to_excel(w, index=False, sheet_name='OCR_Results')
                output.seek(0)
                sc = sanitize_filename(client_name); so = sanitize_filename(base_filename)
                st.download_button("Download OCR Results", data=output, file_name=f"{sc}_{so}_OCR.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="ocr_dl")
        except Exception as e: show_error(e)
    back_button('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows'])


# ---------- Percentage IBNR ----------
def render_percentage_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Percentage IBNR Calculator</h1><p>Simple Method: IBNR = Amount x IBNR Percentage</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: client_name = st.text_input("Client Name", value="Client", key="pct_cn").strip()
    with c2: ibnr_pct_val = st.number_input("IBNR Percentage (%)", 0.0, 100.0, 10.0, 0.5, key="pct_pct") / 100.0
    c1, c2 = st.columns(2)
    with c1: from_date = st.date_input("From Date", date(2020, 1, 1), key="pct_fd")
    with c2: to_date = st.date_input("To Date", date(2025, 12, 31), key="pct_td")
    uploaded = st.file_uploader("Upload Data File (CSV/Excel)", type=["csv", "xlsx", "xls"], key="pct_f")
    if uploaded is None:
        st.info("Upload a file with Date, Line of Business, and Amount columns.")
        back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods']); return
    try:
        df = read_uploaded_file(uploaded)
        df.columns = df.columns.astype(str).str.strip()
        st.dataframe(df.head(5), use_container_width=True)
        cols = df.columns.tolist()
        c1, c2 = st.columns(2)
        with c1: date_col = st.selectbox("Date Column", cols, key="pct_date")
        with c2: lob_col = st.selectbox("Line of Business Column", cols, key="pct_lob")
        amount_candidates = [c for c in cols if c not in [date_col, lob_col] and pd.api.types.is_numeric_dtype(df[c])]
        amount_cols = st.multiselect("Amount Column(s)", amount_candidates, key="pct_amt")
        if not amount_cols: st.warning("Please select at least one Amount column."); return
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        from_dt = pd.Timestamp(str(from_date)); to_dt = pd.Timestamp(str(to_date))
        df = _date_filter(df, date_col, from_date, to_date)
        if df.empty: st.warning("No data in selected date range."); return
        if st.button("Calculate Percentage IBNR", key="pct_run", use_container_width=True):
            if ibnr_pct is None: st.error("Percentage IBNR engine not available."); return
            with st.spinner("Calculating..."):
                summary_df, grand_total = ibnr_pct.calculate_percentage_ibnr(
                    df=df, date_col=date_col, lob_col=lob_col,
                    amount_cols=amount_cols, from_date=from_dt,
                    to_date=to_dt, ibnr_pct=ibnr_pct_val
                )
            st.markdown("### Percentage IBNR Results")
            disp = summary_df.copy()
            for c in disp.columns:
                if c != lob_col: disp[c] = disp[c].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.metric("Total IBNR", f"{grand_total:,.2f}")
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w: summary_df.to_excel(w, index=False, sheet_name='Percentage_IBNR')
            output.seek(0)
            sc = sanitize_filename(client_name)
            st.download_button("Download Results", data=output, file_name=f"{sc}_Percentage_IBNR.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="pct_dl")
    except Exception as e: show_error(e)
    back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])


# ---------- BCL ----------
def render_bcl_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Basic Chain Ladder (BCL) - IBNR</h1><p>Multi-LDF Methods with Inflation & Discounting Support</p></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: client_name = st.text_input("Client Name", value="Client", key="bcl_cn").strip()
    with c2: grain = st.selectbox("Period Grain", ["Yearly", "Quarterly", "Monthly"], key="bcl_gr")
    grain_map = {"Yearly": "Y", "Quarterly": "Q", "Monthly": "M"}; grain_code = grain_map[grain]; ppy = {"Y": 1, "Q": 4, "M": 12}[grain_code]
    with c3: from_date = st.date_input("From Date", date(2020, 1, 1), key="bcl_fd")
    with c4: to_date = st.date_input("To Date", date(2025, 12, 31), key="bcl_td")
    uploaded = st.file_uploader("Upload Claims Data (Loss Date, Report Date, LOB, Amount)", type=["csv", "xlsx", "xls"], key="bcl_f")
    if uploaded is None: st.info("Upload claims data file."); back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods']); return
    try:
        df = read_uploaded_file(uploaded)
        df.columns = df.columns.astype(str).str.strip()
        st.dataframe(df.head(5), use_container_width=True)
        cols = df.columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1: loss_col = st.selectbox("Loss Date Column", cols, key="bcl_ld")
        with c2: rep_col = st.selectbox("Report Date Column", cols, key="bcl_rd")
        with c3: lob_col = st.selectbox("LOB Column", cols, key="bcl_lob")
        amount_candidates = [c for c in cols if c not in [loss_col, rep_col, lob_col] and pd.api.types.is_numeric_dtype(df[c])]
        amount_cols = st.multiselect("Amount Column(s)", amount_candidates, key="bcl_amt")
        if not amount_cols: st.warning("Please select at least one Amount column."); return
        df[loss_col] = pd.to_datetime(df[loss_col], errors='coerce'); df[rep_col] = pd.to_datetime(df[rep_col], errors='coerce')
        for ac in amount_cols: df[ac] = pd.to_numeric(df[ac], errors='coerce').fillna(0)
        df = df.dropna(subset=[loss_col, rep_col])
        from_dt = pd.Timestamp(str(from_date)); to_dt = pd.Timestamp(str(to_date))
        df = _date_filter(df, loss_col, from_date, to_date)
        n_periods = int((to_dt.year - from_dt.year) * ppy) + 1
        st.markdown("### LDF Method Selection")
        selected_method = "volume_weighted"
        if engine_utils is not None and ibnr_bcl is not None and hasattr(ibnr_bcl, 'calculate_all_ldfs'):
            sample_amt = amount_cols[0]
            _, sample_cum, _ = engine_utils.build_triangles(df, loss_col, rep_col, sample_amt, from_dt, grain_code, n_periods)
            all_ldfs = ibnr_bcl.calculate_all_ldfs(sample_cum, n_periods)
            ldf_df = pd.DataFrame({
                "Dev Period": range(1, len(all_ldfs["volume_weighted"]) + 1),
                "Vol-Weighted": all_ldfs["volume_weighted"], "Simple Avg": all_ldfs["simple_average"],
                "Geometric": all_ldfs["geometric"], "Medial": all_ldfs["medial"],
                "Lin Regression": all_ldfs["linear_regression"], "Wtd Last 3": all_ldfs["weighted_last_3"]
            })
            st.dataframe(ldf_df.round(4), use_container_width=True)
            rec_method = "volume_weighted"; min_cv = float('inf')
            for method in ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"]:
                factors = all_ldfs[method]
                if len(factors) >= 3:
                    cv = np.std(factors[:3]) / np.mean(factors[:3]) if np.mean(factors[:3]) > 0 else float('inf')
                    if cv < min_cv: min_cv = cv; rec_method = method
            st.info(f"Recommended: {rec_method.replace('_', ' ').title()} (lowest CV: {min_cv:.2%})")
            selected_method = st.selectbox(
                "Select LDF Method to Use",
                ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"],
                index=["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"].index(rec_method),
                key="bcl_ldf_method"
            )
        st.markdown("### Adjustments")
        c1, c2 = st.columns(2)
        with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key="bcl_inf")
        with c2: use_discounting = st.checkbox("Apply Discounting", key="bcl_disc")
        cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
        if use_inflation: cum_inflation, per_period_rates = load_inflation_data_ui(grain_code, ppy, "bcl")
        if use_discounting: spot_rates, flat_rate = load_discounting_data_ui(grain_code, ppy, "bcl")
        if st.button("Calculate BCL IBNR", key="bcl_run", use_container_width=True):
            if ibnr_bcl is None or engine_utils is None: st.error("Required engines not available."); return
            with st.spinner("Calculating BCL IBNR..."):
                lobs = sorted(df[lob_col].dropna().unique())
                all_results = []
                inc_triangles = {}
                ldfs_per_lob = {}
                for lob in lobs:
                    lob_data = df[df[lob_col] == lob].copy()
                    for idx, ac in enumerate(amount_cols):
                        inc, cum, _ = engine_utils.build_triangles(lob_data, loss_col, rep_col, ac, from_dt, grain_code, n_periods)
                        if idx == 0:
                            inc_triangles[lob] = inc.copy()
                        result = ibnr_bcl.calculate_bcl_ibnr(
                            cum_triangle=cum, start_date=from_dt, period_unit=grain_code,
                            selected_ldf_method=selected_method,
                            use_inflation=use_inflation, cum_inflation=cum_inflation,
                            per_period_rates=per_period_rates,
                            use_discounting=use_discounting, spot_rates=spot_rates, flat_rate=flat_rate
                        )
                        if idx == 0:
                            ldfs_per_lob[lob] = result['dev_factors']
                        res_df = result['results_df']; res_df['LOB'] = lob; res_df['Amount_Col'] = ac
                        all_results.append(res_df)
                final_df = pd.concat(all_results, ignore_index=True)
                summary = final_df.groupby(['LOB', 'Amount_Col'])[['Current_Claims', 'IBNR']].sum().reset_index()
            st.markdown("### BCL IBNR Summary")
            disp = summary.copy()
            for c in ['Current_Claims', 'IBNR']:
                if c in disp.columns: disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            total_ibnr = summary['IBNR'].sum() if 'IBNR' in summary.columns else 0
            st.metric("Total BCL IBNR", f"{total_ibnr:,.2f}")

            export_sheets = {'BCL_Summary': summary, 'BCL_Detail': final_df}
            for lob in lobs:
                if lob in inc_triangles:
                    tri = inc_triangles[lob].copy()
                    tri.columns = [f"Dev_{c}" for c in tri.columns]
                    tri.index = [f"AY_{i}" for i in tri.index]
                    export_sheets[f"Incremental_{lob}"] = tri.reset_index()
                if lob in ldfs_per_lob and ldfs_per_lob[lob]:
                    ldf_df = pd.DataFrame({
                        'Dev_Period': [f"{i}-{i+1}" for i in range(len(ldfs_per_lob[lob]))],
                        'Factor': ldfs_per_lob[lob]
                    })
                    export_sheets[f"LDFs_{lob}"] = ldf_df
                lob_final = final_df[(final_df['LOB'] == lob) & (final_df['Amount_Col'] == amount_cols[0])]
                if not lob_final.empty:
                    export_sheets[f"Ultimate_{lob}"] = lob_final[['Accident_Period_Label', 'Current_Claims', 'Ultimate_Claims', 'IBNR']]
            output, ext, mime = build_download_payload(export_sheets)
            sc = sanitize_filename(client_name)
            st.download_button("Download BCL Results", data=output, file_name=f"{sc}_BCL_IBNR.{ext}", mime=mime, key="bcl_dl")
    except Exception as e: show_error(e)
    back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])


# ---------- Cape Cod ----------
def render_capecod_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Cape Cod - IBNR</h1><p>Uses premiums to derive expected loss ratio</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: client_name = st.text_input("Client Name", value="Client", key="cc_cn").strip()
    with c2: from_date = st.date_input("From Date", date(2020, 1, 1), key="cc_fd")
    with c3: to_date = st.date_input("To Date", date(2025, 12, 31), key="cc_td")
    claims_file = st.file_uploader("Claims Data (Loss Date, Report Date, LOB, Amount)", type=["csv", "xlsx", "xls"], key="cc_cf")

    premium_structure = st.radio(
        "Premium Data Structure",
        ["Per LOB Column (wide format)", "Per Row (Accident Year, LOB, Premium)"],
        key="cc_prem_structure",
        help="'Per LOB Column': One column per LOB with accident years as rows.\n"
             "'Per Row': Each row has Accident Year, LOB, and Premium amount."
    )

    prem_file = st.file_uploader("Premiums Data", type=["csv", "xlsx", "xls"], key="cc_pf")
    if claims_file is None or prem_file is None:
        st.info("Upload both claims and premiums files.")
        back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])
        return

    try:
        df = read_uploaded_file(claims_file)
        df.columns = df.columns.astype(str).str.strip()
        prem_df = read_uploaded_file(prem_file)
        prem_df.columns = prem_df.columns.astype(str).str.strip()

        st.dataframe(df.head(3), use_container_width=True)
        st.dataframe(prem_df.head(3), use_container_width=True)

        cols = df.columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1: loss_col = st.selectbox("Loss Date", cols, key="cc_ld")
        with c2: rep_col = st.selectbox("Report Date", cols, key="cc_rd")
        with c3: lob_col = st.selectbox("LOB", cols, key="cc_lob")
        amount_candidates = [c for c in cols if c not in [loss_col, rep_col, lob_col] and pd.api.types.is_numeric_dtype(df[c])]
        amount_cols = st.multiselect("Amount Column(s)", amount_candidates, key="cc_amt")
        if not amount_cols:
            st.warning("Please select at least one Amount column.")
            return

        df[loss_col] = pd.to_datetime(df[loss_col], errors='coerce')
        df[rep_col] = pd.to_datetime(df[rep_col], errors='coerce')
        for ac in amount_cols: df[ac] = pd.to_numeric(df[ac], errors='coerce').fillna(0)
        df = df.dropna(subset=[loss_col, rep_col])
        from_dt = pd.Timestamp(str(from_date))
        to_dt = pd.Timestamp(str(to_date))
        df = _date_filter(df, loss_col, from_date, to_date)
        lobs = sorted(df[lob_col].dropna().unique())
        grain = "Y"
        ppy = 1
        n_periods = to_dt.year - from_dt.year + 1

        p_cols = prem_df.columns.tolist()
        lob_mapping = {}

        if premium_structure == "Per LOB Column (wide format)":
            prem_period_col = st.selectbox(
                "Accident Year Column (e.g. 2020, 2021, ...)",
                p_cols,
                key="cc_prem_period",
                help="Column containing the calendar year for each row. Values must be years matching the triangle's accident period range."
            )
            avail_cols = [c for c in p_cols if c != prem_period_col]
            if len(avail_cols) < len(lobs):
                st.error(
                    f"Number of premium LOB columns ({len(avail_cols)}) is less than the number of unique LOBs in claims ({len(lobs)}).\n\n"
                    f"Your claims data has LOBs: {', '.join(lobs)}\n"
                    f"Your premium file has these potential LOB columns: {', '.join(avail_cols)}"
                )
                return
            st.markdown("**Map each Line of Business to a premium column:**")
            for lob in lobs:
                options = avail_cols
                default = next((c for c in avail_cols if c.strip().lower() == lob.strip().lower()), avail_cols[0])
                idx = options.index(default)
                lob_mapping[lob] = st.selectbox(f"Column for '{lob}'", options, index=idx, key=f"cc_map_{lob}")
        else:
            st.markdown("**Map premium data columns:**")
            c1, c2, c3 = st.columns(3)
            with c1:
                prem_period_col = st.selectbox(
                    "Accident Year Column",
                    p_cols,
                    key="cc_prem_ay"
                )
            with c2:
                prem_lob_col = st.selectbox("LOB Column", p_cols, key="cc_prem_lob")
            with c3:
                prem_val_col = st.selectbox("Premium Amount Column", [c for c in p_cols if c not in [prem_period_col, prem_lob_col]], key="cc_prem_val")

        st.markdown("### LDF Method Selection")
        selected_method = "volume_weighted"
        if engine_utils is not None and ibnr_cc is not None and hasattr(ibnr_cc, 'calculate_all_ldfs'):
            sample_amt = amount_cols[0]
            _, sample_cum, _ = engine_utils.build_triangles(df, loss_col, rep_col, sample_amt, from_dt, grain, n_periods)
            all_ldfs = ibnr_cc.calculate_all_ldfs(sample_cum, n_periods)
            ldf_df = pd.DataFrame({
                "Dev Period": range(1, len(all_ldfs["volume_weighted"]) + 1),
                "Vol-Weighted": all_ldfs["volume_weighted"], "Simple Avg": all_ldfs["simple_average"],
                "Geometric": all_ldfs["geometric"], "Medial": all_ldfs["medial"],
                "Lin Regression": all_ldfs["linear_regression"], "Wtd Last 3": all_ldfs["weighted_last_3"]
            })
            st.dataframe(ldf_df.round(4), use_container_width=True)
            selected_method = st.selectbox(
                "Select LDF Method",
                ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"],
                key="cc_ldf_m"
            )

        if st.button("Calculate Cape Cod IBNR", key="cc_run", use_container_width=True):
            if ibnr_cc is None or engine_utils is None: st.error("Required engines not available."); return
            with st.spinner("Calculating Cape Cod IBNR..."):
                all_results = []
                for lob in lobs:
                    lob_data = df[df[lob_col] == lob].copy()
                    if premium_structure == "Per LOB Column (wide format)":
                        mapped_col = lob_mapping[lob]
                        prem_series = prem_df.set_index(prem_period_col)[mapped_col].to_dict()
                    else:
                        sub_p = prem_df[prem_df[prem_lob_col].astype(str).str.strip().str.lower() == str(lob).strip().lower()]
                        prem_series = sub_p.set_index(prem_period_col)[prem_val_col].to_dict()

                    expected_years = [from_dt.year + i for i in range(n_periods)]
                    premiums_arr = np.array([float(prem_series.get(y, 0.0) or 0.0) for y in expected_years])

                    for ac in amount_cols:
                        inc, cum, _ = engine_utils.build_triangles(lob_data, loss_col, rep_col, ac, from_dt, grain, n_periods)
                        res = ibnr_cc.calculate_cape_cod_ibnr(
                            cum_triangle=cum, premiums=premiums_arr, start_date=from_dt, period_unit=grain,
                            selected_ldf_method=selected_method
                        )
                        rdf = res['results_df']; rdf['LOB'] = lob; rdf['Amount_Col'] = ac
                        all_results.append(rdf)
                final_df = pd.concat(all_results, ignore_index=True)
                summary = final_df.groupby(['LOB', 'Amount_Col'])[['Current_Claims', 'IBNR']].sum().reset_index()

            st.markdown("### Cape Cod IBNR Summary")
            disp = summary.copy()
            for c in ['Current_Claims', 'IBNR']: disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            total_ibnr = summary['IBNR'].sum()
            st.metric("Total Cape Cod IBNR", f"{total_ibnr:,.2f}")

            output, ext, mime = build_download_payload({'CC_Summary': summary, 'CC_Detail': final_df})
            sc = sanitize_filename(client_name)
            st.download_button("Download Cape Cod Results", data=output, file_name=f"{sc}_CapeCod_IBNR.{ext}", mime=mime, key="cc_dl")
    except Exception as e: show_error(e)
    back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])


# ---------- BF ----------
def render_bf_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Bornhuetter-Ferguson (BF) - IBNR</h1><p>Combines initial expected loss ratios with chain ladder development</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: client_name = st.text_input("Client Name", value="Client", key="bf_cn").strip()
    with c2: from_date = st.date_input("From Date", date(2020, 1, 1), key="bf_fd")
    with c3: to_date = st.date_input("To Date", date(2025, 12, 31), key="bf_td")
    claims_file = st.file_uploader("Claims Data (Loss Date, Report Date, LOB, Amount)", type=["csv", "xlsx", "xls"], key="bf_cf")

    premium_structure = st.radio(
        "Premium Data Structure",
        ["Per LOB Column (wide format)", "Per Row (Accident Year, LOB, Premium)"],
        key="bf_prem_structure"
    )
    prem_file = st.file_uploader("Premiums Data", type=["csv", "xlsx", "xls"], key="bf_pf")
    if claims_file is None or prem_file is None:
        st.info("Upload both claims and premiums files.")
        back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods']); return

    try:
        df = read_uploaded_file(claims_file)
        df.columns = df.columns.astype(str).str.strip()
        prem_df = read_uploaded_file(prem_file)
        prem_df.columns = prem_df.columns.astype(str).str.strip()

        st.dataframe(df.head(3), use_container_width=True)
        cols = df.columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1: loss_col = st.selectbox("Loss Date", cols, key="bf_ld")
        with c2: rep_col = st.selectbox("Report Date", cols, key="bf_rd")
        with c3: lob_col = st.selectbox("LOB", cols, key="bf_lob")
        amount_candidates = [c for c in cols if c not in [loss_col, rep_col, lob_col] and pd.api.types.is_numeric_dtype(df[c])]
        amount_cols = st.multiselect("Amount Column(s)", amount_candidates, key="bf_amt")
        if not amount_cols: st.warning("Please select at least one Amount column."); return

        df[loss_col] = pd.to_datetime(df[loss_col], errors='coerce')
        df[rep_col] = pd.to_datetime(df[rep_col], errors='coerce')
        for ac in amount_cols: df[ac] = pd.to_numeric(df[ac], errors='coerce').fillna(0)
        df = df.dropna(subset=[loss_col, rep_col])
        from_dt = pd.Timestamp(str(from_date)); to_dt = pd.Timestamp(str(to_date))
        df = _date_filter(df, loss_col, from_date, to_date)
        lobs = sorted(df[lob_col].dropna().unique())
        grain = "Y"; n_periods = to_dt.year - from_dt.year + 1

        p_cols = prem_df.columns.tolist()
        lob_mapping = {}

        if premium_structure == "Per LOB Column (wide format)":
            prem_period_col = st.selectbox("Accident Year Column", p_cols, key="bf_prem_period")
            avail_cols = [c for c in p_cols if c != prem_period_col]
            st.markdown("**Map each LOB to premium column:**")
            for lob in lobs:
                default = next((c for c in avail_cols if c.strip().lower() == lob.strip().lower()), avail_cols[0])
                lob_mapping[lob] = st.selectbox(f"Column for '{lob}'", avail_cols, index=avail_cols.index(default), key=f"bf_map_{lob}")
        else:
            c1, c2, c3 = st.columns(3)
            with c1: prem_period_col = st.selectbox("Accident Year Column", p_cols, key="bf_prem_ay")
            with c2: prem_lob_col = st.selectbox("LOB Column", p_cols, key="bf_prem_lob")
            with c3: prem_val_col = st.selectbox("Premium Column", [c for c in p_cols if c not in [prem_period_col, prem_lob_col]], key="bf_prem_val")

        st.markdown("### Expected Loss Ratios (ELR) per LOB")
        elr_inputs = {}
        elr_cols = st.columns(min(len(lobs), 4))
        for i, lob in enumerate(lobs):
            with elr_cols[i % 4]:
                elr_inputs[lob] = st.number_input(f"ELR % for '{lob}'", 0.0, 200.0, 65.0, 1.0, key=f"bf_elr_{lob}") / 100.0

        st.markdown("### LDF Method Selection")
        selected_method = "volume_weighted"
        if engine_utils is not None and ibnr_bf is not None and hasattr(ibnr_bf, 'calculate_all_ldfs'):
            sample_amt = amount_cols[0]
            _, sample_cum, _ = engine_utils.build_triangles(df, loss_col, rep_col, sample_amt, from_dt, grain, n_periods)
            all_ldfs = ibnr_bf.calculate_all_ldfs(sample_cum, n_periods)
            ldf_df = pd.DataFrame({
                "Dev Period": range(1, len(all_ldfs["volume_weighted"]) + 1),
                "Vol-Weighted": all_ldfs["volume_weighted"], "Simple Avg": all_ldfs["simple_average"],
                "Geometric": all_ldfs["geometric"], "Medial": all_ldfs["medial"],
                "Lin Regression": all_ldfs["linear_regression"], "Wtd Last 3": all_ldfs["weighted_last_3"]
            })
            st.dataframe(ldf_df.round(4), use_container_width=True)
            selected_method = st.selectbox(
                "Select LDF Method",
                ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"],
                key="bf_ldf_m"
            )

        if st.button("Calculate BF IBNR", key="bf_run", use_container_width=True):
            if ibnr_bf is None or engine_utils is None: st.error("Required engines not available."); return
            with st.spinner("Calculating BF IBNR..."):
                all_results = []
                for lob in lobs:
                    lob_data = df[df[lob_col] == lob].copy()
                    if premium_structure == "Per LOB Column (wide format)":
                        mapped_col = lob_mapping[lob]
                        prem_series = prem_df.set_index(prem_period_col)[mapped_col].to_dict()
                    else:
                        sub_p = prem_df[prem_df[prem_lob_col].astype(str).str.strip().str.lower() == str(lob).strip().lower()]
                        prem_series = sub_p.set_index(prem_period_col)[prem_val_col].to_dict()

                    expected_years = [from_dt.year + i for i in range(n_periods)]
                    premiums_arr = np.array([float(prem_series.get(y, 0.0) or 0.0) for y in expected_years])
                    elr_val = elr_inputs[lob]

                    for ac in amount_cols:
                        inc, cum, _ = engine_utils.build_triangles(lob_data, loss_col, rep_col, ac, from_dt, grain, n_periods)
                        res = ibnr_bf.calculate_bf_ibnr(
                            cum_triangle=cum, premiums=premiums_arr, elr=elr_val,
                            start_date=from_dt, period_unit=grain, selected_ldf_method=selected_method
                        )
                        rdf = res['results_df']; rdf['LOB'] = lob; rdf['Amount_Col'] = ac
                        all_results.append(rdf)
                final_df = pd.concat(all_results, ignore_index=True)
                summary = final_df.groupby(['LOB', 'Amount_Col'])[['Current_Claims', 'IBNR']].sum().reset_index()

            st.markdown("### BF IBNR Summary")
            disp = summary.copy()
            for c in ['Current_Claims', 'IBNR']: disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            total_ibnr = summary['IBNR'].sum()
            st.metric("Total BF IBNR", f"{total_ibnr:,.2f}")

            output, ext, mime = build_download_payload({'BF_Summary': summary, 'BF_Detail': final_df})
            sc = sanitize_filename(client_name)
            st.download_button("Download BF Results", data=output, file_name=f"{sc}_BF_IBNR.{ext}", mime=mime, key="bf_dl")
    except Exception as e: show_error(e)
    back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])


# ---------- ULAE ----------
def render_ulae_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>ULAE Calculator</h1><p>Unallocated Loss Adjustment Expenses</p></div>', unsafe_allow_html=True)
    client_name = st.text_input("Client Name", value="Client", key="ulae_cn").strip()
    calc_type = st.radio("Calculation Mode", ["Per Portfolio", "Aggregated"], key="ulae_mode", horizontal=True)

    reserves_file = st.file_uploader("Upload Reserves File (LOB, Reserve_Amount)", type=["csv", "xlsx", "xls"], key="ulae_rf")
    if reserves_file is None:
        st.info("Upload reserves summary file.")
        back_button('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows']); return

    try:
        df_res = read_uploaded_file(reserves_file)
        df_res.columns = df_res.columns.astype(str).str.strip()
        st.dataframe(df_res.head(3), use_container_width=True)

        if calc_type == "Per Portfolio":
            ratio_file = st.file_uploader("Upload ULAE Ratios File (LOB, ULAE_Ratio %)", type=["csv", "xlsx", "xls"], key="ulae_pf")
            if ratio_file is None: st.info("Upload portfolio ULAE ratios file."); return
            df_rat = read_uploaded_file(ratio_file)
            st.dataframe(df_rat.head(3), use_container_width=True)

            if st.button("Calculate ULAE (Per Portfolio)", key="ulae_run_p", use_container_width=True):
                if ulae_engine is None: st.error("ULAE engine not available."); return
                with st.spinner("Calculating..."):
                    res_df = ulae_engine.calculate_ulae_per_portfolio(df_reserves=df_res, ulae_ratios=df_rat, is_detailed=True)
                st.markdown("### ULAE Results")
                st.dataframe(res_df, use_container_width=True)
                output, ext, mime = build_download_payload({'ULAE_Portfolio': res_df})
                sc = sanitize_filename(client_name)
                st.download_button("Download Results", data=output, file_name=f"{sc}_ULAE_Portfolio.{ext}", mime=mime, key="ulae_dl_p")
        else:
            ulae_pct = st.number_input("Global ULAE Ratio (%)", 0.0, 100.0, 5.0, 0.5, key="ulae_g_pct") / 100.0
            app_file = st.file_uploader("Upload Apportionment Data (optional)", type=["csv", "xlsx", "xls"], key="ulae_app_f")
            df_app = read_uploaded_file(app_file) if app_file else None

            if st.button("Calculate ULAE (Aggregated)", key="ulae_run_a", use_container_width=True):
                if ulae_engine is None: st.error("ULAE engine not available."); return
                with st.spinner("Calculating..."):
                    res_df, grand_total = ulae_engine.calculate_ulae_aggregated(df_reserves=df_res, ulae_ratio=ulae_pct, apportionment_df=df_app, is_detailed=True)
                st.markdown("### ULAE Results")
                st.dataframe(res_df, use_container_width=True)
                st.metric("Grand Total ULAE", f"{grand_total:,.2f}")
                output, ext, mime = build_download_payload({'ULAE_Aggregated': res_df})
                sc = sanitize_filename(client_name)
                st.download_button("Download Results", data=output, file_name=f"{sc}_ULAE_Aggregated.{ext}", mime=mime, key="ulae_dl_a")
    except Exception as e: show_error(e)
    back_button('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows'])


# ---------- NPR ----------
def render_npr_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Non-Performance Risk (NPR)</h1><p>Reinsurance Credit Risk / Non-Performance</p></div>', unsafe_allow_html=True)
    client_name = st.text_input("Client Name", value="Client", key="npr_cn").strip()
    calc_mode = st.radio("Calculation Mode", ["Aggregated", "Per Portfolio"], key="npr_mode", horizontal=True)

    ri_file = st.file_uploader("Upload Reinsurance Data", type=["csv", "xlsx", "xls"], key="npr_rif")
    lic_file = st.file_uploader("Upload Gross LIC Data", type=["csv", "xlsx", "xls"], key="npr_licf")
    if ri_file is None or lic_file is None:
        st.info("Upload both Reinsurance and Gross LIC files.")
        back_button('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows']); return

    try:
        df_ri = read_uploaded_file(ri_file)
        df_lic = read_uploaded_file(lic_file)
        st.dataframe(df_ri.head(3), use_container_width=True)
        st.dataframe(df_lic.head(3), use_container_width=True)

        if st.button("Calculate NPR", key="npr_run", use_container_width=True):
            if npr_engine is None: st.error("NPR engine not available."); return
            with st.spinner("Calculating NPR..."):
                if calc_mode == "Aggregated":
                    res_ri, res_lic, summary_df, total_npr = npr_engine.calculate_npr_aggregation(df_ri=df_ri, df_lic=df_lic)
                else:
                    portfolios = df_ri['Portfolio'].unique().tolist() if 'Portfolio' in df_ri.columns else []
                    res_ri, res_lic, summary_df, total_npr = npr_engine.calculate_npr_per_portfolio(df_ri=df_ri, df_lic=df_lic, portfolios=portfolios)

            st.markdown("### NPR Results")
            st.dataframe(summary_df, use_container_width=True)
            st.metric("Total NPR", f"{total_npr:,.2f}")
            output, ext, mime = build_download_payload({'NPR_Summary': summary_df, 'RI_Detail': res_ri, 'LIC_Detail': res_lic})
            sc = sanitize_filename(client_name)
            st.download_button("Download NPR Results", data=output, file_name=f"{sc}_NPR_Results.{ext}", mime=mime, key="npr_dl")
    except Exception as e: show_error(e)
    back_button('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows'])


# ---------- Mack RA ----------
def render_mack_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Mack Chain Ladder Risk Adjustment</h1><p>Stochastic Reserving & Value-at-Risk</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: client_name = st.text_input("Client Name", value="Client", key="mack_cn").strip()
    with c2: conf_level = st.slider("Confidence Level (%)", 50.0, 99.5, 75.0, 0.5, key="mack_cl") / 100.0

    uploaded = st.file_uploader("Upload Claims Data (Loss Date, Report Date, LOB, Amount)", type=["csv", "xlsx", "xls"], key="mack_f")
    if uploaded is None:
        st.info("Upload claims data file.")
        back_button('risk_adjustment', ['Home', 'LIC Calculators', 'Risk Adjustment']); return

    try:
        df = read_uploaded_file(uploaded)
        df.columns = df.columns.astype(str).str.strip()
        st.dataframe(df.head(3), use_container_width=True)
        cols = df.columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1: loss_col = st.selectbox("Loss Date", cols, key="mack_ld")
        with c2: rep_col = st.selectbox("Report Date", cols, key="mack_rd")
        with c3: lob_col = st.selectbox("LOB", cols, key="mack_lob")
        amt_col = st.selectbox("Amount Column", [c for c in cols if pd.api.types.is_numeric_dtype(df[c])], key="mack_amt")

        from_date = st.date_input("From Date", date(2020, 1, 1), key="mack_fd")
        to_date = st.date_input("To Date", date(2025, 12, 31), key="mack_td")
        grain = st.selectbox("Period Grain", ["Yearly", "Quarterly", "Monthly"], key="mack_gr")
        grain_code = {"Yearly": "Y", "Quarterly": "Q", "Monthly": "M"}[grain]
        ppy = periods_per_year(grain_code)

        from_dt = pd.Timestamp(str(from_date)); to_dt = pd.Timestamp(str(to_date))
        n_periods = int((to_dt.year - from_dt.year) * ppy) + 1

        if st.button("Calculate Mack Risk Adjustment", key="mack_run", use_container_width=True):
            if mack_engine is None or engine_utils is None: st.error("Mack engine not available."); return
            with st.spinner("Running Mack Chain Ladder calculation..."):
                inc, cum, mask = engine_utils.build_triangles(df, loss_col, rep_col, amt_col, from_dt, grain_code, n_periods)
                res = mack_engine.calculate_mack_chain_ladder(cum_triangle=cum, obs_mask=mask, confidence_level=conf_level)

            st.markdown("### Mack RA Results")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Central Estimate Reserve", f"{res.get('total_reserve', 0):,.2f}")
            with c2: st.metric("Mack Std Error", f"{res.get('total_se', 0):,.2f}")
            with c3: st.metric("Risk Adjustment", f"{res.get('total_ra', 0):,.2f}")

            if 'results_df' in res:
                st.dataframe(res['results_df'], use_container_width=True)

            output, ext, mime = build_download_payload({'Mack_RA': res.get('results_df', pd.DataFrame())})
            sc = sanitize_filename(client_name)
            st.download_button("Download Mack Results", data=output, file_name=f"{sc}_Mack_RA.{ext}", mime=mime, key="mack_dl")
    except Exception as e: show_error(e)
    back_button('risk_adjustment', ['Home', 'LIC Calculators', 'Risk Adjustment'])


# ---------- Bootstrap RA ----------
def render_bootstrap_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>ODP Bootstrap Risk Adjustment</h1><p>Over-dispersed Poisson Bootstrap Simulation</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: client_name = st.text_input("Client Name", value="Client", key="bs_cn").strip()
    with c2: n_sims = st.number_input("Number of Simulations", 100, 10000, 1000, 500, key="bs_sims")
    with c3: conf_level = st.slider("Confidence Level (%)", 50.0, 99.5, 75.0, 0.5, key="bs_cl") / 100.0

    uploaded = st.file_uploader("Upload Claims Data", type=["csv", "xlsx", "xls"], key="bs_f")
    if uploaded is None:
        st.info("Upload claims data file.")
        back_button('risk_adjustment', ['Home', 'LIC Calculators', 'Risk Adjustment']); return

    try:
        df = read_uploaded_file(uploaded)
        df.columns = df.columns.astype(str).str.strip()
        st.dataframe(df.head(3), use_container_width=True)
        cols = df.columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1: loss_col = st.selectbox("Loss Date", cols, key="bs_ld")
        with c2: rep_col = st.selectbox("Report Date", cols, key="bs_rd")
        with c3: amt_col = st.selectbox("Amount Column", [c for c in cols if pd.api.types.is_numeric_dtype(df[c])], key="bs_amt")

        from_date = st.date_input("From Date", date(2020, 1, 1), key="bs_fd")
        to_date = st.date_input("To Date", date(2025, 12, 31), key="bs_td")
        grain = st.selectbox("Period Grain", ["Yearly", "Quarterly", "Monthly"], key="bs_gr")
        grain_code = {"Yearly": "Y", "Quarterly": "Q", "Monthly": "M"}[grain]
        ppy = periods_per_year(grain_code)

        from_dt = pd.Timestamp(str(from_date)); to_dt = pd.Timestamp(str(to_date))
        n_periods = int((to_dt.year - from_dt.year) * ppy) + 1

        if st.button("Run Bootstrap Simulation", key="bs_run", use_container_width=True):
            if bootstrap_engine is None or engine_utils is None: st.error("Bootstrap engine not available."); return
            with st.spinner(f"Simulating {n_sims:,} iterations..."):
                inc, cum, mask = engine_utils.build_triangles(df, loss_col, rep_col, amt_col, from_dt, grain_code, n_periods)
                res = bootstrap_engine.bootstrap_chain_ladder(
                    working_cum=cum, obs_mask=mask, origin=from_dt, grain=grain_code,
                    n_periods=n_periods, n_iterations=n_sims, add_process_variance=True
                )

            st.markdown("### Bootstrap Simulation Results")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Mean Reserve", f"{res.get('mean_reserve', 0):,.2f}")
            with c2: st.metric("Std Dev", f"{res.get('std_reserve', 0):,.2f}")
            with c3: st.metric("Risk Adjustment", f"{res.get('ra_val', 0):,.2f}")

            output, ext, mime = build_download_payload({'Bootstrap_Summary': pd.DataFrame([res])})
            sc = sanitize_filename(client_name)
            st.download_button("Download Bootstrap Results", data=output, file_name=f"{sc}_Bootstrap_RA.{ext}", mime=mime, key="bs_dl")
    except Exception as e: show_error(e)
    back_button('risk_adjustment', ['Home', 'LIC Calculators', 'Risk Adjustment'])


# ---------- Full Valuation ----------
def render_full_valuation():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Full IFRS 17 Valuation</h1><p>Liability for Remaining Coverage & Incurred Claims Integration</p></div>', unsafe_allow_html=True)
    if full_engine is None:
        st.error("Full Valuation module is currently not available.")
        back_button('home', ['Home']); return

    client_name = st.text_input("Client Name", value="Client", key="fv_cn").strip()
    uploaded_file = st.file_uploader("Upload Comprehensive Valuation Model Data", type=["csv", "xlsx", "xls"], key="fv_f")
    if uploaded_file is not None:
        try:
            df = read_uploaded_file(uploaded_file)
            st.dataframe(df.head(5), use_container_width=True)
            if hasattr(full_engine, 'run_full_valuation'):
                if st.button("Run Full IFRS 17 Valuation", key="fv_run", use_container_width=True):
                    with st.spinner("Processing Full Valuation..."):
                        val_results = full_engine.run_full_valuation(df)
                    st.markdown("### Valuation Statement")
                    st.dataframe(val_results, use_container_width=True)
            else:
                st.info("Full Valuation engine loaded.")
        except Exception as e: show_error(e)
    back_button('home', ['Home'])


# =============================================================================
#  MAIN ROUTER
# =============================================================================
page_map = {
    'home': render_home,
    'lrc': render_lrc,
    'lic': render_lic,
    'fulfilment_cashflows': render_fulfilment_cashflows,
    'ibnr_menu': render_ibnr_menu,
    'risk_adjustment': render_risk_adjustment,
    'upr_calculator': render_upr_calculator,
    'loss_component': render_loss_component,
    'ocr_calculator': render_ocr_calculator,
    'percentage_calculator': render_percentage_calculator,
    'bcl_calculator': render_bcl_calculator,
    'capecod_calculator': render_capecod_calculator,
    'bf_calculator': render_bf_calculator,
    'ulae_calculator': render_ulae_calculator,
    'npr_calculator': render_npr_calculator,
    'mack_calculator': render_mack_calculator,
    'bootstrap_calculator': render_bootstrap_calculator,
    'full_valuation': render_full_valuation
}

current_page = st.session_state.get('page', 'home')
render_fn = page_map.get(current_page, render_home)
render_fn()
