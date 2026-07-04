# -*- coding: utf-8 -*-
# =============================================================================
#  NEXT VANTAGE — COMPREHENSIVE ACTUARIAL TOOLKIT
#  Main App with Multi-Page Navigation
#  Run:  streamlit run app.py
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
#  ROBUST PATH & IMPORT SYSTEM
# =============================================================================
import sys
import os
import importlib.util
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def import_file_glob(relative_pattern):
    search_pattern = os.path.join(BASE_DIR, relative_pattern.replace('/', os.sep))
    matched_files = glob.glob(search_pattern)
    if not matched_files:
        search_pattern = os.path.join(BASE_DIR, relative_pattern.replace('/', os.sep).replace(' ', '?'))
        matched_files = glob.glob(search_pattern)
    if not matched_files:
        return None
    abs_path = matched_files[0]
    module_name = os.path.splitext(os.path.basename(abs_path))[0]
    try:
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
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

# --- Load Full Valuation with multiple fallback methods ---
full_engine = None

# Method 1: Try glob pattern
for pattern in ["Full_Valuation/full_LRC_IFRS17.py", "Full_Valuation/*.py"]:
    full_engine = import_file_glob(pattern)
    if full_engine is not None:
        break

# Method 2: Try direct import if glob failed
if full_engine is None:
    fv_path = os.path.join(BASE_DIR, "Full_Valuation", "full_LRC_IFRS17.py")
    if os.path.exists(fv_path):
        try:
            spec = importlib.util.spec_from_file_location("full_LRC_IFRS17", fv_path)
            if spec:
                full_engine = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(full_engine)
        except Exception:
            pass

# =============================================================================
#  MODULE STATUS CHECK
# =============================================================================
module_status = {
    "UPR Engine": upr_engine,
    "Loss Component Engine": loss_comp_engine,
    "OCR Engine": ocr_engine,
    "Percentage IBNR": ibnr_pct,
    "BCL IBNR": ibnr_bcl,
    "Cape Cod IBNR": ibnr_cc,
    "BF IBNR": ibnr_bf,
    "ULAE Engine": ulae_engine,
    "NPR Engine": npr_engine,
    "Mack RA": mack_engine,
    "Bootstrap RA": bootstrap_engine,
    "Engine Utils": engine_utils,
    "Full Valuation": full_engine,
}

critical_modules = ["UPR Engine", "OCR Engine", "Engine Utils"]
missing_critical = [name for name in critical_modules if module_status[name] is None]

if missing_critical:
    st.error("Critical Error: Essential modules could not be loaded.")
    for mod in missing_critical:
        st.error(f"  - {mod}")
    st.stop()

missing_optional = [name for name, mod in module_status.items() 
                    if mod is None and name not in critical_modules]
if missing_optional:
    with st.sidebar.expander("Module Status", expanded=False):
        st.warning("Some optional modules could not be loaded:")
        for mod in missing_optional:
            st.write(f"  - {mod}")

# Debug info
with st.sidebar.expander("Debug Info", expanded=False):
    st.write(f"BASE_DIR: {BASE_DIR}")
    st.write(f"full_engine loaded: {full_engine is not None}")
    if full_engine is not None:
        st.write(f"Has calculate_full_ifrs17_lrc: {hasattr(full_engine, 'calculate_full_ifrs17_lrc')}")
    fv_dir = os.path.join(BASE_DIR, "Full_Valuation")
    if os.path.exists(fv_dir):
        st.write(f"Full_Valuation exists, files: {os.listdir(fv_dir)}")
    else:
        st.write("Full_Valuation directory NOT found")

# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def _date_filter(df, col, from_date, to_date):
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors='coerce')
    fd = pd.Timestamp(from_date)
    td = pd.Timestamp(to_date)
    return df[(df[col] >= fd) & (df[col] <= td)]

def periods_per_year(grain):
    return {"Y": 1, "Q": 4, "M": 12}[grain]

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', '', name).strip() or "Client"

def map_columns(df, required_fields, prefix):
    """Map uploaded file columns to required engine columns."""
    cols = df.columns.tolist()
    mapping = {}
    st.markdown(f"**Map columns:**")
    for field in required_fields:
        default_val = field if field in cols else (cols[0] if cols else "")
        default_idx = cols.index(default_val) if default_val in cols else 0
        mapping[field] = st.selectbox(f"{field}", cols, index=default_idx, key=f"{prefix}_{field}")
    return mapping

# =============================================================================
#  INFLATION & DISCOUNTING UI HELPERS
# =============================================================================

def load_inflation_data_ui(grain_code, ppy, page_key):
    st.markdown("**Inflation Adjustment**")
    inf_file = st.file_uploader(
        "Upload Inflation Curve (CSV/Excel: Period, Rate %)",
        type=["csv", "xlsx", "xls"],
        key=f"inf_{page_key}"
    )
    cum_inflation = None
    per_period_rates = None
    if inf_file:
        try:
            inf_df = pd.read_csv(inf_file) if inf_file.name.endswith('.csv') else pd.read_excel(inf_file)
            inf_df.columns = inf_df.columns.astype(str).str.strip()
            c1, c2 = st.columns(2)
            with c1:
                p_col = st.selectbox("Period Column", inf_df.columns, key=f"inf_p_{page_key}")
            with c2:
                r_col = st.selectbox("Rate Column (%)", inf_df.columns, key=f"inf_r_{page_key}")
            inf_df = inf_df[[p_col, r_col]].dropna()
            inf_df[r_col] = pd.to_numeric(inf_df[r_col], errors='coerce') / 100.0
            rates_inf = inf_df[r_col].values
            ratio = ppy / periods_per_year(grain_code)
            x_inf = np.arange(len(rates_inf)) * ratio
            x_tgt = np.arange(int(x_inf[-1]) + 1) if len(x_inf) > 0 else np.array([0])
            if len(x_inf) >= 4:
                f_interp = interpolate.CubicSpline(x_inf, rates_inf, extrapolate=True)
            else:
                f_interp = interpolate.interp1d(x_inf, rates_inf, kind='linear', fill_value='extrapolate')
            annual_rates_tgt = np.clip(f_interp(x_tgt), -0.5, 2.0)
            per_period_rates = (1 + annual_rates_tgt) ** (1 / ppy) - 1
            cum_inflation = np.cumprod(1 + per_period_rates)
            st.success("Inflation curve loaded and interpolated.")
        except Exception as e:
            st.error(f"Inflation data error: {e}")
    return cum_inflation, per_period_rates


def load_discounting_data_ui(grain_code, ppy, page_key):
    st.markdown("**Discounting**")
    disc_method = st.radio(
        "Discounting Method",
        ["None", "Single Flat Rate", "Yield Curve"],
        key=f"disc_m_{page_key}",
        horizontal=True
    )
    spot_rates = None
    flat_rate = None
    if disc_method == "Yield Curve":
        yc_file = st.file_uploader(
            "Upload Yield Curve (CSV/Excel: Duration_Years, Spot_Rate %)",
            type=["csv", "xlsx", "xls"],
            key=f"yc_{page_key}"
        )
        if yc_file:
            try:
                yc_df = pd.read_csv(yc_file) if yc_file.name.endswith('.csv') else pd.read_excel(yc_file)
                yc_df.columns = yc_df.columns.astype(str).str.strip()
                c1, c2 = st.columns(2)
                with c1:
                    m_col = st.selectbox("Duration Column", yc_df.columns, key=f"yc_m_{page_key}")
                with c2:
                    r_col = st.selectbox("Spot Rate Column (%)", yc_df.columns, key=f"yc_r_{page_key}")
                yc_df = yc_df[[m_col, r_col]].dropna()
                yc_df[m_col] = pd.to_numeric(yc_df[m_col], errors='coerce')
                yc_df[r_col] = pd.to_numeric(yc_df[r_col], errors='coerce') / 100.0
                maturities = yc_df[m_col].values
                rates = yc_df[r_col].values
                if len(maturities) >= 4:
                    f_interp = interpolate.CubicSpline(maturities, rates, extrapolate=True)
                else:
                    f_interp = interpolate.interp1d(maturities, rates, kind='linear', fill_value='extrapolate')
                period_maturities = np.arange(1, 61) / ppy
                spot_rates = np.clip(f_interp(period_maturities), 0, 1.0)
                st.success("Yield curve loaded and interpolated.")
            except Exception as e:
                st.error(f"Yield curve error: {e}")
    elif disc_method == "Single Flat Rate":
        flat_rate = st.number_input(
            "Annual Discount Rate (%)", 0.0, 50.0, 5.0, 0.5, key=f"flat_{page_key}"
        ) / 100.0
    return spot_rates, flat_rate


# =============================================================================
#  STREAMLIT CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Next Vantage Actuarial Toolkit",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #000000; font-family: 'Calisto MT', 'Georgia', serif; font-size: 11pt; }
    h1, h2, h3, h4, h5, h6, p, div, span, label { font-family: 'Calisto MT', 'Georgia', serif !important; }
    .hero { background: linear-gradient(135deg, #000000 0%, #1a1a2e 100%); color: #FFFFFF; padding: 2.5rem 2rem; text-align: center; border-bottom: 3px solid #4A90D9; margin-bottom: 2rem; }
    .hero h1 { color: #4A90D9; font-size: 2.5rem; margin-bottom: 0.5rem; }
    .hero p { font-size: 1.1rem; max-width: 800px; margin: 0 auto; }
    .card { background-color: #F9F9F9; border: 2px solid #4A90D9; border-radius: 10px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1.5rem; text-align: center; min-height: 150px; display: flex; flex-direction: column; justify-content: center; }
    .card h3 { color: #4A90D9; margin-top: 0; font-size: 1.1rem; }
    .card p { font-size: 0.9rem; color: #555; }
    .breadcrumb { background-color: #F0F0F0; padding: 0.5rem 1rem; border-radius: 5px; margin-bottom: 1rem; font-size: 0.85rem; border-left: 4px solid #4A90D9; }
    .breadcrumb span { color: #4A90D9; font-weight: bold; }
    .stButton > button { background-color: #4A90D9 !important; color: #FFFFFF !important; border: none !important; border-radius: 6px !important; font-weight: bold !important; padding: 0.6rem 1.2rem !important; width: 100% !important; font-family: 'Calisto MT', 'Georgia', serif !important; }
    .stButton > button:hover { background-color: #357ABD !important; color: #FFFFFF !important; }
    .stButton > button:disabled { background-color: #CCCCCC !important; color: #888888 !important; }
    .section-container { background-color: #F9F9F9; border: 2px solid #4A90D9; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; }
    .section-container h3 { color: #4A90D9; margin-top: 0; }
    .stFileUploader { border: 2px dashed #4A90D9 !important; border-radius: 10px !important; padding: 1rem !important; }
    .footer { background-color: #000000; color: #FFFFFF; text-align: center; padding: 1.5rem; border-top: 3px solid #4A90D9; margin-top: 3rem; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  SESSION STATE & NAVIGATION
# =============================================================================

if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'breadcrumb' not in st.session_state:
    st.session_state.breadcrumb = ['Home']

def navigate_to(page, breadcrumb_label=None):
    st.session_state.page = page
    if breadcrumb_label:
        st.session_state.breadcrumb = breadcrumb_label
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
    st.markdown('<div class="hero"><h1>Next Vantage</h1><p>Comprehensive Actuarial Reserving Toolkit - IFRS 17 Compliant<br>African Actuarial Consultants</p></div>', unsafe_allow_html=True)
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
    st.markdown('<div class="footer">2025 Next Vantage - African Actuarial Consultants. All rights reserved.</div>', unsafe_allow_html=True)


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
    items = [
        ("OCR", "ocr_calculator", ocr_engine, "Outstanding Claims Reserve"),
        ("IBNR", "ibnr_menu", True, "4 Methods: Percentage, BCL, Cape Cod, BF"),
        ("ULAE", "ulae_calculator", ulae_engine, "Unallocated Loss Adjustment Expenses"),
        ("NPR", "npr_calculator", npr_engine, "Reinsurance Non-Performance Risk"),
    ]
    for i, (title, page, module, desc) in enumerate(items):
        with cols[i]:
            available = module is not None
            status_text = "Available" if available else "Unavailable"
            st.markdown(f'<div class="card"><h3>{title}</h3><p>{desc}</p><p>{status_text}</p></div>', unsafe_allow_html=True)
            if st.button(f"Open {title}", key=f"nav_fcf_{page}", disabled=not available, use_container_width=True):
                navigate_to(page, ['Home', 'LIC Calculators', 'Fulfilment Cashflows', title])
    back_button('lic', ['Home', 'LIC Calculators'])


def render_ibnr_menu():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>IBNR Methods</h1><p>Select a calculation method</p></div>', unsafe_allow_html=True)
    methods = [
        ("Percentage Method", "percentage_calculator", ibnr_pct, "Simple: IBNR = Amount x IBNR%"),
        ("Basic Chain Ladder", "bcl_calculator", ibnr_bcl, "Multi-LDF with inflation & discounting"),
        ("Cape Cod", "capecod_calculator", ibnr_cc, "Uses premiums to derive expected loss ratio"),
        ("Bornhuetter-Ferguson", "bf_calculator", ibnr_bf, "Blends Chain Ladder with expected loss ratio"),
    ]
    for i in range(0, len(methods), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(methods):
                name, page, module, desc = methods[i + j]
                with cols[j]:
                    available = module is not None
                    status_text = "Available" if available else "Unavailable"
                    st.markdown(f'<div class="card"><h3>{name}</h3><p>{desc}</p><p>{status_text}</p></div>', unsafe_allow_html=True)
                    if st.button(f"Open {name}", key=f"nav_ibnr_{page}", disabled=not available, use_container_width=True):
                        navigate_to(page, ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods', name])
    back_button('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows'])


def render_risk_adjustment():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Risk Adjustment</h1><p>RA Methods for IFRS 17</p></div>', unsafe_allow_html=True)
    cols = st.columns(2)
    methods = [
        ("Mack Chain Ladder", "mack_calculator", mack_engine, "Distribution-free standard error of IBNR (Mack 1993)"),
        ("ODP Bootstrap", "bootstrap_calculator", bootstrap_engine, "England & Verrall bootstrap with process variance"),
    ]
    for i, (name, page, module, desc) in enumerate(methods):
        with cols[i]:
            available = module is not None
            status_text = "Available" if available else "Unavailable"
            st.markdown(f'<div class="card"><h3>{name}</h3><p>{desc}</p><p>{status_text}</p></div>', unsafe_allow_html=True)
            if st.button(f"Open {name}", key=f"nav_ra_{page}", disabled=not available, use_container_width=True):
                navigate_to(page, ['Home', 'LIC Calculators', 'Risk Adjustment', name])
    back_button('lic', ['Home', 'LIC Calculators'])


# =============================================================================
#  CALCULATOR: UPR
# =============================================================================

def render_upr_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>UPR Calculator</h1><p>Unearned Premium Reserve - Pro-rata Methods</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        valuation_date = st.date_input("Valuation Date", value=date(2025, 12, 31), key="upr_vd")
    with c2:
        client_name = st.text_input("Client Name", value="Client", key="upr_cn").strip()
    with c3:
        method = st.selectbox("UPR Method", ["365th", "24th", "8th"], key="upr_mt")
    valuation_date_ts = pd.Timestamp(str(valuation_date))
    uploaded_file = st.file_uploader("Upload Premium Register (CSV or Excel)", type=["csv", "xlsx", "xls"], key="upr_f")
    if uploaded_file is not None:
        try:
            original_filename = uploaded_file.name
            base_filename = re.sub(r'\.[^.]*$', '', original_filename)
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            unnamed = [c for c in df.columns if str(c).startswith('Unnamed:')]
            if unnamed:
                df = df.drop(columns=unnamed)
            df.columns = df.columns.astype(str).str.strip()
            st.dataframe(df.head(5), use_container_width=True)
            all_columns = df.columns.tolist()
            c1, c2 = st.columns(2)
            with c1:
                start_date_col = st.selectbox("Start Date Column", [""] + all_columns, key="upr_sd")
            with c2:
                end_date_col = st.selectbox("End Date Column", [""] + all_columns, key="upr_ed")
            if not start_date_col or not end_date_col:
                st.info("Please select Start and End Date columns.")
                return
            grouping_options = [c for c in all_columns if c not in [start_date_col, end_date_col]]
            grouping_cols = st.multiselect("Group By Columns", options=grouping_options, default=[grouping_options[0]] if grouping_options else [], key="upr_gc")
            if not grouping_cols:
                st.info("Please select at least one Group By column.")
                return
            numeric_columns = [c for c in all_columns if c not in [start_date_col, end_date_col] + grouping_cols and pd.api.types.is_numeric_dtype(df[c])]
            selected_value_cols = st.multiselect("Amount Columns", options=numeric_columns, default=numeric_columns[:min(4, len(numeric_columns))], key="upr_vc")
            if not selected_value_cols:
                st.info("Please select at least one Amount column.")
                return
            df_processed = df.rename(columns={start_date_col: 'Start_Date', end_date_col: 'End_Date'}).copy()
            df_processed['Start_Date'] = pd.to_datetime(df_processed['Start_Date'], errors='coerce')
            df_processed['End_Date'] = pd.to_datetime(df_processed['End_Date'], errors='coerce')
            df_processed = df_processed.dropna(subset=['Start_Date', 'End_Date'])
            df_processed = df_processed[df_processed['End_Date'] > df_processed['Start_Date']]
            for c in selected_value_cols:
                df_processed[c] = pd.to_numeric(df_processed[c], errors='coerce').fillna(0)
            df_processed["Duration_Days"] = (df_processed["End_Date"] - df_processed["Start_Date"]).dt.days + 1
            df_processed = df_processed[df_processed["Duration_Days"] > 0]
            if df_processed.empty:
                st.error("No valid policies after data validation.")
                return
            st.success(f"{len(df_processed):,} valid policies loaded")
            if st.button("Calculate UPR", key="upr_calc", use_container_width=True):
                with st.spinner("Calculating UPR..."):
                    df_processed['Remaining_Days'] = (df_processed["End_Date"] - valuation_date_ts).dt.days + 1
                    df_processed['Remaining_Days'] = np.clip(df_processed['Remaining_Days'], 0, df_processed['Duration_Days'])
                    if method == "365th":
                        df_processed['Unearned_Portion'] = df_processed['Remaining_Days'] / df_processed['Duration_Days']
                    elif method == "24th":
                        interval = 365.25 / 24
                        df_processed['Unearned_Portion'] = (df_processed['Remaining_Days'] / interval) / (df_processed['Duration_Days'] / interval)
                    else:
                        interval = 365.25 / 8
                        df_processed['Unearned_Portion'] = (df_processed['Remaining_Days'] / interval) / (df_processed['Duration_Days'] / interval)
                    for c in selected_value_cols:
                        df_processed[f"{c}_UPR"] = df_processed['Unearned_Portion'] * df_processed[c]
                    upr_columns = [f"{c}_UPR" for c in selected_value_cols]
                    result = df_processed.groupby(grouping_cols)[upr_columns].sum().reset_index()
                    result.columns = grouping_cols + selected_value_cols
                st.markdown("### UPR Results")
                disp = result.copy()
                for c in selected_value_cols:
                    disp[c] = disp[c].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
                st.dataframe(disp, use_container_width=True, hide_index=True)
                total_upr = sum(result[c].sum() for c in selected_value_cols)
                st.metric("Total UPR", f"{total_upr:,.2f}")
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as w:
                    result.to_excel(w, index=False, sheet_name='UPR_Results')
                output.seek(0)
                sc = sanitize_filename(client_name)
                so = sanitize_filename(base_filename)
                st.download_button("Download UPR Results", data=output, file_name=f"{sc}_{so}_UPR_{method}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="upr_dl")
        except Exception as e:
            st.error(f"Error: {e}")
    back_button('lrc', ['Home', 'LRC Calculators'])


# =============================================================================
#  CALCULATOR: LOSS COMPONENT
# =============================================================================

def render_loss_component():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Loss Component Calculator</h1><p>Onerous Contract Identification - IFRS 17 PAA</p></div>', unsafe_allow_html=True)
    client_name = st.text_input("Client Name", value="Client", key="lc_cn").strip()
    uploaded_file = st.file_uploader("Upload Data File (CSV or Excel)", type=["csv", "xlsx", "xls"], key="lc_f")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
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
            with c1:
                expenses_col = st.selectbox("Expenses", cols, key="lc_exp")
            with c2:
                opening_upr_col = st.selectbox("Opening UPR", cols, key="lc_oupr")
            closing_upr_col = st.selectbox("Closing UPR", cols, key="lc_cupr")
            if st.button("Calculate Loss Component", key="lc_run", use_container_width=True):
                if loss_comp_engine is None:
                    st.error("Loss Component engine not available.")
                    return
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
                with pd.ExcelWriter(output, engine='openpyxl') as w:
                    result.to_excel(w, index=False, sheet_name='Loss_Component')
                output.seek(0)
                sc = sanitize_filename(client_name)
                st.download_button("Download Results", data=output, file_name=f"{sc}_Loss_Component.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="lc_dl")
        except Exception as e:
            st.error(f"Error: {e}")
    back_button('lrc', ['Home', 'LRC Calculators'])


# =============================================================================
#  CALCULATOR: OCR
# =============================================================================

def render_ocr_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>OCR Calculator</h1><p>Outstanding Claims Reserve - Group & Sum</p></div>', unsafe_allow_html=True)
    client_name = st.text_input("Client Name", value="Client", key="ocr_cn").strip()
    uploaded_file = st.file_uploader("Upload Case Estimates File (CSV or Excel)", type=["csv", "xlsx", "xls"], key="ocr_f")
    if uploaded_file is not None:
        try:
            original_filename = uploaded_file.name
            base_filename = re.sub(r'\.[^.]*$', '', original_filename)
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            unnamed = [c for c in df.columns if str(c).startswith('Unnamed:')]
            if unnamed:
                df = df.drop(columns=unnamed)
            df.columns = df.columns.astype(str).str.strip()
            st.dataframe(df.head(5), use_container_width=True)
            all_columns = df.columns.tolist()
            grouping_cols = st.multiselect("Group By Columns", options=all_columns, default=[all_columns[0]] if all_columns else [], key="ocr_gc")
            if not grouping_cols:
                st.info("Please select at least one Group By column.")
                return
            numeric_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c not in grouping_cols]
            selected_value_cols = st.multiselect("Amount Columns", options=numeric_columns, default=numeric_columns[:min(5, len(numeric_columns))], key="ocr_vc")
            if not selected_value_cols:
                st.info("Please select at least one Amount column.")
                return
            df_processed = df[grouping_cols + selected_value_cols].copy()
            for c in selected_value_cols:
                df_processed[c] = pd.to_numeric(df_processed[c], errors='coerce').fillna(0)
            grouped = df_processed.groupby(grouping_cols)[selected_value_cols].sum().reset_index()
            st.markdown("### OCR Summary")
            disp = grouped.copy()
            for c in selected_value_cols:
                disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            total_ocr = sum(grouped[c].sum() for c in selected_value_cols)
            st.metric("Total OCR", f"{total_ocr:,.2f}")
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w:
                grouped.to_excel(w, index=False, sheet_name='OCR_Results')
            output.seek(0)
            sc = sanitize_filename(client_name)
            so = sanitize_filename(base_filename)
            st.download_button("Download OCR Results", data=output, file_name=f"{sc}_{so}_OCR.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="ocr_dl")
        except Exception as e:
            st.error(f"Error: {e}")
    back_button('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows'])


# =============================================================================
#  CALCULATOR: PERCENTAGE IBNR
# =============================================================================

def render_percentage_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Percentage IBNR Calculator</h1><p>Simple Method: IBNR = Amount x IBNR Percentage</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        client_name = st.text_input("Client Name", value="Client", key="pct_cn").strip()
    with c2:
        ibnr_pct_val = st.number_input("IBNR Percentage (%)", 0.0, 100.0, 10.0, 0.5, key="pct_pct") / 100.0
    c1, c2 = st.columns(2)
    with c1:
        from_date = st.date_input("From Date", date(2020, 1, 1), key="pct_fd")
    with c2:
        to_date = st.date_input("To Date", date(2025, 12, 31), key="pct_td")
    uploaded = st.file_uploader("Upload Data File (CSV/Excel)", type=["csv", "xlsx", "xls"], key="pct_f")
    if uploaded is None:
        st.info("Upload a file with Date, Line of Business, and Amount columns.")
        back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])
        return
    try:
        df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        df.columns = df.columns.astype(str).str.strip()
        st.dataframe(df.head(5), use_container_width=True)
        cols = df.columns.tolist()
        c1, c2 = st.columns(2)
        with c1:
            date_col = st.selectbox("Date Column", cols, key="pct_date")
        with c2:
            lob_col = st.selectbox("Line of Business Column", cols, key="pct_lob")
        amount_candidates = [c for c in cols if c not in [date_col, lob_col] and pd.api.types.is_numeric_dtype(df[c])]
        amount_cols = st.multiselect("Amount Column(s)", amount_candidates, key="pct_amt")
        if not amount_cols:
            st.warning("Please select at least one Amount column.")
            return
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        from_dt = pd.Timestamp(str(from_date))
        to_dt = pd.Timestamp(str(to_date))
        df = _date_filter(df, date_col, from_date, to_date)
        if df.empty:
            st.warning("No data in selected date range.")
            return
        if st.button("Calculate Percentage IBNR", key="pct_run", use_container_width=True):
            if ibnr_pct is None:
                st.error("Percentage IBNR engine not available.")
                return
            with st.spinner("Calculating..."):
                summary_df, grand_total = ibnr_pct.calculate_percentage_ibnr(
                    df=df, date_col=date_col, lob_col=lob_col,
                    amount_cols=amount_cols, from_date=from_dt,
                    to_date=to_dt, ibnr_pct=ibnr_pct_val
                )
            st.markdown("### Percentage IBNR Results")
            disp = summary_df.copy()
            for c in disp.columns:
                if c != lob_col:
                    disp[c] = disp[c].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.metric("Total IBNR", f"{grand_total:,.2f}")
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w:
                summary_df.to_excel(w, index=False, sheet_name='Percentage_IBNR')
            output.seek(0)
            sc = sanitize_filename(client_name)
            st.download_button("Download Results", data=output, file_name=f"{sc}_Percentage_IBNR.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="pct_dl")
    except Exception as e:
        st.error(f"Error: {e}")
    back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])


# =============================================================================
#  CALCULATOR: BCL IBNR
# =============================================================================

def render_bcl_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Basic Chain Ladder (BCL) - IBNR</h1><p>Multi-LDF Methods with Inflation & Discounting Support</p></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        client_name = st.text_input("Client Name", value="Client", key="bcl_cn").strip()
    with c2:
        grain = st.selectbox("Period Grain", ["Yearly", "Quarterly", "Monthly"], key="bcl_gr")
    grain_map = {"Yearly": "Y", "Quarterly": "Q", "Monthly": "M"}
    grain_code = grain_map[grain]
    ppy = {"Y": 1, "Q": 4, "M": 12}[grain_code]
    with c3:
        from_date = st.date_input("From Date", date(2020, 1, 1), key="bcl_fd")
    with c4:
        to_date = st.date_input("To Date", date(2025, 12, 31), key="bcl_td")
    uploaded = st.file_uploader("Upload Claims Data (Loss Date, Report Date, LOB, Amount)", type=["csv", "xlsx", "xls"], key="bcl_f")
    if uploaded is None:
        st.info("Upload claims data file.")
        back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])
        return
    try:
        df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        df.columns = df.columns.astype(str).str.strip()
        st.dataframe(df.head(5), use_container_width=True)
        cols = df.columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1:
            loss_col = st.selectbox("Loss Date Column", cols, key="bcl_ld")
        with c2:
            rep_col = st.selectbox("Report Date Column", cols, key="bcl_rd")
        with c3:
            lob_col = st.selectbox("LOB Column", cols, key="bcl_lob")
        amount_candidates = [c for c in cols if c not in [loss_col, rep_col, lob_col] and pd.api.types.is_numeric_dtype(df[c])]
        amount_cols = st.multiselect("Amount Column(s)", amount_candidates, key="bcl_amt")
        if not amount_cols:
            st.warning("Please select at least one Amount column.")
            return
        df[loss_col] = pd.to_datetime(df[loss_col], errors='coerce')
        df[rep_col] = pd.to_datetime(df[rep_col], errors='coerce')
        for ac in amount_cols:
            df[ac] = pd.to_numeric(df[ac], errors='coerce').fillna(0)
        df = df.dropna(subset=[loss_col, rep_col])
        from_dt = pd.Timestamp(str(from_date))
        to_dt = pd.Timestamp(str(to_date))
        df = _date_filter(df, loss_col, from_date, to_date)
        n_periods = int((to_dt.year - from_dt.year) * ppy) + 1
        st.markdown("### LDF Method Selection")
        selected_method = "volume_weighted"
        if engine_utils is not None and ibnr_bcl is not None:
            sample_amt = amount_cols[0]
            _, sample_cum, _ = engine_utils.build_triangles(df, loss_col, rep_col, sample_amt, from_dt, grain_code, n_periods)
            all_ldfs = ibnr_bcl.calculate_all_ldfs(sample_cum, n_periods)
            ldf_df = pd.DataFrame({
                "Dev Period": range(1, len(all_ldfs["volume_weighted"]) + 1),
                "Vol-Weighted": all_ldfs["volume_weighted"],
                "Simple Avg": all_ldfs["simple_average"],
                "Geometric": all_ldfs["geometric"],
                "Medial": all_ldfs["medial"],
                "Lin Regression": all_ldfs["linear_regression"],
                "Wtd Last 3": all_ldfs["weighted_last_3"]
            })
            st.dataframe(ldf_df.round(4), use_container_width=True)
            rec_method = "volume_weighted"
            min_cv = float('inf')
            for method in ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"]:
                factors = all_ldfs[method]
                if len(factors) >= 3:
                    cv = np.std(factors[:3]) / np.mean(factors[:3]) if np.mean(factors[:3]) > 0 else float('inf')
                    if cv < min_cv:
                        min_cv = cv
                        rec_method = method
            st.info(f"Recommended: {rec_method.replace('_', ' ').title()} (lowest CV: {min_cv:.2%})")
            selected_method = st.selectbox(
                "Select LDF Method to Use",
                ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"],
                index=["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"].index(rec_method),
                key="bcl_ldf_method"
            )
        st.markdown("### Adjustments")
        c1, c2 = st.columns(2)
        with c1:
            use_inflation = st.checkbox("Apply Inflation Adjustment", key="bcl_inf")
        with c2:
            use_discounting = st.checkbox("Apply Discounting", key="bcl_disc")
        cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
        if use_inflation:
            cum_inflation, per_period_rates = load_inflation_data_ui(grain_code, ppy, "bcl")
        if use_discounting:
            spot_rates, flat_rate = load_discounting_data_ui(grain_code, ppy, "bcl")
        if st.button("Calculate BCL IBNR", key="bcl_run", use_container_width=True):
            if ibnr_bcl is None or engine_utils is None:
                st.error("Required engines not available.")
                return
            with st.spinner("Calculating BCL IBNR..."):
                lobs = sorted(df[lob_col].dropna().unique())
                all_results = []
                for lob in lobs:
                    lob_data = df[df[lob_col] == lob].copy()
                    for ac in amount_cols:
                        _, cum, _ = engine_utils.build_triangles(lob_data, loss_col, rep_col, ac, from_dt, grain_code, n_periods)
                        result = ibnr_bcl.calculate_bcl_ibnr(
                            cum_triangle=cum, start_date=from_dt, period_unit=grain_code,
                            selected_ldf_method=selected_method,
                            use_inflation=use_inflation, cum_inflation=cum_inflation,
                            per_period_rates=per_period_rates,
                            use_discounting=use_discounting, spot_rates=spot_rates, flat_rate=flat_rate
                        )
                        res_df = result['results_df']
                        res_df['LOB'] = lob
                        res_df['Amount_Col'] = ac
                        all_results.append(res_df)
                final_df = pd.concat(all_results, ignore_index=True)
                summary = final_df.groupby(['LOB', 'Amount_Col'])[['Current_Claims', 'IBNR']].sum().reset_index()
            st.markdown("### BCL IBNR Summary")
            disp = summary.copy()
            for c in ['Current_Claims', 'IBNR']:
                if c in disp.columns:
                    disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            total_ibnr = summary['IBNR'].sum() if 'IBNR' in summary.columns else 0
            st.metric("Total BCL IBNR", f"{total_ibnr:,.2f}")
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w:
                summary.to_excel(w, index=False, sheet_name='BCL_Summary')
                final_df.to_excel(w, index=False, sheet_name='BCL_Detail')
            output.seek(0)
            sc = sanitize_filename(client_name)
            st.download_button("Download BCL Results", data=output, file_name=f"{sc}_BCL_IBNR.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="bcl_dl")
    except Exception as e:
        st.error(f"Error: {e}")
    back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])


# =============================================================================
#  REMAINING CALCULATORS (Cape Cod, BF, ULAE, NPR, Mack, Bootstrap, Full Valuation)
#  are identical to the last complete version I provided.
#  Due to message length limits, please use the versions from my previous message.
#  The key fix for Full Valuation is the import section at the top of this file.
# =============================================================================


# =============================================================================
#  MAIN ROUTER
# =============================================================================

def main():
    page = st.session_state.page
    page_routes = {
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
        'full_valuation': render_full_valuation,  # This needs the function from previous message
    }
    if page in page_routes:
        page_routes[page]()
    else:
        st.error(f"Unknown page: {page}")
        navigate_to('home', ['Home'])


if __name__ == "__main__":
    main()
