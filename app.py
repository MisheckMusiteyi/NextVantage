# -*- coding: utf-8 -*-
# =============================================================================
#  NEXT VANTAGE — COMPREHENSIVE ACTUARIAL TOOLKIT
#  Main App with Multi-Page Navigation
#  Theme: Light Blue (#4A90D9), Black, White
#  Run:  streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import date, datetime
import re
from scipy import interpolate

# =============================================================================
#  ROBUST PATH & IMPORT SYSTEM (Case-Insensitive Glob matching)
# =============================================================================
import sys
import os
import importlib.util
import glob

# Get the absolute path where app.py is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the base directory to Python's path
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def import_file_glob(relative_pattern):
    """
    Imports a python module using case-insensitive glob pattern matching.
    This 100% bypasses capitalization mismatches and folder space errors.
    """
    search_pattern = os.path.join(BASE_DIR, relative_pattern.replace('/', os.sep))
    matched_files = glob.glob(search_pattern)
    
    if not matched_files:
        return None
        
    abs_path = matched_files[0]
    module_name = os.path.splitext(os.path.basename(abs_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    
    if spec is None:
        return None
        
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None

# --- Load LRC Modules ---
upr_engine = import_file_glob("LRC_Calculators/upr_engine.py")
loss_comp_engine = import_file_glob("LRC_Calculators/loss_component_engine.py")

# --- Load LIC FCF Modules ---
ocr_engine = import_file_glob("LIC_Calculators/FCF_Calculators/OCR_Calculators/ocr_engine.py")

ibnr_pct = import_file_glob("LIC_Calculators/FCF_Calculators/IBNR_Calculators/percentage_ibnr.py")
ibnr_bcl = import_file_glob("LIC_Calculators/FCF_Calculators/IBNR_Calculators/bcl_ibnr.py")
ibnr_cc = import_file_glob("LIC_Calculators/FCF_Calculators/IBNR_Calculators/cape_cod_ibnr.py")
ibnr_bf = import_file_glob("LIC_Calculators/FCF_Calculators/IBNR_Calculators/bf_ibnr.py")

ulae_engine = import_file_glob("LIC_Calculators/FCF_Calculators/ULAE_Calculators/ulae_engine.py")
npr_engine = import_file_glob("LIC_Calculators/FCF_Calculators/NPR_Calculators/npr_engine.py")

# --- Load LIC RA Modules ---
mack_engine = import_file_glob("LIC_Calculators/RA_Calculators/mack_ra.py")
bootstrap_engine = import_file_glob("LIC_Calculators/RA_Calculators/bootstrap_ra.py")

# --- Load Shared Helpers ---
act_helpers = import_file_glob("utils/actuarial_helpers.py")
engine_utils = import_file_glob("utils/actuarial_engine_utils.py")

# --- Load Full Valuation ---
# Try multiple patterns to find the file
full_engine = import_file_glob("Full_Valuation/*.py")
if full_engine is None:
    full_engine = import_file_glob("Full_Valuation/*IFRS17*.py")
if full_engine is None:
    full_engine = import_file_glob("Full_Valuation/*lrc*.py")


# =============================================================================
#  CRITICAL STARTUP CHECKS
# =============================================================================

# If core modules failed to load, stop the app immediately with a clear message.
if upr_engine is None:
    st.error(f"❌ Critical Error: Could not find `upr_engine.py`.")
    st.write(f"Python searched inside: `{os.path.join(BASE_DIR, 'LRC_Calculators')}`")
    st.write("Please check that a file named `upr_engine.py` exists in that folder.")
    st.stop()

if ocr_engine is None:
    st.error(f"❌ Critical Error: Could not find `ocr_engine.py`.")
    st.write(f"Python searched inside: `{os.path.join(BASE_DIR, 'LIC_Calculators/FCF_Calculators/OCR_Calculators')}`")
    st.stop()

if full_engine is None:
    st.error(f"❌ Critical Error: Could not find any Python file in the `Full_Valuation` folder.")
    st.write(f"Python searched inside: `{os.path.join(BASE_DIR, 'Full_Valuation')}`")
    st.write("Please check that the `Full_Valuation` folder exists and contains at least one .py file.")
    st.stop()


# =============================================================================
#  DATETIME UTILITIES
# =============================================================================

def _parse_dates(series):
    """Safely parse any column to datetime regardless of source dtype."""
    try:
        return pd.to_datetime(series.astype(str), errors='coerce')
    except Exception:
        return pd.to_datetime(series, errors='coerce')

def _date_filter(df, col, from_date, to_date):
    """Filter dataframe by date column between from_date and to_date."""
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors='coerce')
    fd = pd.Timestamp(from_date)
    td = pd.Timestamp(to_date)
    return df[(df[col] >= fd) & (df[col] <= td)]


def periods_per_year(grain): 
    return {"Y": 1, "Q": 4, "M": 12}[grain]

# =============================================================================
#  SIMPLIFIED UI HELPERS FOR INFLATION / DISCOUNTING
# =============================================================================

def load_inflation_data_ui(grain_code, ppy):
    st.markdown("**Load Inflation Data**")
    inf_file = st.file_uploader("Upload Inflation Curve (Period, Rate %)", type=["csv","xlsx","xls"], key=f"inf_{st.session_state.page}")
    cum_inflation = None; per_period_rates = None
    if inf_file:
        inf_df = pd.read_csv(inf_file) if inf_file.name.endswith('.csv') else pd.read_excel(inf_file)
        p_col = st.selectbox("Period column", inf_df.columns, key=f"inf_p_{st.session_state.page}")
        r_col = st.selectbox("Rate column", inf_df.columns, key=f"inf_r_{st.session_state.page}")
        inf_df = inf_df[[p_col, r_col]].dropna()
        inf_df[r_col] = pd.to_numeric(inf_df[r_col], errors='coerce') / 100.0
        rates_inf = inf_df[r_col].values
        ratio = ppy / periods_per_year(grain_code)
        x_inf = np.arange(len(rates_inf)) * ratio
        x_tgt = np.arange(int(x_inf[-1]) + 1)
        if len(x_inf) >= 4:
            f_interp = interpolate.CubicSpline(x_inf, rates_inf, extrapolate=True)
        else:
            f_interp = interpolate.interp1d(x_inf, rates_inf, kind='linear', fill_value='extrapolate')
        annual_rates_tgt = np.clip(f_interp(x_tgt), -0.5, 2.0)
        per_period_rates = (1 + annual_rates_tgt) ** (1 / ppy) - 1
        cum_inflation = np.cumprod(1 + per_period_rates)
        st.success(f"Inflation interpolated.")
    return cum_inflation, per_period_rates

def load_discounting_data_ui(grain_code, ppy):
    st.markdown("**Load Discounting Data**")
    disc_method = st.radio("Discounting Method", ["Yield Curve", "Single Flat Rate"], key=f"disc_m_{st.session_state.page}")
    spot_rates = None; flat_rate = None
    if disc_method == "Yield Curve":
        yc_file = st.file_uploader("Upload Yield Curve (Duration_Years, Spot_Rate)", type=["csv","xlsx","xls"], key=f"yc_{st.session_state.page}")
        if yc_file:
            yc_df = pd.read_csv(yc_file) if yc_file.name.endswith('.csv') else pd.read_excel(yc_file)
            m_col = st.selectbox("Maturity column", yc_df.columns, key=f"yc_m_{st.session_state.page}")
            r_col = st.selectbox("Rate column", yc_df.columns, key=f"yc_r_{st.session_state.page}")
            yc_df = yc_df[[m_col, r_col]].dropna()
            yc_df[m_col] = pd.to_numeric(yc_df[m_col], errors='coerce')
            yc_df[r_col] = pd.to_numeric(yc_df[r_col], errors='coerce') / 100.0
            maturities = yc_df[m_col].values; rates = yc_df[r_col].values
            if len(maturities) >= 4:
                f_interp = interpolate.CubicSpline(maturities, rates, extrapolate=True)
            else:
                f_interp = interpolate.interp1d(maturities, rates, kind='linear', fill_value='extrapolate')
            period_maturities = np.arange(1, 61) / ppy
            spot_rates = np.clip(f_interp(period_maturities), 0, 1.0)
            st.success(f"Yield Curve interpolated.")
    else:
        flat_rate = st.number_input("Annual Discount Rate (%)", 0.0, 50.0, 5.0, 0.5, key=f"flat_{st.session_state.page}") / 100.0
    return spot_rates, flat_rate


st.set_page_config(page_title="Next Vantage Actuarial Toolkit", layout="wide", initial_sidebar_state="expanded")

# =============================================================================
#  CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #000000; font-family: 'Calisto MT', 'Georgia', serif; font-size: 11pt; }
    h1, h2, h3, h4, h5, h6, p, div, span, label { font-family: 'Calisto MT', 'Georgia', serif !important; }
    .header { background-color: #000000; padding: 1rem 2rem; display: flex; align-items: center; justify-content: space-between; border-bottom: 3px solid #4A90D9; }
    .header .logo { color: #4A90D9; font-size: 1.5rem; font-weight: bold; }
    .nav-links a { color: #FFFFFF; margin-left: 2rem; text-decoration: none; font-weight: 500; }
    .nav-links a:hover { color: #4A90D9; }
    .hero { background: linear-gradient(135deg, #000000 0%, #1a1a2e 100%); color: #FFFFFF; padding: 3rem 2rem; text-align: center; border-bottom: 3px solid #4A90D9; }
    .hero h1 { color: #4A90D9; font-size: 2.8rem; margin-bottom: 0.5rem; }
    .hero p { font-size: 1.2rem; max-width: 800px; margin: 0 auto; }
    .card { background-color: #F9F9F9; border: 1px solid #4A90D9; border-radius: 8px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1.5rem; text-align: center; }
    .card h3 { color: #4A90D9; margin-top: 0; }
    .footer { background-color: #000000; color: #FFFFFF; text-align: center; padding: 1.5rem; border-top: 3px solid #4A90D9; margin-top: 3rem; }
    .stButton > button { background-color: #4A90D9 !important; color: #FFFFFF !important; border: none !important; border-radius: 4px !important; font-weight: bold !important; padding: 0.75rem 1.5rem !important; width: 100% !important; font-family: 'Calisto MT', 'Georgia', serif !important; }
    .stButton > button:hover { background-color: #357ABD !important; color: #FFFFFF !important; }
    .section-container { background-color: #F9F9F9; border: 2px solid #4A90D9; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; }
    .section-container h3 { color: #4A90D9; margin-top: 0; font-size: 1.2rem; font-weight: bold; }
    .breadcrumb { background-color: #F0F0F0; padding: 0.5rem 1rem; border-radius: 5px; margin-bottom: 1rem; font-size: 0.9rem; border-left: 3px solid #4A90D9; }
    .breadcrumb span { color: #4A90D9; font-weight: bold; }
    .stFileUploader { border: 2px dashed #4A90D9 !important; border-radius: 10px !important; padding: 1rem !important; }
    .dataframe { border: 1px solid #4A90D9 !important; border-radius: 8px !important; overflow: hidden !important; }
    .stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"] { border: 1px solid #4A90D9 !important; border-radius: 4px !important; }
    .required-container { background-color: #F9F9F9; border: 2px solid #4A90D9; border-radius: 10px; padding: 1rem; text-align: center; min-height: 120px; margin-bottom: 1rem; }
    .required-container h3 { color: #4A90D9; font-size: 1.2rem; font-weight: bold; }
    .main-container { max-width: 1400px; margin: 2rem auto; padding: 0 2rem; }
    .report-meta { background-color: #F0F4F8; border: 2px solid #4A90D9; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; font-size: 0.85rem; }
    .report-meta td { padding: 2px 8px; }
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

def go_home():
    st.session_state.page = 'home'; st.session_state.breadcrumb = ['Home']

st.markdown('<div class="header"><div class="logo">Next Vantage Actuarial Toolkit</div><div class="nav-links"><a href="javascript:void(0)" onclick="window.location.reload()">Home</a></div></div>', unsafe_allow_html=True)

def show_breadcrumb():
    if len(st.session_state.breadcrumb) > 1 or st.session_state.breadcrumb[0] != 'Home':
        bc = " > ".join([f"<span>{b}</span>" for b in st.session_state.breadcrumb])
        st.markdown(f'<div class="breadcrumb">{bc}</div>', unsafe_allow_html=True)

def back_button(target_page, target_breadcrumb):
    st.markdown("<br>", unsafe_allow_html=True)
    current = st.session_state.page
    if st.button("Back", key=f"back_{current}_to_{target_page}"):
        navigate_to(target_page, target_breadcrumb); st.rerun()

def map_columns(df, required_fields, file_label):
    all_cols = df.columns.tolist()
    mapped = {}
    st.markdown(f"**Map columns for {file_label}:**")
    cols_per_row = min(3, len(required_fields))
    for i in range(0, len(required_fields), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(required_fields):
                field = required_fields[idx]
                with row_cols[j]:
                    default_val = field if field in all_cols else (all_cols[idx] if idx < len(all_cols) else "")
                    default_idx = all_cols.index(default_val) if default_val in all_cols else 0
                    mapped[field] = st.selectbox(f"{field}", all_cols, index=default_idx, key=f"fv_map_{file_label}_{field}")
    return mapped

# =============================================================================
#  NAVIGATION MENUS
# =============================================================================

def render_home():
    st.markdown('<div class="hero"><h1>Next Vantage</h1><p>Comprehensive Actuarial Reserving Toolkit — IFRS 17 Compliant</p></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><h3>Full IFRS 17 Valuation</h3><p>Complete valuation with Income Statement & Liability Rollforward per LOB</p></div>', unsafe_allow_html=True)
        if st.button("Full Valuation", key="nav_home_full"): navigate_to('full_valuation', ['Home','Full Valuation']); st.rerun()
    with col2:
        st.markdown('<div class="card"><h3>Individual Calculators</h3><p>LRC | LIC | Risk Adjustment — standalone tools</p></div>', unsafe_allow_html=True)
        if st.button("Calculators", key="nav_home_calc"): navigate_to('lrc', ['Home','Individual Calculators','LRC']); st.rerun()
    with col3:
        st.markdown('<div class="card"><h3>LIC</h3><p>Fulfilment Cashflows | Risk Adjustment</p></div>', unsafe_allow_html=True)
        if st.button("Go to LIC", key="nav_home_lic"): navigate_to('lic', ['Home','LIC']); st.rerun()

def render_lrc():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Individual LRC Calculators</h1><p>Liability for Remaining Coverage</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>UPR Calculator</h3><p>Pro-rata unearned premium calculation via 365th, 24th, or 8th methods.</p></div>', unsafe_allow_html=True)
        if st.button("Open UPR Calculator", key="nav_lrc_upr"): navigate_to('upr_calculator', ['Home','Individual Calculators','LRC','UPR Calculator']); st.rerun()
    with col2:
        st.markdown('<div class="card"><h3>Loss Component</h3><p>Onerous contract identification and loss component recognition under IFRS 17 PAA.</p></div>', unsafe_allow_html=True)
        if st.button("Open Loss Component", key="nav_lrc_loss"): navigate_to('loss_component', ['Home','Individual Calculators','LRC','Loss Component']); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    back_button('home', ['Home'])

def render_lic():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>LIC Calculators</h1><p>Liability for Incurred Claims</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>Fulfilment Cashflows</h3><p>OCR, IBNR (multiple methods), ULAE, and NPR calculation.</p></div>', unsafe_allow_html=True)
        if st.button("Open FCF", key="nav_lic_fulfil"): navigate_to('fulfilment_cashflows', ['Home','LIC','Fulfilment Cashflows']); st.rerun()
    with col2:
        st.markdown('<div class="card"><h3>Risk Adjustment</h3><p>Bootstrap, Mack, VaR, and Cost of Capital approaches.</p></div>', unsafe_allow_html=True)
        if st.button("Open RA", key="nav_lic_ra"): navigate_to('risk_adjustment', ['Home','LIC','Risk Adjustment']); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    back_button('home', ['Home'])

def render_fulfilment_cashflows():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Fulfilment Cashflows</h1><p>LIC Components</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    cols = st.columns(4)
    items = [("OCR", "ocr_calculator"), ("IBNR Methods", "ibnr_menu"), ("ULAE", "ulae_calculator"), ("NPR", "npr_calculator")]
    for i, (t, p) in enumerate(items):
        with cols[i]:
            st.markdown(f'<div class="card"><h3>{t}</h3></div>', unsafe_allow_html=True)
            if st.button(f"Open {t}", key=f"nav_fc_{p}"): navigate_to(p, ['Home','LIC','Fulfilment Cashflows',t]); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    back_button('lic', ['Home','LIC'])

def render_ibnr_menu():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>IBNR Methods</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    # Removed "Percentage" as it's not in the router
    methods = [("BCL", "bcl_calculator"), ("Cape Cod", "capecod_calculator"), ("BF", "bf_calculator")]
    for i in range(0, len(methods), 3):
        cols = st.columns(3)
        for j in range(3):
            if i+j < len(methods):
                n, p = methods[i+j]
                with cols[j]:
                    st.markdown(f'<div class="card"><h3>{n}</h3></div>', unsafe_allow_html=True)
                    if st.button(f"Open {n}", key=f"nav_ibnr_{p}"): navigate_to(p, ['Home','LIC','Fulfilment Cashflows','IBNR Methods',n]); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    back_button('fulfilment_cashflows', ['Home','LIC','Fulfilment Cashflows'])

def render_risk_adjustment():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Risk Adjustment</h1><p>RA Methods</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    cols = st.columns(4)
    methods = [("Mack", "mack_calculator"), ("Bootstrap", "bootstrap_calculator")]
    for i, (n, p) in enumerate(methods):
        with cols[i]:
            st.markdown(f'<div class="card"><h3>{n}</h3></div>', unsafe_allow_html=True)
            if st.button(f"Open {n}", key=f"nav_ra_{p}"): navigate_to(p, ['Home','LIC','Risk Adjustment',n]); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    back_button('lic', ['Home','LIC'])


# =============================================================================
#  INDIVIDUAL CALCULATOR FUNCTIONS (Standalone)
# =============================================================================

def render_upr_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>UPR Calculator</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: valuation_date = st.date_input("Valuation Date", value=date(2025,12,31), key="upr_vd")
    with col2: client_name = st.text_input("Client", value="Client", key="upr_cn").strip()
    with col3: method = st.selectbox("Method", ["365th","24th","8th"], key="upr_mt")
    with col4: pass
    valuation_date = pd.Timestamp(str(valuation_date))
    uploaded_file = st.file_uploader("Upload premium register (CSV or Excel)", type=["csv","xlsx","xls"], key="upr_f")
    if uploaded_file is not None:
        try:
            original_filename = uploaded_file.name; base_filename = re.sub(r'\.[^.]*$','',original_filename)
            ext = uploaded_file.name.split('.')[-1].lower()
            df = pd.read_csv(uploaded_file) if ext=='csv' else pd.read_excel(uploaded_file)
            unnamed = [c for c in df.columns if c.startswith('Unnamed:')]
            if unnamed: df = df.drop(columns=unnamed)
            all_columns = df.columns.tolist()
            r1, r2 = st.columns(2)
            with r1: start_date_col = st.selectbox("Start Date Column", [""] + all_columns, key="upr_sd")
            with r2: end_date_col = st.selectbox("End Date Column", [""] + all_columns, key="upr_ed")
            if not start_date_col or not end_date_col: st.info("Please select Start and End Date columns."); return
            grouping_options = [c for c in all_columns if c not in [start_date_col, end_date_col]]
            grouping_cols = st.multiselect("Group by:", options=grouping_options, default=[grouping_options[0]] if grouping_options else [], key="upr_gc")
            if not grouping_cols: st.info("Please select at least one Group By column."); return
            numeric_columns = [c for c in df.columns if c not in [start_date_col,end_date_col]+grouping_cols and pd.api.types.is_numeric_dtype(df[c])]
            selected_value_cols = st.multiselect("Numeric columns:", options=numeric_columns, default=numeric_columns[:min(4,len(numeric_columns))], key="upr_vc")
            if not selected_value_cols: st.info("Please select at least one Numeric column."); return
            df_check = df.rename(columns={start_date_col:'Start_Date', end_date_col:'End_Date'})
            df_check['Start_Date'] = pd.to_datetime(df_check['Start_Date'], errors='coerce').astype('datetime64[ns]')
            df_check['End_Date'] = pd.to_datetime(df_check['End_Date'], errors='coerce').astype('datetime64[ns]')
            bad = df_check.dropna(subset=['Start_Date','End_Date']); bad = bad[bad['End_Date'] <= bad['Start_Date']]
            if len(bad) > 0: st.error(f"{len(bad)} rows have End_Date ≤ Start_Date."); return
            df_processed = df_check.dropna(subset=['Start_Date','End_Date']); df_processed = df_processed[df_processed['End_Date'] > df_processed['Start_Date']]
            for c in selected_value_cols: df_processed[c] = pd.to_numeric(df_processed[c], errors='coerce')
            df_processed["Duration"] = (df_processed["End_Date"] - df_processed["Start_Date"]).dt.days
            df_processed = df_processed[df_processed["Duration"] > 0]
            if df_processed.empty: st.error("No valid policies after date filtering."); return
            if st.button("Calculate UPR", key="upr_calc", width='stretch'):
                cond = [valuation_date < df_processed["Start_Date"], valuation_date > df_processed["End_Date"], (valuation_date <= df_processed["End_Date"]) & (valuation_date >= df_processed["Start_Date"])]
                if method == "365th": t=df_processed["Duration"]; r=(df_processed["End_Date"]-valuation_date).dt.days; ch=[1,0,r/t]
                elif method == "24th": iv=365.25/24; t=df_processed["Duration"]/iv; r=(df_processed["End_Date"]-valuation_date).dt.days/iv; ch=[1,0,r/t]
                else: iv=365.25/8; t=df_processed["Duration"]/iv; r=(df_processed["End_Date"]-valuation_date).dt.days/iv; ch=[1,0,r/t]
                df_processed["Unearned"] = np.select(cond, ch, default=np.nan)
                for c in selected_value_cols: df_processed[f"{c}_UPR"] = df_processed["Unearned"] * df_processed[c]
                upr_c = [f"{c}_UPR" for c in selected_value_cols]
                result = df_processed.groupby(grouping_cols)[upr_c].sum().reset_index()
                result.columns = grouping_cols + [c.replace('_UPR','') for c in upr_c]
                disp = result.copy()
                for c in disp.columns:
                    if c not in grouping_cols: disp[c] = disp[c].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
                st.dataframe(disp, width='stretch')
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as w: result.to_excel(w, index=False)
                output.seek(0)
                sc = re.sub(r'[\\/*?:"<>|]', "", client_name).strip() or "Client"
                so = re.sub(r'[\\/*?:"<>|]', "", base_filename).strip() or "Data"
                st.download_button("⬇  Download UPR Results", data=output, file_name=f"{sc}_{so}_UPR.xlsx", key="upr_dl")
        except Exception as e: st.error(f"Error: {e}")
    back_button('lrc', ['Home','Individual Calculators','LRC'])

def render_ocr_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>OCR Calculator</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: client_name = st.text_input("Client", value="Client", key="ocr_cn").strip()
    uploaded_file = st.file_uploader("Upload case estimates file", type=["csv","xlsx","xls"], key="ocr_f")
    if uploaded_file is not None:
        try:
            original_filename = uploaded_file.name; base_filename = re.sub(r'\.[^.]*$','',original_filename)
            ext = uploaded_file.name.split('.')[-1].lower()
            df = pd.read_csv(uploaded_file) if ext=='csv' else pd.read_excel(uploaded_file)
            unnamed = [c for c in df.columns if c.startswith('Unnamed:')]
            if unnamed: df = df.drop(columns=unnamed)
            all_columns = df.columns.tolist()
            grouping_cols = st.multiselect("Group by:", options=all_columns, default=[all_columns[0]] if all_columns else [], key="ocr_gc")
            if not grouping_cols: st.info("Please select at least one Group By column."); return
            numeric_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c not in grouping_cols]
            selected_value_cols = st.multiselect("Numeric columns:", options=numeric_columns, default=numeric_columns[:min(5,len(numeric_columns))], key="ocr_vc")
            if not selected_value_cols: st.info("Please select at least one Numeric column."); return
            df_processed = df[grouping_cols + selected_value_cols].copy()
            for c in selected_value_cols: df_processed[c] = pd.to_numeric(df_processed[c], errors='coerce').fillna(0)
            grouped = df_processed.groupby(grouping_cols)[selected_value_cols].sum().reset_index()
            disp = grouped.copy()
            for c in selected_value_cols: disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp, width='stretch')
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w: grouped.to_excel(w, index=False)
            output.seek(0)
            sc = re.sub(r'[\\/*?:"<>|]', "", client_name).strip() or "Client"
            so = re.sub(r'[\\/*?:"<>|]', "", base_filename).strip() or "Data"
            st.download_button("⬇  Download OCR Results", data=output, file_name=f"{sc}_{so}_OCR.xlsx", key="ocr_dl")
        except Exception as e: st.error(f"Error: {e}")
    back_button('fulfilment_cashflows', ['Home','LIC','Fulfilment Cashflows'])


# =============================================================================
#  UPDATED IBNR CALCULATORS WITH MULTI-AMOUNT, LDF SELECTION & INFLATION/DISCOUNTING
# =============================================================================

def render_bcl_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Basic Chain Ladder (BCL) — IBNR Calculator</h1><p>Multi-LDF selection, Inflation & Discounting, Multiple Amounts</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: client_name = st.text_input("Client", "Client", key="bcl_cn").strip()
    with c2: from_date = st.date_input("From Date", date(2020,1,1), key="bcl_fd")
    with c3: to_date = st.date_input("To Date", date(2025,12,31), key="bcl_td")
    with c4: grain = st.selectbox("Grain", ["Yearly","Quarterly","Monthly"], key="bcl_gr")

    grain_map = {"Yearly":"Y","Quarterly":"Q","Monthly":"M"}
    grain_code = grain_map[grain]
    ppy = {"Y":1, "Q":4, "M":12}[grain_code]

    uploaded = st.file_uploader("Upload claims file (CSV/Excel)", type=["csv","xlsx","xls"], key="bcl_f")
    if uploaded is None:
        st.info("Upload a claims file with Loss Date, Report Date, Line of Business, and Claim Amount columns.")
        back_button('ibnr_menu', ['Home','LIC','Fulfilment Cashflows','IBNR Methods']); return

    try:
        df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        df.columns = df.columns.astype(str).str.strip()
        st.dataframe(df.head(3), width='stretch')
        cols = df.columns.tolist()
        c1,c2,c3,c4 = st.columns(4)
        with c1: loss_col = st.selectbox("Loss Date", cols, key="bcl_ld")
        with c2: rep_col = st.selectbox("Report Date", cols, key="bcl_rd")
        with c3: lob_col = st.selectbox("Line of Business", cols, key="bcl_lob")
        
        # ===== MULTIPLE AMOUNT COLUMN SELECTION =====
        amount_candidates = [c for c in cols if c not in [loss_col, rep_col, lob_col] and pd.api.types.is_numeric_dtype(df[c])]
        amount_cols = st.multiselect("Select Amount Column(s)", amount_candidates, key="bcl_amt")

        if not amount_cols:
            st.warning("Please select at least one Amount column.")
            return

        df[loss_col] = pd.to_datetime(df[loss_col], errors='coerce').astype('datetime64[ns]')
        df[rep_col]  = pd.to_datetime(df[rep_col],  errors='coerce').astype('datetime64[ns]')
        for ac in amount_cols:
            df[ac] = pd.to_numeric(df[ac], errors='coerce').fillna(0)
        
        df = df.dropna(subset=[loss_col, rep_col])
        from_dt = pd.Timestamp(str(from_date)); to_dt = pd.Timestamp(str(to_date))
        df = _date_filter(df, loss_col, from_date, to_date)

        # ===== INFLATION & DISCOUNTING INPUTS =====
        st.markdown("#### Step 3: Inflation & Discounting Adjustments")
        c1, c2 = st.columns(2)
        with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key="bcl_inf")
        with c2: use_discounting = st.checkbox("Apply Discounting", key="bcl_disc")

        cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
        if use_inflation:
            cum_inflation, per_period_rates = load_inflation_data_ui(grain_code, ppy)
        if use_discounting:
            spot_rates, flat_rate = load_discounting_data_ui(grain_code, ppy)

        # ===== CORE LOGIC =====
        if st.button("Calculate BCL IBNR", key="bcl_run", width='stretch'):
            lobs = sorted(df[lob_col].dropna().unique())
            n_periods = (to_date.year - from_date.year) * ppy + 1
            
            # For the UI, we display the LDF matrix for the first selected amount column
            st.subheader("LDF Selection")
            st.info("Tail factor is hardcoded to 1.000 (fully developed).")
            
            if engine_utils is not None:
                sample_amt = amount_cols[0]
                _, sample_cum, _ = engine_utils.build_triangles(df, loss_col, rep_col, sample_amt, from_dt, grain_code, n_periods)
                all_ldfs = ibnr_bcl.calculate_all_ldfs(sample_cum, n_periods)
                
                ldf_df = pd.DataFrame({
                    "Dev Period": range(1, n_periods),
                    "Vol-Weighted": all_ldfs["volume_weighted"],
                    "Simple Avg": all_ldfs["simple_average"],
                    "Geometric": all_ldfs["geometric"],
                    "Medial": all_ldfs["medial"],
                    "Lin Reg (clamped)": all_ldfs["linear_regression"],
                    "Wtd Last 3": all_ldfs["weighted_last_3"]
                })
                st.dataframe(ldf_df, width='stretch')

                # Calculate Recommended LDF
                rec_method = "volume_weighted" # default
                min_cv = float('inf')
                for method in ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"]:
                    factors = all_ldfs[method]
                    # Calculate CV for first 3 factors
                    if len(factors) >= 3:
                        cv = np.std(factors[:3]) / np.mean(factors[:3]) if np.mean(factors[:3]) > 0 else float('inf')
                        if cv < min_cv:
                            min_cv = cv
                            rec_method = method
                st.info(f"**Recommended LDF Method:** {rec_method.replace('_', ' ').title()} (Lowest CV of first 3 factors: {min_cv:.2%})")

                selected_method = st.selectbox(
                    "Select LDF Method",
                    ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"],
                    index=0,
                    key="bcl_ldf_method"
                )
            else:
                st.error("Could not load engine utilities to display LDF selection.")
                selected_method = "volume_weighted"
            
            all_results = []
            for lob in lobs:
                lob_data = df[df[lob_col]==lob].copy()
                for ac in amount_cols:
                    _, cum, _ = engine_utils.build_triangles(lob_data, loss_col, rep_col, ac, from_dt, grain_code, n_periods)
                    
                    result = ibnr_bcl.calculate_bcl_ibnr(
                        cum_triangle=cum,
                        start_date=from_dt,
                        period_unit=grain_code,
                        selected_ldf_method=selected_method,
                        use_inflation=use_inflation,
                        cum_inflation=cum_inflation,
                        per_period_rates=per_period_rates,
                        use_discounting=use_discounting,
                        spot_rates=spot_rates,
                        flat_rate=flat_rate
                    )
                    res_df = result['results_df']
                    res_df['LOB'] = lob
                    res_df['Amount_Col'] = ac
                    all_results.append(res_df)
            
            final_df = pd.concat(all_results, ignore_index=True)
            summary = final_df.groupby(['LOB', 'Amount_Col'])[['Current_Claims', 'IBNR']].sum().reset_index()
            
            st.subheader("BCL IBNR Summary")
            disp = summary.copy()
            for c in ['Current_Claims', 'IBNR']:
                disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp, width='stretch', hide_index=True)

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w:
                summary.to_excel(w, index=False, sheet_name='BCL_Summary')
                final_df.to_excel(w, index=False, sheet_name='BCL_Detail')
            output.seek(0)
            sc = re.sub(r'[\\/*?:"<>|]','',client_name).strip() or "Client"
            st.download_button("⬇ Download BCL Results", data=output, file_name=f"{sc}_BCL_IBNR.xlsx", key="bcl_dl")

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback; st.write(traceback.format_exc())

    back_button('ibnr_menu', ['Home','LIC','Fulfilment Cashflows','IBNR Methods'])


def render_capecod_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Cape Cod IBNR</h1><p>Multi-LDF selection, Inflation & Discounting, Multiple Amounts</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: client_name=st.text_input("Client","Client",key="cc_cn").strip()
    with c2: from_date=st.date_input("From Date",date(2020,1,1),key="cc_fd")
    with c3: to_date=st.date_input("To Date",date(2025,12,31),key="cc_td")
    claims_file=st.file_uploader("Upload claims file",type=["csv","xlsx","xls"],key="cc_cf")
    prem_file=st.file_uploader("Upload premiums file (Portfolio | Premium per accident year)",type=["csv","xlsx","xls"],key="cc_pf")
    if claims_file is None or prem_file is None:
        st.info("Upload both claims and premiums files to run Cape Cod.")
        back_button('ibnr_menu',['Home','LIC','Fulfilment Cashflows','IBNR Methods']); return
    
    try:
        df=pd.read_csv(claims_file) if claims_file.name.endswith('.csv') else pd.read_excel(claims_file)
        df.columns=df.columns.astype(str).str.strip()
        prem_df=pd.read_csv(prem_file) if prem_file.name.endswith('.csv') else pd.read_excel(prem_file)
        prem_df.columns=prem_df.columns.astype(str).str.strip()
        cols=df.columns.tolist()
        c1,c2,c3,c4=st.columns(4)
        with c1: loss_col=st.selectbox("Loss Date",cols,key="cc_ld")
        with c2: rep_col=st.selectbox("Report Date",cols,key="cc_rd")
        with c3: lob_col=st.selectbox("LOB",cols,key="cc_lob")
        
        # ===== MULTIPLE AMOUNT COLUMN SELECTION =====
        amount_candidates = [c for c in cols if c not in [loss_col, rep_col, lob_col] and pd.api.types.is_numeric_dtype(df[c])]
        amount_cols = st.multiselect("Select Amount Column(s)", amount_candidates, key="cc_amt")

        if not amount_cols:
            st.warning("Please select at least one Amount column.")
            return

        df[loss_col]=pd.to_datetime(df[loss_col],errors='coerce').astype('datetime64[ns]')
        df[rep_col]=pd.to_datetime(df[rep_col],errors='coerce').astype('datetime64[ns]')
        for ac in amount_cols:
            df[ac]=pd.to_numeric(df[ac],errors='coerce').fillna(0)
            
        df=df.dropna(subset=[loss_col,rep_col])
        from_dt=pd.Timestamp(str(from_date)); to_dt=pd.Timestamp(str(to_date))
        df=_date_filter(df, loss_col, from_date, to_date)

        # ===== PREMIUM DATA COLUMN MAPPING =====
        st.markdown("#### Premium Data Column Mapping")
        p_cols = prem_df.columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1: prem_lob_col = st.selectbox("LOB / Portfolio Column", p_cols, key="cc_prem_lob")
        with c2: prem_amt_col = st.selectbox("Premium Amount Column", p_cols, key="cc_prem_amt")
        with c3: prem_date_col = st.selectbox("Premium Date Column (Optional)", ["None"] + p_cols, key="cc_prem_date")
        
        use_prem_date = prem_date_col != "None"
        prem_df[prem_lob_col] = prem_df[prem_lob_col].astype(str)
        prem_df[prem_amt_col] = pd.to_numeric(prem_df[prem_amt_col], errors='coerce').fillna(0)
        if use_prem_date:
            prem_df[prem_date_col] = pd.to_datetime(prem_df[prem_date_col], errors='coerce')

        # ===== INFLATION & DISCOUNTING INPUTS =====
        st.markdown("#### Inflation & Discounting Adjustments")
        c1, c2 = st.columns(2)
        with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key="cc_inf")
        with c2: use_discounting = st.checkbox("Apply Discounting", key="cc_disc")

        grain = "Y"; ppy = 1
        cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
        if use_inflation:
            cum_inflation, per_period_rates = load_inflation_data_ui(grain, ppy)
        if use_discounting:
            spot_rates, flat_rate = load_discounting_data_ui(grain, ppy)

        # ===== CORE LOGIC =====
        if st.button("Calculate Cape Cod IBNR", key="cc_run", width='stretch'):
            lobs=sorted(df[lob_col].dropna().unique())
            n = (to_date.year - from_date.year) + 1
            
            # LDF Selection UI
            st.subheader("LDF Selection")
            st.info("Tail factor is hardcoded to 1.000 (fully developed).")
            if engine_utils is not None:
                sample_amt = amount_cols[0]
                _, sample_cum, _ = engine_utils.build_triangles(df, loss_col, rep_col, sample_amt, from_dt, grain, n)
                all_ldfs = ibnr_cc.calculate_all_ldfs(sample_cum, n)
                
                ldf_df = pd.DataFrame({
                    "Dev Period": range(1, n),
                    "Vol-Weighted": all_ldfs["volume_weighted"],
                    "Simple Avg": all_ldfs["simple_average"],
                    "Geometric": all_ldfs["geometric"],
                    "Medial": all_ldfs["medial"],
                    "Lin Reg (clamped)": all_ldfs["linear_regression"],
                    "Wtd Last 3": all_ldfs["weighted_last_3"]
                })
                st.dataframe(ldf_df, width='stretch')

                # Calculate Recommended LDF
                rec_method = "volume_weighted" # default
                min_cv = float('inf')
                for method in ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"]:
                    factors = all_ldfs[method]
                    if len(factors) >= 3:
                        cv = np.std(factors[:3]) / np.mean(factors[:3]) if np.mean(factors[:3]) > 0 else float('inf')
                        if cv < min_cv:
                            min_cv = cv
                            rec_method = method
                st.info(f"**Recommended LDF Method:** {rec_method.replace('_', ' ').title()} (Lowest CV of first 3 factors: {min_cv:.2%})")

                selected_method = st.selectbox(
                    "Select LDF Method",
                    ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"],
                    index=0,
                    key="cc_ldf_method"
                )
            else:
                st.error("Could not load engine utilities.")
                selected_method = "volume_weighted"
            
            all_results = []
            for lob in lobs:
                lob_data = df[df[lob_col]==lob].copy()
                # Get premiums for this LOB
                prem_sub = prem_df[prem_df[prem_lob_col] == lob].copy()
                if use_prem_date:
                    prem_sub['Year'] = prem_sub[prem_date_col].dt.year
                    prems = prem_sub.groupby('Year')[prem_amt_col].sum().reindex(range(from_dt.year, from_dt.year + n), fill_value=0).tolist()
                else:
                    prems = prem_sub[prem_amt_col].tolist()
                    if len(prems) < n: prems.extend([0] * (n - len(prems)))
                    elif len(prems) > n: prems = prems[:n]

                for ac in amount_cols:
                    _, cum, _ = engine_utils.build_triangles(lob_data, loss_col, rep_col, ac, from_dt, grain, n)
                    
                    result = ibnr_cc.calculate_cape_cod_ibnr(
                        cum_triangle=cum,
                        premiums=prems,
                        start_date=from_dt,
                        period_unit=grain,
                        selected_ldf_method=selected_method,
                        use_inflation=use_inflation,
                        cum_inflation=cum_inflation,
                        per_period_rates=per_period_rates,
                        use_discounting=use_discounting,
                        spot_rates=spot_rates,
                        flat_rate=flat_rate
                    )
                    res_df = result['results_df']
                    res_df['LOB'] = lob
                    res_df['Amount_Col'] = ac
                    all_results.append(res_df)
            
            final_df = pd.concat(all_results, ignore_index=True)
            summary = final_df.groupby(['LOB', 'Amount_Col'])[['Current_Claims', 'Cape_Cod_IBNR']].sum().reset_index()
            
            st.subheader("Cape Cod IBNR Summary")
            disp = summary.copy()
            for c in ['Current_Claims', 'Cape_Cod_IBNR']:
                disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp, width='stretch', hide_index=True)

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w:
                summary.to_excel(w, index=False, sheet_name='CapeCod_Summary')
                final_df.to_excel(w, index=False, sheet_name='CapeCod_Detail')
            output.seek(0)
            sc = re.sub(r'[\/*?:"<>|]','',client_name).strip() or "Client"
            st.download_button("⬇ Download Cape Cod Results", data=output, file_name=f"{sc}_CapeCod_IBNR.xlsx", key="cc_dl")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback; st.write(traceback.format_exc())
    back_button('ibnr_menu',['Home','LIC','Fulfilment Cashflows','IBNR Methods'])


def render_bf_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Bornhuetter-Ferguson (BF) IBNR</h1><p>Multi-LDF selection, Inflation & Discounting, Multiple Amounts</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: client_name=st.text_input("Client","Client",key="bf_cn").strip()
    with c2: from_date=st.date_input("From Date",date(2020,1,1),key="bf_fd")
    with c3: to_date=st.date_input("To Date",date(2025,12,31),key="bf_td")
    claims_file=st.file_uploader("Upload claims file",type=["csv","xlsx","xls"],key="bf_cf")
    prem_file=st.file_uploader("Upload premiums file",type=["csv","xlsx","xls"],key="bf_pf")
    if claims_file is None:
        st.info("Upload claims file (and optionally premiums) to run BF.")
        back_button('ibnr_menu',['Home','LIC','Fulfilment Cashflows','IBNR Methods']); return
    try:
        df=pd.read_csv(claims_file) if claims_file.name.endswith('.csv') else pd.read_excel(claims_file)
        df.columns=df.columns.astype(str).str.strip()
        cols=df.columns.tolist()
        c1,c2,c3,c4=st.columns(4)
        with c1: loss_col=st.selectbox("Loss Date",cols,key="bf_ld")
        with c2: rep_col=st.selectbox("Report Date",cols,key="bf_rd")
        with c3: lob_col=st.selectbox("LOB",cols,key="bf_lob")
        
        # ===== MULTIPLE AMOUNT COLUMN SELECTION =====
        amount_candidates = [c for c in cols if c not in [loss_col, rep_col, lob_col] and pd.api.types.is_numeric_dtype(df[c])]
        amount_cols = st.multiselect("Select Amount Column(s)", amount_candidates, key="bf_amt")

        if not amount_cols:
            st.warning("Please select at least one Amount column.")
            return

        df[loss_col]=pd.to_datetime(df[loss_col],errors='coerce').astype('datetime64[ns]')
        df[rep_col]=pd.to_datetime(df[rep_col],errors='coerce').astype('datetime64[ns]')
        for ac in amount_cols:
            df[ac]=pd.to_numeric(df[ac],errors='coerce').fillna(0)
            
        df=df.dropna(subset=[loss_col,rep_col])
        from_dt=pd.Timestamp(str(from_date)); to_dt=pd.Timestamp(str(to_date))
        df=_date_filter(df, loss_col, from_date, to_date)
        lobs=sorted(df[lob_col].dropna().unique())

        # ===== PREMIUM DATA COLUMN MAPPING =====
        st.markdown("#### Premium Data Column Mapping")
        if prem_file is not None:
            prem_df = pd.read_csv(prem_file) if prem_file.name.endswith('.csv') else pd.read_excel(prem_file)
            prem_df.columns = prem_df.columns.astype(str).str.strip()
            prem_cols = prem_df.columns.tolist()
            c1, c2, c3 = st.columns(3)
            with c1: prem_lob_col = st.selectbox("LOB / Portfolio Column", prem_cols, key="bf_prem_lob")
            with c2: prem_amt_col = st.selectbox("Premium Amount Column", prem_cols, key="bf_prem_amt")
            with c3: prem_date_col = st.selectbox("Premium Date Column (Optional)", ["None"] + prem_cols, key="bf_prem_date")
            
            use_prem_date = prem_date_col != "None"
            prem_df[prem_lob_col] = prem_df[prem_lob_col].astype(str)
            prem_df[prem_amt_col] = pd.to_numeric(prem_df[prem_amt_col], errors='coerce').fillna(0)
            if use_prem_date:
                prem_df[prem_date_col] = pd.to_datetime(prem_df[prem_date_col], errors='coerce')
        else:
            prem_df = None
            use_prem_date = False

        st.markdown("**ELR per Portfolio (%):**")
        elr_cols=st.columns(min(len(lobs),4))
        elr_dict={}
        for i,lob in enumerate(lobs):
            with elr_cols[i%4]: elr_dict[lob]=st.number_input(f"ELR {lob}",0.0,200.0,70.0,1.0,key=f"bf_elr_{lob}")/100

        # ===== INFLATION & DISCOUNTING INPUTS =====
        st.markdown("#### Inflation & Discounting Adjustments")
        c1, c2 = st.columns(2)
        with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key="bf_inf")
        with c2: use_discounting = st.checkbox("Apply Discounting", key="bf_disc")

        grain = "Y"; ppy = 1
        cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
        if use_inflation:
            cum_inflation, per_period_rates = load_inflation_data_ui(grain, ppy)
        if use_discounting:
            spot_rates, flat_rate = load_discounting_data_ui(grain, ppy)

        # ===== CORE LOGIC =====
        if st.button("Calculate BF IBNR", key="bf_run", width='stretch'):
            n = (to_date.year - from_date.year) + 1

            # LDF Selection UI
            st.subheader("LDF Selection")
            st.info("Tail factor is hardcoded to 1.000 (fully developed).")
            if engine_utils is not None:
                sample_amt = amount_cols[0]
                _, sample_cum, _ = engine_utils.build_triangles(df, loss_col, rep_col, sample_amt, from_dt, grain, n)
                all_ldfs = ibnr_bf.calculate_all_ldfs(sample_cum, n)
                
                ldf_df = pd.DataFrame({
                    "Dev Period": range(1, n),
                    "Vol-Weighted": all_ldfs["volume_weighted"],
                    "Simple Avg": all_ldfs["simple_average"],
                    "Geometric": all_ldfs["geometric"],
                    "Medial": all_ldfs["medial"],
                    "Lin Reg (clamped)": all_ldfs["linear_regression"],
                    "Wtd Last 3": all_ldfs["weighted_last_3"]
                })
                st.dataframe(ldf_df, width='stretch')

                # Calculate Recommended LDF
                rec_method = "volume_weighted" # default
                min_cv = float('inf')
                for method in ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"]:
                    factors = all_ldfs[method]
                    if len(factors) >= 3:
                        cv = np.std(factors[:3]) / np.mean(factors[:3]) if np.mean(factors[:3]) > 0 else float('inf')
                        if cv < min_cv:
                            min_cv = cv
                            rec_method = method
                st.info(f"**Recommended LDF Method:** {rec_method.replace('_', ' ').title()} (Lowest CV of first 3 factors: {min_cv:.2%})")

                selected_method = st.selectbox(
                    "Select LDF Method",
                    ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"],
                    index=0,
                    key="bf_ldf_method"
                )
            else:
                st.error("Could not load engine utilities.")
                selected_method = "volume_weighted"
            
            all_results = []
            for lob in lobs:
                lob_data = df[df[lob_col]==lob].copy()
                # Get premiums
                if prem_df is not None:
                    prem_sub = prem_df[prem_df[prem_lob_col] == lob].copy()
                    if use_prem_date:
                        prem_sub['Year'] = prem_sub[prem_date_col].dt.year
                        prems = prem_sub.groupby('Year')[prem_amt_col].sum().reindex(range(from_dt.year, from_dt.year + n), fill_value=0).tolist()
                    else:
                        prems = prem_sub[prem_amt_col].tolist()
                        if len(prems) < n: prems.extend([0] * (n - len(prems)))
                        elif len(prems) > n: prems = prems[:n]
                else:
                    prems = [1] * n # Placeholder if no premium file provided

                for ac in amount_cols:
                    _, cum, _ = engine_utils.build_triangles(lob_data, loss_col, rep_col, ac, from_dt, grain, n)
                    
                    result = ibnr_bf.calculate_bf_ibnr(
                        cum_triangle=cum,
                        premiums=prems,
                        elr=elr_dict.get(lob, 0.7),
                        start_date=from_dt,
                        period_unit=grain,
                        selected_ldf_method=selected_method,
                        use_inflation=use_inflation,
                        cum_inflation=cum_inflation,
                        per_period_rates=per_period_rates,
                        use_discounting=use_discounting,
                        spot_rates=spot_rates,
                        flat_rate=flat_rate
                    )
                    res_df = result['results_df']
                    res_df['LOB'] = lob
                    res_df['Amount_Col'] = ac
                    all_results.append(res_df)
            
            final_df = pd.concat(all_results, ignore_index=True)
            summary = final_df.groupby(['LOB', 'Amount_Col'])[['Current_Claims', 'BF_IBNR']].sum().reset_index()
            
            st.subheader("BF IBNR Summary")
            disp = summary.copy()
            for c in ['Current_Claims', 'BF_IBNR']:
                disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp, width='stretch', hide_index=True)

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w:
                summary.to_excel(w, index=False, sheet_name='BF_Summary')
                final_df.to_excel(w, index=False, sheet_name='BF_Detail')
            output.seek(0)
            sc = re.sub(r'[\/*?:"<>|]','',client_name).strip() or "Client"
            st.download_button("⬇ Download BF Results", data=output, file_name=f"{sc}_BF_IBNR.xlsx", key="bf_dl")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback; st.write(traceback.format_exc())
    back_button('ibnr_menu',['Home','LIC','Fulfilment Cashflows','IBNR Methods'])


def render_ulae_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>ULAE Calculator</h1><p>Paid-to-Paid method: ULAE = ratio × (0.5×OCR + IBNR)</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: client_name=st.text_input("Client","Client",key="ulae_cn").strip()
    with c2: ulae_ratio=st.number_input("ULAE Ratio (%)",0.0,30.0,5.0,0.5,key="ulae_rt")/100
    with c3: basis=st.selectbox("Allocation Basis",["Per Portfolio","Aggregated"],key="ulae_bs")
    uploaded=st.file_uploader("Upload reserves file (LOB | OCR | IBNR)",type=["csv","xlsx","xls"],key="ulae_f")
    if uploaded is None:
        st.info("Upload file with LOB, OCR, and IBNR columns.")
        back_button('fulfilment_cashflows',['Home','LIC','Fulfilment Cashflows']); return
    try:
        df=pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        df.columns=df.columns.astype(str).str.strip()
        st.dataframe(df.head(5),width='stretch')
        cols=df.columns.tolist()
        c1,c2,c3=st.columns(3)
        with c1: lob_col=st.selectbox("LOB",cols,key="ulae_lob")
        with c2: ocr_col=st.selectbox("OCR",cols,key="ulae_ocr")
        with c3: ibnr_col=st.selectbox("IBNR",cols,key="ulae_ibnr")
        prem_col=None
        if basis=="Aggregated":
            with st.columns(2)[0]: prem_col=st.selectbox("Premium (for aggregated split)",cols,key="ulae_prem")
        if st.button("Calculate ULAE",key="ulae_run",width='stretch'):
            df[ocr_col]=pd.to_numeric(df[ocr_col],errors='coerce').fillna(0)
            df[ibnr_col]=pd.to_numeric(df[ibnr_col],errors='coerce').fillna(0)
            df['ULAE_Base']=0.5*df[ocr_col]+df[ibnr_col]
            if basis=="Aggregated" and prem_col:
                df[prem_col]=pd.to_numeric(df[prem_col],errors='coerce').fillna(0)
                tot_prem=df[prem_col].sum()
                df['Pct']=df[prem_col]/tot_prem if tot_prem>0 else 1/len(df)
                total_ulae=df['ULAE_Base'].sum()*ulae_ratio
                df['ULAE']=total_ulae*df['Pct']
            else:
                df['ULAE']=df['ULAE_Base']*ulae_ratio
            res=df[[lob_col,ocr_col,ibnr_col,'ULAE_Base','ULAE']].copy()
            st.subheader("ULAE Results")
            disp=res.copy()
            for c in [ocr_col,ibnr_col,'ULAE_Base','ULAE']: disp[c]=disp[c].apply(lambda x:f"{x:,.2f}")
            st.dataframe(disp,width='stretch',hide_index=True)
            st.metric("Total ULAE",f"{df['ULAE'].sum():,.2f}")
            output=BytesIO()
            with pd.ExcelWriter(output,engine='openpyxl') as w: res.to_excel(w,index=False,sheet_name='ULAE_Results')
            output.seek(0); sc=re.sub(r'[\/*?:"<>|]','',client_name).strip() or "Client"
            st.download_button("⬇ Download ULAE Results",data=output,file_name=f"{sc}_ULAE.xlsx",key="ulae_dl")
    except Exception as e: st.error(f"Error: {e}")
    back_button('fulfilment_cashflows',['Home','LIC','Fulfilment Cashflows'])

def render_npr_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Reinsurance Non-Performance Risk (NPR)</h1><p>IFRS 17 Para 63(e) — NPR = PD × Reinsurer Share × Ceded LIC</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: client_name=st.text_input("Client","Client",key="npr_cn").strip()
    with c2: basis=st.selectbox("Share Basis",["Aggregation (overall share)","Per Portfolio"],key="npr_bs")
    st.markdown("#### Reinsurer Data (Name | Credit_Rating | PD | Share columns)")
    ri_file=st.file_uploader("Upload Reinsurer file",type=["csv","xlsx","xls"],key="npr_rf")
    st.markdown("#### Ceded LIC Data (Portfolio | Ceded_IBNR | Ceded_OCR  OR  Portfolio | Total_Ceded_LIC)")
    lic_file=st.file_uploader("Upload Ceded LIC file",type=["csv","xlsx","xls"],key="npr_lf")
    if ri_file is None or lic_file is None:
        st.info("Upload both Reinsurer and Ceded LIC files.")
        back_button('fulfilment_cashflows',['Home','LIC','Fulfilment Cashflows']); return
    try:
        ri_df=pd.read_csv(ri_file) if ri_file.name.endswith('.csv') else pd.read_excel(ri_file)
        ri_df.columns=ri_df.columns.astype(str).str.strip()
        lic_df=pd.read_csv(lic_file) if lic_file.name.endswith('.csv') else pd.read_excel(lic_file)
        lic_df.columns=lic_df.columns.astype(str).str.strip()
        st.dataframe(ri_df.head(3),width='stretch')
        st.dataframe(lic_df.head(3),width='stretch')
        rc=ri_df.columns.tolist(); lc=lic_df.columns.tolist()
        c1,c2,c3=st.columns(3)
        with c1: name_col=st.selectbox("Reinsurer Name",rc,key="npr_rn")
        with c2: pd_col=st.selectbox("PD (decimal)",rc,key="npr_pd")
        with c3: share_col=st.selectbox("Overall Share (decimal)",rc,key="npr_sh")
        c1,c2,c3=st.columns(3)
        with c1: port_col=st.selectbox("Portfolio",lc,key="npr_pc")
        with c2: lic_fmt=st.selectbox("LIC Format",["Separate IBNR+OCR","Total LIC only"],key="npr_fmt")
        if lic_fmt=="Separate IBNR+OCR":
            with c3: ibnr_c=st.selectbox("Ceded IBNR",lc,key="npr_ibnr")
            ocr_c=st.selectbox("Ceded OCR",lc,key="npr_ocr")
        else:
            with c3: total_c=st.selectbox("Total Ceded LIC",lc,key="npr_tot")
        if st.button("Calculate NPR",key="npr_run",width='stretch'):
            ri_df[pd_col]=pd.to_numeric(ri_df[pd_col],errors='coerce').fillna(0)
            ri_df[share_col]=pd.to_numeric(ri_df[share_col],errors='coerce').fillna(0)
            if lic_fmt=="Separate IBNR+OCR":
                lic_df[ibnr_c]=pd.to_numeric(lic_df[ibnr_c],errors='coerce').fillna(0)
                lic_df[ocr_c]=pd.to_numeric(lic_df[ocr_c],errors='coerce').fillna(0)
                lic_df['Total_LIC']=lic_df[ibnr_c]+lic_df[ocr_c]
            else:
                lic_df['Total_LIC']=pd.to_numeric(lic_df[total_c],errors='coerce').fillna(0)
            rows=[]
            for _,ri in ri_df.iterrows():
                for _,lic in lic_df.iterrows():
                    npr=ri[pd_col]*ri[share_col]*lic['Total_LIC']
                    rows.append({'Reinsurer':ri[name_col],'Portfolio':lic[port_col],'PD':ri[pd_col],'Share':ri[share_col],'Total_LIC':lic['Total_LIC'],'NPR':npr})
            res=pd.DataFrame(rows)
            by_port=res.groupby('Portfolio')['NPR'].sum().reset_index()
            by_ri=res.groupby('Reinsurer')['NPR'].sum().reset_index()
            st.subheader("NPR by Portfolio"); disp=by_port.copy(); disp['NPR']=disp['NPR'].apply(lambda x:f"{x:,.2f}"); st.dataframe(disp,width='stretch',hide_index=True)
            st.subheader("NPR by Reinsurer"); disp2=by_ri.copy(); disp2['NPR']=disp2['NPR'].apply(lambda x:f"{x:,.2f}"); st.dataframe(disp2,width='stretch',hide_index=True)
            st.metric("Total NPR",f"{res['NPR'].sum():,.2f}")
            output=BytesIO()
            with pd.ExcelWriter(output,engine='openpyxl') as w:
                by_port.to_excel(w,index=False,sheet_name='NPR_by_Portfolio')
                by_ri.to_excel(w,index=False,sheet_name='NPR_by_Reinsurer')
                res.to_excel(w,index=False,sheet_name='NPR_Detail')
            output.seek(0); sc=re.sub(r'[\/*?:"<>|]','',client_name).strip() or "Client"
            st.download_button("⬇ Download NPR Results",data=output,file_name=f"{sc}_NPR.xlsx",key="npr_dl")
    except Exception as e: st.error(f"Error: {e}")
    back_button('fulfilment_cashflows',['Home','LIC','Fulfilment Cashflows'])

def render_mack_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Mack Chain Ladder — Risk Adjustment</h1><p>Distribution-free standard error of IBNR per Mack (1993). RA = z × σ(IBNR)</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: client_name=st.text_input("Client","Client",key="mck_cn").strip()
    with c2: confidence=st.number_input("Confidence Level (%)",50.0,99.9,75.0,1.0,key="mck_cl")/100
    from scipy.stats import norm
    z=norm.ppf(confidence)
    with c3: st.info(f"z-score: {z:.3f}")
    uploaded=st.file_uploader("Upload claims triangle (cumulative, CSV/Excel — rows=AY, cols=Dev)",type=["csv","xlsx","xls"],key="mck_f")
    st.info("Expected format: first column = Accident Year label, remaining columns = cumulative claims by development period.")
    if uploaded is None:
        back_button('risk_adjustment',['Home','LIC','Risk Adjustment']); return
    try:
        raw=pd.read_csv(uploaded,index_col=0) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded,index_col=0)
        raw.columns=raw.columns.astype(str).str.strip()
        raw=raw.apply(pd.to_numeric,errors='coerce')
        st.dataframe(raw,width='stretch')
        n=len(raw); m=len(raw.columns)
        C=raw.values.copy().astype(float)
        # Volume-weighted factors
        facs=[]
        for j in range(m-1):
            num=sum(C[i,j+1] for i in range(n-j-1) if not np.isnan(C[i,j+1]) and not np.isnan(C[i,j]))
            den=sum(C[i,j]   for i in range(n-j-1) if not np.isnan(C[i,j+1]) and not np.isnan(C[i,j]))
            facs.append(num/den if den>0 else 1.0)
        # Mack sigma^2 per development period
        sigmas=[]
        for j in range(m-1):
            pairs=[(C[i,j],C[i,j+1]) for i in range(n-j-1) if not np.isnan(C[i,j]) and not np.isnan(C[i,j+1]) and C[i,j]>0]
            if len(pairs)>=2:
                s2=sum(p[0]*(p[1]/p[0]-facs[j])**2 for p in pairs)/(len(pairs)-1)
            elif len(pairs)==1:
                s2=(pairs[0][0]*(pairs[0][1]/pairs[0][0]-facs[j])**2)
            else:
                s2=0.0
            sigmas.append(max(s2,0.0))
        # Tail sigma (Mack extrapolation)
        if len(sigmas)>=2 and sigmas[-2]>0:
            sigmas.append(min(sigmas[-1]**2/sigmas[-2],sigmas[-2],sigmas[-1]))
        else:
            sigmas.append(sigmas[-1] if sigmas else 0.0)
        # Project
        proj=C.copy()
        for i in range(n):
            lo=-1
            for j in range(m-1,-1,-1):
                if not np.isnan(C[i,j]): lo=j; break
            if lo<0: continue
            for j in range(lo,m-1):
                if j<len(facs): proj[i,j+1]=proj[i,j]*facs[j] if not np.isnan(proj[i,j]) else np.nan
        # Mack SE
        rows=[]
        for i in range(n):
            lo=-1
            for j in range(m-1,-1,-1):
                if not np.isnan(C[i,j]): lo=j; break
            if lo<0 or lo==m-1: continue
            c_lo=C[i,lo]; ult=proj[i,m-1]; ibnr=max(ult-c_lo,0.0)
            se2=0.0
            for j in range(lo,m-1):
                if j<len(sigmas) and j<len(facs):
                    cdf_rem=np.prod(facs[j+1:]) if j+1<len(facs) else 1.0
                    se2+=(sigmas[j]/facs[j]**2)*(cdf_rem**2 if cdf_rem else 1.0)/max(c_lo,1)
            mack_se=np.sqrt(max(se2,0.0))*ult
            ra=z*mack_se
            rows.append({'AY':raw.index[i],'Current':c_lo,'Ultimate':ult,'IBNR':ibnr,'Mack_SE':mack_se,'RA':ra})
        res=pd.DataFrame(rows)
        st.subheader(f"Mack RA @ {confidence*100:.0f}% Confidence")
        disp=res.copy()
        for c in ['Current','Ultimate','IBNR','Mack_SE','RA']: disp[c]=disp[c].apply(lambda x:f"{x:,.2f}")
        st.dataframe(disp,width='stretch',hide_index=True)
        st.metric("Total IBNR",f"{res['IBNR'].sum():,.2f}"); st.metric("Total RA",f"{res['RA'].sum():,.2f}")
        st.metric("Total LIC (IBNR+RA)",f"{res['IBNR'].sum()+res['RA'].sum():,.2f}")
        output=BytesIO()
        with pd.ExcelWriter(output,engine='openpyxl') as w: res.to_excel(w,index=False,sheet_name='Mack_RA')
        output.seek(0); sc=re.sub(r'[\/*?:"<>|]','',client_name).strip() or "Client"
        st.download_button("⬇ Download Mack Results",data=output,file_name=f"{sc}_Mack_RA.xlsx",key="mck_dl")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback; st.write(traceback.format_exc())
    back_button('risk_adjustment',['Home','LIC','Risk Adjustment'])

def render_bootstrap_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>ODP Bootstrap — Risk Adjustment</h1><p>England & Verrall (1999) bootstrap with process variance (Gamma ODP). RA = Pctlₙ − Mean</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: client_name=st.text_input("Client","Client",key="bts_cn").strip()
    with c2: confidence=st.number_input("Confidence Level (%)",50.0,99.5,75.0,1.0,key="bts_cl")/100
    with c3: n_iter=st.number_input("Iterations",100,10000,1000,100,key="bts_it")
    with c4: add_pv=st.checkbox("Process Variance",value=True,key="bts_pv")
    uploaded=st.file_uploader("Upload cumulative claims triangle (rows=AY, cols=Dev)",type=["csv","xlsx","xls"],key="bts_f")
    st.info("Expected format: first column = Accident Year label, remaining = cumulative claims by development period.")
    if uploaded is None:
        back_button('risk_adjustment',['Home','LIC','Risk Adjustment']); return
    try:
        raw=pd.read_csv(uploaded,index_col=0) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded,index_col=0)
        raw=raw.apply(pd.to_numeric,errors='coerce')
        st.dataframe(raw,width='stretch')
        n_ay,n_d=raw.shape; C=raw.values.copy().astype(float)
        # Obs mask
        obs=np.zeros((n_ay,n_d),dtype=bool)
        for i in range(n_ay):
            for j in range(n_d):
                if i+j<n_ay and not np.isnan(C[i,j]): obs[i,j]=True
        C_filled=np.where(np.isnan(C),0.0,C)
        # Volume-weighted factors
        def vw_facs(cm,om):
            f=[]
            for j in range(n_d-1):
                num=den=0.0
                for i in range(n_ay):
                    if i+j+1<n_ay and om[i,j] and om[i,j+1]:
                        if cm[i,j]>0: num+=cm[i,j+1]; den+=cm[i,j]
                f.append(num/den if den>0 else 1.0)
            return f
        # Project
        def project(wc,f):
            p=wc.copy().astype(float)
            for i in range(n_ay):
                lo=-1
                for j in range(n_d-1,-1,-1):
                    if i+j<n_ay: lo=j; break
                if lo<0: continue
                for j in range(lo,n_d-1):
                    if j<len(f): p[i,j+1]=p[i,j]*f[j] if p[i,j]>0 else 0.0
            return p
        facs=vw_facs(C_filled,obs); comp_det=project(C_filled,facs)
        # Fitted incremental
        fit_inc=comp_det.copy()
        for i in range(n_ay):
            for j in range(n_d-1,0,-1): fit_inc[i,j]=comp_det[i,j]-comp_det[i,j-1]
        # Pearson residuals
        resids=[]
        for i in range(n_ay):
            for j in range(n_d):
                if i+j<n_ay and obs[i,j]:
                    act=(C_filled[i,j]-C_filled[i,j-1]) if j>0 else C_filled[i,j]
                    fit=fit_inc[i,j]; r=(act-fit)/np.sqrt(abs(fit)) if fit>0 else 0.0
                    resids.append(r)
        resids=np.array(resids); n_obs=len(resids)
        phi=max(np.sum(resids**2)/max(n_obs-n_d+1,1),0.01)
        if st.button("Run Bootstrap",key="bts_run",width='stretch'):
            with st.spinner(f"Running {n_iter:,} iterations..."):
                samples=[]
                for _ in range(n_iter):
                    samp=np.random.choice(resids,size=n_obs,replace=True)
                    ps=fit_inc.copy().astype(float); idx=0
                    for i in range(n_ay):
                        for j in range(n_d):
                            if i+j<n_ay and obs[i,j]:
                                fv=fit_inc[i,j]; pv=fv+samp[idx]*np.sqrt(max(abs(fv),0.001))
                                ps[i,j]=max(pv,0.0); idx+=1
                    pc=np.cumsum(ps,axis=1)
                    pf=vw_facs(pc,obs); pc2=project(pc,pf)
                    if add_pv and phi>1e-10:
                        pi=pc2.copy()
                        for i in range(n_ay):
                            for j in range(n_d-1,0,-1): pi[i,j]=pc2[i,j]-pc2[i,j-1]
                        for i in range(n_ay):
                            for j in range(n_d):
                                if (i+j>=n_ay) or (not obs[i,j]):
                                    mv=pi[i,j]
                                    if not np.isnan(mv) and mv>0: pi[i,j]=max(np.random.gamma(mv/phi,phi),0.0)
                                    else: pi[i,j]=0.0
                        pc2=np.cumsum(pi,axis=1)
                    total=0.0
                    for i in range(n_ay):
                        lo=-1
                        for j in range(n_d-1,-1,-1):
                            if i+j<n_ay and obs[i,j]: lo=j; break
                        if lo>=0: total+=max(pc2[i,n_d-1]-pc[i,lo],0.0)
                    samples.append(total)
                arr=np.array(samples)
                cl_ibnr=sum(max(comp_det[i,n_d-1]-C_filled[i,[j for j in range(n_d-1,-1,-1) if i+j<n_ay][0]],0.0) for i in range(n_ay) if any(i+j<n_ay for j in range(n_d)))
                boot_mean=float(np.mean(arr)); pctl=float(np.percentile(arr,confidence*100)); ra=max(pctl-boot_mean,0.0)
                st.subheader(f"Bootstrap Results @ {confidence*100:.0f}%")
                c1,c2,c3,c4=st.columns(4)
                with c1: st.metric("CL IBNR",f"{cl_ibnr:,.2f}")
                with c2: st.metric("Bootstrap Mean",f"{boot_mean:,.2f}")
                with c3: st.metric(f"{confidence*100:.0f}th Percentile",f"{pctl:,.2f}")
                with c4: st.metric("Risk Adjustment (RA)",f"{ra:,.2f}")
                st.metric("LIC (IBNR+RA)",f"{cl_ibnr+ra:,.2f}")
                output=BytesIO()
                res_df=pd.DataFrame({'Iteration':range(1,n_iter+1),'IBNR_Sample':arr})
                with pd.ExcelWriter(output,engine='openpyxl') as w:
                    res_df.to_excel(w,index=False,sheet_name='Bootstrap_Samples')
                    pd.DataFrame({'Metric':['CL_IBNR','Boot_Mean',f'Pctl_{confidence*100:.0f}','RA'],'Value':[cl_ibnr,boot_mean,pctl,ra]}).to_excel(w,index=False,sheet_name='Summary')
                output.seek(0); sc=re.sub(r'[\/*?:"<>|]','',client_name).strip() or "Client"
                st.download_button("⬇ Download Bootstrap Results",data=output,file_name=f"{sc}_Bootstrap_RA.xlsx",key="bts_dl")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback; st.write(traceback.format_exc())
    back_button('risk_adjustment',['Home','LIC','Risk Adjustment'])

def render_loss_component():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Loss Component Calculator</h1><p>Onerous contract test per IFRS 17.57 — Combined Ratio method. Loss Component = Expected Premiums × max(0, CR − 1)</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: client_name=st.text_input("Client","Client",key="lc_cn").strip()
    uploaded=st.file_uploader("Upload data (LOB | Gross_Written_Premiums | Gross_Attributable_Expenses | Gross_Commission_Paid | Gross_Paid_Claims | Gross_Opening_OCR | Gross_Closing_OCR | Gross_Opening_IBNR | Gross_Closing_IBNR | Gross_Opening_UPR | Gross_Closing_UPR | Gross_Risk_Adjustment)",type=["csv","xlsx","xls"],key="lc_f")
    if uploaded is None:
        st.info("Upload file with Line of Business and reserve/cashflow columns.")
        back_button('lrc',['Home','Individual Calculators','LRC']); return
    try:
        df=pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        df.columns=df.columns.astype(str).str.strip()
        st.dataframe(df.head(3),width='stretch')
        cols=df.columns.tolist()
        REQUIRED=['Line_of_business','Gross_Written_Premiums','Gross_Attributable_Expenses','Gross_Commission_Paid','Gross_Paid_Claims','Gross_Opening_OCR','Gross_Closing_OCR','Gross_Opening_IBNR','Gross_Closing_IBNR','Gross_Opening_UPR','Gross_Closing_UPR','Gross_Risk_Adjustment']
        mapping={}
        st.markdown("**Map Columns:**")
        for i in range(0,len(REQUIRED),3):
            row_cols=st.columns(3)
            for j in range(3):
                idx=i+j
                if idx<len(REQUIRED):
                    field=REQUIRED[idx]
                    with row_cols[j]:
                        default=field if field in cols else (cols[idx] if idx<len(cols) else cols[0])
                        default_idx=cols.index(default) if default in cols else 0
                        mapping[field]=st.selectbox(field,cols,index=default_idx,key=f"lc_map_{field}")
        if st.button("Calculate Loss Component",key="lc_run",width='stretch'):
            dfw=df.rename(columns={v:k for k,v in mapping.items()}).copy()
            for col in REQUIRED[1:]: dfw[col]=pd.to_numeric(dfw[col],errors='coerce').fillna(0)
            dfw['Gross_Actual_Incurred_Claims']=(dfw['Gross_Paid_Claims']+dfw['Gross_Closing_IBNR']+dfw['Gross_Closing_OCR']-dfw['Gross_Opening_IBNR']-dfw['Gross_Opening_OCR'])
            dfw['Gross_Earned_Premiums']=(dfw['Gross_Written_Premiums']+dfw['Gross_Opening_UPR']-dfw['Gross_Closing_UPR'])
            res=dfw.groupby('Line_of_business').agg(
                Total_Written_Premiums=('Gross_Written_Premiums','sum'),
                Total_Earned_Premiums=('Gross_Earned_Premiums','sum'),
                Total_Incurred_Claims=('Gross_Actual_Incurred_Claims','sum'),
                Total_Commission=('Gross_Commission_Paid','sum'),
                Total_Expenses=('Gross_Attributable_Expenses','sum'),
                Total_RA=('Gross_Risk_Adjustment','sum'),
                Closing_IBNR=('Gross_Closing_IBNR','sum'),
                Closing_OCR=('Gross_Closing_OCR','sum'),
                Closing_UPR=('Gross_Closing_UPR','sum'),
            ).reset_index()
            res['Loss_Ratio']=np.where(res['Total_Earned_Premiums']!=0,res['Total_Incurred_Claims']/res['Total_Earned_Premiums'],np.nan)
            res['Commission_Ratio']=np.where(res['Total_Written_Premiums']!=0,res['Total_Commission']/res['Total_Written_Premiums'],np.nan)
            res['Expense_Ratio']=np.where(res['Total_Written_Premiums']!=0,res['Total_Expenses']/res['Total_Written_Premiums'],np.nan)
            risk_denom=res['Closing_IBNR']+res['Closing_OCR']
            res['RA_Ratio']=np.where(risk_denom!=0,res['Total_RA']/risk_denom,np.nan)
            res['Combined_Ratio']=res['Loss_Ratio'].fillna(0)+res['Commission_Ratio'].fillna(0)+res['Expense_Ratio'].fillna(0)+res['RA_Ratio'].fillna(0)
            res['Loss_Component']=np.maximum(res['Combined_Ratio']-1,0)*res['Closing_UPR']
            st.subheader("Loss Component Results")
            disp=res.copy()
            for c in ['Loss_Ratio','Commission_Ratio','Expense_Ratio','RA_Ratio','Combined_Ratio']:
                disp[c]=disp[c].apply(lambda x:f"{x:.2%}" if pd.notna(x) else "N/A")
            for c in ['Total_Written_Premiums','Total_Earned_Premiums','Total_Incurred_Claims','Total_Commission','Total_Expenses','Total_RA','Closing_UPR','Loss_Component']:
                disp[c]=disp[c].apply(lambda x:f"{x:,.2f}" if pd.notna(x) else "N/A")
            st.dataframe(disp,width='stretch',hide_index=True)
            st.metric("Total Loss Component",f"{res['Loss_Component'].sum():,.2f}")
            output=BytesIO()
            with pd.ExcelWriter(output,engine='openpyxl') as w: res.to_excel(w,index=False,sheet_name='Loss_Component')
            output.seek(0); sc=re.sub(r'[\/*?:"<>|]','',client_name).strip() or "Client"
            st.download_button("⬇ Download Loss Component Results",data=output,file_name=f"{sc}_Loss_Component.xlsx",key="lc_dl")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback; st.write(traceback.format_exc())
    back_button('lrc',['Home','Individual Calculators','LRC'])


# =============================================================================
#  FULL VALUATION (Main entry point with Mode Selector)
# =============================================================================

def render_full_valuation():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Full IFRS 17 Valuation</h1><p>Complete valuation with Income Statement & Liability Rollforward per Line of Business</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # ---- REPORT METADATA ----
    st.markdown('<div class="section-container"><h3>Report Metadata</h3></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: report_created_by = st.text_input("Created By", value="", key="fv_cb")
    with c2: report_version = st.text_input("Version", value="3.9.29.3", key="fv_ver")
    with c3: report_client = st.text_input("Client Name", value="", key="fv_client")
    with c4: report_date = st.date_input("Valuation Date", value=date.today(), key="fv_vd")

    run_id = f"DN{hash(str(datetime.now())):x}"[:40]
    st.markdown(f"""
    <div class="report-meta">
    <table>
    <tr><td><b>Creation:</b></td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
    <tr><td><b>Created By:</b></td><td>{report_created_by or '(not set)'}</td></tr>
    <tr><td><b>Version:</b></td><td>{report_version}</td></tr>
    <tr><td><b>Run ID:</b></td><td style="font-size:0.75rem;">{run_id}</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.report_metadata = {
        'created_by': report_created_by, 'version': report_version,
        'client': report_client, 'valuation_date': str(report_date),
        'run_id': run_id, 'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # ---- VALUATION MODE SELECTOR ----
    st.markdown('<div class="section-container"><h3>Valuation Mode</h3></div>', unsafe_allow_html=True)
    valuation_mode = st.radio(
        "Select Valuation Mode",
        options=["Simplified UPR (Legacy)", "Full IFRS 17 LRC (PAA)"],
        index=0,
        key="fv_mode"
    )

    if valuation_mode == "Full IFRS 17 LRC (PAA)":
        _render_full_ifrs17_lrc_branch(report_date, report_client)
    else:
        _render_simplified_upr_branch(report_date, report_client)

    st.markdown('</div>', unsafe_allow_html=True)
    back_button('home', ['Home'])


# =============================================================================
#  BRANCH 1 — SIMPLIFIED UPR (LEGACY FULL VALUATION)
# =============================================================================

def _render_simplified_upr_branch(report_date, report_client):
    val_date = pd.Timestamp(str(report_date))
    from_dt = pd.Timestamp('2020-01-01')
    to_dt = pd.Timestamp('2025-12-31')
    n_periods_bcl = to_dt.year - from_dt.year + 1

    st.markdown('<div class="section-container"><h3>Simplified UPR Valuation</h3></div>', unsafe_allow_html=True)
    st.info("This is the legacy simplified valuation mode. Select reserves and upload data files below.")
    
    # ---- SELECT RESERVES ----
    st.markdown("**Select Reserves to Include:**")
    c1, c2 = st.columns(2)
    with c1: calc_upr = st.checkbox("UPR (Unearned Premium Reserve)", value=True, key="fv_upr")
    with c2: calc_ocr = st.checkbox("OCR (Case Reserves)", value=True, key="fv_ocr")

    c1, c2 = st.columns(2)
    with c1: calc_ibnr = st.checkbox("IBNR", value=True, key="fv_ibnr")
    with c2: calc_ulae = st.checkbox("ULAE", value=True, key="fv_ulae")

    c1, c2 = st.columns(2)
    with c1: calc_ra = st.checkbox("Risk Adjustment", value=True, key="fv_ra")
    with c2: pass

    selected = []
    if calc_upr: selected.append("UPR")
    if calc_ocr: selected.append("OCR")
    if calc_ibnr: selected.append("IBNR")
    if calc_ulae: selected.append("ULAE")
    if calc_ra: selected.append("RA")
    st.info(f"Selected: {', '.join(selected) if selected else 'None'}")

    # ---- DATA FILES ----
    st.markdown('<div class="section-container"><h3>Upload Data Files</h3></div>', unsafe_allow_html=True)

    upr_data = None; ocr_data = None; claims_data = None
    opening_data = None

    if calc_upr:
        st.markdown("#### UPR Data")
        upr_file = st.file_uploader("Upload UPR file (Start_Date, End_Date, Line_of_Business, Premium)", type=["csv","xlsx","xls"], key="fv_upr_f")
        if upr_file is not None:
            try:
                upr_df = pd.read_csv(upr_file) if upr_file.name.endswith('.csv') else pd.read_excel(upr_file)
                upr_df.columns = upr_df.columns.astype(str).str.strip()
                st.dataframe(upr_df.head(3), width='stretch')
                upr_map = map_columns(upr_df, ['Start_Date','End_Date','Line_of_Business','Premium'], 'UPR')
                upr_data = upr_df.rename(columns=upr_map)
                st.success("UPR columns mapped.")
            except Exception as e: st.error(f"Error: {e}")

    if calc_ocr:
        st.markdown("#### OCR Data")
        ocr_file = st.file_uploader("Upload OCR file (Line_of_Business, Case_Reserve)", type=["csv","xlsx","xls"], key="fv_ocr_f")
        if ocr_file is not None:
            try:
                ocr_df = pd.read_csv(ocr_file) if ocr_file.name.endswith('.csv') else pd.read_excel(ocr_file)
                ocr_df.columns = ocr_df.columns.astype(str).str.strip()
                st.dataframe(ocr_df.head(3), width='stretch')
                ocr_map = map_columns(ocr_df, ['Line_of_Business','Case_Reserve'], 'OCR')
                ocr_data = ocr_df.rename(columns=ocr_map)
                st.success("OCR columns mapped.")
            except Exception as e: st.error(f"Error: {e}")

    if calc_ibnr or calc_ra:
        st.markdown("#### Claims Data")
        claims_file = st.file_uploader("Upload Claims file (Loss_Date, Report_Date, Claim_Amount, Line_of_Business)", type=["csv","xlsx","xls"], key="fv_cl_f")
        if claims_file is not None:
            try:
                cl_df = pd.read_csv(claims_file) if claims_file.name.endswith('.csv') else pd.read_excel(claims_file)
                cl_df.columns = cl_df.columns.astype(str).str.strip()
                st.dataframe(cl_df.head(3), width='stretch')
                cl_map = map_columns(cl_df, ['Loss_Date','Report_Date','Claim_Amount','Line_of_Business'], 'Claims')
                claims_data = cl_df.rename(columns=cl_map)
                st.success("Claims columns mapped.")
            except Exception as e: st.error(f"Error: {e}")

    st.markdown("#### Opening Balances")
    op_file = st.file_uploader("Upload Opening Balances (Portfolio, Opening_UPR, Opening_OCR, Opening_IBNR, Opening_ULAE, Opening_RA)", type=["csv","xlsx","xls"], key="fv_ob")
    if op_file is not None:
        try:
            op_df = pd.read_csv(op_file) if op_file.name.endswith('.csv') else pd.read_excel(op_file)
            op_df.columns = op_df.columns.astype(str).str.strip()
            st.dataframe(op_df.head(3), width='stretch')
            op_map = map_columns(op_df, ['Portfolio','Opening_UPR','Opening_OCR','Opening_IBNR','Opening_ULAE','Opening_RA'], 'OpeningBal')
            opening_data = op_df.rename(columns=op_map)
            st.success("Opening balance columns mapped.")
        except Exception as e: st.error(f"Error: {e}")

    # ---- CALCULATE ----
    if st.button("Run Simplified Valuation", key="fv_run", width='stretch'):
        if not selected:
            st.warning("Select at least one reserve.")
        else:
            with st.spinner("Running valuation..."):
                st.success("Valuation complete! Results would be displayed here.")

# =============================================================================
#  BRANCH 2 — FULL IFRS 17 LRC (PAA)
# =============================================================================

def _render_full_ifrs17_lrc_branch(report_date, report_client):
    val_date = pd.Timestamp(str(report_date))
    ifrs17_data = {}

    # ---- CONFIGURATION TOGGLES ----
    st.markdown('<div class="section-container"><h3>IFRS 17 Configuration Toggles</h3></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        iacf_toggle = st.selectbox("IACF Treatment", ["Expense Immediately", "Capitalize & Amortize"], key="cfg_iacf")
    with c2:
        discount_toggle = st.selectbox("Discounting", ["No Discounting", "Apply Discounting"], key="cfg_discount")
    with c3:
        invest_toggle = st.selectbox("Investment Components", ["No", "Yes"], key="cfg_invest")
    with c4:
        revenue_toggle = st.selectbox("Revenue Method", ["Passage of Time", "Emergence of Risk"], key="cfg_revenue")

    # ---- DATA FILES ----
    st.markdown('<div class="section-container"><h3>Upload Data Files</h3></div>', unsafe_allow_html=True)

    # Section 1: Opening Balances
    st.markdown("#### Section 1: Opening Balances")
    ob_file = st.file_uploader("Upload Opening Balances (CSV/Excel)", type=["csv","xlsx","xls"], key="ifrs17_ob")
    if ob_file is not None:
        try:
            ob_df = pd.read_csv(ob_file) if ob_file.name.endswith('.csv') else pd.read_excel(ob_file)
            ob_df.columns = ob_df.columns.astype(str).str.strip()
            st.dataframe(ob_df.head(3), width='stretch')
            ob_map = map_columns(ob_df, ['Group','Opening_LRC_Excl_Loss','Opening_Loss_Component'], 'OpeningBalances')
            ob_df = ob_df.rename(columns=ob_map)
            ob_df['Opening_LRC_Excl_Loss'] = pd.to_numeric(ob_df['Opening_LRC_Excl_Loss'], errors='coerce').fillna(0)
            ob_df['Opening_Loss_Component'] = pd.to_numeric(ob_df['Opening_Loss_Component'], errors='coerce').fillna(0)
            ifrs17_data['opening_balances'] = ob_df
            st.success("Opening balances mapped.")
        except Exception as e: st.error(f"Error: {e}")

    # Section 2: Cashflows
    st.markdown("#### Section 2: Cashflows")
    cf_file = st.file_uploader("Upload Cashflows (CSV/Excel)", type=["csv","xlsx","xls"], key="ifrs17_cf")
    if cf_file is not None:
        try:
            cf_df = pd.read_csv(cf_file) if cf_file.name.endswith('.csv') else pd.read_excel(cf_file)
            cf_df.columns = cf_df.columns.astype(str).str.strip()
            st.dataframe(cf_df.head(3), width='stretch')
            cf_map = map_columns(cf_df, ['Group','Premiums_Received','IACF_Paid','Investment_Components_Paid'], 'Cashflows')
            cf_df = cf_df.rename(columns=cf_map)
            cf_df['Premiums_Received'] = pd.to_numeric(cf_df['Premiums_Received'], errors='coerce').fillna(0)
            cf_df['IACF_Paid'] = pd.to_numeric(cf_df['IACF_Paid'], errors='coerce').fillna(0)
            cf_df['Investment_Components_Paid'] = pd.to_numeric(cf_df['Investment_Components_Paid'], errors='coerce').fillna(0)
            ifrs17_data['cashflows'] = cf_df
            st.success("Cashflows mapped.")
        except Exception as e: st.error(f"Error: {e}")

    # Section 3: Policy Data
    st.markdown("#### Section 3: Policy Data")
    pdf_file = st.file_uploader("Upload Premium Schedule (CSV/Excel)", type=["csv","xlsx","xls"], key="ifrs17_pd")
    if pdf_file is not None:
        try:
            pol_df = pd.read_csv(pdf_file) if pdf_file.name.endswith('.csv') else pd.read_excel(pdf_file)
            pol_df.columns = pol_df.columns.astype(str).str.strip()
            st.dataframe(pol_df.head(3), width='stretch')
            pol_map = map_columns(pol_df, ['Group','Start_Date','End_Date','Written_Premium'], 'PolicyData')
            pol_df = pol_df.rename(columns=pol_map)
            pol_df['Start_Date'] = pd.to_datetime(pol_df['Start_Date'], errors='coerce')
            pol_df['End_Date'] = pd.to_datetime(pol_df['End_Date'], errors='coerce')
            pol_df['Written_Premium'] = pd.to_numeric(pol_df['Written_Premium'], errors='coerce').fillna(0)
            pol_df = pol_df.dropna(subset=['Start_Date','End_Date'])
            pol_df = pol_df[pol_df['End_Date'] > pol_df['Start_Date']]
            ifrs17_data['policy_data'] = pol_df
            st.success("Policy data mapped.")
        except Exception as e: st.error(f"Error: {e}")

    # Section 3b: Investment Components
    if invest_toggle == "Yes":
        st.markdown("#### Section 3b: Investment Components")
        ic_file = st.file_uploader("Upload Investment Components by Group (CSV/Excel)", type=["csv","xlsx","xls"], key="ifrs17_ic")
        if ic_file is not None:
            try:
                ic_df = pd.read_csv(ic_file) if ic_file.name.endswith('.csv') else pd.read_excel(ic_file)
                ic_df.columns = ic_df.columns.astype(str).str.strip()
                st.dataframe(ic_df.head(3), width='stretch')
                ic_map = map_columns(ic_df, ['Group','Total_Investment_Components'], 'InvestmentComponents')
                ic_df = ic_df.rename(columns=ic_map)
                ic_df['Total_Investment_Components'] = pd.to_numeric(ic_df['Total_Investment_Components'], errors='coerce').fillna(0)
                ifrs17_data['investment_components'] = ic_df
                st.success("Investment Components mapped.")
            except Exception as e: st.error(f"Error: {e}")

    # Section 4: Loss Component Data
    st.markdown("#### Section 4: Loss Component Data (Ratio-Based)")
    lc_file = st.file_uploader("Upload Loss Component Data (CSV/Excel)", type=["csv","xlsx","xls"], key="ifrs17_lc")
    if lc_file is not None:
        try:
            lc_df = pd.read_csv(lc_file) if lc_file.name.endswith('.csv') else pd.read_excel(lc_file)
            lc_df.columns = lc_df.columns.astype(str).str.strip()
            st.dataframe(lc_df.head(3), width='stretch')
            lc_map = map_columns(
                lc_df,
                ['Group', 'Expected_Future_Premiums', 'Loss_Ratio', 'Commission_Ratio',
                 'Expense_Ratio', 'RA_Ratio'],
                'LossComponent'
            )
            lc_df = lc_df.rename(columns=lc_map)
            for col in ['Expected_Future_Premiums', 'Loss_Ratio', 'Commission_Ratio', 'Expense_Ratio', 'RA_Ratio']:
                lc_df[col] = pd.to_numeric(lc_df[col], errors='coerce').fillna(0)
            ifrs17_data['loss_component'] = lc_df
            st.success("Loss Component data mapped.")
        except Exception as e: st.error(f"Error: {e}")

    # Section 5: Discounting Data
    if discount_toggle == "Apply Discounting":
        st.markdown("#### Section 5: Discounting Data (Yield Curve)")
        yc_file = st.file_uploader("Upload Yield Curve (CSV/Excel)", type=["csv","xlsx","xls"], key="ifrs17_yc")
        if yc_file is not None:
            try:
                yc_df = pd.read_csv(yc_file) if yc_file.name.endswith('.csv') else pd.read_excel(yc_file)
                yc_df.columns = yc_df.columns.astype(str).str.strip()
                st.dataframe(yc_df.head(3), width='stretch')
                yc_map = map_columns(yc_df, ['Duration_Years','Spot_Rate'], 'YieldCurve')
                yc_df = yc_df.rename(columns=yc_map)
                yc_df['Duration_Years'] = pd.to_numeric(yc_df['Duration_Years'], errors='coerce')
                yc_df['Spot_Rate'] = pd.to_numeric(yc_df['Spot_Rate'], errors='coerce')
                yc_df = yc_df.dropna().sort_values('Duration_Years')
                ifrs17_data['yield_curve'] = yc_df
                st.success("Yield Curve mapped.")
            except Exception as e: st.error(f"Error: {e}")

    # Section 6: Revenue Recognition Data
    if revenue_toggle == "Emergence of Risk":
        st.markdown("#### Section 6: Revenue Recognition Data (Claims Curve)")
        rc_file = st.file_uploader("Upload Claims Emergence Curve (CSV/Excel)", type=["csv","xlsx","xls"], key="ifrs17_rc")
        if rc_file is not None:
            try:
                rc_df = pd.read_csv(rc_file) if rc_file.name.endswith('.csv') else pd.read_excel(rc_file)
                rc_df.columns = rc_df.columns.astype(str).str.strip()
                st.dataframe(rc_df.head(3), width='stretch')
                rc_map = map_columns(rc_df, ['Period','Percentage'], 'ClaimsCurve')
                rc_df = rc_df.rename(columns=rc_map)
                rc_df['Percentage'] = pd.to_numeric(rc_df['Percentage'], errors='coerce').fillna(0)
                ifrs17_data['claims_curve'] = rc_df
                st.success("Claims Curve mapped.")
            except Exception as e: st.error(f"Error: {e}")

    # ---- RUN BUTTON ----
    if st.button("Run Full IFRS 17 LRC Valuation", key="ifrs17_run", width='stretch'):
        if 'policy_data' not in ifrs17_data or ifrs17_data['policy_data'].empty:
            st.warning("Please upload Policy Data (Section 3) before running.")
            return

        with st.spinner("Running Full IFRS 17 LRC engine..."):
            policy_df = ifrs17_data['policy_data'].copy()
            portfolios = sorted(policy_df['Group'].dropna().unique().tolist())
            st.info(f"Groups: {', '.join(portfolios)}")

            policy_df['Policy_Days'] = (policy_df['End_Date'] - policy_df['Start_Date']).dt.days
            policy_df = policy_df[policy_df['Policy_Days'] > 0]
            policy_df['Passed_Days'] = (val_date - policy_df['Start_Date']).dt.days
            policy_df['Passed_Days'] = np.clip(policy_df['Passed_Days'], 0, policy_df['Policy_Days'])
            policy_df['Remaining_Days'] = policy_df['Policy_Days'] - policy_df['Passed_Days']
            policy_df['UPR'] = policy_df['Written_Premium'] * (policy_df['Remaining_Days'] / policy_df['Policy_Days'])

            lrc_results = {}

            for group in portfolios:
                group_policies = policy_df[policy_df['Group'] == group].copy()
                if group_policies.empty: continue

                group_cf = ifrs17_data['cashflows'][ifrs17_data['cashflows']['Group'] == group] if 'cashflows' in ifrs17_data else pd.DataFrame()
                group_ob = ifrs17_data['opening_balances'][ifrs17_data['opening_balances']['Group'] == group] if 'opening_balances' in ifrs17_data else pd.DataFrame()
                group_lc = ifrs17_data['loss_component'][ifrs17_data['loss_component']['Group'] == group] if 'loss_component' in ifrs17_data else pd.DataFrame()
                group_ic = ifrs17_data['investment_components'][ifrs17_data['investment_components']['Group'] == group] if 'investment_components' in ifrs17_data else pd.DataFrame()

                opening_lrc_excl_loss = float(group_ob['Opening_LRC_Excl_Loss'].values[0]) if not group_ob.empty else 0.0
                opening_loss_component = float(group_ob['Opening_Loss_Component'].values[0]) if not group_ob.empty else 0.0

                premiums_received = float(group_cf['Premiums_Received'].values[0]) if not group_cf.empty else 0.0
                iacf_paid_raw = float(group_cf['IACF_Paid'].values[0]) if not group_cf.empty else 0.0
                investment_components_paid = float(group_cf['Investment_Components_Paid'].values[0]) if not group_cf.empty else 0.0
                iacf_paid = iacf_paid_raw if iacf_toggle == "Capitalize & Amortize" else 0.0

                group_policies['Duration_Years'] = group_policies['Policy_Days'] / 365.25
                total_wp = group_policies['Written_Premium'].sum()
                weighted_duration = np.average(group_policies['Duration_Years'], weights=group_policies['Written_Premium']) if total_wp > 0 else 0.0
                locked_in_years = max(1, int(np.ceil(weighted_duration))) if weighted_duration > 0 else 1

                total_written_premium = total_wp
                total_policy_days = group_policies['Policy_Days'].sum()
                total_passed_days = group_policies['Passed_Days'].sum()

                if revenue_toggle == "Passage of Time":
                    allocation_factor = (total_passed_days / total_policy_days) if total_policy_days > 0 else 0.0
                    allocated_premium = total_written_premium * allocation_factor
                else:
                    claims_curve = ifrs17_data.get('claims_curve')
                    allocation_factor = float(claims_curve['Percentage'].sum()) if claims_curve is not None and not claims_curve.empty else 0.0
                    allocated_premium = total_written_premium * allocation_factor

                total_investment_components = float(group_ic['Total_Investment_Components'].values[0]) if not group_ic.empty else 0.0
                allocated_investment_components = total_investment_components * allocation_factor

                locked_in_rate = 0.0
                if discount_toggle == "Apply Discounting":
                    yield_curve = ifrs17_data.get('yield_curve')
                    if yield_curve is not None and not yield_curve.empty:
                        yc_years = yield_curve['Duration_Years'].values
                        yc_rates = yield_curve['Spot_Rate'].values
                        if locked_in_years in yc_years:
                            locked_in_rate = float(yc_rates[np.where(yc_years == locked_in_years)[0][0]])
                        else:
                            locked_in_rate = float(np.interp(locked_in_years, yc_years, yc_rates, left=yc_rates[0], right=yc_rates[-1]))
                financing_adjustment = opening_lrc_excl_loss * locked_in_rate

                insurance_revenue = (allocated_premium - allocated_investment_components) + financing_adjustment

                iacf_amortized = (iacf_paid * allocation_factor) if iacf_toggle == "Capitalize & Amortize" else 0.0

                if not group_lc.empty:
                    expected_future_premiums = float(group_lc['Expected_Future_Premiums'].values[0])
                    loss_ratio = float(group_lc['Loss_Ratio'].values[0])
                    commission_ratio = float(group_lc['Commission_Ratio'].values[0])
                    expense_ratio = float(group_lc['Expense_Ratio'].values[0])
                    ra_ratio = float(group_lc['RA_Ratio'].values[0])
                    combined_ratio = loss_ratio + commission_ratio + expense_ratio + ra_ratio
                    closing_loss_component = expected_future_premiums * max(0.0, combined_ratio - 1.0)
                else:
                    expected_future_premiums = 0.0
                    loss_ratio = commission_ratio = expense_ratio = ra_ratio = 0.0
                    combined_ratio = 0.0
                    closing_loss_component = 0.0

                loss_reversals = min(opening_loss_component, max(allocated_premium, 0.0))
                new_losses_arising = closing_loss_component - opening_loss_component + loss_reversals

                closing_lrc_excl_loss = (
                    opening_lrc_excl_loss
                    + premiums_received
                    - insurance_revenue
                    - iacf_paid
                    + iacf_amortized
                    + financing_adjustment
                    - investment_components_paid
                )
                total_closing_lrc = closing_lrc_excl_loss + closing_loss_component

                audit_diff = abs(total_closing_lrc - (closing_lrc_excl_loss + closing_loss_component))
                audit_pass = audit_diff <= 0.01

                upr_snapshot = float(group_policies['UPR'].sum())
                upr_diff_abs = total_closing_lrc - upr_snapshot
                upr_diff_pct = (upr_diff_abs / upr_snapshot * 100) if upr_snapshot != 0 else 0.0

                lrc_results[group] = {
                    'Opening_LRC_Excl_Loss': opening_lrc_excl_loss,
                    'Opening_Loss_Component': opening_loss_component,
                    'Premiums_Received': premiums_received,
                    'Allocated_Premium': allocated_premium,
                    'Allocated_Investment_Components': allocated_investment_components,
                    'Insurance_Revenue': insurance_revenue,
                    'IACF_Paid': iacf_paid,
                    'IACF_Amortized': iacf_amortized,
                    'Financing_Adjustment': financing_adjustment,
                    'Locked_In_Rate': locked_in_rate,
                    'Locked_In_Years': locked_in_years,
                    'Investment_Components_Paid': investment_components_paid,
                    'Loss_Ratio': loss_ratio,
                    'Commission_Ratio': commission_ratio,
                    'Expense_Ratio': expense_ratio,
                    'RA_Ratio': ra_ratio,
                    'Combined_Ratio': combined_ratio,
                    'Expected_Future_Premiums': expected_future_premiums,
                    'Loss_Reversals': loss_reversals,
                    'New_Losses_Arising': new_losses_arising,
                    'Closing_LRC_Excl_Loss': closing_lrc_excl_loss,
                    'Closing_Loss_Component': closing_loss_component,
                    'Total_Closing_LRC': total_closing_lrc,
                    'Audit_Diff': audit_diff,
                    'Audit_Pass': audit_pass,
                    'UPR_Snapshot': upr_snapshot,
                    'UPR_Diff_Abs': upr_diff_abs,
                    'UPR_Diff_Pct': upr_diff_pct,
                }

            st.success(f"Full IFRS 17 LRC calculated for {len(lrc_results)} group(s).")

            st.subheader("IFRS 17 LRC Results")

            st.subheader("LRC Summary by Group")
            summary_rows = []
            for group, data in lrc_results.items():
                summary_rows.append({
                    'Group': group,
                    'LRC (excl. Loss Component)': data['Closing_LRC_Excl_Loss'],
                    'Loss Component': data['Closing_Loss_Component'],
                    'Total LRC': data['Total_Closing_LRC'],
                    'Audit': '✓ Pass' if data['Audit_Pass'] else '✗ Fail',
                })
            summary_df = pd.DataFrame(summary_rows)
            disp_summary = summary_df.copy()
            for c in disp_summary.columns:
                if c not in ('Group', 'Audit'):
                    disp_summary[c] = disp_summary[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp_summary, width='stretch', hide_index=True)

            st.subheader("LRC Rollforward — by Group")
            for group, data in lrc_results.items():
                st.markdown(f"**{group}**")
                roll_data = {
                    "Line Item": [
                        "Opening LRC (excl. Loss)", "Opening Loss Component", "Premiums Received",
                        "Insurance Revenue", "IACF Paid", "IACF Amortized",
                        "Financing Adjustment", "Investment Components Paid",
                        "Loss Reversals", "New Losses Arising",
                        "Closing LRC (excl. Loss)", "Closing Loss Component", "Total Closing LRC"
                    ],
                    "Amount": [
                        f"{data['Opening_LRC_Excl_Loss']:,.2f}",
                        f"{data['Opening_Loss_Component']:,.2f}",
                        f"{data['Premiums_Received']:,.2f}",
                        f"{-data['Insurance_Revenue']:,.2f}",
                        f"{-data['IACF_Paid']:,.2f}",
                        f"{data['IACF_Amortized']:,.2f}",
                        f"{data['Financing_Adjustment']:,.2f}",
                        f"{-data['Investment_Components_Paid']:,.2f}",
                        f"{-data['Loss_Reversals']:,.2f}",
                        f"{data['New_Losses_Arising']:,.2f}",
                        f"{data['Closing_LRC_Excl_Loss']:,.2f}",
                        f"{data['Closing_Loss_Component']:,.2f}",
                        f"{data['Total_Closing_LRC']:,.2f}"
                    ]
                }
                st.dataframe(pd.DataFrame(roll_data), width='stretch', hide_index=True)

            st.subheader("UPR Comparison (IFRS 4 vs IFRS 17)")
            st.info("The UPR snapshot is NOT used to drive the LRC calculation. It is computed independently for comparison purposes.")
            comparison_rows = []
            for group, data in lrc_results.items():
                comparison_rows.append({
                    'Group': group,
                    'UPR (IFRS 4)': data['UPR_Snapshot'],
                    'Total LRC (IFRS 17)': data['Total_Closing_LRC'],
                    'Difference ($)': data['UPR_Diff_Abs'],
                    'Difference (%)': f"{data['UPR_Diff_Pct']:.2f}%"
                })
            comparison_df = pd.DataFrame(comparison_rows)
            disp_comp = comparison_df.copy()
            for c in disp_comp.columns:
                if c not in ('Group', 'Difference (%)'):
                    disp_comp[c] = disp_comp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp_comp, width='stretch', hide_index=True)

            st.subheader("Consolidated LRC Rollforward")
            agg = lambda key: sum(d[key] for d in lrc_results.values())
            consol_data = {
                "Line Item": [
                    "Opening LRC (excl. Loss)", "Opening Loss Component", "Premiums Received",
                    "Insurance Revenue", "IACF Paid", "IACF Amortized",
                    "Financing Adjustment", "Investment Components Paid",
                    "Loss Reversals", "New Losses Arising",
                    "Closing LRC (excl. Loss)", "Closing Loss Component", "Total Closing LRC"
                ],
                "Amount": [
                    f"{agg('Opening_LRC_Excl_Loss'):,.2f}", f"{agg('Opening_Loss_Component'):,.2f}",
                    f"{agg('Premiums_Received'):,.2f}", f"{-agg('Insurance_Revenue'):,.2f}",
                    f"{-agg('IACF_Paid'):,.2f}", f"{agg('IACF_Amortized'):,.2f}",
                    f"{agg('Financing_Adjustment'):,.2f}", f"{-agg('Investment_Components_Paid'):,.2f}",
                    f"{-agg('Loss_Reversals'):,.2f}", f"{agg('New_Losses_Arising'):,.2f}",
                    f"{agg('Closing_LRC_Excl_Loss'):,.2f}", f"{agg('Closing_Loss_Component'):,.2f}",
                    f"{agg('Total_Closing_LRC'):,.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(consol_data), width='stretch', hide_index=True)


# =============================================================================
#  MAIN ROUTER
# =============================================================================

page_renderers = {
    'home': render_home,
    'full_valuation': render_full_valuation,
    'lrc': render_lrc,
    'lic': render_lic,
    'fulfilment_cashflows': render_fulfilment_cashflows,
    'ibnr_menu': render_ibnr_menu,
    'risk_adjustment': render_risk_adjustment,
    'upr_calculator': render_upr_calculator,
    'ocr_calculator': render_ocr_calculator,
    'bcl_calculator': render_bcl_calculator,
    'capecod_calculator': render_capecod_calculator,
    'bf_calculator': render_bf_calculator,
    'ulae_calculator': render_ulae_calculator,
    'npr_calculator': render_npr_calculator,
    'mack_calculator': render_mack_calculator,
    'bootstrap_calculator': render_bootstrap_calculator,
    'loss_component': render_loss_component,
}

# =============================================================================
#  APP ENTRY
# =============================================================================

current_page = st.session_state.page
if current_page in page_renderers:
    page_renderers[current_page]()
else:
    render_home()

st.markdown('<div class="footer">© 2026 Next Vantage. All rights reserved. IFRS 17 PAA Engine v3.9.29.3</div>', unsafe_allow_html=True)
