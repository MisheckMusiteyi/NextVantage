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
# Note: Using wildcard (*) to ignore casing on the filename
full_engine = import_file_glob("Full_Valuation/*LRC_IFRS17.py")


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
    st.error(f"❌ Critical Error: Could not find `full_LRC_IFRS17.py`.")
    st.write(f"Python searched inside: `{os.path.join(BASE_DIR, 'Full_Valuation')}`")
    st.write("HINT: Make sure the file exists and matches the pattern `*LRC_IFRS17.py` (casing does not matter).")
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
    methods = [("BCL", "bcl_calculator"), ("Cape Cod", "capecod_calculator"), ("BF", "bf_calculator"), ("Percentage", "percentage_calculator")]
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
    st.markdown('<div class="hero"><h1>Reinsurance Non-Performance Risk (NPR)</h1><p>IFRS 17 Para 63(e) — NPR = PD × Reinsurer Share × Ceded LIC</p></div
