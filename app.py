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
    with
