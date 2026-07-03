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
full_engine = import_file_glob("Full_Valuation/*.py")
if full_engine is None:
    full_engine = import_file_glob("Full_Valuation/*IFRS17*.py")


# =============================================================================
#  CRITICAL STARTUP CHECKS
# =============================================================================

if upr_engine is None:
    st.error(f"❌ Critical Error: Could not find `upr_engine.py`.")
    st.write(f"Python searched inside: `{os.path.join(BASE_DIR, 'LRC_Calculators')}`")
    st.stop()

if ocr_engine is None:
    st.error(f"❌ Critical Error: Could not find `ocr_engine.py`.")
    st.write(f"Python searched inside: `{os.path.join(BASE_DIR, 'LIC_Calculators/FCF_Calculators/OCR_Calculators')}`")
    st.stop()


# =============================================================================
#  DATETIME UTILITIES
# =============================================================================

def _parse_dates(series):
    try:
        return pd.to_datetime(series.astype(str), errors='coerce')
    except Exception:
        return pd.to_datetime(series, errors='coerce')

def _date_filter(df, col, from_date, to_date):
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors='coerce')
    fd = pd.Timestamp(from_date)
    td = pd.Timestamp(to_date)
    return df[(df[col] >= fd) & (df[col] <= td)]

def periods_per_year(grain): 
    return {"Y": 1, "Q": 4, "M": 12}[grain]

# =============================================================================
#  UI HELPERS FOR INFLATION / DISCOUNTING
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

def show_breadcrumb():
    if len(st.session_state.breadcrumb) > 1 or st.session_state.breadcrumb[0] != 'Home':
        bc = " > ".join([f"<span>{b}</span>" for b in st.session_state.breadcrumb])
        st.markdown(f'<div class="breadcrumb">{bc}</div>', unsafe_allow_html=True)

def back_button(target_page, target_breadcrumb):
    st.markdown("<br>", unsafe_allow_html=True)
    current = st.session_state.page
    if st.button("Back", key=f"back_{current}_to_{target_page}"):
        navigate_to(target_page, target_breadcrumb)
        st.rerun()

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
    # All 4 IBNR methods including Percentage
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
#  INDIVIDUAL CALCULATOR FUNCTIONS
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
#  PERCENTAGE IBNR CALCULATOR
# =============================================================================

def render_percentage_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Percentage IBNR Calculator</h1><p>Simple IBNR = Amount × IBNR Percentage</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1: client_name = st.text_input("Client", "Client", key="pct_cn").strip()
    with c2: ibnr_pct = st.number_input("IBNR Percentage (%)", 0.0, 100.0, 10.0, 0.5, key="pct_pct") / 100
    
    c1, c2 = st.columns(2)
    with c1: from_date = st.date_input("From Date", date(2020,1,1), key="pct_fd")
    with c2: to_date = st.date_input("To Date", date(2025,12,31), key="pct_td")
    
    uploaded = st.file_uploader("Upload claims file (CSV/Excel)", type=["csv","xlsx","xls"], key="pct_f")
    if uploaded is None:
        st.info("Upload a claims file with Date, Line of Business, and Amount columns.")
        back_button('ibnr_menu', ['Home','LIC','Fulfilment Cashflows','IBNR Methods'])
        return
    
    try:
        df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        df.columns = df.columns.astype(str).str.strip()
        st.dataframe(df.head(3), width='stretch')
        cols = df.columns.tolist()
        
        c1, c2 = st.columns(2)
        with c1: date_col = st.selectbox("Date Column", cols, key="pct_date")
        with c2: lob_col = st.selectbox("Line of Business", cols, key="pct_lob")
        
        # Multiple amount columns
        amount_candidates = [c for c in cols if c not in [date_col, lob_col] and pd.api.types.is_numeric_dtype(df[c])]
        amount_cols = st.multiselect("Select Amount Column(s)", amount_candidates, key="pct_amt")
        
        if not amount_cols:
            st.warning("Please select at least one Amount column.")
            return
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce').astype('datetime64[ns]')
        df = df.dropna(subset=[date_col])
        from_dt = pd.Timestamp(str(from_date)); to_dt = pd.Timestamp(str(to_date))
        df = _date_filter(df, date_col, from_date, to_date)
        
        if st.button("Calculate Percentage IBNR", key="pct_run", width='stretch'):
            if ibnr_pct is not None:
                summary_df, grand_total = ibnr_pct.calculate_percentage_ibnr(
                    df=df,
                    date_col=date_col,
                    lob_col=lob_col,
                    amount_cols=amount_cols,
                    from_date=from_dt,
                    to_date=to_dt,
                    ibnr_pct=ibnr_pct
                )
                
                st.subheader("Percentage IBNR Results")
                disp = summary_df.copy()
                for c in disp.columns:
                    if c not in [lob_col]:
                        disp[c] = disp[c].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
                st.dataframe(disp, width='stretch', hide_index=True)
                st.metric("Total IBNR", f"{grand_total:,.2f}")
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as w:
                    summary_df.to_excel(w, index=False, sheet_name='Percentage_IBNR')
                output.seek(0)
                sc = re.sub(r'[\\/*?:"<>|]','',client_name).strip() or "Client"
                st.download_button("⬇ Download Percentage IBNR Results", data=output, file_name=f"{sc}_Percentage_IBNR.xlsx", key="pct_dl")
                
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback; st.write(traceback.format_exc())
    
    back_button('ibnr_menu', ['Home','LIC','Fulfilment Cashflows','IBNR Methods'])


# =============================================================================
#  BCL, CAPE COD, BF CALCULATORS (with all LDF methods)
# =============================================================================

def render_bcl_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Basic Chain Ladder (BCL) — IBNR Calculator</h1><p>All LDF Methods: Vol-Weighted, Simple Avg, Geometric, Medial, Linear Regression, Weighted Last 3</p></div>', unsafe_allow_html=True)
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

        st.markdown("#### Inflation & Discounting Adjustments")
        c1, c2 = st.columns(2)
        with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key="bcl_inf")
        with c2: use_discounting = st.checkbox("Apply Discounting", key="bcl_disc")

        cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
        if use_inflation:
            cum_inflation, per_period_rates = load_inflation_data_ui(grain_code, ppy)
        if use_discounting:
            spot_rates, flat_rate = load_discounting_data_ui(grain_code, ppy)

        if st.button("Calculate BCL IBNR", key="bcl_run", width='stretch'):
            lobs = sorted(df[lob_col].dropna().unique())
            n_periods = (to_date.year - from_date.year) * ppy + 1
            
            st.subheader("LDF Selection - All Methods")
            st.info("Tail factor is hardcoded to 1.000 (fully developed).")
            
            if engine_utils is not None and ibnr_bcl is not None:
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

                # Recommended LDF based on lowest CV
                rec_method = "volume_weighted"
                min_cv = float('inf')
                for method in ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"]:
                    factors = all_ldfs[method]
                    if len(factors) >= 3:
                        cv = np.std(factors[:3]) / np.mean(factors[:3]) if np.mean(factors[:3]) > 0 else float('inf')
                        if cv < min_cv:
                            min_cv = cv
                            rec_method = method
                st.info(f"**Recommended LDF Method:** {rec_method.replace('_', ' ').title()} (Lowest CV: {min_cv:.2%})")

                selected_method = st.selectbox(
                    "Select LDF Method",
                    ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"],
                    index=0,
                    key="bcl_ldf_method"
                )
            else:
                st.error("Could not load engine utilities.")
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
    st.markdown('<div class="hero"><h1>Cape Cod IBNR</h1><p>All LDF Methods: Vol-Weighted, Simple Avg, Geometric, Medial, Linear Regression, Weighted Last 3</p></div>', unsafe_allow_html=True)
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

        if st.button("Calculate Cape Cod IBNR", key="cc_run", width='stretch'):
            lobs=sorted(df[lob_col].dropna().unique())
            n = (to_date.year - from_date.year) + 1
            
            st.subheader("LDF Selection - All Methods")
            st.info("Tail factor is hardcoded to 1.000 (fully developed).")
            if engine_utils is not None and ibnr_cc is not None:
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

                rec_method = "volume_weighted"
                min_cv = float('inf')
                for method in ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"]:
                    factors = all_ldfs[method]
                    if len(factors) >= 3:
                        cv = np.std(factors[:3]) / np.mean(factors[:3]) if np.mean(factors[:3]) > 0 else float('inf')
                        if cv < min_cv:
                            min_cv = cv
                            rec_method = method
                st.info(f"**Recommended LDF Method:** {rec_method.replace('_', ' ').title()} (Lowest CV: {min_cv:.2%})")

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
    st.markdown('<div class="hero"><h1>Bornhuetter-Ferguson (BF) IBNR</h1><p>All LDF Methods: Vol-Weighted, Simple Avg, Geometric, Medial, Linear Regression, Weighted Last 3</p></div>', unsafe_allow_html=True)
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

        if st.button("Calculate BF IBNR", key="bf_run", width='stretch'):
            n = (to_date.year - from_date.year) + 1

            st.subheader("LDF Selection - All Methods")
            st.info("Tail factor is hardcoded to 1.000 (fully developed).")
            if engine_utils is not None and ibnr_bf is not None:
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

                rec_method = "volume_weighted"
                min_cv = float('inf')
                for method in ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"]:
                    factors = all_ldfs[method]
                    if len(factors) >= 3:
                        cv = np.std(factors[:3]) / np.mean(factors[:3]) if np.mean(factors[:3]) > 0 else float('inf')
                        if cv < min_cv:
                            min_cv = cv
                            rec_method = method
                st.info(f"**Recommended LDF Method:** {rec_method.replace('_', ' ').title()} (Lowest CV: {min_cv:.2%})")

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
                    prems = [1] * n

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


# =============================================================================
#  ULAE, NPR, MACK, BOOTSTRAP, LOSS COMPONENT CALCULATORS
# =============================================================================

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
        facs=[]
        for j in range(m-1):
            num=sum(C[i,j+1] for i in range(n-j-1) if not np.isnan(C[i,j+1]) and not np.isnan(C[i,j]))
            den=sum(C[i,j]   for i in range(n-j-1) if not np.isnan(C[i,j+1]) and not np.isnan(C[i,j]))
            facs.append(num/den if den>0 else 1.0)
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
        if len(sigmas)>=2 and sigmas[-2]>0:
            sigmas.append(min(sigmas[-1]**2/sigmas[-2],sigmas[-2],sigmas[-1]))
        else:
            sigmas.append(sigmas[-1] if sigmas else 0.0)
        proj=C.copy()
        for i in range(n):
            lo=-1
            for j in range(m-1,-1,-1):
                if not np.isnan(C[i,j]): lo=j; break
            if lo<0: continue
            for j in range(lo,m-1):
                if j<len(facs): proj[i,j+1]=proj[i,j]*facs[j] if not np.isnan(proj[i,j]) else np.nan
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
        obs=np.zeros((n_ay,n_d),dtype=bool)
        for i in range(n_ay):
            for j in range(n_d):
                if i+j<n_ay and not np.isnan(C[i,j]): obs[i,j]=True
        C_filled=np.where(np.isnan(C),0.0,C)
        def vw_facs(cm,om):
            f=[]
            for j in range(n_d-1):
                num=den=0.0
                for i in range(n_ay):
                    if i+j+1<n_ay and om[i,j] and om[i,j+1]:
                        if cm[i,j]>0: num+=cm[i,j+1]; den+=cm[i,j]
                f.append(num/den if den>0 else 1.0)
            return f
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
        fit_inc=comp_det.copy()
        for i in range(n_ay):
            for j in range(n_d-1,0,-1): fit_inc[i,j]=comp_det[i,j]-comp_det[i,j-1]
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
#  FULL VALUATION (simplified for now)
# =============================================================================

def render_full_valuation():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Full IFRS 17 Valuation</h1><p>Complete valuation with Income Statement & Liability Rollforward per Line of Business</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.info("Full Valuation module is under development. Please use the individual calculators for now.")
    
    back_button('home', ['Home'])


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
    'percentage_calculator': render_percentage_calculator,
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
