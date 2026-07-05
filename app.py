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
    if optimize: df = _optimize_dtypes(df)
    return df

@st.cache_data(show_spinner=False)
def read_uploaded_triangle(uploaded_file):
    name = uploaded_file.name
    df = pd.read_csv(uploaded_file, index_col=0) if name.endswith('.csv') else pd.read_excel(uploaded_file, index_col=0)
    return df.apply(pd.to_numeric, errors='coerce')

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
        except Exception as e: st.error(f"Inflation data error: {e}")
    return cum_inflation, per_period_rates

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
            except Exception as e: st.error(f"Yield curve error: {e}")
    elif disc_method == "Single Flat Rate":
        flat_rate = st.number_input("Annual Discount Rate (%)", 0.0, 50.0, 5.0, 0.5, key=f"flat_{page_key}") / 100.0
    return spot_rates, flat_rate


# =============================================================================
#  STREAMLIT CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Next Vantage Actuarial Toolkit", layout="wide", initial_sidebar_state="collapsed")
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
#  NAVIGATION PAGES (unchanged)
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
                        navigate_to(page, ['Home', 'LIC', 'Fulfilment Cashflows', 'IBNR Methods', name])
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
                navigate_to(page, ['Home', 'LIC', 'Risk Adjustment', name])
    back_button('lic', ['Home', 'LIC'])


# =============================================================================
#  INDIVIDUAL CALCULATORS (UPR, Loss Component, OCR, Percentage unchanged)
# =============================================================================

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
            df_processed = df.rename(columns={start_date_col: 'Start_Date', end_date_col: 'End_Date'}).copy()
            df_processed['Start_Date'] = pd.to_datetime(df_processed['Start_Date'], errors='coerce')
            df_processed['End_Date'] = pd.to_datetime(df_processed['End_Date'], errors='coerce')
            df_processed = df_processed.dropna(subset=['Start_Date', 'End_Date'])
            df_processed = df_processed[df_processed['End_Date'] > df_processed['Start_Date']]
            for c in selected_value_cols: df_processed[c] = pd.to_numeric(df_processed[c], errors='coerce').fillna(0)
            df_processed["Duration_Days"] = (df_processed["End_Date"] - df_processed["Start_Date"]).dt.days + 1
            df_processed = df_processed[df_processed["Duration_Days"] > 0]
            if df_processed.empty: st.error("No valid policies after data validation."); return
            st.success(f"{len(df_processed):,} valid policies loaded")
            if st.button("Calculate UPR", key="upr_calc", use_container_width=True):
                with st.spinner("Calculating UPR..."):
                    df_processed['Remaining_Days'] = (df_processed["End_Date"] - valuation_date_ts).dt.days + 1
                    df_processed['Remaining_Days'] = np.clip(df_processed['Remaining_Days'], 0, df_processed['Duration_Days'])
                    if method == "365th": df_processed['Unearned_Portion'] = df_processed['Remaining_Days'] / df_processed['Duration_Days']
                    elif method == "24th":
                        interval = 365.25 / 24
                        df_processed['Unearned_Portion'] = (df_processed['Remaining_Days'] / interval) / (df_processed['Duration_Days'] / interval)
                    else:
                        interval = 365.25 / 8
                        df_processed['Unearned_Portion'] = (df_processed['Remaining_Days'] / interval) / (df_processed['Duration_Days'] / interval)
                    for c in selected_value_cols: df_processed[f"{c}_UPR"] = df_processed['Unearned_Portion'] * df_processed[c]
                    upr_columns = [f"{c}_UPR" for c in selected_value_cols]
                    result = df_processed.groupby(grouping_cols)[upr_columns].sum().reset_index()
                    result.columns = grouping_cols + selected_value_cols
                st.markdown("### UPR Results")
                disp = result.copy()
                for c in selected_value_cols: disp[c] = disp[c].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
                st.dataframe(disp, use_container_width=True, hide_index=True)
                total_upr = sum(result[c].sum() for c in selected_value_cols)
                st.metric("Total UPR", f"{total_upr:,.2f}")
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as w: result.to_excel(w, index=False, sheet_name='UPR_Results')
                output.seek(0)
                sc = sanitize_filename(client_name); so = sanitize_filename(base_filename)
                st.download_button("Download UPR Results", data=output, file_name=f"{sc}_{so}_UPR_{method}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="upr_dl")
        except Exception as e: st.error(f"Error: {e}")
    back_button('lrc', ['Home', 'LRC Calculators'])


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
        except Exception as e: st.error(f"Error: {e}")
    back_button('lrc', ['Home', 'LRC Calculators'])


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
            grouping_cols = st.multiselect("Group By Columns", options=all_columns, default=[all_columns[0]] if all_columns else [], key="ocr_gc")
            if not grouping_cols: st.info("Please select at least one Group By column."); return
            numeric_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c not in grouping_cols]
            selected_value_cols = st.multiselect("Amount Columns", options=numeric_columns, default=numeric_columns[:min(5, len(numeric_columns))], key="ocr_vc")
            if not selected_value_cols: st.info("Please select at least one Amount column."); return
            df_processed = df[grouping_cols + selected_value_cols].copy()
            for c in selected_value_cols: df_processed[c] = pd.to_numeric(df_processed[c], errors='coerce').fillna(0)
            grouped = df_processed.groupby(grouping_cols)[selected_value_cols].sum().reset_index()
            st.markdown("### OCR Summary")
            disp = grouped.copy()
            for c in selected_value_cols: disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            total_ocr = sum(grouped[c].sum() for c in selected_value_cols)
            st.metric("Total OCR", f"{total_ocr:,.2f}")
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w: grouped.to_excel(w, index=False, sheet_name='OCR_Results')
            output.seek(0)
            sc = sanitize_filename(client_name); so = sanitize_filename(base_filename)
            st.download_button("Download OCR Results", data=output, file_name=f"{sc}_{so}_OCR.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="ocr_dl")
        except Exception as e: st.error(f"Error: {e}")
    back_button('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows'])


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
    except Exception as e: st.error(f"Error: {e}")
    back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])


# =============================================================================
#  BCL – enhanced with triangles, LDFs, and ultimate claims per LOB
# =============================================================================

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
        if engine_utils is not None and ibnr_bcl is not None:
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
                inc_triangles = {}   # store incremental triangle per LOB (first amount column)
                ldfs_per_lob = {}    # store selected LDFs per LOB
                for lob in lobs:
                    lob_data = df[df[lob_col] == lob].copy()
                    for idx, ac in enumerate(amount_cols):
                        inc, cum, _ = engine_utils.build_triangles(lob_data, loss_col, rep_col, ac, from_dt, grain_code, n_periods)
                        if idx == 0:          # capture first amount column only
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

            # ---- Detailed results per LOB ----
            st.markdown("### Detailed Results per LOB")
            for lob in lobs:
                with st.expander(f"LOB: {lob}", expanded=False):
                    # Incremental triangle
                    if lob in inc_triangles:
                        st.markdown("**Incremental Claims Triangle**")
                        st.dataframe(inc_triangles[lob].style.format("{:,.2f}"), use_container_width=True)
                    # Development factors
                    if lob in ldfs_per_lob:
                        st.markdown("**Selected Development Factors**")
                        ldf_series = pd.Series(ldfs_per_lob[lob], name="Factor")
                        ldf_series.index = [f"{i}-{i+1}" for i in range(len(ldf_series))]
                        st.dataframe(ldf_series.to_frame().T, use_container_width=True)
                    # Ultimate claims
                    lob_df = final_df[(final_df['LOB'] == lob) & (final_df['Amount_Col'] == amount_cols[0])]
                    if not lob_df.empty:
                        st.markdown("**Ultimate Claims**")
                        ult_disp = lob_df[['Accident_Period_Label', 'Current_Claims', 'Ultimate_Claims', 'IBNR']].copy()
                        for c in ['Current_Claims', 'Ultimate_Claims', 'IBNR']:
                            ult_disp[c] = ult_disp[c].apply(lambda x: f"{x:,.2f}")
                        st.dataframe(ult_disp, use_container_width=True, hide_index=True)

            output, ext, mime = build_download_payload({'BCL_Summary': summary, 'BCL_Detail': final_df})
            sc = sanitize_filename(client_name)
            st.download_button("Download BCL Results", data=output, file_name=f"{sc}_BCL_IBNR.{ext}", mime=mime, key="bcl_dl")
    except Exception as e: st.error(f"Error: {e}")
    back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])


# =============================================================================
#  CAPE COD – enhanced with triangles, LDFs, and ultimate claims per LOB
# =============================================================================

def render_capecod_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Cape Cod - IBNR</h1><p>Uses premiums to derive expected loss ratio</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: client_name = st.text_input("Client Name", value="Client", key="cc_cn").strip()
    with c2: from_date = st.date_input("From Date", date(2020, 1, 1), key="cc_fd")
    with c3: to_date = st.date_input("To Date", date(2025, 12, 31), key="cc_td")
    claims_file = st.file_uploader("Claims Data (Loss Date, Report Date, LOB, Amount)", type=["csv", "xlsx", "xls"], key="cc_cf")
    prem_file = st.file_uploader("Premiums Data (Dev_Period + LOB columns)", type=["csv", "xlsx", "xls"], key="cc_pf")
    if claims_file is None or prem_file is None: st.info("Upload both claims and premiums files."); back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods']); return
    try:
        df = read_uploaded_file(claims_file)
        df.columns = df.columns.astype(str).str.strip()
        prem_df = read_uploaded_file(prem_file)
        prem_df.columns = prem_df.columns.astype(str).str.strip()
        st.dataframe(df.head(3), use_container_width=True); st.dataframe(prem_df.head(3), use_container_width=True)
        cols = df.columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1: loss_col = st.selectbox("Loss Date", cols, key="cc_ld")
        with c2: rep_col = st.selectbox("Report Date", cols, key="cc_rd")
        with c3: lob_col = st.selectbox("LOB", cols, key="cc_lob")
        amount_candidates = [c for c in cols if c not in [loss_col, rep_col, lob_col] and pd.api.types.is_numeric_dtype(df[c])]
        amount_cols = st.multiselect("Amount Column(s)", amount_candidates, key="cc_amt")
        if not amount_cols: st.warning("Please select at least one Amount column."); return

        df[loss_col] = pd.to_datetime(df[loss_col], errors='coerce'); df[rep_col] = pd.to_datetime(df[rep_col], errors='coerce')
        for ac in amount_cols: df[ac] = pd.to_numeric(df[ac], errors='coerce').fillna(0)
        df = df.dropna(subset=[loss_col, rep_col])
        from_dt = pd.Timestamp(str(from_date)); to_dt = pd.Timestamp(str(to_date))
        df = _date_filter(df, loss_col, from_date, to_date)
        lobs = sorted(df[lob_col].dropna().unique())

        p_cols = prem_df.columns.tolist()
        prem_period_col = st.selectbox("Development Period Column", p_cols, key="cc_prem_period")
        avail_cols = [c for c in p_cols if c != prem_period_col]
        if len(avail_cols) < len(lobs):
            st.error(f"Number of premium LOB columns ({len(avail_cols)}) is less than the number of unique LOBs in claims ({len(lobs)}).")
            return
        lob_mapping = {}
        st.markdown("**Map each Line of Business to a premium column:**")
        for lob in lobs:
            options = ["None"] + avail_cols
            default = next((c for c in avail_cols if c.strip().lower() == lob.strip().lower()), "None")
            idx = options.index(default) if default in options else 0
            lob_mapping[lob] = st.selectbox(f"Column for '{lob}'", options, index=idx, key=f"cc_map_{lob}")

        grain = "Y"; ppy = 1; n_periods = to_dt.year - from_dt.year + 1
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
            rec_method = "volume_weighted"; min_cv = float('inf')
            for method in ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"]:
                factors = all_ldfs[method]
                if len(factors) >= 3:
                    cv = np.std(factors[:3]) / np.mean(factors[:3]) if np.mean(factors[:3]) > 0 else float('inf')
                    if cv < min_cv: min_cv = cv; rec_method = method
            st.info(f"Recommended: {rec_method.replace('_', ' ').title()}")
            selected_method = st.selectbox(
                "Select LDF Method to Use",
                ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"],
                index=["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"].index(rec_method),
                key="cc_ldf_method"
            )
        st.markdown("### Adjustments")
        c1, c2 = st.columns(2)
        with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key="cc_inf")
        with c2: use_discounting = st.checkbox("Apply Discounting", key="cc_disc")
        cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
        if use_inflation: cum_inflation, per_period_rates = load_inflation_data_ui(grain, ppy, "cc")
        if use_discounting: spot_rates, flat_rate = load_discounting_data_ui(grain, ppy, "cc")
        if st.button("Calculate Cape Cod IBNR", key="cc_run", use_container_width=True):
            if ibnr_cc is None or engine_utils is None: st.error("Required engines not available."); return
            with st.spinner("Calculating Cape Cod IBNR..."):
                all_results = []
                inc_triangles = {}
                ldfs_per_lob = {}
                for lob in lobs:
                    lob_data = df[df[lob_col] == lob].copy()
                    prems = [1.0] * n_periods
                    prem_col = lob_mapping.get(lob, "None")
                    if prem_col != "None":
                        lob_prem = prem_df[[prem_period_col, prem_col]].copy()
                        lob_prem[prem_period_col] = pd.to_numeric(lob_prem[prem_period_col], errors='coerce')
                        lob_prem[prem_col] = pd.to_numeric(lob_prem[prem_col], errors='coerce').fillna(0)
                        lob_prem = lob_prem.dropna(subset=[prem_period_col])
                        lob_prem = lob_prem.sort_values(prem_period_col)
                        if len(lob_prem) != n_periods:
                            st.error(f"Number of premium periods for LOB '{lob}' ({len(lob_prem)}) does not match the number of accident periods ({n_periods}).")
                            return
                        prems = lob_prem[prem_col].tolist()
                    for idx, ac in enumerate(amount_cols):
                        inc, cum, _ = engine_utils.build_triangles(lob_data, loss_col, rep_col, ac, from_dt, grain, n_periods)
                        if idx == 0:
                            inc_triangles[lob] = inc.copy()
                        try:
                            result = ibnr_cc.calculate_cape_cod_ibnr(
                                cum_triangle=cum, premiums=prems, start_date=from_dt,
                                period_unit=grain, selected_ldf_method=selected_method,
                                use_inflation=use_inflation, cum_inflation=cum_inflation,
                                per_period_rates=per_period_rates, use_discounting=use_discounting,
                                spot_rates=spot_rates, flat_rate=flat_rate
                            )
                        except TypeError:
                            result = ibnr_cc.calculate_cape_cod_ibnr(
                                cum_triangle=cum, premiums=prems, start_date=from_dt, period_unit=grain
                            )
                        if idx == 0:
                            ldfs_per_lob[lob] = result.get('dev_factors', [])
                        res_df = result['results_df']; res_df['LOB'] = lob; res_df['Amount_Col'] = ac
                        all_results.append(res_df)
                final_df = pd.concat(all_results, ignore_index=True)
                ibnr_col = next((c for c in final_df.columns if 'IBNR' in c and 'Real' not in c), 'Cape_Cod_IBNR')
                current_col = next((c for c in final_df.columns if 'Current' in c), 'Current_Claims')
                summary = final_df.groupby(['LOB', 'Amount_Col'])[[current_col, ibnr_col]].sum().reset_index()
            st.markdown("### Cape Cod IBNR Summary")
            disp = summary.copy()
            for c in [current_col, ibnr_col]:
                if c in disp.columns: disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            total_ibnr = summary[ibnr_col].sum()
            st.metric("Total Cape Cod IBNR", f"{total_ibnr:,.2f}")

            # ---- Detailed results per LOB ----
            st.markdown("### Detailed Results per LOB")
            for lob in lobs:
                with st.expander(f"LOB: {lob}", expanded=False):
                    if lob in inc_triangles:
                        st.markdown("**Incremental Claims Triangle**")
                        st.dataframe(inc_triangles[lob].style.format("{:,.2f}"), use_container_width=True)
                    if lob in ldfs_per_lob and ldfs_per_lob[lob]:
                        st.markdown("**Selected Development Factors**")
                        ldf_series = pd.Series(ldfs_per_lob[lob], name="Factor")
                        ldf_series.index = [f"{i}-{i+1}" for i in range(len(ldf_series))]
                        st.dataframe(ldf_series.to_frame().T, use_container_width=True)
                    lob_df = final_df[(final_df['LOB'] == lob) & (final_df['Amount_Col'] == amount_cols[0])]
                    if not lob_df.empty:
                        st.markdown("**Ultimate Claims**")
                        # Cape Cod results may use different column names; try common ones
                        ult_cols = [c for c in lob_df.columns if 'Ultimate' in c or 'Current' in c or 'IBNR' in c]
                        if 'Accident_Period_Label' in lob_df.columns:
                            disp_cols = ['Accident_Period_Label'] + [c for c in ult_cols if c in lob_df.columns]
                        else:
                            disp_cols = ult_cols
                        ult_disp = lob_df[disp_cols].copy()
                        for c in ult_disp.columns:
                            if c != 'Accident_Period_Label':
                                ult_disp[c] = ult_disp[c].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
                        st.dataframe(ult_disp, use_container_width=True, hide_index=True)

            output, ext, mime = build_download_payload({'CapeCod_Summary': summary, 'CapeCod_Detail': final_df})
            sc = sanitize_filename(client_name)
            st.download_button("Download Cape Cod Results", data=output, file_name=f"{sc}_CapeCod_IBNR.{ext}", mime=mime, key="cc_dl")
    except Exception as e: st.error(f"Error: {e}")
    back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])


# =============================================================================
#  BF – enhanced with triangles, LDFs, and ultimate claims per LOB
# =============================================================================

def render_bf_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Bornhuetter-Ferguson - IBNR</h1><p>Multi-LDF Methods with Expected Loss Ratio</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: client_name = st.text_input("Client Name", value="Client", key="bf_cn").strip()
    with c2: from_date = st.date_input("From Date", date(2020, 1, 1), key="bf_fd")
    with c3: to_date = st.date_input("To Date", date(2025, 12, 31), key="bf_td")
    claims_file = st.file_uploader("Claims Data (Loss Date, Report Date, LOB, Amount)", type=["csv", "xlsx", "xls"], key="bf_cf")
    prem_file = st.file_uploader("Premiums Data (Dev_Period + LOB columns) - Optional", type=["csv", "xlsx", "xls"], key="bf_pf")
    if claims_file is None: st.info("Upload claims data file."); back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods']); return
    try:
        df = read_uploaded_file(claims_file)
        df.columns = df.columns.astype(str).str.strip()
        st.dataframe(df.head(5), use_container_width=True)
        cols = df.columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1: loss_col = st.selectbox("Loss Date", cols, key="bf_ld")
        with c2: rep_col = st.selectbox("Report Date", cols, key="bf_rd")
        with c3: lob_col = st.selectbox("LOB", cols, key="bf_lob")
        amount_candidates = [c for c in cols if c not in [loss_col, rep_col, lob_col] and pd.api.types.is_numeric_dtype(df[c])]
        amount_cols = st.multiselect("Amount Column(s)", amount_candidates, key="bf_amt")
        if not amount_cols: st.warning("Please select at least one Amount column."); return
        df[loss_col] = pd.to_datetime(df[loss_col], errors='coerce'); df[rep_col] = pd.to_datetime(df[rep_col], errors='coerce')
        for ac in amount_cols: df[ac] = pd.to_numeric(df[ac], errors='coerce').fillna(0)
        df = df.dropna(subset=[loss_col, rep_col])
        from_dt = pd.Timestamp(str(from_date)); to_dt = pd.Timestamp(str(to_date))
        df = _date_filter(df, loss_col, from_date, to_date)
        lobs = sorted(df[lob_col].dropna().unique())

        prem_df = None; prem_period_col = None; lob_mapping = {}
        if prem_file is not None:
            prem_df = read_uploaded_file(prem_file)
            prem_df.columns = prem_df.columns.astype(str).str.strip()
            st.dataframe(prem_df.head(3), use_container_width=True)
            p_cols = prem_df.columns.tolist()
            prem_period_col = st.selectbox("Development Period Column", p_cols, key="bf_prem_period")
            avail_cols = [c for c in p_cols if c != prem_period_col]
            if len(avail_cols) < len(lobs):
                st.error(f"Number of premium LOB columns ({len(avail_cols)}) is less than the number of unique LOBs in claims ({len(lobs)}).")
                return
            st.markdown("**Map each Line of Business to a premium column:**")
            for lob in lobs:
                options = ["None"] + avail_cols
                default = next((c for c in avail_cols if c.strip().lower() == lob.strip().lower()), "None")
                idx = options.index(default) if default in options else 0
                lob_mapping[lob] = st.selectbox(f"Column for '{lob}'", options, index=idx, key=f"bf_map_{lob}")

        st.markdown("### Expected Loss Ratios (ELR) per LOB")
        elr_cols = st.columns(min(len(lobs), 4))
        elr_dict = {}
        for i, lob in enumerate(lobs):
            with elr_cols[i % 4]: elr_dict[lob] = st.number_input(f"ELR {lob} (%)", 0.0, 200.0, 70.0, 1.0, key=f"bf_elr_{lob}") / 100.0
        grain = "Y"; ppy = 1; n_periods = to_dt.year - from_dt.year + 1
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
            rec_method = "volume_weighted"; min_cv = float('inf')
            for method in ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"]:
                factors = all_ldfs[method]
                if len(factors) >= 3:
                    cv = np.std(factors[:3]) / np.mean(factors[:3]) if np.mean(factors[:3]) > 0 else float('inf')
                    if cv < min_cv: min_cv = cv; rec_method = method
            st.info(f"Recommended: {rec_method.replace('_', ' ').title()}")
            selected_method = st.selectbox(
                "Select LDF Method to Use",
                ["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"],
                index=["volume_weighted", "simple_average", "geometric", "medial", "linear_regression", "weighted_last_3"].index(rec_method),
                key="bf_ldf_method"
            )
        st.markdown("### Adjustments")
        c1, c2 = st.columns(2)
        with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key="bf_inf")
        with c2: use_discounting = st.checkbox("Apply Discounting", key="bf_disc")
        cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
        if use_inflation: cum_inflation, per_period_rates = load_inflation_data_ui(grain, ppy, "bf")
        if use_discounting: spot_rates, flat_rate = load_discounting_data_ui(grain, ppy, "bf")
        if st.button("Calculate BF IBNR", key="bf_run", use_container_width=True):
            if ibnr_bf is None or engine_utils is None: st.error("Required engines not available."); return
            with st.spinner("Calculating BF IBNR..."):
                all_results = []
                inc_triangles = {}
                ldfs_per_lob = {}
                for lob in lobs:
                    lob_data = df[df[lob_col] == lob].copy()
                    prems = [1.0] * n_periods
                    prem_col = lob_mapping.get(lob, "None") if prem_file is not None else "None"
                    if prem_col != "None":
                        lob_prem = prem_df[[prem_period_col, prem_col]].copy()
                        lob_prem[prem_period_col] = pd.to_numeric(lob_prem[prem_period_col], errors='coerce')
                        lob_prem[prem_col] = pd.to_numeric(lob_prem[prem_col], errors='coerce').fillna(0)
                        lob_prem = lob_prem.dropna(subset=[prem_period_col])
                        lob_prem = lob_prem.sort_values(prem_period_col)
                        if len(lob_prem) != n_periods:
                            st.error(f"Number of premium periods for LOB '{lob}' ({len(lob_prem)}) does not match the number of accident periods ({n_periods}).")
                            return
                        prems = lob_prem[prem_col].tolist()
                    for idx, ac in enumerate(amount_cols):
                        inc, cum, _ = engine_utils.build_triangles(lob_data, loss_col, rep_col, ac, from_dt, grain, n_periods)
                        if idx == 0:
                            inc_triangles[lob] = inc.copy()
                        try:
                            result = ibnr_bf.calculate_bf_ibnr(
                                cum_triangle=cum, premiums=prems, elr=elr_dict.get(lob, 0.7),
                                start_date=from_dt, period_unit=grain,
                                selected_ldf_method=selected_method,
                                use_inflation=use_inflation, cum_inflation=cum_inflation,
                                per_period_rates=per_period_rates,
                                use_discounting=use_discounting, spot_rates=spot_rates, flat_rate=flat_rate
                            )
                        except TypeError:
                            result = ibnr_bf.calculate_bf_ibnr(
                                cum_triangle=cum, premiums=prems, elr=elr_dict.get(lob, 0.7),
                                start_date=from_dt, period_unit=grain
                            )
                        if idx == 0:
                            ldfs_per_lob[lob] = result.get('dev_factors', [])
                        res_df = result['results_df']; res_df['LOB'] = lob; res_df['Amount_Col'] = ac
                        all_results.append(res_df)
                final_df = pd.concat(all_results, ignore_index=True)
                ibnr_col = next((c for c in final_df.columns if 'IBNR' in c and 'Real' not in c), 'BF_IBNR')
                current_col = next((c for c in final_df.columns if 'Current' in c), 'Current_Claims')
                summary = final_df.groupby(['LOB', 'Amount_Col'])[[current_col, ibnr_col]].sum().reset_index()
            st.markdown("### BF IBNR Summary")
            disp = summary.copy()
            for c in [current_col, ibnr_col]:
                if c in disp.columns: disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            total_ibnr = summary[ibnr_col].sum()
            st.metric("Total BF IBNR", f"{total_ibnr:,.2f}")

            # ---- Detailed results per LOB ----
            st.markdown("### Detailed Results per LOB")
            for lob in lobs:
                with st.expander(f"LOB: {lob}", expanded=False):
                    if lob in inc_triangles:
                        st.markdown("**Incremental Claims Triangle**")
                        st.dataframe(inc_triangles[lob].style.format("{:,.2f}"), use_container_width=True)
                    if lob in ldfs_per_lob and ldfs_per_lob[lob]:
                        st.markdown("**Selected Development Factors**")
                        ldf_series = pd.Series(ldfs_per_lob[lob], name="Factor")
                        ldf_series.index = [f"{i}-{i+1}" for i in range(len(ldf_series))]
                        st.dataframe(ldf_series.to_frame().T, use_container_width=True)
                    lob_df = final_df[(final_df['LOB'] == lob) & (final_df['Amount_Col'] == amount_cols[0])]
                    if not lob_df.empty:
                        st.markdown("**Ultimate Claims**")
                        ult_disp = lob_df[['Accident_Period_Label', 'Current_Claims', 'BF_IBNR']].copy()
                        if 'Ultimate_Claims' in lob_df.columns:
                            ult_disp['Ultimate_Claims'] = lob_df['Ultimate_Claims']
                        else:
                            ult_disp['Ultimate_Claims'] = lob_df['Current_Claims'] + lob_df['BF_IBNR']
                        for c in ['Current_Claims', 'BF_IBNR', 'Ultimate_Claims']:
                            if c in ult_disp.columns:
                                ult_disp[c] = ult_disp[c].apply(lambda x: f"{x:,.2f}")
                        st.dataframe(ult_disp, use_container_width=True, hide_index=True)

            output, ext, mime = build_download_payload({'BF_Summary': summary, 'BF_Detail': final_df})
            sc = sanitize_filename(client_name)
            st.download_button("Download BF Results", data=output, file_name=f"{sc}_BF_IBNR.{ext}", mime=mime, key="bf_dl")
    except Exception as e: st.error(f"Error: {e}")
    back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])


# =============================================================================
#  REMAINING CALCULATORS (ULAE, NPR, Mack, Bootstrap, Full Valuation unchanged)
# =============================================================================

def render_ulae_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>ULAE Calculator</h1><p>Unallocated Loss Adjustment Expenses</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: client_name = st.text_input("Client Name", value="Client", key="ulae_cn").strip()
    with c2: ulae_ratio = st.number_input("ULAE Ratio (%)", 0.0, 30.0, 5.0, 0.5, key="ulae_rt") / 100.0
    with c3: basis = st.selectbox("Allocation Basis", ["Per Portfolio", "Aggregated"], key="ulae_bs")
    uploaded = st.file_uploader("Upload Reserves File (LOB, OCR, IBNR)", type=["csv", "xlsx", "xls"], key="ulae_f")
    if uploaded is None: st.info("Upload file with LOB, OCR, and IBNR columns."); back_button('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows']); return
    try:
        df = read_uploaded_file(uploaded)
        df.columns = df.columns.astype(str).str.strip()
        st.dataframe(df.head(5), use_container_width=True)
        cols = df.columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1: lob_col = st.selectbox("LOB Column", cols, key="ulae_lob")
        with c2: ocr_col = st.selectbox("OCR Column", cols, key="ulae_ocr")
        with c3: ibnr_col = st.selectbox("IBNR Column", cols, key="ulae_ibnr")
        if st.button("Calculate ULAE", key="ulae_run", use_container_width=True):
            df[ocr_col] = pd.to_numeric(df[ocr_col], errors='coerce').fillna(0)
            df[ibnr_col] = pd.to_numeric(df[ibnr_col], errors='coerce').fillna(0)
            df['ULAE_Base'] = 0.5 * df[ocr_col] + df[ibnr_col]
            df['ULAE'] = df['ULAE_Base'] * ulae_ratio
            res = df[[lob_col, ocr_col, ibnr_col, 'ULAE_Base', 'ULAE']].copy()
            st.markdown("### ULAE Results")
            disp = res.copy()
            for c in [ocr_col, ibnr_col, 'ULAE_Base', 'ULAE']: disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.metric("Total ULAE", f"{df['ULAE'].sum():,.2f}")
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w: res.to_excel(w, index=False, sheet_name='ULAE_Results')
            output.seek(0); sc = sanitize_filename(client_name)
            st.download_button("Download ULAE Results", data=output, file_name=f"{sc}_ULAE.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="ulae_dl")
    except Exception as e: st.error(f"Error: {e}")
    back_button('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows'])


def render_npr_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>NPR Calculator</h1><p>Reinsurance Non-Performance Risk - IFRS 17 Para 63(e)</p></div>', unsafe_allow_html=True)
    client_name = st.text_input("Client Name", value="Client", key="npr_cn").strip()
    st.markdown("#### Reinsurer Data (Name, Credit Rating, PD, Share)")
    ri_file = st.file_uploader("Upload Reinsurer File", type=["csv", "xlsx", "xls"], key="npr_rf")
    st.markdown("#### Ceded LIC Data (Portfolio, Ceded IBNR, Ceded OCR)")
    lic_file = st.file_uploader("Upload Ceded LIC File", type=["csv", "xlsx", "xls"], key="npr_lf")
    if ri_file is None or lic_file is None: st.info("Upload both Reinsurer and Ceded LIC files."); back_button('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows']); return
    try:
        ri_df = read_uploaded_file(ri_file)
        ri_df.columns = ri_df.columns.astype(str).str.strip()
        lic_df = read_uploaded_file(lic_file)
        lic_df.columns = lic_df.columns.astype(str).str.strip()
        c1, c2 = st.columns(2)
        with c1: st.caption("Reinsurer Data"); st.dataframe(ri_df.head(3), use_container_width=True)
        with c2: st.caption("Ceded LIC Data"); st.dataframe(lic_df.head(3), use_container_width=True)
        rc = ri_df.columns.tolist(); lc = lic_df.columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1: name_col = st.selectbox("Reinsurer Name", rc, key="npr_rn")
        with c2: pd_col = st.selectbox("PD Column", rc, key="npr_pd")
        with c3: share_col = st.selectbox("Share Column", rc, key="npr_sh")
        c1, c2 = st.columns(2)
        with c1: port_col = st.selectbox("Portfolio Column", lc, key="npr_pc")
        with c2: ibnr_col = st.selectbox("Ceded IBNR Column", lc, key="npr_ibnr")
        ocr_col = st.selectbox("Ceded OCR Column", lc, key="npr_ocr")
        if st.button("Calculate NPR", key="npr_run", use_container_width=True):
            ri_df[pd_col] = pd.to_numeric(ri_df[pd_col], errors='coerce').fillna(0)
            ri_df[share_col] = pd.to_numeric(ri_df[share_col], errors='coerce').fillna(0)
            lic_df[ibnr_col] = pd.to_numeric(lic_df[ibnr_col], errors='coerce').fillna(0)
            lic_df[ocr_col] = pd.to_numeric(lic_df[ocr_col], errors='coerce').fillna(0)
            lic_df['Total_LIC'] = lic_df[ibnr_col] + lic_df[ocr_col]
            ri_small = ri_df[[name_col, pd_col, share_col]].rename(
                columns={name_col: 'Reinsurer', pd_col: 'PD', share_col: 'Share'})
            lic_small = lic_df[[port_col, 'Total_LIC']].rename(columns={port_col: 'Portfolio'})
            res = ri_small.merge(lic_small, how='cross')
            res['NPR'] = res['PD'] * res['Share'] * res['Total_LIC']
            res = res[['Reinsurer', 'Portfolio', 'PD', 'Share', 'Total_LIC', 'NPR']]
            by_port = res.groupby('Portfolio', observed=True)['NPR'].sum().reset_index()
            by_ri = res.groupby('Reinsurer', observed=True)['NPR'].sum().reset_index()
            st.markdown("### NPR Results")
            c1, c2 = st.columns(2)
            with c1:
                disp = by_port.copy(); disp['NPR'] = disp['NPR'].apply(lambda x: f"{x:,.2f}")
                st.dataframe(disp, use_container_width=True, hide_index=True)
            with c2:
                disp2 = by_ri.copy(); disp2['NPR'] = disp2['NPR'].apply(lambda x: f"{x:,.2f}")
                st.dataframe(disp2, use_container_width=True, hide_index=True)
            st.metric("Total NPR", f"{res['NPR'].sum():,.2f}")
            output, ext, mime = build_download_payload({'NPR_by_Portfolio': by_port, 'NPR_by_Reinsurer': by_ri, 'NPR_Detail': res})
            sc = sanitize_filename(client_name)
            st.download_button("Download NPR Results", data=output, file_name=f"{sc}_NPR.{ext}", mime=mime, key="npr_dl")
    except Exception as e: st.error(f"Error: {e}")
    back_button('fulfilment_cashflows', ['Home', 'LIC Calculators', 'Fulfilment Cashflows'])


def run_mack_calculation(triangle, confidence, z_score, client_name, use_inflation=False, cum_inflation=None, per_period_rates=None, use_discounting=False, spot_rates=None, flat_rate=None, grain='Y', origin_date=None):
    with st.spinner("Calculating Mack Chain Ladder..."):
        n_ay, n_dev = triangle.shape
        C = triangle.values.copy().astype(float)
        if mack_engine is not None and hasattr(mack_engine, 'calculate_mack_chain_ladder'):
            obs_mask = pd.DataFrame(False, index=range(n_ay), columns=range(n_dev))
            for i in range(n_ay):
                for j in range(n_dev):
                    if i + j < n_ay and not np.isnan(C[i, j]): obs_mask.iloc[i, j] = True
            if origin_date is None: origin_date = pd.Timestamp('2020-01-01')
            result = mack_engine.calculate_mack_chain_ladder(
                cum_triangle=triangle, obs_mask=obs_mask, confidence_level=confidence,
                use_inflation=use_inflation, cum_inflation=cum_inflation,
                per_period_rates=per_period_rates, use_discounting=use_discounting,
                spot_rates=spot_rates, flat_rate=flat_rate, grain=grain
            )
            res = result['results_df']
        else:
            dev_factors = []
            for j in range(n_dev - 1):
                num = den = 0.0
                for i in range(n_ay):
                    if i + j + 1 < n_ay:
                        c = C[i, j]; n_val = C[i, j + 1]
                        if not np.isnan(c) and not np.isnan(n_val) and c > 0: num += n_val; den += c
                dev_factors.append(num / den if den > 0 else 1.0)
            proj = C.copy()
            for i in range(n_ay):
                last_obs = -1
                for j in range(n_dev - 1, -1, -1):
                    if i + j < n_ay and not np.isnan(C[i, j]): last_obs = j; break
                if last_obs < 0: continue
                for j in range(last_obs, n_dev - 1):
                    if j < len(dev_factors): proj[i, j + 1] = proj[i, j] * dev_factors[j]
            sigma2 = {}
            for j in range(n_dev - 1):
                pairs = []
                for i in range(n_ay):
                    if i + j + 1 < n_ay:
                        c = C[i, j]; n_val = C[i, j + 1]
                        if not np.isnan(c) and not np.isnan(n_val) and c > 0: pairs.append((c, n_val / c))
                if len(pairs) >= 2:
                    f_j = dev_factors[j]; sigma2[j] = sum(c * (r - f_j) ** 2 for c, r in pairs) / (len(pairs) - 1)
                else: sigma2[j] = 0
            rows_list = []
            for i in range(n_ay):
                last_obs = -1
                for j in range(n_dev - 1, -1, -1):
                    if i + j < n_ay and not np.isnan(C[i, j]): last_obs = j; break
                if last_obs < 0 or last_obs >= n_dev - 1: continue
                current = C[i, last_obs]; ultimate = proj[i, n_dev - 1]; ibnr = max(ultimate - current, 0)
                variance = 0
                if ibnr > 0:
                    for k in range(last_obs, n_dev - 1):
                        if k in sigma2 and k < len(dev_factors) and dev_factors[k] > 0:
                            C_ik = proj[i, k]
                            if C_ik > 0:
                                S_k = sum(C[m, k] for m in range(n_ay) if m + k < n_ay and not np.isnan(C[m, k]))
                                variance += sigma2[k] / (dev_factors[k] ** 2) * (1 / C_ik + 1 / max(S_k, 1))
                    variance *= ultimate ** 2
                mack_se = np.sqrt(max(variance, 0)); ra = z_score * mack_se
                rows_list.append({'AY': triangle.index[i] if hasattr(triangle, 'index') else i, 'Current': current, 'Ultimate': ultimate, 'IBNR': ibnr, 'Mack_SE': mack_se, 'RA': ra})
            res = pd.DataFrame(rows_list)
    st.markdown(f"### Mack RA Results at {confidence:.0%} Confidence")
    disp = res.copy()
    for c in ['Current', 'Ultimate', 'IBNR', 'Mack_SE', 'RA']:
        if c in disp.columns: disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
    st.dataframe(disp, use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total IBNR", f"{res['IBNR'].sum():,.2f}")
    with c2: st.metric("Total RA", f"{res['RA'].sum():,.2f}")
    with c3: st.metric("Total LIC", f"{res['IBNR'].sum() + res['RA'].sum():,.2f}")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as w: res.to_excel(w, index=False, sheet_name='Mack_RA')
    output.seek(0); sc = sanitize_filename(client_name)
    st.download_button("Download Mack Results", data=output, file_name=f"{sc}_Mack_RA.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="mck_dl")


def render_mack_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Mack Chain Ladder - Risk Adjustment</h1><p>Distribution-free standard error of IBNR (Mack 1993)</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: client_name = st.text_input("Client Name", value="Client", key="mck_cn").strip()
    with c2: confidence = st.number_input("Confidence Level (%)", 50.0, 99.9, 75.0, 1.0, key="mck_cl") / 100.0
    z_score = scipy_stats.norm.ppf(confidence)
    st.info(f"z-score: {z_score:.3f}")
    input_type = st.radio("Input Data Type", ["Claims-Level Data", "Cumulative Triangle"], key="mck_input_type")
    if input_type == "Cumulative Triangle":
        uploaded = st.file_uploader("Upload Cumulative Claims Triangle (CSV/Excel)", type=["csv", "xlsx", "xls"], key="mck_f_tri")
        st.caption("First column = AY labels, remaining = cumulative claims.")
        if uploaded is None: back_button('risk_adjustment', ['Home', 'LIC Calculators', 'Risk Adjustment']); return
        try:
            raw = read_uploaded_triangle(uploaded)
            st.dataframe(raw, use_container_width=True)
            grain = "Y"; ppy = 1
            st.markdown("### Adjustments")
            c1, c2 = st.columns(2)
            with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key="mck_inf_tri")
            with c2: use_discounting = st.checkbox("Apply Discounting", key="mck_disc_tri")
            cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
            if use_inflation: cum_inflation, per_period_rates = load_inflation_data_ui(grain, ppy, "mck_tri")
            if use_discounting: spot_rates, flat_rate = load_discounting_data_ui(grain, ppy, "mck_tri")
            if st.button("Calculate Mack RA", key="mck_run_tri", use_container_width=True):
                run_mack_calculation(raw, confidence, z_score, client_name, use_inflation, cum_inflation, per_period_rates, use_discounting, spot_rates, flat_rate, grain, origin_date=pd.Timestamp('2020-01-01'))
        except Exception as e: st.error(f"Error: {e}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: grain = st.selectbox("Period Grain", ["Yearly", "Quarterly", "Monthly"], key="mck_gr")
        with c2: from_date = st.date_input("From Date", date(2020, 1, 1), key="mck_fd")
        with c3: to_date = st.date_input("To Date", date(2025, 12, 31), key="mck_td")
        grain_map = {"Yearly": "Y", "Quarterly": "Q", "Monthly": "M"}; grain_code = grain_map[grain]; ppy = {"Y": 1, "Q": 4, "M": 12}[grain_code]
        uploaded = st.file_uploader("Upload Claims Data (Loss Date, Report Date, Amount)", type=["csv", "xlsx", "xls"], key="mck_f_cl")
        if uploaded is None: back_button('risk_adjustment', ['Home', 'LIC Calculators', 'Risk Adjustment']); return
        try:
            df = read_uploaded_file(uploaded)
            df.columns = df.columns.astype(str).str.strip()
            st.dataframe(df.head(5), use_container_width=True)
            cols = df.columns.tolist()
            c1, c2 = st.columns(2)
            with c1: loss_col = st.selectbox("Loss Date", cols, key="mck_ld")
            with c2: rep_col = st.selectbox("Report Date", cols, key="mck_rd")
            amount_col = st.selectbox("Amount Column", cols, key="mck_amt")
            df[loss_col] = pd.to_datetime(df[loss_col], errors='coerce'); df[rep_col] = pd.to_datetime(df[rep_col], errors='coerce')
            df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
            df = df.dropna(subset=[loss_col, rep_col])
            from_dt = pd.Timestamp(str(from_date)); to_dt = pd.Timestamp(str(to_date))
            df = _date_filter(df, loss_col, from_date, to_date)
            n_periods = int((to_dt.year - from_dt.year) * ppy) + 1
            if engine_utils is not None:
                _, cum_triangle, _ = engine_utils.build_triangles(df, loss_col, rep_col, amount_col, from_dt, grain_code, n_periods)
                st.markdown("#### Built Triangle"); st.dataframe(cum_triangle, use_container_width=True)
                st.markdown("### Adjustments")
                c1, c2 = st.columns(2)
                with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key="mck_inf_cl")
                with c2: use_discounting = st.checkbox("Apply Discounting", key="mck_disc_cl")
                cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
                if use_inflation: cum_inflation, per_period_rates = load_inflation_data_ui(grain_code, ppy, "mck_cl")
                if use_discounting: spot_rates, flat_rate = load_discounting_data_ui(grain_code, ppy, "mck_cl")
                if st.button("Calculate Mack RA", key="mck_run_cl", use_container_width=True):
                    run_mack_calculation(cum_triangle, confidence, z_score, client_name, use_inflation, cum_inflation, per_period_rates, use_discounting, spot_rates, flat_rate, grain_code, origin_date=from_dt)
            else: st.error("Engine utils not available.")
        except Exception as e: st.error(f"Error: {e}")
    back_button('risk_adjustment', ['Home', 'LIC Calculators', 'Risk Adjustment'])


def run_bootstrap_calculation(triangle, confidence, n_iter, add_pv, client_name, use_inflation=False, cum_inflation=None, per_period_rates=None, use_discounting=False, spot_rates=None, flat_rate=None, grain='Y', origin_date=None):
    with st.spinner(f"Running {n_iter:,} bootstrap iterations..."):
        n_ay, n_dev = triangle.shape
        C = triangle.values.copy().astype(float)
        if bootstrap_engine is not None and hasattr(bootstrap_engine, 'bootstrap_chain_ladder'):
            obs_mask = pd.DataFrame(False, index=range(n_ay), columns=range(n_dev))
            for i in range(n_ay):
                for j in range(n_dev):
                    if i + j < n_ay and not np.isnan(C[i, j]): obs_mask.iloc[i, j] = True
            if origin_date is None: origin_date = pd.Timestamp('2020-01-01')
            result = bootstrap_engine.bootstrap_chain_ladder(
                working_cum=triangle, obs_mask=obs_mask, origin=origin_date, grain=grain,
                n_periods=n_ay, n_iterations=n_iter, add_process_variance=add_pv,
                use_inflation=use_inflation, per_period_rates=per_period_rates,
                cum_inflation=cum_inflation, use_discounting=use_discounting,
                spot_rates=spot_rates, flat_rate=flat_rate, seed=42
            )
            cl_ibnr = result['cl_ibnr_nominal']; boot_mean = result['bootstrap_mean']
            pctl = result['percentiles_nominal'].get(int(confidence * 100), boot_mean)
            ra = max(pctl - boot_mean, 0); arr = result['ibnr_nominal_samples']; phi = result.get('phi', 0)
        else:
            obs = np.zeros((n_ay, n_dev), dtype=bool)
            for i in range(n_ay):
                for j in range(n_dev):
                    if i + j < n_ay and not np.isnan(C[i, j]): obs[i, j] = True
            C_filled = np.where(np.isnan(C), 0.0, C)
            def vw_facs(cm, om):
                f = []
                for j in range(n_dev - 1):
                    num = den = 0.0
                    for i in range(n_ay):
                        if i + j + 1 < n_ay and om[i, j] and om[i, j + 1]:
                            if cm[i, j] > 0: num += cm[i, j + 1]; den += cm[i, j]
                    f.append(num / den if den > 0 else 1.0)
                return f
            def project_triangle(wc, facs):
                p = wc.copy().astype(float)
                for i in range(n_ay):
                    last_obs = -1
                    for j in range(n_dev - 1, -1, -1):
                        if i + j < n_ay: last_obs = j; break
                    if last_obs < 0: continue
                    for j in range(last_obs, n_dev - 1):
                        if j < len(facs): p[i, j + 1] = p[i, j] * facs[j] if p[i, j] > 0 else 0.0
                return p
            facs = vw_facs(C_filled, obs); comp_det = project_triangle(C_filled, facs)
            fit_inc = comp_det.copy()
            for i in range(n_ay):
                for j in range(n_dev - 1, 0, -1): fit_inc[i, j] = comp_det[i, j] - comp_det[i, j - 1]
            resids = []
            for i in range(n_ay):
                for j in range(n_dev):
                    if i + j < n_ay and obs[i, j]:
                        actual = (C_filled[i, j] - C_filled[i, j - 1]) if j > 0 else C_filled[i, j]
                        fitted = fit_inc[i, j]
                        r = (actual - fitted) / np.sqrt(abs(fitted)) if fitted > 0 else 0.0
                        resids.append(r)
            resids = np.array(resids); n_obs = len(resids)
            phi = max(np.sum(resids ** 2) / max(n_obs - n_dev + 1, 1), 0.01)
            samples = []; progress_bar = st.progress(0)
            for it in range(n_iter):
                samp = np.random.choice(resids, size=n_obs, replace=True)
                ps = fit_inc.copy().astype(float); idx = 0
                for i in range(n_ay):
                    for j in range(n_dev):
                        if i + j < n_ay and obs[i, j]:
                            fv = fit_inc[i, j]; pv = fv + samp[idx] * np.sqrt(max(abs(fv), 0.001))
                            ps[i, j] = max(pv, 0.0); idx += 1
                pc = np.cumsum(ps, axis=1); pf = vw_facs(pc, obs); pc2 = project_triangle(pc, pf)
                if add_pv and phi > 1e-10:
                    pi = pc2.copy()
                    for i in range(n_ay):
                        for j in range(n_dev - 1, 0, -1): pi[i, j] = pc2[i, j] - pc2[i, j - 1]
                    for i in range(n_ay):
                        for j in range(n_dev):
                            if (i + j >= n_ay) or (not obs[i, j]):
                                mv = pi[i, j]
                                if not np.isnan(mv) and mv > 0: pi[i, j] = max(np.random.gamma(mv / phi, phi), 0.0)
                                else: pi[i, j] = 0.0
                    pc2 = np.cumsum(pi, axis=1)
                total = 0.0
                for i in range(n_ay):
                    last_obs = -1
                    for j in range(n_dev - 1, -1, -1):
                        if i + j < n_ay and obs[i, j]: last_obs = j; break
                    if last_obs >= 0: total += max(pc2[i, n_dev - 1] - pc[i, last_obs], 0.0)
                samples.append(total)
                if (it + 1) % 100 == 0: progress_bar.progress((it + 1) / n_iter)
            progress_bar.empty(); arr = np.array(samples)
            cl_ibnr = 0
            for i in range(n_ay):
                last_obs = -1
                for j in range(n_dev - 1, -1, -1):
                    if i + j < n_ay and not np.isnan(C[i, j]): last_obs = j; break
                if last_obs >= 0: cl_ibnr += max(comp_det[i, n_dev - 1] - C_filled[i, last_obs], 0.0)
            boot_mean = float(np.mean(arr)); pctl = float(np.percentile(arr, confidence * 100))
            ra = max(pctl - boot_mean, 0.0)
    st.markdown(f"### Bootstrap Results at {confidence:.0%} Confidence")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("CL IBNR", f"{cl_ibnr:,.2f}")
    with c2: st.metric("Bootstrap Mean", f"{boot_mean:,.2f}")
    with c3: st.metric(f"P{confidence*100:.0f} Percentile", f"{pctl:,.2f}")
    with c4: st.metric("Risk Adjustment", f"{ra:,.2f}")
    st.caption(f"Phi: {phi:.4f} | Iterations: {n_iter:,} | Process Variance: {'Yes' if add_pv else 'No'}")
    st.markdown("#### IBNR Distribution")
    counts, bins = np.histogram(arr, bins=30)
    hist_df = pd.DataFrame({'Bin_Start': bins[:-1], 'Count': counts}).set_index('Bin_Start')
    st.bar_chart(hist_df)
    output = BytesIO()
    pd.DataFrame({'IBNR_Samples': arr}).to_excel(output, index=False)
    output.seek(0); sc = sanitize_filename(client_name)
    st.download_button("Download Bootstrap Samples", data=output, file_name=f"{sc}_Bootstrap_Samples.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="bts_dl")


def render_bootstrap_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>ODP Bootstrap - Risk Adjustment</h1><p>England & Verrall (1999) Bootstrap with Process Variance</p></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: client_name = st.text_input("Client Name", value="Client", key="bts_cn").strip()
    with c2: confidence = st.number_input("Confidence Level (%)", 50.0, 99.5, 75.0, 1.0, key="bts_cl") / 100.0
    with c3: n_iter = st.number_input("Iterations", 100, 10000, 1000, 100, key="bts_it")
    with c4: add_pv = st.checkbox("Process Variance", value=True, key="bts_pv")
    input_type = st.radio("Input Data Type", ["Claims-Level Data", "Cumulative Triangle"], key="bts_input_type")
    if input_type == "Cumulative Triangle":
        uploaded = st.file_uploader("Upload Cumulative Claims Triangle (CSV/Excel)", type=["csv", "xlsx", "xls"], key="bts_f_tri")
        st.caption("First column = AY labels, remaining = cumulative claims.")
        if uploaded is None: back_button('risk_adjustment', ['Home', 'LIC Calculators', 'Risk Adjustment']); return
        try:
            raw = read_uploaded_triangle(uploaded)
            st.dataframe(raw, use_container_width=True)
            grain = "Y"; ppy = 1
            st.markdown("### Adjustments")
            c1, c2 = st.columns(2)
            with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key="bts_inf_tri")
            with c2: use_discounting = st.checkbox("Apply Discounting", key="bts_disc_tri")
            cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
            if use_inflation: cum_inflation, per_period_rates = load_inflation_data_ui(grain, ppy, "bts_tri")
            if use_discounting: spot_rates, flat_rate = load_discounting_data_ui(grain, ppy, "bts_tri")
            if st.button(f"Run Bootstrap ({n_iter:,} iterations)", key="bts_run_tri", use_container_width=True):
                run_bootstrap_calculation(raw, confidence, n_iter, add_pv, client_name, use_inflation, cum_inflation, per_period_rates, use_discounting, spot_rates, flat_rate, grain, origin_date=pd.Timestamp('2020-01-01'))
        except Exception as e: st.error(f"Error: {e}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: grain = st.selectbox("Period Grain", ["Yearly", "Quarterly", "Monthly"], key="bts_gr")
        with c2: from_date = st.date_input("From Date", date(2020, 1, 1), key="bts_fd")
        with c3: to_date = st.date_input("To Date", date(2025, 12, 31), key="bts_td")
        grain_map = {"Yearly": "Y", "Quarterly": "Q", "Monthly": "M"}; grain_code = grain_map[grain]; ppy = {"Y": 1, "Q": 4, "M": 12}[grain_code]
        uploaded = st.file_uploader("Upload Claims Data (Loss Date, Report Date, Amount)", type=["csv", "xlsx", "xls"], key="bts_f_cl")
        if uploaded is None: back_button('risk_adjustment', ['Home', 'LIC Calculators', 'Risk Adjustment']); return
        try:
            df = read_uploaded_file(uploaded)
            df.columns = df.columns.astype(str).str.strip()
            st.dataframe(df.head(5), use_container_width=True)
            cols = df.columns.tolist()
            c1, c2 = st.columns(2)
            with c1: loss_col = st.selectbox("Loss Date", cols, key="bts_ld")
            with c2: rep_col = st.selectbox("Report Date", cols, key="bts_rd")
            amount_col = st.selectbox("Amount Column", cols, key="bts_amt")
            df[loss_col] = pd.to_datetime(df[loss_col], errors='coerce'); df[rep_col] = pd.to_datetime(df[rep_col], errors='coerce')
            df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
            df = df.dropna(subset=[loss_col, rep_col])
            from_dt = pd.Timestamp(str(from_date)); to_dt = pd.Timestamp(str(to_date))
            df = _date_filter(df, loss_col, from_date, to_date)
            n_periods = int((to_dt.year - from_dt.year) * ppy) + 1
            if engine_utils is not None:
                _, cum_triangle, _ = engine_utils.build_triangles(df, loss_col, rep_col, amount_col, from_dt, grain_code, n_periods)
                st.markdown("#### Built Triangle"); st.dataframe(cum_triangle, use_container_width=True)
                st.markdown("### Adjustments")
                c1, c2 = st.columns(2)
                with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key="bts_inf_cl")
                with c2: use_discounting = st.checkbox("Apply Discounting", key="bts_disc_cl")
                cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
                if use_inflation: cum_inflation, per_period_rates = load_inflation_data_ui(grain_code, ppy, "bts_cl")
                if use_discounting: spot_rates, flat_rate = load_discounting_data_ui(grain_code, ppy, "bts_cl")
                if st.button(f"Run Bootstrap ({n_iter:,} iterations)", key="bts_run_cl", use_container_width=True):
                    run_bootstrap_calculation(cum_triangle, confidence, n_iter, add_pv, client_name, use_inflation, cum_inflation, per_period_rates, use_discounting, spot_rates, flat_rate, grain_code, origin_date=from_dt)
            else: st.error("Engine utils not available.")
        except Exception as e: st.error(f"Error: {e}")
    back_button('risk_adjustment', ['Home', 'LIC Calculators', 'Risk Adjustment'])


# =============================================================================
#  FULL VALUATION (IFRS 17 Engine only) – unchanged
# =============================================================================

def render_full_valuation():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Full IFRS 17 Valuation</h1><p>Complete PAA LRC Rollforward</p></div>', unsafe_allow_html=True)
    
    if full_engine is None:
        st.error("Full Valuation engine not available.")
        back_button('home', ['Home'])
        return
    
    # --- Report Metadata ---
    st.markdown("### Report Metadata")
    with st.container():
        st.markdown('<div class="report-meta">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            client = st.text_input("Client Name", value=st.session_state.report_metadata.get('client', ''), key="fv_meta_client")
            st.session_state.report_metadata['client'] = client
        with c2:
            val_date = st.date_input("Valuation Date", value=st.session_state.report_metadata.get('val_date', date(2025,12,31)), key="fv_meta_valdate")
            st.session_state.report_metadata['val_date'] = val_date
        with c3:
            report_title = st.text_input("Report Title", value=st.session_state.report_metadata.get('title', 'IFRS 17 Valuation Report'), key="fv_meta_title")
            st.session_state.report_metadata['title'] = report_title
        with c4:
            prepared_by = st.text_input("Prepared By", value=st.session_state.report_metadata.get('prepared_by', ''), key="fv_meta_prepared")
            st.session_state.report_metadata['prepared_by'] = prepared_by
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### Configuration")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        iacf_toggle = st.selectbox("IACF Treatment", ["Expense Immediately", "Capitalize & Amortize"], key="fv_iacf")
    with c2:
        discount_toggle = st.selectbox("Discounting", ["No Discounting", "Apply Discounting"], key="fv_disc")
    with c3:
        invest_toggle = st.selectbox("Investment Components", ["No", "Yes"], key="fv_inv")
    with c4:
        revenue_toggle = st.selectbox("Revenue Recognition", ["Passage of Time", "Emergence of Risk"], key="fv_rev")
    
    config = {
        'iacf_toggle': iacf_toggle,
        'discount_toggle': discount_toggle,
        'invest_toggle': invest_toggle,
        'revenue_toggle': revenue_toggle
    }
    
    st.markdown("### Upload Files")
    
    c1, c2 = st.columns(2)
    with c1:
        opening_file = st.file_uploader("1. Opening Balances (Group, Opening_LRC_Excl_Loss, Opening_Loss_Component)", type=["csv","xlsx"], key="fv_ob")
        policy_file = st.file_uploader("3. Policy Data (Group, Start_Date, End_Date, Written_Premium)", type=["csv","xlsx"], key="fv_pol")
    with c2:
        cashflows_file = st.file_uploader("2. Cashflows (Group, Premiums_Received, IACF_Paid, Investment_Components_Paid)", type=["csv","xlsx"], key="fv_cf")
        if revenue_toggle == "Emergence of Risk":
            claims_curve_file = st.file_uploader("6. Claims Curve (Period, Percentage) – Optional", type=["csv","xlsx"], key="fv_cc")
        else:
            claims_curve_file = None
    
    # Discounting data – only when toggled on
    yield_curve_df = None
    if discount_toggle == "Apply Discounting":
        st.markdown("#### Discounting Data (Yield Curve)")
        yc_file = st.file_uploader("Upload Yield Curve (Duration_Years, Spot_Rate %)", type=["csv","xlsx"], key="fv_yc_disc")
        if yc_file is not None:
            try:
                yc_df = read_uploaded_file(yc_file)
                yc_df.columns = yc_df.columns.astype(str).str.strip()
                st.markdown("**Yield Curve Column Mapping:**")
                yc_map = map_columns(yc_df, ['Duration_Years', 'Spot_Rate'], 'fv_yc')
                yield_curve_df = yc_df.rename(columns={v: k for k, v in yc_map.items()})
            except Exception as e:
                st.error(f"Yield curve error: {e}")
    
    # Loss Component – always via engine
    st.markdown("#### Loss Component (computed via engine)")
    lc_data_file = st.file_uploader(
        "Upload Loss Component Input Data (LOB, Written Premium, Expenses, Commission, Paid Claims, Opening/Closing OCR, IBNR, UPR, RA)",
        type=["csv","xlsx","xls"], key="fv_lc_data"
    )
    loss_comp_df = None
    lc_computed = False
    
    if lc_data_file is not None:
        try:
            lc_raw = read_uploaded_file(lc_data_file)
            lc_raw.columns = lc_raw.columns.astype(str).str.strip()
            st.dataframe(lc_raw.head(3), use_container_width=True)
            
            cols = lc_raw.columns.tolist()
            st.markdown("**Loss Component Column Mapping:**")
            c1,c2,c3 = st.columns(3)
            with c1:
                lob_col = st.selectbox("Line of Business", cols, key="fv_lc_lob")
                opening_ocr_col = st.selectbox("Opening OCR", cols, key="fv_lc_oocr")
                opening_ibnr_col = st.selectbox("Opening IBNR", cols, key="fv_lc_oibnr")
            with c2:
                wp_col = st.selectbox("Written Premium", cols, key="fv_lc_wp")
                closing_ocr_col = st.selectbox("Closing OCR", cols, key="fv_lc_cocr")
                closing_ibnr_col = st.selectbox("Closing IBNR", cols, key="fv_lc_cibnr")
            with c3:
                commission_col = st.selectbox("Commission Paid", cols, key="fv_lc_comm")
                paid_claims_col = st.selectbox("Paid Claims", cols, key="fv_lc_pc")
                ra_col = st.selectbox("Risk Adjustment", cols, key="fv_lc_ra")
            c1,c2 = st.columns(2)
            with c1:
                expenses_col = st.selectbox("Expenses", cols, key="fv_lc_exp")
            with c2:
                opening_upr_col = st.selectbox("Opening UPR", cols, key="fv_lc_oupr")
            closing_upr_col = st.selectbox("Closing UPR", cols, key="fv_lc_cupr")
            
            if st.button("Compute Loss Component", key="fv_lc_calc"):
                if loss_comp_engine is None:
                    st.error("Loss Component engine not available.")
                else:
                    lc_result = loss_comp_engine.calculate_loss_component(
                        df=lc_raw, lob_col=lob_col, written_premium_col=wp_col,
                        expenses_col=expenses_col, commission_col=commission_col,
                        paid_claims_col=paid_claims_col,
                        opening_ocr_col=opening_ocr_col, closing_ocr_col=closing_ocr_col,
                        opening_ibnr_col=opening_ibnr_col, closing_ibnr_col=closing_ibnr_col,
                        opening_upr_col=opening_upr_col, closing_upr_col=closing_upr_col,
                        risk_adjustment_col=ra_col
                    )
                    st.session_state['lc_computed'] = lc_result
                    lc_computed = True
                    st.success("Loss Component computed. Ready for valuation.")
                    st.dataframe(lc_result)
        except Exception as e:
            st.error(f"Loss Component error: {e}")
    
    required = [opening_file, cashflows_file, policy_file, lc_data_file]
    if all(f is not None for f in required):
        try:
            opening_df = read_uploaded_file(opening_file)
            opening_df.columns = opening_df.columns.astype(str).str.strip()
            cashflows_df = read_uploaded_file(cashflows_file)
            cashflows_df.columns = cashflows_df.columns.astype(str).str.strip()
            policy_df = read_uploaded_file(policy_file)
            policy_df.columns = policy_df.columns.astype(str).str.strip()
            
            st.markdown("### Column Mapping")
            st.markdown("**Opening Balances:**")
            ob_map = map_columns(opening_df, ['Group','Opening_LRC_Excl_Loss','Opening_Loss_Component'], 'fv_ob')
            opening_df = opening_df.rename(columns={v:k for k,v in ob_map.items()})
            st.markdown("**Cashflows:**")
            cf_map = map_columns(cashflows_df, ['Group','Premiums_Received','IACF_Paid','Investment_Components_Paid'], 'fv_cf')
            cashflows_df = cashflows_df.rename(columns={v:k for k,v in cf_map.items()})
            st.markdown("**Policy Data:**")
            pol_map = map_columns(policy_df, ['Group','Start_Date','End_Date','Written_Premium'], 'fv_pol')
            policy_df = policy_df.rename(columns={v:k for k,v in pol_map.items()})
            
            # Build loss_comp_df from computed result + expected future premiums
            if not lc_computed and 'lc_computed' not in st.session_state:
                st.warning("Please compute the Loss Component first.")
                return
            lc_res = st.session_state.get('lc_computed')
            st.markdown("**Expected Future Premiums** – required for full valuation.")
            efp_file = st.file_uploader("Upload Expected Future Premiums (Group, Amount)", type=["csv","xlsx"], key="fv_efp")
            if efp_file is None:
                st.info("Upload the Expected Future Premiums file to proceed.")
                return
            efp_df = read_uploaded_file(efp_file)
            efp_df.columns = efp_df.columns.astype(str).str.strip()
            efp_map = map_columns(efp_df, ['Group','Expected_Future_Premiums'], 'fv_efp')
            efp_df = efp_df.rename(columns={v:k for k,v in efp_map.items()})
            
            # Merge with lc_res (which used lob_col as key)
            lc_res = lc_res.rename(columns={lob_col: 'Group'})
            lc_res = lc_res[['Group','Loss_Ratio','Commission_Ratio','Expense_Ratio','Risk_Adjustment_Ratio']]
            lc_res = lc_res.rename(columns={'Risk_Adjustment_Ratio':'RA_Ratio'})
            loss_comp_df = lc_res.merge(efp_df, on='Group')
            
            # Claims curve (optional)
            claims_curve_df = None
            if claims_curve_file is not None:
                cc_df = read_uploaded_file(claims_curve_file)
                cc_df.columns = cc_df.columns.astype(str).str.strip()
                cc_map = map_columns(cc_df, ['Period','Percentage'], 'fv_cc')
                claims_curve_df = cc_df.rename(columns={v:k for k,v in cc_map.items()})
            
            valuation_date = st.session_state.report_metadata.get('val_date', date(2025,12,31))
            
            if st.button("Run Full Valuation", key="fv_run", use_container_width=True):
                with st.spinner("Running IFRS 17 Valuation..."):
                    results = full_engine.calculate_full_ifrs17_lrc(
                        opening_balances_df=opening_df,
                        cashflows_df=cashflows_df,
                        policy_df=policy_df,
                        loss_component_df=loss_comp_df,
                        yield_curve_df=yield_curve_df,
                        claims_curve_df=claims_curve_df,
                        config=config,
                        valuation_date=valuation_date
                    )
                st.markdown("### Valuation Results by Group")
                for group, data in results.items():
                    with st.expander(f"Group: {group}", expanded=True):
                        c1,c2,c3 = st.columns(3)
                        with c1:
                            st.metric("Opening LRC (excl. Loss)", f"{data['Opening_LRC_Excl_Loss']:,.2f}")
                            st.metric("Premiums Received", f"{data['Premiums_Received']:,.2f}")
                            st.metric("Insurance Revenue", f"{data['Insurance_Revenue']:,.2f}")
                        with c2:
                            st.metric("Opening Loss Component", f"{data['Opening_Loss_Component']:,.2f}")
                            st.metric("IACF Paid", f"{data['IACF_Paid']:,.2f}")
                            st.metric("IACF Amortized", f"{data['IACF_Amortized']:,.2f}")
                        with c3:
                            st.metric("Closing LRC (excl. Loss)", f"{data['Closing_LRC_Excl_Loss']:,.2f}")
                            st.metric("Closing Loss Component", f"{data['Closing_Loss_Component']:,.2f}")
                            st.metric("Total Closing LRC", f"{data['Total_Closing_LRC']:,.2f}")
                        st.markdown(f"**Combined Ratio:** {data.get('Combined_Ratio',0):.2%} | **UPR Snapshot:** {data.get('UPR_Snapshot',0):,.2f}")
        except Exception as e:
            st.error(f"Error: {e}")
            with st.expander("Details"): import traceback; st.code(traceback.format_exc())
    else:
        st.info("Please upload all required files (Opening Balances, Cashflows, Policy, Loss Component Input).")
    
    back_button('home', ['Home'])


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
        'capecod_calculator': render_capecod_calculator,
        'bf_calculator': render_bf_calculator,
        'ulae_calculator': render_ulae_calculator,
        'npr_calculator': render_npr_calculator,
        'mack_calculator': render_mack_calculator,
        'bootstrap_calculator': render_bootstrap_calculator,
        'full_valuation': render_full_valuation,
    }
    if page in page_routes:
        page_routes[page]()
    else:
        st.error(f"Unknown page: {page}")
        navigate_to('home', ['Home'])


if __name__ == "__main__":
    main()
