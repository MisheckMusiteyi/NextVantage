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
for pattern in ["Full_Valuation/full_LRC_IFRS17.py", "Full_Valuation/*.py"]:
    full_engine = import_file_glob(pattern)
    if full_engine is not None:
        break
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
    inf_file = st.file_uploader("Upload Inflation Curve (CSV/Excel: Period, Rate %)", type=["csv","xlsx","xls"], key=f"inf_{page_key}")
    cum_inflation = None
    per_period_rates = None
    if inf_file:
        try:
            inf_df = pd.read_csv(inf_file) if inf_file.name.endswith('.csv') else pd.read_excel(inf_file)
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
    disc_method = st.radio("Discounting Method", ["None", "Single Flat Rate", "Yield Curve"], key=f"disc_m_{page_key}", horizontal=True)
    spot_rates = None
    flat_rate = None
    if disc_method == "Yield Curve":
        yc_file = st.file_uploader("Upload Yield Curve (CSV/Excel: Duration_Years, Spot_Rate %)", type=["csv","xlsx","xls"], key=f"yc_{page_key}")
        if yc_file:
            try:
                yc_df = pd.read_csv(yc_file) if yc_file.name.endswith('.csv') else pd.read_excel(yc_file)
                yc_df.columns = yc_df.columns.astype(str).str.strip()
                c1, c2 = st.columns(2)
                with c1: m_col = st.selectbox("Duration Column", yc_df.columns, key=f"yc_m_{page_key}")
                with c2: r_col = st.selectbox("Spot Rate Column (%)", yc_df.columns, key=f"yc_r_{page_key}")
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
                st.success("Yield curve loaded and interpolated.")
            except Exception as e:
                st.error(f"Yield curve error: {e}")
    elif disc_method == "Single Flat Rate":
        flat_rate = st.number_input("Annual Discount Rate (%)", 0.0, 50.0, 5.0, 0.5, key=f"flat_{page_key}") / 100.0
    return spot_rates, flat_rate


# =============================================================================
#  STREAMLIT CONFIGURATION
# =============================================================================

st.set_page_config(page_title="Next Vantage Actuarial Toolkit", layout="wide", initial_sidebar_state="expanded")

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
    .report-meta { background-color: #F0F4F8; border: 2px solid #4A90D9; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; font-size: 0.85rem; }
    .report-meta td { padding: 2px 8px; }
    .footer { background-color: #000000; color: #FFFFFF; text-align: center; padding: 1.5rem; border-top: 3px solid #4A90D9; margin-top: 3rem; font-size: 0.9rem; }
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
#  NAVIGATION PAGES (unchanged except Full Valuation)
# =============================================================================

# (All navigation functions remain exactly the same as in your working version.
#  I'll only show render_full_valuation() below; the rest is identical to the app.py you pasted.)

# =============================================================================
#  CALCULATOR: FULL VALUATION (UPDATED – conditional uploads, loss component engine)
# =============================================================================

def render_full_valuation():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Full IFRS 17 Valuation</h1><p>Complete PAA LRC Rollforward & Simple PAA</p></div>', unsafe_allow_html=True)
    
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
    
    # --- Mode Selection ---
    mode = st.radio("Valuation Mode", ["Full Valuation (All Files)", "Simple PAA (UPR Rollforward)"], key="fv_mode")
    
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
    
    # ------------------ Full Valuation mode ------------------
    if mode == "Full Valuation (All Files)":
        c1, c2 = st.columns(2)
        with c1:
            opening_file = st.file_uploader("1. Opening Balances (Group, Opening_LRC_Excl_Loss, Opening_Loss_Component)", type=["csv","xlsx"], key="fv_ob")
            policy_file = st.file_uploader("3. Policy Data (Group, Start_Date, End_Date, Written_Premium)", type=["csv","xlsx"], key="fv_pol")
        with c2:
            cashflows_file = st.file_uploader("2. Cashflows (Group, Premiums_Received, IACF_Paid, Investment_Components_Paid)", type=["csv","xlsx"], key="fv_cf")
            claims_curve_file = st.file_uploader("6. Claims Curve (Period, Percentage) - Optional", type=["csv","xlsx"], key="fv_cc")
        
        # ----- Discounting data – only when toggled on -----
        yield_curve_df = None
        if discount_toggle == "Apply Discounting":
            st.markdown("#### Discounting Data (Yield Curve)")
            yc_file = st.file_uploader("Upload Yield Curve (Duration_Years, Spot_Rate %)", type=["csv","xlsx"], key="fv_yc_disc")
            if yc_file is not None:
                try:
                    yc_df = pd.read_csv(yc_file) if yc_file.name.endswith('.csv') else pd.read_excel(yc_file)
                    yc_df.columns = yc_df.columns.astype(str).str.strip()
                    st.markdown("**Yield Curve Column Mapping:**")
                    yc_map = map_columns(yc_df, ['Duration_Years', 'Spot_Rate'], 'fv_yc')
                    yield_curve_df = yc_df.rename(columns={v: k for k, v in yc_map.items()})
                except Exception as e:
                    st.error(f"Yield curve error: {e}")
        
        # ----- Loss Component: engine or pre‑computed -----
        st.markdown("#### Loss Component")
        lc_method = st.radio("Loss Component Input", ["Compute from data (engine)", "Upload pre‑computed file"], key="fv_lc_method")
        loss_comp_df = None
        
        if lc_method == "Compute from data (engine)":
            lc_data_file = st.file_uploader(
                "Upload Loss Component Input Data (LOB, Written Premium, Expenses, Commission, Paid Claims, Opening/Closing OCR, IBNR, UPR, RA)",
                type=["csv","xlsx","xls"], key="fv_lc_data"
            )
            if lc_data_file is not None:
                try:
                    lc_raw = pd.read_csv(lc_data_file) if lc_data_file.name.endswith('.csv') else pd.read_excel(lc_data_file)
                    lc_raw.columns = lc_raw.columns.astype(str).str.strip()
                    st.dataframe(lc_raw.head(3), use_container_width=True)
                    
                    # Column mapping identical to individual Loss Component calculator
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
                            st.success("Loss Component computed.")
                            st.dataframe(lc_result)
                except Exception as e:
                    st.error(f"Loss Component error: {e}")
        else:  # upload pre‑computed
            lc_file = st.file_uploader(
                "4. Loss Component Data (Group, Expected_Future_Premiums, Loss_Ratio, Commission_Ratio, Expense_Ratio, RA_Ratio)",
                type=["csv","xlsx"], key="fv_lc"
            )
            if lc_file is not None:
                try:
                    lc_df = pd.read_csv(lc_file) if lc_file.name.endswith('.csv') else pd.read_excel(lc_file)
                    lc_df.columns = lc_df.columns.astype(str).str.strip()
                    st.markdown("**Loss Component Column Mapping:**")
                    lc_map = map_columns(lc_df, ['Group','Expected_Future_Premiums','Loss_Ratio','Commission_Ratio','Expense_Ratio','RA_Ratio'], 'fv_lc')
                    loss_comp_df = lc_df.rename(columns={v:k for k,v in lc_map.items()})
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Required files check
        required = [opening_file, cashflows_file, policy_file]
        if lc_method == "Upload pre‑computed file":
            required.append(lc_file)
        else:
            # In compute mode we need the computed data in session state
            pass
        
        if all(f is not None for f in required):
            try:
                # Read and map mandatory files
                opening_df = pd.read_csv(opening_file) if opening_file.name.endswith('.csv') else pd.read_excel(opening_file)
                opening_df.columns = opening_df.columns.astype(str).str.strip()
                cashflows_df = pd.read_csv(cashflows_file) if cashflows_file.name.endswith('.csv') else pd.read_excel(cashflows_file)
                cashflows_df.columns = cashflows_df.columns.astype(str).str.strip()
                policy_df = pd.read_csv(policy_file) if policy_file.name.endswith('.csv') else pd.read_excel(policy_file)
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
                
                # Handle Loss Component
                if lc_method == "Compute from data (engine)":
                    if 'lc_computed' not in st.session_state:
                        st.warning("Please compute the Loss Component first.")
                        return
                    lc_res = st.session_state['lc_computed']
                    # We need to convert the engine output to the format expected by full_LRC:
                    # The engine returns columns: <lob_col>, Total_Written_Premiums, Total_Earned_Premiums, ...
                    # and Loss_Ratio, Commission_Ratio, Expense_Ratio, Risk_Adjustment_Ratio, Combined_Ratio, Loss_Component, Closing_UPR, etc.
                    # But full_LRC requires: Group, Expected_Future_Premiums, Loss_Ratio, Commission_Ratio, Expense_Ratio, RA_Ratio
                    # Expected_Future_Premiums is not directly in the output; we'll ask the user to provide it.
                    st.markdown("**Expected Future Premiums** – required for full valuation.")
                    efp_file = st.file_uploader("Upload Expected Future Premiums (Group, Amount)", type=["csv","xlsx"], key="fv_efp")
                    if efp_file is None:
                        st.info("Upload the Expected Future Premiums file to proceed.")
                        return
                    efp_df = pd.read_csv(efp_file) if efp_file.name.endswith('.csv') else pd.read_excel(efp_file)
                    efp_df.columns = efp_df.columns.astype(str).str.strip()
                    efp_map = map_columns(efp_df, ['Group','Expected_Future_Premiums'], 'fv_efp')
                    efp_df = efp_df.rename(columns={v:k for k,v in efp_map.items()})
                    # Merge with lc_res using the LOB column (which is the lob_col name from earlier mapping)
                    lc_res = lc_res.rename(columns={lob_col: 'Group'})
                    lc_res = lc_res[['Group','Loss_Ratio','Commission_Ratio','Expense_Ratio','Risk_Adjustment_Ratio']]
                    lc_res = lc_res.rename(columns={'Risk_Adjustment_Ratio': 'RA_Ratio'})
                    loss_comp_df = lc_res.merge(efp_df, on='Group')
                else:
                    # loss_comp_df already set from uploaded file
                    pass
                
                # Claims curve (optional)
                claims_curve_df = None
                if claims_curve_file is not None:
                    cc_df = pd.read_csv(claims_curve_file) if claims_curve_file.name.endswith('.csv') else pd.read_excel(claims_curve_file)
                    cc_df.columns = cc_df.columns.astype(str).str.strip()
                    cc_map = map_columns(cc_df, ['Period','Percentage'], 'fv_cc')
                    claims_curve_df = cc_df.rename(columns={v:k for k,v in cc_map.items()})
                
                valuation_date = st.session_state.report_metadata.get('val_date', date(2025,12,31))
                
                if st.button("Run Valuation", key="fv_run", use_container_width=True):
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
            st.info("Please upload all required files.")
    
    # ------------------ Simple PAA mode (unchanged) ------------------
    else:
        c1, c2 = st.columns(2)
        with c1:
            policy_file = st.file_uploader("Policy Data (Group, Start_Date, End_Date, Written_Premium)", type=["csv","xlsx"], key="fv_pol_simple")
            opening_file = st.file_uploader("Opening Balances (Optional)", type=["csv","xlsx"], key="fv_ob_simple")
        with c2:
            cashflows_file = st.file_uploader("Cashflows (Optional)", type=["csv","xlsx"], key="fv_cf_simple")
        
        if policy_file is not None:
            try:
                policy_df = pd.read_csv(policy_file) if policy_file.name.endswith('.csv') else pd.read_excel(policy_file)
                policy_df.columns = policy_df.columns.astype(str).str.strip()
                
                opening_df = pd.DataFrame(columns=['Group','Opening_LRC_Excl_Loss','Opening_Loss_Component'])
                cashflows_df = pd.DataFrame(columns=['Group','Premiums_Received','IACF_Paid','Investment_Components_Paid'])
                
                st.markdown("### Column Mapping")
                st.markdown("**Policy Data:**")
                pol_map = map_columns(policy_df, ['Group','Start_Date','End_Date','Written_Premium'], 'fv_pol_s')
                policy_df = policy_df.rename(columns={v:k for k,v in pol_map.items()})
                
                if opening_file is not None:
                    opening_df = pd.read_csv(opening_file) if opening_file.name.endswith('.csv') else pd.read_excel(opening_file)
                    opening_df.columns = opening_df.columns.astype(str).str.strip()
                    ob_map = map_columns(opening_df, ['Group','Opening_LRC_Excl_Loss','Opening_Loss_Component'], 'fv_ob_s')
                    opening_df = opening_df.rename(columns={v:k for k,v in ob_map.items()})
                if cashflows_file is not None:
                    cashflows_df = pd.read_csv(cashflows_file) if cashflows_file.name.endswith('.csv') else pd.read_excel(cashflows_file)
                    cashflows_df.columns = cashflows_df.columns.astype(str).str.strip()
                    cf_map = map_columns(cashflows_df, ['Group','Premiums_Received','IACF_Paid','Investment_Components_Paid'], 'fv_cf_s')
                    cashflows_df = cashflows_df.rename(columns={v:k for k,v in cf_map.items()})
                
                valuation_date = st.session_state.report_metadata.get('val_date', date(2025,12,31))
                
                if st.button("Run Valuation", key="fv_run_simple", use_container_width=True):
                    with st.spinner("Running..."):
                        results = full_engine.calculate_full_ifrs17_lrc(
                            opening_balances_df=opening_df,
                            cashflows_df=cashflows_df,
                            policy_df=policy_df,
                            loss_component_df=pd.DataFrame(columns=['Group','Expected_Future_Premiums','Loss_Ratio','Commission_Ratio','Expense_Ratio','RA_Ratio']),
                            yield_curve_df=None,
                            claims_curve_df=None,
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
        else:
            st.info("Please upload the policy data file.")
    
    back_button('home', ['Home'])


# =============================================================================
#  MAIN ROUTER (all other routes unchanged)
# =============================================================================

# ... (rest of app.py identical to your working version – navigation pages, individual calculators, etc.)

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
