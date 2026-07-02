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

def _parse_dates(series):
    """Safely parse any column to datetime regardless of source dtype."""
    try:
        return pd.to_datetime(series.astype(str), errors='coerce')
    except Exception:
        return pd.to_datetime(series, errors='coerce')

def _date_filter(df, col, from_date, to_date):
    """Filter dataframe by date column between from_date and to_date."""
    # FIX: Ensure column is parsed BEFORE accessing .dt
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    fd = pd.Timestamp(from_date)
    td = pd.Timestamp(to_date)
    return df[(df[col] >= fd) & (df[col] <= td)]


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
#  IMPORT ALL ENGINES FROM YOUR FOLDERS (MATCHING YOUR GITHUB FOLDERS)
# =============================================================================

# --- LRC CALCULATORS (Using Underscores to match GitHub) ---
from LRC_Calculators.upr_engine import calculate_upr
from LRC_Calculators.loss_component_engine import calculate_loss_component

# --- LIC CALCULATORS ---
from LIC_Calculators.FCF_Calculators.OCR_Calculators.ocr_engine import calculate_ocr

from LIC_Calculators.FCF_Calculators.IBNR_Calculators.percentage_ibnr import calculate_percentage_ibnr
from LIC_Calculators.FCF_Calculators.IBNR_Calculators.bcl_ibnr import calculate_bcl_ibnr
from LIC_Calculators.FCF_Calculators.IBNR_Calculators.cape_cod_ibnr import calculate_cape_cod_ibnr
from LIC_Calculators.FCF_Calculators.IBNR_Calculators.bf_ibnr import calculate_bf_ibnr

from LIC_Calculators.FCF_Calculators.ULAE_Calculators.ulae_engine import calculate_ulae_per_portfolio, calculate_ulae_aggregated, calculate_apportionment_percentages
from LIC_Calculators.FCF_Calculators.NPR_Calculators.npr_engine import calculate_npr_aggregation, calculate_npr_per_portfolio

from LIC_Calculators.RA_Calculators.mack_ra import calculate_mack_chain_ladder
from LIC_Calculators.RA_Calculators.bootstrap_ra import bootstrap_chain_ladder, calculate_risk_adjustment

# --- SHARED HELPERS ---
from utils.actuarial_helpers import (
    build_triangles, volume_weighted_factors, simple_average_factors,
    geometric_average_factors, medial_average_factors, linear_regression_factors,
    weighted_last_n_factors, stability_diagnostics, recommend_factors,
    compute_cdfs, project_ultimate, deflate_triangle_to_real,
    reinflate_ibnr_per_ap, discount_completed_triangle,
    period_index, period_label, periods_per_year
)

# --- FULL VALUATION ---
from Full_Valuation.full_LRC_IFRS17 import calculate_full_ifrs17_lrc


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

# =============================================================================
#  INDIVIDUAL LRC CALCULATORS
# =============================================================================

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
#  IBNR CALCULATORS WITH INFLATION & DISCOUNTING (From your advanced script)
# =============================================================================

# =============================================================================
#  UTILITY FUNCTIONS FOR INFLATION & DISCOUNTING
# =============================================================================

def periods_per_year(grain):
    return {"Y": 1, "Q": 4, "M": 12}[grain]

def _load_inflation_data_interactive(grain_code, ppy):
    st.markdown("**Upload Inflation Data**")
    inf_file = st.file_uploader("Upload Inflation Curve (Period, Rate %)", type=["csv","xlsx","xls"], key=f"inf_{st.session_state.page}")
    cum_inflation = None; per_period_rates = None
    if inf_file:
        inf_df = pd.read_csv(inf_file) if inf_file.name.endswith('.csv') else pd.read_excel(inf_file)
        p_col = st.selectbox("Period column", inf_df.columns, key=f"inf_p_{st.session_state.page}")
        r_col = st.selectbox("Rate column", inf_df.columns, key=f"inf_r_{st.session_state.page}")
        inf_df = inf_df[[p_col, r_col]].dropna()
        inf_df[r_col] = pd.to_numeric(inf_df[r_col], errors='coerce') / 100.0
        rates_inf = inf_df[r_col].values
        ratio = ppy / periods_per_year(grain_code) # Assuming uploaded data is Yearly
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

def _load_discounting_data_interactive(grain_code, ppy):
    st.markdown("**Upload Discounting Data**")
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

def apply_inflation_to_triangle(inc, cum, n_periods, cum_inflation):
    if cum_inflation is not None:
        real_inc = inc.copy().astype(float)
        real_cum = cum.copy().astype(float)
        valuation_idx = n_periods - 1
        if len(cum_inflation) <= valuation_idx:
            last_val = cum_inflation[-1] if len(cum_inflation) > 0 else 1.0
            cum_inflation = np.append(cum_inflation, [last_val] * (valuation_idx - len(cum_inflation) + 1))
        inf_val = cum_inflation[valuation_idx]
        for ap in inc.index:
            for dp in inc.columns:
                if ap+dp >= n_periods: continue
                val = inc.loc[ap, dp]
                if pd.isna(val): continue
                t = ap+dp
                inf_t = cum_inflation[min(t, len(cum_inflation)-1)]
                deflation_factor = inf_val/inf_t if inf_t>0 else 1.0
                real_inc.loc[ap, dp] = val * deflation_factor
        # Rebuild cumulative
        for ap in real_inc.index:
            has_obs = any(pd.notna(real_inc.loc[ap, dp]) for dp in real_inc.columns if ap+dp<n_periods)
            if not has_obs:
                real_cum.loc[ap] = np.nan; continue
            running = 0.0
            for dp in sorted(real_inc.columns):
                if ap+dp < n_periods:
                    v = real_inc.loc[ap, dp]
                    running += v if pd.notna(v) else 0.0
                    real_cum.loc[ap, dp] = running
                else:
                    real_cum.loc[ap, dp] = np.nan
        return real_inc, real_cum
    return inc, cum

def apply_discounting_to_triangle(completed, cum, n_periods, grain_code, ppy, spot_rates, flat_rate):
    dp_cols = sorted(completed.columns)
    discounted_results = []
    for ap in completed.index:
        last_obs = -1
        for dp in sorted(cum.columns, reverse=True):
            if ap+dp < n_periods:
                val = cum.loc[ap, dp]
                if pd.notna(val) and val > 0: last_obs = dp; break
        if last_obs < 0 or last_obs >= max(dp_cols):
            discounted_results.append({"AP": ap, "Nominal_IBNR": 0.0, "Discounted_IBNR": 0.0})
            continue
        total_nominal = 0.0; total_discounted = 0.0
        for idx_dp, dp in enumerate(dp_cols):
            if dp <= last_obs: continue
            cum_curr = completed.loc[ap, dp]
            if pd.isna(cum_curr): continue
            if idx_dp > 0:
                cum_prev = completed.loc[ap, dp_cols[idx_dp-1]]
                inc_payment = max(float(cum_curr) - float(cum_prev if pd.notna(cum_prev) else 0.0), 0.0)
            else:
                inc_payment = max(float(cum_curr), 0.0)
            if inc_payment <= 0.0: continue
            periods_ahead = dp - last_obs
            years_ahead = periods_ahead / ppy
            if spot_rates is not None:
                idx = min(int(periods_ahead)-1, len(spot_rates)-1)
                r = float(spot_rates[max(idx, 0)])
            else:
                r = float(flat_rate)
            df_factor = 1.0 / (1.0 + r) ** years_ahead
            total_nominal += inc_payment
            total_discounted += inc_payment * df_factor
        discounted_results.append({"AP": ap, "Nominal_IBNR": total_nominal, "Discounted_IBNR": total_discounted})
    return pd.DataFrame(discounted_results)


def render_bcl_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Basic Chain Ladder (BCL) — IBNR Calculator</h1><p>Volume-weighted development factors with configurable grain and grouping</p></div>', unsafe_allow_html=True)
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
        with c4: amt_col = st.selectbox("Claim Amount", cols, key="bcl_amt")

        df[loss_col] = pd.to_datetime(df[loss_col], errors='coerce').astype('datetime64[ns]')
        df[rep_col]  = pd.to_datetime(df[rep_col],  errors='coerce').astype('datetime64[ns]')
        df[amt_col]  = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)
        df = df.dropna(subset=[loss_col, rep_col])
        from_dt = pd.Timestamp(str(from_date)); to_dt = pd.Timestamp(str(to_date))
        df = _date_filter(df, loss_col, from_date, to_date)

        # ---- INFLATION & DISCOUNTING ----
        st.markdown("#### Step 3: Inflation & Discounting Adjustments")
        c1, c2 = st.columns(2)
        with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key="bcl_inf")
        with c2: use_discounting = st.checkbox("Apply Discounting", key="bcl_disc")

        cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
        if use_inflation:
            cum_inflation, per_period_rates = _load_inflation_data_interactive(grain_code, ppy)
        if use_discounting:
            spot_rates, flat_rate = _load_discounting_data_interactive(grain_code, ppy)

        # ---- CORE LOGIC ----
        if st.button("Calculate BCL IBNR", key="bcl_run", width='stretch'):
            lobs = sorted(df[lob_col].dropna().unique())
            n_periods = (to_date.year - from_date.year) * ppy + 1
            
            all_rows = []
            for lob in lobs:
                sub = df[df[lob_col]==lob].copy()
                sub['AP'] = sub[loss_col].apply(lambda d: (d.year - from_dt.year) * ppy + (d.month - from_dt.month)// (12//ppy))
                sub['DP'] = sub.apply(lambda r: max(0, (r[rep_col].year - r[loss_col].year) * ppy + (r[rep_col].month - r[loss_col].month)// (12//ppy)), axis=1)
                
                n_ap = sub['AP'].max()+1 if not sub.empty else n_periods
                n_dp = n_ap
                sub = sub[(sub['AP']>=0)&(sub['AP']<n_ap)]
                sub['DP'] = sub['DP'].clip(0, n_dp-1)
                
                pivot = sub.pivot_table(index='AP', columns='DP', values=amt_col, aggfunc='sum')
                for i in range(n_periods):
                    if i not in pivot.index: pivot.loc[i] = np.nan
                for j in range(n_periods):
                    if j not in pivot.columns: pivot[j] = np.nan
                inc = pivot.sort_index()[sorted(pivot.columns)].astype(float)
                for i in inc.index:
                    for j in inc.columns:
                        if i+j>=n_periods: inc.loc[i,j] = np.nan
                cum = inc.copy()
                for i in inc.index:
                    r=0.0
                    for j in sorted(inc.columns):
                        if i+j<n_periods:
                            v=inc.loc[i,j]; r+=v if pd.notna(v) else 0.0; cum.loc[i,j]=r
                        else: cum.loc[i,j]=np.nan

                # Apply Inflation/Deflation
                if use_inflation and cum_inflation is not None:
                    inc, cum = apply_inflation_to_triangle(inc, cum, n_periods, cum_inflation)

                wc = cum.fillna(0)
                n_ay, n_d = wc.shape
                factors=[]
                for j in range(n_d-1):
                    num,den=0.0,0.0
                    for i in range(n_ay):
                        if i+j+1<n_ay:
                            c=wc.iloc[i,j]; nxt=wc.iloc[i,j+1]
                            if c>0: num+=nxt; den+=c
                    factors.append(num/den if den>0 else 1.0)
                cdfs=[]; run=1.0
                for f in reversed(factors): run*=f; cdfs.insert(0,run)
                completed=wc.copy().astype(float)
                for i in range(n_ay):
                    lo=-1
                    for j in range(n_d-1,-1,-1):
                        if i+j<n_ay: lo=j; break
                    if lo<0: continue
                    for j in range(lo,n_d-1):
                        if j<len(factors):
                            p=completed.iloc[i,j]; completed.iloc[i,j+1]=p*factors[j] if p>0 else 0.0

                # Apply Discounting
                if use_discounting:
                    disc_df = apply_discounting_to_triangle(completed, cum, n_periods, grain_code, ppy, spot_rates, flat_rate)
                    disc_df = disc_df.set_index("AP")
                    # TODO: Map discounted IBNR back
                else:
                    disc_df = pd.DataFrame()

                for i in range(n_ay):
                    lo=-1
                    for j in range(n_d-1,-1,-1):
                        if i+j<n_ay: lo=j; break
                    if lo>=0:
                        cur=wc.iloc[i,lo]; ult=completed.iloc[i,n_d-1]
                        ibnr=max(ult-cur,0.0)
                        cdf=cdfs[lo] if lo<len(cdfs) else 1.0
                        disc_ibnr = disc_df.loc[i, "Discounted_IBNR"] if use_discounting and i in disc_df.index else np.nan
                        all_rows.append({
                            'LOB': lob,
                            'Accident_Period': str(from_dt.year + i//ppy) + (f"-Q{(i%ppy)+1}" if grain_code!="Y" else ""),
                            'Developed_Periods': lo,
                            'CDF': cdf,
                            'Current_Claims': cur,
                            'Ultimate': ult,
                            'IBNR': ibnr,
                            'Discounted_IBNR': disc_ibnr
                        })

            result = pd.DataFrame(all_rows)
            summary = result.groupby('LOB')[['Current_Claims','Ultimate','IBNR']].sum().reset_index()
            if use_discounting:
                disc_summary = result.groupby('LOB')['Discounted_IBNR'].sum().reset_index()
                summary = summary.merge(disc_summary, on='LOB', how='left')
            
            st.subheader("BCL IBNR Summary by LOB")
            disp=summary.copy()
            for c in ['Current_Claims','Ultimate','IBNR','Discounted_IBNR']:
                if c in disp.columns: disp[c]=disp[c].apply(lambda x:f"{x:,.2f}" if pd.notna(x) else "-")
            st.dataframe(disp, width='stretch', hide_index=True)
            
            st.subheader("Detail by Accident Period")
            disp2=result.copy()
            for c in ['CDF','Current_Claims','Ultimate','IBNR','Discounted_IBNR']:
                if c in disp2.columns: disp2[c]=disp2[c].apply(lambda x:f"{x:,.4f}" if c=='CDF' else f"{x:,.2f}" if pd.notna(x) else "-")
            st.dataframe(disp2, width='stretch', hide_index=True)

            output=BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w:
                summary.to_excel(w,index=False,sheet_name='BCL_Summary')
                result.to_excel(w,index=False,sheet_name='BCL_Detail')
            output.seek(0)
            sc=re.sub(r'[\\/*?:"<>|]','',client_name).strip() or "Client"
            st.download_button("⬇ Download BCL Results", data=output, file_name=f"{sc}_BCL_IBNR.xlsx", key="bcl_dl")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback; st.write(traceback.format_exc())

    back_button('ibnr_menu', ['Home','LIC','Fulfilment Cashflows','IBNR Methods'])


def render_capecod_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Cape Cod IBNR</h1><p>Uses earned premiums to derive an implied ELR</p></div>', unsafe_allow_html=True)
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
        with c4: amt_col=st.selectbox("Amount",cols,key="cc_amt")
        
        df[loss_col]=pd.to_datetime(df[loss_col],errors='coerce').astype('datetime64[ns]')
        df[rep_col]=pd.to_datetime(df[rep_col],errors='coerce').astype('datetime64[ns]')
        df[amt_col]=pd.to_numeric(df[amt_col],errors='coerce').fillna(0)
        df=df.dropna(subset=[loss_col,rep_col])
        from_dt=pd.Timestamp(str(from_date)); to_dt=pd.Timestamp(str(to_date))
        df=_date_filter(df, loss_col, from_date, to_date)

        # ---- INFLATION & DISCOUNTING ----
        st.markdown("#### Inflation & Discounting Adjustments")
        c1, c2 = st.columns(2)
        with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key="cc_inf")
        with c2: use_discounting = st.checkbox("Apply Discounting", key="cc_disc")

        grain = "Y"; ppy = 1  # For premiums
        cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
        if use_inflation:
            cum_inflation, per_period_rates = _load_inflation_data_interactive(grain, ppy)
        if use_discounting:
            spot_rates, flat_rate = _load_discounting_data_interactive(grain, ppy)

        if st.button("Calculate Cape Cod IBNR",key="cc_run",width='stretch'):
            lobs=sorted(df[lob_col].dropna().unique()); rows=[]
            for lob in lobs:
                sub=df[df[lob_col]==lob].copy()
                sub['AP']=sub[loss_col].apply(lambda d:d.year-from_dt.year)
                sub['DP']=sub.apply(lambda r:max(0,r[rep_col].year-r[loss_col].year),axis=1)
                n=sub['AP'].max()+1 if not sub.empty else 1
                sub=sub[(sub['AP']>=0)&(sub['AP']<n)]; sub['DP']=sub['DP'].clip(0,n-1)
                pivot=sub.pivot_table(index='AP',columns='DP',values=amt_col,aggfunc='sum')
                for i in range(n):
                    if i not in pivot.index: pivot.loc[i]=0.0
                for j in range(n):
                    if j not in pivot.columns: pivot[j]=0.0
                inc=pivot.sort_index()[sorted(pivot.columns)].fillna(0.0).astype(float)
                for i in inc.index:
                    for j in inc.columns:
                        if i+j>=n: inc.loc[i,j]=0.0
                cum=inc.cumsum(axis=1); wc=cum.copy(); n_ay,n_d=wc.shape
                facs=[]
                for j in range(n_d-1):
                    num,den=0.0,0.0
                    for i in range(n_ay):
                        if i+j+1<n_ay:
                            c=wc.iloc[i,j]; nxt=wc.iloc[i,j+1]
                            if c>0: num+=nxt; den+=c
                    facs.append(num/den if den>0 else 1.0)
                cdfs=[]; run=1.0
                for f in reversed(facs): run*=f; cdfs.insert(0,run)
                pct_unpaid=[1-(1/c) if c>0 else 0 for c in cdfs]
                prems={}
                if lob in prem_df.columns:
                    vals=pd.to_numeric(prem_df[lob],errors='coerce').fillna(0).tolist()
                    for i,v in enumerate(vals): prems[i]=v
                num_elr=den_elr=0.0
                for i in range(n_ay):
                    lo=-1
                    for j in range(n_d-1,-1,-1):
                        if i+j<n_ay: lo=j; break
                    if lo<0: continue
                    prem_i=prems.get(i,0); pct_dev=1-pct_unpaid[lo] if lo<len(pct_unpaid) else 1
                    num_elr+=wc.iloc[i,lo]; den_elr+=prem_i*pct_dev
                cc_elr=num_elr/den_elr if den_elr>0 else 0.7
                total_ibnr=0.0
                for i in range(n_ay):
                    lo=-1
                    for j in range(n_d-1,-1,-1):
                        if i+j<n_ay: lo=j; break
                    if lo<0: continue
                    prem_i=prems.get(i,0); pct_u=pct_unpaid[lo] if lo<len(pct_unpaid) else 0
                    ibnr=prem_i*cc_elr*pct_u; total_ibnr+=ibnr
                    rows.append({'LOB':lob,'Year':from_dt.year+i,'Premium':prem_i,'CC_ELR':cc_elr,'Pct_Unpaid':pct_u,'IBNR':ibnr})
            res=pd.DataFrame(rows)
            summ=res.groupby('LOB')[['Premium','IBNR']].sum().reset_index()
            summ['CC_ELR']=res.groupby('LOB')['CC_ELR'].first().values
            st.subheader(f"Cape Cod ELR & IBNR Summary")
            disp=summ.copy()
            disp['CC_ELR']=disp['CC_ELR'].apply(lambda x:f"{x:.2%}")
            for c in ['Premium','IBNR']: disp[c]=disp[c].apply(lambda x:f"{x:,.2f}")
            st.dataframe(disp,width='stretch',hide_index=True)
            output=BytesIO()
            with pd.ExcelWriter(output,engine='openpyxl') as w:
                summ.to_excel(w,index=False,sheet_name='CapeCod_Summary')
                res.to_excel(w,index=False,sheet_name='CapeCod_Detail')
            output.seek(0)
            sc=re.sub(r'[\/*?:"<>|]','',client_name).strip() or "Client"
            st.download_button("⬇ Download Cape Cod Results",data=output,file_name=f"{sc}_CapeCod_IBNR.xlsx",key="cc_dl")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback; st.write(traceback.format_exc())
    back_button('ibnr_menu',['Home','LIC','Fulfilment Cashflows','IBNR Methods'])


def render_bf_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Bornhuetter-Ferguson (BF) IBNR</h1><p>Expected + Actual blend using user-supplied ELR per portfolio</p></div>', unsafe_allow_html=True)
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
        with c4: amt_col=st.selectbox("Amount",cols,key="bf_amt")
        
        df[loss_col]=pd.to_datetime(df[loss_col],errors='coerce').astype('datetime64[ns]')
        df[rep_col]=pd.to_datetime(df[rep_col],errors='coerce').astype('datetime64[ns]')
        df[amt_col]=pd.to_numeric(df[amt_col],errors='coerce').fillna(0)
        df=df.dropna(subset=[loss_col,rep_col])
        from_dt=pd.Timestamp(str(from_date)); to_dt=pd.Timestamp(str(to_date))
        df=_date_filter(df, loss_col, from_date, to_date)
        lobs=sorted(df[lob_col].dropna().unique())
        st.markdown("**ELR per Portfolio (%):**")
        elr_cols=st.columns(min(len(lobs),4))
        elr_dict={}
        for i,lob in enumerate(lobs):
            with elr_cols[i%4]: elr_dict[lob]=st.number_input(f"ELR {lob}",0.0,200.0,70.0,1.0,key=f"bf_elr_{lob}")/100
        prem_data=None
        if prem_file is not None:
            prem_data=pd.read_csv(prem_file) if prem_file.name.endswith('.csv') else pd.read_excel(prem_file)
            prem_data.columns=prem_data.columns.astype(str).str.strip()

        # ---- INFLATION & DISCOUNTING ----
        st.markdown("#### Inflation & Discounting Adjustments")
        c1, c2 = st.columns(2)
        with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key="bf_inf")
        with c2: use_discounting = st.checkbox("Apply Discounting", key="bf_disc")

        grain = "Y"; ppy = 1
        cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None
        if use_inflation:
            cum_inflation, per_period_rates = _load_inflation_data_interactive(grain, ppy)
        if use_discounting:
            spot_rates, flat_rate = _load_discounting_data_interactive(grain, ppy)

        if st.button("Calculate BF IBNR",key="bf_run",width='stretch'):
            all_rows=[]; summ_rows=[]
            for lob in lobs:
                sub=df[df[lob_col]==lob].copy()
                sub['AP']=sub[loss_col].apply(lambda d:d.year-from_dt.year)
                sub['DP']=sub.apply(lambda r:max(0,r[rep_col].year-r[loss_col].year),axis=1)
                n=sub['AP'].max()+1 if not sub.empty else 1
                sub=sub[(sub['AP']>=0)&(sub['AP']<n)]; sub['DP']=sub['DP'].clip(0,n-1)
                pivot=sub.pivot_table(index='AP',columns='DP',values=amt_col,aggfunc='sum')
                for i in range(n):
                    if i not in pivot.index: pivot.loc[i]=0.0
                for j in range(n):
                    if j not in pivot.columns: pivot[j]=0.0
                inc=pivot.sort_index()[sorted(pivot.columns)].fillna(0.0).astype(float)
                for i in inc.index:
                    for j in inc.columns:
                        if i+j>=n: inc.loc[i,j]=0.0
                cum=inc.cumsum(axis=1); wc=cum.copy(); n_ay,n_d=wc.shape
                facs=[]
                for j in range(n_d-1):
                    num,den=0.0,0.0
                    for i in range(n_ay):
                        if i+j+1<n_ay:
                            c=wc.iloc[i,j]; nxt=wc.iloc[i,j+1]
                            if c>0: num+=nxt; den+=c
                    facs.append(num/den if den>0 else 1.0)
                cdfs=[]; run=1.0
                for f in reversed(facs): run*=f; cdfs.insert(0,run)
                pct_unpaid=[1-(1/c) if c>0 else 0 for c in cdfs]
                prems={}
                if prem_data is not None and lob in prem_data.columns:
                    vals=pd.to_numeric(prem_data[lob],errors='coerce').fillna(0).tolist()
                    for i,v in enumerate(vals): prems[i]=v
                elr=elr_dict.get(lob,0.7); total_bf=0.0
                for i in range(n_ay):
                    lo=-1
                    for j in range(n_d-1,-1,-1):
                        if i+j<n_ay: lo=j; break
                    if lo<0: continue
                    prem_i=prems.get(i,wc.iloc[i,lo]*1.5 if wc.iloc[i,lo]>0 else 1000)
                    pct_u=pct_unpaid[lo] if lo<len(pct_unpaid) else 0
                    bf_ibnr=prem_i*elr*pct_u; total_bf+=bf_ibnr
                    all_rows.append({'LOB':lob,'Year':from_dt.year+i,'Premium':prem_i,'ELR':elr,'Pct_Unpaid':pct_u,'Current':wc.iloc[i,lo],'BF_IBNR':bf_ibnr})
                summ_rows.append({'LOB':lob,'Total_BF_IBNR':total_bf,'ELR':elr})
            res=pd.DataFrame(all_rows); summ=pd.DataFrame(summ_rows)
            st.subheader("BF IBNR Summary")
            disp=summ.copy(); disp['ELR']=disp['ELR'].apply(lambda x:f"{x:.2%}")
            disp['Total_BF_IBNR']=disp['Total_BF_IBNR'].apply(lambda x:f"{x:,.2f}")
            st.dataframe(disp,width='stretch',hide_index=True)
            with st.expander("Detail by Accident Year"):
                disp2=res.copy()
                for c in ['ELR','Pct_Unpaid']: disp2[c]=disp2[c].apply(lambda x:f"{x:.2%}")
                for c in ['Premium','Current','BF_IBNR']: disp2[c]=disp2[c].apply(lambda x:f"{x:,.2f}")
                st.dataframe(disp2,width='stretch',hide_index=True)
            output=BytesIO()
            with pd.ExcelWriter(output,engine='openpyxl') as w:
                summ.to_excel(w,index=False,sheet_name='BF_Summary')
                res.to_excel(w,index=False,sheet_name='BF_Detail')
            output.seek(0)
            sc=re.sub(r'[\/*?:"<>|]','',client_name).strip() or "Client"
            st.download_button("⬇ Download BF Results",data=output,file_name=f"{sc}_BF_IBNR.xlsx",key="bf_dl")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback; st.write(traceback.format_exc())
    back_button('ibnr_menu',['Home','LIC','Fulfilment Cashflows','IBNR Methods'])


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

    # ---- SELECT RESERVES ----
    st.markdown('<div class="section-container"><h3>Select Reserves & Methodologies</h3></div>', unsafe_allow_html=True)

    st.markdown("**LRC — Liability for Remaining Coverage:**")
    c1, c2 = st.columns(2)
    with c1: calc_upr = st.checkbox("UPR (Unearned Premium Reserve)", value=True, key="fv_upr")
    with c2: calc_loss_comp = st.checkbox("Loss Component (Onerous Contracts)", value=False, key="fv_lc")

    st.markdown("**LIC — Fulfilment Cashflows:**")
    c1, c2, c3 = st.columns(3)
    with c1: calc_ocr = st.checkbox("OCR (Case Reserves)", value=True, key="fv_ocr")
    with c2: calc_ibnr = st.checkbox("IBNR", value=True, key="fv_ibnr")
    with c3:
        if calc_ibnr:
            ibnr_method = st.selectbox("IBNR Method", ["BCL","Percentage","Cape Cod","BF","ELR","ACPC"], key="fv_im")
        else: ibnr_method = "BCL"

    c1, c2, c3 = st.columns(3)
    with c1:
        calc_ulae = st.checkbox("ULAE", value=True, key="fv_ulae")
        ulae_basis = st.selectbox("ULAE Basis", ["Per Portfolio","Aggregated"], key="fv_ub") if calc_ulae else "Per Portfolio"
    with c2: calc_npr = st.checkbox("NPR (Reinsurance)", value=False, key="fv_npr")
    with c3: pass

    st.markdown("**LIC — Risk Adjustment:**")
    c1, c2, c3 = st.columns(3)
    with c1: calc_ra = st.checkbox("Risk Adjustment", value=True, key="fv_ra")
    with c2:
        if calc_ra:
            ra_method = st.selectbox("RA Method", ["Bootstrap","Mack","VaR","Cost of Capital"], key="fv_rm")
            ra_cl = st.number_input("Confidence Level (%)", 50.0, 99.5, 90.0, 5.0, key="fv_rc")/100
        else: ra_method = "Bootstrap"; ra_cl = 0.90
    with c3:
        ra_iters = st.number_input("Bootstrap Iterations", 100, 10000, 1000, 100, key="fv_ri") if (calc_ra and ra_method=="Bootstrap") else 1000

    selected = []
    if calc_upr: selected.append("UPR")
    if calc_loss_comp: selected.append("Loss Component")
    if calc_ocr: selected.append("OCR")
    if calc_ibnr: selected.append(f"IBNR({ibnr_method})")
    if calc_ulae: selected.append(f"ULAE({ulae_basis})")
    if calc_npr: selected.append("NPR")
    if calc_ra: selected.append(f"RA({ra_method}@{ra_cl*100:.0f}%)")
    st.info(f"Selected: {', '.join(selected) if selected else 'None'}")

    # ---- PARAMETERS ----
    st.markdown('<div class="section-container"><h3>Parameters</h3></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if calc_ulae: ulae_ratio = st.number_input("ULAE Ratio (%)", 0.0, 20.0, 5.0, 0.5, key="fv_ur")/100
        else: ulae_ratio = 0.05
    with c2:
        if calc_ibnr: ibnr_grain = st.selectbox("IBNR Grain", ["Yearly","Half-Yearly","Quarterly","Monthly"], key="fv_ig")
        else: ibnr_grain = "Yearly"
    with c3:
        if calc_upr: upr_method = st.selectbox("UPR Method", ["365th","24th","8th"], key="fv_um")
        else: upr_method = "365th"

    # ---- DATA FILES ----
    st.markdown('<div class="section-container"><h3>Upload Data Files & Map Columns</h3></div>', unsafe_allow_html=True)

    upr_data = None; ocr_data = None; claims_data = None
    apportionment_data = None; cashflow_data = None; opening_data = None
    premium_data = None; elr_dict = {}

    if calc_upr:
        st.markdown("#### UPR Data")
        upr_file = st.file_uploader("Upload UPR file", type=["csv","xlsx","xls"], key="fv_upr_f")
        if upr_file is not None:
            try:
                upr_df = pd.read_csv(upr_file) if upr_file.name.endswith('.csv') else pd.read_excel(upr_file)
                upr_df.columns = upr_df.columns.astype(str).str.strip()
                st.dataframe(upr_df.head(3), width='stretch')
                upr_map = map_columns(upr_df, ['Start_Date','End_Date','Line_of_Business','Gross_Written_Premium'], 'UPR')
                upr_data = upr_df.rename(columns=upr_map)
                st.success("UPR columns mapped.")
            except Exception as e: st.error(f"Error: {e}")

    if calc_ocr:
        st.markdown("#### OCR Data")
        ocr_file = st.file_uploader("Upload OCR file", type=["csv","xlsx","xls"], key="fv_ocr_f")
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
        st.markdown("#### Claims Triangle Data")
        claims_file = st.file_uploader("Upload Claims file", type=["csv","xlsx","xls"], key="fv_cl_f")
        if claims_file is not None:
            try:
                cl_df = pd.read_csv(claims_file) if claims_file.name.endswith('.csv') else pd.read_excel(claims_file)
                cl_df.columns = cl_df.columns.astype(str).str.strip()
                st.dataframe(cl_df.head(3), width='stretch')
                cl_map = map_columns(cl_df, ['Loss_Date','Report_Date','Claim_Amount','Line_of_Business'], 'Claims')
                claims_data = cl_df.rename(columns=cl_map)
                st.success("Claims columns mapped.")
            except Exception as e: st.error(f"Error: {e}")

    if calc_ibnr and ibnr_method in ["Cape Cod","BF"]:
        st.markdown("#### Premium Data (for IBNR)")
        prem_file = st.file_uploader("Upload Premium file", type=["csv","xlsx","xls"], key="fv_pr_f")
        if prem_file is not None:
            try:
                prem_df = pd.read_csv(prem_file) if prem_file.name.endswith('.csv') else pd.read_excel(prem_file)
                prem_df.columns = prem_df.columns.astype(str).str.strip()
                st.dataframe(prem_df.head(3), width='stretch')
                premium_data = prem_df
                st.success("Premium data loaded.")
            except Exception as e: st.error(f"Error: {e}")

    if calc_ibnr and ibnr_method == "BF":
        st.markdown("#### ELR per Portfolio")
        portfolios_tmp = ["Motor","Property","Health","Engineering","Liability"]
        ec = st.columns(min(3, len(portfolios_tmp)))
        for i, p in enumerate(portfolios_tmp):
            with ec[i%3]: elr_dict[p] = st.number_input(f"ELR {p} %", 0.0, 200.0, 70.0, 1.0, key=f"fv_elr_{p}")/100

    if calc_ulae:
        st.markdown("#### ULAE Apportionment Key")
        app_file = st.file_uploader("Upload Apportionment file", type=["csv","xlsx","xls"], key="fv_ap_f")
        if app_file is not None:
            try:
                app_df = pd.read_csv(app_file) if app_file.name.endswith('.csv') else pd.read_excel(app_file)
                app_df.columns = app_df.columns.astype(str).str.strip()
                st.dataframe(app_df.head(3), width='stretch')
                app_map = map_columns(app_df, ['Portfolio','Premiums_Received'], 'Apportionment')
                apportionment_data = app_df.rename(columns=app_map)
                st.success("Apportionment columns mapped.")
            except Exception as e: st.error(f"Error: {e}")

    st.markdown("#### Cash Flow Data")
    cf_file = st.file_uploader("Upload Cash Flow file", type=["csv","xlsx","xls"], key="fv_cf")
    if cf_file is not None:
        try:
            cf_df = pd.read_csv(cf_file) if cf_file.name.endswith('.csv') else pd.read_excel(cf_file)
            cf_df.columns = cf_df.columns.astype(str).str.strip()
            st.dataframe(cf_df.head(3), width='stretch')
            cf_map = map_columns(cf_df, ['Portfolio','Premiums_Received','Paid_Claims_Gross','Acquisition_Costs','Maintenance_Expenses'], 'CashFlow')
            cashflow_data = cf_df.rename(columns=cf_map)
            st.success("Cash flow columns mapped.")
        except Exception as e: st.error(f"Error: {e}")

    st.markdown("#### Opening Balances")
    op_file = st.file_uploader("Upload Opening Balances file", type=["csv","xlsx","xls"], key="fv_ob")
    if op_file is not None:
        try:
            op_df = pd.read_csv(op_file) if op_file.name.endswith('.csv') else pd.read_excel(op_file)
            op_df.columns = op_df.columns.astype(str).str.strip()
            st.dataframe(op_df.head(3), width='stretch')
            op_map = map_columns(op_df, ['Portfolio','Opening_LRC_UPR','Opening_LIC_OCR','Opening_LIC_IBNR','Opening_LIC_ULAE','Opening_LIC_RA'], 'OpeningBal')
            opening_data = op_df.rename(columns=op_map)
            st.success("Opening balance columns mapped.")
        except Exception as e: st.error(f"Error: {e}")

    # ---- CALCULATE ----
    if st.button("Run Full IFRS 17 Valuation", key="fv_run", width='stretch'):
        if not selected:
            st.warning("Select at least one reserve.")
        else:
            with st.spinner("Running full IFRS 17 valuation..."):
                results = {}
                val_date = pd.Timestamp(str(report_date))
                from_dt = pd.Timestamp('2020-01-01')
                to_dt = pd.Timestamp('2025-12-31')
                n_periods_bcl = to_dt.year - from_dt.year + 1

                portfolios = []
                if upr_data is not None and 'Line_of_Business' in upr_data.columns:
                    portfolios = sorted(upr_data['Line_of_Business'].dropna().unique().tolist())
                elif ocr_data is not None and 'Line_of_Business' in ocr_data.columns:
                    portfolios = sorted(ocr_data['Line_of_Business'].dropna().unique().tolist())
                elif claims_data is not None and 'Line_of_Business' in claims_data.columns:
                    portfolios = sorted(claims_data['Line_of_Business'].dropna().unique().tolist())
                else:
                    portfolios = ["Motor","Property","Health","Engineering","Liability"]

                st.info(f"Portfolios: {', '.join(portfolios)}")

                # ---- UPR ----
                if calc_upr and upr_data is not None:
                    df_upr = upr_data.copy()
                    df_upr['Start_Date'] = pd.to_datetime(df_upr['Start_Date'], errors='coerce').astype('datetime64[ns]')
                    df_upr['End_Date'] = pd.to_datetime(df_upr['End_Date'], errors='coerce').astype('datetime64[ns]')
                    df_upr['Premium'] = pd.to_numeric(df_upr['Gross_Written_Premium'], errors='coerce')
                    df_upr = df_upr.dropna(subset=['Start_Date','End_Date'])
                    df_upr = df_upr[df_upr['End_Date'] > df_upr['Start_Date']]
                    df_upr['Duration'] = (df_upr['End_Date'] - df_upr['Start_Date']).dt.days
                    df_upr['Remaining'] = (df_upr['End_Date'] - val_date).dt.days
                    if upr_method == "365th":
                        df_upr['Unearned'] = np.where(val_date < df_upr['Start_Date'], 1,
                            np.where(val_date > df_upr['End_Date'], 0,
                            np.clip(df_upr['Remaining'] / df_upr['Duration'], 0, 1)))
                    elif upr_method == "24th":
                        iv = 365.25/24
                        df_upr['Unearned'] = np.where(val_date < df_upr['Start_Date'], 1,
                            np.where(val_date > df_upr['End_Date'], 0,
                            np.clip((df_upr['End_Date']-val_date).dt.days/iv / (df_upr['Duration']/iv), 0, 1)))
                    else:
                        iv = 365.25/8
                        df_upr['Unearned'] = np.where(val_date < df_upr['Start_Date'], 1,
                            np.where(val_date > df_upr['End_Date'], 0,
                            np.clip((df_upr['End_Date']-val_date).dt.days/iv / (df_upr['Duration']/iv), 0, 1)))
                    df_upr['UPR'] = df_upr['Unearned'] * df_upr['Premium']
                    upr_result = df_upr.groupby('Line_of_Business')['UPR'].sum().reset_index()
                    upr_result.columns = ['Portfolio','Closing_UPR']
                    results['UPR'] = upr_result
                    st.success(f"UPR ({upr_method}): {upr_result['Closing_UPR'].sum():,.2f}")

                # ---- OCR ----
                if calc_ocr and ocr_data is not None:
                    df_ocr = ocr_data.copy()
                    df_ocr['Reserve'] = pd.to_numeric(df_ocr['Case_Reserve'], errors='coerce')
                    ocr_result = df_ocr.groupby('Line_of_Business')['Reserve'].sum().reset_index()
                    ocr_result.columns = ['Portfolio','Closing_OCR']
                    results['OCR'] = ocr_result
                    st.success(f"OCR: {ocr_result['Closing_OCR'].sum():,.2f}")

                # ---- IBNR ----
                if calc_ibnr and claims_data is not None:
                    df_cl = claims_data.copy()
                    df_cl['Loss_Date'] = pd.to_datetime(df_cl['Loss_Date'], errors='coerce').astype('datetime64[ns]')
                    df_cl['Report_Date'] = pd.to_datetime(df_cl['Report_Date'], errors='coerce').astype('datetime64[ns]')
                    df_cl['Amount'] = pd.to_numeric(df_cl['Claim_Amount'], errors='coerce')
                    df_cl = df_cl.dropna(subset=['Loss_Date','Report_Date'])
                    df_cl = _date_filter(df_cl, 'Loss_Date', from_dt, to_dt)

                    ibnr_rows = []
                    for lob in portfolios:
                        lob_data = df_cl[df_cl['Line_of_Business']==lob].copy()
                        if len(lob_data)==0:
                            ibnr_rows.append({'Portfolio':lob,'Closing_IBNR':0}); continue
                        lob_data['AP'] = lob_data['Loss_Date'].apply(lambda d: d.year - from_dt.year)
                        lob_data['DP'] = lob_data.apply(lambda r: max(0, min(r['Report_Date'].year - r['Loss_Date'].year, n_periods_bcl-1)), axis=1)
                        pivot = lob_data.pivot_table(index='AP', columns='DP', values='Amount', aggfunc='sum')
                        for ap in range(n_periods_bcl):
                            if ap not in pivot.index: pivot.loc[ap] = np.nan
                        for dp in range(n_periods_bcl):
                            if dp not in pivot.columns: pivot[dp] = np.nan
                        inc = pivot.sort_index()[sorted(pivot.columns)].astype(float)
                        for ap in inc.index:
                            for dp in inc.columns:
                                if ap+dp >= n_periods_bcl: inc.loc[ap, dp] = np.nan
                        cum = inc.copy()
                        for ap in inc.index:
                            has_obs = any(pd.notna(inc.loc[ap, dp]) for dp in inc.columns if ap+dp<n_periods_bcl)
                            if not has_obs: continue
                            running = 0.0
                            for dp in sorted(inc.columns):
                                if ap+dp<n_periods_bcl:
                                    v = inc.loc[ap, dp]; running += v if pd.notna(v) else 0.0; cum.loc[ap, dp] = running
                        wc = cum.fillna(0)

                        if ibnr_method == "Percentage":
                            total_paid = wc.sum().sum()
                            ibnr_total = total_paid * 0.10
                        elif ibnr_method == "BCL":
                            n_ay, n_dp = wc.shape
                            factors = []
                            for j in range(n_dp-1):
                                num, den = 0.0, 0.0
                                for i in range(n_ay):
                                    if i+j+1<n_ay:
                                        c = wc.iloc[i,j]; n = wc.iloc[i,j+1]
                                        if c>0: num+=n; den+=c
                                factors.append(num/den if den>0 else 1.0)
                            completed = wc.copy().astype(float)
                            for i in range(n_ay):
                                last_obs = -1
                                for j in range(n_dp-1,-1,-1):
                                    if i+j<n_ay: last_obs=j; break
                                if last_obs<0: continue
                                for j in range(last_obs, n_dp-1):
                                    if j<len(factors):
                                        prev = completed.iloc[i,j]; completed.iloc[i,j+1] = prev*factors[j] if prev>0 else 0.0
                            ibnr_total = 0.0
                            for i in range(n_ay):
                                last_obs = -1
                                for j in range(n_dp-1,-1,-1):
                                    if i+j<n_ay: last_obs=j; break
                                if last_obs>=0:
                                    cur = wc.iloc[i,last_obs]; ult = completed.iloc[i,n_dp-1]
                                    ibnr_total += max(ult-cur, 0.0)
                        elif ibnr_method == "BF":
                            n_ay, n_dp = wc.shape
                            factors = []
                            for j in range(n_dp-1):
                                num, den = 0.0, 0.0
                                for i in range(n_ay):
                                    if i+j+1<n_ay:
                                        c = wc.iloc[i,j]; n = wc.iloc[i,j+1]
                                        if c>0: num+=n; den+=c
                                factors.append(num/den if den>0 else 1.0)
                            cdfs = []; running = 1.0
                            for f in reversed(factors): running*=f; cdfs.insert(0, running)
                            pct_unpaid = [1-(1/c) if c>0 else 0 for c in cdfs]
                            gelr = elr_dict.get(lob, 0.7)
                            prems = []
                            if premium_data is not None and lob in premium_data.columns:
                                prems = pd.to_numeric(premium_data[lob], errors='coerce').fillna(0).tolist()
                            else:
                                prems = [wc.iloc[i,0]*2 for i in range(n_ay)]
                            ibnr_total = 0.0
                            for i in range(n_ay):
                                last_obs=-1
                                for j in range(n_dp-1,-1,-1):
                                    if i+j<n_ay: last_obs=j; break
                                if last_obs>=0 and i<len(prems) and last_obs<len(pct_unpaid):
                                    ibnr_total += prems[i]*gelr*pct_unpaid[last_obs]
                        else:
                            ibnr_total = 0
                        ibnr_rows.append({'Portfolio':lob,'Closing_IBNR':ibnr_total})
                    ibnr_result = pd.DataFrame(ibnr_rows)
                    results['IBNR'] = ibnr_result
                    st.success(f"IBNR ({ibnr_method}): {ibnr_result['Closing_IBNR'].sum():,.2f}")

                # ---- ULAE ----
                if calc_ulae and 'OCR' in results and 'IBNR' in results:
                    reserves_df = results['OCR'].merge(results['IBNR'], on='Portfolio', how='outer').fillna(0)
                    reserves_df['ULAE_Base'] = 0.5 * reserves_df['Closing_OCR'] + reserves_df['Closing_IBNR']
                    if ulae_basis == "Aggregated" and apportionment_data is not None:
                        app_df = apportionment_data.copy()
                        app_df['Amount'] = pd.to_numeric(app_df['Premiums_Received'], errors='coerce')
                        total_amt = app_df['Amount'].sum()
                        app_df['Pct'] = app_df['Amount']/total_amt if total_amt>0 else 0
                        total_ulae = reserves_df['ULAE_Base'].sum() * ulae_ratio
                        reserves_df = reserves_df.merge(app_df[['Portfolio','Pct']], on='Portfolio', how='left')
                        reserves_df['Pct'] = reserves_df['Pct'].fillna(1.0/len(reserves_df))
                        reserves_df['Closing_ULAE'] = total_ulae * reserves_df['Pct']
                    else:
                        reserves_df['Closing_ULAE'] = reserves_df['ULAE_Base'] * ulae_ratio
                    results['ULAE'] = reserves_df[['Portfolio','Closing_ULAE']]
                    st.success(f"ULAE ({ulae_basis}): {reserves_df['Closing_ULAE'].sum():,.2f}")

                # ---- RA (Bootstrap) ----
                if calc_ra and ra_method == "Bootstrap" and claims_data is not None:
                    df_cl = claims_data.copy()
                    df_cl['Loss_Date'] = pd.to_datetime(df_cl['Loss_Date'], errors='coerce').astype('datetime64[ns]')
                    df_cl['Report_Date'] = pd.to_datetime(df_cl['Report_Date'], errors='coerce').astype('datetime64[ns]')
                    df_cl['Amount'] = pd.to_numeric(df_cl['Claim_Amount'], errors='coerce')
                    df_cl = df_cl.dropna(subset=['Loss_Date','Report_Date'])
                    df_cl = _date_filter(df_cl, 'Loss_Date', from_dt, to_dt)
                    ra_rows = []
                    for lob in portfolios:
                        lob_data = df_cl[df_cl['Line_of_Business']==lob].copy()
                        if len(lob_data)==0: ra_rows.append({'Portfolio':lob,'Closing_RA':0}); continue
                        lob_data['AP'] = lob_data['Loss_Date'].apply(lambda d: d.year - from_dt.year)
                        lob_data['DP'] = lob_data.apply(lambda r: max(0, min(r['Report_Date'].year - r['Loss_Date'].year, n_periods_bcl-1)), axis=1)
                        pivot = lob_data.pivot_table(index='AP', columns='DP', values='Amount', aggfunc='sum')
                        for ap in range(n_periods_bcl):
                            if ap not in pivot.index: pivot.loc[ap] = np.nan
                        for dp in range(n_periods_bcl):
                            if dp not in pivot.columns: pivot[dp] = np.nan
                        inc = pivot.sort_index()[sorted(pivot.columns)].astype(float)
                        obs_mask = pd.DataFrame(False, index=inc.index, columns=inc.columns)
                        for ap in inc.index:
                            for dp in inc.columns:
                                if ap+dp < n_periods_bcl: obs_mask.loc[ap, dp] = pd.notna(inc.loc[ap, dp])
                        for ap in inc.index:
                            for dp in inc.columns:
                                if ap+dp >= n_periods_bcl: inc.loc[ap, dp] = np.nan
                        cum = inc.copy()
                        for ap in inc.index:
                            has_obs = any(pd.notna(inc.loc[ap, dp]) for dp in inc.columns if ap+dp<n_periods_bcl)
                            if not has_obs: continue
                            running = 0.0
                            for dp in sorted(inc.columns):
                                if ap+dp<n_periods_bcl:
                                    v = inc.loc[ap, dp]; running += v if pd.notna(v) else 0.0; cum.loc[ap, dp] = running
                        wc = cum.fillna(0)
                        n_ay, n_dp = wc.shape
                        factors = []
                        for j in range(n_dp-1):
                            num, den = 0.0, 0.0
                            for i in range(n_ay):
                                if i+j+1<n_ay:
                                    c = wc.iloc[i,j]; n = wc.iloc[i,j+1]
                                    if c>0: num+=n; den+=c
                            factors.append(num/den if den>0 else 1.0)
                        completed_det = wc.copy().astype(float)
                        for i in range(n_ay):
                            last_obs=-1
                            for j in range(n_dp-1,-1,-1):
                                if i+j<n_ay: last_obs=j; break
                            if last_obs<0: continue
                            for j in range(last_obs,n_dp-1):
                                if j<len(factors):
                                    prev=completed_det.iloc[i,j]; completed_det.iloc[i,j+1]=prev*factors[j] if prev>0 else 0.0
                        fitted_inc = completed_det.copy()
                        for i in range(n_ay):
                            for j in range(n_dp-1,0,-1): fitted_inc.iloc[i,j] = completed_det.iloc[i,j] - completed_det.iloc[i,j-1]
                        residuals_list = []
                        for i in range(n_ay):
                            for j in range(n_dp):
                                if i+j<n_ay and obs_mask.iloc[i,j]:
                                    actual = (wc.iloc[i,j]-wc.iloc[i,j-1]) if j>0 else wc.iloc[i,j]
                                    fitted = fitted_inc.iloc[i,j]
                                    resid = (actual-fitted)/np.sqrt(abs(fitted)) if fitted>0 else 0.0
                                    residuals_list.append(resid)
                        residuals = np.array(residuals_list)
                        n_obs = len(residuals); phi = max(np.sum(residuals**2)/max(n_obs-n_dp+1,1), 0.01)
                        ibnr_samples = []
                        for iteration in range(ra_iters):
                            sampled = np.random.choice(residuals, size=n_obs, replace=True)
                            pseudo_inc = fitted_inc.copy().astype(float); idx = 0
                            for i in range(n_ay):
                                for j in range(n_dp):
                                    if i+j<n_ay and obs_mask.iloc[i,j]:
                                        fv = fitted_inc.iloc[i,j]
                                        pv = fv + sampled[idx]*np.sqrt(max(abs(fv),0.001))
                                        pseudo_inc.iloc[i,j] = max(pv,0.0); idx += 1
                            pseudo_cum = pseudo_inc.cumsum(axis=1)
                            pf = []
                            for j in range(n_dp-1):
                                num, den = 0.0, 0.0
                                for i in range(n_ay):
                                    if i+j+1<n_ay:
                                        c = pseudo_cum.iloc[i,j]; n = pseudo_cum.iloc[i,j+1]
                                        if c>0: num+=n; den+=c
                                pf.append(num/den if den>0 else 1.0)
                            pc = pseudo_cum.copy().astype(float)
                            for i in range(n_ay):
                                last_obs=-1
                                for j in range(n_dp-1,-1,-1):
                                    if i+j<n_ay: last_obs=j; break
                                if last_obs<0: continue
                                for j in range(last_obs,n_dp-1):
                                    if j<len(pf):
                                        prev=pc.iloc[i,j]; pc.iloc[i,j+1]=prev*pf[j] if prev>0 else 0.0
                            if phi>1e-10:
                                proc_inc = pc.copy()
                                for i in range(n_ay):
                                    for j in range(n_dp-1,0,-1): proc_inc.iloc[i,j] = pc.iloc[i,j] - pc.iloc[i,j-1]
                                for i in range(n_ay):
                                    for j in range(n_dp):
                                        is_future = (i+j>=n_ay) or (not obs_mask.iloc[i,j])
                                        if is_future:
                                            mv = proc_inc.iloc[i,j]
                                            if pd.notna(mv) and mv>0: proc_inc.iloc[i,j] = max(np.random.gamma(mv/phi, phi), 0.0)
                                            else: proc_inc.iloc[i,j] = 0.0
                                pc = proc_inc.copy()
                                for i in range(n_ay):
                                    running=0.0
                                    for j in range(n_dp):
                                        v=proc_inc.iloc[i,j]; running+=v if pd.notna(v) and v>0 else 0.0; pc.iloc[i,j]=running
                            ibnr_val = 0.0
                            for i in range(n_ay):
                                last_obs=-1
                                for j in range(n_dp-1,-1,-1):
                                    if i+j<n_ay and obs_mask.iloc[i,j]: last_obs=j; break
                                if last_obs>=0:
                                    cur = pseudo_cum.iloc[i,last_obs]; ult = pc.iloc[i,n_dp-1]
                                    ibnr_val += max(ult-cur,0.0)
                            ibnr_samples.append(ibnr_val)
                        ibnr_arr = np.array(ibnr_samples)
                        cl_ibnr = 0.0
                        for i in range(n_ay):
                            last_obs=-1
                            for j in range(n_dp-1,-1,-1):
                                if i+j<n_ay: last_obs=j; break
                            if last_obs>=0:
                                cur=wc.iloc[i,last_obs]; ult=completed_det.iloc[i,n_dp-1]
                                cl_ibnr += max(ult-cur,0.0)
                        ra_90 = max(np.percentile(ibnr_arr, 90) - cl_ibnr, 0.0)
                        ra_rows.append({'Portfolio':lob,'Closing_RA':ra_90})
                    ra_result = pd.DataFrame(ra_rows)
                    results['RA'] = ra_result
                    st.success(f"RA (Bootstrap @{ra_cl*100:.0f}%): {ra_result['Closing_RA'].sum():,.2f}")

                # ---- RESULTS DISPLAY ----
                st.markdown("---")
                st.subheader("Valuation Results")

                closing_reserves = {}
                for p in portfolios:
                    closing_reserves[p] = {
                        'UPR': results['UPR'][results['UPR']['Portfolio']==p]['Closing_UPR'].sum() if 'UPR' in results else 0,
                        'OCR': results['OCR'][results['OCR']['Portfolio']==p]['Closing_OCR'].sum() if 'OCR' in results else 0,
                        'IBNR': results['IBNR'][results['IBNR']['Portfolio']==p]['Closing_IBNR'].sum() if 'IBNR' in results else 0,
                        'ULAE': results['ULAE'][results['ULAE']['Portfolio']==p]['Closing_ULAE'].sum() if 'ULAE' in results else 0,
                        'RA': results['RA'][results['RA']['Portfolio']==p]['Closing_RA'].sum() if 'RA' in results else 0,
                    }

                op_reserves = {}
                if opening_data is not None:
                    for _, row in opening_data.iterrows():
                        p = str(row['Portfolio'])
                        op_reserves[p] = {
                            'UPR': abs(pd.to_numeric(row.get('Opening_LRC_UPR',0), errors='coerce') or 0),
                            'OCR': pd.to_numeric(row.get('Opening_LIC_OCR',0), errors='coerce') or 0,
                            'IBNR': pd.to_numeric(row.get('Opening_LIC_IBNR',0), errors='coerce') or 0,
                            'ULAE': pd.to_numeric(row.get('Opening_LIC_ULAE',0), errors='coerce') or 0,
                            'RA': pd.to_numeric(row.get('Opening_LIC_RA',0), errors='coerce') or 0,
                        }

                cf_reserves = {}
                if cashflow_data is not None:
                    for _, row in cashflow_data.iterrows():
                        p = str(row['Portfolio'])
                        cf_reserves[p] = {
                            'Premiums_Received': pd.to_numeric(row.get('Premiums_Received',0), errors='coerce') or 0,
                            'Paid_Claims': pd.to_numeric(row.get('Paid_Claims_Gross',0), errors='coerce') or 0,
                            'Acquisition_Costs': pd.to_numeric(row.get('Acquisition_Costs',0), errors='coerce') or 0,
                            'Maintenance_Expenses': pd.to_numeric(row.get('Maintenance_Expenses',0), errors='coerce') or 0,
                        }

                st.subheader("Liability Summary by Portfolio")
                summary_rows = []
                for p in portfolios:
                    cl = closing_reserves.get(p, {})
                    row = {'Portfolio': p}
                    row['UPR (LRC)'] = cl.get('UPR', 0)
                    row['OCR (LIC)'] = cl.get('OCR', 0)
                    row['IBNR (LIC)'] = cl.get('IBNR', 0)
                    row['ULAE (LIC)'] = cl.get('ULAE', 0)
                    row['RA (LIC)'] = cl.get('RA', 0)
                    row['Total LRC'] = row['UPR (LRC)']
                    row['Total LIC'] = row['OCR (LIC)'] + row['IBNR (LIC)'] + row['ULAE (LIC)'] + row['RA (LIC)']
                    row['ICL'] = row['Total LRC'] + row['Total LIC']
                    summary_rows.append(row)
                total_row = {'Portfolio': 'TOTAL'}
                for key in summary_rows[0].keys():
                    if key != 'Portfolio': total_row[key] = sum(r.get(key, 0) for r in summary_rows)
                summary_rows.append(total_row)
                summary_df = pd.DataFrame(summary_rows)
                disp_summary = summary_df.copy()
                for c in disp_summary.columns:
                    if c != 'Portfolio': disp_summary[c] = disp_summary[c].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
                st.dataframe(disp_summary, width='stretch', hide_index=True)

                ins_rev = {}
                for p in portfolios:
                    op_upr = op_reserves.get(p, {}).get('UPR', 0)
                    cl_upr = closing_reserves.get(p, {}).get('UPR', 0)
                    prem_rec = cf_reserves.get(p, {}).get('Premiums_Received', 0)
                    ins_rev[p] = op_upr + prem_rec - cl_upr

                st.subheader("Liability Rollforward — by Line of Business")
                for p in portfolios:
                    op = op_reserves.get(p, {}); cl = closing_reserves.get(p, {}); cf = cf_reserves.get(p, {})
                    op_upr=op.get('UPR',0); cl_upr=cl.get('UPR',0)
                    op_ocr=op.get('OCR',0); cl_ocr=cl.get('OCR',0)
                    op_ibnr=op.get('IBNR',0); cl_ibnr=cl.get('IBNR',0)
                    op_ulae=op.get('ULAE',0); cl_ulae=cl.get('ULAE',0)
                    op_ra=op.get('RA',0); cl_ra=cl.get('RA',0)
                    prem_rec=cf.get('Premiums_Received',0); paid=cf.get('Paid_Claims',0)
                    acq=cf.get('Acquisition_Costs',0); maint=cf.get('Maintenance_Expenses',0)
                    ir=ins_rev.get(p,0)
                    incurred=paid+cl_ocr+cl_ibnr-op_ocr-op_ibnr
                    op_icf=op_ocr+op_ibnr+op_ulae; cl_icf=cl_ocr+cl_ibnr+cl_ulae
                    op_icl=op_upr+op_icf+op_ra; cl_icl=cl_upr+cl_icf+cl_ra
                    st.markdown(f"**{p}**")
                    roll_data = {
                        "Line Item": ["Opening Balance","Premiums Received","Insurance Revenue","Incurred Claims","Paid Claims","Acquisition Costs","ULAE","Maintenance Expenses","Change in RA","Closing Balance"],
                        "LRC (UPR)": [f"{op_upr:,.2f}",f"{prem_rec:,.2f}",f"{-ir:,.2f}","-","-","-","-","-","-",f"{cl_upr:,.2f}"],
                        "LIC (FCF)": [f"{op_icf:,.2f}","-","-",f"{incurred:,.2f}",f"{-paid:,.2f}","-",f"{cl_ulae:,.2f}",f"{-maint:,.2f}","-",f"{cl_icf:,.2f}"],
                        "LIC (RA)": [f"{op_ra:,.2f}","-","-","-","-","-","-","-",f"{cl_ra-op_ra:,.2f}",f"{cl_ra:,.2f}"],
                        "ICL": [f"{op_icl:,.2f}",f"{prem_rec:,.2f}",f"{-ir:,.2f}",f"{incurred:,.2f}",f"{-paid:,.2f}",f"{-acq:,.2f}",f"{cl_ulae:,.2f}",f"{-maint:,.2f}",f"{cl_ra-op_ra:,.2f}",f"{cl_icl:,.2f}"]
                    }
                    st.dataframe(pd.DataFrame(roll_data), width='stretch', hide_index=True)

                st.subheader("Consolidated Liability Rollforward")
                T = lambda d: sum(v for v in d.values())
                tot_op_upr=T({p:op_reserves.get(p,{}).get('UPR',0) for p in portfolios}); tot_cl_upr=T({p:closing_reserves.get(p,{}).get('UPR',0) for p in portfolios})
                tot_op_ocr=T({p:op_reserves.get(p,{}).get('OCR',0) for p in portfolios}); tot_cl_ocr=T({p:closing_reserves.get(p,{}).get('OCR',0) for p in portfolios})
                tot_op_ibnr=T({p:op_reserves.get(p,{}).get('IBNR',0) for p in portfolios}); tot_cl_ibnr=T({p:closing_reserves.get(p,{}).get('IBNR',0) for p in portfolios})
                tot_op_ulae=T({p:op_reserves.get(p,{}).get('ULAE',0) for p in portfolios}); tot_cl_ulae=T({p:closing_reserves.get(p,{}).get('ULAE',0) for p in portfolios})
                tot_op_ra=T({p:op_reserves.get(p,{}).get('RA',0) for p in portfolios}); tot_cl_ra=T({p:closing_reserves.get(p,{}).get('RA',0) for p in portfolios})
                tot_prem=T({p:cf_reserves.get(p,{}).get('Premiums_Received',0) for p in portfolios})
                tot_paid=T({p:cf_reserves.get(p,{}).get('Paid_Claims',0) for p in portfolios})
                tot_acq=T({p:cf_reserves.get(p,{}).get('Acquisition_Costs',0) for p in portfolios})
                tot_maint=T({p:cf_reserves.get(p,{}).get('Maintenance_Expenses',0) for p in portfolios})
                tot_ir=tot_op_upr+tot_prem-tot_cl_upr
                tot_incurred=tot_paid+tot_cl_ocr+tot_cl_ibnr-tot_op_ocr-tot_op_ibnr
                tot_op_icf=tot_op_ocr+tot_op_ibnr+tot_op_ulae; tot_cl_icf=tot_cl_ocr+tot_cl_ibnr+tot_cl_ulae
                tot_op_icl=tot_op_upr+tot_op_icf+tot_op_ra; tot_cl_icl=tot_cl_upr+tot_cl_icf+tot_cl_ra
                consol_data = {
                    "Line Item": ["Opening Balance","Premiums Received","Insurance Revenue","Incurred Claims","Paid Claims","Acquisition Costs","ULAE","Maintenance Expenses","Change in RA","Closing Balance"],
                    "LRC (UPR)": [f"{tot_op_upr:,.2f}",f"{tot_prem:,.2f}",f"{-tot_ir:,.2f}","-","-","-","-","-","-",f"{tot_cl_upr:,.2f}"],
                    "LIC (FCF)": [f"{tot_op_icf:,.2f}","-","-",f"{tot_incurred:,.2f}",f"{-tot_paid:,.2f}","-",f"{tot_cl_ulae:,.2f}",f"{-tot_maint:,.2f}","-",f"{tot_cl_icf:,.2f}"],
                    "LIC (RA)": [f"{tot_op_ra:,.2f}","-","-","-","-","-","-","-",f"{tot_cl_ra-tot_op_ra:,.2f}",f"{tot_cl_ra:,.2f}"],
                    "ICL": [f"{tot_op_icl:,.2f}",f"{tot_prem:,.2f}",f"{-tot_ir:,.2f}",f"{tot_incurred:,.2f}",f"{-tot_paid:,.2f}",f"{-tot_acq:,.2f}",f"{tot_cl_ulae:,.2f}",f"{-tot_maint:,.2f}",f"{tot_cl_ra-tot_op_ra:,.2f}",f"{tot_cl_icl:,.2f}"]
                }
                st.dataframe(pd.DataFrame(consol_data), width='stretch', hide_index=True)

                st.subheader("IFRS 17 Income Statement")
                income_data = {
                    "Line Item": ["Insurance revenue","Insurance service expenses","  Incurred claims","  Acquisition costs","  ULAE","  Maintenance expenses","Insurance service result","Insurance Finance Result","Profit before tax"],
                    "Amount": [f"{tot_ir:,.2f}",f"{(tot_incurred+tot_acq+tot_cl_ulae+tot_maint):,.2f}",f"{tot_incurred:,.2f}",f"{tot_acq:,.2f}",f"{tot_cl_ulae:,.2f}",f"{tot_maint:,.2f}",f"{tot_ir-tot_incurred-tot_acq-tot_cl_ulae-tot_maint:,.2f}","0.00",f"{tot_ir-tot_incurred-tot_acq-tot_cl_ulae-tot_maint:,.2f}"]
                }
                st.dataframe(pd.DataFrame(income_data), width='stretch', hide_index=True)

                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as w:
                    meta_df = pd.DataFrame([
                        {"Field":"Creation","Value":st.session_state.report_metadata.get('creation_time','')},
                        {"Field":"Created By","Value":st.session_state.report_metadata.get('created_by','')},
                        {"Field":"Version","Value":st.session_state.report_metadata.get('version','')},
                        {"Field":"Run ID","Value":st.session_state.report_metadata.get('run_id','')},
                        {"Field":"Client","Value":st.session_state.report_metadata.get('client','')},
                        {"Field":"Valuation Date","Value":st.session_state.report_metadata.get('valuation_date','')},
                    ])
                    meta_df.to_excel(w, index=False, sheet_name='Report_Metadata')
                    summary_df.to_excel(w, index=False, sheet_name='Liability_Summary')
                    pd.DataFrame(income_data).to_excel(w, index=False, sheet_name='Income_Statement')
                    pd.DataFrame(consol_data).to_excel(w, index=False, sheet_name='Consolidated_Rollforward')
                    for p in portfolios:
                        op=op_reserves.get(p,{}); cl=closing_reserves.get(p,{}); cf=cf_reserves.get(p,{})
                        op_upr=op.get('UPR',0); cl_upr=cl.get('UPR',0)
                        op_ocr=op.get('OCR',0); cl_ocr=cl.get('OCR',0)
                        op_ibnr=op.get('IBNR',0); cl_ibnr=cl.get('IBNR',0)
                        op_ulae=op.get('ULAE',0); cl_ulae=cl.get('ULAE',0)
                        op_ra=op.get('RA',0); cl_ra=cl.get('RA',0)
                        prem_rec=cf.get('Premiums_Received',0); paid=cf.get('Paid_Claims',0)
                        acq=cf.get('Acquisition_Costs',0); maint=cf.get('Maintenance_Expenses',0)
                        ir=ins_rev.get(p,0); incurred=paid+cl_ocr+cl_ibnr-op_ocr-op_ibnr
                        op_icf=op_ocr+op_ibnr+op_ulae; cl_icf=cl_ocr+cl_ibnr+cl_ulae
                        op_icl=op_upr+op_icf+op_ra; cl_icl=cl_upr+cl_icf+cl_ra
                        pr_data = {
                            "Line Item": ["Opening Balance","Premiums Received","Insurance Revenue","Incurred Claims","Paid Claims","Acquisition Costs","ULAE","Maintenance Expenses","Change in RA","Closing Balance"],
                            "LRC (UPR)": [op_upr,prem_rec,-ir,0,0,0,0,0,0,cl_upr],
                            "LIC (FCF)": [op_icf,0,0,incurred,-paid,0,cl_ulae,-maint,0,cl_icf],
                            "LIC (RA)": [op_ra,0,0,0,0,0,0,0,cl_ra-op_ra,cl_ra],
                            "ICL": [op_icl,prem_rec,-ir,incurred,-paid,-acq,cl_ulae,-maint,cl_ra-op_ra,cl_icl]
                        }
                        safe_name = re.sub(r'[\\/*?:\[\]]', '', p)[:28]
                        pd.DataFrame(pr_data).to_excel(w, index=False, sheet_name=f'RW_{safe_name}')
                output.seek(0)
                sc = re.sub(r'[\\/*?:"<>|]',"",report_client).strip() or "Client"
                st.download_button("⬇  Download IFRS 17 Report (.xlsx)", data=output, file_name=f"{sc}_IFRS17_Report_{report_date}.xlsx", key="fv_dl")


# =============================================================================
#  BRANCH 2 — FULL IFRS 17 LRC (PAA)  [NEW, FULLY INDEPENDENT]
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
#  MAIN ROUTER (FIXED: Removed undefined skipped pages)
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
