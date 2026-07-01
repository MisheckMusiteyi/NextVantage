# -*- coding: utf-8 -*-
# =============================================================================
#  NEXT VANTAGE - MAIN APP ENTRY POINT
#  Modularized LIC Individual Calculators + Full IFRS 17 LRC Mode
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import date, datetime
import re
from scipy import interpolate  # <-- Added for Inflation/Discounting interpolation

# =============================================================================
#  IMPORT PURE ENGINES FROM LIC CALCULATORS FOLDERS
# =============================================================================

# UPR
from LRC_Calculators.upr_engine import calculate_upr

# Loss Component
from LRC_Calculators.loss_component_engine import calculate_loss_component

# OCR
from LIC_Calculators.FCF_Calculators.OCR_Calculators.ocr_engine import calculate_ocr

# IBNR
from LIC_Calculators.FCF_Calculators.IBNR_Calculators.percentage_ibnr import calculate_percentage_ibnr
from LIC_Calculators.FCF_Calculators.IBNR_Calculators.bcl_ibnr import calculate_bcl_ibnr
from LIC_Calculators.FCF_Calculators.IBNR_Calculators.cape_cod_ibnr import calculate_cape_cod_ibnr
from LIC_Calculators.FCF_Calculators.IBNR_Calculators.bf_ibnr import calculate_bf_ibnr

# ULAE & NPR
from LIC_Calculators.FCF_Calculators.ULAE_Calculators.ulae_engine import calculate_ulae_per_portfolio, calculate_ulae_aggregated, calculate_apportionment_percentages
from LIC_Calculators.FCF_Calculators.NPR_Calculators.npr_engine import calculate_npr_aggregation, calculate_npr_per_portfolio

# RA (Risk Adjustment)
from LIC_Calculators.RA_Calculators.mack_ra import calculate_mack_chain_ladder
from LIC_Calculators.RA_Calculators.bootstrap_ra import bootstrap_chain_ladder, calculate_risk_adjustment

# SHARED HELPERS
from utils.actuarial_helpers import (
    build_triangles, volume_weighted_factors, simple_average_factors,
    geometric_average_factors, medial_average_factors, linear_regression_factors,
    weighted_last_n_factors, stability_diagnostics, recommend_factors,
    compute_cdfs, project_ultimate, deflate_triangle_to_real,
    reinflate_ibnr_per_ap, discount_completed_triangle,
    period_index, period_label, periods_per_year
)

# FULL IFRS 17 LRC ENGINE (SEPARATE FILE)
from Full_Valuation.full_LRC_IFRS17 import calculate_full_ifrs17_lrc


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


# =============================================================================
#  UI HELPER: COLUMN MAPPER
# =============================================================================

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
                    mapped[field] = st.selectbox(f"{field}", all_cols, index=default_idx, key=f"map_{file_label}_{field}")
    return mapped


# =============================================================================
#  SHARED TRIANGLE CALCULATOR UI HELPER
# =============================================================================

def render_triangle_calculator(title, client_name_key, engine_callback):
    """
    Generic UI for BCL, Cape Cod, and BF.
    engine_callback is a function that takes (cum_triangle, obs_mask, factors, from_dt, grain, ...) and returns results.
    """
    show_breadcrumb()
    st.markdown(f'<div class="hero"><h1>{title}</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    client_name = st.text_input("Client name", value="Client", key=client_name_key).strip()

    # ---- Step 1: Load Claims Data ----
    st.markdown("#### Step 1: Upload Claims Data")
    claims_file = st.file_uploader("Upload Claims Data (CSV/Excel)", type=["csv","xlsx","xls"], key=f"{client_name_key}_f")
    if claims_file is None:
        st.info("Please upload a claims data file to proceed.")
        back_button('ibnr_menu', ['Home','LIC','Fulfilment Cashflows','IBNR Methods'])
        return

    try:
        # Load and preview
        df = pd.read_csv(claims_file) if claims_file.name.endswith('.csv') else pd.read_excel(claims_file)
        df.columns = df.columns.astype(str).str.strip()
        st.dataframe(df.head(5), use_container_width=True)

        # ---- Step 2: Map Columns ----
        st.markdown("#### Step 2: Map Columns")
        all_cols = df.columns.tolist()
        c1, c2, c3, c4 = st.columns(4)
        with c1: loss_col = st.selectbox("Loss Date Column", all_cols, key=f"{client_name_key}_loss")
        with c2: report_col = st.selectbox("Report Date Column", all_cols, key=f"{client_name_key}_rep")
        with c3: amt_col = st.selectbox("Amount Column", [c for c in all_cols if c not in [loss_col, report_col] and pd.api.types.is_numeric_dtype(df[c])], key=f"{client_name_key}_amt")
        with c4: 
            grouping_opts = [c for c in all_cols if c not in [loss_col, report_col, amt_col]]
            grouping_col = st.selectbox("Group By (Optional)", ["None"] + grouping_opts, key=f"{client_name_key}_grp")

        # ---- Step 3: Period & Granularity ----
        st.markdown("#### Step 3: IBNR Period & Granularity")
        c1, c2, c3 = st.columns(3)
        with c1: from_date = st.date_input("From Date", value=date(2020,1,1), key=f"{client_name_key}_from")
        with c2: to_date = st.date_input("To Date", value=date(2025,12,31), key=f"{client_name_key}_to")
        with c3: grain = st.selectbox("Granularity", ["Yearly", "Half-Yearly", "Quarterly", "Monthly"], key=f"{client_name_key}_grain")

        # ---- Step 4: Data Validation ----
        st.markdown("#### Step 4: Data Validation")
        df[loss_col] = pd.to_datetime(df[loss_col], errors='coerce')
        df[report_col] = pd.to_datetime(df[report_col], errors='coerce')
        from_dt = pd.to_datetime(from_date)
        to_dt = pd.to_datetime(to_date)
        df = df[(df[loss_col] >= from_dt) & (df[loss_col] <= to_dt)].copy()

        # Missing values
        missing_rows = df[[loss_col, report_col, amt_col]].isnull().any(axis=1)
        if missing_rows.any():
            st.error(f"❌ {missing_rows.sum()} rows have missing values.")
            st.stop()
        else: st.success("✅ No missing values.")

        # Date consistency
        inconsistent_dates = df[df[loss_col] > df[report_col]]
        if not inconsistent_dates.empty:
            st.error(f"❌ {len(inconsistent_dates)} rows have Loss Date > Report Date.")
            st.stop()
        else: st.success("✅ Date consistency check passed.")

        # Duplicates
        before = len(df); df = df.drop_duplicates()
        if len(df) < before: st.warning(f"⚠️ {before - len(df)} duplicate rows removed.")

        # Calculate periods
        g_map = {"Yearly": "Y", "Half-Yearly": "H", "Quarterly": "Q", "Monthly": "M"}
        g_code = g_map[grain]
        ppy = {"Y":1, "H":2, "Q":4, "M":12}[g_code]
        if g_code == "Y": n_periods = to_date.year - from_date.year + 1
        elif g_code == "M": n_periods = (to_date.year - from_date.year)*12 + (to_date.month - from_date.month) + 1
        elif g_code == "Q": n_periods = (to_date.year - from_date.year)*4 + ((to_date.month-1)//3 - (from_date.month-1)//3) + 1
        elif g_code == "H": n_periods = (to_date.year - from_date.year)*2 + ((to_date.month-1)//6 - (from_date.month-1)//6) + 1

        # ---- Step 5: Inflation & Discounting ----
        st.markdown("#### Step 5: Inflation & Discounting Adjustments")
        c1, c2 = st.columns(2)
        with c1: use_inflation = st.checkbox("Apply Inflation Adjustment", key=f"{client_name_key}_inf")
        with c2: use_discounting = st.checkbox("Apply Discounting", key=f"{client_name_key}_disc")

        cum_inflation = None; per_period_rates = None; spot_rates = None; flat_rate = None

        if use_inflation:
            st.markdown("**Upload Inflation Data**")
            inf_file = st.file_uploader("Upload Inflation Curve", type=["csv","xlsx","xls"], key=f"{client_name_key}_inf_f")
            if inf_file:
                inf_df = pd.read_csv(inf_file) if inf_file.name.endswith('.csv') else pd.read_excel(inf_file)
                p_col = st.selectbox("Period column", inf_df.columns, key=f"{client_name_key}_inf_p")
                r_col = st.selectbox("Rate column", inf_df.columns, key=f"{client_name_key}_inf_r")
                inf_df = inf_df[[p_col, r_col]].dropna()
                inf_df[r_col] = pd.to_numeric(inf_df[r_col], errors='coerce') / 100.0
                rates_inf = inf_df[r_col].values
                ratio = ppy / {"Y":1, "H":2, "Q":4, "M":12}[g_code]
                x_inf = np.arange(len(rates_inf)) * ratio
                x_tgt = np.arange(int(x_inf[-1]) + 1)
                if len(x_inf) >= 4: f_interp = interpolate.CubicSpline(x_inf, rates_inf, extrapolate=True)
                else: f_interp = interpolate.interp1d(x_inf, rates_inf, kind='linear', fill_value='extrapolate')
                annual_rates_tgt = np.clip(f_interp(x_tgt), -0.5, 2.0)
                per_period_rates = (1 + annual_rates_tgt) ** (1 / ppy) - 1
                cum_inflation = np.cumprod(1 + per_period_rates)
                st.success(f"Inflation interpolated to {grain}.")

        if use_discounting:
            st.markdown("**Upload Yield Curve**")
            disc_method = st.radio("Discounting Method", ["Yield Curve", "Single Flat Rate"], key=f"{client_name_key}_disc_method")
            if disc_method == "Yield Curve":
                yc_file = st.file_uploader("Upload Yield Curve", type=["csv","xlsx","xls"], key=f"{client_name_key}_yc")
                if yc_file:
                    yc_df = pd.read_csv(yc_file) if yc_file.name.endswith('.csv') else pd.read_excel(yc_file)
                    m_col = st.selectbox("Maturity column", yc_df.columns, key=f"{client_name_key}_yc_m")
                    r_col = st.selectbox("Rate column", yc_df.columns, key=f"{client_name_key}_yc_r")
                    yc_df = yc_df[[m_col, r_col]].dropna()
                    yc_df[m_col] = pd.to_numeric(yc_df[m_col], errors='coerce')
                    yc_df[r_col] = pd.to_numeric(yc_df[r_col], errors='coerce') / 100.0
                    maturities = yc_df[m_col].values; rates = yc_df[r_col].values
                    if len(maturities) >= 4: f_interp = interpolate.CubicSpline(maturities, rates, extrapolate=True)
                    else: f_interp = interpolate.interp1d(maturities, rates, kind='linear', fill_value='extrapolate')
                    period_maturities = np.arange(1, 61) / ppy
                    spot_rates = np.clip(f_interp(period_maturities), 0, 1.0)
                    st.success(f"Yield Curve interpolated to {grain}.")
            else:
                flat_rate = st.number_input("Annual Discount Rate (%)", 0.0, 50.0, 5.0, 0.5, key=f"{client_name_key}_flat") / 100.0

        # ---- Step 6: Run Calculation ----
        if st.button("Run Calculation", key=f"{client_name_key}_run", use_container_width=True):
            # Build triangle
            inc, cum, obs_mask = build_triangles(df, loss_col, report_col, amt_col, from_dt, g_code, n_periods)
            
            # Inflation
            if use_inflation and cum_inflation is not None:
                _, real_cum = deflate_triangle_to_real(inc, cum_inflation, n_periods)
                working_cum = real_cum
            else: working_cum = cum

            # Calculate all LDFs
            vw = volume_weighted_factors(working_cum)
            sa = simple_average_factors(working_cum)
            geo = geometric_average_factors(working_cum)
            med = medial_average_factors(working_cum)
            lr_tuple = linear_regression_factors(working_cum)
            lr, slope, intercept, r2 = lr_tuple
            wln = weighted_last_n_factors(working_cum, n=3)

            cvs = stability_diagnostics(working_cum)
            recs, mean_cv = recommend_factors(vw, sa, geo, med, lr_tuple, cvs)

            # Display LDF Selection
            st.subheader("LDF Selection")
            factor_df = pd.DataFrame({
                "Dev Period": range(1, len(vw)+1),
                "Vol-Weighted": vw, "Simple Avg": sa, "Geometric": geo,
                "Medial Avg": med, "Lin Reg": lr, "Wtd Last 3": wln
            })
            st.dataframe(factor_df, use_container_width=True)
            st.write(f"**Stability Diagnostics:** Mean CV: {mean_cv:.2%} | R²: {r2:.4f}")
            st.info(f"**Recommendation:** {recs[0][0]} — {recs[0][1]}")

            selected_method = st.selectbox(
                "Select LDF Method",
                ["Volume-Weighted", "Simple Average", "Geometric", "Medial Average", "Linear Regression", "Weighted Last 3"],
                key=f"{client_name_key}_method"
            )
            if selected_method == "Volume-Weighted": chosen = vw
            elif selected_method == "Simple Average": chosen = sa
            elif selected_method == "Geometric": chosen = geo
            elif selected_method == "Medial Average": chosen = med
            elif selected_method == "Linear Regression": chosen = lr
            elif selected_method == "Weighted Last 3": chosen = wln

            # Call the specific engine
            results = engine_callback(working_cum, chosen, from_dt, g_code)

            # Display results
            st.subheader("Results")
            st.dataframe(results['results_df'], use_container_width=True)
            st.write(f"**Total IBNR: {results['total_ibnr']:,.2f}**")

            # Export
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w:
                results['results_df'].to_excel(w, index=False, sheet_name="IBNR_Results")
            output.seek(0)
            st.download_button("⬇ Download Report", data=output, file_name=f"{client_name}_IBNR.xlsx", key=f"{client_name_key}_dl")

    except Exception as e: st.error(f"Error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
    back_button('ibnr_menu', ['Home','LIC','Fulfilment Cashflows','IBNR Methods'])


# =============================================================================
#  STREAMLIT UI RENDERERS
# =============================================================================

# -----------------------------------------------------------------------------
# 1. OCR CALCULATOR
# -----------------------------------------------------------------------------

def render_ocr_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>OCR Calculator</h1><p>Outstanding Claims Reserve Aggregation</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    client_name = st.text_input("Client name", value="Client", key="ocr_cn").strip()
    file = st.file_uploader("Upload Claims Data (CSV/Excel)", type=["csv","xlsx","xls"], key="ocr_f")
    if file is None:
        st.info("Please upload a file to proceed.")
        back_button('fulfilment_cashflows', ['Home','LIC','Fulfilment Cashflows'])
        return

    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df.columns = df.columns.astype(str).str.strip()
        st.dataframe(df.head(5), use_container_width=True)
        cols = df.columns.tolist()

        grouping_cols = st.multiselect("Select Grouping Columns (e.g., Line of Business)", cols, key="ocr_grp")
        if not grouping_cols: return

        numeric_cols = [c for c in cols if c not in grouping_cols and pd.api.types.is_numeric_dtype(df[c])]
        amount_cols = st.multiselect("Select Amount Columns", numeric_cols, key="ocr_amt")
        if not amount_cols: return

        if st.button("Calculate OCR", key="ocr_run", use_container_width=True):
            results, report, grand_total = calculate_ocr(df, grouping_cols, amount_cols, clean_data=True)
            st.subheader("OCR Results")
            st.dataframe(results, use_container_width=True)
            st.success(f"Grand Total OCR: {grand_total:,.2f}")

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w:
                results.to_excel(w, index=False, sheet_name="OCR_Results")
            output.seek(0)
            st.download_button("⬇ Download Report", data=output, file_name=f"{client_name}_OCR.xlsx", key="ocr_dl")
    except Exception as e: st.error(f"Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
    back_button('fulfilment_cashflows', ['Home','LIC','Fulfilment Cashflows'])


# -----------------------------------------------------------------------------
# 2. PERCENTAGE IBNR CALCULATOR
# -----------------------------------------------------------------------------

def render_percentage_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>IBNR Percentage Method</h1><p>IBNR = Amount × Percentage</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    client_name = st.text_input("Client name", value="Client", key="pct_cn").strip()
    c1, c2, c3 = st.columns(3)
    with c1: from_date = st.date_input("From Date", value=date(2025,1,1), key="pct_from")
    with c2: to_date = st.date_input("To Date", value=date(2025,12,31), key="pct_to")
    with c3: ibnr_pct = st.number_input("IBNR %", 0.0, 100.0, 10.0, 0.5, key="pct_pct") / 100.0

    file = st.file_uploader("Upload Data", type=["csv","xlsx","xls"], key="pct_f")
    if file is None:
        st.info("Please upload a file.")
        back_button('ibnr_menu', ['Home','LIC','Fulfilment Cashflows','IBNR Methods'])
        return

    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df.columns = df.columns.astype(str).str.strip()
        st.dataframe(df.head(5), use_container_width=True)
        cols = df.columns.tolist()
        date_col = st.selectbox("Date Column", cols, key="pct_date")
        lob_col = st.selectbox("Line of Business Column", cols, key="pct_lob")
        num_cols = [c for c in cols if c not in [date_col, lob_col] and pd.api.types.is_numeric_dtype(df[c])]
        amount_cols = st.multiselect("Amount Columns", num_cols, key="pct_amt")
        if not amount_cols: return

        if st.button("Calculate", key="pct_run", use_container_width=True):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            results, total = calculate_percentage_ibnr(df, date_col, lob_col, amount_cols, pd.to_datetime(from_date), pd.to_datetime(to_date), ibnr_pct)
            st.subheader("Results")
            st.dataframe(results, use_container_width=True)
            st.success(f"Grand Total IBNR: {total:,.2f}")

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w:
                results.to_excel(w, index=False, sheet_name="IBNR_Summary")
            output.seek(0)
            st.download_button("⬇ Download", data=output, file_name=f"{client_name}_IBNR_Percentage.xlsx", key="pct_dl")
    except Exception as e: st.error(f"Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
    back_button('ibnr_menu', ['Home','LIC','Fulfilment Cashflows','IBNR Methods'])


# -----------------------------------------------------------------------------
# 3. BCL (CHAIN LADDER) CALCULATOR
# -----------------------------------------------------------------------------

def render_bcl_calculator():
    from LIC Calculators.FCF Calculators.IBNR Calculators.bcl_ibnr import calculate_bcl_ibnr
    render_triangle_calculator("BCL Chain-Ladder", "bcl", calculate_bcl_ibnr)


# -----------------------------------------------------------------------------
# 4. CAPE COD CALCULATOR
# -----------------------------------------------------------------------------

def render_capecod_calculator():
    from LIC Calculators.FCF Calculators.IBNR Calculators.cape_cod_ibnr import calculate_cape_cod_ibnr
    render_triangle_calculator("Cape Cod IBNR", "cc", calculate_cape_cod_ibnr)


# -----------------------------------------------------------------------------
# 5. BF (BORNHUETTER-FERGUSON) CALCULATOR
# -----------------------------------------------------------------------------

def render_bf_calculator():
    from LIC Calculators.FCF Calculators.IBNR Calculators.bf_ibnr import calculate_bf_ibnr
    render_triangle_calculator("Bornhuetter-Ferguson IBNR", "bf", calculate_bf_ibnr)


# -----------------------------------------------------------------------------
# 6. ULAE CALCULATOR
# -----------------------------------------------------------------------------

def render_ulae_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>ULAE Calculator</h1><p>Unallocated Loss Adjustment Expenses</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    client_name = st.text_input("Client name", value="Client", key="ulae_cn").strip()
    c1, c2 = st.columns(2)
    with c1: basis = st.selectbox("Basis", ["Per Portfolio", "Aggregated"], key="ulae_basis")
    with c2: is_detailed = st.checkbox("Detailed (OCR + IBNR)", value=True, key="ulae_detailed")

    file = st.file_uploader("Upload Reserves File", type=["csv","xlsx","xls"], key="ulae_f")
    if file is None:
        st.info("Please upload a file.")
        back_button('fulfilment_cashflows', ['Home','LIC','Fulfilment Cashflows'])
        return

    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df.columns = df.columns.astype(str).str.strip()
        st.dataframe(df.head(5), use_container_width=True)
        cols = df.columns.tolist()
        
        # Map columns
        portfolio_col = st.selectbox("Portfolio Column", cols, key="ulae_port")
        if is_detailed:
            ocr_col = st.selectbox("OCR Column", [c for c in cols if c != portfolio_col], key="ulae_ocr")
            ibnr_col = st.selectbox("IBNR Column", [c for c in cols if c not in [portfolio_col, ocr_col]], key="ulae_ibnr")
            df = df.rename(columns={portfolio_col: "Portfolio", ocr_col: "OCR", ibnr_col: "IBNR"})
            df["Total_Reserves"] = df["OCR"] + df["IBNR"]
        else:
            total_col = st.selectbox("Total Reserves Column", [c for c in cols if c != portfolio_col], key="ulae_total")
            df = df.rename(columns={portfolio_col: "Portfolio", total_col: "Total_Reserves"})
            df["OCR"] = np.nan; df["IBNR"] = np.nan

        portfolios = df["Portfolio"].unique().tolist()
        st.markdown("### ULAE Ratios")
        ratio_method = st.selectbox("Ratio Input Method", ["Overall", "Per Portfolio Manual", "Per Portfolio File"], key="ulae_rm")

        if ratio_method == "Overall":
            ratio = st.number_input("ULAE Ratio (%)", 0.0, 100.0, 5.0, 0.5, key="ulae_ratio") / 100.0
            ulae_ratios = {p: ratio for p in portfolios}
        elif ratio_method == "Per Portfolio Manual":
            ulae_ratios = {}
            for p in portfolios:
                ulae_ratios[p] = st.number_input(f"Ratio for {p} (%)", 0.0, 100.0, 5.0, 0.5, key=f"ulae_ratio_{p}") / 100.0
        else:
            st.warning("File-based ratios not yet implemented in this demo.")

        apportionment_df = None
        if basis == "Aggregated":
            st.markdown("### Apportionment Key")
            app_file = st.file_uploader("Upload Apportionment File", type=["csv","xlsx","xls"], key="ulae_app")
            if app_file:
                app_df = pd.read_csv(app_file) if app_file.name.endswith('.csv') else pd.read_excel(app_file)
                app_col = st.selectbox("Apportionment Amount Column", app_df.columns, key="ulae_app_amt")
                app_df = app_df.rename(columns={app_col: "Amount"})
                apportionment_df = calculate_apportionment_percentages(app_df)

        if st.button("Calculate ULAE", key="ulae_run", use_container_width=True):
            if basis == "Per Portfolio":
                results = calculate_ulae_per_portfolio(df, ulae_ratios, is_detailed)
            else:
                overall_ratio = list(ulae_ratios.values())[0]
                results, total_base = calculate_ulae_aggregated(df, overall_ratio, apportionment_df, is_detailed)
            
            st.subheader("Results")
            st.dataframe(results, use_container_width=True)
    except Exception as e: st.error(f"Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
    back_button('fulfilment_cashflows', ['Home','LIC','Fulfilment Cashflows'])


# -----------------------------------------------------------------------------
# 7. NPR CALCULATOR
# -----------------------------------------------------------------------------

def render_npr_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>NPR Calculator</h1><p>Reinsurance Non-Performance Risk</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    client_name = st.text_input("Client name", value="Client", key="npr_cn").strip()
    per_portfolio = st.radio("Share Basis", ["Per Portfolio", "Aggregation"], key="npr_basis") == "Per Portfolio"

    st.markdown("### Reinsurer File")
    ri_file = st.file_uploader("Upload Reinsurer Data", type=["csv","xlsx","xls"], key="npr_ri")
    st.markdown("### Ceded LIC File")
    lic_file = st.file_uploader("Upload Ceded LIC Data", type=["csv","xlsx","xls"], key="npr_lic")

    if ri_file and lic_file:
        try:
            df_ri = pd.read_csv(ri_file) if ri_file.name.endswith('.csv') else pd.read_excel(ri_file)
            df_lic = pd.read_csv(lic_file) if lic_file.name.endswith('.csv') else pd.read_excel(lic_file)
            
            st.subheader("Reinsurer Data Preview")
            st.dataframe(df_ri.head(5))
            st.subheader("LIC Data Preview")
            st.dataframe(df_lic.head(5))

            # Simplified mapping for demo
            st.info("Full column mapping and calculation logic is available in the `npr_engine.py` file. This UI is a placeholder.")
        except Exception as e: st.error(f"Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
    back_button('fulfilment_cashflows', ['Home','LIC','Fulfilment Cashflows'])


# -----------------------------------------------------------------------------
# 8. MACK RA CALCULATOR
# -----------------------------------------------------------------------------

def render_mack_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Mack Risk Adjustment</h1><p>Standard Error Method (1993)</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.warning("Mack RA requires the full triangle UI. Using the shared helper flow...")
    st.markdown("The Mack engine (`mack_ra.py`) is complete. The UI is ready to be hooked up.")

    st.markdown('</div>', unsafe_allow_html=True)
    back_button('risk_adjustment', ['Home','LIC','Risk Adjustment'])


# -----------------------------------------------------------------------------
# 9. BOOTSTRAP RA CALCULATOR
# -----------------------------------------------------------------------------

def render_bootstrap_calculator():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Bootstrap RA</h1><p>ODP Bootstrap (England & Verrall 2002)</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.warning("Bootstrap RA requires the full triangle UI. Using the shared helper flow...")
    st.markdown("The Bootstrap engine (`bootstrap_ra.py`) is complete. The UI is ready to be hooked up.")

    st.markdown('</div>', unsafe_allow_html=True)
    back_button('risk_adjustment', ['Home','LIC','Risk Adjustment'])


# =============================================================================
#  NAVIGATION MENUS & HOME PAGE
# =============================================================================

def render_home():
    st.markdown('<div class="hero"><h1>Next Vantage</h1><p>Comprehensive Actuarial Reserving Toolkit — IFRS 17 Compliant</p></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><h3>Full IFRS 17 Valuation</h3><p>LRC (PAA) + UPR Modes</p></div>', unsafe_allow_html=True)
        if st.button("Full Valuation", key="nav_home_full"): navigate_to('full_valuation', ['Home','Full Valuation']); st.rerun()
    with col2:
        st.markdown('<div class="card"><h3>Individual Calculators</h3><p>LRC | LIC | Risk Adjustment</p></div>', unsafe_allow_html=True)
        if st.button("Calculators", key="nav_home_calc"): navigate_to('lrc', ['Home','Individual Calculators','LRC']); st.rerun()
    with col3:
        st.markdown('<div class="card"><h3>LIC</h3><p>Fulfilment Cashflows | Risk Adjustment</p></div>', unsafe_allow_html=True)
        if st.button("Go to LIC", key="nav_home_lic"): navigate_to('lic', ['Home','LIC']); st.rerun()


def render_lrc():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Individual LRC Calculators</h1><p>Liability for Remaining Coverage</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>UPR Calculator</h3></div>', unsafe_allow_html=True)
        if st.button("Open UPR", key="nav_lrc_upr"): navigate_to('upr_calculator', ['Home','Individual Calculators','LRC','UPR']); st.rerun()
    with col2:
        st.markdown('<div class="card"><h3>Loss Component</h3></div>', unsafe_allow_html=True)
        if st.button("Open Loss Component", key="nav_lrc_loss"): navigate_to('loss_component', ['Home','Individual Calculators','LRC','Loss Component']); st.rerun()
    back_button('home', ['Home'])


def render_lic():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>LIC Calculators</h1><p>Liability for Incurred Claims</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>Fulfilment Cashflows</h3><p>OCR, IBNR, ULAE, NPR</p></div>', unsafe_allow_html=True)
        if st.button("Open FCF", key="nav_lic_fulfil"): navigate_to('fulfilment_cashflows', ['Home','LIC','Fulfilment Cashflows']); st.rerun()
    with col2:
        st.markdown('<div class="card"><h3>Risk Adjustment</h3><p>Bootstrap, Mack</p></div>', unsafe_allow_html=True)
        if st.button("Open RA", key="nav_lic_ra"): navigate_to('risk_adjustment', ['Home','LIC','Risk Adjustment']); st.rerun()
    back_button('home', ['Home'])


def render_fulfilment_cashflows():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Fulfilment Cashflows</h1><p>LIC Components</p></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    items = [("OCR", "ocr_calculator"), ("IBNR Methods", "ibnr_menu"), ("ULAE", "ulae_calculator"), ("NPR", "npr_calculator")]
    for i, (t, p) in enumerate(items):
        with cols[i]:
            st.markdown(f'<div class="card"><h3>{t}</h3></div>', unsafe_allow_html=True)
            if st.button(f"Open {t}", key=f"nav_fc_{p}"): navigate_to(p, ['Home','LIC','Fulfilment Cashflows',t]); st.rerun()
    back_button('lic', ['Home','LIC'])


def render_ibnr_menu():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>IBNR Methods</h1></div>', unsafe_allow_html=True)
    methods = [
        ("BCL", "bcl_calculator"),
        ("Cape Cod", "capecod_calculator"),
        ("BF", "bf_calculator"),
        ("Percentage", "percentage_calculator"),
        ("ELR", "elr_calculator"),
        ("ACPC", "acpc_calculator")
    ]
    for i in range(0, len(methods), 3):
        cols = st.columns(3)
        for j in range(3):
            if i+j < len(methods):
                n, p = methods[i+j]
                with cols[j]:
                    st.markdown(f'<div class="card"><h3>{n}</h3></div>', unsafe_allow_html=True)
                    if st.button(f"Open {n}", key=f"nav_ibnr_{p}"): navigate_to(p, ['Home','LIC','Fulfilment Cashflows','IBNR Methods',n]); st.rerun()
    back_button('fulfilment_cashflows', ['Home','LIC','Fulfilment Cashflows'])


def render_risk_adjustment():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Risk Adjustment</h1><p>RA Methods</p></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    methods = [("Mack", "mack_calculator"), ("Bootstrap", "bootstrap_calculator"), ("VaR", "var_calculator"), ("Cost of Capital", "coc_calculator")]
    for i, (n, p) in enumerate(methods):
        with cols[i]:
            st.markdown(f'<div class="card"><h3>{n}</h3></div>', unsafe_allow_html=True)
            if st.button(f"Open {n}", key=f"nav_ra_{p}"): navigate_to(p, ['Home','LIC','Risk Adjustment',n]); st.rerun()
    back_button('lic', ['Home','LIC'])


# =============================================================================
#  FULL VALUATION (LRC MODE)
# =============================================================================

def render_full_valuation():
    show_breadcrumb()
    st.markdown('<div class="hero"><h1>Full IFRS 17 LRC Valuation</h1><p>Liability for Remaining Coverage (PAA)</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # ---- REPORT METADATA ----
    st.markdown('<div class="section-container"><h3>Report Metadata</h3></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: report_created_by = st.text_input("Created By", value="", key="fv_cb")
    with c2: report_version = st.text_input("Version", value="3.9.29.3", key="fv_ver")
    with c3: report_client = st.text_input("Client Name", value="", key="fv_client")
    with c4: report_date = st.date_input("Valuation Date", value=date.today(), key="fv_vd")

    st.session_state.report_metadata = {
        'created_by': report_created_by, 'version': report_version,
        'client': report_client, 'valuation_date': str(report_date)
    }

    # ---- VALUATION MODE SELECTOR ----
    st.markdown('<div class="section-container"><h3>Valuation Mode</h3></div>', unsafe_allow_html=True)
    valuation_mode = st.radio(
        "Select Valuation Mode",
        options=["Simplified UPR", "Full IFRS 17 LRC (PAA)"],
        index=0,
        key="fv_mode"
    )

    # =========================================================================
    #  BRANCH 1: SIMPLIFIED UPR (IFRS 4 Style)
    # =========================================================================
    if valuation_mode == "Simplified UPR":
        st.info("Simplified UPR Mode: Calculates Unearned Premium Reserve using 365th / 24th / 8th methods.")
        col1, col2, col3 = st.columns(3)
        with col1: upr_method = st.selectbox("UPR Method", ["365th", "24th", "8th"], key="upr_method")
        
        file = st.file_uploader("Upload Premium Register (CSV/Excel)", type=["csv","xlsx","xls"], key="upr_file")
        if file:
            try:
                df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                st.dataframe(df.head(5))
                
                if st.button("Calculate UPR", key="calc_upr"):
                    st.success("UPR calculation would run here.")
            except Exception as e: st.error(f"Error: {e}")

    # =========================================================================
    #  BRANCH 2: FULL IFRS 17 LRC (PAA)  [NEW ENGINE]
    # =========================================================================
    else:
        st.markdown('<div class="section-container"><h3>IFRS 17 Configuration Toggles</h3></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: iacf_toggle = st.selectbox("IACF Treatment", ["Expense Immediately", "Capitalize & Amortize"], key="cfg_iacf")
        with c2: discount_toggle = st.selectbox("Discounting", ["No Discounting", "Apply Discounting"], key="cfg_discount")
        with c3: invest_toggle = st.selectbox("Investment Components", ["No", "Yes"], key="cfg_invest")
        with c4: revenue_toggle = st.selectbox("Revenue Method", ["Passage of Time", "Emergence of Risk"], key="cfg_revenue")

        st.markdown("#### Upload Data Files")
        
        # 6 Sections of Data
        ob_file = st.file_uploader("Opening Balances (Group, Opening_LRC_Excl_Loss, Opening_Loss_Component)", type=["csv","xlsx","xls"], key="ifrs17_ob")
        cf_file = st.file_uploader("Cashflows (Group, Premiums_Received, IACF_Paid, Investment_Components_Paid)", type=["csv","xlsx","xls"], key="ifrs17_cf")
        pd_file = st.file_uploader("Policy Data (Group, Start_Date, End_Date, Written_Premium)", type=["csv","xlsx","xls"], key="ifrs17_pd")
        lc_file = st.file_uploader("Loss Component Data (Group, Expected_Future_Premiums, Loss_Ratio, Commission_Ratio, Expense_Ratio, RA_Ratio)", type=["csv","xlsx","xls"], key="ifrs17_lc")
        
        yc_file = None
        if discount_toggle == "Apply Discounting":
            yc_file = st.file_uploader("Yield Curve (Duration_Years, Spot_Rate)", type=["csv","xlsx","xls"], key="ifrs17_yc")
        
        rc_file = None
        if revenue_toggle == "Emergence of Risk":
            rc_file = st.file_uploader("Claims Curve (Period, Percentage)", type=["csv","xlsx","xls"], key="ifrs17_rc")

        if st.button("Run IFRS 17 LRC", key="ifrs17_run"):
            if pd_file is None:
                st.warning("Please upload Policy Data (Section 3).")
                return

            try:
                # Load data into DataFrames
                opening_balances = pd.read_csv(ob_file) if ob_file else pd.DataFrame()
                cashflows = pd.read_csv(cf_file) if cf_file else pd.DataFrame()
                policy = pd.read_csv(pd_file) if pd_file else pd.DataFrame()
                loss_component = pd.read_csv(lc_file) if lc_file else pd.DataFrame()
                yield_curve = pd.read_csv(yc_file) if yc_file else None
                claims_curve = pd.read_csv(rc_file) if rc_file else None

                config = {
                    'iacf_toggle': iacf_toggle,
                    'discount_toggle': discount_toggle,
                    'invest_toggle': invest_toggle,
                    'revenue_toggle': revenue_toggle
                }

                # CALL THE PURE ENGINE FROM THE SEPARATE FILE
                results = calculate_full_ifrs17_lrc(
                    opening_balances, cashflows, policy, loss_component,
                    yield_curve, claims_curve, config, report_date
                )

                st.success("Full IFRS 17 LRC Calculated Successfully!")
                st.dataframe(pd.DataFrame(results).T)

            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
    back_button('home', ['Home'])


# =============================================================================
#  MAIN ROUTER
# =============================================================================

page_renderers = {
    'home': render_home,
    'lrc': render_lrc,
    'lic': render_lic,
    'fulfilment_cashflows': render_fulfilment_cashflows,
    'ibnr_menu': render_ibnr_menu,
    'risk_adjustment': render_risk_adjustment,
    'ocr_calculator': render_ocr_calculator,
    'percentage_calculator': render_percentage_calculator,
    'bcl_calculator': render_bcl_calculator,
    'capecod_calculator': render_capecod_calculator,
    'bf_calculator': render_bf_calculator,
    'ulae_calculator': render_ulae_calculator,
    'npr_calculator': render_npr_calculator,
    'mack_calculator': render_mack_calculator,
    'bootstrap_calculator': render_bootstrap_calculator,
    'elr_calculator': render_elr_calculator,
    'acpc_calculator': render_acpc_calculator,
    'var_calculator': render_var_calculator,
    'coc_calculator': render_coc_calculator,
    'upr_calculator': render_upr_calculator,
    'loss_component': render_loss_component,
    'full_valuation': render_full_valuation,
}

# =============================================================================
#  APP ENTRY
# =============================================================================

if __name__ == "__main__":
    current_page = st.session_state.page
    if current_page in page_renderers:
        page_renderers[current_page]()
    else:
        render_home()
