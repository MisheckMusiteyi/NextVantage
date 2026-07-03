# =============================================================================
#  CALCULATOR: BF IBNR (WITH MULTI-LDF SUPPORT)
# =============================================================================

def render_bf_calculator():
    show_breadcrumb()
    st.markdown("""
    <div class="hero">
        <h1>Bornhuetter-Ferguson — IBNR</h1>
        <p>Multi-LDF Methods with Expected Loss Ratio</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        client_name = st.text_input("Client Name", value="Client", key="bf_cn").strip()
    with c2:
        from_date = st.date_input("From Date", date(2020, 1, 1), key="bf_fd")
    with c3:
        to_date = st.date_input("To Date", date(2025, 12, 31), key="bf_td")
    
    claims_file = st.file_uploader(
        "📂 Claims Data (Loss Date, Report Date, LOB, Amount)",
        type=["csv", "xlsx", "xls"],
        key="bf_cf"
    )
    
    if claims_file is None:
        st.info("👆 Upload claims data file.")
        back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])
        return
    
    try:
        df = pd.read_csv(claims_file) if claims_file.name.endswith('.csv') else pd.read_excel(claims_file)
        df.columns = df.columns.astype(str).str.strip()
        
        st.markdown("#### Data Preview")
        st.dataframe(df.head(5), use_container_width=True)
        cols = df.columns.tolist()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            loss_col = st.selectbox("Loss Date", cols, key="bf_ld")
        with c2:
            rep_col = st.selectbox("Report Date", cols, key="bf_rd")
        with c3:
            lob_col = st.selectbox("LOB", cols, key="bf_lob")
        
        amount_candidates = [
            c for c in cols 
            if c not in [loss_col, rep_col, lob_col] 
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        amount_cols = st.multiselect("Amount Column(s)", amount_candidates, key="bf_amt")
        
        if not amount_cols:
            st.warning("👆 Please select at least one Amount column.")
            return
        
        df[loss_col] = pd.to_datetime(df[loss_col], errors='coerce')
        df[rep_col] = pd.to_datetime(df[rep_col], errors='coerce')
        for ac in amount_cols:
            df[ac] = pd.to_numeric(df[ac], errors='coerce').fillna(0)
        
        df = df.dropna(subset=[loss_col, rep_col])
        from_dt = pd.Timestamp(str(from_date))
        to_dt = pd.Timestamp(str(to_date))
        df = _date_filter(df, loss_col, from_date, to_date)
        
        lobs = sorted(df[lob_col].dropna().unique())
        
        st.markdown("### 🎯 Expected Loss Ratios (ELR) per LOB")
        elr_cols = st.columns(min(len(lobs), 4))
        elr_dict = {}
        for i, lob in enumerate(lobs):
            with elr_cols[i % 4]:
                elr_dict[lob] = st.number_input(
                    f"ELR {lob} (%)", 0.0, 200.0, 70.0, 1.0, key=f"bf_elr_{lob}"
                ) / 100.0
        
        grain = "Y"
        n_periods = to_dt.year - from_dt.year + 1
        
        # Show LDF selection
        st.markdown("### 📈 LDF Method")
        
        if st.button("🔢 Calculate BF IBNR", key="bf_run", use_container_width=True):
            if ibnr_bf is None or engine_utils is None:
                st.error("❌ Required engines not available.")
                return
            
            with st.spinner("Calculating BF IBNR..."):
                # Calculate all LDFs for display
                sample_amt = amount_cols[0]
                lob_data_sample = df[df[lob_col] == lobs[0]].copy() if lobs else df.copy()
                _, sample_cum, _ = engine_utils.build_triangles(
                    lob_data_sample, loss_col, rep_col, sample_amt, from_dt, grain, n_periods
                )
                
                all_ldfs = ibnr_bf.calculate_all_ldfs(sample_cum, n_periods)
                
                st.markdown("#### Development Factors (All Methods)")
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
                
                selected_method = st.selectbox(
                    "Select LDF Method",
                    ["volume_weighted", "simple_average", "geometric", 
                     "medial", "linear_regression", "weighted_last_3"],
                    key="bf_ldf_method"
                )
                
                all_results = []
                for lob in lobs:
                    lob_data = df[df[lob_col] == lob].copy()
                    prems = [1.0] * n_periods
                    
                    for ac in amount_cols:
                        _, cum, _ = engine_utils.build_triangles(
                            lob_data, loss_col, rep_col, ac, from_dt, grain, n_periods
                        )
                        
                        result = ibnr_bf.calculate_bf_ibnr(
                            cum_triangle=cum,
                            premiums=prems,
                            elr=elr_dict.get(lob, 0.7),
                            start_date=from_dt,
                            period_unit=grain,
                            selected_ldf_method=selected_method,
                            use_inflation=False,
                            cum_inflation=None,
                            per_period_rates=None,
                            use_discounting=False,
                            spot_rates=None,
                            flat_rate=None
                        )
                        res_df = result['results_df']
                        res_df['LOB'] = lob
                        res_df['Amount_Col'] = ac
                        all_results.append(res_df)
                
                final_df = pd.concat(all_results, ignore_index=True)
                summary = final_df.groupby(['LOB', 'Amount_Col'])[['Current_Claims', 'BF_IBNR']].sum().reset_index()
            
            st.markdown("### 📊 BF IBNR Summary")
            
            disp = summary.copy()
            for c in ['Current_Claims', 'BF_IBNR']:
                if c in disp.columns:
                    disp[c] = disp[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            
            total_ibnr = summary['BF_IBNR'].sum()
            st.metric("Total BF IBNR", f"{total_ibnr:,.2f}")
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as w:
                summary.to_excel(w, index=False, sheet_name='BF_Summary')
                final_df.to_excel(w, index=False, sheet_name='BF_Detail')
            output.seek(0)
            
            sc = sanitize_filename(client_name)
            st.download_button(
                "⬇ Download BF Results",
                data=output,
                file_name=f"{sc}_BF_IBNR.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="bf_dl"
            )
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        with st.expander("Show details"):
            import traceback
            st.code(traceback.format_exc())
    
    back_button('ibnr_menu', ['Home', 'LIC Calculators', 'Fulfilment Cashflows', 'IBNR Methods'])
