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
    
    if mode == "Full Valuation (All Files)":
        c1, c2 = st.columns(2)
        with c1:
            opening_file = st.file_uploader("1. Opening Balances (Group, Opening_LRC_Excl_Loss, Opening_Loss_Component)", type=["csv","xlsx"], key="fv_ob")
            policy_file = st.file_uploader("3. Policy Data (Group, Start_Date, End_Date, Written_Premium)", type=["csv","xlsx"], key="fv_pol")
        with c2:
            cashflows_file = st.file_uploader("2. Cashflows (Group, Premiums_Received, IACF_Paid, Investment_Components_Paid)", type=["csv","xlsx"], key="fv_cf")
            claims_curve_file = st.file_uploader("6. Claims Curve (Period, Percentage) – Optional", type=["csv","xlsx"], key="fv_cc")
        
        # Discounting data – only when toggled on
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
                lc_raw = pd.read_csv(lc_data_file) if lc_data_file.name.endswith('.csv') else pd.read_excel(lc_data_file)
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
        
        # Required files: opening, cashflows, policy (loss component will be derived)
        required = [opening_file, cashflows_file, policy_file, lc_data_file]
        if all(f is not None for f in required):
            try:
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
                efp_df = pd.read_csv(efp_file) if efp_file.name.endswith('.csv') else pd.read_excel(efp_file)
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
                    cc_df = pd.read_csv(claims_curve_file) if claims_curve_file.name.endswith('.csv') else pd.read_excel(claims_curve_file)
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
    
    else:  # Simple PAA mode (unchanged)
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
                if st.button("Run Simple PAA Valuation", key="fv_run_simple", use_container_width=True):
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
