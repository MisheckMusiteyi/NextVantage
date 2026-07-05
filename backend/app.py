# backend/app.py
import os
import sys
from io import BytesIO
from datetime import date, datetime
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy import stats as scipy_stats

# ------------------------------------------------------------
#  Make sure we can import the engines from the parent folder
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)          # NextVantage root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import all engines (they are in PROJECT_ROOT / LRC_Calculators etc.)
from LRC_Calculators.upr_engine import calculate_upr
from LRC_Calculators.loss_component_engine import calculate_loss_component
from LIC_Calculators.FCF_Calculators.OCR_Calculators.ocr_engine import calculate_ocr
from LIC_Calculators.FCF_Calculators.IBNR_Calculators.percentage_ibnr import calculate_percentage_ibnr
from LIC_Calculators.FCF_Calculators.IBNR_Calculators.bcl_ibnr import calculate_bcl_ibnr
from LIC_Calculators.FCF_Calculators.IBNR_Calculators.cape_cod_ibnr import calculate_cape_cod_ibnr
from LIC_Calculators.FCF_Calculators.IBNR_Calculators.bf_ibnr import calculate_bf_ibnr
from LIC_Calculators.FCF_Calculators.ULAE_Calculators.ulae_engine import calculate_ulae_per_portfolio
from LIC_Calculators.FCF_Calculators.NPR_Calculators.npr_engine import calculate_npr_aggregation
from LIC_Calculators.RA_Calculators.mack_ra import calculate_mack_chain_ladder
from LIC_Calculators.RA_Calculators.bootstrap_ra import bootstrap_chain_ladder, calculate_risk_adjustment
from Full_Valuation.full_LRC_IFRS17 import calculate_full_ifrs17_lrc
from utils.actuarial_engine_utils import build_triangles, period_label

# ------------------------------------------------------------
app = Flask(__name__)
CORS(app)   # allow requests from any origin (frontend)

# ------------------------------------------------------------
#  Helper: read uploaded file into DataFrame
# ------------------------------------------------------------
def read_upload(file_key):
    """Read CSV or Excel from request.files, return (df, filename)."""
    f = request.files.get(file_key)
    if f is None:
        return None, None
    name = f.filename.lower()
    df = pd.read_csv(f) if name.endswith('.csv') else pd.read_excel(f)
    # clean up unnamed columns
    unnamed = [c for c in df.columns if str(c).startswith('Unnamed:')]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)
    df.columns = df.columns.astype(str).str.strip()
    return df, f.filename

# ------------------------------------------------------------
#  UPR
# ------------------------------------------------------------
@app.route('/api/upr', methods=['POST'])
def upr_endpoint():
    df, _ = read_upload('file')
    if df is None:
        return jsonify({'error': 'No file uploaded'}), 400

    # read parameters from form (or JSON – here we use form)
    start_date_col = request.form['start_date_col']
    end_date_col   = request.form['end_date_col']
    valuation_date = pd.to_datetime(request.form['valuation_date'])
    method         = request.form.get('method', '365th')
    grouping_cols  = request.form.getlist('grouping_cols')
    value_cols     = request.form.getlist('value_cols')

    try:
        result = calculate_upr(
            df, start_date_col, end_date_col,
            value_cols, grouping_cols,
            valuation_date, method
        )
        # result is a DataFrame; convert to list of dicts
        return jsonify(result.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ------------------------------------------------------------
#  Loss Component
# ------------------------------------------------------------
@app.route('/api/loss_component', methods=['POST'])
def loss_component_endpoint():
    df, _ = read_upload('file')
    if df is None:
        return jsonify({'error': 'No file uploaded'}), 400

    params = {k: request.form[k] for k in [
        'lob_col', 'written_premium_col', 'expenses_col', 'commission_col',
        'paid_claims_col', 'opening_ocr_col', 'closing_ocr_col',
        'opening_ibnr_col', 'closing_ibnr_col',
        'opening_upr_col', 'closing_upr_col', 'risk_adjustment_col'
    ]}
    try:
        result = calculate_loss_component(df, **params)
        return jsonify(result.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ------------------------------------------------------------
#  OCR
# ------------------------------------------------------------
@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    df, _ = read_upload('file')
    if df is None:
        return jsonify({'error': 'No file uploaded'}), 400

    grouping_cols = request.form.getlist('grouping_cols')
    value_cols    = request.form.getlist('value_cols')
    try:
        result = calculate_ocr(df, grouping_cols, value_cols)
        return jsonify(result.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ------------------------------------------------------------
#  Percentage IBNR
# ------------------------------------------------------------
@app.route('/api/percentage_ibnr', methods=['POST'])
def pct_ibnr_endpoint():
    df, _ = read_upload('file')
    if df is None:
        return jsonify({'error': 'No file uploaded'}), 400

    date_col    = request.form['date_col']
    lob_col     = request.form['lob_col']
    amount_cols = request.form.getlist('amount_cols')
    from_date   = pd.to_datetime(request.form['from_date'])
    to_date     = pd.to_datetime(request.form['to_date'])
    ibnr_pct    = float(request.form['ibnr_pct']) / 100.0

    try:
        summary, total = calculate_percentage_ibnr(
            df, date_col, lob_col, amount_cols,
            from_date, to_date, ibnr_pct
        )
        return jsonify({
            'summary': summary.to_dict(orient='records'),
            'total': total
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ------------------------------------------------------------
#  BCL IBNR
# ------------------------------------------------------------
@app.route('/api/bcl_ibnr', methods=['POST'])
def bcl_ibnr_endpoint():
    df, _ = read_upload('file')
    if df is None:
        return jsonify({'error': 'No file uploaded'}), 400

    loss_col   = request.form['loss_col']
    rep_col    = request.form['rep_col']
    lob_col    = request.form['lob_col']
    amount_cols= request.form.getlist('amount_cols')
    grain      = request.form.get('grain', 'Y')
    from_date  = pd.to_datetime(request.form['from_date'])
    to_date    = pd.to_datetime(request.form['to_date'])
    method     = request.form.get('selected_ldf_method', 'volume_weighted')
    use_inflation = request.form.get('use_inflation', 'false').lower() == 'true'
    use_disc   = request.form.get('use_discounting', 'false').lower() == 'true'

    # inflation / discounting data would be uploaded as separate files; simplified here
    n_periods = (to_date.year - from_date.year) + 1
    lobs = sorted(df[lob_col].dropna().unique())
    results = []
    for lob in lobs:
        lob_data = df[df[lob_col] == lob].copy()
        for ac in amount_cols:
            _, cum, _ = build_triangles(lob_data, loss_col, rep_col, ac, from_date, grain, n_periods)
            res = calculate_bcl_ibnr(cum, from_date, grain, selected_ldf_method=method)
            results.append(res)

    # flatten results
    flat = []
    for r in results:
        flat.extend(r['results_df'].to_dict(orient='records'))
    return jsonify(flat)

# ------------------------------------------------------------
#  Cape Cod IBNR
# ------------------------------------------------------------
@app.route('/api/capecod_ibnr', methods=['POST'])
def capecod_ibnr_endpoint():
    claims_df, _ = read_upload('claims_file')
    prem_df, _   = read_upload('prem_file')
    if claims_df is None or prem_df is None:
        return jsonify({'error': 'Both claims and premiums files required'}), 400

    loss_col = request.form['loss_col']
    rep_col  = request.form['rep_col']
    lob_col  = request.form['lob_col']
    amount_cols = request.form.getlist('amount_cols')
    from_date   = pd.to_datetime(request.form['from_date'])
    to_date     = pd.to_datetime(request.form['to_date'])
    prem_lob_col   = request.form['prem_lob_col']
    prem_amt_col   = request.form['prem_amt_col']
    prem_period_col= request.form['prem_period_col']
    method = request.form.get('selected_ldf_method', 'volume_weighted')
    grain = 'Y'; n_periods = to_date.year - from_date.year + 1
    lobs = sorted(claims_df[lob_col].dropna().unique())
    results = []
    for lob in lobs:
        lob_data = claims_df[claims_df[lob_col] == lob].copy()
        prem_sub = prem_df[prem_df[prem_lob_col].astype(str).str.lower() == lob.lower()].copy()
        prem_sub[prem_amt_col] = pd.to_numeric(prem_sub[prem_amt_col], errors='coerce').fillna(0)
        prem_sub[prem_period_col] = pd.to_numeric(prem_sub[prem_period_col], errors='coerce')
        prem_sub = prem_sub.sort_values(prem_period_col)
        prems = prem_sub[prem_amt_col].tolist()
        if len(prems) != n_periods:
            return jsonify({'error': f'LOB {lob}: premium periods count mismatch'}), 400
        for ac in amount_cols:
            _, cum, _ = build_triangles(lob_data, loss_col, rep_col, ac, from_date, grain, n_periods)
            res = calculate_cape_cod_ibnr(cum, prems, from_date, grain, selected_ldf_method=method)
            results.append(res)
    flat = []
    for r in results:
        flat.extend(r['results_df'].to_dict(orient='records'))
    return jsonify(flat)

# ------------------------------------------------------------
#  BF IBNR
# ------------------------------------------------------------
@app.route('/api/bf_ibnr', methods=['POST'])
def bf_ibnr_endpoint():
    claims_df, _ = read_upload('claims_file')
    prem_df, _   = read_upload('prem_file')   # optional – can be None
    if claims_df is None:
        return jsonify({'error': 'Claims file required'}), 400

    # same parameters as cape cod plus ELR dictionary
    loss_col = request.form['loss_col']
    rep_col  = request.form['rep_col']
    lob_col  = request.form['lob_col']
    amount_cols = request.form.getlist('amount_cols')
    from_date   = pd.to_datetime(request.form['from_date'])
    to_date     = pd.to_datetime(request.form['to_date'])
    method = request.form.get('selected_ldf_method', 'volume_weighted')
    grain = 'Y'; n_periods = to_date.year - from_date.year + 1
    lobs = sorted(claims_df[lob_col].dropna().unique())
    # ELR per LOB (passed as JSON string)
    elr_dict = request.form.get('elr_dict')
    if elr_dict:
        import json; elr_dict = json.loads(elr_dict)
    else:
        elr_dict = {lob: 0.7 for lob in lobs}

    results = []
    for lob in lobs:
        lob_data = claims_df[claims_df[lob_col] == lob].copy()
        prems = [1.0] * n_periods
        if prem_df is not None:
            prem_period_col = request.form.get('prem_period_col')
            prem_amt_col    = request.form.get('prem_amt_col')
            if prem_period_col and prem_amt_col:
                prem_sub = prem_df[prem_df[request.form['prem_lob_col']].astype(str).str.lower() == lob.lower()].copy()
                prem_sub[prem_amt_col] = pd.to_numeric(prem_sub[prem_amt_col], errors='coerce').fillna(0)
                prem_sub[prem_period_col] = pd.to_numeric(prem_sub[prem_period_col], errors='coerce')
                prem_sub = prem_sub.sort_values(prem_period_col)
                prems = prem_sub[prem_amt_col].tolist()
                if len(prems) != n_periods:
                    return jsonify({'error': f'LOB {lob}: premium periods count mismatch'}), 400
        for ac in amount_cols:
            _, cum, _ = build_triangles(lob_data, loss_col, rep_col, ac, from_date, grain, n_periods)
            res = calculate_bf_ibnr(cum, prems, elr_dict[lob], from_date, grain, selected_ldf_method=method)
            results.append(res)
    flat = []
    for r in results:
        flat.extend(r['results_df'].to_dict(orient='records'))
    return jsonify(flat)

# ------------------------------------------------------------
#  ULAE
# ------------------------------------------------------------
@app.route('/api/ulae', methods=['POST'])
def ulae_endpoint():
    df, _ = read_upload('file')
    if df is None:
        return jsonify({'error': 'No file uploaded'}), 400

    lob_col   = request.form['lob_col']
    ocr_col   = request.form['ocr_col']
    ibnr_col  = request.form['ibnr_col']
    ulae_ratio= float(request.form['ulae_ratio']) / 100.0
    basis     = request.form.get('basis', 'Per Portfolio')

    # prepare data
    df[ocr_col]  = pd.to_numeric(df[ocr_col], errors='coerce').fillna(0)
    df[ibnr_col] = pd.to_numeric(df[ibnr_col], errors='coerce').fillna(0)
    df['ULAE_Base'] = 0.5 * df[ocr_col] + df[ibnr_col]
    df['ULAE'] = df['ULAE_Base'] * ulae_ratio
    res = df[[lob_col, ocr_col, ibnr_col, 'ULAE_Base', 'ULAE']].to_dict(orient='records')
    return jsonify(res)

# ------------------------------------------------------------
#  NPR
# ------------------------------------------------------------
@app.route('/api/npr', methods=['POST'])
def npr_endpoint():
    ri_df, _  = read_upload('ri_file')
    lic_df, _ = read_upload('lic_file')
    if ri_df is None or lic_df is None:
        return jsonify({'error': 'Both reinsurer and LIC files required'}), 400

    name_col  = request.form['name_col']
    pd_col    = request.form['pd_col']
    share_col = request.form['share_col']
    port_col  = request.form['port_col']
    ibnr_col  = request.form['ibnr_col']
    ocr_col   = request.form['ocr_col']

    detail, by_port, by_ri, total = calculate_npr_aggregation(
        ri_df.rename(columns={name_col:'Reinsurer_Name', pd_col:'PD', share_col:'Overall_Share'}),
        lic_df.rename(columns={port_col:'Portfolio', ibnr_col:'Ceded_IBNR', ocr_col:'Ceded_OCR'})
    )
    return jsonify({
        'detail': detail.to_dict(orient='records'),
        'by_portfolio': by_port.to_dict(orient='records'),
        'by_reinsurer': by_ri.to_dict(orient='records'),
        'total': total
    })

# ------------------------------------------------------------
#  Mack RA
# ------------------------------------------------------------
@app.route('/api/mack', methods=['POST'])
def mack_endpoint():
    # expects either a cumulative triangle JSON or claims file
    if 'triangle' in request.files:
        df, _ = read_upload('triangle')
        triangle = df.apply(pd.to_numeric, errors='coerce').values
    else:
        df, _ = read_upload('file')
        loss_col = request.form['loss_col']
        rep_col  = request.form['rep_col']
        amount_col = request.form['amount_col']
        grain = request.form.get('grain', 'Y')
        from_date = pd.to_datetime(request.form['from_date'])
        to_date   = pd.to_datetime(request.form['to_date'])
        n_periods = (to_date.year - from_date.year) + 1
        _, cum, _ = build_triangles(df, loss_col, rep_col, amount_col, from_date, grain, n_periods)
        triangle = cum

    confidence = float(request.form.get('confidence', 0.75))
    result = calculate_mack_chain_ladder(triangle, confidence)
    return jsonify(result['results_df'].to_dict(orient='records'))

# ------------------------------------------------------------
#  Bootstrap RA
# ------------------------------------------------------------
@app.route('/api/bootstrap', methods=['POST'])
def bootstrap_endpoint():
    # similar to Mack – accepts triangle or claims data
    if 'triangle' in request.files:
        df, _ = read_upload('triangle')
        triangle = df.apply(pd.to_numeric, errors='coerce')
    else:
        df, _ = read_upload('file')
        # build triangle as above
        # ... (omitted for brevity, similar to Mack)
        pass

    confidence = float(request.form.get('confidence', 0.75))
    n_iter = int(request.form.get('n_iter', 1000))
    add_pv = request.form.get('add_pv', 'true').lower() == 'true'

    boot_result = bootstrap_chain_ladder(triangle, n_iterations=n_iter, add_process_variance=add_pv)
    ra_result = calculate_risk_adjustment(boot_result, confidence, 'cl')
    return jsonify({
        'ra': ra_result,
        'samples': boot_result['ibnr_nominal_samples'].tolist()
    })

# ------------------------------------------------------------
#  Full IFRS 17 Valuation
# ------------------------------------------------------------
@app.route('/api/full_valuation', methods=['POST'])
def full_valuation_endpoint():
    # requires multiple files and a config JSON
    opening_df, _   = read_upload('opening_balances')
    cashflows_df, _ = read_upload('cashflows')
    policy_df, _    = read_upload('policy')
    loss_comp_df, _ = read_upload('loss_component')
    yc_df, _        = read_upload('yield_curve') if 'yield_curve' in request.files else (None, None)
    cc_df, _        = read_upload('claims_curve') if 'claims_curve' in request.files else (None, None)

    config = request.form.get('config')
    if config:
        import json; config = json.loads(config)
    else:
        config = {
            'iacf_toggle': 'Expense Immediately',
            'discount_toggle': 'No Discounting',
            'invest_toggle': 'No',
            'revenue_toggle': 'Passage of Time'
        }
    valuation_date = pd.to_datetime(request.form.get('valuation_date', '2025-12-31'))

    try:
        results = calculate_full_ifrs17_lrc(
            opening_df, cashflows_df, policy_df, loss_comp_df,
            yc_df, cc_df, config, valuation_date
        )
        # convert results dict of DataFrames to JSON
        out = {}
        for group, data in results.items():
            out[group] = {k: v for k, v in data.items() if not isinstance(v, pd.DataFrame)}
        return jsonify(out)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
