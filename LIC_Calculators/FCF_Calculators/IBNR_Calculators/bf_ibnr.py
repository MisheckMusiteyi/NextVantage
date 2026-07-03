# -*- coding: utf-8 -*-
# =============================================================================
#  BORNHUETTER-FERGUSON IBNR ENGINE
#  Multi-LDF, Inflation, Discounting Support
#  Tail factor hardcoded to 1.000 (fully developed)
# =============================================================================

import pandas as pd
import numpy as np
from utils.actuarial_engine_utils import (
    period_label, project_ultimate, deflate_triangle_to_real,
    reinflate_ibnr_per_ap, discount_completed_triangle
)

def calculate_all_ldfs(cum, n_dp):
    """Calculate all 6 LDF methods for comparison."""
    ldfs = {
        "volume_weighted": [], "simple_average": [], "geometric": [],
        "medial": [], "linear_regression": [], "weighted_last_3": []
    }

    for j in range(n_dp - 1):
        ratios = []
        weights = []
        for i in range(cum.shape[0]):
            if i + j + 1 < cum.shape[0]:
                c = cum.iloc[i, j]
                n = cum.iloc[i, j + 1]
                if pd.notna(c) and pd.notna(n) and c > 0:
                    ratios.append(n / c)
                    weights.append(c)

        if not ratios:
            for k in ldfs.keys():
                ldfs[k].append(1.0)
            continue

        # Volume-Weighted
        num = sum(w * r for r, w in zip(ratios, weights))
        den = sum(weights)
        ldfs["volume_weighted"].append(num / den if den > 0 else 1.0)

        # Simple Average
        ldfs["simple_average"].append(np.mean(ratios))

        # Geometric
        geo_mean = np.exp(np.mean(np.log(ratios))) if all(r > 0 for r in ratios) else 1.0
        ldfs["geometric"].append(geo_mean)

        # Medial (trim 20%)
        sorted_ratios = np.sort(ratios)
        trim = int(len(sorted_ratios) * 0.2)
        trimmed = sorted_ratios[trim:-trim] if trim > 0 else sorted_ratios
        ldfs["medial"].append(np.mean(trimmed) if len(trimmed) > 0 else 1.0)

        # Linear Regression (extrapolated to latest)
        x = np.arange(len(ratios))
        y = np.array(ratios)
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            ldfs["linear_regression"].append(max(1.0, intercept + slope * (len(ratios) - 1)))
        else:
            ldfs["linear_regression"].append(ratios[0] if ratios else 1.0)

        # Weighted Last 3
        if len(ratios) >= 3:
            w3 = [1, 2, 3]
            ldfs["weighted_last_3"].append(sum(w * r for w, r in zip(w3, ratios[-3:])) / sum(w3))
        elif ratios:
            ldfs["weighted_last_3"].append(np.mean(ratios))
        else:
            ldfs["weighted_last_3"].append(1.0)

    return ldfs


def calculate_bf_ibnr(cum_triangle, premiums, elr, start_date, period_unit,
                     selected_ldf_method="volume_weighted",
                     use_inflation=False, cum_inflation=None, per_period_rates=None,
                     use_discounting=False, spot_rates=None, flat_rate=None):
    """
    Calculate BF IBNR with multi-LDF, inflation, and discounting support.
    
    Parameters:
    -----------
    cum_triangle : pd.DataFrame
        Cumulative claims triangle (rows=accident periods, cols=development periods)
    premiums : list
        Premium for each accident period
    elr : float
        Expected Loss Ratio (e.g., 0.70 for 70%)
    start_date : pd.Timestamp
        Start date of first accident period
    period_unit : str
        'Y', 'Q', or 'M'
    selected_ldf_method : str
        One of: volume_weighted, simple_average, geometric, medial, 
        linear_regression, weighted_last_3
    use_inflation : bool
        Whether to apply inflation adjustment
    cum_inflation : np.array or None
        Cumulative inflation indices
    per_period_rates : np.array or None
        Per-period inflation rates for re-inflation
    use_discounting : bool
        Whether to apply discounting
    spot_rates : np.array or None
        Spot rates for discounting
    flat_rate : float or None
        Flat discount rate
    
    Returns:
    --------
    dict with:
        - results_df: pd.DataFrame with IBNR per accident period
        - total_ibnr: float
        - total_ibnr_discounted: float or None
        - all_ldfs: dict of all LDF methods
        - dev_factors: list of selected LDFs
        - cdfs: list of cumulative development factors
        - pct_unpaid: list of percentage unpaid per development period
    """
    n_ap, n_dp = cum_triangle.shape

    # 1. Get real working triangle if inflation is on
    if use_inflation and cum_inflation is not None:
        inc = cum_triangle.diff(axis=1).fillna(cum_triangle.iloc[:, 0]).fillna(0)
        _, working_cum = deflate_triangle_to_real(inc, cum_inflation, n_ap)
        deflation_factor = cum_inflation[n_ap - 1] / cum_inflation[0] if cum_inflation[0] > 0 else 1.0
        working_premiums = [p / deflation_factor for p in premiums]
    else:
        working_cum = cum_triangle.copy()
        working_premiums = premiums

    # 2. Calculate LDFs using selected method
    all_ldfs = calculate_all_ldfs(working_cum, n_dp)
    factors = all_ldfs[selected_ldf_method]
    completed_real = project_ultimate(working_cum, factors)

    # 3. Compute CDFs and percentage unpaid
    cdfs = []
    run = 1.0
    for f in reversed(factors):
        run *= f
        cdfs.insert(0, run)
    pct_unpaid = [1 - (1 / cdf) if cdf > 0 else 0 for cdf in cdfs]

    # 4. Calculate BF IBNR on real basis
    rows = []
    total_ibnr_real = 0.0
    for i in range(n_ap):
        last_obs = -1
        for j in range(n_dp - 1, -1, -1):
            if i + j < n_ap and pd.notna(working_cum.iloc[i, j]):
                last_obs = j
                break
        if last_obs == -1:
            continue

        current = working_cum.iloc[i, last_obs]
        expected_ultimate_real = working_premiums[i] * elr

        if last_obs < len(pct_unpaid):
            pct_unpaid_val = pct_unpaid[last_obs]
            cdf_val = cdfs[last_obs] if last_obs < len(cdfs) else 1.0
            bf_ibnr_real = expected_ultimate_real * pct_unpaid_val
        else:
            pct_unpaid_val = 0
            cdf_val = 1.0
            bf_ibnr_real = 0

        total_ibnr_real += bf_ibnr_real
        rows.append({
            'Accident_Period': i,
            'Accident_Period_Label': period_label(i, start_date, period_unit),
            'Current_Claims': current,
            'Premium': working_premiums[i],
            'ELR': elr,
            'CDF_to_Ultimate': cdf_val,
            'Pct_Unpaid': pct_unpaid_val,
            'Expected_Ultimate': expected_ultimate_real,
            'BF_IBNR_Real': bf_ibnr_real,
            'Selected_LDF_Method': selected_ldf_method
        })

    # 5. Re-inflate Real IBNR to Nominal
    if use_inflation and per_period_rates is not None:
        nominal_map = reinflate_ibnr_per_ap(completed_real, working_cum, n_ap, per_period_rates)
        for i, row in enumerate(rows):
            row['BF_IBNR'] = nominal_map.get(i, row['BF_IBNR_Real'])
    else:
        for i, row in enumerate(rows):
            row['BF_IBNR'] = row['BF_IBNR_Real']

    # 6. Discount nominal IBNR if required
    if use_discounting:
        _, total_ibnr_disc = discount_completed_triangle(
            completed_real, working_cum, n_ap, period_unit, spot_rates, flat_rate
        )
    else:
        total_ibnr_disc = None

    return {
        'results_df': pd.DataFrame(rows),
        'total_ibnr': sum(r['BF_IBNR'] for r in rows),
        'total_ibnr_discounted': total_ibnr_disc,
        'all_ldfs': calculate_all_ldfs(cum_triangle, n_dp),
        'dev_factors': factors,
        'cdfs': cdfs,
        'pct_unpaid': pct_unpaid
    }
