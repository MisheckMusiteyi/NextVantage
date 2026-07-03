# -*- coding: utf-8 -*-
# =============================================================================
#  BASIC CHAIN LADDER (BCL) IBNR ENGINE
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
            for k in ldfs.keys(): ldfs[k].append(1.0)
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

        # Medial (Trim 20%)
        sorted_ratios = np.sort(ratios)
        trim = int(len(sorted_ratios) * 0.2)
        trimmed = sorted_ratios[trim:-trim] if trim > 0 else sorted_ratios
        ldfs["medial"].append(np.mean(trimmed) if len(trimmed) > 0 else 1.0)

        # Linear Regression (Intercept at latest)
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

def calculate_bcl_ibnr(cum_triangle, start_date, period_unit, 
                       selected_ldf_method="volume_weighted",
                       use_inflation=False, cum_inflation=None, per_period_rates=None,
                       use_discounting=False, spot_rates=None, flat_rate=None):
    n_ap, n_dp = cum_triangle.shape

    # 1. Calculate all LDFs on the nominal triangle (for UI display)
    all_ldfs = calculate_all_ldfs(cum_triangle, n_dp)

    # 2. Get real working triangle if inflation is on
    if use_inflation and cum_inflation is not None:
        inc = cum_triangle.diff(axis=1).fillna(cum_triangle.iloc[:, 0]).fillna(0)
        _, working_cum = deflate_triangle_to_real(inc, cum_inflation, n_ap)
    else:
        working_cum = cum_triangle.copy()

    # 3. Calculate LDFs from the REAL triangle (if inflated), or nominal if not.
    if selected_ldf_method == "volume_weighted": factors = calculate_all_ldfs(working_cum, n_dp)["volume_weighted"]
    elif selected_ldf_method == "simple_average": factors = calculate_all_ldfs(working_cum, n_dp)["simple_average"]
    elif selected_ldf_method == "geometric": factors = calculate_all_ldfs(working_cum, n_dp)["geometric"]
    elif selected_ldf_method == "medial": factors = calculate_all_ldfs(working_cum, n_dp)["medial"]
    elif selected_ldf_method == "linear_regression": factors = calculate_all_ldfs(working_cum, n_dp)["linear_regression"]
    elif selected_ldf_method == "weighted_last_3": factors = calculate_all_ldfs(working_cum, n_dp)["weighted_last_3"]
    else: factors = calculate_all_ldfs(working_cum, n_dp)["volume_weighted"]

    # 4. Project REAL ultimate
    completed_real = project_ultimate(working_cum, factors)

    # 5. Re-inflate the projected payments (to nominal) if required
    if use_inflation and per_period_rates is not None:
        nominal_map = reinflate_ibnr_per_ap(completed_real, working_cum, n_ap, per_period_rates)
        # Build nominal completed triangle for discounting
        completed_nominal = completed_real.copy()
        for ap in range(n_ap):
            for dp in range(n_dp):
                if ap + dp >= n_ap: # Future periods
                    if nominal_map.get(ap) is not None:
                        completed_nominal.iloc[ap, dp] = nominal_map[ap] + working_cum.iloc[ap, n_dp-1] # Placeholder for distribution
    else:
        nominal_map = None
        completed_nominal = completed_real

    # 6. Discount nominal IBNR if required
    if use_discounting:
        _, total_ibnr_disc = discount_completed_triangle(completed_nominal, cum_triangle, n_ap, period_unit, spot_rates, flat_rate)
    else:
        total_ibnr_disc = None

    # 7. Build Results
    rows = []
    total_ibnr = 0.0
    for i in range(n_ap):
        last_obs = -1
        for j in range(n_dp - 1, -1, -1):
            if i + j < n_ap and pd.notna(cum_triangle.iloc[i, j]):
                last_obs = j
                break
        if last_obs < 0: continue

        current = cum_triangle.iloc[i, last_obs]
        ultimate = completed_real.iloc[i, n_dp - 1]
        if nominal_map is not None and i in nominal_map:
            ibnr = nominal_map[i]
        else:
            ibnr = max(ultimate - current, 0.0)
        
        total_ibnr += ibnr
        rows.append({
            'Accident_Period': i,
            'Accident_Period_Label': period_label(i, start_date, period_unit),
            'Current_Claims': current,
            'Ultimate_Claims': ultimate,
            'IBNR': ibnr,
            'Selected_LDF_Method': selected_ldf_method
        })

    return {
        'results_df': pd.DataFrame(rows),
        'total_ibnr': total_ibnr,
        'total_ibnr_discounted': total_ibnr_disc,
        'dev_factors': factors,
        'all_ldfs': all_ldfs
    }
