# -*- coding: utf-8 -*-
# =============================================================================
#  CAPE COD IBNR ENGINE
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
            for k in ldfs.keys():
                ldfs[k].append(1.0)
            continue

        num = sum(w * r for r, w in zip(ratios, weights))
        den = sum(weights)
        ldfs["volume_weighted"].append(num / den if den > 0 else 1.0)
        ldfs["simple_average"].append(np.mean(ratios))
        geo_mean = np.exp(np.mean(np.log(ratios))) if all(r > 0 for r in ratios) else 1.0
        ldfs["geometric"].append(geo_mean)
        sorted_ratios = np.sort(ratios)
        trim = int(len(sorted_ratios) * 0.2)
        trimmed = sorted_ratios[trim:-trim] if trim > 0 else sorted_ratios
        ldfs["medial"].append(np.mean(trimmed) if len(trimmed) > 0 else 1.0)
        x = np.arange(len(ratios))
        y = np.array(ratios)
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            ldfs["linear_regression"].append(max(1.0, intercept + slope * (len(ratios) - 1)))
        else:
            ldfs["linear_regression"].append(ratios[0] if ratios else 1.0)
        if len(ratios) >= 3:
            w3 = [1, 2, 3]
            ldfs["weighted_last_3"].append(sum(w * r for w, r in zip(w3, ratios[-3:])) / sum(w3))
        elif ratios:
            ldfs["weighted_last_3"].append(np.mean(ratios))
        else:
            ldfs["weighted_last_3"].append(1.0)

    return ldfs


def calculate_cape_cod_ibnr(cum_triangle, premiums, start_date, period_unit,
                            selected_ldf_method="volume_weighted",
                            use_inflation=False, cum_inflation=None, per_period_rates=None,
                            use_discounting=False, spot_rates=None, flat_rate=None):
    """
    Calculate Cape Cod IBNR with multi-LDF, inflation, and discounting support.
    Tail factor = 1.000 (fully developed).
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

    # 2. Calculate LDFs and CDFs
    factors = calculate_all_ldfs(working_cum, n_dp)[selected_ldf_method]
    completed_real = project_ultimate(working_cum, factors)

    cdfs = []
    run = 1.0
    for f in reversed(factors):
        run *= f
        cdfs.insert(0, run)
    pct_developed = [1 / cdf if cdf > 0 else 1.0 for cdf in cdfs]

    # 3. Cape Cod ELR calculation
    developed_claims = []
    used_up_premiums_list = []

    for i in range(n_ap):
        last_obs = -1
        for j in range(n_dp - 1, -1, -1):
            if i + j < n_ap and pd.notna(working_cum.iloc[i, j]):
                last_obs = j
                break
        if last_obs == -1:
            developed_claims.append(0)
            used_up_premiums_list.append(0)
            continue

        current = working_cum.iloc[i, last_obs]
        pct_dev = pct_developed[last_obs] if last_obs < len(pct_developed) else 1.0
        developed_claims.append(current)
        used_up_premiums_list.append(working_premiums[i] * pct_dev)

    total_developed = sum(developed_claims)
    total_used_up = sum(used_up_premiums_list)
    cape_cod_lr = total_developed / total_used_up if total_used_up > 0 else 0

    # 4. Calculate IBNR per accident period
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
        pct_dev = pct_developed[last_obs] if last_obs < len(pct_developed) else 1.0
        pct_unpaid = 1 - pct_dev
        expected_ultimate = working_premiums[i] * cape_cod_lr
        cc_ibnr_real = expected_ultimate * pct_unpaid
        total_ibnr_real += cc_ibnr_real

        rows.append({
            'Accident_Period': i,
            'Accident_Period_Label': period_label(i, start_date, period_unit),
            'Current_Claims': current,
            'Premium': working_premiums[i],
            'Used_Up_Premium': used_up_premiums_list[i],
            'Pct_Developed': pct_dev * 100,
            'Cape_Cod_LR': cape_cod_lr,
            'Expected_Ultimate': expected_ultimate,
            'Cape_Cod_IBNR_Real': cc_ibnr_real,
        })

    # 5. Re-inflate Real IBNR to Nominal
    if use_inflation and per_period_rates is not None:
        nominal_map = reinflate_ibnr_per_ap(completed_real, working_cum, n_ap, per_period_rates)
        for i, row in enumerate(rows):
            row['Cape_Cod_IBNR'] = nominal_map.get(i, row['Cape_Cod_IBNR_Real'])
    else:
        for i, row in enumerate(rows):
            row['Cape_Cod_IBNR'] = row['Cape_Cod_IBNR_Real']

    # 6. Discount nominal IBNR if required
    if use_discounting:
        _, total_ibnr_disc = discount_completed_triangle(
            completed_real, working_cum, n_ap, period_unit, spot_rates, flat_rate
        )
    else:
        total_ibnr_disc = None

    return {
        'results_df': pd.DataFrame(rows),
        'total_ibnr': sum(r['Cape_Cod_IBNR'] for r in rows),
        'total_ibnr_discounted': total_ibnr_disc,
        'cape_cod_lr': cape_cod_lr,
        'total_developed': total_developed,
        'total_used_up': total_used_up,
        'all_ldfs': calculate_all_ldfs(cum_triangle, n_dp)
    }
