# -*- coding: utf-8 -*-
# =============================================================================
#  MACK CHAIN LADDER RISK ADJUSTMENT ENGINE  (CORRECTED)
#  Mack (1993) Standard Error Methodology
#  Tail factor hardcoded to 1.000 (fully developed)
#
#  FIXES APPLIED vs original:
#   [FIX-1] Tail sigma^2 extrapolation: corrected exponents (values in `sigma2`
#           are already sigma^2, not sigma) and corrected the storage index so
#           the extrapolated value is actually the one read by the variance
#           loop (index n_dev-2, not the unused n_dev-1).
#   [FIX-2] Total reserve variance now includes Mack's cross-accident-year
#           covariance term, matching R's ChainLadder::MackChainLadder
#           "Total Mack S.E." (previously total_variance was just a naive
#           sum of the per-accident-year variances).
#   [FIX-3] Discounting-with-inflation stub ("pass") replaced with a
#           proportional reallocation of nominal IBNR over future periods.
# =============================================================================

import pandas as pd
import numpy as np
from scipy import stats
from utils.actuarial_engine_utils import (
    period_label, period_index, build_triangles, volume_weighted_factors,
    project_ultimate, deflate_triangle_to_real, reinflate_ibnr_per_ap,
    discount_completed_triangle
)

def _build_nominal_completed_triangle(completed_real, cum_real, n_periods, per_period_rates):
    """
    Reconstructs a full nominal cumulative triangle from a real-terms
    completed triangle, using the EXACT same cell-by-cell forward-inflation
    logic as utils.reinflate_ibnr_per_ap (rather than approximating with a
    single row-level scale factor). This makes the nominal reserve fed into
    discount_completed_triangle reconcile exactly with the nominal IBNR
    reported elsewhere in this engine.
    """
    valuation_idx = n_periods - 1
    last_rate = per_period_rates[-1] if len(per_period_rates) > 0 else 0.0

    def forward_inflation(t_future):
        factor = 1.0
        for k in range(valuation_idx + 1, t_future + 1):
            ki = k - valuation_idx - 1
            r = per_period_rates[ki] if ki < len(per_period_rates) else last_rate
            factor *= (1.0 + r)
        return factor

    dp_cols = sorted(completed_real.columns)
    nominal_completed = completed_real.copy().astype(float)
    for ap in completed_real.index:
        last_obs = -1
        for dp in sorted(cum_real.columns, reverse=True):
            if ap + dp < n_periods and pd.notna(cum_real.loc[ap, dp]):
                last_obs = dp
                break
        if last_obs < 0:
            continue
        running_nominal = float(cum_real.loc[ap, last_obs]) if pd.notna(cum_real.loc[ap, last_obs]) else 0.0
        nominal_completed.loc[ap, last_obs] = running_nominal
        for idx_dp, dp in enumerate(dp_cols):
            if dp <= last_obs:
                continue
            cum_curr = completed_real.loc[ap, dp]
            if pd.isna(cum_curr):
                continue
            cum_prev = completed_real.loc[ap, dp_cols[idx_dp - 1]] if idx_dp > 0 else 0.0
            inc_real = max(float(cum_curr) - float(cum_prev if pd.notna(cum_prev) else 0.0), 0.0)
            running_nominal += inc_real * forward_inflation(ap + dp)
            nominal_completed.loc[ap, dp] = running_nominal
    return nominal_completed


def calculate_mack_chain_ladder(cum_triangle, obs_mask, confidence_level=0.995,
                                use_inflation=False, cum_inflation=None, per_period_rates=None,
                                use_discounting=False, spot_rates=None, flat_rate=None,
                                grain='Y'):
    n_ay, n_dev = cum_triangle.shape

    # 1. Deflate if required
    if use_inflation and cum_inflation is not None:
        inc = cum_triangle.diff(axis=1).fillna(cum_triangle.iloc[:, 0]).fillna(0)
        _, working_cum = deflate_triangle_to_real(inc, cum_inflation, n_ay)
    else:
        working_cum = cum_triangle.copy()

    # 2. Calc Development Factors & Column Sums
    dev_factors = volume_weighted_factors(working_cum, obs_mask)
    S = {}
    for j in range(n_dev):
        col_sum = 0
        for i in range(n_ay):
            if obs_mask.iloc[i, j] and pd.notna(working_cum.iloc[i, j]):
                col_sum += working_cum.iloc[i, j]
        S[j] = col_sum

    # 3. Calculate Sigma^2 (Process Variance) per development period
    sigma2 = {}
    for j in range(n_dev - 1):
        factors_list = []
        weights = []
        for i in range(n_ay):
            if obs_mask.iloc[i, j] and obs_mask.iloc[i, j + 1] and working_cum.iloc[i, j] > 0:
                factor = working_cum.iloc[i, j + 1] / working_cum.iloc[i, j]
                factors_list.append(factor)
                weights.append(working_cum.iloc[i, j])
        if len(factors_list) > 1 and j < len(dev_factors):
            f_j = dev_factors[j]
            weighted_sq_sum = sum(w * (f - f_j) ** 2 for f, w in zip(factors_list, weights))
            sigma2[j] = weighted_sq_sum / (len(factors_list) - 1)
        else:
            sigma2[j] = 0

    # [FIX-1] Tail sigma^2 extrapolation (Mack 1993 recommended rule):
    #   sigma2_tail = min( sigma2_{k-1}^2 / sigma2_{k-2},  sigma2_{k-1},  sigma2_{k-2} )
    # `sigma2[j]` values are already variance-scale (sigma^2), so the
    # extrapolation formula must NOT re-square them. The tail index that
    # actually needs filling is n_dev-2 (the last development-factor slot,
    # which the variance loop below iterates up to), not n_dev-1.
    non_zero_sigma = [(j, val) for j, val in sigma2.items() if val > 0]
    tail_idx = n_dev - 2
    if tail_idx >= 0 and sigma2.get(tail_idx, 0) == 0:
        if len(non_zero_sigma) >= 2:
            (last_j, last_val), (second_j, second_val) = non_zero_sigma[-1], non_zero_sigma[-2]
            option1 = (last_val ** 2) / second_val if second_val > 0 else 0.0
            option2 = last_val
            option3 = second_val
            sigma2[tail_idx] = min(option1, option2, option3)
        elif len(non_zero_sigma) == 1:
            sigma2[tail_idx] = non_zero_sigma[-1][1]
        # else: insufficient data anywhere in the triangle to estimate a
        # tail sigma^2 -- leave at 0 rather than fabricate a number.

    # 4. Project Ultimate Triangle
    completed = project_ultimate(working_cum, dev_factors, obs_mask)

    # 5. Re-inflate nominal IBNR
    if use_inflation and per_period_rates is not None:
        nominal_map = reinflate_ibnr_per_ap(completed, working_cum, n_ay, per_period_rates)
    else:
        nominal_map = None

    # 6. Calculate IBNR and Mack Variance per Accident Period
    results = []
    total_ibnr = 0
    total_variance = 0.0
    ultimates = {}
    last_obs_map = {}
    # cross_sum_by_i[i] = sum_{k=last_obs(i)}^{n_dev-2} 2*sigma2_k / (f_k^2 * S_k)
    # -- the building block for Mack's inter-accident-year covariance term.
    cross_sum_by_i = {}

    for i in range(n_ay):
        last_obs = -1
        for j in range(n_dev - 1, -1, -1):
            if obs_mask.iloc[i, j] and pd.notna(working_cum.iloc[i, j]):
                last_obs = j
                break
        last_obs_map[i] = last_obs

        ultimate = completed.iloc[i, n_dev - 1]
        current = working_cum.iloc[i, last_obs] if last_obs >= 0 else 0
        ultimates[i] = ultimate

        if nominal_map is not None and i in nominal_map:
            ibnr = nominal_map[i]
        else:
            ibnr = ultimate - current

        # Variance loop uses the COMPLETED (projected) triangle for future
        # terms, per Mack's closed-form single-accident-year MSEP formula.
        variance = 0.0
        cross_sum = 0.0
        if last_obs < n_dev - 1 and last_obs >= 0 and ibnr > 0:
            total_sum = 0.0
            for k in range(last_obs, n_dev - 1):
                C_ik = completed.iloc[i, k]
                if C_ik > 0 and k in sigma2 and sigma2[k] > 0 and k < len(dev_factors):
                    f_k = dev_factors[k]
                    S_k = S.get(k, 0)
                    term_a = sigma2[k] / (f_k ** 2 * C_ik)
                    term_b = sigma2[k] / (f_k ** 2 * S_k) if S_k > 0 else 0.0
                    total_sum += term_a + term_b
                    if S_k > 0:
                        cross_sum += 2 * sigma2[k] / (f_k ** 2 * S_k)
            variance = (ultimate ** 2) * total_sum

        total_ibnr += ibnr
        total_variance += variance
        cross_sum_by_i[i] = cross_sum

        results.append({
            'accident_period': i,
            'accident_period_label': period_label(i, 0, 'Y'),
            'current': current,
            'ultimate': ultimate,
            'ibnr': ibnr,
            'variance': variance,
            'std_error': np.sqrt(variance) if variance > 0 else 0
        })

    # [FIX-2] Mack's cross-accident-year covariance term.
    # msep(sum R_i) = sum_i msep(R_i)
    #               + sum_i  C_i,ult * (sum_{j>i} C_j,ult) * cross_sum_by_i[i]
    # This captures the correlation between accident years induced by using
    # the same estimated development factors for all of them. Without it,
    # the total standard error is understated and will NOT match R's
    # ChainLadder::MackChainLadder "Total Mack S.E.".
    ay_indices = sorted(cross_sum_by_i.keys())
    for pos, i in enumerate(ay_indices):
        if cross_sum_by_i[i] == 0:
            continue
        later_ultimate_sum = sum(ultimates[j] for j in ay_indices[pos + 1:])
        if later_ultimate_sum > 0:
            total_variance += ultimates[i] * later_ultimate_sum * cross_sum_by_i[i]

    results_df = pd.DataFrame(results)

    # 7. Discounting (nominal reinflation properly threaded through)
    if use_discounting:
        if use_inflation and per_period_rates is not None:
            # [FIX-3] Previously a no-op ("pass"). Reconstruct the nominal
            # cumulative triangle cell-by-cell using the same forward-
            # inflation logic as reinflate_ibnr_per_ap, so the discounted
            # reserve reconciles exactly with the nominal IBNR reported
            # elsewhere (rather than approximating with a row-level scale
            # factor).
            completed_nominal = _build_nominal_completed_triangle(
                completed, working_cum, n_ay, per_period_rates
            )
            _, total_ibnr_disc = discount_completed_triangle(
                completed_nominal, working_cum, n_ay, grain, spot_rates, flat_rate
            )
        else:
            _, total_ibnr_disc = discount_completed_triangle(
                completed, working_cum, n_ay, grain, spot_rates, flat_rate
            )
    else:
        total_ibnr_disc = None

    # 8. Risk Adjustment
    total_se = np.sqrt(total_variance)
    z_score = stats.norm.ppf(confidence_level)
    risk_adjustment = total_se * z_score
    lic = total_ibnr + risk_adjustment

    return {
        'results_df': results_df,
        'total_ibnr': total_ibnr,
        'total_ibnr_discounted': total_ibnr_disc,
        'total_se': total_se,
        'risk_adjustment': risk_adjustment,
        'lic': lic,
        'dev_factors': dev_factors
    }
