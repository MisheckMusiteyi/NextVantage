# -*- coding: utf-8 -*-
# =============================================================================
#  BOOTSTRAP RISK ADJUSTMENT ENGINE  (CORRECTED)
#  England & Verrall (2002) ODP Bootstrap Methodology
#  Tail factor hardcoded to 1.000 (fully developed)
#
#  FIXES APPLIED vs original:
#   [FIX-1] (CRITICAL) The "fitted" incremental triangle used to compute
#           Pearson residuals was built from project_ultimate's forward
#           completion, which leaves the OBSERVED region equal to the
#           ACTUAL data. Since residuals are only computed over the
#           observed region, fitted == actual there and residuals collapse
#           to ~0, phi ~ 0, and the whole bootstrap distribution degenerates
#           to a point mass. Fixed by reconstructing the proper
#           England & Verrall "retrospective" fitted cumulative triangle:
#           for each row, divide the projected ultimate by the
#           cumulative-development-factor-to-ultimate at each development
#           period, working backwards from the latest diagonal. This is
#           exactly what R's ChainLadder::BootChainLadder does, and it
#           produces genuinely non-zero fitted-vs-actual residuals.
#   [FIX-2] Degrees of freedom for phi only subtracted development-period
#           parameters (n_dp - 1). The standard ODP GLM parameterization
#           has n_ay + n_dp - 1 free parameters (accident effects +
#           development effects, minus one redundant constant). Fixed to
#           dof = n_obs - (n_ay + n_dp - 1), matching R.
#   [FIX-3] `cum_filled` was computed from the pre-deflation triangle, but
#           `factors`/`completed_det` were computed from the post-deflation
#           (real-terms) `working_cum` whenever use_inflation=True -- mixing
#           nominal cumulative values with real-terms development factors.
#           Fixed by refreshing cum_filled after deflation.
#   [FIX-4] Discounting-with-inflation stub ("pass") replaced with a
#           proportional reallocation of nominal IBNR over future periods,
#           mirroring the same fix applied to the Mack engine.
# =============================================================================

import pandas as pd
import numpy as np
from utils.actuarial_engine_utils import (
    period_label, period_index, build_triangles, volume_weighted_factors,
    project_ultimate, deflate_triangle_to_real, reinflate_ibnr_per_ap,
    discount_completed_triangle
)

def run_chain_ladder(cum, factors, origin, grain):
    n_ap, n_dp = cum.shape
    completed = project_ultimate(cum, factors)
    cdfs = []
    run = 1.0
    for f in reversed(factors):
        run *= f
        cdfs.insert(0, run)
    rows = []
    for i in range(n_ap):
        last_obs = -1
        for j in range(n_dp - 1, -1, -1):
            if i + j < n_ap and pd.notna(cum.iloc[i, j]):
                last_obs = j
                break
        if last_obs < 0:
            continue
        current = cum.iloc[i, last_obs]
        ultimate = completed.iloc[i, n_dp - 1]
        ibnr = max(ultimate - current, 0.0)
        cdf = cdfs[last_obs] if last_obs < len(cdfs) else 1.0
        rows.append({
            "Accident_Period": i,
            "Accident_Period_Label": period_label(i, origin, grain),
            "Periods_Developed": last_obs,
            "CDF_to_Ultimate": cdf,
            "Current_Claims": current,
            "Ultimate_Claims": ultimate,
            "IBNR": ibnr
        })
    return pd.DataFrame(rows), completed

def _build_nominal_completed_triangle(completed_real, cum_real, n_periods, per_period_rates):
    """
    Reconstructs a full nominal cumulative triangle from a real-terms
    completed triangle, using the EXACT same cell-by-cell forward-inflation
    logic as utils.reinflate_ibnr_per_ap (rather than approximating with a
    single row-level scale factor). This makes the nominal reserve fed into
    discount_completed_triangle reconcile exactly with the nominal IBNR
    total already being reported for that same pseudo-sample.
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


def _build_retrospective_fitted_triangle(cum_filled, factors, obs_mask, n_ay, n_dp):
    """
    England & Verrall (2002) / R ChainLadder::BootChainLadder style
    "backwards recalculated" fitted cumulative & incremental triangle.

    Unlike project_ultimate (which only fills in FUTURE cells and leaves
    observed cells equal to actual data), this reconstructs fitted values
    for the ENTIRE triangle -- including the observed region -- by taking
    each row's projected ultimate and dividing by the cumulative
    development factor to ultimate at each development period. This is
    what makes Pearson residuals in the observed region meaningfully
    different from zero.
    """
    completed = project_ultimate(cum_filled, factors, obs_mask)

    # cumulative-to-ultimate factor at each development period (tail = 1.0)
    cdf_to_ult = [1.0] * n_dp
    running = 1.0
    for k in range(len(factors) - 1, -1, -1):
        running *= factors[k]
        cdf_to_ult[k] = running

    fitted_cum = completed.copy().astype(float)
    for i in range(n_ay):
        ultimate_i = completed.iloc[i, n_dp - 1]
        for j in range(n_dp):
            denom = cdf_to_ult[j]
            fitted_cum.iloc[i, j] = ultimate_i / denom if denom > 0 else ultimate_i

    fitted_inc = fitted_cum.copy()
    for i in range(n_ay):
        for j in range(n_dp - 1, 0, -1):
            fitted_inc.iloc[i, j] = fitted_cum.iloc[i, j] - fitted_cum.iloc[i, j - 1]
        # j == 0 column is already correct (equals fitted_cum at j=0)

    return fitted_inc, completed


def bootstrap_chain_ladder(working_cum, obs_mask, origin, grain, n_periods,
                           n_iterations, add_process_variance,
                           use_inflation, per_period_rates, cum_inflation,
                           use_discounting, spot_rates, flat_rate, seed=None):
    if seed is not None:
        np.random.seed(seed)

    n_ay, n_dp = working_cum.shape
    cum_filled = working_cum.fillna(0)

    # 1. Deflate if required
    if use_inflation and cum_inflation is not None:
        inc = cum_filled.diff(axis=1).fillna(cum_filled.iloc[:, 0]).fillna(0)
        _, working_cum = deflate_triangle_to_real(inc, cum_inflation, n_ay)
        # [FIX-3] Refresh cum_filled to the REAL-terms triangle so it stays
        # consistent with `factors`/`completed_det`, which are computed
        # from `working_cum` below. Previously cum_filled remained the
        # pre-deflation (nominal) fill, mixing nominal levels with
        # real-terms development factors.
        cum_filled = working_cum.fillna(0)

    # 2. Deterministic CL
    factors = volume_weighted_factors(working_cum, obs_mask)

    # 3. Retrospective fitted triangle + residuals
    # [FIX-1] Use the proper backwards-recalculated fitted triangle instead
    # of project_ultimate's forward-only completion.
    fitted_inc, completed_det = _build_retrospective_fitted_triangle(
        cum_filled, factors, obs_mask, n_ay, n_dp
    )

    residuals_list = []
    for i in range(n_ay):
        for j in range(n_dp):
            if i + j < n_ay and obs_mask.iloc[i, j]:
                actual = (cum_filled.iloc[i, j] - cum_filled.iloc[i, j - 1] if j > 0 else cum_filled.iloc[i, j])
                fitted = fitted_inc.iloc[i, j]
                resid = (actual - fitted) / np.sqrt(abs(fitted)) if fitted > 0 else 0.0
                residuals_list.append(resid)

    residuals = np.array(residuals_list)
    n_obs = len(residuals)
    # [FIX-2] Standard ODP GLM parameter count: n_ay accident-year effects
    # + n_dp development-period effects - 1 shared constant.
    dof = max(n_obs - (n_ay + n_dp - 1), 1)
    phi = max(np.sum(residuals ** 2) / dof, 0.01)

    # 4. Bootstrap Simulation
    ibnr_nominal_samples = []
    ibnr_discounted_samples = []

    for iteration in range(n_iterations):
        sampled = np.random.choice(residuals, size=n_obs, replace=True)
        pseudo_inc = fitted_inc.copy().astype(float)
        idx = 0
        for i in range(n_ay):
            for j in range(n_dp):
                if i + j < n_ay and obs_mask.iloc[i, j]:
                    fv = fitted_inc.iloc[i, j]
                    pv = fv + sampled[idx] * np.sqrt(max(abs(fv), 0.001))
                    pseudo_inc.iloc[i, j] = max(pv, 0.0)
                    idx += 1

        pseudo_cum = pseudo_inc.cumsum(axis=1)
        pseudo_factors = volume_weighted_factors(pseudo_cum, obs_mask)
        pseudo_completed = project_ultimate(pseudo_cum, pseudo_factors, obs_mask)

        if add_process_variance and phi > 1e-10:
            proc_inc = pseudo_completed.copy()
            for i in range(n_ay):
                for j in range(n_dp - 1, 0, -1):
                    proc_inc.iloc[i, j] = pseudo_completed.iloc[i, j] - pseudo_completed.iloc[i, j - 1]
            for i in range(n_ay):
                for j in range(n_dp):
                    is_future = (i + j >= n_ay) or (not obs_mask.iloc[i, j])
                    if is_future:
                        mean_val = proc_inc.iloc[i, j]
                        if pd.notna(mean_val) and mean_val > 0:
                            shape = mean_val / phi
                            scale = phi
                            proc_inc.iloc[i, j] = max(np.random.gamma(shape, scale), 0.0)
                        else:
                            proc_inc.iloc[i, j] = 0.0
            pseudo_completed = proc_inc.copy()
            for i in range(n_ay):
                running = 0.0
                for j in range(n_dp):
                    v = proc_inc.iloc[i, j]
                    running += v if pd.notna(v) and v > 0 else 0.0
                    pseudo_completed.iloc[i, j] = running

        ibnr_working = 0.0
        for i in range(n_ay):
            last_obs = -1
            for j in range(n_dp - 1, -1, -1):
                if i + j < n_ay and obs_mask.iloc[i, j]:
                    last_obs = j
                    break
            if last_obs >= 0:
                current = pseudo_cum.iloc[i, last_obs]
                ultimate = pseudo_completed.iloc[i, n_dp - 1]
                ibnr_working += max(ultimate - current, 0.0)

        nominal_map = None
        if use_inflation and per_period_rates is not None:
            nominal_map = reinflate_ibnr_per_ap(pseudo_completed, pseudo_cum, n_periods, per_period_rates)
            ibnr_nominal_this = sum(nominal_map.values())
        else:
            ibnr_nominal_this = ibnr_working
        ibnr_nominal_samples.append(ibnr_nominal_this)

        if use_discounting:
            if use_inflation and per_period_rates is not None:
                # [FIX-4] Previously a no-op ("pass"). Reconstruct the
                # nominal cumulative triangle cell-by-cell using the same
                # forward-inflation logic as reinflate_ibnr_per_ap, so the
                # discounted reserve reconciles exactly with the nominal
                # IBNR already computed for this pseudo-sample above
                # (rather than approximating with a row-level scale factor).
                nom_completed = _build_nominal_completed_triangle(
                    pseudo_completed, pseudo_cum, n_periods, per_period_rates
                )
            else:
                nom_completed = pseudo_completed
            _, total_disc = discount_completed_triangle(nom_completed, pseudo_cum, n_periods, grain, spot_rates, flat_rate)
            ibnr_discounted_samples.append(total_disc)

    ibnr_nominal_arr = np.array(ibnr_nominal_samples)
    ibnr_discounted_arr = np.array(ibnr_discounted_samples) if ibnr_discounted_samples else None

    cl_df, cl_completed = run_chain_ladder(cum_filled, factors, origin, grain)
    cl_ibnr_working = cl_df["IBNR"].sum()
    if use_inflation and per_period_rates is not None:
        nom_map = reinflate_ibnr_per_ap(cl_completed, working_cum, n_periods, per_period_rates)
        cl_ibnr_nominal = sum(nom_map.values())
    else:
        cl_ibnr_nominal = cl_ibnr_working

    results = {
        "cl_ibnr_nominal": cl_ibnr_nominal,
        "bootstrap_mean": float(np.mean(ibnr_nominal_arr)),
        "bootstrap_median": float(np.median(ibnr_nominal_arr)),
        "bootstrap_std": float(np.std(ibnr_nominal_arr, ddof=1)),
        "bootstrap_cv": (float(np.std(ibnr_nominal_arr, ddof=1) / np.mean(ibnr_nominal_arr)) if np.mean(ibnr_nominal_arr) > 0 else 0.0),
        "phi": float(phi),
        "process_variance": add_process_variance,
        "ibnr_nominal_samples": ibnr_nominal_arr,
        "ibnr_discounted_samples": ibnr_discounted_arr,
        "dev_factors": factors,
        "residuals": residuals,
        "n_iterations": n_iterations,
        "percentiles_nominal": {},
        "percentiles_discounted": {},
    }
    for p in [50, 70, 75, 80, 85, 90, 95, 99, 99.5]:
        results["percentiles_nominal"][p] = float(np.percentile(ibnr_nominal_arr, p))
        if ibnr_discounted_arr is not None:
            results["percentiles_discounted"][p] = float(np.percentile(ibnr_discounted_arr, p))

    return results

def calculate_risk_adjustment(boot_results, confidence_level, ra_base):
    base = boot_results["cl_ibnr_nominal"] if ra_base == "cl" else boot_results["bootstrap_mean"]
    pctl = boot_results["percentiles_nominal"].get(confidence_level, base)
    ra = max(pctl - base, 0.0)
    return {
        "confidence_level": confidence_level,
        "ra_base": ra_base,
        "base_ibnr": base,
        "percentile_ibnr": pctl,
        "risk_adjustment": ra
    }
