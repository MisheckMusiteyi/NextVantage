# -*- coding: utf-8 -*-
# =============================================================================
#  SHARED ACTUARIAL ENGINE HELPERS
#  Used by BCL, Cape Cod, BF, Mack, and Bootstrap engines
# =============================================================================

import pandas as pd
import numpy as np

def periods_per_year(grain):
    return {"Y": 1, "H": 2, "Q": 4, "M": 12}[grain]

def period_index(dt, origin, grain):
    if grain == "Y": return dt.year - origin.year
    elif grain == "H":
        oy = origin.year * 2 + (0 if origin.month <= 6 else 1)
        dy = dt.year * 2 + (0 if dt.month <= 6 else 1)
        return dy - oy
    elif grain == "Q":
        oq = origin.year * 4 + (origin.month - 1) // 3
        dq = dt.year * 4 + (dt.month - 1) // 3
        return dq - oq
    elif grain == "M":
        return (dt.year - origin.year) * 12 + (dt.month - origin.month)

def period_label(idx, origin, grain):
    if grain == "Y": return str(origin.year + idx)
    elif grain == "H":
        base = origin.year * 2 + (0 if origin.month <= 6 else 1) + idx
        y, h = base // 2, base % 2 + 1
        return f"{y}-H{h}"
    elif grain == "Q":
        base = origin.year * 4 + (origin.month - 1) // 3 + idx
        y, q = base // 4, base % 4 + 1
        return f"{y}-Q{q}"
    elif grain == "M":
        total = origin.year * 12 + origin.month - 1 + idx
        y, m = total // 12, total % 12 + 1
        return f"{y}-{m:02d}"

def build_triangles(df, loss_col, report_col, amount_col, origin, grain, n_periods):
    df = df.copy()
    df["__ap"] = df[loss_col].apply(lambda d: period_index(d, origin, grain))
    df["__dp"] = df.apply(
        lambda r: max(0, min(period_index(r[report_col], origin, grain) - period_index(r[loss_col], origin, grain), n_periods - 1)),
        axis=1
    )
    df = df[(df["__ap"] >= 0) & (df["__ap"] < n_periods)]
    df_valid = df.dropna(subset=[amount_col])

    if len(df_valid) > 0:
        pivot = df_valid.pivot_table(index="__ap", columns="__dp", values=amount_col, aggfunc="sum")
    else:
        pivot = pd.DataFrame(index=pd.RangeIndex(n_periods), columns=pd.RangeIndex(n_periods), dtype=float)

    for ap in range(n_periods):
        if ap not in pivot.index: pivot.loc[ap] = np.nan
    for dp in range(n_periods):
        if dp not in pivot.columns: pivot[dp] = np.nan

    inc = pivot.sort_index()[sorted(pivot.columns)].copy().astype(float)
    obs_mask = pd.DataFrame(False, index=inc.index, columns=inc.columns)
    for ap in inc.index:
        for dp in inc.columns:
            if ap + dp < n_periods:
                obs_mask.loc[ap, dp] = pd.notna(inc.loc[ap, dp])

    for ap in inc.index:
        for dp in inc.columns:
            if ap + dp >= n_periods: inc.loc[ap, dp] = np.nan

    cum = inc.copy()
    for ap in inc.index:
        has_obs = any(pd.notna(inc.loc[ap, dp]) for dp in inc.columns if ap + dp < n_periods)
        if not has_obs:
            cum.loc[ap] = np.nan
            continue
        running = 0.0
        for dp in sorted(inc.columns):
            if ap + dp < n_periods:
                v = inc.loc[ap, dp]
                running += v if pd.notna(v) else 0.0
                cum.loc[ap, dp] = running
            else:
                cum.loc[ap, dp] = np.nan
    return inc, cum, obs_mask

def project_ultimate(cum, factors, obs_mask=None):
    n_ap, n_dp = cum.shape
    completed = cum.copy().astype(float)
    for i in range(n_ap):
        last_obs = -1
        for j in range(n_dp - 1, -1, -1):
            if obs_mask is not None:
                if obs_mask.iloc[i, j] and pd.notna(cum.iloc[i, j]):
                    last_obs = j
                    break
            else:
                if i + j < n_ap and pd.notna(cum.iloc[i, j]):
                    last_obs = j
                    break
        if last_obs < 0:
            continue
        for j in range(last_obs, n_dp - 1):
            if j < len(factors):
                prev = completed.iloc[i, j]
                completed.iloc[i, j + 1] = prev * factors[j] if pd.notna(prev) and prev > 0 else 0.0
    return completed

def volume_weighted_factors(cum, obs_mask=None):
    n_ap, n_dp = cum.shape
    factors = []
    for j in range(n_dp - 1):
        num, den = 0.0, 0.0
        for i in range(n_ap):
            if i + j + 1 < n_ap:
                if obs_mask is not None:
                    if not obs_mask.iloc[i, j] or not obs_mask.iloc[i, j + 1]:
                        continue
                c = cum.iloc[i, j]
                n = cum.iloc[i, j + 1]
                if pd.notna(c) and pd.notna(n) and c > 0:
                    num += n
                    den += c
        factors.append(num / den if den > 0 else 1.0)
    return factors

def deflate_triangle_to_real(inc, cum_inflation, n_periods):
    valuation_idx = n_periods - 1
    if len(cum_inflation) <= valuation_idx:
        cum_inflation = np.append(cum_inflation, [cum_inflation[-1]] * (valuation_idx - len(cum_inflation) + 1))
    inf_val = cum_inflation[valuation_idx]
    real_inc = inc.copy().astype(float)
    for ap in inc.index:
        for dp in inc.columns:
            if ap + dp >= n_periods: continue
            val = inc.loc[ap, dp]
            if pd.isna(val): continue
            t = ap + dp
            inf_t = cum_inflation[min(t, len(cum_inflation) - 1)]
            real_inc.loc[ap, dp] = val * (inf_val / inf_t) if inf_t > 0 else val
    real_cum = real_inc.copy()
    for ap in real_inc.index:
        has_obs = any(pd.notna(real_inc.loc[ap, dp]) for dp in real_inc.columns if ap + dp < n_periods)
        if not has_obs:
            real_cum.loc[ap] = np.nan
            continue
        running = 0.0
        for dp in sorted(real_inc.columns):
            if ap + dp < n_periods:
                v = real_inc.loc[ap, dp]
                running += v if pd.notna(v) else 0.0
                real_cum.loc[ap, dp] = running
            else:
                real_cum.loc[ap, dp] = np.nan
    return real_inc, real_cum

def deflate_premiums(premiums, cum_inflation, n_periods):
    """
    Deflate each accident period's premium to valuation-date real terms.

    FIX (was): a single blanket factor cum_inflation[n_ap-1] / cum_inflation[0]
    was applied to every accident period, which is only correct for ap=0.
    Every other accident period's premium was over-deflated.

    NOW: mirrors deflate_triangle_to_real's per-cell convention -- premium
    for accident period `ap` is treated as occurring at calendar time t=ap
    (written/earned at the start of that accident period, consistent with
    how the dp=0 cell of the claims triangle is deflated), and is deflated
    using the inflation index at that specific time.
    """
    valuation_idx = n_periods - 1
    if len(cum_inflation) <= valuation_idx:
        cum_inflation = np.append(
            cum_inflation, [cum_inflation[-1]] * (valuation_idx - len(cum_inflation) + 1)
        )
    inf_val = cum_inflation[valuation_idx]

    working_premiums = []
    for ap, p in enumerate(premiums):
        inf_t = cum_inflation[min(ap, len(cum_inflation) - 1)]
        working_premiums.append(p * (inf_val / inf_t) if inf_t > 0 else p)
    return working_premiums

def build_real_emergence_triangle(cum_triangle_real, expected_ultimate_real, pct_developed, n_periods):
    """
    Build a full real "completed" cumulative triangle for BF / Cape Cod,
    where future emergence is distributed across development periods in
    proportion to the chain-ladder-implied %-developed pattern, but scaled
    to each accident period's OWN expected_ultimate_real (the BF or Cape
    Cod ultimate), not the plain chain-ladder ultimate.

    FIX (was): BF and Cape Cod computed their own real IBNR correctly
    (BF_IBNR_Real / Cape_Cod_IBNR_Real) but then reinflated and discounted
    a completely different, plain chain-ladder completed_real triangle --
    so the "nominal" and "discounted" figures reported had nothing to do
    with the BF/Cape Cod ultimate. This builds the correct real pattern to
    feed into reinflate_ibnr_per_ap / build_nominal_triangle_for_discounting.

    Parameters
    ----------
    cum_triangle_real : observed real cumulative triangle (working_cum)
    expected_ultimate_real : list/array, one value per accident period
        (BF: premium*ELR ; Cape Cod: premium*cape_cod_lr, on real basis)
    pct_developed : list, one value per development-period column j
        (i.e. 1/CDF_j)
    n_periods : n_ap == n_dp (square triangle)
    """
    dp_cols = sorted(cum_triangle_real.columns)
    completed = cum_triangle_real.copy().astype(float)

    for ap in cum_triangle_real.index:
        last_obs = -1
        for dp in sorted(cum_triangle_real.columns, reverse=True):
            if ap + dp < n_periods and pd.notna(cum_triangle_real.loc[ap, dp]):
                last_obs = dp
                break
        if last_obs < 0:
            continue

        eu = expected_ultimate_real[ap]
        pct_last_obs = pct_developed[last_obs] if last_obs < len(pct_developed) else 1.0
        base_val = cum_triangle_real.loc[ap, last_obs]

        for dp in dp_cols:
            if dp <= last_obs:
                continue
            pct_dp = pct_developed[dp] if dp < len(pct_developed) else 1.0
            if pct_last_obs >= 1.0:
                completed.loc[ap, dp] = base_val
            else:
                share = (pct_dp - pct_last_obs) / (1.0 - pct_last_obs)
                completed.loc[ap, dp] = base_val + share * (eu - base_val)
    return completed

def reinflate_ibnr_per_ap(completed_real, cum_triangle_real, n_periods, per_period_rates):
    valuation_idx = n_periods - 1
    last_rate = per_period_rates[-1] if len(per_period_rates) > 0 else 0.0

    def forward_inflation(t_future):
        factor = 1.0
        for k in range(valuation_idx + 1, t_future + 1):
            ki = k - valuation_idx - 1
            r = per_period_rates[ki] if ki < len(per_period_rates) else last_rate
            factor *= (1.0 + r)
        return factor

    nominal_by_ap = {}
    dp_cols = sorted(completed_real.columns)
    for ap in completed_real.index:
        last_obs = -1
        for dp in sorted(cum_triangle_real.columns, reverse=True):
            if ap + dp < n_periods and pd.notna(cum_triangle_real.loc[ap, dp]):
                last_obs = dp
                break
        if last_obs < 0:
            nominal_by_ap[ap] = 0.0
            continue
        total = 0.0
        for idx_dp, dp in enumerate(dp_cols):
            if dp <= last_obs: continue
            if ap + dp >= 2 * n_periods: break
            cum_curr = completed_real.loc[ap, dp]
            if pd.isna(cum_curr): continue
            cum_prev = completed_real.loc[ap, dp_cols[idx_dp - 1]] if idx_dp > 0 else 0.0
            inc_real = max(float(cum_curr) - float(cum_prev if pd.notna(cum_prev) else 0.0), 0.0)
            if inc_real <= 0.0: continue
            total += inc_real * forward_inflation(ap + dp)
        nominal_by_ap[ap] = total
    return nominal_by_ap

def build_nominal_triangle_for_discounting(completed_real, cum_triangle_nominal, n_periods, per_period_rates):
    """
    Build a proper nominal cumulative triangle cell-by-cell, suitable for
    discount_completed_triangle.

    FIX (was, in BCL): future cells were set with
        nominal_map[ap] + working_cum.iloc[ap, n_dp-1]
    which is NaN for almost every accident period (that column is only
    observed for the oldest, fully-developed row), and dumped one flat
    total into every future cell instead of an incremental buildup -- so
    discount_completed_triangle saw NaN or zero increments for nearly
    every accident period and barely discounted anything.

    NOW: observed cells are taken directly from cum_triangle_nominal
    (real data). Future cells are built by taking each REAL incremental
    amount off completed_real, inflating THAT SPECIFIC increment forward
    from the valuation date to its own calendar period, and cumulating on
    top of the last known nominal value.
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

    nominal = cum_triangle_nominal.copy().astype(float)
    dp_cols = sorted(completed_real.columns)

    for ap in completed_real.index:
        last_obs = -1
        for dp in sorted(cum_triangle_nominal.columns, reverse=True):
            if ap + dp < n_periods and pd.notna(cum_triangle_nominal.loc[ap, dp]):
                last_obs = dp
                break
        if last_obs < 0:
            continue

        running_nominal = float(cum_triangle_nominal.loc[ap, last_obs])
        for idx_dp, dp in enumerate(dp_cols):
            if dp <= last_obs:
                continue
            cum_curr = completed_real.loc[ap, dp]
            if pd.isna(cum_curr):
                continue
            cum_prev = completed_real.loc[ap, dp_cols[idx_dp - 1]] if idx_dp > 0 else 0.0
            inc_real = max(float(cum_curr) - float(cum_prev if pd.notna(cum_prev) else 0.0), 0.0)
            inc_nominal = inc_real * forward_inflation(ap + dp)
            running_nominal += inc_nominal
            nominal.loc[ap, dp] = running_nominal

    return nominal

def discount_completed_triangle(completed_triangle, cum_triangle, n_periods, grain, spot_rates=None, flat_rate=None):
    ppy = periods_per_year(grain)
    dp_cols = sorted(completed_triangle.columns)
    total_nominal = 0.0
    total_discounted = 0.0
    for ap in completed_triangle.index:
        last_obs = -1
        for dp in sorted(cum_triangle.columns, reverse=True):
            if ap + dp < n_periods and pd.notna(cum_triangle.loc[ap, dp]):
                last_obs = dp
                break
        if last_obs < 0 or last_obs >= max(dp_cols):
            continue
        for idx_dp, dp in enumerate(dp_cols):
            if dp <= last_obs: continue
            cum_curr = completed_triangle.loc[ap, dp]
            if pd.isna(cum_curr): continue
            cum_prev = completed_triangle.loc[ap, dp_cols[idx_dp - 1]] if idx_dp > 0 else 0.0
            inc_payment = max(float(cum_curr) - float(cum_prev if pd.notna(cum_prev) else 0.0), 0.0)
            if inc_payment <= 0.0: continue
            periods_ahead = dp - last_obs
            years_ahead = periods_ahead / ppy
            if spot_rates is not None:
                idx = min(int(periods_ahead) - 1, len(spot_rates) - 1)
                r = float(spot_rates[max(idx, 0)])
            else:
                r = float(flat_rate)
            df_factor = 1.0 / (1.0 + r) ** years_ahead
            total_nominal += inc_payment
            total_discounted += inc_payment * df_factor
    return total_nominal, total_discounted
