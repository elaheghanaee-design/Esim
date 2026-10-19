"""
Day-Ahead + aFRR (R2) + mFRR (R3) 15-year BESS scheduling model
Python / Pyomo / HiGHS translation of the supplied Julia / JuMP / CPLEX model.

Main features
-------------
- 15-year multi-year simulation with continuous SoC/capacity carry-over
- 48 h rolling horizon, 24 h implemented
- Day-ahead market scheduling
- Secondary reserve capacity (R2 / aFRR) up/down
- Activated secondary regulation energy up/down
- Tertiary reserve (R3 / mFRR) up/down with activated energy based on use ratios
- Wind + BESS operation
- No grid charging
- Piecewise-linear cycle-aging cost inside MILP
- Linear calendar-aging cost inside MILP
- Rainflow cycle-aging post-processing
- Nonlinear calendar-aging post-processing
- Weekly usable-capacity update
- Excel export matching the Julia model structure

Install:
    pip install numpy pandas pyomo highspy openpyxl rainflow

Run:
    python main_DA_aFRR_mFRR_15years_WindFarmDan.py

Place these Excel files in the same folder as this script:
    priceSPOTDan.xlsx
    WindFarmDan.xlsx
    data_stroe2017.xlsx
    priceBR2.xlsx
    ratioBR2.xlsx
    priceER2up.xlsx
    priceER2down.xlsx
    useER2up.xlsx
    useER2down.xlsx
    priceER3up.xlsx
    priceER3down.xlsx
    useER3up.xlsx
    useER3down.xlsx
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import rainflow


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

PRICE_FILE = BASE_DIR / "priceSPOTDan.xlsx"
WIND_FILE = BASE_DIR / "WindFarmR3_scaled_308_over_35.xlsx"
STROE_FILE = BASE_DIR / "data_stroe2017.xlsx"

PRICE_BR2_FILE = BASE_DIR / "priceBR2.xlsx"
RATIO_BR2_FILE = BASE_DIR / "ratioBR2.xlsx"
PRICE_ER2UP_FILE = BASE_DIR / "priceER2up.xlsx"
PRICE_ER2DOWN_FILE = BASE_DIR / "priceER2down.xlsx"
USE_ER2UP_FILE = BASE_DIR / "useER2up.xlsx"
USE_ER2DOWN_FILE = BASE_DIR / "useER2down.xlsx"

# Tertiary regulation / mFRR data.
# FASBATT names these ER3 up/down prices and expected use ratios.
PRICE_ER3UP_FILE = BASE_DIR / "priceER3up.xlsx"
PRICE_ER3DOWN_FILE = BASE_DIR / "priceER3down.xlsx"
USE_ER3UP_FILE = BASE_DIR / "useER3up.xlsx"
USE_ER3DOWN_FILE = BASE_DIR / "useER3down.xlsx"


# =========================================================
# READ EXCEL DATA
# =========================================================

def read_single_excel_column(
    file_path: Path,
    sheet_name: str,
    column: str,
    start_row: int,
    nrows: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Read one Excel column and replace nonnumeric/missing values by fill_value."""
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        usecols=column,
        skiprows=start_row - 1,
        nrows=nrows,
        header=None,
        engine="openpyxl",
    )

    values = (
        pd.to_numeric(df.iloc[:, 0], errors="coerce")
        .fillna(fill_value)
        .astype(float)
        .to_numpy()
    )
    return values


def read_8760_series(file_path: Path) -> np.ndarray:
    """Equivalent to Julia's read_8760_series: Sheet1!C2:C8761."""
    values = read_single_excel_column(
        file_path=file_path,
        sheet_name="Sheet1",
        column="C",
        start_row=2,
        nrows=8760,
        fill_value=0.0,
    )
    if len(values) != 8760:
        raise ValueError(
            f"Expected 8760 values in {file_path.name}; got {len(values)}."
        )
    return values


def read_stroe_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read A2:B56 from sheet data_stroe, clean invalid values and sort by x."""
    df = pd.read_excel(
        file_path,
        sheet_name="data_stroe",
        usecols="A:B",
        skiprows=1,
        nrows=55,
        header=None,
        engine="openpyxl",
    )

    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    order = np.argsort(x)
    return x[order], y[order]


# =========================================================
# GENERAL FUNCTIONS
# =========================================================

def interp_linear_extrap(
    x: np.ndarray,
    y: np.ndarray,
    xq: float,
) -> float:
    """Linear interpolation with linear extrapolation at both ends."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        raise ValueError("At least two points are required for interpolation.")

    if xq <= x[0]:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return float(y[0] + slope * (xq - x[0]))

    if xq >= x[-1]:
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return float(y[-2] + slope * (xq - x[-2]))

    return float(np.interp(xq, x, y))


def clean_soc_profile(
    values: List[float] | np.ndarray,
    tol: float = 1e-6,
) -> np.ndarray:
    """Remove consecutive repeated/near-equal SoC values."""
    cleaned: List[float] = []

    for value in values:
        value = float(value)
        if not cleaned or abs(value - cleaned[-1]) > tol:
            cleaned.append(value)

    return np.asarray(cleaned, dtype=float)


# =========================================================
# RAINFLOW POST-PROCESSING
#
# DOD is always normalized by rated capacity = 60 MWh.
# =========================================================

def rainflow_damage_and_cost(
    soc_profile: List[float] | np.ndarray,
    socbess_ref: float,
    x_stroe: np.ndarray,
    y_stroe: np.ndarray,
    ecap_cost: float,
    tol: float = 1e-6,
) -> Tuple[float, float]:
    """
    Compute cumulative Rainflow damage and cost.

    Python package 'rainflow' returns cycles as:
        (range, mean, count, start_index, end_index)

    Count is normally 1.0 for a full cycle or 0.5 for a half cycle.
    """
    soc_clean = clean_soc_profile(soc_profile, tol=tol)

    if (
        len(soc_clean) < 3
        or float(np.max(soc_clean) - np.min(soc_clean)) < tol
    ):
        return 0.0, 0.0

    total_damage = 0.0
    total_cost = 0.0

    for cycle_range, _mean, count, _i_start, _i_end in rainflow.extract_cycles(
        soc_clean
    ):
        cycle_range = float(cycle_range)
        count = float(count)

        if cycle_range < tol:
            continue

        if not (np.isclose(count, 1.0) or np.isclose(count, 0.5)):
            continue

        dcr = cycle_range / socbess_ref
        dcr_percent = 100.0 * dcr

        ncycles = interp_linear_extrap(
            x_stroe,
            y_stroe,
            dcr_percent,
        )

        if ncycles <= 0.0 or not np.isfinite(ncycles):
            continue

        damage = count / ncycles
        cost = count * ecap_cost / ncycles

        total_damage += damage
        total_cost += cost

    return float(total_damage), float(total_cost)


# =========================================================
# NONLINEAR CALENDAR AGING FOR WEEKLY CAPACITY UPDATE
#
# SoC percentage is always normalized by rated capacity.
# =========================================================

def calendar_polynomial_hourly_weekly_percent(
    soc_week: List[float] | np.ndarray,
    rated_capacity: float,
    first_hour_index: int,
    commissioning_hours: float = 0.0,
    hours_per_month: float = 720.0,
) -> float:
    """
    first_hour_index is 1-based within the current operating year,
    matching the Julia implementation.
    """
    weekly_percent = 0.0

    for h0, soc_value in enumerate(soc_week):
        global_hour_now = first_hour_index + h0
        global_hour_prev = global_hour_now - 1

        t_now = commissioning_hours + global_hour_now
        t_prev = commissioning_hours + global_hour_prev

        time_increment = (
            (t_now / hours_per_month) ** 0.8
            - (t_prev / hours_per_month) ** 0.8
        )

        soc_percent = 100.0 * float(soc_value) / rated_capacity

        weekly_percent += time_increment * (
            0.0064 * soc_percent ** 0.823
            + 0.1751
        )

    return float(weekly_percent)


# =========================================================
# PARAMETERS
# =========================================================

T = 48
IMPLEMENT = 24

# =========================================================
# MAIN PLANT PARAMETERS
# Values aligned with the previous DA-only Python model.
# =========================================================

NODE_CAPACITY_G = 300.0
NODE_CAPACITY_C = 20.0

# WindFarmDan.xlsx corresponds to a 35 MW wind farm.
# The raw wind series is first converted to per-unit:
#     wind_pu = wind_raw / WIND_DATA_BASE_CAPACITY_MW
# and then scaled to the desired wind-farm capacity:
#     wind = wind_pu * WIND_FARM_CAPACITY_MW
#
# To test a different wind-farm size later, change ONLY
# WIND_FARM_CAPACITY_MW.
WIND_DATA_BASE_CAPACITY_MW = 35.0
WIND_FARM_CAPACITY_MW = 308.0

WP_MAX = WIND_FARM_CAPACITY_MW
PBESS = 20.0

RATED_CAPACITY = 60.0
SOCBESS_INITIAL = RATED_CAPACITY  # fixed reference for aging equations

ETA = 0.86

# Multi-year reserve simulation.
MAX_YEARS = 25
SOH_STOP_THRESHOLD = 0.80

SIMULATION_INITIAL_CAPACITY = RATED_CAPACITY

# Same initial-SoC assumption as the previous DA-only model:
# the simulation starts only once at 50% SoC.
# SoC is then carried continuously from day to day and year to year.
INITIAL_SOC = 0.5 * RATED_CAPACITY

ECAP_COST = 6_200_000.0

DAYS_PER_WEEK = 7
HOURS_PER_MONTH = 720.0

CAL_SOC_COEFF = 0.0028
CAL_CAPACITY_COEFF = 0.001939
# Calendar-aging economic coefficient from the reference formulation:
#     gamma = BIC / (0.2 * E_rated)
# where BIC = total battery investment/replacement cost [EUR]
# and 0.2 corresponds to the capacity loss from SOH=1.0 to EOL=0.8.
CAL_COST_FACTOR = ECAP_COST / (0.2 * SOCBESS_INITIAL)

INCLUDE_CALENDAR_IN_CAPACITY_UPDATE = True

SLOPE_BESS_COST1 = np.array(
    [1.34499, 2.2341250142, 2.5039895004],
    dtype=float,
)
SLOPE_BESS_COST = (ECAP_COST/1600000.0) * SLOPE_BESS_COST1

BETA1 = np.array(
    [0.0, -44.4567550828, -66.0459139776],
    dtype=float,
)
BETA = (ECAP_COST/1600000.0) * BETA1

OUTPUT_FILE = BASE_DIR / "8results_With charge_DA_aFRR_mFRR_WindFarmDan.xlsx"


# =========================================================
# DAILY PYOMO MILP
# =========================================================

def solve_daily_model(
    price: np.ndarray,
    wind: np.ndarray,
    price_br2: np.ndarray,
    ratio_br2: np.ndarray,
    price_er2up: np.ndarray,
    price_er2down: np.ndarray,
    use_er2up: np.ndarray,
    use_er2down: np.ndarray,
    price_er3up: np.ndarray,
    price_er3down: np.ndarray,
    use_er3up: np.ndarray,
    use_er3down: np.ndarray,
    t_current: int,
    current_capacity: float,
    soc_initial: float,
    commissioning_hours: float,
    auxiliar1: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    """Build and solve one rolling-horizon DA + aFRR + mFRR MILP with Pyomo + HiGHS."""

    model = pyo.ConcreteModel(name="DA_aFRR_mFRR_BESS")

    model.T = pyo.RangeSet(0, t_current - 1)
    model.J = pyo.RangeSet(0, 2)

    soc_min_current = 0.10 * current_capacity
    soc_max_current = current_capacity

    # =====================================================
    # VARIABLES
    # =====================================================

    model.wp = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, WP_MAX),
    )

    model.wp2b = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, PBESS),
    )

    model.soc = pyo.Var(
        model.T,
        domain=pyo.Reals,
        bounds=(soc_min_current, soc_max_current),
    )

    model.ed_discharged = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
    )

    # Julia fixes grid charging to zero.
    model.ed_charged = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, NODE_CAPACITY_C),
    )

    model.p_DA = pyo.Var(
        model.T,
        domain=pyo.Reals,
    )

    model.aux1 = pyo.Var(
        model.T,
        domain=pyo.Binary,
    )

    model.dcr = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
    )

    model.bess_cost = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
    )

    model.pb = pyo.Var(
        model.T,
        domain=pyo.Reals,
    )

    # Secondary reserve capacity.
    model.br2up = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, 2.0 * PBESS),
    )

    model.br2down = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, 2.0 * PBESS),
    )

    # Activated regulation energy.
    model.er2up = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
    )

    model.er2down = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
    )

    model.aux_er2 = pyo.Var(
        model.T,
        domain=pyo.Binary,
    )

    # Tertiary reserve / mFRR capacity.
    # In FASBATT there is no separate mFRR capacity-payment term;
    # activated mFRR energy is use_ratio * BR3.
    model.br3up = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, 2.0 * PBESS),
    )

    model.br3down = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, 2.0 * PBESS),
    )

    model.er3up = pyo.Expression(
        model.T,
        rule=lambda m, t: float(use_er3up[t]) * m.br3up[t],
    )

    model.er3down = pyo.Expression(
        model.T,
        rule=lambda m, t: float(use_er3down[t]) * m.br3down[t],
    )

    model.y = pyo.Var(
        model.T,
        model.J,
        domain=pyo.Binary,
    )

    model.ud = pyo.Var(
        model.T,
        domain=pyo.Binary,
    )

    model.AG_cal = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
    )

    model.C_cal = pyo.Var(
        model.T,
        domain=pyo.NonNegativeReals,
    )

    # =====================================================
    # POWER BALANCE
    # =====================================================

    def power_balance_rule(m, t):
        return (
            m.wp[t]
            + m.ed_discharged[t]
            - m.ed_charged[t]
            == m.p_DA[t]
        )

    model.power_balance = pyo.Constraint(
        model.T,
        rule=power_balance_rule,
    )

    # =====================================================
    # WIND LIMIT
    # =====================================================

    def wind_limit_rule(m, t):
        return m.wp[t] + m.wp2b[t] <= float(wind[t])

    model.wind_limit = pyo.Constraint(
        model.T,
        rule=wind_limit_rule,
    )

    # =====================================================
    # CHARGE / DISCHARGE LOGIC
    # =====================================================

    def charge_mode_rule(m, t):
        return (
            m.ed_charged[t] + m.wp2b[t]
            <= PBESS * (1.0 - m.aux1[t])
        )

    model.charge_mode = pyo.Constraint(
        model.T,
        rule=charge_mode_rule,
    )

    def discharge_mode_rule(m, t):
        return (
            m.ed_discharged[t]
            <= PBESS * m.aux1[t]
        )

    model.discharge_mode = pyo.Constraint(
        model.T,
        rule=discharge_mode_rule,
    )

    def battery_power_rule(m, t):
        return (
            m.ed_discharged[t]
            - m.ed_charged[t]
            - m.wp2b[t]
            == m.pb[t]
        )

    model.battery_power = pyo.Constraint(
        model.T,
        rule=battery_power_rule,
    )

    # =====================================================
    # SECONDARY RESERVE CAPACITY (R2)
    # =====================================================

    def br2_up_limit_rule(m, t):
        return m.br2up[t] <= PBESS - m.pb[t]

    model.br2_up_limit = pyo.Constraint(
        model.T,
        rule=br2_up_limit_rule,
    )

    def br2_down_limit_rule(m, t):
        return m.br2down[t] <= m.pb[t] + PBESS

    model.br2_down_limit = pyo.Constraint(
        model.T,
        rule=br2_down_limit_rule,
    )

    def br2_ratio_rule(m, t):
        return (
            m.br2up[t]
            == float(ratio_br2[t]) * (m.br2up[t] + m.br2down[t])
        )

    model.br2_ratio = pyo.Constraint(
        model.T,
        rule=br2_ratio_rule,
    )

    # =====================================================
    # ACTIVATED SECONDARY REGULATION ENERGY
    # =====================================================

    def er2_activation_rule(m, t):
        return (
            m.er2up[t] - m.er2down[t]
            == float(use_er2up[t]) * m.br2up[t]
            - float(use_er2down[t]) * m.br2down[t]
        )

    model.er2_activation = pyo.Constraint(
        model.T,
        rule=er2_activation_rule,
    )

    def er2_up_mode_rule(m, t):
        return m.er2up[t] <= PBESS * m.aux_er2[t]

    model.er2_up_mode = pyo.Constraint(
        model.T,
        rule=er2_up_mode_rule,
    )

    def er2_down_mode_rule(m, t):
        return m.er2down[t] <= PBESS * (1.0 - m.aux_er2[t])

    model.er2_down_mode = pyo.Constraint(
        model.T,
        rule=er2_down_mode_rule,
    )

    # =====================================================
    # TERTIARY RESERVE CAPACITY (R3 / mFRR)
    #
    # FASBATT stacking logic for DA + aFRR + mFRR:
    #   BR3_up   <= PBESS - pb - BR2_up
    #   BR3_down <= pb + PBESS - BR2_down
    # Therefore aFRR capacity is committed first and mFRR
    # can use only the remaining battery power headroom.
    # =====================================================

    def br3_up_limit_rule(m, t):
        return m.br3up[t] <= PBESS - m.pb[t] - m.br2up[t]

    model.br3_up_limit = pyo.Constraint(
        model.T,
        rule=br3_up_limit_rule,
    )

    def br3_down_limit_rule(m, t):
        return m.br3down[t] <= m.pb[t] + PBESS - m.br2down[t]

    model.br3_down_limit = pyo.Constraint(
        model.T,
        rule=br3_down_limit_rule,
    )

    # =====================================================
    # SOC DYNAMICS
    # =====================================================

    def soc_rule(m, t):
        if t == 0:
            return (
                m.soc[t]
                == soc_initial
                + (m.wp2b[t] + m.ed_charged[t]) * ETA
                - m.ed_discharged[t]
                + ETA * m.er2down[t]
                - m.er2up[t]
                + ETA * m.er3down[t]
                - m.er3up[t]
            )

        return (
            m.soc[t]
            == m.soc[t - 1]
            + (m.wp2b[t] + m.ed_charged[t]) * ETA
            - m.ed_discharged[t]
            + ETA * m.er2down[t]
            - m.er2up[t]
            + ETA * m.er3down[t]
            - m.er3up[t]
        )

    model.soc_balance = pyo.Constraint(
        model.T,
        rule=soc_rule,
    )

    # =====================================================
    # DCR EQUATION
    #
    # Always normalized by original rated capacity = 60 MWh.
    # =====================================================

    def dcr_rule(m, t):
        if t == 0:
            return (
                m.dcr[t]
                >= (soc_initial - m.soc[t]) / SOCBESS_INITIAL
            )

        return (
            m.dcr[t]
            >= (m.soc[t - 1] - m.soc[t]) / SOCBESS_INITIAL
        )

    model.dcr_constraint = pyo.Constraint(
        model.T,
        rule=dcr_rule,
    )

    # =====================================================
    # CONEJO PWL FORMULATION
    # =====================================================

    def dcr_ud_rule(m, t):
        return m.dcr[t] <= m.ud[t]

    model.dcr_ud = pyo.Constraint(
        model.T,
        rule=dcr_ud_rule,
    )

    def segment_select_rule(m, t):
        return sum(m.y[t, j] for j in m.J) <= m.ud[t]

    model.segment_select = pyo.Constraint(
        model.T,
        rule=segment_select_rule,
    )

    def dcr_segment_lb_rule(m, t):
        return (
            m.dcr[t]
            >= (
                0.0 * m.y[t, 0]
                + 0.5001 * m.y[t, 1]
                + 0.8001 * m.y[t, 2]
            )
        )

    model.dcr_segment_lb = pyo.Constraint(
        model.T,
        rule=dcr_segment_lb_rule,
    )

    def dcr_segment_ub_rule(m, t):
        return (
            m.dcr[t]
            <= (
                0.5 * m.y[t, 0]
                + 0.8 * m.y[t, 1]
                + 1.0 * m.y[t, 2]
            )
        )

    model.dcr_segment_ub = pyo.Constraint(
        model.T,
        rule=dcr_segment_ub_rule,
    )

    def bess_cost_rule(m, t, j):
        return (
            m.bess_cost[t]
            >= (
                100.0
                * m.dcr[t]
                * float(SLOPE_BESS_COST[j])
                + float(BETA[j]) * m.ud[t]
            )
        )

    model.bess_cost_constraint = pyo.Constraint(
        model.T,
        model.J,
        rule=bess_cost_rule,
    )

    # =====================================================
    # LINEAR CALENDAR AGING INSIDE OPTIMIZATION
    # =====================================================

    time_factor: Dict[int, float] = {}

    for k in range(t_current):
        local_hour_now = float(auxiliar1[k])

        if k == 0:
            local_hour_prev = local_hour_now - 1.0
        else:
            local_hour_prev = float(auxiliar1[k - 1])

        time_factor[k] = (
            (
                (local_hour_now + commissioning_hours)
                / HOURS_PER_MONTH
            ) ** 0.8
            - (
                (local_hour_prev + commissioning_hours)
                / HOURS_PER_MONTH
            ) ** 0.8
        )

    def calendar_damage_rule(m, k):
        return (
            m.AG_cal[k]
            == time_factor[k]
            * (
                CAL_SOC_COEFF * m.soc[k]
                + CAL_CAPACITY_COEFF * SOCBESS_INITIAL
            )
        )

    model.calendar_damage = pyo.Constraint(
        model.T,
        rule=calendar_damage_rule,
    )

    def calendar_cost_rule(m, k):
        return (
            m.C_cal[k]
            == CAL_COST_FACTOR * m.AG_cal[k]
        )

    model.calendar_cost = pyo.Constraint(
        model.T,
        rule=calendar_cost_rule,
    )

    # =====================================================
    # GRID LIMITS
    # =====================================================

    def grid_export_rule(m, t):
        return (
            m.p_DA[t]
            + m.er2up[t]
            - m.er2down[t]
            + m.er3up[t]
            - m.er3down[t]
            <= NODE_CAPACITY_G
        )

    model.grid_export = pyo.Constraint(
        model.T,
        rule=grid_export_rule,
    )

    def grid_import_rule(m, t):
        return (
            m.p_DA[t]
            + m.er2up[t]
            - m.er2down[t]
            + m.er3up[t]
            - m.er3down[t]
            >= -NODE_CAPACITY_C
        )

    model.grid_import = pyo.Constraint(
        model.T,
        rule=grid_import_rule,
    )

    # =====================================================
    # OBJECTIVE
    # =====================================================

    model.objective = pyo.Objective(
        expr=sum(
            float(price[t]) * model.p_DA[t]
            + float(price_br2[t]) * (
                model.br2up[t] + model.br2down[t]
            )
            + float(price_er2up[t]) * model.er2up[t]
            - float(price_er2down[t]) * model.er2down[t]
            + float(price_er3up[t]) * model.er3up[t]
            - float(price_er3down[t]) * model.er3down[t]
            - model.bess_cost[t]
            - model.C_cal[t]
            for t in model.T
        ),
        sense=pyo.maximize,
    )

    # =====================================================
    # SOLVE WITH HIGHS
    # =====================================================

    solver = pyo.SolverFactory("highs")

    if not solver.available(exception_flag=False):
        raise RuntimeError(
            "HiGHS is not available. Install dependencies with: "
            "pip install pyomo highspy"
        )

    solver.options["mip_rel_gap"] = 0.0
    solver.options["time_limit"] = 3600.0

    solve_result = solver.solve(
        model,
        tee=False,
        load_solutions=True,
    )

    termination = solve_result.solver.termination_condition

    acceptable = {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.maxTimeLimit,
        pyo.TerminationCondition.feasible,
    }

    if termination not in acceptable:
        raise RuntimeError(
            "Model did not solve correctly. "
            f"Termination condition: {termination}"
        )

    # =====================================================
    # EXTRACT VALUES
    # =====================================================

    def var_vector(var_component) -> np.ndarray:
        return np.asarray(
            [
                pyo.value(var_component[t])
                for t in range(t_current)
            ],
            dtype=float,
        )

    return {
        "soc": var_vector(model.soc),
        "p_DA": var_vector(model.p_DA),
        "wp": var_vector(model.wp),
        "wp2b": var_vector(model.wp2b),
        "ed_discharged": var_vector(model.ed_discharged),
        "ed_charged": var_vector(model.ed_charged),
        "pb": var_vector(model.pb),
        "br2up": var_vector(model.br2up),
        "br2down": var_vector(model.br2down),
        "er2up": var_vector(model.er2up),
        "er2down": var_vector(model.er2down),
        "br3up": var_vector(model.br3up),
        "br3down": var_vector(model.br3down),
        "er3up": var_vector(model.er3up),
        "er3down": var_vector(model.er3down),
        "dcr": var_vector(model.dcr),
        "bess_cost": var_vector(model.bess_cost),
        "AG_cal": var_vector(model.AG_cal),
        "C_cal": var_vector(model.C_cal),
        "objective_value": float(pyo.value(model.objective)),
    }


# =========================================================
# EXCEL HELPERS
# =========================================================

def write_vector_sheet(
    writer: pd.ExcelWriter,
    sheet_name: str,
    values: List[float] | np.ndarray,
) -> None:
    pd.DataFrame(
        np.asarray(values).reshape(-1, 1)
    ).to_excel(
        writer,
        sheet_name=sheet_name,
        index=False,
        header=False,
    )


def write_scalar_sheet(
    writer: pd.ExcelWriter,
    sheet_name: str,
    value: float,
) -> None:
    pd.DataFrame([[value]]).to_excel(
        writer,
        sheet_name=sheet_name,
        index=False,
        header=False,
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    required_files = (
        PRICE_FILE,
        WIND_FILE,
        STROE_FILE,
        PRICE_BR2_FILE,
        RATIO_BR2_FILE,
        PRICE_ER2UP_FILE,
        PRICE_ER2DOWN_FILE,
        USE_ER2UP_FILE,
        USE_ER2DOWN_FILE,
        PRICE_ER3UP_FILE,
        PRICE_ER3DOWN_FILE,
        USE_ER3UP_FILE,
        USE_ER3DOWN_FILE,
    )

    for required_file in required_files:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Missing input file: {required_file.name}\n"
                "Put all required Excel files in the same folder as this script."
            )

    # =========================================================
    # LOAD THE 8760-HOUR INPUT SERIES
    #
    # The same historical year is repeated for each operating year,
    # while battery age, SoC and usable capacity continue across years.
    # =========================================================

    price_all = read_8760_series(PRICE_FILE)

    # Input file is already scaled to 308 MW.
    wind_all = read_8760_series(WIND_FILE)

    price_br2_all = read_8760_series(PRICE_BR2_FILE)
    ratio_br2_all = read_8760_series(RATIO_BR2_FILE)
    price_er2up_all = read_8760_series(PRICE_ER2UP_FILE)
    price_er2down_all = read_8760_series(PRICE_ER2DOWN_FILE)
    use_er2up_all = read_8760_series(USE_ER2UP_FILE)
    use_er2down_all = read_8760_series(USE_ER2DOWN_FILE)

    # FASBATT ER3 treatment: negative prices are replaced by zero;
    # expected-use values above 1 p.u. are capped at 1, and missing
    # values have already been read as zero by read_8760_series().
    price_er3up_all = np.maximum(0.0, read_8760_series(PRICE_ER3UP_FILE))
    price_er3down_all = np.maximum(0.0, read_8760_series(PRICE_ER3DOWN_FILE))
    use_er3up_all = np.minimum(1.0, read_8760_series(USE_ER3UP_FILE))
    use_er3down_all = np.minimum(1.0, read_8760_series(USE_ER3DOWN_FILE))

    x_stroe, y_stroe = read_stroe_data(STROE_FILE)

    # =========================================================
    # RESULT CONTAINERS
    # =========================================================

    soc_all: List[float] = []
    pda_all: List[float] = []
    wp_all: List[float] = []
    wp2b_all: List[float] = []
    grid_charge_all: List[float] = []
    total_charge_all: List[float] = []
    discharge_all: List[float] = []

    br2up_all: List[float] = []
    br2down_all: List[float] = []
    er2up_all: List[float] = []
    er2down_all: List[float] = []

    br3up_all: List[float] = []
    br3down_all: List[float] = []
    er3up_all: List[float] = []
    er3down_all: List[float] = []

    dcr_all: List[float] = []
    bess_cost_all: List[float] = []
    cal_damage_hourly_all: List[float] = []
    cal_cost_hourly_all: List[float] = []

    capacity_hourly_all: List[float] = []
    soc_min_hourly_all: List[float] = []
    soc_max_hourly_all: List[float] = []

    hour_year_all: List[int] = []
    hour_in_year_all: List[int] = []
    global_hour_all: List[int] = []

    profit_daily_all: List[float] = []
    da_revenue_daily_all: List[float] = []
    reserve_capacity_revenue_daily_all: List[float] = []
    regulation_energy_revenue_daily_all: List[float] = []
    mfrr_energy_revenue_daily_all: List[float] = []
    objective_daily_all: List[float] = []

    day_year_all: List[int] = []
    day_in_year_all: List[int] = []
    global_day_all: List[int] = []

    rf_damage_weekly_all: List[float] = []
    rf_cost_weekly_all: List[float] = []
    rf_damage_incremental_weekly_all: List[float] = []
    rf_cost_incremental_weekly_all: List[float] = []

    opt_cycle_aging_daily_all: List[float] = []
    opt_cycle_aging_cumulative_all: List[float] = []
    opt_cycle_aging_weekly_all: List[float] = []
    opt_cycle_aging_weekly_cumulative_all: List[float] = []

    cal_damage_daily_all: List[float] = []
    cal_cost_daily_all: List[float] = []
    cal_damage_cumulative_all: List[float] = []
    cal_cost_cumulative_all: List[float] = []

    cal_damage_weekly_all: List[float] = []
    cal_cost_weekly_all: List[float] = []
    cal_damage_weekly_cumulative_all: List[float] = []
    cal_cost_weekly_cumulative_all: List[float] = []

    week_year_all: List[int] = []
    week_day_all: List[int] = []
    week_global_day_all: List[int] = []
    capacity_weekly_all: List[float] = []
    soh_weekly_all: List[float] = []

    # Annual summary.
    year_all: List[int] = []
    year_days_simulated_all: List[int] = []
    year_initial_capacity_all: List[float] = []
    year_final_capacity_all: List[float] = []
    year_initial_soh_all: List[float] = []
    year_final_soh_all: List[float] = []

    year_da_revenue_all: List[float] = []
    year_reserve_capacity_revenue_all: List[float] = []
    year_regulation_energy_revenue_all: List[float] = []
    year_mfrr_energy_revenue_all: List[float] = []
    year_total_profit_all: List[float] = []
    year_objective_all: List[float] = []

    year_discharge_all: List[float] = []
    year_wind_charge_all: List[float] = []
    year_efc_all: List[float] = []

    year_opt_cycle_cost_all: List[float] = []
    year_cal_cost_objective_all: List[float] = []
    year_cal_loss_objective_all: List[float] = []

    year_rf_damage_increment_all: List[float] = []
    year_rf_cost_increment_all: List[float] = []
    year_rf_loss_increment_all: List[float] = []
    year_cal_loss_increment_all: List[float] = []
    year_cal_cost_postprocess_all: List[float] = []
    year_capacity_loss_increment_all: List[float] = []

    # =========================================================
    # INITIAL STATE
    # =========================================================

    soc_initial_ref = float(INITIAL_SOC)
    battery_capacity_ref = float(SIMULATION_INITIAL_CAPACITY)

    stop_simulation = False

    # =========================================================
    # LOOP OVER OPERATING YEARS
    # =========================================================

    for operating_year in range(1, MAX_YEARS + 1):

        print("\n########################################################")
        print(f"STARTING OPERATING YEAR {operating_year}")
        print(f"Initial capacity = {battery_capacity_ref} MWh")
        print(
            "Initial SOH = "
            f"{battery_capacity_ref / SOCBESS_INITIAL}"
        )
        print(f"Initial SOC = {soc_initial_ref} MWh")
        print("########################################################")

        # Battery age at the beginning of this year.
        commissioning_hours = (operating_year - 1) * 8760.0

        year_initial_capacity = float(battery_capacity_ref)
        year_initial_soh = year_initial_capacity / SOCBESS_INITIAL

        day_idx_start = len(profit_daily_all)
        hour_idx_start = len(soc_all)

        rf_damage_start = (
            0.0 if not rf_damage_weekly_all
            else rf_damage_weekly_all[-1]
        )
        rf_cost_start = (
            0.0 if not rf_cost_weekly_all
            else rf_cost_weekly_all[-1]
        )
        cal_damage_start = (
            0.0 if not cal_damage_weekly_cumulative_all
            else cal_damage_weekly_cumulative_all[-1]
        )
        cal_cost_start = (
            0.0 if not cal_cost_weekly_cumulative_all
            else cal_cost_weekly_cumulative_all[-1]
        )

        days_simulated_this_year = 0

        # =====================================================
        # LOOP OVER DAYS
        # =====================================================

        for d in range(1, 366):

            print("================================")
            print(f"Running year {operating_year}, day {d}")
            print("================================")

            current_capacity = float(battery_capacity_ref)

            soc_min_current = 0.10 * current_capacity
            soc_max_current = current_capacity

            # Keep the rolling state feasible after degradation.
            soc_initial_ref = min(soc_initial_ref, soc_max_current)
            soc_initial_ref = max(soc_initial_ref, soc_min_current)
            soc_initial = float(soc_initial_ref)

            print(f"Current battery capacity = {current_capacity}")
            print(f"SOC min = {soc_min_current}")
            print(f"SOC max = {soc_max_current}")
            print(f"SOC initial = {soc_initial}")

            t_current = 24 if d == 365 else T

            # Input data are repeated annually.
            start_idx = (d - 1) * IMPLEMENT
            end_idx = start_idx + t_current

            price = price_all[start_idx:end_idx]
            wind = wind_all[start_idx:end_idx]

            price_br2 = price_br2_all[start_idx:end_idx]
            ratio_br2 = ratio_br2_all[start_idx:end_idx]
            price_er2up = price_er2up_all[start_idx:end_idx]
            price_er2down = price_er2down_all[start_idx:end_idx]
            use_er2up = use_er2up_all[start_idx:end_idx]
            use_er2down = use_er2down_all[start_idx:end_idx]

            price_er3up = price_er3up_all[start_idx:end_idx]
            price_er3down = price_er3down_all[start_idx:end_idx]
            use_er3up = use_er3up_all[start_idx:end_idx]
            use_er3down = use_er3down_all[start_idx:end_idx]

            series = (
                price,
                wind,
                price_br2,
                ratio_br2,
                price_er2up,
                price_er2down,
                use_er2up,
                use_er2down,
                price_er3up,
                price_er3down,
                use_er3up,
                use_er3down,
            )

            if any(len(x) != t_current for x in series):
                raise IndexError(
                    "Look-ahead window exceeds input data "
                    f"on year={operating_year}, day={d}."
                )

            auxiliar1 = np.arange(
                start_idx + 1,
                start_idx + t_current + 1,
                dtype=float,
            )

            day_result = solve_daily_model(
                price=price,
                wind=wind,
                price_br2=price_br2,
                ratio_br2=ratio_br2,
                price_er2up=price_er2up,
                price_er2down=price_er2down,
                use_er2up=use_er2up,
                use_er2down=use_er2down,
                price_er3up=price_er3up,
                price_er3down=price_er3down,
                use_er3up=use_er3up,
                use_er3down=use_er3down,
                t_current=t_current,
                current_capacity=current_capacity,
                soc_initial=soc_initial,
                commissioning_hours=commissioning_hours,
                auxiliar1=auxiliar1,
            )

            # =================================================
            # SAVE IMPLEMENTED HOURS
            # =================================================

            save_hours = min(IMPLEMENT, t_current)
            implemented = slice(0, save_hours)

            soc_impl = np.asarray(day_result["soc"][implemented], dtype=float)
            pda_impl = np.asarray(day_result["p_DA"][implemented], dtype=float)
            wp_impl = np.asarray(day_result["wp"][implemented], dtype=float)
            wp2b_impl = np.asarray(day_result["wp2b"][implemented], dtype=float)
            discharge_impl = np.asarray(
                day_result["ed_discharged"][implemented],
                dtype=float,
            )

            br2up_impl = np.asarray(day_result["br2up"][implemented], dtype=float)
            br2down_impl = np.asarray(day_result["br2down"][implemented], dtype=float)
            er2up_impl = np.asarray(day_result["er2up"][implemented], dtype=float)
            er2down_impl = np.asarray(day_result["er2down"][implemented], dtype=float)

            br3up_impl = np.asarray(day_result["br3up"][implemented], dtype=float)
            br3down_impl = np.asarray(day_result["br3down"][implemented], dtype=float)
            er3up_impl = np.asarray(day_result["er3up"][implemented], dtype=float)
            er3down_impl = np.asarray(day_result["er3down"][implemented], dtype=float)
            grid_charge_impl = np.asarray(
                day_result["ed_charged"][implemented], dtype=float
            )
            # Gross charging terms before ETA; not physical subhourly charging.
            # One-hour intervals: values are MWh.
            total_charge_impl = (
                wp2b_impl + grid_charge_impl + er2down_impl + er3down_impl
            )
            grid_charge_all.extend(grid_charge_impl.tolist())
            total_charge_all.extend(total_charge_impl.tolist())


            dcr_impl = np.asarray(day_result["dcr"][implemented], dtype=float)
            bess_cost_impl = np.asarray(
                day_result["bess_cost"][implemented],
                dtype=float,
            )
            ag_cal_impl = np.asarray(day_result["AG_cal"][implemented], dtype=float)
            c_cal_impl = np.asarray(day_result["C_cal"][implemented], dtype=float)

            soc_all.extend(soc_impl.tolist())
            pda_all.extend(pda_impl.tolist())
            wp_all.extend(wp_impl.tolist())
            wp2b_all.extend(wp2b_impl.tolist())
            discharge_all.extend(discharge_impl.tolist())

            br2up_all.extend(br2up_impl.tolist())
            br2down_all.extend(br2down_impl.tolist())
            er2up_all.extend(er2up_impl.tolist())
            er2down_all.extend(er2down_impl.tolist())

            br3up_all.extend(br3up_impl.tolist())
            br3down_all.extend(br3down_impl.tolist())
            er3up_all.extend(er3up_impl.tolist())
            er3down_all.extend(er3down_impl.tolist())

            dcr_all.extend(dcr_impl.tolist())
            bess_cost_all.extend(bess_cost_impl.tolist())
            cal_damage_hourly_all.extend(ag_cal_impl.tolist())
            cal_cost_hourly_all.extend(c_cal_impl.tolist())

            capacity_hourly_all.extend([current_capacity] * save_hours)
            soc_min_hourly_all.extend([soc_min_current] * save_hours)
            soc_max_hourly_all.extend([soc_max_current] * save_hours)

            local_hour_start = (d - 1) * IMPLEMENT + 1

            for h in range(save_hours):
                hour_year_all.append(operating_year)
                hour_in_year_all.append(local_hour_start + h)
                global_hour_all.append(
                    (operating_year - 1) * 8760
                    + local_hour_start
                    + h
                )

            # =================================================
            # DAILY AGING / REVENUES
            # =================================================

            daily_opt_cycle_aging = float(np.sum(bess_cost_impl))
            opt_cycle_aging_daily_all.append(daily_opt_cycle_aging)
            opt_cycle_aging_cumulative_all.append(
                float(np.sum(opt_cycle_aging_daily_all))
            )

            daily_cal_damage = float(np.sum(ag_cal_impl))
            daily_cal_cost = float(np.sum(c_cal_impl))

            cal_damage_daily_all.append(daily_cal_damage)
            cal_cost_daily_all.append(daily_cal_cost)

            cal_damage_cumulative_all.append(
                float(np.sum(cal_damage_daily_all))
            )
            cal_cost_cumulative_all.append(
                float(np.sum(cal_cost_daily_all))
            )

            daily_da_revenue = float(
                np.dot(price[:save_hours], pda_impl)
            )

            daily_reserve_capacity_revenue = float(
                np.sum(
                    price_br2[:save_hours]
                    * (br2up_impl + br2down_impl)
                )
            )

            daily_regulation_energy_revenue = float(
                np.sum(
                    price_er2up[:save_hours] * er2up_impl
                    - price_er2down[:save_hours] * er2down_impl
                )
            )

            # FASBATT mFRR/ER3 pays activated energy only; there is no
            # separate BR3 capacity-payment term in its objective.
            daily_mfrr_energy_revenue = float(
                np.sum(
                    price_er3up[:save_hours] * er3up_impl
                    - price_er3down[:save_hours] * er3down_impl
                )
            )

            daily_profit = (
                daily_da_revenue
                + daily_reserve_capacity_revenue
                + daily_regulation_energy_revenue
                + daily_mfrr_energy_revenue
            )

            daily_objective = float(
                np.sum(
                    price[:save_hours] * pda_impl
                    + price_br2[:save_hours] * (
                        br2up_impl + br2down_impl
                    )
                    + price_er2up[:save_hours] * er2up_impl
                    - price_er2down[:save_hours] * er2down_impl
                    + price_er3up[:save_hours] * er3up_impl
                    - price_er3down[:save_hours] * er3down_impl
                    - bess_cost_impl
                    - c_cal_impl
                )
            )

            da_revenue_daily_all.append(daily_da_revenue)
            reserve_capacity_revenue_daily_all.append(
                daily_reserve_capacity_revenue
            )
            regulation_energy_revenue_daily_all.append(
                daily_regulation_energy_revenue
            )
            mfrr_energy_revenue_daily_all.append(
                daily_mfrr_energy_revenue
            )
            profit_daily_all.append(daily_profit)
            objective_daily_all.append(daily_objective)

            day_year_all.append(operating_year)
            day_in_year_all.append(d)
            global_day_all.append(
                (operating_year - 1) * 365 + d
            )

            days_simulated_this_year = d

            # SoC carries to the next rolling-horizon day.
            soc_initial_ref = float(soc_impl[-1])

            print(f"SOC next day = {soc_initial_ref}")

            # =================================================
            # WEEKLY RAINFLOW + CALENDAR CAPACITY UPDATE
            # =================================================

            if d % DAYS_PER_WEEK == 0 or d == 365:

                print("---------- Weekly degradation update ----------")

                # Complete multi-year SoC history.
                # Initial SoC is included only once, at simulation start.
                soc_history = np.concatenate(
                    [
                        np.array([INITIAL_SOC], dtype=float),
                        np.asarray(soc_all, dtype=float),
                    ]
                )

                total_rf_damage, total_rf_cost = (
                    rainflow_damage_and_cost(
                        soc_history,
                        SOCBESS_INITIAL,
                        x_stroe,
                        y_stroe,
                        ECAP_COST,
                    )
                )

                previous_rf_damage = (
                    0.0 if not rf_damage_weekly_all
                    else rf_damage_weekly_all[-1]
                )
                previous_rf_cost = (
                    0.0 if not rf_cost_weekly_all
                    else rf_cost_weekly_all[-1]
                )

                weekly_rf_damage = (
                    total_rf_damage - previous_rf_damage
                )
                weekly_rf_cost = (
                    total_rf_cost - previous_rf_cost
                )

                # Local (within-year) week range.
                week_start_day = (
                    ((d - 1) // DAYS_PER_WEEK) * DAYS_PER_WEEK + 1
                )
                week_start_hour = (
                    (week_start_day - 1) * IMPLEMENT + 1
                )
                week_end_hour = d * IMPLEMENT

                # Global multi-year slice.
                global_week_start_hour = (
                    (operating_year - 1) * 8760
                    + week_start_hour
                )
                global_week_end_hour = min(
                    (operating_year - 1) * 8760
                    + week_end_hour,
                    len(soc_all),
                )

                soc_week = np.asarray(
                    soc_all[
                        global_week_start_hour - 1:
                        global_week_end_hour
                    ],
                    dtype=float,
                )

                weekly_cal_percent = (
                    calendar_polynomial_hourly_weekly_percent(
                        soc_week,
                        SOCBESS_INITIAL,
                        week_start_hour,
                        commissioning_hours=commissioning_hours,
                        hours_per_month=HOURS_PER_MONTH,
                    )
                )

                weekly_cal_damage = (
                    SOCBESS_INITIAL
                    * weekly_cal_percent
                    / 100.0
                )

                previous_cumulative_cal_damage = (
                    0.0
                    if not cal_damage_weekly_cumulative_all
                    else cal_damage_weekly_cumulative_all[-1]
                )

                cumulative_cal_damage = (
                    previous_cumulative_cal_damage
                    + weekly_cal_damage
                )

                weekly_cal_cost = (
                    CAL_COST_FACTOR
                    * weekly_cal_damage
                )

                previous_cumulative_cal_cost = (
                    0.0
                    if not cal_cost_weekly_cumulative_all
                    else cal_cost_weekly_cumulative_all[-1]
                )

                cumulative_cal_cost = (
                    previous_cumulative_cal_cost
                    + weekly_cal_cost
                )

                cal_damage_weekly_all.append(weekly_cal_damage)
                cal_cost_weekly_all.append(weekly_cal_cost)
                cal_damage_weekly_cumulative_all.append(
                    cumulative_cal_damage
                )
                cal_cost_weekly_cumulative_all.append(
                    cumulative_cal_cost
                )

                # Capacity is always calculated relative to the original
                # rated capacity, not relative to the previous week's capacity.
                rf_capacity_loss_mwh = (
                    0.2*total_rf_damage * SOCBESS_INITIAL
                )

                cal_capacity_loss_mwh = cumulative_cal_damage

                if INCLUDE_CALENDAR_IN_CAPACITY_UPDATE:
                    total_capacity_loss_mwh = (
                        rf_capacity_loss_mwh
                        + cal_capacity_loss_mwh
                    )
                else:
                    total_capacity_loss_mwh = (
                        rf_capacity_loss_mwh
                    )

                new_capacity = (
                    SIMULATION_INITIAL_CAPACITY
                    - total_capacity_loss_mwh
                )

                new_capacity = max(
                    new_capacity,
                    0.10 * SOCBESS_INITIAL,
                )

                battery_capacity_ref = float(new_capacity)

                soh = (
                    battery_capacity_ref
                    / SOCBESS_INITIAL
                )

                week_year_all.append(operating_year)
                week_day_all.append(d)
                week_global_day_all.append(
                    (operating_year - 1) * 365 + d
                )
                capacity_weekly_all.append(
                    battery_capacity_ref
                )
                soh_weekly_all.append(soh)

                rf_damage_incremental_weekly_all.append(
                    weekly_rf_damage
                )
                rf_cost_incremental_weekly_all.append(
                    weekly_rf_cost
                )
                rf_damage_weekly_all.append(total_rf_damage)
                rf_cost_weekly_all.append(total_rf_cost)

                global_week_start_day = (
                    (operating_year - 1) * 365
                    + week_start_day
                )
                global_day = (
                    (operating_year - 1) * 365
                    + d
                )

                weekly_opt_cycle_aging = float(
                    np.sum(
                        opt_cycle_aging_daily_all[
                            global_week_start_day - 1:
                            global_day
                        ]
                    )
                )

                cumulative_opt_cycle_aging = float(
                    np.sum(
                        opt_cycle_aging_daily_all[
                            :global_day
                        ]
                    )
                )

                opt_cycle_aging_weekly_all.append(
                    weekly_opt_cycle_aging
                )
                opt_cycle_aging_weekly_cumulative_all.append(
                    cumulative_opt_cycle_aging
                )

                print(f"Day = {d}")
                print(f"Weekly RF damage = {weekly_rf_damage}")
                print(f"Cumulative RF damage = {total_rf_damage}")
                print(f"Weekly RF cost = {weekly_rf_cost}")
                print(f"Cumulative RF cost = {total_rf_cost}")
                print(
                    "Weekly calendar aging = "
                    f"{weekly_cal_percent} %"
                )
                print(
                    "Weekly calendar capacity loss = "
                    f"{weekly_cal_damage} MWh"
                )
                print(
                    "Cumulative calendar capacity loss = "
                    f"{cumulative_cal_damage} MWh"
                )
                print(
                    "Updated battery capacity for next week = "
                    f"{battery_capacity_ref}"
                )
                print(f"SOH = {soh}")

                # Keep next-day SoC feasible after capacity reduction.
                soc_initial_ref = min(
                    soc_initial_ref,
                    battery_capacity_ref,
                )
                soc_initial_ref = max(
                    soc_initial_ref,
                    0.10 * battery_capacity_ref,
                )

                if soh <= SOH_STOP_THRESHOLD:
                    print(
                        "SOH threshold reached: "
                        f"{soh} <= {SOH_STOP_THRESHOLD}"
                    )
                    stop_simulation = True

            if stop_simulation:
                break

        # =====================================================
        # ANNUAL SUMMARY
        # =====================================================

        day_idx_end = len(profit_daily_all)
        hour_idx_end = len(soc_all)

        annual_da_revenue = float(
            np.sum(da_revenue_daily_all[day_idx_start:day_idx_end])
        )
        annual_reserve_capacity_revenue = float(
            np.sum(
                reserve_capacity_revenue_daily_all[
                    day_idx_start:day_idx_end
                ]
            )
        )
        annual_regulation_energy_revenue = float(
            np.sum(
                regulation_energy_revenue_daily_all[
                    day_idx_start:day_idx_end
                ]
            )
        )
        annual_mfrr_energy_revenue = float(
            np.sum(
                mfrr_energy_revenue_daily_all[
                    day_idx_start:day_idx_end
                ]
            )
        )
        annual_total_profit = float(
            np.sum(profit_daily_all[day_idx_start:day_idx_end])
        )
        annual_objective = float(
            np.sum(objective_daily_all[day_idx_start:day_idx_end])
        )

        annual_discharge = float(
            np.sum(discharge_all[hour_idx_start:hour_idx_end])
        )
        annual_wind_charge = float(
            np.sum(wp2b_all[hour_idx_start:hour_idx_end])
        )
        annual_efc = annual_discharge / SOCBESS_INITIAL

        annual_opt_cycle_cost = float(
            np.sum(
                opt_cycle_aging_daily_all[
                    day_idx_start:day_idx_end
                ]
            )
        )
        annual_cal_cost_obj = float(
            np.sum(
                cal_cost_daily_all[
                    day_idx_start:day_idx_end
                ]
            )
        )
        annual_cal_loss_obj = float(
            np.sum(
                cal_damage_daily_all[
                    day_idx_start:day_idx_end
                ]
            )
        )

        rf_damage_end = (
            rf_damage_start
            if not rf_damage_weekly_all
            else rf_damage_weekly_all[-1]
        )
        rf_cost_end = (
            rf_cost_start
            if not rf_cost_weekly_all
            else rf_cost_weekly_all[-1]
        )
        cal_damage_end = (
            cal_damage_start
            if not cal_damage_weekly_cumulative_all
            else cal_damage_weekly_cumulative_all[-1]
        )
        cal_cost_end = (
            cal_cost_start
            if not cal_cost_weekly_cumulative_all
            else cal_cost_weekly_cumulative_all[-1]
        )

        annual_rf_damage_increment = (
            rf_damage_end - rf_damage_start
        )
        annual_rf_cost_increment = (
            rf_cost_end - rf_cost_start
        )
        annual_rf_loss_increment = (
            0.20 *annual_rf_damage_increment
            * SOCBESS_INITIAL
        )
        annual_cal_loss_increment = (
            cal_damage_end - cal_damage_start
        )
        annual_cal_cost_postprocess = (
            cal_cost_end - cal_cost_start
        )

        annual_capacity_loss_increment = (
            annual_rf_loss_increment
            + (
                annual_cal_loss_increment
                if INCLUDE_CALENDAR_IN_CAPACITY_UPDATE
                else 0.0
            )
        )

        year_all.append(operating_year)
        year_days_simulated_all.append(days_simulated_this_year)
        year_initial_capacity_all.append(year_initial_capacity)
        year_final_capacity_all.append(battery_capacity_ref)
        year_initial_soh_all.append(year_initial_soh)
        year_final_soh_all.append(
            battery_capacity_ref / SOCBESS_INITIAL
        )

        year_da_revenue_all.append(annual_da_revenue)
        year_reserve_capacity_revenue_all.append(
            annual_reserve_capacity_revenue
        )
        year_regulation_energy_revenue_all.append(
            annual_regulation_energy_revenue
        )
        year_mfrr_energy_revenue_all.append(
            annual_mfrr_energy_revenue
        )
        year_total_profit_all.append(annual_total_profit)
        year_objective_all.append(annual_objective)

        year_discharge_all.append(annual_discharge)
        year_wind_charge_all.append(annual_wind_charge)
        year_efc_all.append(annual_efc)

        year_opt_cycle_cost_all.append(annual_opt_cycle_cost)
        year_cal_cost_objective_all.append(annual_cal_cost_obj)
        year_cal_loss_objective_all.append(annual_cal_loss_obj)

        year_rf_damage_increment_all.append(
            annual_rf_damage_increment
        )
        year_rf_cost_increment_all.append(
            annual_rf_cost_increment
        )
        year_rf_loss_increment_all.append(
            annual_rf_loss_increment
        )
        year_cal_loss_increment_all.append(
            annual_cal_loss_increment
        )
        year_cal_cost_postprocess_all.append(
            annual_cal_cost_postprocess
        )
        year_capacity_loss_increment_all.append(
            annual_capacity_loss_increment
        )

        print(
            f"YEAR {operating_year} COMPLETE | "
            f"Final capacity = {battery_capacity_ref:.6f} MWh | "
            f"Final SOH = {battery_capacity_ref / SOCBESS_INITIAL:.6f}"
        )

        if stop_simulation:
            print(
                "Simulation stopped because SOH reached "
                f"the {SOH_STOP_THRESHOLD:.0%} threshold."
            )
            break

    # =========================================================
    # TOTALS
    # =========================================================

    total_profit = float(np.sum(profit_daily_all))
    total_da_revenue = float(np.sum(da_revenue_daily_all))
    total_reserve_capacity_revenue = float(
        np.sum(reserve_capacity_revenue_daily_all)
    )
    total_regulation_energy_revenue = float(
        np.sum(regulation_energy_revenue_daily_all)
    )
    total_mfrr_energy_revenue = float(
        np.sum(mfrr_energy_revenue_daily_all)
    )
    total_objective = float(np.sum(objective_daily_all))

    total_discharge_mwh = float(np.sum(discharge_all))
    total_wind_charge_mwh = float(np.sum(wp2b_all))
    equivalent_full_cycles = (
        total_discharge_mwh / SOCBESS_INITIAL
    )

    total_opt_cycle_aging = float(
        np.sum(opt_cycle_aging_daily_all)
    )

    total_cal_damage = (
        0.0
        if not cal_damage_weekly_cumulative_all
        else cal_damage_weekly_cumulative_all[-1]
    )
    total_cal_cost = (
        0.0
        if not cal_cost_weekly_cumulative_all
        else cal_cost_weekly_cumulative_all[-1]
    )
    total_cal_fraction = (
        total_cal_damage / SOCBESS_INITIAL
    )

    total_cal_cost_objective = float(
        np.sum(cal_cost_daily_all)
    )
    total_cal_loss_objective = float(
        np.sum(cal_damage_daily_all)
    )

    total_rf_cost = (
        0.0 if not rf_cost_weekly_all
        else rf_cost_weekly_all[-1]
    )
    total_rf_damage = (
        0.0 if not rf_damage_weekly_all
        else rf_damage_weekly_all[-1]
    )
    total_rf_loss_mwh = (
        0.20 *total_rf_damage * SOCBESS_INITIAL
    )

    if INCLUDE_CALENDAR_IN_CAPACITY_UPDATE:
        total_capacity_loss_mwh = (
            total_rf_loss_mwh + total_cal_damage
        )
    else:
        total_capacity_loss_mwh = total_rf_loss_mwh

    total_capacity_loss_fraction = (
        total_capacity_loss_mwh / SOCBESS_INITIAL
    )

    final_capacity = float(battery_capacity_ref)
    final_soh = final_capacity / SOCBESS_INITIAL

    print("================================")
    print(f"YEARS SIMULATED = {len(year_all)}")
    print(f"TOTAL PROFIT = {total_profit}")
    print(f"TOTAL DA REVENUE = {total_da_revenue}")
    print(
        "TOTAL R2 CAPACITY REVENUE = "
        f"{total_reserve_capacity_revenue}"
    )
    print(
        "TOTAL aFRR/R2 REGULATION ENERGY REVENUE = "
        f"{total_regulation_energy_revenue}"
    )
    print(
        "TOTAL mFRR/R3 REGULATION ENERGY REVENUE = "
        f"{total_mfrr_energy_revenue}"
    )
    print(f"TOTAL OBJECTIVE = {total_objective}")
    print(f"FINAL BATTERY CAPACITY = {final_capacity}")
    print(f"FINAL SOH = {final_soh}")
    print("================================")

    # =========================================================
    # EXPORT RESULTS
    # =========================================================

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl",
    ) as writer:

        # Hourly.
        write_vector_sheet(writer, "SOC", soc_all)
        write_vector_sheet(writer, "HOUR_YEAR", hour_year_all)
        write_vector_sheet(writer, "HOUR_IN_YEAR", hour_in_year_all)
        write_vector_sheet(writer, "GLOBAL_HOUR", global_hour_all)

        write_vector_sheet(writer, "P_DA", pda_all)
        write_vector_sheet(writer, "WP", wp_all)
        write_vector_sheet(writer, "WP2B", wp2b_all)
        write_vector_sheet(writer, "GRID_CHARGE_MWH", grid_charge_all)
        write_vector_sheet(writer, "TOTAL_CHARGE_MWH", total_charge_all)
        write_scalar_sheet(
            writer, "TOTAL_CHARGING_ENERGY_MWH",
            float(np.sum(total_charge_all)),
        )

        write_vector_sheet(writer, "DISCHARGE", discharge_all)

        write_vector_sheet(writer, "BR2_UP", br2up_all)
        write_vector_sheet(writer, "BR2_DOWN", br2down_all)
        write_vector_sheet(writer, "ER2_UP", er2up_all)
        write_vector_sheet(writer, "ER2_DOWN", er2down_all)

        write_vector_sheet(writer, "BR3_UP", br3up_all)
        write_vector_sheet(writer, "BR3_DOWN", br3down_all)
        write_vector_sheet(writer, "ER3_UP", er3up_all)
        write_vector_sheet(writer, "ER3_DOWN", er3down_all)

        write_vector_sheet(writer, "DCR", dcr_all)
        write_vector_sheet(writer, "BESS_COST", bess_cost_all)

        write_vector_sheet(
            writer,
            "CAL_DAMAGE_HOURLY",
            cal_damage_hourly_all,
        )
        write_vector_sheet(
            writer,
            "CAL_COST_HOURLY",
            cal_cost_hourly_all,
        )

        write_vector_sheet(
            writer,
            "CAPACITY_HOURLY",
            capacity_hourly_all,
        )
        write_vector_sheet(
            writer,
            "SOC_MIN_HOURLY",
            soc_min_hourly_all,
        )
        write_vector_sheet(
            writer,
            "SOC_MAX_HOURLY",
            soc_max_hourly_all,
        )

        # Daily.
        write_vector_sheet(writer, "DAY_YEAR", day_year_all)
        write_vector_sheet(writer, "DAY_IN_YEAR", day_in_year_all)
        write_vector_sheet(writer, "GLOBAL_DAY", global_day_all)

        write_vector_sheet(
            writer,
            "DA_REVENUE_DAILY",
            da_revenue_daily_all,
        )
        write_vector_sheet(
            writer,
            "R2_CAPACITY_REVENUE_DAILY",
            reserve_capacity_revenue_daily_all,
        )
        write_vector_sheet(
            writer,
            "R2_ENERGY_REVENUE_DAILY",
            regulation_energy_revenue_daily_all,
        )
        write_vector_sheet(
            writer,
            "R3_ENERGY_REVENUE_DAILY",
            mfrr_energy_revenue_daily_all,
        )
        write_vector_sheet(
            writer,
            "DAILY_PROFIT",
            profit_daily_all,
        )
        write_vector_sheet(
            writer,
            "DAILY_OBJECTIVE",
            objective_daily_all,
        )

        # Optimization aging.
        write_vector_sheet(
            writer,
            "OPT_CYCLE_AGING_DAILY",
            opt_cycle_aging_daily_all,
        )
        write_vector_sheet(
            writer,
            "OPT_CYCLE_AGING_CUM",
            opt_cycle_aging_cumulative_all,
        )
        write_vector_sheet(
            writer,
            "OPT_CYCLE_AGING_WEEKLY",
            opt_cycle_aging_weekly_all,
        )
        write_vector_sheet(
            writer,
            "OPT_CYCLE_AGING_WEEKLY_CUM",
            opt_cycle_aging_weekly_cumulative_all,
        )

        write_vector_sheet(
            writer,
            "CAL_DAMAGE_DAILY",
            cal_damage_daily_all,
        )
        write_vector_sheet(
            writer,
            "CAL_COST_DAILY",
            cal_cost_daily_all,
        )
        write_vector_sheet(
            writer,
            "CAL_DAMAGE_CUM",
            cal_damage_cumulative_all,
        )
        write_vector_sheet(
            writer,
            "CAL_COST_CUM",
            cal_cost_cumulative_all,
        )

        # Rainflow.
        write_vector_sheet(
            writer,
            "RF_COST_WEEKLY",
            rf_cost_incremental_weekly_all,
        )
        write_vector_sheet(
            writer,
            "RF_COST_CUMULATIVE",
            rf_cost_weekly_all,
        )
        write_vector_sheet(
            writer,
            "RF_DAMAGE_WEEKLY",
            rf_damage_incremental_weekly_all,
        )
        write_vector_sheet(
            writer,
            "RF_DAMAGE_CUMULATIVE",
            rf_damage_weekly_all,
        )

        # Weekly capacity.
        weekly_df = pd.DataFrame(
            {
                "Year": week_year_all,
                "DayInYear": week_day_all,
                "GlobalDay": week_global_day_all,
                "BatteryCapacity_MWh": capacity_weekly_all,
                "SOH": soh_weekly_all,
                "Weekly_RF_Damage":
                    rf_damage_incremental_weekly_all,
                "Cumulative_RF_Damage":
                    rf_damage_weekly_all,
                "Weekly_RF_Cost":
                    rf_cost_incremental_weekly_all,
                "Cumulative_RF_Cost":
                    rf_cost_weekly_all,
                "Weekly_Opt_Cycle_Aging":
                    opt_cycle_aging_weekly_all,
                "Cumulative_Opt_Cycle_Aging":
                    opt_cycle_aging_weekly_cumulative_all,
                "Weekly_Calendar_Loss_MWh":
                    cal_damage_weekly_all,
                "Cumulative_Calendar_Loss_MWh":
                    cal_damage_weekly_cumulative_all,
                "Weekly_Calendar_Cost":
                    cal_cost_weekly_all,
                "Cumulative_Calendar_Cost":
                    cal_cost_weekly_cumulative_all,
            }
        )

        weekly_df.to_excel(
            writer,
            sheet_name="WEEKLY_CAPACITY",
            index=False,
        )

        # Annual summary.
        annual_df = pd.DataFrame(
            {
                "Year": year_all,
                "DaysSimulated": year_days_simulated_all,
                "InitialCapacity_MWh":
                    year_initial_capacity_all,
                "FinalCapacity_MWh":
                    year_final_capacity_all,
                "InitialSOH": year_initial_soh_all,
                "FinalSOH": year_final_soh_all,

                "DA_Revenue_EUR":
                    year_da_revenue_all,
                "R2_Capacity_Revenue_EUR":
                    year_reserve_capacity_revenue_all,
                "R2_Energy_Revenue_EUR":
                    year_regulation_energy_revenue_all,
                "R3_mFRR_Energy_Revenue_EUR":
                    year_mfrr_energy_revenue_all,
                "Total_Market_Revenue_EUR":
                    year_total_profit_all,
                "Objective_EUR":
                    year_objective_all,

                "Discharge_MWh":
                    year_discharge_all,
                "WindCharge_MWh":
                    year_wind_charge_all,
                "EFC":
                    year_efc_all,

                "OptCycleCost_EUR":
                    year_opt_cycle_cost_all,
                "CalCostObjective_EUR":
                    year_cal_cost_objective_all,
                "CalLossObjective_MWh":
                    year_cal_loss_objective_all,

                "RF_Damage_Increment":
                    year_rf_damage_increment_all,
                "RF_Cost_Increment_EUR":
                    year_rf_cost_increment_all,
                "RF_Loss_Increment_MWh":
                    year_rf_loss_increment_all,
                "CalLoss_Increment_MWh":
                    year_cal_loss_increment_all,
                "CalCostPostProcess_EUR":
                    year_cal_cost_postprocess_all,
                "CapacityLoss_Increment_MWh":
                    year_capacity_loss_increment_all,
            }
        )

        annual_df.to_excel(
            writer,
            sheet_name="ANNUAL_SUMMARY",
            index=False,
        )

        # Scalar totals.
        scalar_values = {
            "TOTAL_PROFIT": total_profit,
            "TOTAL_DA_REVENUE": total_da_revenue,
            "TOTAL_R2_CAPACITY_REVENUE":
                total_reserve_capacity_revenue,
            "TOTAL_R2_ENERGY_REVENUE":
                total_regulation_energy_revenue,
            "TOTAL_R3_MFRR_ENERGY_REVENUE":
                total_mfrr_energy_revenue,
            "TOTAL_OBJECTIVE": total_objective,
            "TOTAL_DISCHARGE_MWH": total_discharge_mwh,
            "TOTAL_WIND_CHARGE_MWH": total_wind_charge_mwh,
            "EQUIVALENT_FULL_CYCLES": equivalent_full_cycles,
            "TOTAL_OPT_CYCLE_AGING": total_opt_cycle_aging,
            "TOTAL_CAL_COST": total_cal_cost,
            "TOTAL_CAL_DAMAGE": total_cal_damage,
            "TOTAL_CAL_FRACTION": total_cal_fraction,
            "TOTAL_CAL_COST_OBJECTIVE":
                total_cal_cost_objective,
            "TOTAL_CAL_LOSS_OBJECTIVE":
                total_cal_loss_objective,
            "TOTAL_RF_COST": total_rf_cost,
            "TOTAL_RF_DAMAGE": total_rf_damage,
            "TOTAL_RF_LOSS_MWH": total_rf_loss_mwh,
            "TOTAL_CAPACITY_LOSS_MWH":
                total_capacity_loss_mwh,
            "TOTAL_CAPACITY_LOSS_FRACTION":
                total_capacity_loss_fraction,
            "FINAL_CAPACITY": final_capacity,
            "FINAL_SOH": final_soh,
        }

        for sheet_name, value in scalar_values.items():
            write_scalar_sheet(
                writer,
                sheet_name,
                float(value),
            )

    print("Results exported successfully!")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Years simulated: {len(year_all)}")
    print(f"Final SOH: {final_soh}")


if __name__ == "__main__":
    main()
