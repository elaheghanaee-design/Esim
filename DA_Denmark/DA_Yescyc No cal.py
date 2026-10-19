"""
DA-only multi-year BESS scheduling model with plot-only mode
Python / Pyomo / HiGHS translation of DA_only_multiyear_25y.jl

Main features
-------------
- 25-year multi-year simulation
- 48 h rolling horizon, 24 h implemented
- Day-ahead market scheduling
- Wind + BESS operation
- No grid charging
- Piecewise-linear cycle-aging cost inside MILP
- Linear calendar-aging cost inside MILP
- Rainflow cycle-aging post-processing
- Nonlinear calendar-aging post-processing
- Weekly usable-capacity update
- Annual summaries and Excel export

Install:
    pip install -r requirements.txt

Run:
    python main.py

Place these Excel files in the same folder as main.py:
    priceSPOT.xlsx
    WindFarm.xlsx
    data_stroe2017.xlsx
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import rainflow
import matplotlib.pyplot as plt


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

PRICE_FILE = BASE_DIR / "priceSPOT.xlsx"
WIND_FILE = BASE_DIR / "WindFarm_HyDesign_308MW.xlsx"
STROE_FILE = BASE_DIR / "data_stroe2017.xlsx"

OUTPUT_FILE = BASE_DIR / "results_multiyear_Den only cyc.xlsx"


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
    """
    Read one Excel column.

    Example:
        column="C", start_row=2, nrows=8760
    corresponds to Excel range C2:C8761.
    """
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


def read_stroe_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    
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
    matching the original Julia implementation.
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

NODE_CAPACITY_G = 300.0
NODE_CAPACITY_C = 00.0

WP_MAX = 308.0
PBESS = 20.0

RATED_CAPACITY = 60.0
SOCBESS_INITIAL = RATED_CAPACITY  # rated/reference capacity used by aging equations
INITIAL_SOC = 0.5 * RATED_CAPACITY  # 30 MWh = 50% initial SoC

ETA = 0.86

MAX_YEARS = 25
SOH_STOP_THRESHOLD = 0.80

SIMULATION_INITIAL_CAPACITY = RATED_CAPACITY

ECAP_COST = 6_200_000.0

DAYS_PER_WEEK = 7
HOURS_PER_MONTH = 720.0

CAL_SOC_COEFF = 0.0028
CAL_CAPACITY_COEFF = 0.001939
CAL_COST_FACTOR = ECAP_COST / (0.2 * SOCBESS_INITIAL)

####################################
#for considering cal aging
CALENDAR_COST_WEIGHT = 0.0

INCLUDE_CALENDAR_IN_CAPACITY_UPDATE = False
####################################

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


# =========================================================
# RESULT CONTAINER
# =========================================================

@dataclass
class Results:
    soc_all: List[float]
    pda_all: List[float]
    wp_all: List[float]
    wp2b_all: List[float]
    discharge_all: List[float]
    charged_all: List[float]
    pb_all: List[float]
    aux1_all: List[float]
    ud_all: List[float]
    y1_all: List[float]
    y2_all: List[float]
    y3_all: List[float]

    hour_year_all: List[int]
    hour_in_year_all: List[int]
    global_hour_all: List[int]

    day_year_all: List[int]
    day_in_year_all: List[int]
    global_day_all: List[int]

    profit_daily_all: List[float]
    objective_daily_all: List[float]

    dcr_all: List[float]
    bess_cost_all: List[float]

    capacity_hourly_all: List[float]
    soc_min_hourly_all: List[float]
    soc_max_hourly_all: List[float]

    week_year_all: List[int]
    week_day_all: List[int]
    week_global_day_all: List[int]
    capacity_weekly_all: List[float]
    soh_weekly_all: List[float]

    rf_damage_weekly_all: List[float]
    rf_cost_weekly_all: List[float]

    rf_damage_incremental_weekly_all: List[float]
    rf_cost_incremental_weekly_all: List[float]

    opt_cycle_aging_daily_all: List[float]
    opt_cycle_aging_cumulative_all: List[float]

    opt_cycle_aging_weekly_all: List[float]
    opt_cycle_aging_weekly_cumulative_all: List[float]

    cal_damage_hourly_all: List[float]
    cal_cost_hourly_all: List[float]

    cal_damage_daily_all: List[float]
    cal_cost_daily_all: List[float]

    cal_damage_cumulative_all: List[float]
    cal_cost_cumulative_all: List[float]

    cal_damage_weekly_all: List[float]
    cal_cost_weekly_all: List[float]

    cal_damage_weekly_cumulative_all: List[float]
    cal_cost_weekly_cumulative_all: List[float]

    year_all: List[int]
    year_days_simulated_all: List[int]
    year_initial_capacity_all: List[float]
    year_final_capacity_all: List[float]
    year_initial_soh_all: List[float]
    year_final_soh_all: List[float]
    year_da_revenue_all: List[float]
    year_objective_all: List[float]
    year_discharge_all: List[float]
    year_wind_charge_all: List[float]
    year_efc_all: List[float]
    year_opt_cycle_cost_all: List[float]
    year_cal_cost_objective_all: List[float]
    year_cal_loss_objective_all: List[float]
    year_rf_damage_increment_all: List[float]
    year_rf_cost_increment_all: List[float]
    year_rf_loss_increment_all: List[float]
    year_cal_loss_increment_all: List[float]
    year_capacity_loss_increment_all: List[float]

    @classmethod
    def empty(cls) -> "Results":
        return cls(
            **{
                field_name: []
                for field_name in cls.__dataclass_fields__
            }
        )


# =========================================================
# DAILY PYOMO MILP
# =========================================================

def solve_daily_model(
    price: np.ndarray,
    wind: np.ndarray,
    t_current: int,
    current_capacity: float,
    soc_initial: float,
    commissioning_hours: float,
    auxiliar1: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    """
    Build and solve one rolling-horizon day with Pyomo + HiGHS.
    """

    model = pyo.ConcreteModel(name="DA_BESS")

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

    # Original Julia code fixes grid charging to zero.
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
        return (
            m.wp[t] + m.wp2b[t]
            <= float(wind[t])
        )

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
    # SOC DYNAMICS
    # =====================================================

    def soc_rule(m, t):
        total_charging = (
            m.wp2b[t]
            + m.ed_charged[t]
        )

        if t == 0:
            return (
                m.soc[t]
                == soc_initial
                + ETA * total_charging
                - m.ed_discharged[t]
            )

        return (
            m.soc[t]
            == m.soc[t - 1]
            + ETA * total_charging
            - m.ed_discharged[t]
        )

    model.soc_balance = pyo.Constraint(
        model.T,
        rule=soc_rule,
    )

        # =====================================================
    # DCR EQUATION
    # =====================================================

    def dcr_rule(m, t):
        return (
            m.dcr[t]
            == m.ed_discharged[t] / SOCBESS_INITIAL
        )

    model.dcr_constraint = pyo.Constraint(
        model.T,
        rule=dcr_rule,
    )
    # =====================================================
    # CONEJO FORMULATION
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
        return m.p_DA[t] <= NODE_CAPACITY_G

    model.grid_export = pyo.Constraint(
        model.T,
        rule=grid_export_rule,
    )

    def grid_import_rule(m, t):
        return m.p_DA[t] >= -NODE_CAPACITY_C

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
            - model.bess_cost[t]
            - CALENDAR_COST_WEIGHT*model.C_cal[t]
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

    # HiGHS options.
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

    y_values = np.asarray(
        [
            [
                pyo.value(model.y[t, j])
                for j in range(3)
            ]
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
        "aux1": var_vector(model.aux1),
        "dcr": var_vector(model.dcr),
        "bess_cost": var_vector(model.bess_cost),
        "ud": var_vector(model.ud),
        "y": y_values,
        "AG_cal": var_vector(model.AG_cal),
        "C_cal": var_vector(model.C_cal),
        "objective_value": float(pyo.value(model.objective)),
    }


# =========================================================
# EXCEL EXPORT HELPERS
# =========================================================

def write_vector_sheet(
    writer: pd.ExcelWriter,
    sheet_name: str,
    values: List[float] | List[int] | np.ndarray,
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


def export_results(
    results: Results,
    output_file: Path,
    totals: Dict[str, float],
) -> None:

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
    ) as writer:

        # Hourly
        write_vector_sheet(writer, "SOC", results.soc_all)
        write_vector_sheet(writer, "HOUR_YEAR", results.hour_year_all)
        write_vector_sheet(writer, "HOUR_IN_YEAR", results.hour_in_year_all)
        write_vector_sheet(writer, "GLOBAL_HOUR", results.global_hour_all)

        write_vector_sheet(writer, "P_DA", results.pda_all)
        write_vector_sheet(writer, "WP", results.wp_all)
        write_vector_sheet(writer, "WP2B", results.wp2b_all)
        write_vector_sheet(writer, "DISCHARGE", results.discharge_all)
        write_vector_sheet(writer, "GRID_CHARGE", results.charged_all)
        write_vector_sheet(writer, "PB", results.pb_all)
        write_vector_sheet(writer, "AUX1", results.aux1_all)
        write_vector_sheet(writer, "UD", results.ud_all)
        write_vector_sheet(writer, "Y1", results.y1_all)
        write_vector_sheet(writer, "Y2", results.y2_all)
        write_vector_sheet(writer, "Y3", results.y3_all)

        write_vector_sheet(writer, "DCR", results.dcr_all)
        write_vector_sheet(writer, "BESS_COST", results.bess_cost_all)

        write_vector_sheet(
            writer,
            "CAL_DAMAGE_HOURLY",
            results.cal_damage_hourly_all,
        )
        write_vector_sheet(
            writer,
            "CAL_COST_HOURLY",
            results.cal_cost_hourly_all,
        )

        write_vector_sheet(
            writer,
            "CAPACITY_HOURLY",
            results.capacity_hourly_all,
        )
        write_vector_sheet(
            writer,
            "SOC_MIN_HOURLY",
            results.soc_min_hourly_all,
        )
        write_vector_sheet(
            writer,
            "SOC_MAX_HOURLY",
            results.soc_max_hourly_all,
        )

        # Optimization cycle aging
        write_vector_sheet(
            writer,
            "OPT_CYCLE_AGING_DAILY",
            results.opt_cycle_aging_daily_all,
        )
        write_vector_sheet(
            writer,
            "OPT_CYCLE_AGING_CUM",
            results.opt_cycle_aging_cumulative_all,
        )
        write_vector_sheet(
            writer,
            "OPT_CYCLE_AGING_WEEKLY",
            results.opt_cycle_aging_weekly_all,
        )
        write_vector_sheet(
            writer,
            "OPT_CYCLE_AGING_WEEKLY_CUM",
            results.opt_cycle_aging_weekly_cumulative_all,
        )

        # Calendar aging inside objective
        write_vector_sheet(
            writer,
            "CAL_DAMAGE_DAILY",
            results.cal_damage_daily_all,
        )
        write_vector_sheet(
            writer,
            "CAL_COST_DAILY",
            results.cal_cost_daily_all,
        )
        write_vector_sheet(
            writer,
            "CAL_DAMAGE_CUM",
            results.cal_damage_cumulative_all,
        )
        write_vector_sheet(
            writer,
            "CAL_COST_CUM",
            results.cal_cost_cumulative_all,
        )

        # Rainflow
        write_vector_sheet(
            writer,
            "RF_COST_WEEKLY",
            results.rf_cost_incremental_weekly_all,
        )
        write_vector_sheet(
            writer,
            "RF_COST_CUMULATIVE",
            results.rf_cost_weekly_all,
        )
        write_vector_sheet(
            writer,
            "RF_DAMAGE_WEEKLY",
            results.rf_damage_incremental_weekly_all,
        )
        write_vector_sheet(
            writer,
            "RF_DAMAGE_CUMULATIVE",
            results.rf_damage_weekly_all,
        )

        # Weekly capacity
        weekly_df = pd.DataFrame(
            {
                "Year": results.week_year_all,
                "DayInYear": results.week_day_all,
                "GlobalDay": results.week_global_day_all,
                "BatteryCapacity_MWh": results.capacity_weekly_all,
                "SOH": results.soh_weekly_all,
                "Weekly_RF_Damage":
                    results.rf_damage_incremental_weekly_all,
                "Cumulative_RF_Damage":
                    results.rf_damage_weekly_all,
                "Weekly_RF_Cost":
                    results.rf_cost_incremental_weekly_all,
                "Cumulative_RF_Cost":
                    results.rf_cost_weekly_all,
                "Weekly_Opt_Cycle_Aging":
                    results.opt_cycle_aging_weekly_all,
                "Cumulative_Opt_Cycle_Aging":
                    results.opt_cycle_aging_weekly_cumulative_all,
                "Weekly_Calendar_Loss_MWh":
                    results.cal_damage_weekly_all,
                "Cumulative_Calendar_Loss_MWh":
                    results.cal_damage_weekly_cumulative_all,
                "Weekly_Calendar_Cost":
                    results.cal_cost_weekly_all,
                "Cumulative_Calendar_Cost":
                    results.cal_cost_weekly_cumulative_all,
            }
        )
        weekly_df.to_excel(
            writer,
            sheet_name="WEEKLY_CAPACITY",
            index=False,
        )

        # Daily
        write_vector_sheet(
            writer,
            "DAILY_PROFIT",
            results.profit_daily_all,
        )
        write_vector_sheet(
            writer,
            "DAILY_OBJECTIVE",
            results.objective_daily_all,
        )
        write_vector_sheet(writer, "DAY_YEAR", results.day_year_all)
        write_vector_sheet(writer, "DAY_IN_YEAR", results.day_in_year_all)
        write_vector_sheet(writer, "GLOBAL_DAY", results.global_day_all)

        # Annual summary
        annual_df = pd.DataFrame(
            {
                "Year": results.year_all,
                "DaysSimulated": results.year_days_simulated_all,
                "InitialCapacity_MWh":
                    results.year_initial_capacity_all,
                "FinalCapacity_MWh":
                    results.year_final_capacity_all,
                "InitialSOH": results.year_initial_soh_all,
                "FinalSOH": results.year_final_soh_all,
                "DA_Revenue_EUR":
                    results.year_da_revenue_all,
                "Objective_EUR":
                    results.year_objective_all,
                "Discharge_MWh":
                    results.year_discharge_all,
                "WindCharge_MWh":
                    results.year_wind_charge_all,
                "EFC":
                    results.year_efc_all,
                "OptCycleCost_EUR":
                    results.year_opt_cycle_cost_all,
                "CalCostObjective_EUR":
                    results.year_cal_cost_objective_all,
                "CalLossObjective_MWh":
                    results.year_cal_loss_objective_all,
                "RF_Damage_Increment":
                    results.year_rf_damage_increment_all,
                "RF_Cost_Increment_EUR":
                    results.year_rf_cost_increment_all,
                "RF_Loss_Increment_MWh":
                    results.year_rf_loss_increment_all,
                "CalLoss_Increment_MWh":
                    results.year_cal_loss_increment_all,
                "CapacityLoss_Increment_MWh":
                    results.year_capacity_loss_increment_all,
            }
        )
        annual_df.to_excel(
            writer,
            sheet_name="ANNUAL_SUMMARY",
            index=False,
        )

        # Scalar totals
        scalar_sheet_map = {
            "TOTAL_PROFIT": "total_profit",
            "TOTAL_OBJECTIVE": "total_objective",
            "TOTAL_DISCHARGE_MWH": "total_discharge_mwh",
            "TOTAL_WIND_CHARGE_MWH": "total_wind_charge_mwh",
            "EQUIVALENT_FULL_CYCLES": "equivalent_full_cycles",
            "TOTAL_OPT_CYCLE_AGING": "total_opt_cycle_aging",
            "TOTAL_CAL_COST": "total_cal_cost",
            "TOTAL_CAL_DAMAGE": "total_cal_damage",
            "TOTAL_RF_COST": "total_rf_cost",
            "TOTAL_RF_DAMAGE": "total_rf_damage",
            "TOTAL_CAPACITY_LOSS_MWH": "total_capacity_loss_mwh",
            "TOTAL_CAPACITY_LOSS_FRACTION":
                "total_capacity_loss_fraction",
            "TOTAL_CAL_COST_OBJECTIVE":
                "total_cal_cost_objective",
            "TOTAL_CAL_LOSS_OBJECTIVE":
                "total_cal_loss_objective",
            "FINAL_CAPACITY": "final_capacity",
            "FINAL_SOH": "final_soh",
        }

        for sheet_name, key in scalar_sheet_map.items():
            write_scalar_sheet(
                writer,
                sheet_name,
                totals[key],
            )


# =========================================================
# MAIN MULTI-YEAR SIMULATION
# =========================================================
def plot_annual_results(
    results: Results,
    output_file: Path,
) -> None:
    years = np.asarray(
        results.year_all,
        dtype=int,
    )

    soh_percent = (
        100.0
        * np.asarray(
            results.year_final_soh_all,
            dtype=float,
        )
    )

    da_revenue = np.asarray(
        results.year_da_revenue_all,
        dtype=float,
    )

    objective_profit = np.asarray(
        results.year_objective_all,
        dtype=float,
    )

    # Post-processing aging costs
    rf_cost = np.asarray(
        results.year_rf_cost_increment_all,
        dtype=float,
    )

    calendar_loss = np.asarray(
        results.year_cal_loss_increment_all,
        dtype=float,
    )

    calendar_cost = (
        CAL_COST_FACTOR * calendar_loss
    )

    # Net profit after nonlinear calendar aging
    # and Rainflow cycle aging
    net_profit_after_aging = (
        da_revenue
        - rf_cost
        - calendar_cost
    )

    # Convert financial values to million EUR
    da_revenue_million = da_revenue / 1e6
    objective_million = objective_profit / 1e6
    net_profit_million = (
        net_profit_after_aging / 1e6
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 10),
        constrained_layout=True,
    )

    # ============================================
    # SOH
    # ============================================

    axes[0].plot(
        years,
        soh_percent,
        marker="o",
        linewidth=2.2,
        color="darkgreen",
        label="Final annual SOH",
    )

    axes[0].axhline(
        80.0,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="80% SOH",
    )

    axes[0].axhline(
        40.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Stopping threshold",
    )

    axes[0].set_title(
        "Battery State of Health"
    )
    axes[0].set_xlabel("Operating year")
    axes[0].set_ylabel("SOH (%)")
    axes[0].set_xticks(years)
    axes[0].grid(
        True,
        alpha=0.3,
    )
    axes[0].legend()

    # ============================================
    # REVENUE AND PROFIT
    # ============================================

    width = 0.25

    axes[1].bar(
        years - width,
        da_revenue_million,
        width=width,
        color="steelblue",
        label="DA market revenue",
    )

    axes[1].bar(
        years,
        objective_million,
        width=width,
        color="darkorange",
        label="Optimization objective",
    )

    axes[1].bar(
        years + width,
        net_profit_million,
        width=width,
        color="seagreen",
        label="Net profit after RF + calendar aging",
    )

    axes[1].axhline(
        0.0,
        color="black",
        linewidth=0.8,
    )

    axes[1].set_title(
        "Annual Revenue and Profit"
    )
    axes[1].set_xlabel("Operating year")
    axes[1].set_ylabel("Million EUR")
    axes[1].set_xticks(years)
    axes[1].grid(
        axis="y",
        alpha=0.3,
    )
    axes[1].legend()

    plot_file = output_file.with_name(
        output_file.stem
        + "_annual_results.png"
    )

    fig.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"Annual plot saved: {plot_file}")


def plot_existing_results(output_file: Path) -> None:
    """Plot annual results already stored in Excel without rerunning Pyomo."""
    if not output_file.exists():
        raise FileNotFoundError(
            f"Existing result file was not found: {output_file}"
        )

    annual = pd.read_excel(
        output_file,
        sheet_name="ANNUAL_SUMMARY",
    )

    required_columns = {
        "Year",
        "FinalSOH",
        "DA_Revenue_EUR",
        "Objective_EUR",
        "RF_Cost_Increment_EUR",
        "CalLoss_Increment_MWh",
    }
    missing_columns = required_columns.difference(annual.columns)
    if missing_columns:
        raise ValueError(
            "ANNUAL_SUMMARY is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    stored_results = Results.empty()
    stored_results.year_all = annual["Year"].tolist()
    stored_results.year_final_soh_all = annual["FinalSOH"].tolist()
    stored_results.year_da_revenue_all = annual["DA_Revenue_EUR"].tolist()
    stored_results.year_objective_all = annual["Objective_EUR"].tolist()
    stored_results.year_rf_cost_increment_all = annual[
        "RF_Cost_Increment_EUR"
    ].tolist()
    stored_results.year_cal_loss_increment_all = annual[
        "CalLoss_Increment_MWh"
    ].tolist()

    plot_annual_results(stored_results, output_file)


 ######################   
def main() -> None:

    for required_file in (PRICE_FILE, WIND_FILE, STROE_FILE):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Missing input file: {required_file.name}\n"
                "Put all three Excel input files in the same folder as main.py."
            )

    price_all = read_single_excel_column(
        PRICE_FILE,
        sheet_name="Sheet1",
        column="C",
        start_row=2,
        nrows=8760,
    )

    wind_all = read_single_excel_column(
        WIND_FILE,
        sheet_name="Sheet1",
        column="C",
        start_row=2,
        nrows=8760,
    )

    x_stroe, y_stroe = read_stroe_data(
        STROE_FILE
    )

    if len(price_all) != 8760:
        raise ValueError(
            f"Expected 8760 price values; got {len(price_all)}."
        )

    if len(wind_all) != 8760:
        raise ValueError(
            f"Expected 8760 wind values; got {len(wind_all)}."
        )

    results = Results.empty()

    soc_initial_ref = float(
        INITIAL_SOC
    )
    battery_capacity_ref = float(
        SIMULATION_INITIAL_CAPACITY
    )

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
        print("########################################################")

        commissioning_hours = (
            (operating_year - 1) * 8760.0
        )

        year_initial_capacity = (
            battery_capacity_ref
        )

        year_initial_soh = (
            year_initial_capacity
            / SOCBESS_INITIAL
        )

        day_idx_start = len(
            results.profit_daily_all
        )
        hour_idx_start = len(
            results.soc_all
        )

        rf_damage_start = (
            0.0
            if not results.rf_damage_weekly_all
            else results.rf_damage_weekly_all[-1]
        )

        rf_cost_start = (
            0.0
            if not results.rf_cost_weekly_all
            else results.rf_cost_weekly_all[-1]
        )

        cal_damage_start = (
            0.0
            if not results.cal_damage_weekly_cumulative_all
            else results.cal_damage_weekly_cumulative_all[-1]
        )

        days_simulated_this_year = 0

        # =====================================================
        # LOOP OVER DAYS
        # =====================================================

        for d in range(1, 366):

            print("================================")
            print(
                f"Running year {operating_year}, day {d}"
            )
            print("================================")

            current_capacity = (
                battery_capacity_ref
            )

            soc_min_current = (
                0.10 * current_capacity
            )
            soc_max_current = (
                current_capacity
            )

            soc_initial_ref = min(
                soc_initial_ref,
                soc_max_current,
            )

            soc_initial_ref = max(
                soc_initial_ref,
                soc_min_current,
            )

            soc_initial = (
                soc_initial_ref
            )

            print(
                "Current battery capacity = "
                f"{current_capacity}"
            )
            print(f"SOC min = {soc_min_current}")
            print(f"SOC max = {soc_max_current}")
            print(f"SOC initial = {soc_initial}")

            t_current = (
                24 if d == 365 else T
            )

            # Julia indexing:
            # start_hour = (d - 1)*24 + 1
            # Python equivalent:
            start_idx = (
                (d - 1) * IMPLEMENT
            )
            end_idx = (
                start_idx + t_current
            )

            price = price_all[
                start_idx:end_idx
            ]
            wind = wind_all[
                start_idx:end_idx
            ]

            if (
                len(price) != t_current
                or len(wind) != t_current
            ):
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
                t_current=t_current,
                current_capacity=current_capacity,
                soc_initial=soc_initial,
                commissioning_hours=commissioning_hours,
                auxiliar1=auxiliar1,
            )

            # =================================================
            # SAVE IMPLEMENTED HOURS
            # =================================================

            save_hours = min(
                IMPLEMENT,
                t_current,
            )
            implemented = slice(
                0,
                save_hours,
            )

            soc_impl = day_result["soc"][implemented]
            pda_impl = day_result["p_DA"][implemented]
            wp_impl = day_result["wp"][implemented]
            wp2b_impl = day_result["wp2b"][implemented]
            discharge_impl = day_result["ed_discharged"][implemented]
            charged_impl = day_result["ed_charged"][implemented]
            pb_impl = day_result["pb"][implemented]
            aux1_impl = day_result["aux1"][implemented]
            ud_impl = day_result["ud"][implemented]
            y_impl = day_result["y"][implemented, :]
            dcr_impl = day_result["dcr"][implemented]
            bess_cost_impl = day_result["bess_cost"][implemented]
            ag_cal_impl = day_result["AG_cal"][implemented]
            c_cal_impl = day_result["C_cal"][implemented]

            results.soc_all.extend(
                np.asarray(soc_impl).tolist()
            )
            results.pda_all.extend(
                np.asarray(pda_impl).tolist()
            )
            results.wp_all.extend(
                np.asarray(wp_impl).tolist()
            )
            results.wp2b_all.extend(
                np.asarray(wp2b_impl).tolist()
            )
            results.discharge_all.extend(
                np.asarray(discharge_impl).tolist()
            )
            results.charged_all.extend(
                np.asarray(charged_impl).tolist()
            )
            results.pb_all.extend(
                np.asarray(pb_impl).tolist()
            )
            results.aux1_all.extend(
                np.asarray(aux1_impl).tolist()
            )
            results.ud_all.extend(
                np.asarray(ud_impl).tolist()
            )
            results.y1_all.extend(
                np.asarray(y_impl[:, 0]).tolist()
            )
            results.y2_all.extend(
                np.asarray(y_impl[:, 1]).tolist()
            )
            results.y3_all.extend(
                np.asarray(y_impl[:, 2]).tolist()
            )

            local_hour_start = (
                (d - 1) * IMPLEMENT + 1
            )

            for h in range(save_hours):
                results.hour_year_all.append(
                    operating_year
                )
                results.hour_in_year_all.append(
                    local_hour_start + h
                )
                results.global_hour_all.append(
                    (operating_year - 1) * 8760
                    + local_hour_start
                    + h
                )

            results.dcr_all.extend(
                np.asarray(dcr_impl).tolist()
            )
            results.bess_cost_all.extend(
                np.asarray(bess_cost_impl).tolist()
            )
            results.cal_damage_hourly_all.extend(
                np.asarray(ag_cal_impl).tolist()
            )
            results.cal_cost_hourly_all.extend(
                np.asarray(c_cal_impl).tolist()
            )

            results.capacity_hourly_all.extend(
                [current_capacity] * save_hours
            )
            results.soc_min_hourly_all.extend(
                [soc_min_current] * save_hours
            )
            results.soc_max_hourly_all.extend(
                [soc_max_current] * save_hours
            )

            # Daily optimization cycle-aging cost
            daily_opt_cycle_aging = float(
                np.sum(bess_cost_impl)
            )

            results.opt_cycle_aging_daily_all.append(
                daily_opt_cycle_aging
            )

            results.opt_cycle_aging_cumulative_all.append(
                float(
                    np.sum(
                        results.opt_cycle_aging_daily_all
                    )
                )
            )

            # Daily calendar aging in optimization
            daily_cal_damage = float(
                np.sum(ag_cal_impl)
            )
            daily_cal_cost = float(
                np.sum(c_cal_impl)
            )

            results.cal_damage_daily_all.append(
                daily_cal_damage
            )
            results.cal_cost_daily_all.append(
                daily_cal_cost
            )

            results.cal_damage_cumulative_all.append(
                float(
                    np.sum(
                        results.cal_damage_daily_all
                    )
                )
            )

            results.cal_cost_cumulative_all.append(
                float(
                    np.sum(
                        results.cal_cost_daily_all
                    )
                )
            )

            # DA revenue and implemented objective
            daily_da_revenue = float(
                np.dot(
                    price[:save_hours],
                    pda_impl,
                )
            )

            daily_profit = (
                daily_da_revenue
            )

            daily_objective = float(
                np.sum(
                    price[:save_hours] * pda_impl
                    - bess_cost_impl
                )
            )

            results.profit_daily_all.append(
                daily_profit
            )
            results.objective_daily_all.append(
                daily_objective
            )

            results.day_year_all.append(
                operating_year
            )
            results.day_in_year_all.append(
                d
            )
            results.global_day_all.append(
                (operating_year - 1) * 365 + d
            )

            days_simulated_this_year = d

            # SoC for next rolling-horizon day
            soc_initial_ref = float(
                soc_impl[-1]
            )

            print(
                f"SOC next day = {soc_initial_ref}"
            )

            # =================================================
            # WEEKLY RAINFLOW + CALENDAR CAPACITY UPDATE
            # =================================================

            if (
                d % DAYS_PER_WEEK == 0
                or d == 365
            ):

                print(
                    "---------- Weekly degradation update ----------"
                )

                # Complete multi-year SoC history
                soc_history = np.concatenate(
                    [
                        np.array(
                            [INITIAL_SOC],
                            dtype=float,
                        ),
                        np.asarray(
                            results.soc_all,
                            dtype=float,
                        ),
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
                    0.0
                    if not results.rf_damage_weekly_all
                    else results.rf_damage_weekly_all[-1]
                )

                previous_rf_cost = (
                    0.0
                    if not results.rf_cost_weekly_all
                    else results.rf_cost_weekly_all[-1]
                )

                weekly_rf_damage = (
                    total_rf_damage
                    - previous_rf_damage
                )

                weekly_rf_cost = (
                    total_rf_cost
                    - previous_rf_cost
                )

                week_start_day = max(
                    1,
                    d - DAYS_PER_WEEK + 1,
                )

                week_start_hour = (
                    (week_start_day - 1)
                    * IMPLEMENT
                    + 1
                )

                week_end_hour = (
                    d * IMPLEMENT
                )

                global_week_start_hour = (
                    (operating_year - 1)
                    * 8760
                    + week_start_hour
                )

                global_week_end_hour = min(
                    (operating_year - 1)
                    * 8760
                    + week_end_hour,
                    len(results.soc_all),
                )

                # Convert Julia 1-based inclusive slicing
                # to Python zero-based end-exclusive slicing.
                start0 = (
                    global_week_start_hour - 1
                )
                end0 = (
                    global_week_end_hour
                )

                soc_week = np.asarray(
                    results.soc_all[start0:end0],
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
                    if not results.cal_damage_weekly_cumulative_all
                    else results.cal_damage_weekly_cumulative_all[-1]
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
                    if not results.cal_cost_weekly_cumulative_all
                    else results.cal_cost_weekly_cumulative_all[-1]
                )

                cumulative_cal_cost = (
                    previous_cumulative_cal_cost
                    + weekly_cal_cost
                )

                results.cal_damage_weekly_all.append(
                    weekly_cal_damage
                )
                results.cal_cost_weekly_all.append(
                    weekly_cal_cost
                )
                results.cal_damage_weekly_cumulative_all.append(
                    cumulative_cal_damage
                )
                results.cal_cost_weekly_cumulative_all.append(
                    cumulative_cal_cost
                )

                rf_capacity_loss_mwh = (
                    total_rf_damage
                    * SOCBESS_INITIAL
                )

                cal_capacity_loss_mwh = (
                    cumulative_cal_damage
                )

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

                battery_capacity_ref = (
                    new_capacity
                )

                soh = (
                    battery_capacity_ref
                    / SOCBESS_INITIAL
                )

                results.week_year_all.append(
                    operating_year
                )
                results.week_day_all.append(
                    d
                )
                results.week_global_day_all.append(
                    (operating_year - 1) * 365 + d
                )
                results.capacity_weekly_all.append(
                    battery_capacity_ref
                )
                results.soh_weekly_all.append(
                    soh
                )

                results.rf_damage_incremental_weekly_all.append(
                    weekly_rf_damage
                )
                results.rf_cost_incremental_weekly_all.append(
                    weekly_rf_cost
                )
                results.rf_damage_weekly_all.append(
                    total_rf_damage
                )
                results.rf_cost_weekly_all.append(
                    total_rf_cost
                )

                global_week_start_day = (
                    (operating_year - 1)
                    * 365
                    + week_start_day
                )

                global_day = (
                    (operating_year - 1)
                    * 365
                    + d
                )

                weekly_opt_cycle_aging = float(
                    np.sum(
                        results.opt_cycle_aging_daily_all[
                            global_week_start_day - 1:
                            global_day
                        ]
                    )
                )

                cumulative_opt_cycle_aging = float(
                    np.sum(
                        results.opt_cycle_aging_daily_all[
                            :global_day
                        ]
                    )
                )

                results.opt_cycle_aging_weekly_all.append(
                    weekly_opt_cycle_aging
                )

                results.opt_cycle_aging_weekly_cumulative_all.append(
                    cumulative_opt_cycle_aging
                )

                print(f"Day = {d}")
                print(
                    f"Weekly RF damage = {weekly_rf_damage}"
                )
                print(
                    f"Cumulative RF damage = {total_rf_damage}"
                )
                print(
                    f"Weekly RF cost = {weekly_rf_cost}"
                )
                print(
                    f"Cumulative RF cost = {total_rf_cost}"
                )
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
                    "Weekly calendar cost = "
                    f"{weekly_cal_cost}"
                )
                print(
                    "Cumulative calendar cost = "
                    f"{cumulative_cal_cost}"
                )
                print(
                    "RF capacity loss = "
                    f"{rf_capacity_loss_mwh} MWh"
                )
                print(
                    "Total capacity loss used for capacity = "
                    f"{total_capacity_loss_mwh} MWh"
                )
                print(
                    "Weekly Opt cycle aging = "
                    f"{weekly_opt_cycle_aging}"
                )
                print(
                    "Cumulative Opt cycle aging = "
                    f"{cumulative_opt_cycle_aging}"
                )
                print(
                    "Updated battery capacity for next week = "
                    f"{battery_capacity_ref}"
                )
                print(f"SOH = {soh}")

                # Keep next-day initial SoC feasible.
                soc_initial_ref = min(
                    soc_initial_ref,
                    battery_capacity_ref,
                )

                soc_initial_ref = max(
                    soc_initial_ref,
                    0.10 * battery_capacity_ref,
                )

                if (
                    soh
                    <= SOH_STOP_THRESHOLD
                ):
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

        day_idx_end = len(
            results.profit_daily_all
        )
        hour_idx_end = len(
            results.soc_all
        )

        annual_da_revenue = (
            float(
                np.sum(
                    results.profit_daily_all[
                        day_idx_start:day_idx_end
                    ]
                )
            )
            if day_idx_end > day_idx_start
            else 0.0
        )

        annual_objective = (
            float(
                np.sum(
                    results.objective_daily_all[
                        day_idx_start:day_idx_end
                    ]
                )
            )
            if day_idx_end > day_idx_start
            else 0.0
        )

        annual_discharge = (
            float(
                np.sum(
                    results.discharge_all[
                        hour_idx_start:hour_idx_end
                    ]
                )
            )
            if hour_idx_end > hour_idx_start
            else 0.0
        )

        annual_wind_charge = (
            float(
                np.sum(
                    results.wp2b_all[
                        hour_idx_start:hour_idx_end
                    ]
                )
            )
            if hour_idx_end > hour_idx_start
            else 0.0
        )

        annual_efc = (
            annual_discharge
            / SOCBESS_INITIAL
        )

        annual_opt_cycle_cost = (
            float(
                np.sum(
                    results.opt_cycle_aging_daily_all[
                        day_idx_start:day_idx_end
                    ]
                )
            )
            if day_idx_end > day_idx_start
            else 0.0
        )

        annual_cal_cost_obj = (
            float(
                np.sum(
                    results.cal_cost_daily_all[
                        day_idx_start:day_idx_end
                    ]
                )
            )
            if day_idx_end > day_idx_start
            else 0.0
        )

        annual_cal_loss_obj = (
            float(
                np.sum(
                    results.cal_damage_daily_all[
                        day_idx_start:day_idx_end
                    ]
                )
            )
            if day_idx_end > day_idx_start
            else 0.0
        )

        rf_damage_end = (
            rf_damage_start
            if not results.rf_damage_weekly_all
            else results.rf_damage_weekly_all[-1]
        )

        rf_cost_end = (
            rf_cost_start
            if not results.rf_cost_weekly_all
            else results.rf_cost_weekly_all[-1]
        )

        cal_damage_end = (
            cal_damage_start
            if not results.cal_damage_weekly_cumulative_all
            else results.cal_damage_weekly_cumulative_all[-1]
        )

        annual_rf_damage_increment = (
            rf_damage_end
            - rf_damage_start
        )

        annual_rf_cost_increment = (
            rf_cost_end
            - rf_cost_start
        )

        annual_rf_loss_increment = (
            annual_rf_damage_increment
            * SOCBESS_INITIAL
        )

        annual_cal_loss_increment = (
            cal_damage_end
            - cal_damage_start
        )

        annual_capacity_loss_increment = (
            annual_rf_loss_increment
            + (
                annual_cal_loss_increment
                if INCLUDE_CALENDAR_IN_CAPACITY_UPDATE
                else 0.0
            )
        )

        results.year_all.append(
            operating_year
        )
        results.year_days_simulated_all.append(
            days_simulated_this_year
        )
        results.year_initial_capacity_all.append(
            year_initial_capacity
        )
        results.year_final_capacity_all.append(
            battery_capacity_ref
        )
        results.year_initial_soh_all.append(
            year_initial_soh
        )
        results.year_final_soh_all.append(
            battery_capacity_ref
            / SOCBESS_INITIAL
        )
        results.year_da_revenue_all.append(
            annual_da_revenue
        )
        results.year_objective_all.append(
            annual_objective
        )
        results.year_discharge_all.append(
            annual_discharge
        )
        results.year_wind_charge_all.append(
            annual_wind_charge
        )
        results.year_efc_all.append(
            annual_efc
        )
        results.year_opt_cycle_cost_all.append(
            annual_opt_cycle_cost
        )
        results.year_cal_cost_objective_all.append(
            annual_cal_cost_obj
        )
        results.year_cal_loss_objective_all.append(
            annual_cal_loss_obj
        )
        results.year_rf_damage_increment_all.append(
            annual_rf_damage_increment
        )
        results.year_rf_cost_increment_all.append(
            annual_rf_cost_increment
        )
        results.year_rf_loss_increment_all.append(
            annual_rf_loss_increment
        )
        results.year_cal_loss_increment_all.append(
            annual_cal_loss_increment
        )
        results.year_capacity_loss_increment_all.append(
            annual_capacity_loss_increment
        )

        print(
            f"YEAR {operating_year} COMPLETE"
        )
        print(
            f"Days simulated = {days_simulated_this_year}"
        )
        print(
            f"Annual DA revenue = {annual_da_revenue}"
        )
        print(
            "Final capacity = "
            f"{battery_capacity_ref} MWh"
        )
        print(
            "Final SOH = "
            f"{battery_capacity_ref / SOCBESS_INITIAL}"
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

    total_profit = float(
        np.sum(
            results.profit_daily_all
        )
    )

    total_objective = float(
        np.sum(
            results.objective_daily_all
        )
    )

    total_opt_cycle_aging = float(
        np.sum(
            results.opt_cycle_aging_daily_all
        )
    )

    total_cal_damage = (
        0.0
        if not results.cal_damage_weekly_cumulative_all
        else results.cal_damage_weekly_cumulative_all[-1]
    )

    total_cal_cost = (
        0.0
        if not results.cal_cost_weekly_cumulative_all
        else results.cal_cost_weekly_cumulative_all[-1]
    )

    total_cal_fraction = (
        total_cal_damage
        / SOCBESS_INITIAL
    )

    total_cal_cost_objective = float(
        np.sum(
            results.cal_cost_daily_all
        )
    )

    total_cal_loss_objective = float(
        np.sum(
            results.cal_damage_daily_all
        )
    )

    total_discharge_mwh = float(
        np.sum(
            results.discharge_all
        )
    )

    total_wind_charge_mwh = float(
        np.sum(
            results.wp2b_all
        )
    )

    equivalent_full_cycles = (
        total_discharge_mwh
        / SOCBESS_INITIAL
    )

    total_rf_cost = (
        0.0
        if not results.rf_cost_weekly_all
        else results.rf_cost_weekly_all[-1]
    )

    total_rf_damage = (
        0.0
        if not results.rf_damage_weekly_all
        else results.rf_damage_weekly_all[-1]
    )

    total_rf_loss_mwh = (
        total_rf_damage
        * SOCBESS_INITIAL
    )

    if INCLUDE_CALENDAR_IN_CAPACITY_UPDATE:
        total_capacity_loss_mwh = (
            total_rf_loss_mwh
            + total_cal_damage
        )
    else:
        total_capacity_loss_mwh = (
            total_rf_loss_mwh
        )

    total_capacity_loss_fraction = (
        total_capacity_loss_mwh
        / SOCBESS_INITIAL
    )

    final_soh = (
        battery_capacity_ref
        / SOCBESS_INITIAL
    )

    print("================================")
    print(
        f"TOTAL PROFIT = {total_profit}"
    )
    print(
        f"TOTAL OBJECTIVE = {total_objective}"
    )
    print(
        "TOTAL DISCHARGE = "
        f"{total_discharge_mwh} MWh"
    )
    print(
        "TOTAL WIND CHARGING = "
        f"{total_wind_charge_mwh} MWh"
    )
    print(
        "EQUIVALENT FULL CYCLES = "
        f"{equivalent_full_cycles}"
    )
    print(
        "TOTAL OPT CYCLE AGING = "
        f"{total_opt_cycle_aging}"
    )
    print(
        "TOTAL CALENDAR AGING COST = "
        f"{total_cal_cost}"
    )
    print(
        "TOTAL CALENDAR AGING LOSS = "
        f"{total_cal_damage} MWh"
    )
    print(
        "TOTAL CALENDAR AGING FRACTION = "
        f"{total_cal_fraction}"
    )
    print(
        "TOTAL CALENDAR AGING COST IN OBJECTIVE = "
        f"{total_cal_cost_objective}"
    )
    print(
        "TOTAL CALENDAR AGING LOSS IN OBJECTIVE = "
        f"{total_cal_loss_objective} MWh"
    )
    print(
        f"TOTAL RF COST = {total_rf_cost}"
    )
    print(
        "TOTAL RF LOSS = "
        f"{total_rf_loss_mwh} MWh"
    )
    print(
        "TOTAL RF DAMAGE FRACTION = "
        f"{total_rf_damage}"
    )
    print(
        "TOTAL CAPACITY LOSS USED FOR CAPACITY = "
        f"{total_capacity_loss_mwh} MWh"
    )
    print(
        "TOTAL CAPACITY LOSS FRACTION = "
        f"{total_capacity_loss_fraction}"
    )
    print(
        "FINAL BATTERY CAPACITY = "
        f"{battery_capacity_ref}"
    )
    print(
        f"FINAL SOH = {final_soh}"
    )
    print("================================")

    totals = {
        "total_profit": total_profit,
        "total_objective": total_objective,
        "total_discharge_mwh": total_discharge_mwh,
        "total_wind_charge_mwh": total_wind_charge_mwh,
        "equivalent_full_cycles": equivalent_full_cycles,
        "total_opt_cycle_aging": total_opt_cycle_aging,
        "total_cal_cost": total_cal_cost,
        "total_cal_damage": total_cal_damage,
        "total_rf_cost": total_rf_cost,
        "total_rf_damage": total_rf_damage,
        "total_capacity_loss_mwh":
            total_capacity_loss_mwh,
        "total_capacity_loss_fraction":
            total_capacity_loss_fraction,
        "total_cal_cost_objective":
            total_cal_cost_objective,
        "total_cal_loss_objective":
            total_cal_loss_objective,
        "final_capacity":
            battery_capacity_ref,
        "final_soh":
            final_soh,
    }

    export_results(
        results,
        OUTPUT_FILE,
        totals,
    )

    plot_annual_results(
        results,
        OUTPUT_FILE,
    )

    print("Results exported successfully!")
    print(f"Output file: {OUTPUT_FILE}")
    print(
        f"Years simulated: {len(results.year_all)}"
    )
    print(f"Final SOH: {final_soh}")


if __name__ == "__main__":
    import sys

    if "--plot-only" in sys.argv:
        plot_existing_results(OUTPUT_FILE)
    else:
        main()
