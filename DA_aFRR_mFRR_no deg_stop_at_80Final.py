# DA + aFRR + mFRR with fixed battery capacity; no degradation.
# Install: pip install numpy pandas pyomo highspy openpyxl rainflow
# Place this script beside the market and scaled wind Excel inputs.
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import rainflow
import numpy as np
import pandas as pd
import pyomo.environ as pyo
BASE_DIR = Path(__file__).resolve().parent
PRICE_FILE = BASE_DIR / 'priceSPOTDan.xlsx'
WIND_FILE = BASE_DIR / 'WindFarmR3_scaled_308_over_35.xlsx'
PRICE_BR2_FILE = BASE_DIR / 'priceBR2.xlsx'
RATIO_BR2_FILE = BASE_DIR / 'ratioBR2.xlsx'
PRICE_ER2UP_FILE = BASE_DIR / 'priceER2up.xlsx'
PRICE_ER2DOWN_FILE = BASE_DIR / 'priceER2down.xlsx'
USE_ER2UP_FILE = BASE_DIR / 'useER2up.xlsx'
USE_ER2DOWN_FILE = BASE_DIR / 'useER2down.xlsx'
PRICE_ER3UP_FILE = BASE_DIR / 'priceER3up.xlsx'
PRICE_ER3DOWN_FILE = BASE_DIR / 'priceER3down.xlsx'
USE_ER3UP_FILE = BASE_DIR / 'useER3up.xlsx'
USE_ER3DOWN_FILE = BASE_DIR / 'useER3down.xlsx'

def read_single_excel_column(file_path: Path, sheet_name: str, column: str, start_row: int, nrows: int, fill_value: float=0.0) -> np.ndarray:
    """Read one Excel column and replace nonnumeric/missing values by fill_value."""
    df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=column, skiprows=start_row - 1, nrows=nrows, header=None, engine='openpyxl')
    values = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(fill_value).astype(float).to_numpy()
    return values

def read_8760_series(file_path: Path) -> np.ndarray:
    """Equivalent to Julia's read_8760_series: Sheet1!C2:C8761."""
    values = read_single_excel_column(file_path=file_path, sheet_name='Sheet1', column='C', start_row=2, nrows=8760, fill_value=0.0)
    if len(values) != 8760:
        raise ValueError(f'Expected 8760 values in {file_path.name}; got {len(values)}.')
    return values
T = 48
IMPLEMENT = 24
NODE_CAPACITY_G = 300.0
NODE_CAPACITY_C = 20.0
WIND_FARM_CAPACITY_MW = 308.0
WP_MAX = WIND_FARM_CAPACITY_MW
PBESS = 20.0
RATED_CAPACITY = 60.0
ETA = 0.86
MAX_YEARS = 20
INITIAL_SOC = 0.5 * RATED_CAPACITY
SOH_STOP_THRESHOLD = 0.80
BATTERY_CAPEX_EUR = 6_200_000.0
DISCOUNT_RATE = 0.09

def solve_daily_model(price: np.ndarray, wind: np.ndarray, price_br2: np.ndarray, ratio_br2: np.ndarray, price_er2up: np.ndarray, price_er2down: np.ndarray, use_er2up: np.ndarray, use_er2down: np.ndarray, price_er3up: np.ndarray, price_er3down: np.ndarray, use_er3up: np.ndarray, use_er3down: np.ndarray, t_current: int, current_capacity: float, soc_initial: float) -> Dict[str, np.ndarray | float]:
    """Build and solve one rolling-horizon DA + aFRR + mFRR MILP with Pyomo + HiGHS."""
    model = pyo.ConcreteModel(name='DA_aFRR_mFRR_BESS')
    model.T = pyo.RangeSet(0, t_current - 1)
    soc_min_current = 0.1 * current_capacity
    soc_max_current = current_capacity
    model.wp = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, WP_MAX))
    model.wp2b = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, PBESS))
    model.soc = pyo.Var(model.T, domain=pyo.Reals, bounds=(soc_min_current, soc_max_current))
    model.ed_discharged = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.ed_charged = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, NODE_CAPACITY_C))
    model.p_DA = pyo.Var(model.T, domain=pyo.Reals)
    model.aux1 = pyo.Var(model.T, domain=pyo.Binary)
    model.pb = pyo.Var(model.T, domain=pyo.Reals)
    model.br2up = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, 2.0 * PBESS))
    model.br2down = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, 2.0 * PBESS))
    model.er2up = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.er2down = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.aux_er2 = pyo.Var(model.T, domain=pyo.Binary)
    model.br3up = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, 2.0 * PBESS))
    model.br3down = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, 2.0 * PBESS))
    model.er3up = pyo.Expression(model.T, rule=lambda m, t: float(use_er3up[t]) * m.br3up[t])
    model.er3down = pyo.Expression(model.T, rule=lambda m, t: float(use_er3down[t]) * m.br3down[t])

    def power_balance_rule(m, t):
        return m.wp[t] + m.ed_discharged[t] - m.ed_charged[t] == m.p_DA[t]
    model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

    def wind_limit_rule(m, t):
        return m.wp[t] + m.wp2b[t] <= float(wind[t])
    model.wind_limit = pyo.Constraint(model.T, rule=wind_limit_rule)

    def charge_mode_rule(m, t):
        return m.ed_charged[t] + m.wp2b[t] <= PBESS * (1.0 - m.aux1[t])
    model.charge_mode = pyo.Constraint(model.T, rule=charge_mode_rule)

    def discharge_mode_rule(m, t):
        return m.ed_discharged[t] <= PBESS * m.aux1[t]
    model.discharge_mode = pyo.Constraint(model.T, rule=discharge_mode_rule)

    def battery_power_rule(m, t):
        return m.ed_discharged[t] - m.ed_charged[t] - m.wp2b[t] == m.pb[t]
    model.battery_power = pyo.Constraint(model.T, rule=battery_power_rule)

    def br2_up_limit_rule(m, t):
        return m.br2up[t] <= PBESS - m.pb[t]
    model.br2_up_limit = pyo.Constraint(model.T, rule=br2_up_limit_rule)

    def br2_down_limit_rule(m, t):
        return m.br2down[t] <= m.pb[t] + PBESS
    model.br2_down_limit = pyo.Constraint(model.T, rule=br2_down_limit_rule)

    def br2_ratio_rule(m, t):
        return m.br2up[t] == float(ratio_br2[t]) * (m.br2up[t] + m.br2down[t])
    model.br2_ratio = pyo.Constraint(model.T, rule=br2_ratio_rule)

    def er2_activation_rule(m, t):
        return m.er2up[t] - m.er2down[t] == float(use_er2up[t]) * m.br2up[t] - float(use_er2down[t]) * m.br2down[t]
    model.er2_activation = pyo.Constraint(model.T, rule=er2_activation_rule)

    def er2_up_mode_rule(m, t):
        return m.er2up[t] <= PBESS * m.aux_er2[t]
    model.er2_up_mode = pyo.Constraint(model.T, rule=er2_up_mode_rule)

    def er2_down_mode_rule(m, t):
        return m.er2down[t] <= PBESS * (1.0 - m.aux_er2[t])
    model.er2_down_mode = pyo.Constraint(model.T, rule=er2_down_mode_rule)

    def br3_up_limit_rule(m, t):
        return m.br3up[t] <= PBESS - m.pb[t] - m.br2up[t]
    model.br3_up_limit = pyo.Constraint(model.T, rule=br3_up_limit_rule)

    def br3_down_limit_rule(m, t):
        return m.br3down[t] <= m.pb[t] + PBESS - m.br2down[t]
    model.br3_down_limit = pyo.Constraint(model.T, rule=br3_down_limit_rule)

    def soc_rule(m, t):
        if t == 0:
            return m.soc[t] == soc_initial + (m.wp2b[t] + m.ed_charged[t]) * ETA - m.ed_discharged[t] + ETA * m.er2down[t] - m.er2up[t] + ETA * m.er3down[t] - m.er3up[t]
        return m.soc[t] == m.soc[t - 1] + (m.wp2b[t] + m.ed_charged[t]) * ETA - m.ed_discharged[t] + ETA * m.er2down[t] - m.er2up[t] + ETA * m.er3down[t] - m.er3up[t]
    model.soc_balance = pyo.Constraint(model.T, rule=soc_rule)

    def grid_export_rule(m, t):
        return m.p_DA[t] + m.er2up[t] - m.er2down[t] + m.er3up[t] - m.er3down[t] <= NODE_CAPACITY_G
    model.grid_export = pyo.Constraint(model.T, rule=grid_export_rule)

    def grid_import_rule(m, t):
        return m.p_DA[t] + m.er2up[t] - m.er2down[t] + m.er3up[t] - m.er3down[t] >= -NODE_CAPACITY_C
    model.grid_import = pyo.Constraint(model.T, rule=grid_import_rule)
    model.objective = pyo.Objective(expr=sum((float(price[t]) * model.p_DA[t] + float(price_br2[t]) * (model.br2up[t] + model.br2down[t]) + float(price_er2up[t]) * model.er2up[t] - float(price_er2down[t]) * model.er2down[t] + float(price_er3up[t]) * model.er3up[t] - float(price_er3down[t]) * model.er3down[t] for t in model.T)), sense=pyo.maximize)
    solver = pyo.SolverFactory('highs')
    if not solver.available(exception_flag=False):
        raise RuntimeError('HiGHS is not available. Install dependencies with: pip install pyomo highspy')
    solver.options['mip_rel_gap'] = 0.0
    solver.options['time_limit'] = 3600.0
    solve_result = solver.solve(model, tee=False, load_solutions=True)
    termination = solve_result.solver.termination_condition
    acceptable = {pyo.TerminationCondition.optimal, pyo.TerminationCondition.maxTimeLimit, pyo.TerminationCondition.feasible}
    if termination not in acceptable:
        raise RuntimeError(f'Model did not solve correctly. Termination condition: {termination}')

    def var_vector(var_component) -> np.ndarray:
        return np.asarray([pyo.value(var_component[t]) for t in range(t_current)], dtype=float)
    return {'soc': var_vector(model.soc), 'p_DA': var_vector(model.p_DA), 'wp': var_vector(model.wp), 'wp2b': var_vector(model.wp2b), 'ed_discharged': var_vector(model.ed_discharged), 'ed_charged': var_vector(model.ed_charged), 'pb': var_vector(model.pb), 'br2up': var_vector(model.br2up), 'br2down': var_vector(model.br2down), 'er2up': var_vector(model.er2up), 'er2down': var_vector(model.er2down), 'br3up': var_vector(model.br3up), 'br3down': var_vector(model.br3down), 'er3up': var_vector(model.er3up), 'er3down': var_vector(model.er3down), 'objective_value': float(pyo.value(model.objective))}

def write_vector_sheet(writer: pd.ExcelWriter, sheet_name: str, values: List[float] | np.ndarray) -> None:
    pd.DataFrame(np.asarray(values).reshape(-1, 1)).to_excel(writer, sheet_name=sheet_name, index=False, header=False)

def write_scalar_sheet(writer: pd.ExcelWriter, sheet_name: str, value: float) -> None:
    pd.DataFrame([[value]]).to_excel(writer, sheet_name=sheet_name, index=False, header=False)

OUTPUT_FILE = BASE_DIR / "results_DA_aFRR_mFRR_FIXED_CAPACITY_POSTPROCESS.xlsx"

def read_stroe_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read A2:B56 from sheet data_stroe, clean invalid values and sort by x."""
    df = pd.read_excel(file_path, sheet_name='data_stroe', usecols='A:B', skiprows=1, nrows=55, header=None, engine='openpyxl')
    x = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy(dtype=float)
    y = pd.to_numeric(df.iloc[:, 1], errors='coerce').to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    return (x[order], y[order])

def interp_linear_extrap(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """Linear interpolation with linear extrapolation at both ends."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        raise ValueError('At least two points are required for interpolation.')
    if xq <= x[0]:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return float(y[0] + slope * (xq - x[0]))
    if xq >= x[-1]:
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return float(y[-2] + slope * (xq - x[-2]))
    return float(np.interp(xq, x, y))

def clean_soc_profile(values: List[float] | np.ndarray, tol: float=1e-06) -> np.ndarray:
    """Remove consecutive repeated/near-equal SoC values."""
    cleaned: List[float] = []
    for value in values:
        value = float(value)
        if not cleaned or abs(value - cleaned[-1]) > tol:
            cleaned.append(value)
    return np.asarray(cleaned, dtype=float)

def rainflow_damage_and_cost(soc_profile: List[float] | np.ndarray, socbess_ref: float, x_stroe: np.ndarray, y_stroe: np.ndarray, ecap_cost: float, tol: float=1e-06) -> Tuple[float, float]:
    """
    Compute cumulative Rainflow damage and cost.

    Python package 'rainflow' returns cycles as:
        (range, mean, count, start_index, end_index)

    Count is normally 1.0 for a full cycle or 0.5 for a half cycle.
    """
    soc_clean = clean_soc_profile(soc_profile, tol=tol)
    if len(soc_clean) < 3 or float(np.max(soc_clean) - np.min(soc_clean)) < tol:
        return (0.0, 0.0)
    total_damage = 0.0
    total_cost = 0.0
    for cycle_range, _mean, count, _i_start, _i_end in rainflow.extract_cycles(soc_clean):
        cycle_range = float(cycle_range)
        count = float(count)
        if cycle_range < tol:
            continue
        if not (np.isclose(count, 1.0) or np.isclose(count, 0.5)):
            continue
        dcr = cycle_range / socbess_ref
        dcr_percent = 100.0 * dcr
        ncycles = interp_linear_extrap(x_stroe, y_stroe, dcr_percent)
        if ncycles <= 0.0 or not np.isfinite(ncycles):
            continue
        damage = count / ncycles
        cost = count * ecap_cost / ncycles
        total_damage += damage
        total_cost += cost
    return (float(total_damage), float(total_cost))

def calendar_polynomial_hourly_weekly_percent(soc_week: List[float] | np.ndarray, rated_capacity: float, first_hour_index: int, commissioning_hours: float=0.0, hours_per_month: float=720.0) -> float:
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
        time_increment = (t_now / hours_per_month) ** 0.8 - (t_prev / hours_per_month) ** 0.8
        soc_percent = 100.0 * float(soc_value) / rated_capacity
        weekly_percent += time_increment * (0.0064 * soc_percent ** 0.823 + 0.1751)
    return float(weekly_percent)

AGING_REFERENCE_COST_EUR = BATTERY_CAPEX_EUR
HOURS_PER_MONTH = 720.0
CAL_COST_FACTOR = AGING_REFERENCE_COST_EUR / (0.2 * RATED_CAPACITY)
STROE_FILE = BASE_DIR / "data_stroe2017.xlsx"

def calculate_postprocess(soc_values, annual):
    x_stroe, y_stroe = read_stroe_data(STROE_FILE)
    weekly = []
    prev_rf_damage = prev_rf_cost = cal_loss = cal_cost = 0.0
    previous_end = 0
    for year in range(1, len(annual) + 1):
        year_rf_start, year_rf_cost_start = prev_rf_damage, prev_rf_cost
        year_cal_start, year_cal_cost_start = cal_loss, cal_cost
        days_simulated = int(annual[year - 1]["DaysSimulated"])
        checkpoints = list(range(7, days_simulated + 1, 7))
        if not checkpoints or checkpoints[-1] != days_simulated:
            checkpoints.append(days_simulated)
        for day in checkpoints:
            end = (year - 1) * 8760 + day * 24
            history = np.concatenate(([INITIAL_SOC], np.asarray(soc_values[:end])))
            rf_damage, rf_cost = rainflow_damage_and_cost(
                history, RATED_CAPACITY, x_stroe, y_stroe,
                AGING_REFERENCE_COST_EUR
            )
            # Non-overlapping intervals: the final update covers day 365 only.
            first_hour = previous_end - (year - 1) * 8760 + 1
            cal_percent = calendar_polynomial_hourly_weekly_percent(
                soc_values[previous_end:end], RATED_CAPACITY, first_hour,
                commissioning_hours=(year-1)*8760,
                hours_per_month=HOURS_PER_MONTH,
            )
            loss_increment = RATED_CAPACITY * cal_percent / 100.0
            cost_increment = CAL_COST_FACTOR * loss_increment
            cal_loss += loss_increment
            cal_cost += cost_increment
            total_loss = 0.20 * rf_damage * RATED_CAPACITY + cal_loss
            diagnostic_capacity = max(RATED_CAPACITY-total_loss, 0.1*RATED_CAPACITY)
            weekly.append({
                "Year": year, "DayInYear": day, "GlobalDay": (year-1)*365+day,
                "OperatingCapacity_MWh": RATED_CAPACITY,
                "PostProcessCapacity_MWh": diagnostic_capacity,
                "PostProcessSOH": diagnostic_capacity/RATED_CAPACITY,
                "Weekly_RF_Damage": rf_damage-prev_rf_damage,
                "Cumulative_RF_Damage": rf_damage,
                "Weekly_RF_Cost": rf_cost-prev_rf_cost,
                "Cumulative_RF_Cost": rf_cost,
                "Weekly_Calendar_Loss_MWh": loss_increment,
                "Cumulative_Calendar_Loss_MWh": cal_loss,
                "Weekly_Calendar_Cost": cost_increment,
                "Cumulative_Calendar_Cost": cal_cost,
                "CumulativeCapacityLoss_MWh": total_loss,
            })
            previous_end=end
            prev_rf_damage,prev_rf_cost=rf_damage,rf_cost
        annual[year-1].update({
            "RF_Damage_Increment": prev_rf_damage-year_rf_start,
            "RF_Cost_Increment_EUR": prev_rf_cost-year_rf_cost_start,
            "RF_Loss_Increment_MWh": 0.20 * (prev_rf_damage-year_rf_start)*RATED_CAPACITY,
            "CalLoss_Increment_MWh": cal_loss-year_cal_start,
            "CalCostPostProcess_EUR": cal_cost-year_cal_cost_start,
            "PostProcessFinalCapacity_MWh": diagnostic_capacity,
            "PostProcessFinalSOH": diagnostic_capacity/RATED_CAPACITY,
            "RevenueMinusPostCosts_EUR": annual[year-1]["Total_Market_Revenue_EUR"]
                -(prev_rf_cost-year_rf_cost_start)-(cal_cost-year_cal_cost_start),
        })
    return pd.DataFrame(weekly)

def main():
    if not STROE_FILE.exists():
        raise FileNotFoundError(f"Missing post-processing input: {STROE_FILE}")
    files = {
        "price": PRICE_FILE, "wind": WIND_FILE,
        "price_br2": PRICE_BR2_FILE, "ratio_br2": RATIO_BR2_FILE,
        "price_er2up": PRICE_ER2UP_FILE, "price_er2down": PRICE_ER2DOWN_FILE,
        "use_er2up": USE_ER2UP_FILE, "use_er2down": USE_ER2DOWN_FILE,
        "price_er3up": PRICE_ER3UP_FILE, "price_er3down": PRICE_ER3DOWN_FILE,
        "use_er3up": USE_ER3UP_FILE, "use_er3down": USE_ER3DOWN_FILE,
    }
    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
    # Wind file already contains the 308 MW profile. Do not rescale it.
    data = {key: read_8760_series(path) for key, path in files.items()}
    # Preserve the original R3 input treatment.
    for key in ("price_er3up", "price_er3down"):
        data[key] = np.maximum(0.0, data[key])
    for key in ("use_er3up", "use_er3down"):
        data[key] = np.minimum(1.0, data[key])

    sheet_map = {
        "soc": "SOC", "p_DA": "P_DA", "wp": "WP", "wp2b": "WP2B",
        "ed_discharged": "DISCHARGE", "ed_charged": "GRID_CHARGE_MWH",
        "pb": "PB", "br2up": "BR2_UP", "br2down": "BR2_DOWN",
        "er2up": "ER2_UP", "er2down": "ER2_DOWN",
        "br3up": "BR3_UP", "br3down": "BR3_DOWN",
        "er3up": "ER3_UP", "er3down": "ER3_DOWN",
    }
    hourly = {sheet: [] for sheet in sheet_map.values()}
    for name in ("HOUR_YEAR", "HOUR_IN_YEAR", "GLOBAL_HOUR", "CAPACITY_HOURLY",
                 "SOC_MIN_HOURLY", "SOC_MAX_HOURLY", "TOTAL_CHARGE_MWH"):
        hourly[name] = []
    daily = []
    annual = []
    soc_initial = INITIAL_SOC
    x_stroe_online, y_stroe_online = read_stroe_data(STROE_FILE)
    online_cal_loss_mwh = 0.0
    online_previous_end = 0
    stop_simulation = False
    revenue_columns = ["DA_Revenue_EUR", "R2_Capacity_Revenue_EUR",
                       "R2_Energy_Revenue_EUR", "R3_mFRR_Energy_Revenue_EUR"]
    for year in range(1, MAX_YEARS + 1):
        year_rows = []
        for day in range(1, 366):
            start = (day - 1) * IMPLEMENT
            horizon = min(T, 8760 - start)
            window = {key: values[start:start + horizon] for key, values in data.items()}
            result = solve_daily_model(
                **window, t_current=horizon,
                current_capacity=RATED_CAPACITY, soc_initial=soc_initial,
            )
            n = min(IMPLEMENT, horizon)
            implemented = {key: np.asarray(result[key][:n], dtype=float) for key in sheet_map}
            for key, sheet in sheet_map.items():
                hourly[sheet].extend(implemented[key].tolist())
            # Gross positive terms in the original SoC equation before ETA.
            # Not reconstructed physical subhourly charging when reserve offsets DA.
            charging = (implemented["wp2b"] + implemented["ed_charged"]
                        + implemented["er2down"] + implemented["er3down"])
            hourly["TOTAL_CHARGE_MWH"].extend(charging.tolist())
            hourly["HOUR_YEAR"].extend([year] * n)
            hourly["HOUR_IN_YEAR"].extend(range(start + 1, start + n + 1))
            hourly["GLOBAL_HOUR"].extend(range((year-1)*8760+start+1, (year-1)*8760+start+n+1))
            hourly["CAPACITY_HOURLY"].extend([RATED_CAPACITY] * n)
            hourly["SOC_MIN_HOURLY"].extend([0.1 * RATED_CAPACITY] * n)
            hourly["SOC_MAX_HOURLY"].extend([RATED_CAPACITY] * n)
            row = {"Year": year, "Day": day, "GlobalDay": (year-1)*365+day}
            row["DA_Revenue_EUR"] = float(np.dot(window["price"][:n], implemented["p_DA"]))
            row["R2_Capacity_Revenue_EUR"] = float(np.dot(window["price_br2"][:n], implemented["br2up"]+implemented["br2down"]))
            row["R2_Energy_Revenue_EUR"] = float(np.dot(window["price_er2up"][:n], implemented["er2up"])-np.dot(window["price_er2down"][:n], implemented["er2down"]))
            row["R3_mFRR_Energy_Revenue_EUR"] = float(np.dot(window["price_er3up"][:n], implemented["er3up"])-np.dot(window["price_er3down"][:n], implemented["er3down"]))
            row["Total_Market_Revenue_EUR"] = sum(row[key] for key in revenue_columns)
            row["Objective_EUR"] = row["Total_Market_Revenue_EUR"]
            row["GridCharge_MWh"] = float(implemented["ed_charged"].sum())
            row["WindCharge_MWh"] = float(implemented["wp2b"].sum())
            row["GrossChargingTerms_MWh"] = float(charging.sum())
            daily.append(row)
            year_rows.append(row)
            soc_initial = float(implemented["soc"][-1])
            print(f"Year {year}, day {day}: market revenue = {row['Total_Market_Revenue_EUR']:.2f} EUR")

            # Weekly online post-processing check. Aging does not change the
            # fixed 60 MWh capacity used by the optimizer; it only determines
            # when the simulated battery reaches its 80% end-of-life limit.
            if day % 7 == 0 or day == 365:
                online_end = len(hourly["SOC"])
                online_history = np.concatenate(
                    ([INITIAL_SOC], np.asarray(hourly["SOC"][:online_end]))
                )
                online_rf_damage, _online_rf_cost = rainflow_damage_and_cost(
                    online_history,
                    RATED_CAPACITY,
                    x_stroe_online,
                    y_stroe_online,
                    AGING_REFERENCE_COST_EUR,
                )
                first_hour_in_year = (
                    online_previous_end - (year - 1) * 8760 + 1
                )
                online_cal_percent = (
                    calendar_polynomial_hourly_weekly_percent(
                        hourly["SOC"][online_previous_end:online_end],
                        RATED_CAPACITY,
                        first_hour_in_year,
                        commissioning_hours=(year - 1) * 8760,
                        hours_per_month=HOURS_PER_MONTH,
                    )
                )
                online_cal_loss_mwh += (
                    RATED_CAPACITY * online_cal_percent / 100.0
                )
                online_total_loss_mwh = (
                    0.20 * online_rf_damage * RATED_CAPACITY
                    + online_cal_loss_mwh
                )
                online_capacity_mwh = max(
                    RATED_CAPACITY - online_total_loss_mwh,
                    0.0,
                )
                online_soh = online_capacity_mwh / RATED_CAPACITY
                online_previous_end = online_end

                print(
                    f"Weekly post-process SOH = {online_soh:.6f} "
                    f"(capacity = {online_capacity_mwh:.6f} MWh)"
                )

                if online_soh <= SOH_STOP_THRESHOLD:
                    print(
                        "SOH threshold reached at "
                        f"year {year}, day {day}: "
                        f"{online_soh:.6f} <= {SOH_STOP_THRESHOLD:.2f}"
                    )
                    stop_simulation = True
                    break
        summary = {"Year": year, "DaysSimulated": len(year_rows), "Capacity_MWh": RATED_CAPACITY}
        for key in revenue_columns + ["Total_Market_Revenue_EUR", "Objective_EUR", "GridCharge_MWh", "WindCharge_MWh", "GrossChargingTerms_MWh"]:
            summary[key] = sum(row[key] for row in year_rows)
        annual.append(summary)

        if stop_simulation:
            break

    weekly_df = calculate_postprocess(hourly["SOC"], annual)
    daily_df = pd.DataFrame(daily)
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        write_scalar_sheet(writer, "BATTERY_CAPEX_EUR", BATTERY_CAPEX_EUR)
        write_scalar_sheet(writer, "DISCOUNT_RATE", DISCOUNT_RATE)
        write_scalar_sheet(writer, "SOH_STOP_THRESHOLD", SOH_STOP_THRESHOLD)
        for sheet, values in hourly.items():
            write_vector_sheet(writer, sheet, values)
        daily_df.to_excel(writer, sheet_name="DAILY_SUMMARY", index=False)
        pd.DataFrame(annual).to_excel(writer, sheet_name="ANNUAL_SUMMARY", index=False)
        for sheet, column in {
            "DA_REVENUE_DAILY": "DA_Revenue_EUR",
            "R2_CAPACITY_REVENUE_DAILY": "R2_Capacity_Revenue_EUR",
            "R2_ENERGY_REVENUE_DAILY": "R2_Energy_Revenue_EUR",
            "R3_ENERGY_REVENUE_DAILY": "R3_mFRR_Energy_Revenue_EUR",
            "DAILY_PROFIT": "Total_Market_Revenue_EUR",
            "DAILY_OBJECTIVE": "Objective_EUR",
        }.items():
            write_vector_sheet(writer, sheet, daily_df[column].tolist())
        for sheet, column in {
            "TOTAL_PROFIT": "Total_Market_Revenue_EUR",
            "TOTAL_OBJECTIVE": "Objective_EUR",
            "TOTAL_DA_REVENUE": "DA_Revenue_EUR",
            "TOTAL_R2_CAPACITY_REVENUE": "R2_Capacity_Revenue_EUR",
            "TOTAL_R2_ENERGY_REVENUE": "R2_Energy_Revenue_EUR",
            "TOTAL_R3_MFRR_ENERGY_REVENUE": "R3_mFRR_Energy_Revenue_EUR",
            "TOTAL_CHARGING_ENERGY_MWH": "GrossChargingTerms_MWh",
        }.items():
            write_scalar_sheet(writer, sheet, float(daily_df[column].sum()))
        write_scalar_sheet(writer, "FINAL_CAPACITY", RATED_CAPACITY)
        weekly_df.to_excel(writer, sheet_name="WEEKLY_CAPACITY", index=False)
        for sheet, column in {
            "RF_COST_WEEKLY": "Weekly_RF_Cost",
            "RF_COST_CUMULATIVE": "Cumulative_RF_Cost",
            "RF_DAMAGE_WEEKLY": "Weekly_RF_Damage",
            "RF_DAMAGE_CUMULATIVE": "Cumulative_RF_Damage",
        }.items():
            write_vector_sheet(writer, sheet, weekly_df[column].tolist())
        last = weekly_df.iloc[-1]
        for sheet, value in {
            "TOTAL_RF_COST": last["Cumulative_RF_Cost"],
            "TOTAL_RF_DAMAGE": last["Cumulative_RF_Damage"],
            "TOTAL_RF_LOSS_MWH": 0.20 * last["Cumulative_RF_Damage"]*RATED_CAPACITY,
            "TOTAL_CAL_COST": last["Cumulative_Calendar_Cost"],
            "TOTAL_CAL_DAMAGE": last["Cumulative_Calendar_Loss_MWh"],
            "TOTAL_CAL_FRACTION": last["Cumulative_Calendar_Loss_MWh"]/RATED_CAPACITY,
            "TOTAL_CAPACITY_LOSS_MWH": last["CumulativeCapacityLoss_MWh"],
            "POST_FINAL_CAPACITY": last["PostProcessCapacity_MWh"],
            "POST_FINAL_SOH": last["PostProcessSOH"],
        }.items():
            write_scalar_sheet(writer, sheet, float(value))

    print(f"Results exported: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
