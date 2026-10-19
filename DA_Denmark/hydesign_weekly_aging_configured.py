"""Weekly Rainflow and nonlinear calendar-aging analysis for HyDesign SOC output.

Examples
--------
SOC in MWh:
    python hydesign_weekly_rainflow_calendar_aging.py hydesign_output.xlsx \
        --soc-column SOC --soc-unit mwh --rated-capacity 60 \
        --cycle-curve data_stroe2017.xlsx

SOC in percent:
    python hydesign_weekly_rainflow_calendar_aging.py hydesign_output.csv \
        --soc-column battery_soc --soc-unit percent --rated-capacity 60 \
        --cycle-curve data_stroe2017.xlsx

Required packages:
    pip install numpy pandas openpyxl rainflow
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import rainflow


HOURS_PER_WEEK = 168
HOURS_PER_YEAR = 8760
HOURS_PER_MONTH = 720.0


def read_table(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported input format: {suffix}")


def select_soc_column(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is not None:
        if column in df.columns:
            return df[column]
        try:
            return df.iloc[:, int(column)]
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"SOC column {column!r} was not found. Available columns: "
                f"{list(df.columns)}"
            ) from exc

    candidates = [
        c for c in df.columns
        if "soc" in str(c).lower().replace("_", "")
        or "stateofcharge" in str(c).lower().replace("_", "").replace(" ", "")
    ]
    if len(candidates) == 1:
        return df[candidates[0]]
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    raise ValueError(
        "Could not identify the SOC column uniquely. Use --soc-column NAME. "
        f"Available columns: {list(df.columns)}"
    )


def prepare_soc_mwh(
    values: Iterable[float],
    unit: str,
    rated_capacity: float,
) -> np.ndarray:
    soc = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    if np.isnan(soc).any():
        bad = np.flatnonzero(np.isnan(soc))[:10] + 1
        raise ValueError(f"SOC contains missing/non-numeric values at rows {bad.tolist()}.")

    if unit == "percent":
        soc = soc * rated_capacity / 100.0
    elif unit == "pu":
        soc = soc * rated_capacity
    elif unit != "mwh":
        raise ValueError("SOC unit must be mwh, percent, or pu.")

    tolerance = 1e-6 * max(1.0, rated_capacity)
    if np.any(soc < -tolerance) or np.any(soc > rated_capacity + tolerance):
        raise ValueError(
            f"SOC must be between 0 and rated capacity ({rated_capacity} MWh). "
            f"Observed range: {soc.min()} to {soc.max()} MWh."
        )
    return np.clip(soc, 0.0, rated_capacity)


def read_cycle_curve(path: Path, sheet_name: str) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        raw = pd.read_excel(path, sheet_name=sheet_name, usecols="A:B")
    else:
        raw = read_table(path).iloc[:, :2]

    x = pd.to_numeric(raw.iloc[:, 0], errors="coerce").to_numpy(float)
    y = pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        raise ValueError("The cycle curve must contain at least two valid points.")
    order = np.argsort(x)
    return x[order], y[order]


def interp_linear_extrap(x: np.ndarray, y: np.ndarray, query: float) -> float:
    if query <= x[0]:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return float(y[0] + slope * (query - x[0]))
    if query >= x[-1]:
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return float(y[-1] + slope * (query - x[-1]))
    return float(np.interp(query, x, y))


def clean_soc_profile(soc: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    if len(soc) == 0:
        return soc.copy()
    keep = np.r_[True, np.abs(np.diff(soc)) > tolerance]
    return soc[keep]


def cumulative_rainflow(
    soc_history_mwh: np.ndarray,
    rated_capacity: float,
    curve_dod_percent: np.ndarray,
    curve_cycles: np.ndarray,
    replacement_cost: float,
) -> tuple[float, float, float, float, float]:
    """Return damage, capacity loss, cost, counted cycles, and EFC."""
    profile = clean_soc_profile(soc_history_mwh)
    if len(profile) < 2 or np.ptp(profile) < 1e-6:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    damage = 0.0
    counted_cycles = 0.0
    equivalent_full_cycles = 0.0
    for cycle_range, _mean, count, _start, _end in rainflow.extract_cycles(profile):
        if cycle_range <= 1e-6 or count <= 0:
            continue
        dod_percent = 100.0 * float(cycle_range) / rated_capacity
        cycle_life = interp_linear_extrap(
            curve_dod_percent,
            curve_cycles,
            dod_percent,
        )
        if np.isfinite(cycle_life) and cycle_life > 0:
            damage += float(count) / cycle_life
        counted_cycles += float(count)
        equivalent_full_cycles += (
            float(count) * float(cycle_range) / rated_capacity
        )

    capacity_loss = damage * rated_capacity
    cost = damage * replacement_cost
    return damage, capacity_loss, cost, counted_cycles, equivalent_full_cycles


def rainflow_by_week(
    soc_mwh: np.ndarray,
    initial_soc_mwh: float | None,
    rated_capacity: float,
    curve_dod_percent: np.ndarray,
    curve_cycles: np.ndarray,
    replacement_cost: float,
) -> dict[str, np.ndarray]:
    """Run Rainflow once and assign each extracted cycle to its ending week."""
    prefix = (
        np.array([initial_soc_mwh], dtype=float)
        if initial_soc_mwh is not None
        else np.empty(0, dtype=float)
    )
    raw = np.concatenate([prefix, soc_mwh])
    keep = np.r_[True, np.abs(np.diff(raw)) > 1e-6]
    profile = raw[keep]
    original_indices = np.flatnonzero(keep)
    number_of_weeks = int(np.ceil(len(soc_mwh) / HOURS_PER_WEEK))
    number_of_years = int(np.ceil(len(soc_mwh) / HOURS_PER_YEAR))
    damage = np.zeros(number_of_weeks)
    counted = np.zeros(number_of_weeks)
    efc = np.zeros(number_of_weeks)
    damage_year = np.zeros(number_of_years)
    counted_year = np.zeros(number_of_years)
    efc_year = np.zeros(number_of_years)

    if len(profile) >= 2 and np.ptp(profile) >= 1e-6:
        hour_offset = 0 if initial_soc_mwh is not None else 1
        for cycle_range, _mean, count, _start, end_index in rainflow.extract_cycles(profile):
            if cycle_range <= 1e-6 or count <= 0:
                continue
            ending_hour = int(original_indices[int(end_index)] + hour_offset)
            ending_hour = min(max(ending_hour, 1), len(soc_mwh))
            week_index = (ending_hour - 1) // HOURS_PER_WEEK
            year_index = (ending_hour - 1) // HOURS_PER_YEAR
            dod_percent = 100.0 * float(cycle_range) / rated_capacity
            cycle_life = interp_linear_extrap(
                curve_dod_percent, curve_cycles, dod_percent
            )
            if np.isfinite(cycle_life) and cycle_life > 0:
                cycle_damage = float(count) / cycle_life
                damage[week_index] += cycle_damage
                damage_year[year_index] += cycle_damage
            counted[week_index] += float(count)
            counted_year[year_index] += float(count)
            cycle_efc = float(count) * float(cycle_range) / rated_capacity
            efc[week_index] += cycle_efc
            efc_year[year_index] += cycle_efc

    return {
        "damage": damage,
        "counted": counted,
        "efc": efc,
        "cost": damage * replacement_cost,
        "capacity_loss": damage * rated_capacity,
        "damage_year": damage_year,
        "counted_year": counted_year,
        "efc_year": efc_year,
    }


def nonlinear_calendar_hourly(
    soc_mwh: np.ndarray,
    rated_capacity: float,
    starting_age_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return hourly calendar loss in percent and MWh.

    Formula:
      delta(t^0.8) * (0.0064 * SOC_percent^0.823 + 0.1751)
    with time measured in 720-hour months.
    """
    hour_now = starting_age_hours + np.arange(1, len(soc_mwh) + 1, dtype=float)
    hour_prev = hour_now - 1.0
    time_increment = (
        (hour_now / HOURS_PER_MONTH) ** 0.8
        - (hour_prev / HOURS_PER_MONTH) ** 0.8
    )
    soc_percent = 100.0 * soc_mwh / rated_capacity
    loss_percent = time_increment * (
        0.0064 * soc_percent ** 0.823 + 0.1751
    )
    loss_mwh = rated_capacity * loss_percent / 100.0
    return loss_percent, loss_mwh


def analyse_weekly(
    soc_mwh: np.ndarray,
    rated_capacity: float,
    curve_dod_percent: np.ndarray,
    curve_cycles: np.ndarray,
    replacement_cost: float,
    starting_age_hours: float,
    initial_soc_mwh: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calendar_cost_factor = replacement_cost / (0.2 * rated_capacity)
    cal_percent_hourly, cal_mwh_hourly = nonlinear_calendar_hourly(
        soc_mwh,
        rated_capacity,
        starting_age_hours,
    )

    hourly = pd.DataFrame({
        "Hour": np.arange(1, len(soc_mwh) + 1),
        "BatteryAgeHours": starting_age_hours + np.arange(1, len(soc_mwh) + 1),
        "SOC_MWh": soc_mwh,
        "SOC_percent": 100.0 * soc_mwh / rated_capacity,
        "CalendarLoss_percent": cal_percent_hourly,
        "CalendarLoss_MWh": cal_mwh_hourly,
        "CalendarCost_EUR": cal_mwh_hourly * calendar_cost_factor,
    })

    rf = rainflow_by_week(
        soc_mwh,
        initial_soc_mwh,
        rated_capacity,
        curve_dod_percent,
        curve_cycles,
        replacement_cost,
    )
    cumulative_calendar_mwh = 0.0
    rows: list[dict[str, float | int]] = []

    for week, start in enumerate(range(0, len(soc_mwh), HOURS_PER_WEEK), start=1):
        end = min(start + HOURS_PER_WEEK, len(soc_mwh))
        weekly_rf_damage = float(rf["damage"][week - 1])
        weekly_counted_cycles = float(rf["counted"][week - 1])
        weekly_efc = float(rf["efc"][week - 1])
        rf_damage = float(rf["damage"][:week].sum())
        counted_cycles = float(rf["counted"][:week].sum())
        equivalent_full_cycles = float(rf["efc"][:week].sum())
        rf_loss_mwh = float(rf["capacity_loss"][:week].sum())
        rf_cost = float(rf["cost"][:week].sum())
        weekly_cal_mwh = float(cal_mwh_hourly[start:end].sum())
        weekly_cal_percent = float(cal_percent_hourly[start:end].sum())
        cumulative_calendar_mwh += weekly_cal_mwh
        weekly_calendar_cost = weekly_cal_mwh * calendar_cost_factor
        cumulative_calendar_cost = cumulative_calendar_mwh * calendar_cost_factor
        weekly_rf_cost = weekly_rf_damage * replacement_cost

        rows.append({
            "Week": week,
            "StartHour": start + 1,
            "EndHour": end,
            "HoursInWeek": end - start,
            "MeanSOC_MWh": float(soc_mwh[start:end].mean()),
            "MeanSOC_percent": float(100.0 * soc_mwh[start:end].mean() / rated_capacity),
            "Weekly_RF_Damage": weekly_rf_damage,
            "Cumulative_RF_Damage": rf_damage,
            "Weekly_RainflowCycleCount": weekly_counted_cycles,
            "Cumulative_RainflowCycleCount": counted_cycles,
            "Weekly_EquivalentFullCycles": weekly_efc,
            "Cumulative_EquivalentFullCycles": equivalent_full_cycles,
            "Weekly_RF_CapacityLoss_MWh": weekly_rf_damage * rated_capacity,
            "Cumulative_RF_CapacityLoss_MWh": rf_loss_mwh,
            "Weekly_RF_Cost_EUR": weekly_rf_cost,
            "Cumulative_RF_Cost_EUR": rf_cost,
            "Weekly_CalendarLoss_percent": weekly_cal_percent,
            "Weekly_CalendarLoss_MWh": weekly_cal_mwh,
            "Cumulative_CalendarLoss_MWh": cumulative_calendar_mwh,
            "Weekly_CalendarCost_EUR": weekly_calendar_cost,
            "Cumulative_CalendarCost_EUR": cumulative_calendar_cost,
            "Weekly_TotalAgingCost_EUR": weekly_rf_cost + weekly_calendar_cost,
            "Cumulative_TotalAgingCost_EUR": rf_cost + cumulative_calendar_cost,
            "Total_CumulativeCapacityLoss_MWh": rf_loss_mwh + cumulative_calendar_mwh,
            "RemainingCapacity_MWh": max(
                0.0,
                rated_capacity - rf_loss_mwh - cumulative_calendar_mwh,
            ),
            "SOH": max(
                0.0,
                1.0 - rf_loss_mwh / rated_capacity
                - cumulative_calendar_mwh / rated_capacity,
            ),
        })
    annual_rows: list[dict[str, float | int]] = []
    cumulative_rf_loss = 0.0
    cumulative_cal_loss = 0.0
    for year, start in enumerate(range(0, len(soc_mwh), HOURS_PER_YEAR), start=1):
        end = min(start + HOURS_PER_YEAR, len(soc_mwh))
        annual_rf_damage = float(rf["damage_year"][year - 1])
        annual_rf_loss = annual_rf_damage * rated_capacity
        annual_cal_loss = float(cal_mwh_hourly[start:end].sum())
        annual_cal_percent = float(cal_percent_hourly[start:end].sum())
        cumulative_rf_loss += annual_rf_loss
        cumulative_cal_loss += annual_cal_loss
        total_cumulative_loss = cumulative_rf_loss + cumulative_cal_loss
        annual_rf_cost = annual_rf_damage * replacement_cost
        cumulative_rf_cost = float(rf["damage_year"][:year].sum()) * replacement_cost
        annual_calendar_cost = annual_cal_loss * calendar_cost_factor
        cumulative_calendar_cost = cumulative_cal_loss * calendar_cost_factor
        annual_rows.append({
            "Year": year,
            "StartHour": start + 1,
            "EndHour": end,
            "HoursInYear": end - start,
            "MeanSOC_MWh": float(soc_mwh[start:end].mean()),
            "MeanSOC_percent": float(100.0 * soc_mwh[start:end].mean() / rated_capacity),
            "Annual_RainflowCycleCount": float(rf["counted_year"][year - 1]),
            "Cumulative_RainflowCycleCount": float(rf["counted_year"][:year].sum()),
            "Annual_EquivalentFullCycles": float(rf["efc_year"][year - 1]),
            "Cumulative_EquivalentFullCycles": float(rf["efc_year"][:year].sum()),
            "Annual_RF_Damage": annual_rf_damage,
            "Cumulative_RF_Damage": float(rf["damage_year"][:year].sum()),
            "Annual_RF_CapacityLoss_MWh": annual_rf_loss,
            "Cumulative_RF_CapacityLoss_MWh": cumulative_rf_loss,
            "Annual_RF_Cost_EUR": annual_rf_cost,
            "Cumulative_RF_Cost_EUR": cumulative_rf_cost,
            "Annual_CalendarLoss_percent": annual_cal_percent,
            "Annual_CalendarLoss_MWh": annual_cal_loss,
            "Cumulative_CalendarLoss_MWh": cumulative_cal_loss,
            "Annual_CalendarCost_EUR": annual_calendar_cost,
            "Cumulative_CalendarCost_EUR": cumulative_calendar_cost,
            "Annual_TotalAgingCost_EUR": annual_rf_cost + annual_calendar_cost,
            "Cumulative_TotalAgingCost_EUR": cumulative_rf_cost + cumulative_calendar_cost,
            "Annual_TotalCapacityLoss_MWh": annual_rf_loss + annual_cal_loss,
            "Cumulative_TotalCapacityLoss_MWh": total_cumulative_loss,
            "RemainingCapacity_MWh": max(0.0, rated_capacity - total_cumulative_loss),
            "SOH": max(0.0, 1.0 - total_cumulative_loss / rated_capacity),
        })

    return hourly, pd.DataFrame(rows), pd.DataFrame(annual_rows)


def get_config() -> SimpleNamespace:
    """Edit only this block when file names or battery data change."""
    folder = Path(__file__).resolve().parent
    return SimpleNamespace(
        input=folder / "socR.xlsx",
        soc_column="SoC_MWh",
        soc_unit="mwh",             # mwh, percent, or pu
        rated_capacity=60.0,         # MWh
        cycle_curve=folder / "data_stroe2017.xlsx",
        input_sheet=0,
        curve_sheet="data_stroe",
        replacement_cost=6_200_000.0,  # EUR
        starting_age_hours=0.0,
        initial_soc=None,            # Hour=0 is detected automatically
        output=folder / "Fassbatnocharge den.xlsx",
    )


def main() -> None:
    args = get_config()
    if args.rated_capacity <= 0:
        raise ValueError("--rated-capacity must be positive.")
    if args.starting_age_hours < 0:
        raise ValueError("--starting-age-hours cannot be negative.")
    if args.replacement_cost < 0:
        raise ValueError("--replacement-cost cannot be negative.")

    input_sheet: str | int
    try:
        input_sheet = int(args.input_sheet)
    except (TypeError, ValueError):
        input_sheet = args.input_sheet

    source = read_table(args.input, input_sheet)
    soc_series = select_soc_column(source, args.soc_column)
    soc_mwh = prepare_soc_mwh(soc_series, args.soc_unit, args.rated_capacity)
    if len(soc_mwh) == 0:
        raise ValueError("The selected SOC column is empty.")

    initial_soc_mwh = None
    if args.initial_soc is not None:
        initial_soc_mwh = float(prepare_soc_mwh(
            [args.initial_soc], args.soc_unit, args.rated_capacity
        )[0])
    elif "Hour" in source.columns:
        hours = pd.to_numeric(source["Hour"], errors="coerce").to_numpy(float)
        if len(hours) == len(soc_mwh) and len(hours) > 1 and hours[0] == 0:
            initial_soc_mwh = float(soc_mwh[0])
            soc_mwh = soc_mwh[1:]

    curve_dod, curve_life = read_cycle_curve(args.cycle_curve, args.curve_sheet)
    hourly, weekly, annual = analyse_weekly(
        soc_mwh,
        args.rated_capacity,
        curve_dod,
        curve_life,
        args.replacement_cost,
        args.starting_age_hours,
        initial_soc_mwh,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        hourly.to_excel(writer, sheet_name="HOURLY", index=False)
        weekly.to_excel(writer, sheet_name="WEEKLY_SUMMARY", index=False)
        annual.to_excel(writer, sheet_name="ANNUAL_SUMMARY", index=False)
        pd.DataFrame({
            "Metric": [
                "RatedCapacity_MWh",
                "HoursAnalysed",
                "WeeksAnalysed",
                "Final_RF_Damage",
                "Final_RainflowCycleCount",
                "Final_EquivalentFullCycles",
                "Final_RF_CapacityLoss_MWh",
                "Final_RF_Cost_EUR",
                "Final_CalendarLoss_MWh",
                "Final_CalendarCost_EUR",
                "Final_TotalAgingCost_EUR",
                "Final_TotalCapacityLoss_MWh",
                "Final_RemainingCapacity_MWh",
                "Final_SOH",
            ],
            "Value": [
                args.rated_capacity,
                len(hourly),
                len(weekly),
                weekly.iloc[-1]["Cumulative_RF_Damage"],
                weekly.iloc[-1]["Cumulative_RainflowCycleCount"],
                weekly.iloc[-1]["Cumulative_EquivalentFullCycles"],
                weekly.iloc[-1]["Cumulative_RF_CapacityLoss_MWh"],
                weekly.iloc[-1]["Cumulative_RF_Cost_EUR"],
                weekly.iloc[-1]["Cumulative_CalendarLoss_MWh"],
                weekly.iloc[-1]["Cumulative_CalendarCost_EUR"],
                weekly.iloc[-1]["Cumulative_TotalAgingCost_EUR"],
                weekly.iloc[-1]["Total_CumulativeCapacityLoss_MWh"],
                weekly.iloc[-1]["RemainingCapacity_MWh"],
                weekly.iloc[-1]["SOH"],
            ],
        }).to_excel(writer, sheet_name="TOTALS", index=False)

    print(
        f"Analysed {len(hourly)} hourly SOC values in "
        f"{len(weekly)} weeks and {len(annual)} years."
    )
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
