"""Wind-only reference revenue for battery NPV comparisons.

The script uses the same hourly price, wind profile, and 300 MW export limit
as the Wind+BESS model.  There is no battery and therefore no charging,
discharging, SoC, degradation, or optimization cost.

One run evaluates both battery lifetimes currently being compared:
    1) Post-process-only degradation: 2 full years + 259 days
    2) Degradation-aware optimization: 3 full years + 231 days
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

PRICE_FILE = BASE_DIR / "priceSPOTDan.xlsx"
WIND_FILE = BASE_DIR / "WindFarmR3_scaled_308_over_35.xlsx"
OUTPUT_FILE = BASE_DIR / "wind_only_reference_revenue_Spain_DA aFRR mFRR.xlsx"

PRICE_SHEET = "Sheet1"
WIND_SHEET = "Sheet1"
PRICE_COLUMN = "C"
WIND_COLUMN = "C"

HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365
HOURS_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR
NODE_CAPACITY_G = 300.0
DISCOUNT_RATE = 0.09

# Run the wind-only reference over the exact lifetime of each battery case.
SCENARIOS = {
    "postprocess_only": (5, 273),
    "degradation_in_optimization": (6, 315),
}


def read_hourly_column(
    path: Path,
    sheet_name: str,
    column: str,
) -> np.ndarray:
    values = pd.read_excel(
        path,
        sheet_name=sheet_name,
        usecols=column,
        skiprows=1,
        nrows=HOURS_PER_YEAR,
        header=None,
        engine="openpyxl",
    ).iloc[:, 0]

    values = pd.to_numeric(values, errors="coerce")
    if values.isna().any():
        raise ValueError(f"Missing or nonnumeric values found in {path.name}")
    if len(values) != HOURS_PER_YEAR:
        raise ValueError(
            f"{path.name} must contain {HOURS_PER_YEAR} hourly values; "
            f"found {len(values)}."
        )
    return values.to_numpy(dtype=float)


def simulate_wind_only(
    scenario: str,
    full_years: int,
    final_year_days: int,
    price: np.ndarray,
    wind: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not 0 <= final_year_days <= DAYS_PER_YEAR:
        raise ValueError("final_year_days must be between 0 and 365")

    total_days = full_years * DAYS_PER_YEAR + final_year_days
    total_hours = total_days * HOURS_PER_DAY

    hour_index = np.arange(total_hours, dtype=int)
    source_hour = hour_index % HOURS_PER_YEAR
    year = hour_index // HOURS_PER_YEAR + 1
    hour_in_year = source_hour + 1
    day_in_year = source_hour // HOURS_PER_DAY + 1

    price_used = price[source_hour]
    available_wind = np.maximum(wind[source_hour], 0.0)

    # This reproduces the wind-only optimum of the original model:
    # export up to the grid limit at positive prices and curtail at negative
    # prices. At a zero price, export choice has no effect on revenue, so zero
    # is used for a clean and deterministic reference.
    wind_export = np.where(
        price_used > 0.0,
        np.minimum(available_wind, NODE_CAPACITY_G),
        0.0,
    )
    curtailed_wind = available_wind - wind_export
    hourly_revenue = price_used * wind_export

    hourly = pd.DataFrame(
        {
            "Scenario": scenario,
            "GlobalHour": hour_index + 1,
            "Year": year,
            "HourInYear": hour_in_year,
            "DayInYear": day_in_year,
            "Price_EUR_per_MWh": price_used,
            "WindAvailable_MW": available_wind,
            "WindExport_MW": wind_export,
            "WindCurtailed_MW": curtailed_wind,
            "WindOnlyRevenue_EUR": hourly_revenue,
        }
    )

    annual = (
        hourly.groupby("Year", as_index=False)
        .agg(
            DaysSimulated=("DayInYear", "max"),
            WindAvailable_MWh=("WindAvailable_MW", "sum"),
            WindExport_MWh=("WindExport_MW", "sum"),
            WindCurtailed_MWh=("WindCurtailed_MW", "sum"),
            WindOnlyRevenue_EUR=("WindOnlyRevenue_EUR", "sum"),
        )
    )
    annual.insert(0, "Scenario", scenario)
    annual["DiscountTime_years"] = np.where(
        annual["DaysSimulated"] < DAYS_PER_YEAR,
        annual["Year"] - 1 + annual["DaysSimulated"] / DAYS_PER_YEAR,
        annual["Year"],
    )
    annual["PV_WindOnlyRevenue_EUR"] = (
        annual["WindOnlyRevenue_EUR"]
        / (1.0 + DISCOUNT_RATE) ** annual["DiscountTime_years"]
    )

    summary = {
        "Scenario": scenario,
        "FullYears": full_years,
        "FinalYearDays": final_year_days,
        "LifetimeYears": full_years + final_year_days / DAYS_PER_YEAR,
        "TotalDays": total_days,
        "WindOnlyRevenue_EUR": float(hourly_revenue.sum()),
        "PV_WindOnlyRevenue_EUR": float(
            annual["PV_WindOnlyRevenue_EUR"].sum()
        ),
        "DiscountRate": DISCOUNT_RATE,
    }
    return hourly, annual, summary


def main() -> None:
    for required_file in (PRICE_FILE, WIND_FILE):
        if not required_file.exists():
            raise FileNotFoundError(
                f"Missing {required_file.name}. Put it beside this script."
            )

    price = read_hourly_column(PRICE_FILE, PRICE_SHEET, PRICE_COLUMN)
    wind = read_hourly_column(WIND_FILE, WIND_SHEET, WIND_COLUMN)

    all_annual = []
    all_summaries = []
    hourly_outputs = {}

    for scenario, (full_years, final_year_days) in SCENARIOS.items():
        hourly, annual, summary = simulate_wind_only(
            scenario,
            full_years,
            final_year_days,
            price,
            wind,
        )
        hourly_outputs[scenario] = hourly
        all_annual.append(annual)
        all_summaries.append(summary)

    annual_summary = pd.concat(all_annual, ignore_index=True)
    scenario_summary = pd.DataFrame(all_summaries)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        scenario_summary.to_excel(
            writer, sheet_name="SCENARIO_SUMMARY", index=False
        )
        annual_summary.to_excel(
            writer, sheet_name="ANNUAL_SUMMARY", index=False
        )
        for scenario, hourly in hourly_outputs.items():
            sheet_name = (
                "HOURLY_POSTPROCESS"
                if scenario == "postprocess_only"
                else "HOURLY_DEG_OPT"
            )
            hourly.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Wind-only reference calculation completed.")
    print(f"Output: {OUTPUT_FILE}")
    print(scenario_summary.to_string(index=False))


if __name__ == "__main__":
    main()
