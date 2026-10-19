import numpy as np
import pandas as pd
import rainflow
import warnings
from pathlib import Path

# در Colab، در صورت نیاز ابتدا اجرا کن:
# %pip install rainflow openpyxl

RATED_CAPACITY = 60.0       # MWh
ECAP_COST = 6_200_000.0     # EUR
EOL_SOH = 0.80
FADE_AT_EOL = 1.0 - EOL_SOH

HOURS_PER_YEAR = 8760
HOURS_PER_MONTH = 720.0

# سن باتری در ابتدای داده؛ برای باتری نو صفر
INITIAL_AGE_HOURS = 0.0

# اگر soc_input_mwh شامل حالت اولیه + N حالت پایان ساعت است:
SOC_INCLUDES_INITIAL = False

# اگر آرایه فقط SOC پایان ساعت‌هاست، مقدار واقعی شروع را اینجا بده.
# فقط اگر واقعاً از 50% شروع شده: 30.0
INITIAL_SOC_MWH = 30


INPUT_DIR = Path(__file__).resolve().parent

SOC_FILE = INPUT_DIR / "soc_sim.xlsx"
CURVE_FILE = INPUT_DIR / "data_stroe2017.xlsx"

soc_frame = pd.read_excel(SOC_FILE, sheet_name="Sheet1")
if "SOC_MWh" not in soc_frame.columns:
    raise ValueError("socR.xlsx must have a SOC_MWh column.")
soc_input_mwh = pd.to_numeric(soc_frame["SOC_MWh"], errors="raise").to_numpy()
df_stroe = pd.read_excel(CURVE_FILE, sheet_name="data_stroe")

curve = df_stroe[["x", "y"]].apply(pd.to_numeric, errors="raise")
curve = curve.dropna(how="all")
if curve.isna().any().any():
    raise ValueError("Each Stroe row must contain both x and y.")
x_stroe = curve["x"].to_numpy(dtype=float)
y_stroe = curve["y"].to_numpy(dtype=float)

print(x_stroe[:10])
print(y_stroe[:10])
OUTPUT_DIR = Path(__file__).resolve().parent

# ============================================================
# VALIDATE SOC
# ============================================================
# ============================================================
# GET SOC FROM HYDESIGN
# ============================================================



# ============================================================
# VALIDATE SOC
# ============================================================

raw_soc = np.asarray(soc_input_mwh, dtype=float).ravel()


if raw_soc.size == 0 or not np.isfinite(raw_soc).all():
    raise ValueError("SOC is empty or contains NaN/Inf.")

if np.any(raw_soc < -1e-6) or np.any(
    raw_soc > RATED_CAPACITY + 1e-6
):
    raise ValueError("SOC must be in MWh and within rated capacity.")

raw_soc = np.clip(raw_soc, 0.0, RATED_CAPACITY)

if SOC_INCLUDES_INITIAL:
    if raw_soc.size < 2:
        raise ValueError("Initial SOC plus at least one hourly state is required.")
    initial_soc = float(raw_soc[0])
    soc = raw_soc[1:]
else:
    soc = raw_soc
    initial_soc = INITIAL_SOC_MWH

if initial_soc is None:
    warnings.warn(
        "Initial SOC was not supplied: Rainflow omits the transition "
        "from the initial state to the first recorded hourly state."
    )
else:
    initial_soc = float(initial_soc)
    if not np.isfinite(initial_soc) or not 0 <= initial_soc <= RATED_CAPACITY:
        raise ValueError("Invalid initial SOC.")

# ============================================================
# VALIDATE CYCLE-LIFE CURVE
# x: cycle depth in percent; y: cycles to 80% capacity
# ============================================================

dod_curve = np.asarray(x_stroe, dtype=float).ravel()
life_curve = np.asarray(y_stroe, dtype=float).ravel()

if (
    len(dod_curve) != len(life_curve)
    or len(dod_curve) < 2
    or not np.isfinite(dod_curve).all()
    or not np.isfinite(life_curve).all()
    or np.any(life_curve <= 0)
):
    raise ValueError("Invalid Stroe curve.")

order = np.argsort(dod_curve)
dod_curve, life_curve = dod_curve[order], life_curve[order]

if np.any(np.diff(dod_curve) <= 0) or dod_curve[0] < 0 or dod_curve[-1] > 100:
    raise ValueError("Cycle depths must be unique, increasing percentages.")

def cycles_to_80(depth_percent):
    # Linear interpolation/extrapolation, matching your Python model.
    if depth_percent < dod_curve[0]:
        value = life_curve[0] + (
            (life_curve[1] - life_curve[0]) / (dod_curve[1] - dod_curve[0])
        ) * (depth_percent - dod_curve[0])
    elif depth_percent > dod_curve[-1]:
        value = life_curve[-1] + (
            (life_curve[-1] - life_curve[-2]) / (dod_curve[-1] - dod_curve[-2])
        ) * (depth_percent - dod_curve[-1])
    else:
        value = np.interp(depth_percent, dod_curve, life_curve)

    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"Invalid cycle life at DoD={depth_percent:.6f}%."
        )
    return float(value)

def cumulative_rainflow(states):
    if len(states) < 2:
        return 0.0, 0.0, 0.0

    damage = efc = cycle_count = 0.0

    for cycle_range, mean, count, start, end in rainflow.extract_cycles(states):
        if cycle_range <= 1e-6:
            continue

        depth = cycle_range / RATED_CAPACITY
        n_cycles = cycles_to_80(100.0 * depth)

        damage += count / n_cycles
        efc += count * depth
        cycle_count += count

    return damage, efc, cycle_count

# ============================================================
# CALENDAR LOSS: hourly increments, continuous battery age
# Existing nonlinear SOC model, at its assumed temperature
# ============================================================

if not np.isfinite(INITIAL_AGE_HOURS) or INITIAL_AGE_HOURS < 0:
    raise ValueError("INITIAL_AGE_HOURS must be finite and nonnegative.")
if RATED_CAPACITY <= 0 or ECAP_COST < 0:
    raise ValueError("Invalid rated capacity or battery cost.")
if not np.isclose(EOL_SOH, 0.80):
    raise ValueError("The supplied cycle-life curve is calibrated to 80% SoH.")

hours = np.arange(1, len(soc) + 1, dtype=float)
age_now = INITIAL_AGE_HOURS + hours
age_previous = age_now - 1.0

time_increment = (
    (age_now / HOURS_PER_MONTH) ** 0.8
    - (age_previous / HOURS_PER_MONTH) ** 0.8
)

soc_percent = 100.0 * soc / RATED_CAPACITY

cal_loss_percent_hourly = time_increment * (
    0.0064 * soc_percent ** 0.823 + 0.1751
)
cal_loss_mwh_hourly = (
    RATED_CAPACITY * cal_loss_percent_hourly / 100.0
)
cal_cost_hourly = (
    ECAP_COST * cal_loss_mwh_hourly
    / (FADE_AT_EOL * RATED_CAPACITY)
)

# Linear approximation used inside your optimization (comparison only).
# The nonlinear postprocessing above determines reported capacity and SoH.
cal_linear_loss_hourly = time_increment * (
    0.0028 * soc + 0.001939 * RATED_CAPACITY
)
cal_linear_cost_hourly = (
    ECAP_COST * cal_linear_loss_hourly / (FADE_AT_EOL * RATED_CAPACITY)
)

# ============================================================
# YEAR-BY-YEAR RESULTS
# Rainflow is evaluated on the complete history at each boundary.
# ============================================================

records = []
previous_damage = previous_efc = previous_count = 0.0
cumulative_cal_loss = 0.0

for start in range(0, len(soc), HOURS_PER_YEAR):
    end = min(start + HOURS_PER_YEAR, len(soc))
    year = start // HOURS_PER_YEAR + 1

    history = soc[:end]
    if initial_soc is not None:
        history = np.concatenate(([initial_soc], history))

    damage, efc, count = cumulative_rainflow(history)

    annual_damage = damage - previous_damage
    annual_rf_loss = FADE_AT_EOL * RATED_CAPACITY * annual_damage
    cumulative_rf_loss = FADE_AT_EOL * RATED_CAPACITY * damage

    annual_cal_loss = float(cal_loss_mwh_hourly[start:end].sum())
    cumulative_cal_loss += annual_cal_loss

    annual_rf_cost = ECAP_COST * annual_damage
    annual_cal_cost = float(cal_cost_hourly[start:end].sum())

    total_loss = cumulative_rf_loss + cumulative_cal_loss
    remaining = max(0.0, RATED_CAPACITY - total_loss)
    soh = remaining / RATED_CAPACITY

    records.append({
        "Year": year,
        "Hours": end - start,
        "DaysSimulated": (end - start) / 24.0,
        "InitialCapacity_MWh": RATED_CAPACITY - (
            FADE_AT_EOL * RATED_CAPACITY * previous_damage
            + cumulative_cal_loss - annual_cal_loss
        ),
        "Mean_SOC_MWh": float(soc[start:end].mean()),
        "Mean_SOC_percent": float(soc_percent[start:end].mean()),
        "Annual_RF_Damage": annual_damage,
        "Cumulative_RF_Damage": damage,
        "Annual_RF_Loss_MWh": annual_rf_loss,
        "Cumulative_RF_Loss_MWh": cumulative_rf_loss,
        "Annual_RF_Cost_EUR": annual_rf_cost,
        "Cumulative_RF_Cost_EUR": ECAP_COST * damage,
        "Annual_EFC": efc - previous_efc,
        "Cumulative_EFC": efc,
        "Annual_RF_Count": count - previous_count,
        "Cumulative_RF_Count": count,
        "Annual_CalendarLoss_percent":
            float(cal_loss_percent_hourly[start:end].sum()),
        "Annual_CalendarLoss_MWh": annual_cal_loss,
        "Cumulative_CalendarLoss_MWh": cumulative_cal_loss,
        "Annual_CalendarCost_EUR": annual_cal_cost,
        "Annual_LinearCalendarLoss_MWh": float(cal_linear_loss_hourly[start:end].sum()),
        "Annual_LinearCalendarCost_EUR": float(cal_linear_cost_hourly[start:end].sum()),
        "Cumulative_CalendarCost_EUR":
            ECAP_COST * cumulative_cal_loss
            / (FADE_AT_EOL * RATED_CAPACITY),
        "Annual_TotalCapacityLoss_MWh": annual_rf_loss + annual_cal_loss,
        "Cumulative_TotalCapacityLoss_MWh": total_loss,
        "Annual_TotalAgingCost_EUR": annual_rf_cost + annual_cal_cost,
        "Cumulative_TotalAgingCost_EUR":
            ECAP_COST * damage
            + ECAP_COST * cumulative_cal_loss
            / (FADE_AT_EOL * RATED_CAPACITY),
        "RemainingCapacity_MWh": remaining,
        "SOH": soh,
        "Below_80pct_at_period_end": soh <= EOL_SOH,
    })

    print(
        f"Year {year} | RF loss={annual_rf_loss:.6f} MWh | "
        f"Calendar loss={annual_cal_loss:.6f} MWh | "
        f"Capacity={remaining:.6f} MWh | SoH={soh:.4%}"
    )

    previous_damage = damage
    previous_efc = efc
    previous_count = count

annual_aging_df = pd.DataFrame(records)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_file = OUTPUT_DIR / "socR_RF_Calendar_Year_by_Year.xlsx"
hourly_df = pd.DataFrame({
    "Hour": hours.astype(int),
    "BatteryAge_Hours": age_now,
    "SOC_MWh": soc,
    "SOC_percent_of_initial_capacity": soc_percent,
    "CAL_DAMAGE_HOURLY_Linear_MWh": cal_linear_loss_hourly,
    "CAL_COST_HOURLY_Linear_EUR": cal_linear_cost_hourly,
    "CAL_DAMAGE_HOURLY_Postprocess_MWh": cal_loss_mwh_hourly,
    "CAL_COST_HOURLY_Postprocess_EUR": cal_cost_hourly,
    "Cumulative_CalendarLoss_Postprocess_MWh": np.cumsum(cal_loss_mwh_hourly),
})
settings_df = pd.DataFrame({
    "Parameter": ["RatedCapacity_MWh", "BatteryCost_EUR", "EOL_SOH", "InitialAge_Hours",
                  "InitialSOC_MWh", "SOC_INCLUDES_INITIAL", "Capacity_method", "Stop_at_80", "SOC_Source", "Curve_Source", "Calendar_model", "Time_step_hours"],
    "Value": [RATED_CAPACITY, ECAP_COST, EOL_SOH, INITIAL_AGE_HOURS,
              initial_soc, SOC_INCLUDES_INITIAL,
              "0.2 * E0 * cumulative RF damage + nonlinear calendar loss", False, str(SOC_FILE.resolve()), str(CURVE_FILE.resolve()), "dt_month^0.8 * (0.0064 * SOC_percent^0.823 + 0.1751); percent to MWh", 1.0],
})
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    annual_aging_df.to_excel(writer, sheet_name="ANNUAL_AGING", index=False)
    # Split large exports before reaching Excel's row limit.
    for part, offset in enumerate(range(0, len(hourly_df), 1_000_000), 1):
        hourly_df.iloc[offset:offset+1_000_000].to_excel(
            writer, sheet_name=f"HOURLY_AGING_{part}", index=False
        )
    settings_df.to_excel(writer, sheet_name="SETTINGS", index=False)

print(f"\nSaved: {output_file.resolve()}")
print("All supplied hours were processed; no automatic cutoff or replacement reset.")
print("Reported capacity is diagnostic and does not alter the supplied SOC.")