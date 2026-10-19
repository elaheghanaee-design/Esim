# BESS Multi-Year Optimization — Pyomo + HiGHS

This repository is a Python translation of the Julia/JuMP model
`DA_only_multiyear_25y.jl`.

## Solver

The model uses **Pyomo + HiGHS**, so no CPLEX/Gurobi license is required.

## Files

Put these three input files in the repository root:

- `priceSPOT.xlsx`
- `WindFarm.xlsx`
- `data_stroe2017.xlsx`

The script expects:

- price data in `priceSPOT.xlsx`, sheet `Sheet1`, range `C2:C8761`
- wind data in `WindFarm.xlsx`, sheet `Sheet1`, range `C2:C8761`
- degradation curve in `data_stroe2017.xlsx`, sheet `data_stroe`, range `A2:B56`

## Run in GitHub Codespaces

Open the repository in Codespaces, then run:

```bash
pip install -r requirements.txt
python main.py
```

The result workbook will be written as:

```text
results_multiyear_DA_beta_0_50_80_100.xlsx
```

## Quick test

Before running all 25 years, edit:

```python
MAX_YEARS = 1
```

This is strongly recommended for the first test.

## Notes

The model preserves the original logic:

- 48-hour optimization horizon
- first 24 hours implemented
- day-ahead revenue
- wind-to-battery charging
- no grid charging
- cycle-aging penalty inside MILP
- Rainflow degradation post-processing
- nonlinear calendar-aging post-processing
- weekly battery-capacity update
- stop at SOH threshold
