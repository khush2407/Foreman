# AI-Enabled Predictive Maintenance Using Explainable Machine Learning

ISE 298-05 Graduate Project · Khush Hemant Patel · Supervisor: Dr. Behin Elahi · San José State University

Predicts machine failure from six sensor readings, explains which reading caused it, marks the part to inspect, and has Claude write a maintenance work order grounded in a maintenance knowledge base. A reliability layer models tool life, the best time to change the tool, a reliability block diagram of the machine, and how failures of different parts depend on each other.

## What's in this repository

| Folder | What it is |
|---|---|
| `pdm/` | The analysis pipeline: 10 stages from data understanding to reliability, root cause, spare parts and the production line. `bash run_all.sh` runs everything and writes figures and tables to `pdm/outputs/`. |
| `foreman/` | **Foreman**, the web app: Single cycle, Score a shift, Reliability, Demand & spares, Production line, How it works. |

## Results (AI4I 2020, 10,000 cycles)

- XGBoost at a cost-optimal alarm threshold of 0.42: 91.2% of failures caught, 45.9% precision
- Projected cost $971,000 a year against $4,080,000 for run-to-failure: $3,109,000 saved (76.2%)
- 120 tool lifetimes in the data: Weibull shape 18.4 (strong wear-out), B10 life 206 min, cost-optimal tool change at about 179 min

## Run it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r foreman/requirements.txt
cd foreman
python3 test_headless.py        # should end with ALL CHECKS PASSED
streamlit run app.py
```

To let Claude write the work orders, set `ANTHROPIC_API_KEY` (and `ANTHROPIC_WORKSPACE_ID` for account-linked keys) before launching. Without a key, a built-in writer produces the same work order.

## Online version

Deployed on Streamlit Community Cloud with `foreman/app.py` as the main file. Keys go in the app's Secrets (see `.streamlit/secrets.toml.example`), never in the code.

## Data

AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository (Matzka, 2020), CC BY 4.0.
