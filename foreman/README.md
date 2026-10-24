# Foreman
### AI maintenance assistant — ISE 298 Graduate Project

Khush Hemant Patel · Supervisor: Dr. Behin Elahi · San José State University

Predicts machine failure from sensor readings, says which component is at
risk and why, retrieves the matching maintenance procedure, and writes the
work order a technician acts on.

---

## Run it

```bash
pip install -r requirements.txt
python train_modes.py        # trains the four failure-mode classifiers
streamlit run app.py         # opens at localhost:8501
```

Run from the project root — the app reads `data/` and `models/` by relative path.

**To use Claude for the work orders**, set a key before launching. Without
one the app still works; a built-in writer produces the same work order
from the same inputs.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export FOREMAN_MODEL=claude-sonnet-5-5   # optional; this is the default
export ANTHROPIC_WORKSPACE_ID=wrkspc_...  # only for account-linked keys (sk-ant-usr...) not scoped to a workspace
```

Verify everything without a browser:

```bash
python test_headless.py
```

---

## The five layers

| | Layer | What happens |
|---|---|---|
| 1 | **Input** | Six sensor readings, a CSV of a shift, or a real example row from the dataset |
| 2 | **Predict** | Gradient boosting gives overall risk; four classifiers give the failure mode |
| 3 | **Explain** | Attribution ranks the readings that drove it; the mode maps to a physical component |
| 4 | **Ground** | The matching procedures are retrieved from the maintenance knowledge base |
| 5 | **Generate** | Claude writes the work order from *only* those passages; the cost model sets urgency |

The grounding layer is the point. The assistant receives the prediction,
the factors and the retrieved passages, and is instructed to use nothing
else. It cannot state a threshold that isn't in the knowledge base. The
work order names the entries it drew on, and the full prompt is visible
in the UI under the work order.

---

## Failure mode results

| Mode | Positives | Recall | Precision | PR-AUC |
|---|---|---|---|---|
| Heat dissipation (HDF) | 115 | 1.000 | 1.000 | 1.000 |
| Overstrain (OSF) | 98 | 1.000 | 0.952 | 0.998 |
| Power (PWF) | 95 | 0.947 | 1.000 | 0.986 |
| Tool wear (TWF) | 46 | 0.111 | 0.042 | 0.052 |

The first three are recovered almost exactly. That is the expected result
and the interesting one: their generating rules are written in terms of a
temperature difference, a power, and a wear-torque product — precisely the
quantities the engineered features encode. The pipeline rediscovers the
physics rather than memorising rows.

**Tool wear scores poorly on purpose.** The dataset fires that mode at a
*randomly chosen* wear time between 200 and 240 minutes, so two identical
rows can disagree. Foreman reports the wear clock directly instead of a
model probability for it.

**Random failures (RNF) are excluded entirely.** They are generated with a
flat 0.1% probability independent of every reading — unpredictable by
construction. They are about 5.6% of flagged failures, which puts a real
ceiling on achievable recall, and saying so is more honest than reporting
a meaningless score.

---

## Why nothing here is circular

- The data stays the public **AI4I 2020** set from the UCI repository. Nothing is generated and then analysed.
- The **knowledge base is authored domain content** — failure thresholds from the dataset documentation, inspection procedures condensed from published CNC troubleshooting guidance, cited per entry. It is a teaching artefact for a simulated machine, not a manufacturer's service manual, and says so.
- The **language model never touches training or evaluation.** It reads finished output and writes prose. Every number in a work order traces to a model run or a knowledge base entry.
- Using TWF/HDF/PWF/OSF as *targets* is not the leakage problem. Using them as *inputs* to predict the aggregate failure flag would be — that is why the binary pipeline drops them.

---

## Layout

```
foreman/
├── app.py                  Streamlit UI — four tabs
├── train_modes.py          Trains the four mode classifiers
├── test_headless.py        Full test suite, no browser needed
├── data/ai4i2020.csv       The dataset
├── kb/kb.json              Maintenance knowledge base
├── models/                 Trained models, scaler, feature names
└── foreman/
    ├── engine.py           Prediction, attribution, what-if, fix search
    ├── knowledge.py        Retrieval from the knowledge base
    ├── assistant.py        Claude prompt + deterministic fallback writer
    └── schematic.py        Machine diagram with the at-risk part marked
```

---

## Deploying it

Push to GitHub and connect the repo at [share.streamlit.io](https://share.streamlit.io)
— free, and gives a public URL. Add `ANTHROPIC_API_KEY` under the app's
Secrets if you want Claude writing the orders in the hosted version.

---

## Known limits

- One machine, not a fleet. No machine identity in the data.
- No time dimension. Every row is an independent cycle with no timestamp and no sequence, so sequence models (LSTM, TCN) have nothing to learn from. This is a property of the dataset, not a choice.
- Cost figures are stated assumptions — $12,000 for a missed failure, $1,500 for a planned fix, $400 for a false alarm. A sensitivity test across a 10x–100x range is in the main analysis.
- The knowledge base covers four failure modes. A real deployment would point this layer at the plant's actual manuals.
