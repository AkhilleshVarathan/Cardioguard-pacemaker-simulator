# CARDIOGUARD — Full Analysis Dashboard

## Run locally
```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Deploy
Replace the existing deployed `app.py` with this version and ensure `requirements.txt` is committed. Streamlit Cloud will rebuild the app.

## Included
- Four synthetic rhythm scenarios and live input/output ECG-like traces
- Selectable ABC, DE and GAS controller gain sets
- Selected-controller metrics: MSE, rise time, settling time, overshoot, peak, peak time, stability, objective cost
- Selected step response and tracking-error graph
- All-controller response overlay
- Comparison table and cost/MSE/overshoot/settling/rise-time charts
- PID gain comparison, best-per-metric table, closed-loop poles, model equations and CSV export

## Important
Educational simulation only. ECG-like waveforms are synthetic, the plant is simplified, and BPM mapping is illustrative. GAS was demonstrated with Qiskit simulation and does not establish quantum speedup. Not for clinical use.
