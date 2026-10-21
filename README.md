# CARDIOGUARD — Live Pacemaker Control Simulation

Educational Streamlit dashboard for synthetic ECG-like input scenarios and a simplified closed-loop PID model.

## Run locally

```bash
py -m pip install --upgrade streamlit
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`). Stop the previous app with Ctrl+C before launching this version.

## Demo scenarios
- Normal rhythm
- Bradycardia
- Missed beats
- Irregular rhythm

Change the scenario, controller (ABC/DE/GAS), or target. The model-driven display transition restarts from the selected controller's unit-step response and runs over several seconds. Use Reset simulation to restart the simulation clock.

## Important limitations
This is a software-only educational demonstration. All ECG-like signals are synthetic. The transfer function is a simplified control-system model, not a validated physiological heart model. The output-rate mapping is illustrative, not a clinical pacing law. It does not read a real ECG, detect a patient's rhythm, or control a pacemaker. Do not connect it to a person or use it for diagnosis or treatment.
