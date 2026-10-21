import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import TransferFunction, step

st.set_page_config(page_title="CARDIOGUARD | Pacemaker Simulation", page_icon="💓", layout="wide")
GAINS = {
    "ABC": (2.460736, 0.120961, 0.600000),
    "DE": (1.5011866853102536, 0.33717359960085164, 0.37848409731959337),
    "GAS": (1.500000, 0.350000, 0.40714285714285714),
}
CONTROLLER_LABELS = {
    "ABC": "Existing ABC baseline",
    "DE": "Differential Evolution (classical)",
    "GAS": "Grover Adaptive Search (Qiskit simulation)",
}
T_FINAL = 4.0
DT = 0.001
MSE_SCALE = 0.02

st.markdown("""<style>
.stApp{background:#07131b;color:#e8f4fb}
[data-testid="stMetric"]{background:#10232d;border:1px solid #244451;padding:12px;border-radius:12px}
[data-testid="stMetricLabel"]{color:#9ab6c4}
[data-testid="stMetricValue"]{color:#e8f4fb}
.block-container{padding-top:1.1rem}
h1,h2,h3{color:#e8f4fb}
</style>""", unsafe_allow_html=True)

st.title("💓 CARDIOGUARD")
st.caption("MODEL-DRIVEN CARDIAC PACEMAKER DIGITAL SIMULATION")
st.warning(
    "SIMULATION ONLY — not a medical device, not for diagnosis or treatment, and never connect to a person. "
    "All ECG-like signals are synthetic. The transfer function is a simplified control-system model, "
    "not a validated physiological heart model."
)

with st.sidebar:
    st.header("Simulation controls")
    scenario = st.selectbox("Input rhythm", ["Normal rhythm", "Bradycardia", "Missed beats", "Irregular rhythm"])
    controller = st.selectbox("Controller gain set", ["ABC", "DE", "GAS"], index=1)
    threshold = st.slider("Demo bradycardia threshold (BPM)", 35, 70, 50)
    target = st.slider("Demo output target (BPM)", 50, 90, 60)
    speed = st.slider("Simulation speed", 0.5, 2.0, 1.0, 0.25)
    running = st.toggle("▶ Run live simulation", value=True)
    reset_clicked = st.button("↻ Reset simulation", use_container_width=True)
    if reset_clicked:
        st.session_state.sim_epoch = time.monotonic()
        st.session_state.paused_elapsed = 0.0
        st.session_state.transition_start_elapsed = 0.0
    st.caption("These are demonstration settings, not clinical recommendations.")

if "sim_epoch" not in st.session_state:
    st.session_state.sim_epoch = time.monotonic()
if "paused_elapsed" not in st.session_state:
    st.session_state.paused_elapsed = 0.0
if "was_running" not in st.session_state:
    st.session_state.was_running = running
if st.session_state.was_running and not running:
    st.session_state.paused_elapsed = (time.monotonic() - st.session_state.sim_epoch) * speed
if not st.session_state.was_running and running:
    st.session_state.sim_epoch = time.monotonic() - st.session_state.paused_elapsed / speed
st.session_state.was_running = running
current_elapsed = (time.monotonic() - st.session_state.sim_epoch) * speed if running else st.session_state.paused_elapsed
current_elapsed = max(0.0, current_elapsed)
transition_signature = (scenario, controller, target)
if "transition_signature" not in st.session_state or st.session_state.transition_signature != transition_signature:
    st.session_state.transition_signature = transition_signature
    st.session_state.transition_start_elapsed = current_elapsed
if "transition_start_elapsed" not in st.session_state:
    st.session_state.transition_start_elapsed = 0.0

def beat_times(kind, end):
    out, t, i = [], 0.0, 0
    while t <= end + 2:
        if kind == "Normal rhythm":
            interval, keep = 60 / 72, True
        elif kind == "Bradycardia":
            interval, keep = 60 / 38, True
        elif kind == "Missed beats":
            interval, keep = 60 / 68, (i % 5 != 2)
        else:
            interval, keep = [0.52, 0.68, 1.02, 0.58, 1.24, 0.76][i % 6], True
        if keep:
            out.append(t)
        t += interval
        i += 1
    return np.asarray(out, dtype=float)

def ecg(t, beats):
    y = np.zeros_like(t, dtype=float)
    for b in beats:
        y += 0.12 * np.exp(-0.5 * ((t - (b - 0.18)) / 0.045) ** 2)
        y += -0.16 * np.exp(-0.5 * ((t - (b - 0.025)) / 0.012) ** 2)
        y += np.exp(-0.5 * ((t - b) / 0.014) ** 2)
        y += -0.22 * np.exp(-0.5 * ((t - (b + 0.028)) / 0.014) ** 2)
        y += 0.28 * np.exp(-0.5 * ((t - (b + 0.22)) / 0.075) ** 2)
    return y + 0.018 * np.sin(2 * np.pi * 0.33 * t)

@st.cache_data(show_spinner=False)
def model_response(g, end=T_FINAL, dt=DT):
    kp, ki, kd = g
    num = [1352 * kd, 1352 * kp, 1352 * ki]
    den = [1, 28.8, 166.4 + 1352 * kd, 1352 * kp, 1352 * ki]
    t = np.arange(0.0, end + dt / 2, dt)
    tout, y = step(TransferFunction(num, den), T=t)
    return tout, np.asarray(y, dtype=float)

def calc_metrics(t, y, g):
    kp, ki, kd = g
    error = 1.0 - y
    mse = float(np.trapezoid(error ** 2, t) / (t[-1] - t[0]))
    peak_index = int(np.argmax(y))
    peak = float(y[peak_index])
    peak_time = float(t[peak_index])
    overshoot = max(0.0, (peak - 1.0) * 100.0)
    i10 = np.flatnonzero(y >= 0.1)
    i90 = np.flatnonzero(y >= 0.9)
    rise = float(t[i90[0]] - t[i10[0]]) if len(i10) and len(i90) else float(t[-1])
    outside = np.flatnonzero(np.abs(y - 1.0) > 0.02)
    settling = 0.0 if not len(outside) else (float(t[outside[-1] + 1]) if outside[-1] + 1 < len(t) else float(t[-1]))
    poles = np.roots([1, 28.8, 166.4 + 1352 * kd, 1352 * kp, 1352 * ki])
    stable = bool(np.all(np.real(poles) < 0))
    cost = 0.50 * (mse / 0.02) + 0.30 * (overshoot / 20.0) + 0.20 * settling
    if settling >= 3.999:
        cost += 2.0
    if not stable:
        cost += 100.0
    return {
        "Kp": kp, "Ki": ki, "Kd": kd, "MSE": mse, "Rise time (s)": rise,
        "Settling time (s)": settling, "Overshoot (%)": overshoot,
        "Peak time (s)": peak_time, "Peak output": peak, "Stable": stable,
        "Objective cost": float(cost), "Closed-loop poles": poles,
    }

def draw_ecg(t, y, color, left, right, markers=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color=color, width=1.8), name="Synthetic ECG"))
    if markers is not None:
        for beat in markers:
            if left <= beat <= right:
                fig.add_vline(x=float(beat), line_width=1, line_dash="dot", line_color="#f6c96b", opacity=0.6)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#07131b", plot_bgcolor="#07131b", font=dict(color="#dcecf4"),
        xaxis=dict(range=[left, right], title="Simulation time (s)", gridcolor="#18303d"),
        yaxis=dict(title="Amplitude (a.u.)", range=[-0.25, 1.15], gridcolor="#18303d"), showlegend=False)
    return fig

def live_panel():
    elapsed = (time.monotonic() - st.session_state.sim_epoch) * speed if running else st.session_state.paused_elapsed
    elapsed = max(0.0, elapsed)
    response_elapsed = max(0.0, elapsed - st.session_state.transition_start_elapsed)
    window = 6.0
    left, right = max(0.0, elapsed - window), max(window, elapsed)
    tplot = np.arange(left, right, 0.004)
    all_input = beat_times(scenario, right + 2)
    if scenario == "Normal rhythm":
        input_bpm = 72.0
    elif scenario == "Bradycardia":
        input_bpm = 38.0
    elif scenario == "Missed beats":
        input_bpm = 54.4
    else:
        input_bpm = 60 / np.mean(np.diff(beat_times(scenario, 12)[:7]))
    abnormal = scenario != "Normal rhythm"
    brady = input_bpm < threshold or scenario == "Bradycardia"
    mt, my = model_response(GAINS[controller], 4.0)
    progress = float(np.clip(np.interp(min(response_elapsed, mt[-1]), mt, my), 0.0, 1.0))
    out_bpm = input_bpm + (target - input_bpm) * progress if abnormal else input_bpm
    out_bpm = float(np.clip(out_bpm, 30.0, 120.0))
    inbeats = all_input[(all_input >= left - 0.4) & (all_input <= right + 0.5)]
    interval = 60.0 / out_bpm
    outbeats = np.arange(0.0, right + interval, interval)
    outbeats = outbeats[(outbeats >= left - 0.4) & (outbeats <= right + 0.5)]
    yin, yout = ecg(tplot, inbeats), ecg(tplot, outbeats)
    a, b, c, d = st.columns(4)
    a.metric("Input rate", f"{input_bpm:.1f} BPM")
    b.metric("Input rhythm", "NORMAL" if not abnormal else scenario.upper())
    c.metric("Selected controller", controller, f"Kp={GAINS[controller][0]:.3f}")
    d.metric("Simulated output rate", f"{out_bpm:.1f} BPM" if abnormal else "No pacing requested")
    if brady:
        st.error(f"SIMULATED BRADYCARDIA — input {input_bpm:.1f} BPM; selected demo threshold {threshold} BPM.")
    elif abnormal:
        st.warning("SIMULATED RHYTHM ABNORMALITY — synthetic scenario, not clinical detection.")
    else:
        st.success("NORMAL INPUT SCENARIO — pacing not requested.")
    lc, rc = st.columns(2)
    with lc:
        st.subheader("1 · Input ECG — condition retained")
        st.plotly_chart(draw_ecg(tplot, yin, "#66e6a1", left, right), use_container_width=True, key="input_trace")
        st.caption("Synthetic input: slow, missed or irregular beats remain visible.")
    with rc:
        st.subheader("2 · Simulated output — model-driven")
        st.plotly_chart(draw_ecg(tplot, yout, "#71baff", left, right, outbeats if abnormal else None), use_container_width=True, key="output_trace")
        st.caption(f"Illustrative output rate: {out_bpm:.1f} BPM. This is a display mapping, not a pacing law.")
    st.caption(f"Simulation time: {elapsed:.1f} s · Controller transition: {response_elapsed:.2f} s · ECG panels refresh independently of the static analysis below.")

st.subheader("Live synthetic ECG comparison")
if hasattr(st, "fragment"):
    st.fragment(run_every=0.15)(live_panel)()
else:
    live_panel()

# Calculate all selected controller metrics from one shared, MATLAB-aligned evaluation setup.
results = {}
responses = {}
for name, gains in GAINS.items():
    tx, yy = model_response(gains, T_FINAL, DT)
    responses[name] = (tx, yy)
    results[name] = calc_metrics(tx, yy, gains)
selected_t, selected_y = responses[controller]
selected = results[controller]

st.divider()
st.header("3 · Selected controller: complete analysis")
st.caption(f"Selected model: {CONTROLLER_LABELS[controller]} · The selected controller updates all metrics, plots, and model details below.")
metric_cols = st.columns(4)
metric_cols[0].metric("MSE", f"{selected['MSE']:.6f}")
metric_cols[1].metric("Rise time", f"{selected['Rise time (s)']:.3f} s")
metric_cols[2].metric("Settling time", f"{selected['Settling time (s)']:.3f} s")
metric_cols[3].metric("Overshoot", f"{selected['Overshoot (%)']:.2f}%")
metric_cols = st.columns(4)
metric_cols[0].metric("Peak output", f"{selected['Peak output']:.6f}")
metric_cols[1].metric("Peak time", f"{selected['Peak time (s)']:.3f} s")
metric_cols[2].metric("Closed-loop stability", "STABLE" if selected["Stable"] else "UNSTABLE")
metric_cols[3].metric("Objective cost", f"{selected['Objective cost']:.6f}")

st.subheader("Selected PID gains")
g1, g2, g3 = st.columns(3)
g1.metric("Proportional gain (Kp)", f"{selected['Kp']:.9f}")
g2.metric("Integral gain (Ki)", f"{selected['Ki']:.9f}")
g3.metric("Derivative gain (Kd)", f"{selected['Kd']:.9f}")

st.subheader("Selected controller step response")
fig = go.Figure()
fig.add_trace(go.Scatter(x=selected_t, y=selected_y, mode="lines", line=dict(color="#c49bff", width=2.4), name=controller))
fig.add_hline(y=1, line_dash="dash", line_color="#94a3b8", annotation_text="Reference = 1")
fig.add_hrect(y0=0.98, y1=1.02, fillcolor="#66e6a1", opacity=0.08, line_width=0, annotation_text="±2% settling band")
fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="#07131b",
    plot_bgcolor="#07131b", font=dict(color="#dcecf4"),
    xaxis=dict(title="Time (s)", gridcolor="#18303d"), yaxis=dict(title="Normalized output", gridcolor="#18303d"))
st.plotly_chart(fig, use_container_width=True)

err_fig = go.Figure()
err_fig.add_trace(go.Scatter(x=selected_t, y=1-selected_y, mode="lines", line=dict(color="#f6c96b", width=1.8), name="Tracking error"))
err_fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
err_fig.update_layout(height=260, margin=dict(l=10, r=10, t=15, b=10), paper_bgcolor="#07131b",
    plot_bgcolor="#07131b", font=dict(color="#dcecf4"),
    xaxis=dict(title="Time (s)", gridcolor="#18303d"), yaxis=dict(title="Error = reference − output", gridcolor="#18303d"))
st.subheader("Selected controller tracking error")
st.plotly_chart(err_fig, use_container_width=True)

with st.expander("Detailed metric definitions and selected closed-loop poles"):
    st.markdown("""
- **MSE:** time-averaged squared tracking error, integrated using the trapezoidal rule over the 0–4 s simulation.
- **Rise time:** first crossing of 90% of the unit reference minus first crossing of 10%.
- **Settling time:** last sample outside the ±2% band, followed by the next sample; if it never settles in the window, the displayed value is the end of the simulation.
- **Overshoot:** maximum response above the unit reference, expressed as a percentage.
- **Peak output / peak time:** maximum normalized output and the time it occurs.
- **Stability:** all roots of the closed-loop characteristic polynomial have negative real parts.
- **Objective cost:** `0.50 × (MSE / 0.02) + 0.30 × (overshoot (%) / 20) + 0.20 × settling time (s)`, plus 2 if settling time is at least 3.999 s and plus 100 if unstable. These weights are project design choices, not clinical thresholds.
""")
    st.write("Closed-loop poles (real + imaginary parts):")
    poles_df = pd.DataFrame({
        "Pole index": np.arange(1, len(selected["Closed-loop poles"])+1),
        "Real part": np.real(selected["Closed-loop poles"]),
        "Imaginary part": np.imag(selected["Closed-loop poles"]),
    })
    st.dataframe(poles_df.style.format({"Real part":"{:.6f}", "Imaginary part":"{:.6f}"}), use_container_width=True, hide_index=True)

st.divider()
st.header("4 · ABC vs DE vs GAS: complete comparison")
st.caption("All three gain sets are evaluated with the same plant, unit-step reference, 4.000 s duration, and 0.001 s time step. Changing the selected controller above changes the detailed selected-controller section.")
rows = []
for name, gains in GAINS.items():
    r = results[name]
    rows.append({
        "Controller": name, "Method": CONTROLLER_LABELS[name],
        "Kp": r["Kp"], "Ki": r["Ki"], "Kd": r["Kd"],
        "MSE": r["MSE"], "Rise time (s)": r["Rise time (s)"],
        "Settling time (s)": r["Settling time (s)"], "Overshoot (%)": r["Overshoot (%)"],
        "Peak time (s)": r["Peak time (s)"], "Peak output": r["Peak output"],
        "Stable": r["Stable"], "Objective cost": r["Objective cost"],
    })
comparison = pd.DataFrame(rows)
st.dataframe(comparison.style.format({
    "Kp":"{:.9f}", "Ki":"{:.9f}", "Kd":"{:.9f}", "MSE":"{:.6f}",
    "Rise time (s)":"{:.3f}", "Settling time (s)":"{:.3f}",
    "Overshoot (%)":"{:.2f}", "Peak time (s)":"{:.3f}",
    "Peak output":"{:.6f}", "Objective cost":"{:.6f}",
}), use_container_width=True, hide_index=True)

st.subheader("All-controller step-response overlay")
overlay = go.Figure()
overlay.add_trace(go.Scatter(x=selected_t, y=np.ones_like(selected_t), mode="lines",
    line=dict(color="#94a3b8", dash="dash"), name="Unit reference"))
for name, (tx, yy) in responses.items():
    overlay.add_trace(go.Scatter(x=tx, y=yy, mode="lines", name=name, line=dict(width=2.0)))
overlay.update_layout(height=400, margin=dict(l=10, r=10, t=15, b=10), paper_bgcolor="#07131b",
    plot_bgcolor="#07131b", font=dict(color="#dcecf4"),
    xaxis=dict(title="Time (s)", gridcolor="#18303d"), yaxis=dict(title="Normalized output", gridcolor="#18303d"))
st.plotly_chart(overlay, use_container_width=True)

def comparison_bar(column, title, y_title, lower_better=True):
    ordered = comparison.sort_values(column, ascending=lower_better)
    fig = go.Figure(go.Bar(x=ordered["Controller"], y=ordered[column], text=ordered[column], textposition="outside"))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="#07131b", plot_bgcolor="#07131b", font=dict(color="#dcecf4"),
        xaxis=dict(title="Controller", gridcolor="#18303d"), yaxis=dict(title=y_title, gridcolor="#18303d"))
    st.plotly_chart(fig, use_container_width=True)

chart1, chart2 = st.columns(2)
with chart1:
    comparison_bar("Objective cost", "Objective cost (lower is better)", "Cost")
with chart2:
    comparison_bar("MSE", "Mean squared error (lower is better)", "MSE")
chart3, chart4 = st.columns(2)
with chart3:
    comparison_bar("Overshoot (%)", "Overshoot (lower is better)", "Overshoot (%)")
with chart4:
    comparison_bar("Settling time (s)", "Settling time (lower is better)", "Seconds")
chart5, chart6 = st.columns(2)
with chart5:
    comparison_bar("Rise time (s)", "Rise time (lower is faster)", "Seconds")
with chart6:
    gain_fig = go.Figure()
    for col in ["Kp", "Ki", "Kd"]:
        gain_fig.add_trace(go.Bar(x=comparison["Controller"], y=comparison[col], name=col))
    gain_fig.update_layout(barmode="group", title="PID gain comparison", height=320,
        margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor="#07131b", plot_bgcolor="#07131b",
        font=dict(color="#dcecf4"), xaxis=dict(title="Controller", gridcolor="#18303d"),
        yaxis=dict(title="Gain value", gridcolor="#18303d"))
    st.plotly_chart(gain_fig, use_container_width=True)

st.subheader("Best controller by each metric")
best_rows = [
    ("Lowest MSE", comparison.loc[comparison["MSE"].idxmin(), "Controller"], comparison["MSE"].min()),
    ("Fastest rise time", comparison.loc[comparison["Rise time (s)"].idxmin(), "Controller"], comparison["Rise time (s)"].min()),
    ("Fastest settling", comparison.loc[comparison["Settling time (s)"].idxmin(), "Controller"], comparison["Settling time (s)"].min()),
    ("Lowest overshoot", comparison.loc[comparison["Overshoot (%)"].idxmin(), "Controller"], comparison["Overshoot (%)"].min()),
    ("Lowest combined cost", comparison.loc[comparison["Objective cost"].idxmin(), "Controller"], comparison["Objective cost"].min()),
]
st.dataframe(pd.DataFrame(best_rows, columns=["Metric", "Best controller", "Value"]), use_container_width=True, hide_index=True)

st.subheader("5 · Mathematical model, setup, and limitations")
with st.expander("Transfer functions, equations, simulation settings, and methodology", expanded=True):
    st.markdown("**Heart model**")
    st.latex(r"G_h(s)=\frac{169}{s(s+20.8)}")

    st.markdown("**Pacemaker model**")
    st.latex(r"G_p(s)=\frac{8}{s+8}")

    st.markdown("**Combined pacemaker–heart plant**")
    st.latex(r"G_{cp}(s)=G_p(s)G_h(s)=\frac{1352}{s(s+20.8)(s+8)}")
    st.latex(r"G_{cp}(s)=\frac{1352}{s^3+28.8s^2+166.4s}")

    st.markdown("**PID controller (unity feedback)**")
    st.latex(r"C(s)=K_p+\frac{K_i}{s}+K_ds")

    st.markdown("**Closed-loop transfer function**")
    st.latex(r"T(s)=\frac{C(s)G_{cp}(s)}{1+C(s)G_{cp}(s)}")
    st.latex(r"T(s)=\frac{1352(K_ds^2+K_ps+K_i)}{s^4+28.8s^3+(166.4+1352K_d)s^2+1352K_ps+1352K_i}")

    st.markdown("**Simulation configuration**")
    st.write("Unit-step reference = 1 · Duration = 4.000 s · Time step = 0.001 s · Settling band = ±2% · Rise-time thresholds = 10% and 90%.")

    st.markdown("**Interpretation**")
    st.write("ABC is the existing baseline. DE is a classical differential-evolution optimizer. GAS denotes Grover Adaptive Search demonstrated through a Qiskit statevector simulation; the candidate-cost table was calculated classically, so this does not demonstrate quantum speedup. The dashboard reuses the selected gain sets; it does not rerun the optimizers live.")

    st.markdown("**Limitations**")
    st.write("The transfer function is a simplified control-system model, not a validated physiological model. ECG-like waveforms are synthetic. The BPM transition is a display-only illustrative mapping, not a physiological pacing equation. No patient ECG is read, no diagnosis is performed, and no real pacemaker is controlled. This is educational software only.")

csv = comparison.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download full controller metrics (CSV)", data=csv,
    file_name="CARDIOGUARD_full_controller_metrics.csv", mime="text/csv", use_container_width=False)
st.caption("CARDIOGUARD · deterministic educational simulation · NOT FOR CLINICAL USE")
