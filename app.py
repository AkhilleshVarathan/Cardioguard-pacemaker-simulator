import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import TransferFunction, step

st.set_page_config(page_title="CARDIOGUARD | Pacemaker Simulation", page_icon="💓", layout="wide")
GAINS={"ABC":(2.460736,.120961,.600000),"DE":(1.501187,.337174,.378484),"GAS":(1.500000,.350000,.407143)}
st.markdown("""<style>
.stApp{background:#07131b;color:#e8f4fb}
[data-testid="stMetric"]{background:#10232d;border:1px solid #244451;padding:12px;border-radius:12px}
[data-testid="stMetricLabel"]{color:#9ab6c4}
[data-testid="stMetricValue"]{color:#e8f4fb}
.block-container{padding-top:1.1rem}
</style>""",unsafe_allow_html=True)
st.title("💓 CARDIOGUARD")
st.caption("MODEL-DRIVEN CARDIAC PACEMAKER DIGITAL SIMULATION")
st.warning("SIMULATION ONLY — not a medical device, not for diagnosis or treatment, and never connect to a person. All ECG-like signals are synthetic. The transfer function is a simplified control-system model, not a validated physiological heart model.")

with st.sidebar:
    st.header("Simulation controls")
    scenario=st.selectbox("Input rhythm",["Normal rhythm","Bradycardia","Missed beats","Irregular rhythm"],index=1)
    controller=st.selectbox("Controller gain set",["ABC","DE","GAS"],index=1)
    threshold=st.slider("Demo bradycardia threshold (BPM)",35,70,50)
    target=st.slider("Demo output target (BPM)",50,90,60)
    speed=st.slider("Simulation speed",.5,2.,1.,.25)
    running=st.toggle("▶ Run live simulation",value=True)
    reset_clicked=st.button("↻ Reset simulation",use_container_width=True)
    if reset_clicked:
        st.session_state.sim_epoch=time.monotonic()
        st.session_state.paused_elapsed=0.0
        st.session_state.transition_start_elapsed=0.0
    st.caption("These are demonstration settings, not clinical recommendations.")

if "sim_epoch" not in st.session_state: st.session_state.sim_epoch=time.monotonic()
if "paused_elapsed" not in st.session_state: st.session_state.paused_elapsed=0.
if "was_running" not in st.session_state: st.session_state.was_running=running
if st.session_state.was_running and not running:
    st.session_state.paused_elapsed=(time.monotonic()-st.session_state.sim_epoch)*speed
if not st.session_state.was_running and running:
    st.session_state.sim_epoch=time.monotonic()-st.session_state.paused_elapsed/speed
st.session_state.was_running=running
current_elapsed=(time.monotonic()-st.session_state.sim_epoch)*speed if running else st.session_state.paused_elapsed
current_elapsed=max(0.0,current_elapsed)
# Restart the model-driven transition whenever the scenario, controller, or target changes.
transition_signature=(scenario,controller,target)
if "transition_signature" not in st.session_state or st.session_state.transition_signature != transition_signature:
    st.session_state.transition_signature=transition_signature
    st.session_state.transition_start_elapsed=current_elapsed
if "transition_start_elapsed" not in st.session_state:
    st.session_state.transition_start_elapsed=0.0

def beat_times(kind,end):
    out=[]; t=0.; i=0
    while t<=end+2:
        if kind=="Normal rhythm": interval,keep=60/72,True
        elif kind=="Bradycardia": interval,keep=60/38,True
        elif kind=="Missed beats": interval,keep=60/68,(i%5!=2)
        else:
            interval,keep=[.52,.68,1.02,.58,1.24,.76][i%6],True
        if keep: out.append(t)
        t+=interval; i+=1
    return np.asarray(out,dtype=float)

def ecg(t,beats):
    y=np.zeros_like(t,dtype=float)
    for b in beats:
        y+=.12*np.exp(-.5*((t-(b-.18))/.045)**2)
        y+=-.16*np.exp(-.5*((t-(b-.025))/.012)**2)
        y+=np.exp(-.5*((t-b)/.014)**2)
        y+=-.22*np.exp(-.5*((t-(b+.028))/.014)**2)
        y+=.28*np.exp(-.5*((t-(b+.22))/.075)**2)
    return y+.018*np.sin(2*np.pi*.33*t)

@st.cache_data(show_spinner=False)
def model_response(g,end,dt=.001):
    kp,ki,kd=g
    # Existing plant: 1352/[s^3 + 28.8s^2 + 166.4s]
    # Unity-feedback PID closed-loop transfer function.
    num=[1352*kd,1352*kp,1352*ki]
    den=[1,28.8,166.4+1352*kd,1352*kp,1352*ki]
    t=np.arange(0,end+dt,dt)
    tout,y=step(TransferFunction(num,den),T=t)
    return tout,np.asarray(y,dtype=float)

def calc_metrics(t,y):
    mse=float(np.mean((1-y)**2))
    over=float(max(0,(np.max(y)-1)*100))
    idx=np.where(np.abs(y-1)>.02)[0]
    settle=0. if not len(idx) else (float(t[idx[-1]+1]) if idx[-1]+1<len(t) else float(t[-1]))
    return mse,over,settle

def draw_ecg(t,y,color,left,right,markers=None):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=t,y=y,mode="lines",line=dict(color=color,width=1.8),name="Synthetic ECG"))
    if markers is not None:
        for b in markers:
            if left<=b<=right: fig.add_vline(x=float(b),line_width=1,line_dash="dot",line_color="#f6c96b",opacity=.6)
    fig.update_layout(height=300,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor="#07131b",plot_bgcolor="#07131b",
        font=dict(color="#dcecf4"),xaxis=dict(range=[left,right],title="Simulation time (s)",gridcolor="#18303d"),
        yaxis=dict(title="Amplitude (a.u.)",range=[-.25,1.15],gridcolor="#18303d"),showlegend=False)
    return fig

def live_panel():
    elapsed=(time.monotonic()-st.session_state.sim_epoch)*speed if running else st.session_state.paused_elapsed
    elapsed=max(0.,elapsed)
    response_elapsed=max(0.0,elapsed-st.session_state.transition_start_elapsed)
    window=6.
    left=max(0.,elapsed-window); right=max(window,elapsed)
    tplot=np.arange(left,right,.004)
    all_input=beat_times(scenario,right+2)
    if scenario=="Normal rhythm": input_bpm=72.
    elif scenario=="Bradycardia": input_bpm=38.
    elif scenario=="Missed beats": input_bpm=54.4
    else: input_bpm=60/np.mean(np.diff(beat_times(scenario,12)[:7]))
    abnormal=scenario!="Normal rhythm"
    brady=input_bpm<threshold or scenario=="Bradycardia"

    # Selected controller's response from the original transfer-function model.
    # Its normalized step response is mapped to a bounded illustrative rate transition.
    mt,my=model_response(GAINS[controller],4.0)
    progress=float(np.clip(np.interp(min(response_elapsed,mt[-1]),mt,my),0.,1.))
    # Every non-normal scenario demonstrates recovery; no hidden "slow-only" gate.
    if abnormal:
        out_bpm=input_bpm+(target-input_bpm)*progress
    else:
        out_bpm=input_bpm
    out_bpm=float(np.clip(out_bpm,30.,120.))

    inbeats=all_input[(all_input>=left-.4)&(all_input<=right+.5)]
    interval=60./out_bpm
    outbeats=np.arange(0.,right+interval,interval)
    outbeats=outbeats[(outbeats>=left-.4)&(outbeats<=right+.5)]
    yin=ecg(tplot,inbeats); yout=ecg(tplot,outbeats)

    a,b,c,d=st.columns(4)
    a.metric("Input rate",f"{input_bpm:.1f} BPM")
    b.metric("Input rhythm","NORMAL" if not abnormal else scenario.upper())
    c.metric("Selected controller",controller,f"Kp={GAINS[controller][0]:.3f}")
    d.metric("Simulated output rate",f"{out_bpm:.1f} BPM" if abnormal else "No pacing requested")

    if brady: st.error(f"SIMULATED BRADYCARDIA — input {input_bpm:.1f} BPM; selected demo threshold {threshold} BPM.")
    elif abnormal: st.warning("SIMULATED RHYTHM ABNORMALITY — synthetic scenario, not clinical detection.")
    else: st.success("NORMAL INPUT SCENARIO — pacing not requested.")

    lc,rc=st.columns(2)
    with lc:
        st.subheader("1 · Input ECG — condition retained")
        st.plotly_chart(draw_ecg(tplot,yin,"#66e6a1",left,right),use_container_width=True,key="input_trace")
        st.caption("Green shows the selected input condition: slow, missed or irregular beats remain visible.")
    with rc:
        st.subheader("2 · Simulated output — model-driven")
        st.plotly_chart(draw_ecg(tplot,yout,"#71baff",left,right,outbeats if abnormal else None),use_container_width=True,key="output_trace")
        st.caption(f"Blue output uses {controller}'s model response to transition toward the demo target. Output rate: {out_bpm:.1f} BPM.")
    st.caption(f"Live simulation time: {elapsed:.1f} s · Controller transition: {response_elapsed:.2f} s · Only the ECG panel refreshes; the rest of the page remains stationary.")

st.subheader("Live ECG comparison")
if hasattr(st,"fragment"):
    st.fragment(run_every=0.15)(live_panel)()
else:
    st.error("Update Streamlit for smooth partial refresh: `py -m pip install --upgrade streamlit`")
    live_panel()

st.subheader("3 · Original mathematical control-model response")
t,y=model_response(GAINS[controller],4.)
fig=go.Figure(go.Scatter(x=t,y=y,mode="lines",line=dict(color="#c49bff",width=2.4),name=controller))
fig.add_hline(y=1,line_dash="dash",line_color="#94a3b8",annotation_text="Reference = 1")
fig.update_layout(height=270,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor="#07131b",plot_bgcolor="#07131b",
    font=dict(color="#dcecf4"),xaxis=dict(title="Time (s)",gridcolor="#18303d"),yaxis=dict(title="Normalized output",gridcolor="#18303d"))
st.plotly_chart(fig,use_container_width=True)
mse,over,settle=calc_metrics(t,y)
a,b,c=st.columns(3); a.metric("MSE (unit-step)",f"{mse:.6f}"); b.metric("Overshoot",f"{over:.2f}%"); c.metric("Settling time",f"{settle:.3f} s")

st.subheader("4 · ABC vs DE vs GAS")
rows=[]
for name,g in GAINS.items():
    tx,yy=model_response(g,4.); mm,oo,ss=calc_metrics(tx,yy)
    rows.append({"Controller":name,"Kp":g[0],"Ki":g[1],"Kd":g[2],"MSE":mm,"Overshoot (%)":oo,"Settling time (s)":ss})
st.dataframe(pd.DataFrame(rows).style.format({"Kp":"{:.6f}","Ki":"{:.6f}","Kd":"{:.6f}","MSE":"{:.6f}","Overshoot (%)":"{:.2f}","Settling time (s)":"{:.3f}"}),use_container_width=True,hide_index=True)

with st.expander("Mathematical model and limitations"):
    st.markdown("""**Existing plant:** `Gcp(s)=1352/[s³+28.8s²+166.4s]`  
**Unity-feedback PID closed loop:** `T(s)=1352(Kd s²+Kp s+Ki)/[s⁴+28.8s³+(166.4+1352Kd)s²+1352Kp s+1352Ki]`

The selected controller's unit-step response is clipped to 0–1 and used as a documented interface mapping from input BPM toward a display-only target. This is an illustrative rule, not a physiological equation. ECG morphology and beat timing are synthetic. No diagnosis, real pacing, or clinical treatment is implemented. Optimizers are not rerun live; existing ABC/DE/GAS gains are reused.""")
st.caption("CARDIOGUARD · deterministic live simulation · educational use only · NOT FOR CLINICAL USE")
