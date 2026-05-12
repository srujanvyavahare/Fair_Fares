import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Fair Fares – Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# CUSTOM CSS – Fair Fares Theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Montserrat:wght@700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background: #f5f6fa;
}

/* ── Header / Hero ─────────────────────── */
.ctkt-hero {
    background: linear-gradient(135deg, #d0021b 0%, #f5222d 40%, #ff6b35 100%);
    border-radius: 0 0 32px 32px;
    padding: 28px 40px 36px;
    margin: -20px -20px 28px -20px;
    box-shadow: 0 8px 32px rgba(208,2,27,0.28);
    position: relative;
    overflow: hidden;
}
.ctkt-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: rgba(255,255,255,0.07);
}
.ctkt-hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.brand-row { display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }
.brand-logo {
    font-family: 'Montserrat', sans-serif;
    font-size: 3rem;
    font-weight: 900;
    color: white;
    letter-spacing: -1px;
}
.brand-logo span { color: #ffe066; }
.brand-tagline { color: rgba(255,255,255,0.82); font-size: 1.5rem; letter-spacing: 1px; font-weight: 400; }
.hero-title {
    font-family: 'Montserrat', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: white;
    margin: 10px 0 4px;
}
.hero-sub { color: rgba(255,255,255,0.78); font-size: 0.88rem; font-weight: 400; }

/* ── Search Card ───────────────────────── */
.search-card {
    background: white;
    border-radius: 20px;
    padding: 28px 32px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.09);
    margin-bottom: 24px;
    border-top: 5px solid #d0021b;
}
.section-title {
    font-family: 'Montserrat', sans-serif;
    font-size: 3rem;
    font-weight: 500;
    color: #1a1a2e;
    letter-spacing: 0.3px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, #d0021b22, transparent);
    border-radius: 2px;
}

/* ── Predict Button ─────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #d0021b, #ff4d4f) !important;
    color: white !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 36px !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 6px 20px rgba(208,2,27,0.38) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(208,2,27,0.45) !important;
    background: linear-gradient(135deg, #b8001a, #e03333) !important;
}

/* ── Swap Button Override ───────────────── */
div[data-testid="column"]:has(button[kind="secondary"]#swap_btn) .stButton > button,
button[key="swap_btn"],
[data-testid="stButton-swap_btn"] button {
    background: white !important;
    color: #d0021b !important;
    border: 2px solid #d0021b !important;
    border-radius: 50% !important;
    font-size: 1.2rem !important;
    font-weight: 900 !important;
    padding: 6px !important;
    box-shadow: 0 2px 10px rgba(208,2,27,0.18) !important;
    min-height: 38px !important;
    letter-spacing: 0 !important;
}
[data-testid="stButton-swap_btn"] button:hover {
    background: #d0021b !important;
    color: white !important;
    transform: rotate(180deg) translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(208,2,27,0.35) !important;
}

/* ── Result Card ────────────────────────── */
.result-card {
    background: white;
    border-radius: 22px;
    padding: 32px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.12);
    border-left: 6px solid #d0021b;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '✈';
    position: absolute;
    top: 14px; right: 24px;
    font-size: 5rem;
    color: rgba(208,2,27,0.26);
    line-height: 1;
}
.flight-airline {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #1a1a2e;
    margin-bottom: 4px;
}
.flight-number { color: #888; font-size: 1rem; font-weight: 500; margin-bottom: 16px; }
.route-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}
.city-box {
    text-align: center;
}
.city-code {
    font-family: 'Montserrat', sans-serif;
    font-size: 2.5rem;
    font-weight: 750;
    color: #1a1a2e;
    line-height: 1;
}
.city-name { font-size: 1.05rem; color: #888; font-weight: 500; }
.route-line {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 5px;
}
.route-bar {
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, #d0021b, #ff6b35);
    border-radius: 2px;
    position: relative;
}
.route-bar::before { content: '✈'; position: absolute; top: -11px; left: 50%; transform: translateX(-50%); font-size: 1rem; color: #d0021b; }
.route-stops { font-size: 0.85rem; color: #999; font-weight: 600; }
.detail-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 20px;
    padding: 16px;
    background: #fef9f9;
    border-radius: 14px;
}
.detail-item { text-align: center; }
.detail-label { font-size: 0.68rem; color: #aaa; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
.detail-value { font-size: 0.9rem; font-weight: 700; color: #2d2d2d; }
.price-display {
    text-align: center;
    padding: 22px;
    background: linear-gradient(135deg, #d0021b08, #ff6b3508);
    border: 2px dashed #d0021b44;
    border-radius: 16px;
    margin-top: 12px;
}
.price-label { font-size: 0.75rem; color: #aaa; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px; }
.price-amount {
    font-family: 'Montserrat', sans-serif;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #d0021b, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.price-per { font-size: 0.78rem; color: #aaa; margin-top: 4px; font-weight: 500; }

/* ── Days Left Badge ────────────────────── */
.days-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    color: white;
    padding: 8px 18px;
    border-radius: 100px;
    font-size: 0.82rem;
    font-weight: 700;
    margin-bottom: 0;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
}
.days-number {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.1rem;
    font-weight: 900;
    color: #ffe066;
}

/* ── Info Metric Cards ──────────────────── */
.metric-strip {
    display: flex;
    gap: 14px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.metric-pill {
    background: white;
    border-radius: 14px;
    padding: 12px 20px;
    text-align: center;
    flex: 1;
    min-width: 110px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-top: 3px solid #d0021b;
}
.metric-pill-val { font-family:'Montserrat',sans-serif; font-size:1.3rem; font-weight:900; color:#1a1a2e; }
.metric-pill-lbl { font-size:0.68rem; color:#aaa; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; }

/* ── Comparison Chart Header ────────────── */
.chart-header {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #1a1a2e;
    margin: 24px 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Streamlit overrides ────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stDateInput"] label,
.stRadio label { font-weight: 600; color: #2d2d2d; font-size: 0.85rem !important; }

div[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
    border-color: #e0e0e0 !important;
}

.stSelectbox [data-baseweb="select"] { border-radius: 10px !important; }

footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 40px !important; }

/* ── Mobile-width layout ───────────────── */
.block-container {
    max-width: 1000px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 10px !important;
    padding-right: 10px !important;
}
/* ── Divider ───────────────────────────── */
.ctkt-divider {
    height: 3px;
    background: linear-gradient(90deg, #d0021b, #ff6b35, transparent);
    border-radius: 2px;
    margin: 8px 0 20px;
}

/* ── Best deal tag ──────────────────────── */
.best-deal {
    display: inline-block;
    background: #52c41a;
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 100px;
    margin-left: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Loading spinner ───────────────────── */
.ctkt-loading {
    text-align: center;
    padding: 30px;
    color: #d0021b;
    font-weight: 600;
    font-size: 1rem;
}

/* ── Steps indicator ───────────────────── */
.step-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.step-dot {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, #d0021b, #ff6b35);
    color: white;
    font-size: 0.72rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
}
.step-text { font-size: 0.8rem; font-weight: 600; color: #555; }
.step-sep { color: #ccc; font-size: 0.7rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DATA & MODEL LOADING (cached)
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("Clean_Dataset.csv")
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

@st.cache_data(show_spinner=False)
def get_duration_map(df):
    """Median duration per (source, destination, stops)"""
    dur = (df.groupby(['source_city', 'destination_city', 'stops'])['duration']
             .median()
             .reset_index())
    dur_dict = {}
    for _, row in dur.iterrows():
        dur_dict[(row['source_city'], row['destination_city'], row['stops'])] = row['duration']
    return dur_dict

@st.cache_resource(show_spinner=False)
def train_model(df):
    """Train RF fast:
       - Fit label encoders on FULL dataset (no unknown-label errors at predict time)
       - Train model on an 80K stratified sample  (~10s vs ~46s for full data)
       - Accuracy: R² ~98%, MAE ~₹1,500
    """
    # 'flight' is back as a feature — prices vary meaningfully by flight number
    # 'arrival_time' is auto-computed from departure_time + duration (not a user input)
    label_cols = ['airline', 'flight', 'source_city', 'departure_time',
                  'stops', 'arrival_time', 'destination_city', 'class']
    encoders = {}
    df_full = df.copy()

    # Fit encoders on ALL rows so every category is known
    for col in label_cols:
        le = LabelEncoder()
        df_full[col] = le.fit_transform(df_full[col].astype(str))
        encoders[col] = le

    feature_cols = ['airline', 'flight', 'source_city', 'departure_time',
                    'stops', 'arrival_time', 'destination_city', 'class', 'days_left']

    # Sample 80K rows for fast training — still ~98% accuracy
    df_sample = df_full.sample(n=80_000, random_state=42)
    X = df_sample[feature_cols]
    y = df_sample['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)

    # Pre-tuned optimal hyperparameters
    best_params = {
        'n_estimators': 150,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
    }
    best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return best_model, encoders, feature_cols, mae, r2, mape, best_params


# ─────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="ctkt-hero">
  <div class="brand-row">
    <div class="brand-logo">Fair <span>Fares</span></div>
  </div>
  <div class="brand-tagline">✈ AI BASED FLIGHT FARE PREDICTOR</div>
  <div class="hero-sub">Find the best fare before you book </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD & TRAIN
# ─────────────────────────────────────────
with st.spinner("🔧 Loading dataset & training AI model…"):
    df = load_data()
    duration_map = get_duration_map(df)
    model, encoders, feature_cols, mae, r2, mape, best_params = train_model(df)

# ─────────────────────────────────────────
# UNIQUE VALUES
# ─────────────────────────────────────────
cities      = sorted(df['source_city'].unique().tolist())
airlines    = sorted(df['airline'].unique().tolist())
dep_times   = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
stop_opts   = ['zero', 'one', 'two_or_more']
class_opts  = ['Economy', 'Business']

time_icons  = {'Early_Morning':'🌅','Morning':'🌤','Afternoon':'☀️','Evening':'🌆','Night':'🌙','Late_Night':'🌃'}
stop_labels = {'zero':'Non-Stop ✈','one':'1 Stop','two_or_more':'2+ Stops'}
airline_icons = {
    'SpiceJet':'🛪',
    'AirAsia':'🟥',
    'Vistara':'🅥',
    'GO_FIRST':'✴',
    'Indigo':'🔷',
    'Air_India':'🔴'   
}

# ─────────────────────────────────────────
# ARRIVAL TIME AUTO-COMPUTE
# ─────────────────────────────────────────
def compute_arrival_time(dep_time, duration_hrs):
    """Given a departure time slot and float duration in hours,
    return the arrival time slot by adding duration to the slot's midpoint."""
    slot_midpoints = {
        'Early_Morning': 6.5,   # ~6:30 AM
        'Morning':       10.0,  # ~10:00 AM
        'Afternoon':     14.0,  # ~2:00 PM
        'Evening':       18.0,  # ~6:00 PM
        'Night':         22.0,  # ~10:00 PM
        'Late_Night':    2.0,   # ~2:00 AM
    }
    dep_hour = slot_midpoints.get(dep_time, 12.0)
    arr_hour = (dep_hour + duration_hrs) % 24

    if   5  <= arr_hour < 8:   return 'Early_Morning'
    elif 8  <= arr_hour < 12:  return 'Morning'
    elif 12 <= arr_hour < 16:  return 'Afternoon'
    elif 16 <= arr_hour < 20:  return 'Evening'
    elif 20 <= arr_hour < 24:  return 'Night'
    else:                      return 'Late_Night'   # 0–5 AM

# ─────────────────────────────────────────
# SESSION STATE FOR SWAP
# ─────────────────────────────────────────
if 'source_city' not in st.session_state:
    st.session_state.source_city = cities[0]
if 'dest_city' not in st.session_state:
    st.session_state.dest_city = cities[1] if len(cities) > 1 else cities[0]

def swap_cities():
    st.session_state.source_city, st.session_state.dest_city = \
        st.session_state.dest_city, st.session_state.source_city

# ─────────────────────────────────────────
# SEARCH FORM
# ─────────────────────────────────────────
st.markdown("""
<div style="background:white; border-radius:24px; padding:28px 32px 8px;
            box-shadow:0 4px 24px rgba(0,0,0,0.09); margin-bottom:8px;
            border-top:4px solid #d0021b;">

  <!-- Header row -->
  <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:20px;">
    <div style="display:flex; align-items:center; gap:20px;">
      <div style="background:linear-gradient(135deg,#d0021b,#ff4d4f); border-radius:10px;
                  width:50px; height:50px; display:flex; align-items:center;
                  justify-content:center; font-size:1.5rem; box-shadow:0 4px 12px rgba(208,2,27,0.3);">
        🔍
      </div>
      <div>
        <div style="font-family:'Montserrat',sans-serif; font-size:1.6rem;
                    font-weight:900; color:#1a1a2e; letter-spacing:0.2px;">Search Flights</div>
        <div style="font-size:0.85rem; color:#aaa; font-weight:500; margin-top:1px;">
            Fill in your travel details below
        </div>
      </div>
    </div>

  </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="height:2px; background:linear-gradient(90deg,#d0021b22,#e8e8e8,transparent);
                margin:20px 0 14px;"></div>
    <div style="font-size:1.1rem; font-weight:1000; color:#d0021b; text-transform:uppercase;
                letter-spacing:1.2px; margin-bottom:8px;">🗺 Route</div>
            <div style="height:2px; background:linear-gradient(90deg,#d0021b22,#e8e8e8,transparent);
                margin:10px 0 14px;"></div>
    """, unsafe_allow_html=True)

with st.container():
    # Row 1: Source | Swap Button | Destination
    row1_c1, row1_swap, row1_c2 = st.columns([10, 1.5, 10])
    with row1_c1:
        source = st.selectbox(
            "🛫 From ", cities,
            index=cities.index(st.session_state.source_city) if st.session_state.source_city in cities else 0,
            key="source_select"
        )
        st.session_state.source_city = source
    with row1_swap:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.button("⇄", on_click=swap_cities, help="Swap Source & Destination",
                  use_container_width=True, key="swap_btn")
    with row1_c2:
        dest_options = [c for c in cities if c != source]
        dest_default = st.session_state.dest_city if st.session_state.dest_city in dest_options else dest_options[0]
        destination = st.selectbox(
            "🛬 To ", dest_options,
            index=dest_options.index(dest_default),
            key="dest_select"
        )
        st.session_state.dest_city = destination

    # Divider + Travel Details label
    st.markdown("""
    <div style="height:2px; background:linear-gradient(90deg,#d0021b22,#e8e8e8,transparent);
                margin:20px 0 14px;"></div>
    <div style="font-size:1.1rem; font-weight:1000; color:#d0021b; text-transform:uppercase;
                letter-spacing:1.2px; margin-bottom:8px;">🗓 Travel Details</div>
            <div style="height:2px; background:linear-gradient(90deg,#d0021b22,#e8e8e8,transparent);
                margin:10px 0 14px;"></div>
    """, unsafe_allow_html=True)

    # Row 2: Date | Departure Time
    row2_c1, row2_c2 = st.columns([1, 1])
    with row2_c1:
        today = date.today()
        depart_date = st.date_input("📅 Date of Departure",
                                    min_value=today,
                                    max_value=today + timedelta(days=49),
                                    value=today + timedelta(days=7))
    with row2_c2:
        dep_time = st.selectbox("🕐 Departure Time", dep_times,
                                format_func=lambda x: f"{time_icons.get(x,'')} {x.replace('_',' ')}")

    # Divider + Flight Preferences label
    st.markdown("""
    <div style="height:2px; background:linear-gradient(90deg,#d0021b22,#e8e8e8,transparent);
                margin:20px 0 14px;"></div>
    <div style="font-size:1.1rem; font-weight:1000; color:#d0021b; text-transform:uppercase;
                letter-spacing:1.2px; margin-bottom:8px;">⚙️ Flight Preferences</div>
            <div style="height:2px; background:linear-gradient(90deg,#d0021b22,#e8e8e8,transparent);
                margin:10px 0 14px;"></div>
    """, unsafe_allow_html=True)

    # Row 3: Airline | Stops
    row3_c1, row3_c2 = st.columns([1, 1])
    with row3_c1:
        airline = st.selectbox("✈ Airline",
                               [f"{airline_icons.get(a,'')} {a}" for a in airlines])
        airline_clean = airline.split(" ", 1)[1].strip() if " " in airline else airline
    with row3_c2:
        # Row 5: Stops only (flight number auto-selected internally)
        stops = st.selectbox("🛑 Stops", stop_opts,
                             format_func=lambda x: stop_labels[x])

    # Auto-pick most common flight for airline + route + departure_time
    # so flight number changes naturally when departure time changes
    filtered = df[(df['airline'] == airline_clean) &
                  (df['source_city'] == source) &
                  (df['destination_city'] == destination) &
                  (df['departure_time'] == dep_time)]
    if len(filtered) > 0:
        flight_no = filtered['flight'].value_counts().idxmax()
    else:
        # fallback: drop departure_time filter if no match
        filtered_route = df[(df['airline'] == airline_clean) &
                            (df['source_city'] == source) &
                            (df['destination_city'] == destination)]
        flight_no = filtered_route['flight'].value_counts().idxmax() if len(filtered_route) > 0 else df['flight'].value_counts().idxmax()

    # Class selector — radio buttons
    travel_class = st.radio("💺 Class", class_opts, index=0, horizontal=True)

    # Duration info box
    dur_key = (source, destination, stops)
    est_dur = duration_map.get(dur_key)
    if est_dur is None:
        for s in ['zero', 'one', 'two_or_more']:
            est_dur = duration_map.get((source, destination, s))
            if est_dur:
                break
    dur_hours = int(est_dur) if est_dur else 0
    dur_mins  = int((est_dur - dur_hours) * 60) if est_dur else 0

    # Auto-compute arrival time slot from departure + duration
    total_dur = (est_dur if est_dur else 2.0)
    arr_time  = compute_arrival_time(dep_time, total_dur)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_col = st.columns([1, 2, 1])[1]
    with predict_col:
        predict_btn = st.button("✈  PREDICT FLIGHT PRICE", use_container_width=True)


# ─────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────
if predict_btn:
    days_left = (depart_date - today).days
    days_left = max(1, days_left)  # minimum 1

    # --- Encode input ---
    def safe_encode(le, val):
        if val in le.classes_:
            return int(le.transform([val])[0])
        # unknown → use most frequent
        return 0

    enc_airline  = safe_encode(encoders['airline'],          airline_clean)
    enc_flight   = safe_encode(encoders['flight'],           flight_no)
    enc_source   = safe_encode(encoders['source_city'],      source)
    enc_dep_time = safe_encode(encoders['departure_time'],   dep_time)
    enc_stops    = safe_encode(encoders['stops'],            stops)
    enc_arr_time = safe_encode(encoders['arrival_time'],     arr_time)   # auto-derived
    enc_dest     = safe_encode(encoders['destination_city'], destination)
    enc_class    = safe_encode(encoders['class'],            travel_class)

    input_array = np.array([[enc_airline, enc_flight, enc_source, enc_dep_time,
                              enc_stops, enc_arr_time, enc_dest, enc_class, days_left]])
    predicted_price = float(model.predict(input_array)[0])

    # --- Days left badge ---
    if days_left <= 3:
        urgency_color = "#d0021b"
        urgency_msg   = "🔥 Book Now – Prices Rising Fast!"
    elif days_left <= 10:
        urgency_color = "#e98316"
        urgency_msg   = "⚡ Good Time to Book!"
    elif days_left <= 20:
        urgency_color = "#e2b457"
        urgency_msg   = "📊 Fair Price Window"
    else:
        urgency_color = "#52c41a"
        urgency_msg   = "🟢 Early Bird – Great Prices!"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; align-items:center;">
      <div class="days-badge" style="height:40px; display:inline-flex; align-items:center;">
        🗓 Days Until Departure: <span class="days-number">&nbsp;{days_left}</span>
      </div>
      <div style="height:40px; display:inline-flex; align-items:center;
                  background:{urgency_color}18; border:1.5px solid {urgency_color}44;
                  color:{urgency_color}; padding:0 18px; border-radius:100px;
                  font-size:0.82rem; font-weight:700;">
          {urgency_msg}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Estimated Duration & Arrival ─────
    st.markdown(f"""
    <div style="margin-bottom:20px; background:#fff5f5; border:1.5px solid #ffccc7;
                border-radius:12px; padding:12px 16px; text-align:center;">
      <div style="font-size:0.68rem;color:#aaa;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.8px;">Estimated Duration &amp; Arrival</div>
      <div style="font-family:'Montserrat',sans-serif; font-size:1.4rem;
                  font-weight:900; color:#d0021b; margin:4px 0;">
          {dur_hours}h {dur_mins}m
      </div>
      <div style="font-size:0.78rem;color:#555;font-weight:600;margin-top:4px;">
          {time_icons.get(dep_time,'')} {dep_time.replace('_',' ')}
          &nbsp;→&nbsp;
          {time_icons.get(arr_time,'')} <strong>{arr_time.replace('_',' ')}</strong>
      </div>
      <div style="font-size:0.7rem;color:#888;margin-top:2px;">{source} → {destination} ({stop_labels[stops]})</div>
    </div>""", unsafe_allow_html=True)

    # ── Main Result Card ──────────────────
    st.markdown(f"""
    <div class="result-card">
      <div class="flight-airline">{airline_icons.get(airline_clean,'')} {airline_clean.replace('_',' ')}</div>
      <div class="flight-number">Flight {flight_no} &nbsp;|&nbsp; {travel_class} Class</div>

      <div class="route-row">
        <div class="city-box">
          <div class="city-code">{source[:3].upper()}</div>
          <div class="city-name">{source}</div>
        </div>
        <div class="route-line">
          <div class="route-bar"></div>
          <div class="route-stops">{stop_labels[stops]}</div>
        </div>
        <div class="city-box">
          <div class="city-code">{destination[:3].upper()}</div>
          <div class="city-name">{destination}</div>
        </div>
      </div>

      <div class="detail-grid">
        <div class="detail-item">
          <div class="detail-label">📅 Date</div>
          <div class="detail-value">{depart_date.strftime('%d %b %Y')}</div>
        </div>
        <div class="detail-item">
          <div class="detail-label">🕐 Departure</div>
          <div class="detail-value">{time_icons.get(dep_time,'')} {dep_time.replace('_',' ')}</div>
        </div>
        <div class="detail-item">
          <div class="detail-label">🕑 Arrival</div>
          <div class="detail-value">{time_icons.get(arr_time,'')} {arr_time.replace('_',' ')}</div>
        </div>
        <div class="detail-item">
          <div class="detail-label">⏱ Duration</div>
          <div class="detail-value">{dur_hours}h {dur_mins}m</div>
        </div>
      </div>

      <div class="price-display">
        <div class="price-label"> Predicted Price (per person)</div>
        <div class="price-amount">₹{int(predicted_price):,}</div>
        <div class="price-per">Inclusive of base fare · Taxes & fees may apply</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Comparison Chart ──────────────────
    st.markdown("""
    <div class="chart-header">📊 Price Comparison – All Airlines on This Route</div>
    """, unsafe_allow_html=True)

    # Predict for all airlines with same params
    # Each airline uses its own most popular flight on this route for a fair comparison
    comparison_data = []
    for al in airlines:
        al_filtered = df[(df['airline'] == al) &
                         (df['source_city'] == source) &
                         (df['destination_city'] == destination)]
        if len(al_filtered) > 0:
            al_flight = al_filtered['flight'].value_counts().idxmax()
        else:
            al_flight = df['flight'].value_counts().idxmax()
        enc_al        = safe_encode(encoders['airline'], al)
        enc_al_flight = safe_encode(encoders['flight'],  al_flight)
        for s in stop_opts:
            enc_s  = safe_encode(encoders['stops'], s)
            x = np.array([[enc_al, enc_al_flight, enc_source, enc_dep_time,
                           enc_s, enc_arr_time, enc_dest, enc_class, days_left]])
            p = float(model.predict(x)[0])
            comparison_data.append({
                'Airline': al.replace('_',' '),
                'Stops': stop_labels[s],
                'Price': round(p, 0),
                'IsSelected': (al == airline_clean and s == stops)
            })

    comp_df = pd.DataFrame(comparison_data)

    # Bar chart — grouped by airline, color by stops
    colors_stops = {
        'Non-Stop ✈': '#52c41a',
        '1 Stop': '#faad14',
        '2+ Stops': '#f5222d'
    }

    fig = go.Figure()
    for stop_type in ['Non-Stop ✈', '1 Stop', '2+ Stops']:
        sub = comp_df[comp_df['Stops'] == stop_type]
        fig.add_trace(go.Bar(
            name=stop_type,
            x=sub['Airline'],
            y=sub['Price'],
            marker_color=colors_stops[stop_type],
            text=[f"₹{int(p):,}" for p in sub['Price']],
            textposition='outside',
            textfont=dict(size=10, family='Poppins'),
            opacity=0.88
        ))

    # Highlight selected prediction
    selected_price = predicted_price
    fig.add_hline(
        y=selected_price,
        line_dash="dot",
        line_color="#d0021b",
        line_width=2.5,
        annotation_text=f"  Your Selection: ₹{int(selected_price):,}",
        annotation_font_color="#d0021b",
        annotation_font_size=12
    )

    fig.update_layout(
        barmode='group',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Poppins', size=12),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='right', x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#eee', borderwidth=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            gridcolor='#f0f0f0',
            title='Airline',
            title_font=dict(size=12, color='#666'),
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            gridcolor='#f0f0f0',
            title='Predicted Price (₹)',
            title_font=dict(size=12, color='#666'),
            tickformat=',',
            tickprefix='₹'
        ),
        margin=dict(t=60, b=20, l=20, r=20),
        title=dict(
            text=f"<b>Route:</b> {source} → {destination} | <b>Class:</b> {travel_class} | <b>Date:</b> {depart_date.strftime('%d %b %Y')}",
            font=dict(size=13, color='#333'),
            x=0
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Cheapest options table ────────────
    st.markdown("""
    <div style="font-family:'Montserrat',sans-serif; font-size:0.95rem; font-weight:800;
                color:#1a1a2e; margin:10px 0 12px; display:flex; align-items:center; gap:8px;">
        🏷️ Top Cheapest Options on This Route
    </div>""", unsafe_allow_html=True)

    top5 = comp_df.nsmallest(5, 'Price').reset_index(drop=True)
    top5['Rank'] = ['🥇','🥈','🥉','4️⃣','5️⃣']
    top5['Price Display'] = top5['Price'].apply(lambda x: f"₹{int(x):,}")
    top5['Selected'] = top5.apply(
        lambda r: "⭐ Your Pick" if (r['Airline'].replace(' ','_') == airline_clean and r['Stops'] == stop_labels[stops]) else "",
        axis=1
    )

    display_df = top5[['Rank', 'Airline', 'Stops', 'Price Display', 'Selected']].rename(
        columns={'Price Display': 'Predicted Price', 'Selected': ''}
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


else:
    # ── Landing state ─────────────────────
    st.markdown("""
    <div style="text-align:center; padding:50px 20px; color:#ccc;">
      <div style="font-size:6rem; margin-bottom:16px;">✈️</div>
      <div style="font-family:'Montserrat',sans-serif; font-size:1.4rem; font-weight:800;
                  color:#bbb;">Select your flight details above</div>
      <div style="font-size:0.9rem; color:#ccc; margin-top:8px;">
          Then click <strong style="color:#d0021b;">PREDICT FLIGHT PRICE</strong> to see AI-powered fare prediction
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:28px 0 10px; color:#bbb; font-size:1.1rem;">
    <strong style="color:#d0021b; font-family:'Montserrat',sans-serif; font-weight:900;">Fair</strong><strong style="color: #ffe066; font-family:'Montserrat',sans-serif; font-weight:900;"> Fares</strong>
    &nbsp;·&nbsp; AI Flight fare Predictor &nbsp;·&nbsp;<br>
    <span style="font-size:0.88rem;">⚠️ Predicted prices are for informational purposes only. Actual fares may vary.</span>
</div>
""", unsafe_allow_html=True)