import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="OncoFin – Онкологиялық скрининг экономикалық болжам",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background-color: #f7fafc;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #3182ce;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
st.sidebar.title("⚙️ Параметрлер")
st.sidebar.markdown("---")

st.sidebar.subheader("Негізгі шығындар")
capex = st.sidebar.number_input("CapEx (бастапқы инвестиция, тг)", value=20_500_000, step=100_000, format="%d")
opex = st.sidebar.number_input("OpEx (жылдық, тг)", value=7_600_000, step=100_000, format="%d")

st.sidebar.subheader("Экономикалық пайда")
cost_early = st.sidebar.number_input("1-2 стадия емдеу құны (тг)", value=800_000, step=50_000, format="%d")
cost_late = st.sidebar.number_input("3-4 стадия емдеу құны (тг)", value=5_500_000, step=100_000, format="%d")
savings_per_patient = cost_late - cost_early

num_early = st.sidebar.slider("Ерте анықталған пациенттер саны (жылына)", 5, 50, 20)

st.sidebar.subheader("Инвестициялық параметрлер")
discount_rate = st.sidebar.slider("Дисконттау ставкасы (%)", 5.0, 20.0, 10.0, 0.5) / 100
years = st.sidebar.slider("Болжам мерзімі (жыл)", 1, 5, 3)

st.sidebar.markdown("---")
scenario = st.sidebar.selectbox(
    "Сценарий",
    ["Базалық", "Пессимистік", "Оптимистік"],
    index=0
)

# Scenario adjustments
if scenario == "Пессимистік":
    num_early = 5
    opex = int(opex * 1.20)
elif scenario == "Оптимистік":
    num_early = 35

annual_benefit = num_early * savings_per_patient
net_cash_flow = annual_benefit - opex

# ==================== HEADER ====================
st.markdown('<p class="main-header">🏥 OncoFin</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Онкологиялық скринингті оңтайландыру: экономикалық тиімділік және 3 жылдық болжам</p>', unsafe_allow_html=True)

# ==================== KEY METRICS ====================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Жылдық пайда (Benefit)", f"{annual_benefit:,.0f} ₸".replace(",", " "))
with col2:
    st.metric("Таза ақша ағыны", f"{net_cash_flow:,.0f} ₸".replace(",", " "))
with col3:
    payback_months = (capex / net_cash_flow) * 12 if net_cash_flow > 0 else float("inf")
    st.metric("Өзін-өзі өтеу мерзімі", f"{payback_months:.1f} ай" if payback_months < 100 else "∞")
with col4:
    st.metric("1 пациент үнемі", f"{savings_per_patient:,.0f} ₸".replace(",", " "))

st.markdown("---")

# ==================== NPV CALCULATION ====================
def calculate_npv(cash_flows, rate, initial_investment):
    npv = -initial_investment
    for t, cf in enumerate(cash_flows, 1):
        npv += cf / ((1 + rate) ** t)
    return npv

cash_flows = [net_cash_flow] * years
npv = calculate_npv(cash_flows, discount_rate, capex)

# Discounted cash flows table
disc_cfs = []
cum_disc = -capex
rows = []
for t in range(1, years + 1):
    disc = net_cash_flow / ((1 + discount_rate) ** t)
    cum_disc += disc
    disc_cfs.append(disc)
    rows.append({
        "Жыл": t,
        "Таза ақша ағыны (тг)": f"{net_cash_flow:,.0f}",
        "Дисконтталған (тг)": f"{disc:,.0f}",
        "Жинақталған NPV (тг)": f"{cum_disc:,.0f}"
    })

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Негізгі көрсеткіштер",
    "📈 3 жылдық болжам",
    "🤖 ML болжам",
    "💰 Сценарийлер",
    "ℹ️ Модель туралы"
])

# ---------- TAB 1 ----------
with tab1:
    st.subheader("Инвестициялық талдау")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        **CapEx (бастапқы инвестиция):** {capex:,.0f} ₸  
        **OpEx (жылдық):** {opex:,.0f} ₸  
        **Жылдық Benefit:** {annual_benefit:,.0f} ₸  
        **Таза ақша ағыны:** {net_cash_flow:,.0f} ₸  
        **Дисконттау ставкасы:** {discount_rate*100:.1f}%  
        """)
    with c2:
        st.markdown(f"""
        **NPV ({years} жыл):** **{npv:,.0f} ₸**  
        **Өзін-өзі өтеу:** {payback_months:.1f} ай  
        **ICER (1 ерте жағдай):** {capex / num_early if num_early > 0 else 0:,.0f} ₸  
        **Үнем коэффициенті:** {(cost_late / (capex / num_early) if num_early > 0 else 0):.1f}x  
        """)
    
    st.subheader("Дисконтталған ақша ағындары")
    df_npv = pd.DataFrame(rows)
    st.dataframe(df_npv, use_container_width=True, hide_index=True)
    
    # NPV waterfall-like chart
    fig_npv = go.Figure()
    fig_npv.add_trace(go.Bar(
        x=["Бастапқы инвестиция"] + [f"Жыл {t}" for t in range(1, years+1)] + ["Қорытынды NPV"],
        y=[-capex] + disc_cfs + [npv],
        marker_color=["#e53e3e"] + ["#38a169"]*years + ["#3182ce"],
        text=[f"{v/1e6:.1f}M" for v in [-capex] + disc_cfs + [npv]],
        textposition="outside"
    ))
    fig_npv.update_layout(
        title="NPV құрылымы (млн ₸)",
        yaxis_title="Сома (тг)",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_npv, use_container_width=True)

# ---------- TAB 2 ----------
with tab2:
    st.subheader(f"Алдағы {years} жылға экономикалық болжам")
    
    # Build forecast dataframe
    forecast_data = []
    cum_cash = -capex
    for t in range(1, years + 1):
        benefit_t = annual_benefit
        opex_t = opex
        net_t = benefit_t - opex_t
        disc_t = net_t / ((1 + discount_rate) ** t)
        cum_cash += disc_t
        forecast_data.append({
            "Жыл": 2026 + t - 1,
            "Ерте анықтау (адам)": num_early,
            "Benefit (тг)": benefit_t,
            "OpEx (тг)": opex_t,
            "Таза ағын (тг)": net_t,
            "Дисконтталған (тг)": disc_t,
            "Жинақталған NPV (тг)": cum_cash
        })
    
    df_forecast = pd.DataFrame(forecast_data)
    
    # Charts
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig1 = px.bar(
            df_forecast, x="Жыл", y=["Benefit (тг)", "OpEx (тг)"],
            barmode="group",
            title="Жылдық Benefit vs OpEx",
            labels={"value": "Сома (тг)", "variable": "Көрсеткіш"},
            color_discrete_map={"Benefit (тг)": "#38a169", "OpEx (тг)": "#e53e3e"}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_b:
        fig2 = px.line(
            df_forecast, x="Жыл", y="Жинақталған NPV (тг)",
            markers=True,
            title="Жинақталған NPV динамикасы",
            labels={"Жинақталған NPV (тг)": "NPV (тг)"}
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.dataframe(
        df_forecast.style.format({
            "Benefit (тг)": "{:,.0f}",
            "OpEx (тг)": "{:,.0f}",
            "Таза ағын (тг)": "{:,.0f}",
            "Дисконтталған (тг)": "{:,.0f}",
            "Жинақталған NPV (тг)": "{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

# ---------- TAB 3: ML FORECAST ----------
with tab3:
    st.subheader("🤖 Машиналық оқыту негізіндегі болжам")
    st.markdown("""
    Модель тарихи (синтетикалық) деректерге негізделген **Linear Regression + Polynomial** пайдаланады.  
    Деректер: 2021–2025 жылдардағы шартты көрсеткіштер (ерте анықтау саны, Benefit).
    """)
    
    # Synthetic historical data based on project logic
    years_hist = np.array([2021, 2022, 2023, 2024, 2025]).reshape(-1, 1)
    # Assume gradual improvement: early detections growing
    early_hist = np.array([8, 11, 14, 17, 20])
    benefit_hist = early_hist * savings_per_patient
    
    # Train simple model
    poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_model.fit(years_hist, early_hist)
    
    # Future years
    future_years = np.array([2026, 2027, 2028, 2029, 2030]).reshape(-1, 1)
    pred_early = poly_model.predict(future_years)
    pred_early = np.clip(pred_early, 5, 50).astype(int)
    
    # Also linear for benefit
    lin_model = LinearRegression()
    lin_model.fit(years_hist, benefit_hist)
    pred_benefit = lin_model.predict(future_years)
    
    # Combine
    ml_df = pd.DataFrame({
        "Жыл": future_years.flatten(),
        "Болжалды ерте анықтау": pred_early,
        "Болжалды Benefit (тг)": pred_benefit.astype(int),
        "Болжалды OpEx (тг)": [opex] * 5,
        "Болжалды таза ағын (тг)": (pred_benefit - opex).astype(int)
    })
    
    # Historical + forecast plot
    hist_df = pd.DataFrame({
        "Жыл": years_hist.flatten(),
        "Ерте анықтау": early_hist,
        "Тип": "Тарихи"
    })
    fut_df = pd.DataFrame({
        "Жыл": future_years.flatten(),
        "Ерте анықтау": pred_early,
        "Тип": "Болжам"
    })
    plot_df = pd.concat([hist_df, fut_df])
    
    fig_ml = px.line(
        plot_df, x="Жыл", y="Ерте анықтау", color="Тип",
        markers=True,
        title="Ерте анықтау саны: тарихи + ML болжам",
        color_discrete_map={"Тарихи": "#3182ce", "Болжам": "#e53e3e"}
    )
    st.plotly_chart(fig_ml, use_container_width=True)
    
    st.dataframe(
        ml_df.style.format({
            "Болжалды Benefit (тг)": "{:,.0f}",
            "Болжалды OpEx (тг)": "{:,.0f}",
            "Болжалды таза ағын (тг)": "{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.info("📌 Ескерту: ML модель синтетикалық тарихи деректерге үйретілген. Нақты деректер қосылғанда дәлдік артады.")

# ---------- TAB 4: SCENARIOS ----------
with tab4:
    st.subheader("Сценарийлік талдау")
    
    scenarios = {
        "Пессимистік": {"early": 5, "opex_mult": 1.20},
        "Базалық": {"early": 20, "opex_mult": 1.00},
        "Оптимистік": {"early": 35, "opex_mult": 1.00}
    }
    
    scen_rows = []
    for name, params in scenarios.items():
        e = params["early"]
        o = int(opex * params["opex_mult"]) if name == "Пессимистік" else opex
        ben = e * savings_per_patient
        net = ben - o
        cfs = [net] * 3
        npv_s = calculate_npv(cfs, 0.10, capex)
        pb = (capex / net * 12) if net > 0 else float("inf")
        scen_rows.append({
            "Сценарий": name,
            "Ерте анықтау": e,
            "OpEx (тг)": o,
            "Таза ағын (тг)": net,
            "3 жылдық NPV (тг)": npv_s,
            "Өзін-өзі өтеу (ай)": round(pb, 1) if pb < 100 else "—"
        })
    
    df_scen = pd.DataFrame(scen_rows)
    st.dataframe(
        df_scen.style.format({
            "OpEx (тг)": "{:,.0f}",
            "Таза ағын (тг)": "{:,.0f}",
            "3 жылдық NPV (тг)": "{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    fig_scen = px.bar(
        df_scen, x="Сценарий", y="3 жылдық NPV (тг)",
        color="Сценарий",
        title="Сценарийлер бойынша 3 жылдық NPV",
        color_discrete_map={
            "Пессимистік": "#e53e3e",
            "Базалық": "#3182ce",
            "Оптимистік": "#38a169"
        },
        text=[f"{v/1e6:.0f}M" for v in df_scen["3 жылдық NPV (тг)"]]
    )
    fig_scen.update_traces(textposition="outside")
    st.plotly_chart(fig_scen, use_container_width=True)

# ---------- TAB 5 ----------
with tab5:
    st.subheader("Модель туралы")
    st.markdown("""
    ### OncoFin экономикалық моделі
    
    **Негізгі формулалар (жобадан):**
    
    - **1 пациентті ерте анықтау үнемі** = 3-4 стадия құны − 1-2 стадия құны = 5 500 000 − 800 000 = **4 700 000 ₸**
    - **Жылдық Benefit** = ерте анықталған пациенттер × 4 700 000 ₸
    - **Таза ақша ағыны** = Benefit − OpEx
    - **NPV** = −CapEx + Σ (CF_t / (1 + r)^t)
    - **Өзін-өзі өтеу мерзімі** = CapEx / Таза ақша ағыны × 12 (ай)
    
    **Базалық параметрлер (жоба бойынша):**
    | Көрсеткіш | Мәні |
    |-----------|------|
    | CapEx | 20 500 000 ₸ |
    | OpEx | 7 600 000 ₸/жыл |
    | Базалық ерте анықтау | 20 адам/жыл |
    | Дисконттау | 10% |
    | 3 жылдық NPV (базалық) | ≈ +194 364 013 ₸ |
    | Өзін-өзі өтеу | ≈ 2.8 ай |
    
    **Микроэкономикалық әсер (1 кабинет):**
    - Өткізу қабілеті: 15 → 22 пациент/күн (+46%)
    - Бос тұру уақыты: 2.5 → 0.8 сағат (−68%)
    - 1 тексеру өзіндік құны: 12 500 → 8 523 ₸ (−31.8%)
    
    ---
    *Назарбаев Зияткерлік Мектептері · Ақтау · 10-сынып*  
    *Авторлары: Сарсенбиғалиқызы Зере, Назарбай Аружан*  
    *Жетекші: Қонырбаева Ж.М.*
    """)

# Footer
st.markdown("---")
st.caption("OncoFin v1.0 · Экономикалық болжам модулі · Streamlit · Python")
