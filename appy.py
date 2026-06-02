# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURACIÓN DE COLORES (AZUL PROFESIONAL)
# ==============================================================================
COLORS = {
    "primary": "#2DD4BF",
    "primary_blue": "#3B82F6",
    "primary_dark": "#0F766E",
    "bg": "#0A0F1E",
    "bg_card": "#111827",
    "text": "#F3F4F6",
    "text_muted": "#9CA3AF",
    "border": "#1F2937",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "chart_line": "#2DD4BF",
    "chart_points": "#38BDF8"
}

st.set_page_config(page_title="El Gallito | Modelamiento Matemático", page_icon="🌾", layout="wide")

# CSS con tooltips
st.markdown(f"""
<style>
    .stApp {{ background: {COLORS['bg']}; }}
    .stDataFrame, .stDataEditor {{ background: {COLORS['bg_card']} !important; border-radius: 12px !important; border: 1px solid {COLORS['border']} !important; font-size: 0.85rem !important; }}
    h1, h2, h3 {{ color: {COLORS['primary_blue']} !important; font-weight: 500 !important; margin-bottom: 0.25rem !important; }}
    .stButton button {{ background: linear-gradient(135deg, {COLORS['primary_blue']}, {COLORS['primary_dark']}); color: white !important; font-weight: 500 !important; border: none; border-radius: 8px; padding: 0.25rem 0.75rem; font-size: 0.8rem; }}
    .stButton button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px -4px {COLORS['primary_blue']}80; }}
    .info-card {{ background: {COLORS['bg_card']}; padding: 0.75rem; border-radius: 12px; border: 1px solid {COLORS['border']}; margin: 0.25rem 0; font-size: 0.85rem; }}
    .alert-danger {{ background: #2D1A2B; padding: 0.4rem 0.75rem; border-radius: 8px; border-left: 3px solid {COLORS['danger']}; margin: 0.2rem 0; font-size: 0.8rem; }}
    .alert-warning {{ background: #2D271A; padding: 0.4rem 0.75rem; border-radius: 8px; border-left: 3px solid {COLORS['warning']}; margin: 0.2rem 0; font-size: 0.8rem; }}
    .alert-success {{ background: #0E2A1F; padding: 0.4rem 0.75rem; border-radius: 8px; border-left: 3px solid {COLORS['success']}; margin: 0.2rem 0; font-size: 0.8rem; }}
    .formula-box {{ background: {COLORS['bg_card']}; padding: 0.5rem; border-radius: 12px; font-family: monospace; text-align: center; font-size: 0.9rem; border: 1px solid {COLORS['border']}; margin: 0.5rem 0; }}
    .sidebar-card {{ background: {COLORS['bg_card']}; padding: 0.75rem; border-radius: 16px; border: 1px solid {COLORS['border']}; text-align: center; margin-bottom: 1rem; }}
    .footer {{ text-align: center; padding: 0.75rem 0; color: {COLORS['text_muted']}; font-size: 0.7rem; border-top: 1px solid {COLORS['border']}; margin-top: 1rem; }}
    .metric-tooltip {{ border-bottom: 1px dashed {COLORS['primary_blue']}; cursor: help; display: inline-block; }}
    .metric-card {{ background: {COLORS['bg_card']}; border-radius: 12px; padding: 0.5rem; text-align: center; border: 1px solid {COLORS['border']}; }}
    .metric-value {{ font-size: 1.8rem; font-weight: bold; color: {COLORS['primary_blue']}; line-height: 1.2; }}
    .metric-label {{ font-size: 0.85rem; color: {COLORS['text_muted']}; margin-top: 0.25rem; }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATOS
# ==============================================================================
@st.cache_data
def load_real_data():
    return pd.DataFrame({
        "Mes": ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"],
        "Periodo": list(range(1,13)),
        "Ventas_Soles": [227300,251920,278700,389000,445200,411700,336100,307400,279900,351420,446700,508000],
        "Clientes": [950,1000,1050,1200,1300,1250,1150,1100,1050,1100,1300,1400]
    })

@st.cache_data
def load_synthetic_data():
    ventas = [215000,230000,243000,258000,267000,274000,279000,282000,283000,278000,275000,270000]
    return pd.DataFrame({
        "Mes": ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"],
        "Periodo": list(range(1,13)),
        "Ventas_Soles": ventas,
        "Clientes": [950,1000,1050,1200,1300,1250,1150,1100,1050,1100,1300,1400]
    })

if "df" not in st.session_state:
    st.session_state.df = load_real_data()
if "synthetic_mode" not in st.session_state:
    st.session_state.synthetic_mode = False

# ==============================================================================
# BARRA LATERAL
# ==============================================================================
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-card">
        <h3 style="margin:0;">TECSUP</h3>
        <p style="margin:0; font-size:0.8rem;">Proyecto Integrador</p>
        <p style="color:{COLORS['text_muted']}; font-size:0.7rem;">Matemática Aplicada</p>
        <hr>
        <p><strong style="color:{COLORS['primary_blue']};">C30S-S</strong> | Grupo D</p>
        <hr>
        <p style="color:{COLORS['text_muted']}; font-size:0.7rem;">Agro-Distribuciones El Gallito<br>La Joya, Arequipa</p>
    </div>
    """, unsafe_allow_html=True)
    
    synthetic_toggle = st.checkbox("Usar dataset sintético (un solo pico)", value=st.session_state.synthetic_mode)
    if synthetic_toggle != st.session_state.synthetic_mode:
        st.session_state.synthetic_mode = synthetic_toggle
        st.session_state.df = load_synthetic_data() if synthetic_toggle else load_real_data()
        st.rerun()
    
    st.markdown(f"<div class='info-card' style='text-align:center;'><span style='color:{COLORS['text_muted']};'>Próxima revisión</span><br><span style='color:{COLORS['primary_blue']};'>{ (datetime.now()+timedelta(days=180)).strftime('%d/%m/%Y') }</span></div>", unsafe_allow_html=True)

# ==============================================================================
# EDITOR DE DATOS
# ==============================================================================
st.markdown("### Editor de datos en vivo")
edited_df = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Periodo": st.column_config.NumberColumn("Periodo", min_value=1, max_value=12, step=1),
        "Ventas_Soles": st.column_config.NumberColumn("Ventas (Soles)", step=1000, format="%.0f"),
        "Clientes": st.column_config.NumberColumn("Clientes", step=50)
    }
)
if st.button("Actualizar modelo", use_container_width=True):
    st.session_state.df = edited_df.copy()
    st.rerun()

# ==============================================================================
# FUNCIÓN PARA MÉTRICAS
# ==============================================================================
def get_metrics(df, deg):
    X = df["Periodo"].values.reshape(-1,1)
    y = df["Ventas_Soles"].values
    poly = PolynomialFeatures(degree=deg)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred)/y))*100
    return model, poly, r2, rmse, mae, mape, y_pred

# ==============================================================================
# COMPARACIÓN DE GRADOS
# ==============================================================================
st.markdown("## Comparación de grados polinomiales (1 a 5)")

degrees_to_compare = [1, 2, 3, 4, 5]
results = []

for d in degrees_to_compare:
    X_temp = st.session_state.df["Periodo"].values.reshape(-1,1)
    y_temp = st.session_state.df["Ventas_Soles"].values
    poly_temp = PolynomialFeatures(degree=d)
    X_poly_temp = poly_temp.fit_transform(X_temp)
    model_temp = LinearRegression().fit(X_poly_temp, y_temp)
    y_pred_temp = model_temp.predict(X_poly_temp)
    r2_temp = r2_score(y_temp, y_pred_temp)
    rmse_temp = np.sqrt(mean_squared_error(y_temp, y_pred_temp))
    mae_temp = mean_absolute_error(y_temp, y_pred_temp)
    mape_temp = np.mean(np.abs((y_temp - y_pred_temp)/y_temp))*100
    results.append({"Grado": d, "R²": r2_temp, "RMSE": rmse_temp, "MAE": mae_temp, "MAPE (%)": mape_temp})

df_compare = pd.DataFrame(results)
st.dataframe(df_compare.style.format({
    "R²": "{:.4f}", 
    "RMSE": "{:,.0f}", 
    "MAE": "{:,.0f}", 
    "MAPE (%)": "{:.1f}"
}), use_container_width=True)

# Gráfico comparativo
X_plot = np.linspace(1, 12, 200)
fig_compare = go.Figure()
fig_compare.add_trace(go.Scatter(
    x=st.session_state.df["Periodo"], 
    y=st.session_state.df["Ventas_Soles"],
    mode='markers', 
    name='📊 Datos reales',
    marker=dict(size=10, color=COLORS['chart_points']),
    text=st.session_state.df["Mes"]
))

colores_grado = ['#F59E0B', '#3B82F6', '#10B981', '#EF4444', '#8B5CF6']
X_temp = st.session_state.df["Periodo"].values.reshape(-1,1)
y_temp = st.session_state.df["Ventas_Soles"].values

for i, d in enumerate(degrees_to_compare):
    poly_temp = PolynomialFeatures(degree=d)
    X_poly_temp = poly_temp.fit_transform(X_temp)
    model_temp = LinearRegression().fit(X_poly_temp, y_temp)
    y_curve = model_temp.predict(poly_temp.transform(X_plot.reshape(-1,1)))
    fig_compare.add_trace(go.Scatter(
        x=X_plot, y=y_curve, mode='lines',
        name=f'Grado {d}', 
        line=dict(color=colores_grado[i], width=2, dash='solid' if d==2 else 'dot')
    ))

fig_compare.update_layout(
    height=400, 
    title="Ajuste de diferentes grados polinomiales",
    xaxis_title="Mes", 
    yaxis_title="Ventas (S/)",
    plot_bgcolor=COLORS['bg'], 
    paper_bgcolor=COLORS['bg'],
    font=dict(color=COLORS['text']),
    xaxis=dict(gridcolor=COLORS['border']),
    yaxis=dict(gridcolor=COLORS['border']),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_compare, use_container_width=True)

# ==============================================================================
# EXPLICACIÓN MATEMÁTICA
# ==============================================================================
with st.expander("📐 ¿Por qué los grados altos (≥4) son inestables?"):
    st.markdown(r"""
    **1. Fenómeno de Runge** – Oscilaciones violentas en los extremos.  
    **2. Sobreajuste** – Con 12 puntos, grado 5 fuerza el paso exacto pero genera picos falsos.  
    **3. Varianza elevada** – Pequeños cambios en datos → cambios enormes en coeficientes.  
    **4. Extrapolación catastrófica** – Para t>12, los polinomios altos se disparan a ±∞.  
    **Conclusión:** Usar grado 2 para un solo pico; para dos picos (mayo/diciembre) usar modelos más robustos.
    """)

# ==============================================================================
# MODELO SELECCIONADO
# ==============================================================================
st.markdown("## Modelo seleccionado y análisis detallado")
degree = st.selectbox("Grado del polinomio (seleccione para análisis)", [1,2,3,4,5], index=1)

model, poly, r2, rmse, mae, mape, y_pred = get_metrics(st.session_state.df, degree)
X = st.session_state.df["Periodo"].values
y = st.session_state.df["Ventas_Soles"].values

# Tooltips
tooltips = {
    "R²": "Coeficiente de determinación: % de variación de ventas que explica el modelo.",
    "RMSE": "Error típico de predicción en soles.",
    "MAE": "Error absoluto medio en soles.",
    "MAPE": "Error porcentual absoluto medio."
}

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("R²", f"{r2:.4f}")
    st.caption(f"ℹ️ {tooltips['R²']}")
with col2:
    st.metric("RMSE", f"S/ {rmse:,.0f}")
    st.caption(f"ℹ️ {tooltips['RMSE']}")
with col3:
    st.metric("MAE", f"S/ {mae:,.0f}")
    st.caption(f"ℹ️ {tooltips['MAE']}")
with col4:
    st.metric("MAPE", f"{mape:.1f}%")
    st.caption(f"ℹ️ {tooltips['MAPE']}")

# Ecuación
coefs = model.coef_
intercept = model.intercept_
terms = []
if abs(intercept) > 1e-6:
    terms.append(f"{intercept:,.2f}")
for i in range(1, degree+1):
    if i < len(coefs) and abs(coefs[i]) > 1e-6:
        sign = "+" if coefs[i] > 0 else "-"
        terms.append(f"{sign} {abs(coefs[i]):,.2f}·t^{i}" if i>1 else f"{sign} {abs(coefs[i]):,.2f}·t")
eq = "V(t) = " + " ".join(terms).replace("+ -", "- ")
st.markdown(f"<div class='formula-box'><code>{eq}</code></div>", unsafe_allow_html=True)

# Gráfico
t_smooth = np.linspace(1, 12, 200)
y_smooth = model.predict(poly.transform(t_smooth.reshape(-1,1)))
fig = go.Figure()
fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Reales', marker=dict(size=8, color=COLORS['chart_points']), text=st.session_state.df["Mes"]))
fig.add_trace(go.Scatter(x=t_smooth, y=y_smooth, mode='lines', name=f'Grado {degree}', line=dict(color=COLORS['chart_line'], width=2)))
fig.update_layout(height=350, plot_bgcolor=COLORS['bg'], paper_bgcolor=COLORS['bg'], font=dict(color=COLORS['text']), xaxis=dict(gridcolor=COLORS['border']), yaxis=dict(gridcolor=COLORS['border']))
st.plotly_chart(fig, use_container_width=True)

# Análisis
st.markdown("### Análisis de resultados")
if degree >= 4:
    st.error(f"❌ Grado {degree} → SOBREAJUSTE PELIGROSO. No usar para decisiones.")
elif degree == 2 and not st.session_state.synthetic_mode and r2 < 0.7:
    st.warning(f"⚠️ R² = {r2:.4f} bajo porque los datos tienen DOS picos (mayo y diciembre).")
else:
    st.success(f"✅ R² = {r2:.4f}")

# ==============================================================================
# PREDICCIONES
# ==============================================================================
st.markdown("### Predicciones para el próximo año")
if degree >= 4:
    st.error("Grados ≥4 no son confiables.")
else:
    future = list(range(13,25))
    meses = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    preds = [model.predict(poly.transform(np.array([[t]])))[0] for t in future]
    df_future = pd.DataFrame({"Mes": meses, "Proyección (S/)": [f"{p:,.0f}" for p in preds]})
    st.dataframe(df_future, use_container_width=True, hide_index=True)

st.markdown(f"""
<div class="footer">
    TECSUP – Matemática Aplicada a la Mecánica | C30S-S | Grupo D | Agro-Distribuciones El Gallito
</div>
""", unsafe_allow_html=True)
