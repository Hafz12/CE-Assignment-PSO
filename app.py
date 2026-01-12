import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Delhi Metro Passenger Prediction & Optimization (PSO)",
    layout="wide"
)

# =====================================================
# Load PSO model and scaler (Prediction Model)
# =====================================================
model = joblib.load("pso_model.pkl")
scaler = joblib.load("scaler.pkl")

weights = np.array(model["weights"])
bias = float(model["bias"])

feature_names = ["Distance_km", "Fare", "Cost_per_passenger"]

# =====================================================
# Load Dataset
# =====================================================
data = pd.read_csv("delhi_metro_updated.csv")
data = data[['Distance_km', 'Fare', 'Cost_per_passenger', 'Passengers']].dropna()

distance_arr = data["Distance_km"].values
fare_arr = data["Fare"].values

# =====================================================
# Title
# =====================================================
st.title("🚇 Delhi Metro Passenger Prediction & Optimization (PSO)")

st.markdown(
    """
    This application integrates **Particle Swarm Optimization (PSO)** for:

    - 📊 **Passenger demand prediction** (PSO-optimized regression)
    - 📈 **Multi-objective optimization** of **Distance and Fare**
    """
)

# =====================================================
# Sidebar – User Inputs (Prediction)
# =====================================================
st.sidebar.header("🔢 Passenger Prediction Inputs")

distance = st.sidebar.number_input("Distance (km)", 0.0, 100.0, 12.94)
fare = st.sidebar.number_input("Fare (₹)", 0.0, 200.0, 77.99)
cost = st.sidebar.number_input("Cost per Passenger (₹)", 0.0, 100.0, 18.27)

X_input = pd.DataFrame([{
    "Distance_km": distance,
    "Fare": fare,
    "Cost_per_passenger": cost
}])

X_scaled = scaler.transform(X_input)

# =====================================================
# Prediction
# =====================================================
y_pred = np.dot(X_scaled, weights) + bias
y_pred = max(0, y_pred.item())

# =====================================================
# Prediction Results
# =====================================================
st.subheader("📊 Passenger Demand Prediction (PSO Model)")

pcol1, pcol2, pcol3, pcol4 = st.columns(4)

pcol1.metric("Distance (km)", f"{distance:.2f}")
pcol2.metric("Fare (₹)", f"{fare:.2f}")
pcol3.metric("Cost / Passenger (₹)", f"{cost:.2f}")
pcol4.metric("Predicted Passengers", f"{y_pred:.2f}")

# =====================================================
# Feature Contribution & Sensitivity
# =====================================================
st.subheader("📈 Feature Contribution & Sensitivity Analysis")

weights_flat = weights.flatten()
contribution_raw = X_scaled[0] * weights_flat
total_contribution = contribution_raw.sum()

if total_contribution != 0:
    contribution_scaled = contribution_raw / total_contribution * y_pred
else:
    contribution_scaled = np.zeros_like(contribution_raw)

contrib_df = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": contribution_scaled,
    "Impact": np.abs(contribution_scaled)
})

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Feature Contribution**")
    st.bar_chart(contrib_df.set_index("Feature")["Contribution"])

with c2:
    st.markdown("**Sensitivity Analysis**")
    st.bar_chart(contrib_df.set_index("Feature")["Impact"])

# =====================================================
# MULTI-OBJECTIVE PSO (Distance & Fare)
# =====================================================
st.divider()
st.subheader("📈 Multi-Objective Optimization (Distance vs Fare)")

st.markdown(
    """
    This section applies **PSO-inspired multi-objective analysis**
    to identify **Pareto-optimal trade-offs** between **Distance** and **Fare**.
    """
)

# =====================================================
# Pareto Front Function
# =====================================================
def pareto_front(dist, cost):
    pareto = []
    for i in range(len(dist)):
        dominated = False
        for j in range(len(dist)):
            if (
                dist[j] <= dist[i] and
                cost[j] <= cost[i] and
                (dist[j] < dist[i] or cost[j] < cost[i])
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    return pareto

pareto_idx = pareto_front(distance_arr, fare_arr)

# =====================================================
# Pareto Front Plot
# =====================================================
plot_df = pd.DataFrame({
    "Distance": distance_arr,
    "Fare": fare_arr,
    "Type": [
        "Pareto-optimal" if i in pareto_idx else "Dominated"
        for i in range(len(distance_arr))
    ]
})

chart = alt.Chart(plot_df).mark_circle(size=80).encode(
    x="Distance",
    y="Fare",
    color=alt.Color(
        "Type",
        scale=alt.Scale(
            domain=["Pareto-optimal", "Dominated"],
            range=["red", "lightgray"]
        ),
        legend=alt.Legend(title="Solution Type")
    ),
    tooltip=["Distance", "Fare", "Type"]
)

st.altair_chart(chart, use_container_width=True)

st.info("🔴 Red points represent Pareto-optimal solutions.")

# =====================================================
# Dataset Preview
# =====================================================
st.subheader("🗂 Dataset Preview")

with st.expander("Show dataset sample"):
    st.dataframe(data.head(10))

# =====================================================
# PSO Explanation
# =====================================================
with st.expander("🧠 How PSO Is Used in This System"):
    st.markdown(
        """
        **Prediction Module**
        - PSO optimizes regression weights and bias
        - Fitness function minimizes Mean Squared Error (MSE)

        **Optimization Module**
        - Distance and Fare are treated as conflicting objectives
        - Pareto front identifies optimal trade-offs
        - Supports decision-making without collapsing objectives
        """
    )

# =====================================================
# Conclusion
# =====================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
    This integrated system demonstrates how **Particle Swarm Optimization**
    can be applied to both **prediction** and **multi-objective optimization**
    for intelligent metro transport planning.
    """
)
