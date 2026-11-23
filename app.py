import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).parent / "artifacts"
st.set_page_config(page_title="Sri Lankan Car Price Ensembles", page_icon="🚗", layout="wide")

COLORS = {"Decision Tree (baseline)": "#adb5bd", "Random Forest": "#2a9d8f", "XGBoost": "#e76f51",
          "Stacking": "#e9a400", "Hybrid RF + ANN": "#7b2cbf", "ANN (alone)": "#8d99ae"}


@st.cache_data
def load_tables():
    return (pd.read_csv(ART / "results.csv"), pd.read_csv(ART / "feature_importance.csv", index_col=0),
            pd.read_csv(ART / "test_predictions.csv"), pd.read_csv(ART / "clean_data.csv"),
            json.load(open(ART / "meta.json")), json.load(open(ART / "stacking_weights.json")))


@st.cache_resource
def load_models():
    return {"Random Forest": joblib.load(ART / "random_forest.pkl"), "XGBoost": joblib.load(ART / "xgboost.pkl"),
            "Stacking": joblib.load(ART / "stacking.pkl"), "Hybrid RF + ANN": joblib.load(ART / "hybrid_rf_ann.pkl")}


def predict(m, name, X):
    if name == "Hybrid RF + ANN":
        Xp = m["prep"].transform(X)
        return np.expm1(m["ann"].predict(np.column_stack([Xp, m["rf"].predict(Xp)])))[0]
    return np.expm1(m.predict(X))[0]


def takeaway(text):
    st.markdown(f"**Takeaway:** {text}")


results, fi, preds, data, meta, weights = load_tables()
R = results.set_index("Model")

st.title("🚗 Sri Lankan Used Car Price Prediction")
st.caption("Bagging vs boosting vs stacking vs hybrid · Data: ikman.lk & riyasewana.lk (Kaggle) · "
           "Prices in lakhs (1 lakh = Rs 100,000)")

c2, c3, c4 = st.columns(3)
c2.metric("Lowest error", "Random Forest", f"{R.loc['Random Forest', 'MAE (lakhs)']:.2f} lakhs", delta_color="off")
c3.metric("Best R²", "Stacking", f"{R.loc['Stacking', 'R2']:.3f}", delta_color="off")
c4.metric("Fastest ensemble", "XGBoost", f"{R.loc['XGBoost', 'Train time (s)']:.1f} s", delta_color="off")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🏆 Models", "🧬 Stacking & hybrid", "🧮 Predict a price"])

# ---------------- Data ----------------
with tab1:
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Car ads", f"{len(data):,}")
    d2.metric("Brands", data["Brand"].nunique())
    d3.metric("Models", f"{data['Model'].nunique():,}")
    d4.metric("Median price", f"{data['Price'].median():.0f} lakhs")

    def med_bar(col, title, top=None, horizontal=False):
        g = data.groupby(col)["Price"].agg(["median", "count"])
        if top:
            g = g.query("count >= 20").sort_values("count", ascending=False).head(top)
        g = g.sort_values("median", ascending=horizontal).reset_index()
        fig = px.bar(g, x="median" if horizontal else col, y=col if horizontal else "median",
                     orientation="h" if horizontal else "v", text_auto=".0f", title=title,
                     labels={"median": "Median price (lakhs)", col: ""}, color_discrete_sequence=["#2a9d8f"])
        fig.update_traces(textangle=0, textposition="outside", cliponaxis=False)
        return fig.update_layout(height=380)

    l, r = st.columns(2)
    l.plotly_chart(med_bar("Brand", "Median price by brand (10 most advertised)", top=10, horizontal=True), width="stretch")
    age = data.assign(Age=2025 - data["YOM"]).query("Age <= 40").groupby("Age")["Price"].median().reset_index()
    r.plotly_chart(px.line(age, x="Age", y="Price", markers=True, title="Median price by car age",
                           labels={"Age": "Age (years)", "Price": "Median price (lakhs)"},
                           color_discrete_sequence=["#2a9d8f"]).update_layout(height=380), width="stretch")

    l, r = st.columns(2)
    l.plotly_chart(med_bar("Fuel Type", "Median price by fuel type"), width="stretch")
    r.plotly_chart(med_bar("Gear", "Median price by gear"), width="stretch")

    st.info("""**Key findings**
- **Brand and age matter most.** Mercedes-Benz costs about twice as much as Toyota (BMW and Audi even more), and prices drop steadily as cars get older.
- **Hybrids and automatics cost more**, partly because they are newer cars.
- **Mileage was dropped:** it was calculated from the year (correlation -1.0), not real readings. 252 duplicate ads were also removed.
- **Prices are skewed** (a few very expensive cars), so the models were trained on log price.""")

# ---------------- Models ----------------
with tab2:
    metric = st.radio("Compare by", ["MAE (lakhs)", "R2"], horizontal=True,
                      format_func=lambda m: "Average error (MAE)" if m.startswith("MAE") else "R²")
    res = results.sort_values(metric, ascending=metric != "R2")
    fig = px.bar(res, x=metric, y="Model", color="Model", color_discrete_map=COLORS, orientation="h",
                 text_auto=".3f" if metric == "R2" else ".2f",
                 title="Lower is better" if metric.startswith("MAE") else "Higher is better")
    fig.update_layout(showlegend=False, yaxis={"categoryorder": "array", "categoryarray": res["Model"].tolist()[::-1]})
    if metric == "R2":
        fig.update_xaxes(range=[0.8, 0.92])
    st.plotly_chart(fig, width="stretch")
    takeaway("All ensembles beat the single tree. Random Forest has the lowest average error; "
             "stacking explains the most variation (best R²).")

    disp = results.copy()
    disp["Train time (s)"] = disp["Train time (s)"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1f}")
    st.dataframe(disp.drop(columns="Family").style.format(
        {"R2": "{:.3f}", "MAE (lakhs)": "{:.2f}", "RMSE (lakhs)": "{:.2f}", "MAPE (%)": "{:.1f}"}),
        width="stretch", hide_index=True)

    st.subheader("Random Forest vs XGBoost")
    l, r = st.columns(2)
    mx = preds["actual"].max()
    for col, name in [(l, "Random Forest"), (r, "XGBoost")]:
        fig = px.scatter(preds, x="actual", y=name, opacity=0.4, log_x=True, log_y=True,
                         color_discrete_sequence=[COLORS[name]], title=f"{name}: actual vs predicted",
                         labels={"actual": "Actual (lakhs)", name: "Predicted (lakhs)"})
        fig.add_trace(go.Scatter(x=[2, mx], y=[2, mx], mode="lines", line=dict(dash="dash", color="gray"),
                                 showlegend=False))
        col.plotly_chart(fig, width="stretch")
    takeaway(f"RF is slightly more accurate ({R.loc['Random Forest', 'MAE (lakhs)']:.2f} vs "
             f"{R.loc['XGBoost', 'MAE (lakhs)']:.2f} lakhs), but XGBoost trains about "
             f"{R.loc['Random Forest', 'Train time (s)'] / R.loc['XGBoost', 'Train time (s)']:.0f}x faster. "
             "RF averages independent trees (less variance); XGBoost builds trees that fix earlier mistakes (less bias).")

    l, r = st.columns(2)
    with l:
        band_models = ["Decision Tree (baseline)", "Random Forest", "XGBoost", "Stacking"]
        b = preds.assign(Band=pd.cut(preds["actual"], [0, 25, 50, 100, 1000],
                                     labels=["< 25", "25-50", "50-100", "> 100"]))
        bm = pd.DataFrame({m: (b[m] - b["actual"]).abs() for m in band_models}).assign(Band=b["Band"]) \
            .groupby("Band", observed=True).mean().reset_index().melt(id_vars="Band", var_name="Model", value_name="MAE")
        st.plotly_chart(px.bar(bm, x="Band", y="MAE", color="Model", barmode="group", color_discrete_map=COLORS,
                               title="Average error by price (lakhs)",
                               labels={"Band": "Car price (lakhs)", "MAE": "Error (lakhs)"}), width="stretch")
        takeaway("Most cars are within 3 to 5 lakhs. Cars over 100 lakhs are off by about 20, because luxury cars are rare.")
    with r:
        fl = fi.reset_index().rename(columns={"index": "Feature"}).melt(id_vars="Feature", var_name="Model",
                                                                         value_name="Importance")
        order = fi.sort_values("Random Forest").index.tolist()
        fig = px.bar(fl, x="Importance", y="Feature", color="Model", barmode="group", orientation="h",
                     color_discrete_map=COLORS, category_orders={"Feature": order[::-1]},
                     title="What drives the price")
        fig.update_layout(height=450)
        st.plotly_chart(fig, width="stretch")
        takeaway("Gear, model, year and brand matter most. Gear ranks high partly because manual cars are mostly old.")

    st.divider()
    st.subheader("📚 Research: Random Forest vs gradient boosting")
    rr = pd.read_csv(ART / "research_results.csv")
    l, r = st.columns([3, 2])
    with l:
        st.markdown("""
| | Random Forest | XGBoost | LightGBM | CatBoost |
|---|---|---|---|---|
| How it builds trees | In parallel, then averages | One after another | One after another, leaf-wise | One after another, balanced trees |
| Mainly reduces | Variance | Bias | Bias | Bias |
| Tuning needed | Little | More | More | Medium |
| Speed | Medium | Fast | Fastest | Slower |
| Categorical data | Needs encoding | Needs encoding | Some native support | Best native support |
""")
    with r:
        st.markdown("**Tested on the same test set**")
        st.dataframe(rr.drop(columns="Family").style.format(
            {"R2": "{:.3f}", "MAE (lakhs)": "{:.2f}", "RMSE (lakhs)": "{:.2f}", "MAPE (%)": "{:.1f}",
             "Train time (s)": "{:.1f}"}), width="stretch", hide_index=True)
    takeaway("RF had the lowest average error. The boosting models trained much faster, and CatBoost and LightGBM "
             "had a higher R². With hyperparameter tuning, boosting models usually improve further.")

    l, r = st.columns(2)
    with l:
        st.markdown("""
**✅ Tabular data (like this project)**
- Tree ensembles are usually the best choice.
- They handle mixed columns (numbers, categories, yes/no) with little preprocessing.
- They capture sharp rules, like "cars made after 2015 cost more".
- Here, the neural network alone had the highest error of all models.
""")
    with r:
        st.markdown("""
**❌ Image and text data**
- Trees treat each pixel or word as a separate column.
- They can't see shapes in images or word order and meaning in text.
- CNNs (images) and transformers (text) work much better.
- Trees can still be used on features extracted by those networks.
""")
    st.caption("All results come from one 80/20 split, so very small gaps (like RF 5.39 vs stacking 5.44) could change "
               "on a different split. The large gaps (ensembles vs single tree, trees vs neural network) are reliable.")

# ---------------- Stacking & hybrid ----------------
with tab3:
    l, r = st.columns(2)
    with l:
        st.subheader("Stacking")
        st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica"];
 F [label="Car features", fillcolor="#f1f3f5"];
 RF [label="Random Forest", fillcolor="#c9f0e9"]; X [label="XGBoost", fillcolor="#fde2d8"];
 M [label="Ridge", fillcolor="#fff3bf"]; P [label="Price", fillcolor="#ffe8cc"];
 F -> RF; F -> X; RF -> M; X -> M; M -> P }""")
        st.markdown("RF and XGBoost each predict the price (using 5-fold cross-validation, so no answers leak). "
                    "A Ridge model then learns the best way to combine the two.")
        wdf = pd.DataFrame({"Model": list(weights), "Weight": list(weights.values())})
        st.plotly_chart(px.bar(wdf, x="Model", y="Weight", color="Model", color_discrete_map=COLORS, text_auto=".2f",
                               title="Weight given to each model").update_layout(showlegend=False, height=300),
                        width="stretch")
        takeaway(f"An almost 50/50 blend. Stacking gets the best R² ({R.loc['Stacking', 'R2']:.3f}) with error close "
                 f"to RF ({R.loc['Stacking', 'MAE (lakhs)']:.2f} vs {R.loc['Random Forest', 'MAE (lakhs)']:.2f}).")
    with r:
        st.subheader("Hybrid: RF + ANN")
        st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica"];
 F [label="Car features", fillcolor="#f1f3f5"]; RF [label="Random Forest", fillcolor="#c9f0e9"];
 A [label="Neural network", fillcolor="#e2d4f5"]; P [label="Price", fillcolor="#ffe8cc"];
 F -> RF -> A; F -> A; A -> P }""")
        st.markdown("The RF's prediction is given to a neural network as an extra input, "
                    "next to the normal car features.")
        hy = results[results["Model"].isin(["ANN (alone)", "Hybrid RF + ANN", "Random Forest"])]
        st.plotly_chart(px.bar(hy, x="Model", y="MAE (lakhs)", color="Model", color_discrete_map=COLORS,
                               text_auto=".2f", title="Average error (lakhs)").update_layout(showlegend=False, height=300),
                        width="stretch")
        gain = (1 - R.loc["Hybrid RF + ANN", "MAE (lakhs)"] / R.loc["ANN (alone)", "MAE (lakhs)"]) * 100
        takeaway(f"The hybrid cuts the neural network's error by {gain:.0f}%, but doesn't beat plain RF.")
    st.caption("RF + ARIMA was not possible: the data covers only Dec 2024 to Feb 2025, so there is no time series.")

# ---------------- Predict ----------------
with tab4:
    models = load_models()
    brands = sorted(meta["brand_models"])
    lc = meta["low_card"]
    pick = lambda col, label, opts, default: col.selectbox(label, opts, index=opts.index(default) if default in opts else 0)

    c1, c2, c3, c4 = st.columns(4)
    brand = pick(c1, "Brand", brands, "TOYOTA")
    opts = meta["brand_models"][brand]
    common = data.loc[data["Brand"] == brand, "Model"].mode()
    model_name = pick(c2, "Model", opts, common.iloc[0] if len(common) else opts[0])
    similar = data[(data["Brand"] == brand) & (data["Model"] == model_name)]
    lo, hi = meta["yom_range"]
    yom = c3.slider("Year", lo, hi, int(similar["YOM"].median()) if len(similar) else 2015)
    engine = c4.number_input("Engine (cc)", 600, 7000, int(similar["Engine (cc)"].median()) if len(similar) else 1500, step=50)

    c1, c2, c3, c4 = st.columns(4)
    gear = pick(c1, "Gear", lc["Gear"], "Automatic")
    fuel = pick(c2, "Fuel type", lc["Fuel Type"], "Petrol")
    town = pick(c3, "Town", meta["towns"], "Colombo")
    cond = pick(c4, "Condition", lc["Condition"], "USED")
    with st.expander("More options"):
        c1, c2, c3, c4, c5 = st.columns(5)
        leasing = pick(c1, "Leasing", lc["Leasing"], "No Leasing")
        extras = {k: pick(col, k.title(), lc[k], "Available") for col, k in
                  zip([c2, c3, c4, c5], ["AIR CONDITION", "POWER STEERING", "POWER MIRROR", "POWER WINDOW"])}

    row = pd.DataFrame([{"Brand": brand, "Model": model_name, "Town": town, "Gear": gear, "Fuel Type": fuel,
                         "Leasing": leasing, "Condition": cond, **extras, "YOM": yom, "Age": 2025 - yom,
                         "Engine (cc)": engine}])[meta["features"]]
    out = pd.DataFrame({"Model": list(models), "Price": [predict(m, n, row) for n, m in models.items()]})

    l, r = st.columns([1, 2])
    with l:
        best = out.set_index("Model").loc["Stacking", "Price"]
        st.metric("Estimated price (stacking)", f"{best:.1f} lakhs", f"≈ Rs {best * 1e5:,.0f}", delta_color="off")
        if len(similar):
            st.caption(f"Real ads for {brand} {model_name}: {len(similar)}, median {similar['Price'].median():.1f} lakhs.")
        else:
            st.caption("No ads for this exact model in the data, so the estimate is less reliable.")
    with r:
        st.plotly_chart(px.bar(out, x="Price", y="Model", color="Model", color_discrete_map=COLORS, orientation="h",
                               text_auto=".1f", labels={"Price": "Predicted price (lakhs)"})
                        .update_layout(showlegend=False, height=280), width="stretch")

st.divider()
st.caption("AI assistance: Claude (Anthropic) helped generate the code and explanations. Dataset choice, training, deployment and review of results by Harshana A.M.K.")
