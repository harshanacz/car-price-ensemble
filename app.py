import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="Ensemble Learning Dashboard", page_icon="🚗", layout="wide")

MODEL_FILES = {
    "Random Forest": "random_forest.pkl",
    "Bagging": "bagging.pkl",
    "XGBoost": "xgboost.pkl",
    "LightGBM": "lightgbm.pkl",
    "CatBoost": "catboost.pkl",
    "AdaBoost": "adaboost.pkl",
    "Stacking": "stacking.pkl",
}
FAMILY_COLORS = {"Bagging": "#2a9d8f", "Boosting": "#e76f51", "Stacking": "#e9a400", "Hybrid": "#7b2cbf",
                 "Neural network": "#8d99ae", "Single model": "#adb5bd"}


@st.cache_data
def load_tables():
    results = pd.read_csv(ART / "results.csv")
    fi = pd.read_csv(ART / "feature_importance.csv", index_col=0)
    preds = pd.read_csv(ART / "test_predictions.csv")
    data = pd.read_csv(ART / "clean_data.csv")
    meta = json.load(open(ART / "meta.json"))
    return results, fi, preds, data, meta


@st.cache_resource
def load_models():
    models = {name: joblib.load(ART / f) for name, f in MODEL_FILES.items()}
    models["Hybrid RF + ANN"] = joblib.load(ART / "hybrid_rf_ann.pkl")
    return models


def predict(models, name, X):
    m = models[name]
    if name == "Hybrid RF + ANN":
        Xp = m["prep"].transform(X)
        return np.expm1(m["ann"].predict(np.column_stack([Xp, m["rf"].predict(Xp)])))
    return np.expm1(m.predict(X))


results, fi, preds, data, meta = load_tables()

st.title("🚗 Ensemble Learning Dashboard")
st.caption("Predicting Sri Lankan used car prices (ikman.lk & riyasewana.lk data) with bagging, boosting, stacking and a hybrid RF + ANN model. Prices are in lakhs (1 lakh = Rs 100,000).")

best_mae = results.loc[results["MAE (lakhs)"].idxmin()]
best_r2 = results.loc[results["R2"].idxmax()]
fastest = results[(results["Train time (s)"] > 0) & (results["Family"].isin(["Bagging", "Boosting"]))].sort_values("Train time (s)").iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cars in dataset", f"{len(data):,}")
c2.metric("Lowest error", best_mae["Model"], f"{best_mae['MAE (lakhs)']:.2f} lakhs MAE", delta_color="off")
c3.metric("Best R²", best_r2["Model"], f"{best_r2['R2']:.3f}", delta_color="off")
c4.metric("Fastest ensemble to train", fastest["Model"], f"{fastest['Train time (s)']:.1f} s", delta_color="off")

tabs = st.tabs(["📊 Dataset", "🏆 Model comparison", "⚔️ RF vs XGBoost", "🔍 Feature importance",
                "🧮 Price predictor", "🧬 Stacking & hybrid", "📚 Research"])

# ---------------- Dataset ----------------
with tabs[0]:
    st.subheader("Dataset overview")
    a, b, c = st.columns(3)
    a.metric("Brands", data["Brand"].nunique())
    b.metric("Models", data["Model"].nunique())
    c.metric("Median price", f"{data['Price'].median():.1f} lakhs")

    l, r = st.columns(2)
    with l:
        log = st.toggle("Log scale", value=False)
        fig = px.histogram(data, x=np.log1p(data["Price"]) if log else "Price", nbins=60,
                           labels={"x": "log(1 + Price)", "Price": "Price (lakhs)"},
                           title="Price distribution" + (" (log)" if log else ", right-skewed"))
        st.plotly_chart(fig, width="stretch")
    with r:
        top = data.groupby("Brand")["Price"].agg(["median", "count"]).query("count >= 20") \
            .sort_values("median", ascending=False).head(15).reset_index()
        fig = px.bar(top, x="median", y="Brand", orientation="h", title="Median price by brand (brands with 20+ ads)",
                     labels={"median": "Median price (lakhs)"})
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

    l, r = st.columns(2)
    with l:
        yr = data.groupby("YOM")["Price"].median().reset_index()
        st.plotly_chart(px.line(yr, x="YOM", y="Price", markers=True, title="Median price by year of manufacture",
                                labels={"Price": "Median price (lakhs)"}), width="stretch")
    with r:
        st.markdown("#### ⚠️ Data quality finding")
        st.info("The original **Mileage** column had exactly as many unique values as Year of Manufacture "
                "and a correlation of **-1.0** with it, about 11,000 km per year of age. "
                "It was generated from the year, not real odometer readings, so it adds no information "
                "and was **dropped**. `Date` (all ads from Dec 2024 to Feb 2025) was also dropped.")
        st.markdown("**Features used:** " + ", ".join(f"`{f}`" for f in meta["features"]))

    st.markdown("#### Relationships with price")
    l, r = st.columns(2)
    with l:
        fig = px.scatter(data, x="Engine (cc)", y="Price", color="Fuel Type", opacity=0.5, log_y=True,
                         color_discrete_map={"Petrol": "#4361ee", "Diesel": "#f4a261", "Hybrid": "#2a9d8f",
                                             "Electric": "#e63946"},
                         title="Engine capacity vs price", labels={"Price": "Price (lakhs, log scale)"})
        st.plotly_chart(fig, width="stretch")
        corr = data["Engine (cc)"].corr(np.log1p(data["Price"]))
        st.caption(f"Overall the link looks weak (correlation with log price: {corr:.2f}) because most big-engine cars "
                   "are old diesels (median year 1996). Comparing cars of the same year, bigger engines do cost more "
                   "(partial correlation about 0.53). Hybrids sit highest: they are newer (median 2015).")
    with r:
        towns = data["Town"].value_counts().head(12).index
        tdf = data[data["Town"].isin(towns)]
        order = tdf.groupby("Town")["Price"].median().sort_values(ascending=False).index.tolist()
        fig = px.box(tdf, x="Town", y="Price", log_y=True, points=False, category_orders={"Town": order},
                     title="Price by town (12 towns with the most ads)", labels={"Price": "Price (lakhs, log scale)"})
        fig.update_xaxes(tickangle=-40)
        st.plotly_chart(fig, width="stretch")
        st.caption("Dehiwala-Mount Lavinia, Colombo and Negombo have the highest median prices; Kurunegala and Gampaha "
                   "the lowest. The ranges overlap a lot, so location matters much less than brand, year and engine size.")

    with st.expander("Browse the cleaned data"):
        st.dataframe(data, width="stretch", height=300)

# ---------------- Model comparison ----------------
with tabs[1]:
    st.subheader("All models on the same test set (20% of data)")
    fams = st.multiselect("Show families", list(FAMILY_COLORS), default=list(FAMILY_COLORS))
    res = results[results["Family"].isin(fams)].sort_values("MAE (lakhs)")
    metric = st.radio("Metric", ["MAE (lakhs)", "RMSE (lakhs)", "R2", "MAPE (%)", "Train time (s)"], horizontal=True)
    asc = metric != "R2"
    fig = px.bar(res.sort_values(metric, ascending=not asc), x=metric, y="Model", color="Family",
                 color_discrete_map=FAMILY_COLORS, orientation="h", text_auto=".3f" if metric == "R2" else ".2f",
                 title=f"{metric} ({'lower' if asc else 'higher'} is better)")
    if metric == "R2":
        fig.update_xaxes(range=[0.7, 1])
    st.plotly_chart(fig, width="stretch")
    disp = res.copy()
    disp["Train time (s)"] = disp["Train time (s)"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1f}")
    st.dataframe(disp.style.format({"R2": "{:.3f}", "MAE (lakhs)": "{:.2f}", "RMSE (lakhs)": "{:.2f}",
                                    "MAPE (%)": "{:.1f}"})
                 .highlight_min(subset=["MAE (lakhs)", "RMSE (lakhs)", "MAPE (%)"], color="#b7e4c7")
                 .highlight_max(subset=["R2"], color="#b7e4c7"),
                 width="stretch", hide_index=True)

    fam = results[results["Family"].isin(["Bagging", "Boosting"])].groupby("Family")[["MAE (lakhs)", "R2"]].mean()
    st.markdown(f"**Bagging vs Boosting (average):** bagging MAE {fam.loc['Bagging', 'MAE (lakhs)']:.2f} vs "
                f"boosting MAE {fam.loc['Boosting', 'MAE (lakhs)']:.2f} lakhs. Bagging, gradient boosting and stacking all beat the single "
                f"decision tree; AdaBoost is the exception, because re-weighting makes it chase noisy, extreme ads.")
    st.caption("ANN training time was not recorded separately (it is included in the hybrid's time). "
               "Stacking was trained on a different machine than the other models, so its time is approximate.")

    st.markdown("#### Where do the errors come from? Average error by price band")
    band_models = ["Decision Tree (baseline)", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "Stacking"]
    bands = preds.assign(Band=pd.cut(preds["actual"], [0, 25, 50, 100, 1000],
                                     labels=["< 25 lakhs", "25-50", "50-100", "> 100 lakhs"]))
    bmae = pd.DataFrame({m: (bands[m] - bands["actual"]).abs() for m in band_models})
    bmae["Band"] = bands["Band"]
    bmae = bmae.groupby("Band", observed=True).mean().reset_index().melt(id_vars="Band", var_name="Model", value_name="MAE")
    st.plotly_chart(px.bar(bmae, x="Band", y="MAE", color="Model", barmode="group", text_auto=".1f",
                           color_discrete_map={"Decision Tree (baseline)": "#adb5bd", "Random Forest": "#2a9d8f",
                                               "XGBoost": "#e76f51", "LightGBM": "#f4a582", "CatBoost": "#b23a48",
                                               "Stacking": "#e9a400"},
                           labels={"MAE": "Average error (lakhs)"}), width="stretch")
    worst = {m: (preds[m] - preds["actual"]).abs().max() for m in ["Random Forest", "CatBoost", "Stacking"]}
    st.info(f"Errors are 3 to 5 lakhs for cars under 100 lakhs, but about 20 lakhs above that, because luxury cars are rare "
            f"in the data. **Among single models, Random Forest has the lowest (or tied lowest) error in every band, even for "
            f"expensive cars** (19.8 vs 21.4 lakhs for CatBoost). "
            f"CatBoost's higher R² comes from a handful of extreme cars: its single worst miss is "
            f"{worst['CatBoost']:.0f} lakhs vs {worst['Random Forest']:.0f} for RF, and R² squares errors, so those few cars "
            f"dominate it. Stacking blends both: it is lowest of all in the cheapest and most expensive bands.")
    st.caption("All results come from one 80/20 split. Gaps of a few tenths of a lakh in MAE or 0.01 in R² are small "
               "and could change order on a different split; cross-validation would be needed to call them significant.")

# ---------------- RF vs XGBoost ----------------
with tabs[2]:
    st.subheader("Random Forest vs XGBoost")
    rx = results.set_index("Model").loc[["Random Forest", "XGBoost"]]
    cols = st.columns(4)
    for col, m in zip(cols, ["MAE (lakhs)", "RMSE (lakhs)", "R2", "Train time (s)"]):
        rf_v, xg_v = rx.loc["Random Forest", m], rx.loc["XGBoost", m]
        col.metric(m, f"RF {rf_v:.3f}" if m == "R2" else f"RF {rf_v:.2f}",
                   f"XGB {xg_v:.3f}" if m == "R2" else f"XGB {xg_v:.2f}", delta_color="off")

    l, r = st.columns(2)
    mx = preds["actual"].max()
    for col, name, colr in [(l, "Random Forest", "#2a9d8f"), (r, "XGBoost", "#e76f51")]:
        fig = px.scatter(preds, x="actual", y=name, opacity=0.4, log_x=True, log_y=True,
                         color_discrete_sequence=[colr], title=f"{name}: actual vs predicted",
                         labels={"actual": "Actual (lakhs)", name: "Predicted (lakhs)"})
        fig.add_trace(go.Scatter(x=[2, mx], y=[2, mx], mode="lines", line=dict(dash="dash", color="black"),
                                 name="perfect"))
        col.plotly_chart(fig, width="stretch")

    err = pd.DataFrame({"Random Forest": (preds["Random Forest"] - preds["actual"]).abs(),
                        "XGBoost": (preds["XGBoost"] - preds["actual"]).abs(), "actual": preds["actual"]})
    err["Price band"] = pd.cut(err["actual"], [0, 25, 50, 100, 1000],
                               labels=["< 25 lakhs", "25-50", "50-100", "> 100 lakhs"])
    band = err.groupby("Price band", observed=True)[["Random Forest", "XGBoost"]].mean().reset_index() \
        .melt(id_vars="Price band", var_name="Model", value_name="MAE")
    st.plotly_chart(px.bar(band, x="Price band", y="MAE", color="Model", barmode="group",
                           color_discrete_map={"Random Forest": "#2a9d8f", "XGBoost": "#e76f51"},
                           title="Average error by price band (lakhs)"), width="stretch")
    st.markdown("""
- **Random Forest** builds trees independently on bootstrap samples and averages them, so it mainly reduces **variance**.
- **XGBoost** builds trees one after another, each fixing the previous errors, so it mainly reduces **bias**.
- Both errors grow for expensive cars: there are few luxury cars in the data, so all models are less sure there.
""")

# ---------------- Feature importance ----------------
with tabs[3]:
    st.subheader("Which features drive the price?")
    fi_long = fi.reset_index().rename(columns={"index": "Feature"}).melt(id_vars="Feature", var_name="Model",
                                                                          value_name="Importance")
    order = fi.sort_values("Random Forest").index.tolist()
    fig = px.bar(fi_long, x="Importance", y="Feature", color="Model", barmode="group", orientation="h",
                 category_orders={"Feature": order[::-1]},
                 color_discrete_map={"Random Forest": "#2a9d8f", "XGBoost": "#e76f51"},
                 title="Normalised feature importance")
    fig.update_layout(height=550)
    st.plotly_chart(fig, width="stretch")
    st.caption("RF importance = average impurity decrease across trees. XGBoost importance = gain. "
               "The two models can rank features differently because boosting focuses on correcting errors.")

# ---------------- Predictor ----------------
with tabs[4]:
    st.subheader("Estimate a car's price with every model")
    models = load_models()
    brands = sorted(meta["brand_models"])
    c1, c2, c3 = st.columns(3)
    brand = c1.selectbox("Brand", brands, index=brands.index("TOYOTA") if "TOYOTA" in brands else 0)
    model_opts = meta["brand_models"][brand]
    common = data.loc[data["Brand"] == brand, "Model"].mode()
    model_name = c2.selectbox("Model", model_opts,
                              index=model_opts.index(common.iloc[0]) if len(common) and common.iloc[0] in model_opts else 0)
    subset = data[(data["Brand"] == brand) & (data["Model"] == model_name)]
    lo, hi = meta["yom_range"]
    yom = c3.slider("Year of manufacture", lo, hi, int(subset["YOM"].median()) if len(subset) else 2015)

    c1, c2, c3, c4 = st.columns(4)
    engine = c1.number_input("Engine (cc)", 600, 7000,
                             int(subset["Engine (cc)"].median()) if len(subset) else 1500, step=50)
    lc = meta["low_card"]
    pick = lambda col, label, opts, default: col.selectbox(label, opts, index=opts.index(default) if default in opts else 0)
    gear = pick(c2, "Gear", lc["Gear"], "Automatic")
    fuel = pick(c3, "Fuel type", lc["Fuel Type"], "Petrol")
    town = pick(c4, "Town", meta["towns"], "Colombo")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    leasing = pick(c1, "Leasing", lc["Leasing"], "No Leasing")
    cond = pick(c2, "Condition", lc["Condition"], "USED")
    extras = {}
    for col, k in zip([c3, c4, c5, c6], ["AIR CONDITION", "POWER STEERING", "POWER MIRROR", "POWER WINDOW"]):
        extras[k] = pick(col, k.title(), lc[k], "Available")

    row = pd.DataFrame([{"Brand": brand, "Model": model_name, "Town": town, "Gear": gear, "Fuel Type": fuel,
                         "Leasing": leasing, "Condition": cond, **extras, "YOM": yom, "Age": 2025 - yom,
                         "Engine (cc)": engine}])[meta["features"]]

    out = pd.DataFrame({"Model": list(models), "Predicted price (lakhs)": [predict(models, n, row)[0] for n in models]})
    out["Family"] = out["Model"].map(results.set_index("Model")["Family"])
    avg = out["Predicted price (lakhs)"].mean()

    l, r = st.columns([1, 2])
    with l:
        st.metric("Average of all models", f"{avg:.1f} lakhs", f"≈ Rs {avg * 100000:,.0f}", delta_color="off")
        if len(subset):
            st.caption(f"{len(subset)} similar ads in the data ({brand} {model_name}), median "
                       f"{subset['Price'].median():.1f} lakhs.")
        else:
            st.caption("No ads for this exact model in the data, so predictions are less reliable.")
    with r:
        fig = px.bar(out.sort_values("Predicted price (lakhs)"), x="Predicted price (lakhs)", y="Model",
                     color="Family", color_discrete_map=FAMILY_COLORS, orientation="h", text_auto=".1f")
        fig.add_vline(x=avg, line_dash="dash", annotation_text="average")
        st.plotly_chart(fig, width="stretch")

# ---------------- Hybrid ----------------
with tabs[5]:
    st.subheader("Stacking: combining RF, XGBoost, LightGBM and CatBoost")
    sw = json.load(open(ART / "stacking_weights.json"))
    l, r = st.columns([3, 2])
    with l:
        st.markdown("""
**How it works**
1. Four **base models** (Random Forest, XGBoost, LightGBM, CatBoost) each predict the price.
   Their training-set predictions come from **5-fold cross-validation** (out-of-fold), so the next step
   never sees predictions on cars a model was trained on.
2. A **meta-model** (Ridge regression) learns the best weighted combination of the four predictions.
3. Bagging (RF) is strong on typical cars and boosting limits extreme misses, so blending them uses both strengths.
""")
    with r:
        st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f1f3f5", fontname="Helvetica"];
 F [label="Car features"]; P [label="Price", fillcolor="#ffe8cc"];
 M [label="Ridge\n(meta-model)", fillcolor="#fff3bf"];
 RF [label="Random Forest", fillcolor="#c9f0e9"]; X [label="XGBoost", fillcolor="#fde2d8"];
 L [label="LightGBM", fillcolor="#fde2d8"]; C [label="CatBoost", fillcolor="#fde2d8"];
 F -> RF; F -> X; F -> L; F -> C; RF -> M; X -> M; L -> M; C -> M; M -> P }""")
    l, r = st.columns(2)
    with l:
        wdf = pd.DataFrame({"Base model": list(sw), "Weight": list(sw.values())})
        st.plotly_chart(px.bar(wdf, x="Weight", y="Base model", orientation="h", text_auto=".2f",
                               title="How much the meta-model trusts each base model"), width="stretch")
    with r:
        sm = results[results["Model"].isin(["Random Forest", "CatBoost", "Stacking"])].set_index("Model")
        st.dataframe(sm[["MAE (lakhs)", "R2", "RMSE (lakhs)"]].style.format("{:.3f}"), width="stretch")
        st.success(f"Stacking matches CatBoost's R² ({sm.loc['Stacking', 'R2']:.3f}) while keeping an error close to "
                   f"Random Forest ({sm.loc['Stacking', 'MAE (lakhs)']:.2f} vs {sm.loc['Random Forest', 'MAE (lakhs)']:.2f} lakhs). "
                   "The meta-model gives Random Forest the largest weight.")

    st.divider()
    st.subheader("Hybrid model: Random Forest + ANN (a stacking variant)")
    l, r = st.columns([3, 2])
    with l:
        st.markdown("""
**How it works**
1. A **Random Forest** is trained and predicts the price for every training car using 5-fold
   *out-of-fold* predictions, so the next model never sees leaked answers.
2. An **ANN** (MLP neural network, layers 64 → 32) gets the **normal features + the RF prediction** as input.
3. RF captures sharp rules (brand, year thresholds). The ANN learns smooth corrections on top.

**Why try it?** A neural network alone is weak on small tabular data. Giving it the forest's
"opinion" as an input lets it start from a strong guess.
""")
    with r:
        st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f1f3f5", fontname="Helvetica"];
 F [label="Car features"]; RF [label="Random Forest", fillcolor="#c9f0e9"];
 A [label="ANN (MLP)", fillcolor="#e2d4f5"]; P [label="Price", fillcolor="#ffe8cc"];
 F -> RF -> A [label=" RF prediction"]; F -> A [label=" features"]; A -> P }""")

    hy = results[results["Model"].isin(["ANN (alone)", "Random Forest", "Hybrid RF + ANN"])]
    st.plotly_chart(px.bar(hy, x="Model", y="MAE (lakhs)", color="Family", color_discrete_map=FAMILY_COLORS,
                           text_auto=".2f", title="Error: ANN alone vs RF vs Hybrid"), width="stretch")
    h = results.set_index("Model")
    gain = (1 - h.loc["Hybrid RF + ANN", "MAE (lakhs)"] / h.loc["ANN (alone)", "MAE (lakhs)"]) * 100
    st.success(f"The hybrid cuts the ANN's error by **{gain:.0f}%**. "
               f"Compared with plain Random Forest the difference is small "
               f"({h.loc['Hybrid RF + ANN', 'MAE (lakhs)']:.2f} vs {h.loc['Random Forest', 'MAE (lakhs)']:.2f} lakhs), "
               "so on this tabular data the extra complexity gives little gain over RF itself.")
    st.caption("RF + ARIMA (the time-series hybrid) needs data over time. This dataset is a snapshot of ads from "
               "Dec 2024 to Feb 2025, so it was not applicable here.")

# ---------------- Research ----------------
with tabs[6]:
    st.subheader("Comparing ensemble methods")
    st.markdown("""
| | **Random Forest** (bagging) | **XGBoost** | **LightGBM** | **CatBoost** |
|---|---|---|---|---|
| How trees are built | Independently, in parallel, on bootstrap samples | Sequentially, level-wise, with regularisation | Sequentially, leaf-wise, histogram based | Sequentially, symmetric (balanced) trees |
| Mainly reduces | Variance | Bias | Bias | Bias |
| Tuning effort | Low | Medium to high | Medium to high | Low to medium |
| Overfitting risk | Low | Medium (needs regularisation) | Higher on small data | Low (ordered boosting) |
| Speed | Fast, parallel | Fast | Fastest | Slower to train |
| Categorical features | Need encoding | Need encoding (basic native support) | Native support | Best native support |
""")
    l, r = st.columns(2)
    with l:
        st.markdown("""
#### ✅ Tabular data (like this project)
- Tree ensembles are usually the **strongest choice**: they handle mixed numeric and categorical features,
  need little preprocessing and are robust to outliers.
- On tables, they often **beat neural networks**. This project shows it too: the ANN alone had the highest error.
""")
    with r:
        st.markdown("""
#### ❌ Image and text data
- Trees see each pixel or word as an unrelated column. They can't learn **spatial patterns** (edges, shapes)
  or **word order and meaning**.
- CNNs (images) and transformers (text) are far better there.
- Trees can still help **on top of extracted features**, e.g. CNN embeddings or TF-IDF vectors.
""")
    st.markdown("#### Key findings from this project")
    rf, cb = results.set_index("Model").loc["Random Forest"], results.set_index("Model").loc["CatBoost"]
    st_ = results.set_index("Model").loc["Stacking"]
    st.markdown(f"""
- **Random Forest** had the lowest average error ({rf['MAE (lakhs)']:.2f} lakhs), while **CatBoost** had the best R² among
  single models ({cb['R2']:.3f}). RF was better for typical cars; CatBoost's R² edge comes from smaller misses on a few extreme cars.
- **Stacking** all four combined these strengths: R² {st_['R2']:.3f} with an error of {st_['MAE (lakhs)']:.2f} lakhs.
- Bagging, gradient boosting and stacking all beat the **single decision tree**, confirming that combining models reduces error.
  AdaBoost was the exception: its re-weighting over-focuses on noisy, extreme ads.
- The **hybrid RF + ANN** greatly improved the ANN but did not beat RF, showing tree ensembles are already strong on tabular data.
""")
