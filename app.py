import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).parent / "artifacts"
st.set_page_config(page_title="Sri Lankan Used Car Price Prediction", page_icon="🚗", layout="wide")

COLORS = {"Decision Tree (baseline)": "#adb5bd", "Random Forest": "#2a9d8f", "XGBoost": "#e76f51",
          "Stacking": "#e9a400", "Hybrid RF + ANN": "#7b2cbf", "ANN (alone)": "#8d99ae",
          "LightGBM": "#f4a582", "CatBoost": "#b23a48"}
LABEL = {"Decision Tree (baseline)": "Decision Tree (Baseline)", "Random Forest": "Random Forest (Bagging)",
         "XGBoost": "XGBoost (Boosting)", "Stacking": "Stacking (RF + XGB + Ridge)",
         "Hybrid RF + ANN": "Hybrid (RF + ANN)", "ANN (alone)": "Neural Network (Reference)"}

st.markdown("""
<style>
.hero {background: linear-gradient(135deg, #1d3557 0%, #2a6f97 100%); padding: 28px 32px; border-radius: 14px;
       color: white; margin-bottom: 22px;}
.hero h1 {color: white; font-size: 2rem; margin: 0 0 8px 0;}
.hero p {color: #e3eef7; margin: 0; font-size: 1.02rem;}
.cards {display: grid; gap: 14px; margin: 8px 0 18px 0;}
.card {background: #ffffff; border-radius: 12px; padding: 14px 16px; border-top: 5px solid #2a6f97;
       box-shadow: 0 2px 8px rgba(0,0,0,0.08); color: #1d3557;}
.card.best {border-top-color: #2a9d8f; background: #eefaf6;}
.card .t {font-size: 0.75rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; color: #5c6b7a;}
.card .v {font-size: 1.7rem; font-weight: 800; margin: 4px 0;}
.card .s {font-size: 0.82rem; color: #5c6b7a;}
.summary {background: #e9f5ee; border-left: 5px solid #2a9d8f; padding: 12px 16px; border-radius: 8px;
          color: #1b4332; margin: 6px 0 16px 0;}
.interp {background: #f1f5fb; border-left: 5px solid #2a6f97; padding: 12px 16px; border-radius: 8px;
         color: #1d3557; margin: 4px 0 18px 0; font-size: 0.95rem;}
.notice {background: #fff6e5; border-left: 5px solid #e9a400; padding: 12px 16px; border-radius: 8px;
         color: #5c4400; margin: 4px 0 16px 0;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load_tables():
    return (pd.read_csv(ART / "results.csv"), pd.read_csv(ART / "feature_importance.csv", index_col=0),
            pd.read_csv(ART / "test_predictions.csv"), pd.read_csv(ART / "clean_data.csv"),
            json.load(open(ART / "meta.json")), json.load(open(ART / "stacking_weights.json")),
            pd.read_csv(ART / "research_results.csv"), pd.read_csv(ART / "cv_results.csv"))


@st.cache_resource
def load_models():
    return {"Random Forest": joblib.load(ART / "random_forest.pkl"), "XGBoost": joblib.load(ART / "xgboost.pkl"),
            "Stacking": joblib.load(ART / "stacking.pkl"), "Hybrid RF + ANN": joblib.load(ART / "hybrid_rf_ann.pkl")}


def predict(m, name, X):
    if name == "Hybrid RF + ANN":
        Xp = m["prep"].transform(X)
        return np.expm1(m["ann"].predict(np.column_stack([Xp, m["rf"].predict(Xp)])))[0]
    return np.expm1(m.predict(X))[0]


def hero():
    st.markdown('<div class="hero"><h1>🚗 Sri Lankan Used Car Price Prediction</h1>'
                '<p>Predicting used car prices from ikman.lk & riyasewana.lk ads, comparing Bagging (Random Forest), '
                'Boosting (XGBoost), Stacking (RF + XGB + Ridge) and a Hybrid (RF + Neural Network).</p></div>',
                unsafe_allow_html=True)


def cards(items, per_row=None):
    n = per_row or min(len(items), 5)
    html = "".join(f'<div class="card{" best" if best else ""}"><div class="t">{t}</div>'
                   f'<div class="v">{v}</div><div class="s">{s}</div></div>' for t, v, s, best in items)
    st.markdown(f'<div class="cards" style="grid-template-columns: repeat({n}, 1fr);">{html}</div>',
                unsafe_allow_html=True)


def box(kind, html):
    st.markdown(f'<div class="{kind}">{html}</div>', unsafe_allow_html=True)


results, fi, preds, data, meta, weights, research, cv = load_tables()
R = results.set_index("Model")
RAW_ROWS, DUPES = 9788, 252
N_TEST = len(preds)
N_TRAIN = len(data) - N_TEST

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 🚗 Navigation")
    page = st.radio("Select page", ["1. Overview", "2. Model Comparison", "3. Research Comparison",
                                    "4. Price Prediction", "5. About the Project"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown(f"""
**Course:** CM4371 – ML & Pattern Recognition

**Dataset:** Sri Lankan used car ads (ikman.lk & riyasewana.lk, Kaggle)

**Coverage:** Dec 2024 – Feb 2025

**Split:** Random 80/20 ({N_TRAIN:,} train / {N_TEST:,} test), plus 5-fold cross-validation

**Target:** Price in lakhs (1 lakh = Rs 100,000)
""")
    st.divider()
    st.caption("AI assistance: Claude (Anthropic) helped generate the code and explanations. Dataset choice, "
               "training, deployment and review of results by Harshana A.M.K.")

hero()

# ---------------- 1. Overview ----------------
if page.startswith("1"):
    cards([("Car ads used", f"{len(data):,}", f"From {RAW_ROWS:,} raw ads", False),
           ("Brands", data["Brand"].nunique(), f"{data['Model'].nunique():,} different models", False),
           ("Median price", f"{data['Price'].median():.0f} lakhs", f"Mean {data['Price'].mean():.1f} · Max {data['Price'].max():.0f}", False),
           ("Year range", f"{data['YOM'].min()}–{data['YOM'].max()}", f"Median year {int(data['YOM'].median())}", False)])

    st.subheader("Price distribution")
    fig = px.histogram(data, x="Price", nbins=80, labels={"Price": "Price (lakhs)"}, color_discrete_sequence=["#2a6f97"])
    fig.add_vline(x=data["Price"].median(), line_dash="dash", line_color="#2a9d8f",
                  annotation_text=f"Median {data['Price'].median():.0f}")
    fig.add_vline(x=data["Price"].mean(), line_dash="dot", line_color="#e76f51",
                  annotation_text=f"Mean {data['Price'].mean():.1f}", annotation_position="top left")
    st.plotly_chart(fig.update_layout(height=340, yaxis_title="Number of ads"), width="stretch")
    box("interp", "<b>Interpretation:</b> Prices are strongly right-skewed. Half of all cars cost under 43 lakhs, "
                  "but a few luxury cars reach up to 790 lakhs. To stop these few cars from dominating training, "
                  "the models were trained on <b>log(price)</b> and predictions converted back to lakhs.")

    def med_bar(col, top=None, horizontal=False):
        g = data.groupby(col)["Price"].agg(["median", "count"])
        if top:
            g = g.query("count >= 20").sort_values("count", ascending=False).head(top)
        g = g.sort_values("median", ascending=horizontal).reset_index()
        fig = px.bar(g, x="median" if horizontal else col, y=col if horizontal else "median",
                     orientation="h" if horizontal else "v", text_auto=".0f",
                     labels={"median": "Median price (lakhs)", col: ""}, color_discrete_sequence=["#2a9d8f"])
        fig.update_traces(textangle=0, textposition="outside", cliponaxis=False)
        return fig.update_layout(height=360)

    l, r = st.columns(2)
    with l:
        st.subheader("Median price by brand")
        st.plotly_chart(med_bar("Brand", top=10, horizontal=True), width="stretch")
        box("interp", "<b>Interpretation:</b> Among the 10 most advertised brands, Mercedes-Benz costs about twice "
                      "as much as Toyota, while budget brands (Micro, Hyundai, Mitsubishi) are much cheaper. "
                      "Brand and model are strong price signals.")
    with r:
        st.subheader("Median price by car age")
        age = data.assign(Age=2025 - data["YOM"]).query("Age <= 40").groupby("Age")["Price"].median().reset_index()
        st.plotly_chart(px.line(age, x="Age", y="Price", markers=True, color_discrete_sequence=["#2a6f97"],
                                labels={"Age": "Age (years)", "Price": "Median price (lakhs)"}).update_layout(height=360),
                        width="stretch")
        box("interp", "<b>Interpretation:</b> Prices fall as cars get older. Very new cars vary a lot because few "
                      "are listed, but from about 15 years onward the drop is steady.")

    l, r = st.columns(2)
    with l:
        st.subheader("Median price by fuel type")
        st.plotly_chart(med_bar("Fuel Type"), width="stretch")
    with r:
        st.subheader("Median price by gear")
        st.plotly_chart(med_bar("Gear"), width="stretch")
    box("interp", "<b>Interpretation:</b> Hybrid and electric cars cost the most, and automatics cost far more than "
                  "manuals. Part of this is age: hybrids and automatics are mostly newer cars, while many diesel and "
                  "manual cars are old.")

    st.subheader("Data cleaning")
    box("summary", f"<b>Mileage was dropped:</b> it had exactly the same 62 unique values as year of manufacture and "
                   f"a correlation of <b>-1.0</b> with it (about 11,000 km per year of age), so it was calculated from "
                   f"the year, not real readings. <b>{DUPES} duplicate ads</b> (the same car posted twice) and the "
                   f"posting date were also removed, leaving <b>{len(data):,}</b> cars.")

# ---------------- 2. Model Comparison ----------------
elif page.startswith("2"):
    st.subheader("Test-set performance comparison")
    box("summary", f"<b>Lowest error (MAE):</b> Random Forest ({R.loc['Random Forest', 'MAE (lakhs)']:.2f} lakhs) | "
                   f"<b>Best RMSE & R²:</b> Stacking (RMSE {R.loc['Stacking', 'RMSE (lakhs)']:.2f} lakhs, "
                   f"R² {R.loc['Stacking', 'R2']:.3f}) | <b>Fastest ensemble:</b> XGBoost "
                   f"({R.loc['XGBoost', 'Train time (s)']:.1f} s)")

    order = ["Random Forest", "XGBoost", "Stacking", "Hybrid RF + ANN", "Decision Tree (baseline)", "ANN (alone)"]
    tags = {"Random Forest": " 🏆 Best MAE", "Stacking": " 🏆 Best R²"}
    cards([(LABEL[m] + tags.get(m, ""), f"{R.loc[m, 'MAE (lakhs)']:.2f}",
            f"MAE in lakhs (lower is better)<br>RMSE: {R.loc[m, 'RMSE (lakhs)']:.2f} · R²: {R.loc[m, 'R2']:.3f}",
            m in tags) for m in order], per_row=3)

    c1, c2, c3 = st.columns(3)
    for col, metric, title in [(c1, "MAE (lakhs)", "MAE (lakhs), lower is better"),
                               (c2, "RMSE (lakhs)", "RMSE (lakhs), lower is better"),
                               (c3, "R2", "R², higher is better")]:
        d = results.set_index("Model").loc[order].reset_index()
        fig = px.bar(d, x="Model", y=metric, color="Model", color_discrete_map=COLORS,
                     text_auto=".3f" if metric == "R2" else ".2f", title=title)
        fig.update_layout(showlegend=False, height=360, xaxis_title="", xaxis_tickangle=-35)
        if metric == "R2":
            fig.update_yaxes(range=[0.8, 0.92])
        col.plotly_chart(fig, width="stretch")

    st.subheader("Detailed analysis")
    t0, t1, t2, t3, t5, t4 = st.tabs(["Stacking & hybrid", "Actual vs predicted", "Error by price range",
                                      "Feature importance", "Stability check (5-fold CV)", "Full results table"])
    with t5:
        cvd = cv.copy()
        cvd["Model"] = cvd["Model"].map(LABEL)
        cvd["MAE (lakhs)"] = cvd.apply(lambda r: f"{r['MAE mean']:.2f} ± {r['MAE std']:.2f}", axis=1)
        cvd["R²"] = cvd.apply(lambda r: f"{r['R2 mean']:.3f} ± {r['R2 std']:.3f}", axis=1)
        st.dataframe(cvd[["Model", "MAE (lakhs)", "R²", "MAE folds"]].rename(columns={"MAE folds": "MAE in each fold"}),
                     width="stretch", hide_index=True)
        fig = go.Figure()
        for _, rw in cv.iterrows():
            fig.add_trace(go.Bar(x=[LABEL[rw["Model"]]], y=[rw["MAE mean"]], error_y=dict(type="data", array=[rw["MAE std"]]),
                                 marker_color=COLORS[rw["Model"]], text=f"{rw['MAE mean']:.2f}", textposition="outside",
                                 showlegend=False))
        st.plotly_chart(fig.update_layout(height=380, yaxis_title="MAE (lakhs), mean ± std over 5 folds"), width="stretch")
        C = cv.set_index("Model")
        box("interp", f"<b>Interpretation:</b> the whole dataset was split into 5 parts, and each model was trained 5 times, "
                      f"each time tested on a different part. The ranking stays the same as the single split: "
                      f"<b>Random Forest</b> has the lowest error ({C.loc['Random Forest', 'MAE mean']:.2f} ± {C.loc['Random Forest', 'MAE std']:.2f}) "
                      f"and <b>Stacking</b> the best R² ({C.loc['Stacking', 'R2 mean']:.3f}). RF and stacking are within each other's "
                      f"spread, so they are effectively tied, but RF beats XGBoost "
                      f"({C.loc['XGBoost', 'MAE mean']:.2f}) and every ensemble clearly beats the single tree "
                      f"({C.loc['Decision Tree (baseline)', 'MAE mean']:.2f}). R² varies more between folds because it depends on "
                      f"which expensive cars land in each test part.")
    with t0:
        l, r = st.columns(2)
        with l:
            st.subheader("Stacking: RF + XGBoost → Ridge")
            st.graphviz_chart("""
    digraph { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica"];
     F [label="Car features", fillcolor="#f1f3f5"];
     RF [label="Random Forest", fillcolor="#c9f0e9"]; X [label="XGBoost", fillcolor="#fde2d8"];
     M [label="Ridge\\n(meta-model)", fillcolor="#fff3bf"]; P [label="Price", fillcolor="#ffe8cc"];
     F -> RF; F -> X; RF -> M; X -> M; M -> P }""")
            st.markdown("""
    1. Random Forest and XGBoost each predict the price.
    2. Their training predictions come from **5-fold cross-validation** (out-of-fold), so the meta-model never sees predictions on cars a model was trained on.
    3. A **Ridge regression** learns the best weighted combination.
    """)
            wdf = pd.DataFrame({"Model": list(weights), "Weight": list(weights.values())})
            st.plotly_chart(px.bar(wdf, x="Model", y="Weight", color="Model", color_discrete_map=COLORS, text_auto=".2f",
                                   title="Weight given to each base model").update_layout(showlegend=False, height=300),
                            width="stretch")
            box("interp", f"<b>Interpretation:</b> Ridge gives the two models almost equal weight "
                          f"({weights['Random Forest']:.2f} RF, {weights['XGBoost']:.2f} XGBoost). The blend has the best "
                          f"R² ({R.loc['Stacking', 'R2']:.3f}) and RMSE, with error close to RF "
                          f"({R.loc['Stacking', 'MAE (lakhs)']:.2f} vs {R.loc['Random Forest', 'MAE (lakhs)']:.2f} lakhs).")
        with r:
            st.subheader("Hybrid: Random Forest + Neural Network")
            st.graphviz_chart("""
    digraph { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica"];
     F [label="Car features", fillcolor="#f1f3f5"]; RF [label="Random Forest", fillcolor="#c9f0e9"];
     A [label="Neural network\\n(64 → 32)", fillcolor="#e2d4f5"]; P [label="Price", fillcolor="#ffe8cc"];
     F -> RF -> A; F -> A; A -> P }""")
            st.markdown("""
    1. Random Forest predicts the price (again using out-of-fold predictions for training).
    2. The **neural network** gets the normal car features **plus** the RF prediction as an extra input.
    3. RF captures sharp rules; the network learns smooth corrections on top.
    """)
            hy = results.set_index("Model").loc[["ANN (alone)", "Hybrid RF + ANN", "Random Forest"]].reset_index()
            st.plotly_chart(px.bar(hy, x="Model", y="MAE (lakhs)", color="Model", color_discrete_map=COLORS,
                                   text_auto=".2f", title="Average error (lakhs)").update_layout(showlegend=False, height=300),
                            width="stretch")
            gain = (1 - R.loc["Hybrid RF + ANN", "MAE (lakhs)"] / R.loc["ANN (alone)", "MAE (lakhs)"]) * 100
            box("interp", f"<b>Interpretation:</b> The hybrid cuts the neural network's error by <b>{gain:.0f}%</b> "
                          f"({R.loc['ANN (alone)', 'MAE (lakhs)']:.2f} → {R.loc['Hybrid RF + ANN', 'MAE (lakhs)']:.2f}), "
                          f"but it does not beat plain Random Forest. RF already captures almost all the signal in this "
                          f"tabular data.")
        box("notice", "<b>RF + ARIMA</b> (the time-series hybrid) was not possible: the data is a snapshot of ads from "
                      "Dec 2024 to Feb 2025, not a time series.")
    with t1:
        pick = st.selectbox("Model", order, format_func=lambda m: LABEL[m])
        mx = preds["actual"].max()
        fig = px.scatter(preds, x="actual", y=pick, opacity=0.4, log_x=True, log_y=True,
                         color_discrete_sequence=[COLORS[pick]],
                         labels={"actual": "Actual price (lakhs)", pick: "Predicted price (lakhs)"})
        fig.add_trace(go.Scatter(x=[2, mx], y=[2, mx], mode="lines", line=dict(dash="dash", color="gray"),
                                 name="Perfect prediction"))
        st.plotly_chart(fig.update_layout(height=450), width="stretch")
        box("interp", "<b>Interpretation:</b> Points close to the dashed line are accurate predictions. "
                      "The ensembles keep most cars close to the line; the single tree and the neural network spread "
                      "more widely. All models spread more for very expensive cars.")
    with t2:
        band_models = ["Decision Tree (baseline)", "Random Forest", "XGBoost", "Stacking"]
        b = preds.assign(Band=pd.cut(preds["actual"], [0, 25, 50, 100, 1000],
                                     labels=["Under 25", "25 to 50", "50 to 100", "Over 100"]))
        bm = pd.DataFrame({m: (b[m] - b["actual"]).abs() for m in band_models}).assign(Band=b["Band"]) \
            .groupby("Band", observed=True).mean().reset_index().melt(id_vars="Band", var_name="Model", value_name="MAE")
        counts = b["Band"].value_counts().reindex(["Under 25", "25 to 50", "50 to 100", "Over 100"])
        st.plotly_chart(px.bar(bm, x="Band", y="MAE", color="Model", barmode="group", color_discrete_map=COLORS,
                               text_auto=".1f", labels={"Band": "Car price (lakhs)", "MAE": "Average error (lakhs)"})
                        .update_layout(height=420), width="stretch")
        box("interp", f"<b>Interpretation:</b> Most cars are predicted within 3 to 5 lakhs. Cars over 100 lakhs are "
                      f"off by about 20 lakhs, because only {counts['Over 100']} of the {N_TEST:,} test cars are that "
                      f"expensive and their prices vary a lot. Random Forest has the lowest error of the single models "
                      f"in most price ranges.")
    with t3:
        fl = fi.reset_index().rename(columns={"index": "Feature"}).melt(id_vars="Feature", var_name="Model",
                                                                         value_name="Importance")
        forder = fi.sort_values("Random Forest").index.tolist()
        fig = px.bar(fl, x="Importance", y="Feature", color="Model", barmode="group", orientation="h",
                     color_discrete_map=COLORS, category_orders={"Feature": forder[::-1]})
        st.plotly_chart(fig.update_layout(height=520), width="stretch")
        box("interp", "<b>Interpretation:</b> Gear, model, year/age and brand drive the price. Gear ranks very high "
                      "(especially for XGBoost) partly because manual cars are mostly old and cheap, so the first "
                      "split on gear separates old and new cars. Town, fuel type and equipment matter much less.")
    with t4:
        disp = results.copy()
        disp["Model"] = disp["Model"].map(LABEL)
        disp["Train time (s)"] = disp["Train time (s)"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1f}")
        st.dataframe(disp.style.format({"R2": "{:.3f}", "MAE (lakhs)": "{:.2f}", "RMSE (lakhs)": "{:.2f}",
                                        "MAPE (%)": "{:.1f}"}), width="stretch", hide_index=True)
        st.caption("MAPE = average error as a percentage of the car's price. ANN training time was not recorded "
                   "separately (it is included in the hybrid's time).")

    st.subheader("Model performance interpretation")
    speed = R.loc["Random Forest", "Train time (s)"] / R.loc["XGBoost", "Train time (s)"]
    st.markdown(f"""
**1. Which model performed best?**
- **Random Forest (Bagging)** had the lowest average error (MAE {R.loc['Random Forest', 'MAE (lakhs)']:.2f} lakhs, about 13% of the car's price).
- **Stacking (RF + XGB + Ridge)** had the best RMSE ({R.loc['Stacking', 'RMSE (lakhs)']:.2f}) and R² ({R.loc['Stacking', 'R2']:.3f}), so it won two of the three main metrics.
- All ensembles beat the **single decision tree** (MAE {R.loc['Decision Tree (baseline)', 'MAE (lakhs)']:.2f}), which shows the main benefit of ensemble learning.

**2. Random Forest vs XGBoost**
- RF averages many independent trees trained on random samples, which mainly **reduces variance**. XGBoost builds trees one after another, each correcting the previous errors, which mainly **reduces bias**.
- On this data RF was more accurate on all three metrics, but XGBoost trained about **{speed:.0f}x faster**. We used default-style settings; boosting models usually gain more from tuning.

**3. Why did stacking improve RMSE and R² but not MAE?**
- RF and XGBoost make partly different mistakes, so blending them cancels some errors.
- The Ridge meta-model minimises **squared** error, so it mainly shrinks the biggest misses. That improves RMSE and R² (which square errors), while the average error stays slightly above RF.

**4. Why was the neural network weakest?**
- Neural networks learn smooth patterns and need more data and tuning. Car prices follow sharp rules (brand, year thresholds) that trees capture naturally.
""")
    box("notice", "<b>Reliability:</b> the 5-fold cross-validation (see the Stability check tab) gives the same ranking "
                  "as the single 80/20 split, so the main conclusions hold. RF and stacking are effectively tied on MAE.")

# ---------------- 4. Research ----------------
elif page.startswith("3"):
    st.subheader("Random Forest vs gradient boosting (XGBoost, LightGBM, CatBoost)")
    RR = research.set_index("Model")
    cards([(m + (" 🏆 Best MAE" if m == "Random Forest" else " 🏆 Best R²" if m == "CatBoost" else ""),
            f"{RR.loc[m, 'MAE (lakhs)']:.2f}",
            f"MAE in lakhs<br>R²: {RR.loc[m, 'R2']:.3f} · Train: {RR.loc[m, 'Train time (s)']:.1f} s",
            m in ("Random Forest", "CatBoost")) for m in RR.index])

    l, r = st.columns(2)
    with l:
        st.plotly_chart(px.bar(research, x="Model", y="MAE (lakhs)", color="Model", color_discrete_map=COLORS,
                               text_auto=".2f", title="Average error (lower is better)")
                        .update_layout(showlegend=False, height=330), width="stretch")
    with r:
        st.plotly_chart(px.bar(research, x="Model", y="Train time (s)", color="Model", color_discrete_map=COLORS,
                               text_auto=".1f", title="Training time in seconds (lower is better)")
                        .update_layout(showlegend=False, height=330), width="stretch")
    box("interp", "<b>Interpretation:</b> All four were tested on the same test set. Random Forest had the lowest "
                  "average error; CatBoost and LightGBM had a slightly higher R². The three boosting models trained "
                  "3 to 8 times faster than Random Forest. With hyperparameter tuning, boosting models usually improve further.")

    st.subheader("How they differ")
    st.markdown("""
| | Random Forest | XGBoost | LightGBM | CatBoost |
|---|---|---|---|---|
| Ensemble type | Bagging | Boosting | Boosting | Boosting |
| How trees are built | In parallel, then averaged | One after another, level by level | One after another, leaf-wise | One after another, balanced (symmetric) trees |
| Mainly reduces | Variance | Bias | Bias | Bias |
| Tuning needed | Little | More | More | Medium |
| Speed | Medium | Fast | Fastest | Slower |
| Categorical data | Needs encoding | Needs encoding | Some native support | Best native support |
""")

    st.subheader("Strengths and limitations: tabular vs image/text data")
    l, r = st.columns(2)
    with l:
        box("summary", """<b>✅ Tabular data (like this project)</b><br>
• Tree ensembles are usually the best choice.<br>
• They handle mixed columns (numbers, categories, yes/no) with little preprocessing.<br>
• They capture sharp rules, like "cars made after 2015 cost more".<br>
• In this project, the neural network alone had the highest error of all models.""")
    with r:
        box("notice", """<b>❌ Image and text data</b><br>
• Trees treat each pixel or word as a separate column.<br>
• They can't see shapes in images, or word order and meaning in text.<br>
• CNNs (images) and transformers (text) work much better.<br>
• Trees can still be used on features extracted by those networks.""")

# ---------------- 5. Prediction ----------------
elif page.startswith("4"):
    st.subheader("Interactive used car price predictor")
    box("notice", "<b>Experimental estimate:</b> these predictions come from models trained on advertised prices "
                  "from Dec 2024 to Feb 2025. They are for educational demonstration only, not an official valuation.")
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
    yom = c3.slider("Year of manufacture", lo, hi, int(similar["YOM"].median()) if len(similar) else 2015)
    engine = c4.number_input("Engine (cc)", 600, 7000, int(similar["Engine (cc)"].median()) if len(similar) else 1500, step=50)

    c1, c2, c3, c4 = st.columns(4)
    gear = pick(c1, "Gear", lc["Gear"], "Automatic")
    fuel = pick(c2, "Fuel type", lc["Fuel Type"], "Petrol")
    town = pick(c3, "Town", meta["towns"], "Colombo")
    cond = pick(c4, "Condition", lc["Condition"], "USED")
    c1, c2, c3, c4, c5 = st.columns(5)
    leasing = pick(c1, "Leasing", lc["Leasing"], "No Leasing")
    extras = {k: pick(col, k.title(), lc[k], "Available") for col, k in
              zip([c2, c3, c4, c5], ["AIR CONDITION", "POWER STEERING", "POWER MIRROR", "POWER WINDOW"])}

    row = pd.DataFrame([{"Brand": brand, "Model": model_name, "Town": town, "Gear": gear, "Fuel Type": fuel,
                         "Leasing": leasing, "Condition": cond, **extras, "YOM": yom, "Age": 2025 - yom,
                         "Engine (cc)": engine}])[meta["features"]]
    out = {n: predict(m, n, row) for n, m in models.items()}
    ref = similar["Price"].median() if len(similar) else None

    st.subheader("Predicted price")
    cards([(LABEL[n], f"{p:.1f} lakhs",
            (f"Δ vs ads median: {p - ref:+.1f}" if ref is not None else "No similar ads") + f"<br>≈ Rs {p * 1e5:,.0f}",
            n == "Stacking") for n, p in out.items()]
          + [("Real ads median", f"{ref:.1f} lakhs" if ref is not None else "n/a",
              f"{len(similar)} ads for {brand} {model_name}" if ref is not None else "No ads for this model", False)])
    chart = pd.DataFrame({"Method": [LABEL[n] for n in out] + (["Real ads median"] if ref is not None else []),
                          "Price": list(out.values()) + ([ref] if ref is not None else [])})
    st.plotly_chart(px.bar(chart, x="Method", y="Price", color="Method", text_auto=".1f",
                           color_discrete_sequence=[COLORS[n] for n in out] + ["#adb5bd"],
                           title=f"Prediction comparison for {brand} {model_name} ({yom})",
                           labels={"Price": "Price (lakhs)", "Method": ""}).update_layout(showlegend=False, height=380),
                    width="stretch")
    st.caption("The real ads median covers all years of this model, so it is a rough reference only.")

# ---------------- 6. About ----------------
else:
    st.subheader("Project objective")
    st.markdown("This project was developed for **CM4371 – Machine Learning & Pattern Recognition** (Hands-on "
                "Project: Ensemble Learning Dashboard). Its goal is to predict the price of a used car in Sri Lanka "
                "from its details, and to compare **Bagging, Boosting, Stacking** and a **Hybrid** model on local data.")

    st.subheader("Dataset & data source")
    st.markdown(f"""
- **Dataset:** [SriLankan Vehicle (Car) Price Dataset](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset) (Kaggle, Prasad Nirmal)
- **Source websites:** ikman.lk and riyasewana.lk
- **Raw records:** {RAW_ROWS:,} ads, 17 columns, no missing values
- **After cleaning:** {len(data):,} ads ({DUPES} duplicates removed)
- **Period:** ads posted Dec 2024 – Feb 2025
- **Target:** price in lakhs (1 lakh = Rs 100,000)
""")

    st.subheader("Features used")
    st.markdown("""
| Feature | Description | Encoding |
|---|---|---|
| Brand, Model, Town | Manufacturer, model name, ad location (50 / 1,555 / 107 values) | Target encoding (average log price per category) |
| Gear, Fuel Type | Automatic/Manual; Petrol/Hybrid/Diesel/Electric | One-hot |
| Leasing, Condition | Ongoing lease or not; Used/New | One-hot |
| Air condition, Power steering, Power mirror, Power window | Equipment available or not | One-hot |
| YOM, Age | Year of manufacture, and age = 2025 − YOM | Numeric |
| Engine (cc) | Engine capacity | Numeric |

**Removed:** Mileage (calculated from year, correlation -1.0), posting date (only a 2-month window), row index.
""")

    st.subheader("Method")
    st.markdown(f"""
1. **Cleaning:** removed duplicates, mileage and date.
2. **Target transform:** trained on log(1 + price) because prices are skewed; predictions converted back to lakhs.
3. **Split:** random 80/20, {N_TRAIN:,} training and {N_TEST:,} test cars. All models use the same test set.
4. **Pipelines:** encoding is fitted only on the training data inside scikit-learn Pipelines, so the test set never leaks into training.
5. **Validation:** besides the 80/20 split, Decision Tree, Random Forest, XGBoost and Stacking were checked with 5-fold cross-validation on the full dataset.
6. **Models:** Decision Tree (baseline), Random Forest (bagging), XGBoost (boosting), Stacking (RF + XGBoost → Ridge, 5-fold out-of-fold), Hybrid (RF + neural network), and a neural network alone for reference. LightGBM and CatBoost were also tested for the research comparison.
7. **Metrics:** MAE (average error in lakhs), RMSE (punishes big errors more), R² (share of price variation explained), MAPE (error as a % of price).
""")

    st.subheader("Limitations")
    st.markdown("""
- Default-style hyperparameters; no tuning.
- RF and stacking are within each other's cross-validation spread, so their small difference is not significant.
- Advertised prices, not final sale prices.
- No real mileage, accident history or trim level in the data.
- Only 2 months of ads; the Sri Lankan car market changes quickly with import rules.
""")

    st.subheader("Tools & links")
    st.markdown("""
- **Training:** Google Colab, scikit-learn, XGBoost (LightGBM and CatBoost for research)
- **Dashboard:** Streamlit, Plotly
- **Code:** [github.com/harshanacz/car-price-ensemble](https://github.com/harshanacz/car-price-ensemble)
""")
    st.caption("AI assistance: Claude (Anthropic) helped generate the code and explanations. Dataset choice, "
               "training, deployment and review of results by Harshana A.M.K.")
