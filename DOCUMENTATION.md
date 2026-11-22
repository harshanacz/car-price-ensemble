# Ensemble Learning for Sri Lankan Used Car Price Prediction

**Project documentation: end-to-end technical breakdown**

| | |
|---|---|
| **Author** | Harshana A.M.K |
| **Repository** | https://github.com/harshanacz/car-price-ensemble |
| **Live dashboard** | https://car-price-ensemble.streamlit.app/ |
| **Dataset** | [SriLankan Vehicle (Car) Price Dataset](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset), scraped from ikman.lk and riyasewana.lk |
| **Training environment** | Google Colab (Python 3, scikit-learn 1.6.1) |
| **Dashboard stack** | Streamlit, Plotly, joblib |

---

## Table of contents

1. [Executive summary and problem formulation](#1-executive-summary-and-problem-formulation)
2. [Dataset details and in-depth EDA](#2-dataset-details-and-in-depth-eda)
3. [Data preprocessing and feature engineering](#3-data-preprocessing-and-feature-engineering)
4. [Model architectures and mechanics](#4-model-architectures-and-mechanics)
5. [Evaluation metrics and head-to-head comparison](#5-evaluation-metrics-and-head-to-head-comparison)
6. [Streamlit dashboard architecture](#6-streamlit-dashboard-architecture)
7. [Research analysis: tabular vs unstructured data](#7-research-analysis-tabular-vs-unstructured-data)
8. [Limitations and future work](#8-limitations-and-future-work)
9. [How to reproduce](#9-how-to-reproduce)

---

## 1. Executive summary and problem formulation

### 1.1 Problem

Buying or selling a used car in Sri Lanka involves a lot of guesswork. Prices on ikman.lk and riyasewana.lk vary widely for similar vehicles, and the market has been heavily distorted by import restrictions since 2020, which pushed used car prices up sharply. A model that estimates a fair price from a car's basic attributes (brand, model, year, engine size, fuel type, gear, location and equipment) is useful to buyers, sellers and dealers.

### 1.2 Formulation

- **Task type:** supervised **regression**
- **Target variable:** `Price`, the advertised price in **lakhs** (1 lakh = Rs 100,000). Median price is 43 lakhs (Rs 4.3 million).
- **Input:** 14 features describing the car (after cleaning)
- **Training objective:** minimise squared error on `log(1 + Price)`
- **Primary evaluation metric:** Mean Absolute Error (MAE) in lakhs, supported by R², RMSE and MAPE on the original price scale

### 1.3 Objectives (from the assignment brief)

1. Build an **ensemble learning dashboard** covering bagging, boosting and stacking.
2. **Compare Random Forest vs XGBoost** on a local (Sri Lankan) dataset.
3. **Try hybridisation**: Random Forest + ANN for tabular regression.
4. **Research**: compare Random Forest against gradient boosting methods (XGBoost, LightGBM, CatBoost), and discuss their strengths and limitations on tabular vs image/text data.

### 1.4 Headline results

| Finding | Value |
|---|---|
| Lowest average error | **Random Forest**, MAE **5.39 lakhs** |
| Highest R² | **Stacking**, R² **0.913** (CatBoost best single model, 0.912) |
| Fastest gradient boosting model | **XGBoost**, 1.47 s training |
| Stacking (RF + XGBoost + LightGBM + CatBoost → Ridge) | Matches CatBoost's R² with an error close to RF (MAE **5.45 lakhs**) |
| Ensembles vs single decision tree | Bagging, gradient boosting and stacking all beat the tree (MAE 6.96 lakhs); AdaBoost did not |
| Hybrid RF + ANN | Cut the ANN's error by **26%** (7.72 → 5.69 lakhs) but did not beat plain RF |
| Key data quality finding | `Millage(KM)` was synthetic (perfectly derived from year) and was dropped |
| Key EDA finding | Year of manufacture **confounds** the engine size vs price relationship |

---

## 2. Dataset details and in-depth EDA

### 2.1 Source

The dataset was scraped from two of Sri Lanka's largest vehicle classifieds sites, ikman.lk and riyasewana.lk. Each row is one advertisement. Ads were posted between **3 December 2024 and 5 February 2025**.

- **Rows:** 9,788
- **Columns:** 17
- **File:** `car_price_dataset.csv` (1.3 MB)

### 2.2 All columns and data types

| # | Column | Type | Unique values | Description | Used? |
|---|---|---|---|---|---|
| 1 | `Unnamed: 0` | int64 | 9,788 | Row index left over from export | ❌ Dropped (no information) |
| 2 | `Brand` | string | 50 | Manufacturer, e.g. TOYOTA, SUZUKI | ✅ |
| 3 | `Model` | string | 1,555 | Model name, e.g. VITZ, ALTO | ✅ |
| 4 | `YOM` | int64 | 62 | Year of manufacture (1956 to 2024) | ✅ |
| 5 | `Engine (cc)` | float64 | 278 | Engine capacity in cubic centimetres | ✅ |
| 6 | `Gear` | string | 2 | Automatic / Manual | ✅ |
| 7 | `Fuel Type` | string | 4 | Petrol / Hybrid / Diesel / Electric | ✅ |
| 8 | `Millage(KM)` | float64 | 62 | Mileage in km | ❌ Dropped (synthetic, see 2.6) |
| 9 | `Town` | string | 107 | Town where the ad was posted | ✅ |
| 10 | `Date` | string | 65 | Ad posting date | ❌ Dropped (2-month window, no signal) |
| 11 | `Leasing` | string | 2 | No Leasing / Ongoing Lease | ✅ |
| 12 | `Condition` | string | 2 | USED / NEW | ✅ |
| 13 | `AIR CONDITION` | string | 2 | Available / Not_Available | ✅ |
| 14 | `POWER STEERING` | string | 2 | Available / Not_Available | ✅ |
| 15 | `POWER MIRROR` | string | 2 | Available / Not_Available | ✅ |
| 16 | `POWER WINDOW` | string | 2 | Available / Not_Available | ✅ |
| 17 | `Price` | float64 | 1,697 | Advertised price in lakhs | 🎯 Target |

### 2.3 Initial hygiene checks

| Check | Result | Action |
|---|---|---|
| Missing values | **0** in every column | None needed |
| Exact duplicate rows (all original columns, excluding the index) | 18 | Removed |
| Duplicates after dropping `Date` and `Millage(KM)` | 252 | Removed (same car re-posted on a different date) |
| Rows after cleaning | **9,536** | |
| Impossible values | Engine 573 to 4,800 cc, year 1956 to 2024, price 2.65 to 790 lakhs | All plausible, kept |

The 252 duplicates are almost all the **same advertisement re-posted** on a different date. Keeping them would let identical cars appear in both the training and the test set, which would inflate test scores (a form of data leakage). Removing them gives a more honest evaluation.

### 2.4 Categorical column distributions

| Column | Distribution |
|---|---|
| `Gear` | Automatic 6,305 (64%), Manual 3,483 (36%) |
| `Fuel Type` | Petrol 7,866 (80%), Hybrid 1,239 (13%), Diesel 614 (6%), Electric 69 (1%) |
| `Leasing` | No Leasing 9,384 (96%), Ongoing Lease 404 (4%) |
| `Condition` | USED 9,694 (99%), NEW 94 (1%) |
| `AIR CONDITION` | Available 9,590 (98%), Not available 198 (2%) |
| `POWER STEERING` | Available 8,918 (91%), Not available 870 (9%) |
| `POWER MIRROR` | Available 7,281 (74%), Not available 2,507 (26%) |
| `POWER WINDOW` | Available 8,003 (82%), Not available 1,785 (18%) |

`Condition` and `AIR CONDITION` are extremely imbalanced (99% and 98% in one class), so they carry very little information. They were kept because tree models simply ignore uninformative features, and they could still help for the rare cases.

**Most common brands:** Toyota (3,089 ads), Suzuki (2,473), Nissan (1,248), Honda (681), Mitsubishi (284).
**Most common models:** Alto (392), Vitz (366), Wagon R Stingray (282), Maruti (229), Wagon R (204).

This reflects the Sri Lankan market well: small, fuel-efficient Japanese and Indian cars dominate.

### 2.5 Target variable: distribution, skewness and the log1p transformation

**Summary statistics of `Price` (lakhs):**

| Statistic | Value |
|---|---|
| Count | 9,788 |
| Mean | 53.46 |
| Standard deviation | 43.63 |
| Minimum | 2.65 |
| 25th percentile | 26.25 |
| Median | 43.00 |
| 75th percentile | 69.75 |
| Maximum | 790.00 |
| **Skewness** | **3.46** |
| **Kurtosis** | **24.9** |

**Interpretation:** the distribution is strongly **right-skewed** (skewness 3.46, where 0 is symmetric) with a very heavy right tail (kurtosis 24.9, where a normal distribution has 0 excess kurtosis). Most cars are between 26 and 70 lakhs, but a small number of luxury cars (BMW, Mercedes-Benz, Land Rover) reach several hundred lakhs. The mean (53.5) is well above the median (43.0), another sign of right skew.

**Why `log1p` was applied:**

The transformation used is `y = log(1 + Price)`, and predictions are converted back with `Price = exp(y) - 1` (`np.expm1`).

1. **It makes the target close to symmetric.** Skewness drops from **3.46 to -0.22**.
2. **It stops expensive cars from dominating training.** Models minimise squared error. On the raw scale, a 50-lakh error on a 500-lakh BMW counts 2,500 times more than a 1-lakh error on a 20-lakh Alto, even though both are a similar *percentage* mistake. On the log scale, errors become roughly **relative (percentage) errors**, so every price range gets fair attention.
3. **It matches how prices behave.** Car prices change multiplicatively: one extra year of age reduces the price by a percentage, not a fixed amount. Log space turns these multiplicative effects into additive ones, which trees and neural networks model more easily.
4. **It helps the neural network.** ANNs train better when the target has a compact, roughly normal range.
5. **Why `log1p` and not `log`?** `log1p(x) = log(1 + x)` is numerically stable and safe for values near zero. No price here is zero, but `log1p` is the standard safe choice.

All metrics in Section 5 are computed **after converting predictions back to lakhs**, so they are directly interpretable.

### 2.6 Key finding 1: Mileage is synthetic and was dropped

Three observations show that `Millage(KM)` is not real odometer data:

1. `Millage(KM)` has **exactly 62 unique values**, the same as `YOM`.
2. Every year of manufacture maps to **exactly one** mileage value.
3. The Pearson correlation between `YOM` and `Millage(KM)` is **-1.000000**. A linear fit gives exactly **-11,000 km per year**:

```
Mileage = 11,000 × (2025 − YOM)
```

So the dataset creator most likely **filled mileage in with a formula** (11,000 km per year of age) where the real value was missing. Real mileage would vary a lot between cars of the same year.

**Consequences and decision:**

- The column contains **zero information beyond `YOM`**. It is perfectly collinear.
- Keeping it would not help accuracy, but it would **mislead feature importance** (the importance would be split arbitrarily between two identical signals) and **mislead anyone using the predictor** into thinking mileage matters independently.
- It was **dropped**.

This is an important lesson: always check features against each other, not only against the target.

### 2.7 Key finding 2: Year confounds the engine size vs price relationship

At first glance, engine size appears almost **unrelated** to price:

| Measure | Value |
|---|---|
| Correlation: `Engine (cc)` vs `log(Price)` | **0.06** (almost none) |
| Correlation: `Engine (cc)` vs `YOM` | **-0.40** (bigger engines are older) |
| Correlation: `YOM` vs `log(Price)` | **0.70** (newer cars cost much more) |
| **Partial correlation**: `Engine (cc)` vs `log(Price)`, controlling for `YOM` | **0.53** |

**Why this happens:** most large-engine vehicles in the data are **old diesel** vehicles. The median diesel car has a 1,970 cc engine and was made in **1996**, while the median hybrid has a 1,490 cc engine and was made in **2015**.

| Fuel type | Median engine (cc) | Median year | Median price (lakhs) |
|---|---|---|---|
| Diesel | 1,970 | 1996 | 24.75 |
| Petrol | 1,300 | 2008 | 38.50 |
| Electric | 1,500 | 2014 | 54.20 |
| Hybrid | 1,490 | 2015 | 73.68 |

Two effects cancel each other out:
- **Bigger engine → higher price** (direct effect)
- **Bigger engine → usually older car → lower price** (indirect effect through age)

`YOM` is a **confounder**: it is related to both engine size and price. To see the true effect of engine size, we remove the influence of year from both variables (regress each on `YOM` and correlate the residuals). This **partial correlation is 0.53**, showing that **among cars of the same age, bigger engines do cost clearly more.**

Tree ensembles handle this automatically, because they can split on year first and then on engine size within each year range. A simple correlation table would have wrongly suggested that engine size does not matter.

### 2.8 Key finding 3: Brand and year distributions

**Median price by brand (brands with at least 20 ads):**

| Brand | Median price (lakhs) | Ads |
|---|---|---|
| BMW | 215.0 | 80 |
| Audi | 139.0 | 56 |
| Mercedes-Benz | 132.3 | 168 |
| Honda | 75.8 | 681 |
| Toyota | 66.5 | 3,089 |

European luxury brands command roughly **2 to 3 times** the price of the popular Japanese brands. At the other end, Indian and budget brands (Maruti, Micro, Perodua, Datsun) are much cheaper.

**Median price by year of manufacture:** prices are roughly flat and low for very old cars (1960s to 1980s, some of which are collectable classics), then rise steadily from about 1990. They jump sharply for **2015 and newer** cars, which reflects both normal depreciation and the import restrictions that made recent models scarce in Sri Lanka.

### 2.9 Key finding 4: Town distribution

There are 107 towns. The busiest are Colombo (1,008 ads), Kurunegala (631), Gampaha (628) and Kandy (618).

**Median price in the 12 towns with the most ads:**

| Highest | Median (lakhs) | Lowest | Median (lakhs) |
|---|---|---|---|
| Dehiwala-Mount Lavinia | 66.0 | Kurunegala | 34.8 |
| Colombo | 58.9 | Gampaha | 35.9 |
| Negombo | 55.6 | Homagama | 37.9 |

Colombo and the nearby coastal suburbs have higher median prices, which likely reflects buyers with higher incomes and more luxury cars listed there. However, the **box plots overlap heavily**: within every town, prices span from very cheap to very expensive cars. Location therefore matters much less than brand, model and year, which the feature importance in Section 5.5 confirms.

### 2.10 EDA summary

| Insight | Effect on modelling |
|---|---|
| Price is heavily right-skewed | Train on `log1p(Price)` |
| Mileage is a formula of year | Drop `Millage(KM)` |
| Date covers only 2 months | Drop `Date` |
| 252 re-posted duplicate ads | Remove to avoid train/test leakage |
| Year confounds engine size | Use non-linear models that capture interactions |
| Brand/Model have many categories | Use target encoding (Section 3) |
| Very imbalanced binary columns | Keep; trees will ignore them if useless |

---

## 3. Data preprocessing and feature engineering

### 3.1 Cleaning steps (in order)

1. Drop `Unnamed: 0`, `Date` and `Millage(KM)`.
2. Drop duplicate rows (252 removed, 9,536 remain).
3. Create the engineered feature **`Age = 2025 − YOM`**.
4. Transform the target: `y = log1p(Price)`.

### 3.2 Final feature set (14 features)

| Group | Features | Count |
|---|---|---|
| High cardinality categorical | `Brand`, `Model`, `Town` | 3 |
| Low cardinality categorical | `Gear`, `Fuel Type`, `Leasing`, `Condition`, `AIR CONDITION`, `POWER STEERING`, `POWER MIRROR`, `POWER WINDOW` | 8 |
| Numeric | `YOM`, `Age`, `Engine (cc)` | 3 |

> **Note on `Age`:** `Age` is a linear function of `YOM`, so it adds no new information to tree models. It was added because "age" is more intuitive to explain, and it costs nothing for trees. The side effect is that importance gets split between `YOM` and `Age` (see 5.5). It could be removed without changing accuracy.

### 3.3 Encoding strategy

Machine learning models need numeric input, so every text column must be converted. The right method depends on how many categories a column has (its **cardinality**).

#### 3.3.1 Target encoding for high-cardinality features (Brand, Model, Town)

`Model` has **1,555** categories, `Town` has **107** and `Brand` has **50**. One-hot encoding these would create about **1,700 extra columns**, most of them almost always zero. That causes:
- a huge, sparse feature matrix that is slow to train on,
- many rare categories with too few examples to learn from (many models appear only once or twice),
- trees that waste splits on one category at a time.

**Target encoding** replaces each category with a **single number: the average target value (log price) for that category**. For example, `BMW` becomes a high number and `MARUTI` a low one. This compresses 1,555 columns into **one meaningful numeric column** that directly tells the model how expensive that model tends to be.

scikit-learn's `TargetEncoder` was used, which includes two safeguards against its main risks:

1. **Smoothing (shrinkage).** For rare categories, the category mean is unreliable (one ad is not enough). The encoder blends the category mean with the overall mean, weighting by how many examples the category has:

   ```
   encoding = (n × category_mean + m × global_mean) / (n + m)
   ```

   A model seen once is pulled strongly towards the global average. A model seen 300 times keeps its own average. scikit-learn chooses the smoothing amount `m` automatically (`smooth="auto"`) using an empirical Bayes estimate.

2. **Internal cross-fitting to prevent target leakage.** If each training row were encoded using an average that *includes its own price*, the model would effectively see the answer, and it would overfit badly. During `fit_transform`, scikit-learn's `TargetEncoder` splits the training data into 5 folds and encodes each fold using statistics computed **only from the other 4 folds**. At prediction time (`transform`), it uses statistics from the whole training set. Categories never seen in training receive the global mean.

Because the encoder is placed inside each model's scikit-learn `Pipeline`, it is fitted **only on the training set**. The test set never influences the encoding.

#### 3.3.2 One-hot encoding for low-cardinality features

The 8 low-cardinality columns have only 2 to 4 categories each. For these, **one-hot encoding** creates one 0/1 column per category (for example `Fuel Type_Hybrid = 1`).

- One-hot encoding does **not impose any order** (Petrol is not "less than" Diesel).
- With so few categories, it adds only **18 columns** in total.
- It is lossless and easy to interpret.

Settings: `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. `handle_unknown="ignore"` means an unseen category at prediction time becomes all zeros instead of crashing.

#### 3.3.3 Resulting matrix

After the `ColumnTransformer`, each car becomes **24 numeric columns**:
- 3 target-encoded columns (Brand, Model, Town)
- 18 one-hot columns
- 3 numeric columns (YOM, Age, Engine)

### 3.4 Scaling strategy

| Model type | Scaling | Reason |
|---|---|---|
| All tree-based models (Decision Tree, Bagging, RF, AdaBoost, XGBoost, LightGBM, CatBoost) | **None** (`passthrough`) | Trees split on thresholds such as "Engine > 1,500". Scaling changes the threshold value but not which rows go left or right, so results are identical. Skipping it keeps the pipeline simpler. |
| ANN and hybrid RF + ANN | **`StandardScaler`** on numeric columns (mean 0, standard deviation 1) | Neural networks use gradient descent. Unscaled inputs (engine ≈ 1,300 vs a one-hot value of 1) make training slow and unstable, because large-valued inputs dominate the weight updates. |

In the ANN pipeline, the one-hot columns are already 0/1 and the target-encoded columns are already on the log-price scale (roughly 1 to 6), so only the 3 numeric columns needed standardisation.

### 3.5 Train/test split

- **Method:** `train_test_split(test_size=0.2, random_state=42)`
- **Training set:** 7,628 cars (80%)
- **Test set:** 1,908 cars (20%)
- The **same split** is used for every model, so the comparison is fair.
- `random_state=42` makes the split reproducible.
- The test set is used **only for final evaluation**. Every fitted step (encoders, scalers, models) learns only from the training set, because everything is wrapped in a `Pipeline`.

A random split (not stratified) was used because the target is continuous. With 1,908 test cars, every price range is well represented: 432 cars under 25 lakhs, 629 between 25 and 50, 666 between 50 and 100, and 181 above 100.

### 3.6 Pipeline structure

Every tree model is wrapped like this:

```python
Pipeline([
    ("prep", ColumnTransformer([
        ("te",  TargetEncoder(random_state=42), ["Brand", "Model", "Town"]),
        ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOW_CARD),
        ("num", "passthrough", ["YOM", "Age", "Engine (cc)"]),
    ])),
    ("model", <estimator>),
])
```

Benefits:
- **No leakage**: preprocessing is fitted inside `fit()` on training data only.
- **One object to save**: the saved `.pkl` file contains both preprocessing and model, so the dashboard can predict straight from raw inputs like `"TOYOTA"`, `"VITZ"`, `2010`.

---

## 4. Model architectures and mechanics

All models predict `log1p(Price)`. Hyperparameters were set to sensible, commonly used values. No automated hyperparameter search was run (see Section 8).

### 4.1 Baseline: single decision tree

```python
DecisionTreeRegressor(random_state=42)   # no depth limit
```

**How it works.** A regression tree repeatedly splits the data on one feature at a time, choosing the split that most reduces the squared error (variance) in the resulting groups. For example: "Is `Model` encoding > 4.1?", then "Is `YOM` > 2012?", and so on. Each final leaf predicts the **average log price** of the training cars that end up in it.

**Why it overfits (high variance).** With no depth limit, the tree keeps splitting until leaves are tiny, often a single car. It then memorises the training data, including its noise (for example, one seller who priced their car unusually high). The consequences:
- Training error is close to zero, but test error is much higher.
- It is **unstable**: a small change in the training data can produce a completely different tree, because one different early split changes everything below it.

In bias-variance terms, a deep tree has **low bias** (it can fit any pattern) but **high variance** (its predictions change a lot depending on the exact training sample).

```
Expected error = Bias² + Variance + Irreducible noise
```

Ensembles attack this: **bagging reduces variance** and **boosting reduces bias**.

**Result:** MAE **6.96 lakhs**, R² **0.870**. This is the bar every ensemble must beat.

### 4.2 Bagging family

#### 4.2.1 Bagging regressor

```python
BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=2),
                 n_estimators=100, n_jobs=-1, random_state=42)
```

**Bootstrap aggregating (bagging):**

1. **Bootstrap sampling.** Create 100 new training sets by sampling the 7,628 training rows **with replacement**. Each bootstrap sample has the same size as the original, but some rows appear several times and others not at all. On average each sample contains about **63.2%** of the unique rows. The remaining ~36.8% are "out-of-bag" for that tree.

   (The probability a given row is never picked in *n* draws is (1 − 1/n)ⁿ ≈ e⁻¹ ≈ 0.368.)

2. **Train one deep tree per sample**, independently and **in parallel** (`n_jobs=-1` uses all CPU cores).

3. **Aggregate** by averaging the 100 predictions.

**Why averaging reduces variance.** If we average *B* models that each have variance σ², and the models have pairwise correlation ρ, the variance of the average is:

```
Var(average) = ρσ² + (1 − ρ)σ² / B
```

- As *B* grows, the second term shrinks towards zero.
- The first term, **ρσ²**, remains. So the benefit is limited by how **correlated** the trees are.

Each tree still overfits its own sample, but in **different ways**, so their individual errors partly cancel out when averaged. Bias stays roughly the same as a single deep tree. Variance drops a lot.

`min_samples_leaf=2` requires at least 2 cars per leaf. This slightly reduces each tree's overfitting and roughly halves the saved model size, with no loss in accuracy.

**Result:** MAE **5.50 lakhs**, R² **0.900**, a **21% error reduction** over the single tree.

#### 4.2.2 Random Forest

```python
RandomForestRegressor(n_estimators=300, max_features=0.5,
                      min_samples_leaf=2, n_jobs=-1, random_state=42)
```

Random Forest is bagging **plus feature subsampling**.

**The problem with plain bagging:** if one feature is very strong (here, the encoded `Model` or `YOM`), almost every bagged tree will use it for the first split. The trees then look alike, their correlation ρ is high, and the ρσ² term in the variance formula stays large.

**The fix:** at **every split**, each tree may only choose from a **random subset of features**. Here `max_features=0.5` means a random 50% of the 24 columns (12) are considered at each split.

Effects:
- Trees are forced to sometimes split on other useful features (Brand, Engine, Gear).
- Trees become **more different (decorrelated)**, so ρ falls.
- Averaging decorrelated trees reduces variance further than plain bagging.

Each individual tree is slightly weaker, but the **ensemble** is stronger. Note that if `max_features` equalled all features, Random Forest would be identical to bagging.

**Hyperparameters explained:**

| Parameter | Value | Meaning |
|---|---|---|
| `n_estimators` | 300 | Number of trees. More trees never overfit in RF, they just cost more time. Gains flatten after a few hundred. |
| `max_features` | 0.5 | Fraction of features considered per split. Controls decorrelation. (The common rule `p/3` for regression would be 8 of 24.) |
| `min_samples_leaf` | 2 | Minimum cars per leaf. |
| `n_jobs` | -1 | Train trees in parallel on all CPU cores. |

**Result:** MAE **5.39 lakhs**, R² **0.901**, the **lowest error of all models**.

### 4.3 Boosting family

**Core idea.** Instead of training many strong trees independently and averaging them, boosting trains many **weak** (shallow) models **sequentially**. Each new model focuses on the mistakes of the ensemble built so far. The final prediction is a **weighted sum** of all models. Boosting mainly **reduces bias**: each step makes the overall model able to fit more of the pattern.

| | Bagging | Boosting |
|---|---|---|
| Training order | Parallel, independent | Sequential, each depends on the previous |
| Base learners | Deep trees (low bias, high variance) | Shallow trees (high bias, low variance) |
| Main goal | Reduce variance | Reduce bias |
| Combination | Simple average | Weighted sum |
| Overfitting risk | Low | Higher, controlled by learning rate, depth and regularisation |

#### 4.3.1 AdaBoost (Adaptive Boosting)

```python
AdaBoostRegressor(DecisionTreeRegressor(max_depth=8),
                  n_estimators=200, learning_rate=0.05, random_state=42)
```

AdaBoost (here the AdaBoost.R2 variant for regression) boosts by **re-weighting the training samples**:

1. Start with equal weights on all training cars.
2. Train a tree (depth 8) on a sample drawn according to the weights.
3. Measure each car's error relative to the largest error (linear loss by default).
4. **Increase the weight** of cars with large errors, so the next tree focuses on them.
5. Give the tree a voting weight based on its overall accuracy (more accurate trees count more).
6. Repeat 200 times.
7. Final prediction: the **weighted median** of all trees' predictions.

`learning_rate=0.05` shrinks each tree's contribution so that no single tree dominates.

**Why AdaBoost was weakest here (MAE 7.10, worse than the single tree):** AdaBoost is very **sensitive to noisy and extreme samples**. Sale ads contain noise (unusual asking prices, typos, rare luxury cars). AdaBoost keeps increasing the weight on these hard cases, so later trees concentrate on a handful of outliers instead of the general pattern. Modern gradient boosting replaces re-weighting with gradient-based residual fitting plus regularisation, which is much more robust.

#### 4.3.2 Gradient boosting: the shared mechanism

XGBoost, LightGBM and CatBoost are all **gradient boosting** implementations. The shared algorithm:

1. Start with a constant prediction F₀ (the mean log price).
2. For each round *m* = 1 … M:
   - Compute **residuals** (more precisely, the **negative gradients** of the loss) for every training car. For squared error, this is simply `actual − current prediction`.
   - Fit a small tree *hₘ* to predict these residuals.
   - Update the model: **Fₘ = Fₘ₋₁ + η · hₘ**, where η is the **learning rate**.
3. Final prediction = F₀ + η·h₁ + η·h₂ + … + η·h_M

Each new tree learns "what the current model is still getting wrong", and adds a **small** correction.

**Learning rate (η = 0.05 for all three).** Each tree contributes only 5% of its prediction. Small steps generalise better, because the model approaches the solution gradually instead of over-correcting. The trade-off is that more trees are needed, hence 800 to 1,000 rounds.

**Why gradient boosting reduces bias.** A single shallow tree is too simple (high bias). By adding hundreds of corrections, the ensemble can represent very complex functions, gradually removing the remaining error.

#### 4.3.3 XGBoost (eXtreme Gradient Boosting)

```python
XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6,
             subsample=0.8, colsample_bytree=0.8, random_state=42)
```

XGBoost adds several improvements to basic gradient boosting:

- **Second-order optimisation.** It uses both the gradient (first derivative) and the **Hessian** (second derivative) of the loss to choose splits and leaf values, which gives more accurate steps.
- **Built-in regularisation.** Its objective penalises complexity:

  ```
  Objective = Σ loss(yᵢ, ŷᵢ) + Σ [ γ·T + ½·λ·Σ w² ]
  ```

  where *T* is the number of leaves and *w* are the leaf values. This discourages overly complex trees.
- **Level-wise tree growth** up to `max_depth=6` (balanced trees).
- **Row subsampling** (`subsample=0.8`): each tree sees a random 80% of the rows (stochastic gradient boosting), which adds randomness and reduces overfitting.
- **Column subsampling** (`colsample_bytree=0.8`): each tree uses a random 80% of the features, similar to Random Forest.
- **Histogram-based split finding** and multi-threading for speed.

**Result:** MAE **5.87 lakhs**, R² **0.889**, training time **1.47 s** (about 7.6x faster than RF in Colab).

#### 4.3.4 LightGBM (Light Gradient Boosting Machine)

```python
LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=31,
              subsample=0.8, colsample_bytree=0.8, random_state=42)
```

LightGBM, from Microsoft, is designed for speed:

- **Leaf-wise (best-first) growth.** Instead of growing all branches of a level together, it always splits the **single leaf that reduces loss the most**. This produces deeper, unbalanced trees that reach lower loss with the same number of leaves. The tree size is controlled by `num_leaves=31` rather than depth. On small datasets, leaf-wise growth can overfit, so it needs regularisation.
- **Histogram binning.** Continuous features are bucketed into discrete bins (255 by default), so split search is much faster.
- **GOSS** (Gradient-based One-Side Sampling) and **EFB** (Exclusive Feature Bundling) are further speed optimisations for large datasets.

> Note: in LightGBM, `subsample` only takes effect if `subsample_freq > 0`. With the default `subsample_freq=0`, row subsampling was effectively not applied. Column subsampling (`colsample_bytree=0.8`) was applied.

**Result:** MAE **5.88 lakhs**, R² **0.908** (second-best R²), training time **1.49 s**.

#### 4.3.5 CatBoost (Categorical Boosting)

```python
CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6,
                  random_seed=42, verbose=0)
```

CatBoost, from Yandex, has two signature features:

- **Ordered boosting.** Standard gradient boosting computes residuals using a model that was trained on the same rows, which causes a subtle bias called **prediction shift**. CatBoost uses random permutations so that each row's residual is computed from a model that has **not seen that row**. This reduces overfitting, especially on small or noisy data.
- **Symmetric (oblivious) trees.** Every node at the same depth uses the **same split condition**. A depth-6 tree is effectively a lookup table of 2⁶ = 64 leaves. These trees are fast to evaluate and act as a strong regulariser.
- **Native categorical handling** (ordered target statistics). CatBoost can encode categorical columns itself. In this project, categorical columns were already encoded by the shared pipeline, so all models received identical inputs for a fair comparison. This means CatBoost's native categorical advantage was **not** used.

**Result:** MAE **5.98 lakhs**, R² **0.912**, the **best R² and best RMSE** of all models, training time **3.55 s**.

### 4.4 Neural network baseline: ANN alone

```python
MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True,
             max_iter=500, random_state=42)
```

**Architecture:**

```
Input (24 features, scaled)
  → Dense 128, ReLU
  → Dense 64, ReLU
  → Output 1 (linear): predicted log price
```

- **Optimiser:** Adam (scikit-learn default), learning rate 0.001
- **Loss:** squared error
- **Regularisation:** L2 penalty α = 0.0001 (default), plus **early stopping**: 10% of the training data is held out for validation, and training stops when the validation score has not improved for 10 consecutive epochs.

**Result:** MAE **7.72 lakhs**, R² **0.835**, the **worst model**. The reasons are discussed in Section 7.

### 4.5 Stacking: RF + XGBoost + LightGBM + CatBoost → Ridge

```python
StackingRegressor(
    estimators=[("rf", RandomForestRegressor(...)),   # same settings as 4.2.2
                ("xgb", XGBRegressor(...)),           # same settings as 4.3.3
                ("lgbm", LGBMRegressor(...)),         # same settings as 4.3.4
                ("cat", CatBoostRegressor(...))],     # same settings as 4.3.5
    final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)),
    cv=5)
```

**How stacking works:**

1. **Level 0 (base models):** four diverse models, one from the bagging family (RF) and three from the boosting family (XGBoost, LightGBM, CatBoost), each predict log price.
2. **Out-of-fold predictions:** `StackingRegressor` uses 5-fold cross-validation on the training set. For each fold, every base model is trained on the other 4 folds and predicts the held-out fold. This gives every training car four predictions from models that never saw it (the same leakage-prevention idea explained in 4.6.3).
3. **Level 1 (meta-model):** a **Ridge regression** is trained on those four out-of-fold predictions. It learns the best weighted combination. `RidgeCV` picks its regularisation strength by cross-validation (chosen: α ≈ 31.6). Ridge is a good meta-model because the four base predictions are highly correlated, and Ridge's L2 penalty keeps the weights stable instead of letting them swing to large positive and negative values.
4. **Final fit:** each base model is retrained on the full training set. At prediction time, the four base predictions go through the Ridge weights.

**Why stacking can beat each base model:** the base models make **different kinds of mistakes**. Bagging (RF) is very stable on typical cars, while boosting reduces large misses on extreme cars (see 5.3). A weighted blend can cancel part of each model's errors.

**Learned meta-model weights:**

| Base model | Weight |
|---|---|
| Random Forest | 0.33 |
| XGBoost | 0.25 |
| LightGBM | 0.22 |
| CatBoost | 0.21 |

The weights sum to about 1.0 (intercept ≈ -0.02), so the meta-model is essentially a **weighted average**, with the most trust in Random Forest.

**Result:** MAE **5.45 lakhs** (2nd best, close to RF's 5.39), R² **0.913** and RMSE **13.73** (best of all models, tied with CatBoost), MAPE 13.43%. Stacking combines RF's low typical error with CatBoost's strong squared-error performance.

> Implementation note: the stacking model was added after the main Colab run. It was trained on the **identical train/test split** (verified: the test set's actual prices match the Colab run exactly) with scikit-learn 1.6.1, so its metrics are directly comparable. Its training time (≈27 s) was measured on a different machine, so it is approximate. The notebook now includes the stacking step.

### 4.6 Hybrid: Random Forest + ANN (a stacking variant)

#### 4.6.1 Concept

This hybrid is a special case of stacking (4.5) with **one** base model and a **neural network** as the meta-model. It also passes the original features to the meta-model. Here:

- **Base model:** Random Forest (strong at sharp, rule-like patterns such as brand and year thresholds)
- **Meta-model:** ANN (smooth function approximator that can learn corrections)

The ANN receives the **original 24 features plus the RF prediction** as a 25th input.

#### 4.6.2 Architecture

```
                    ┌──────────────────────┐
 Car features ─────►│ Random Forest (300)  │──► RF prediction ─┐
 (24 scaled cols)   └──────────────────────┘                   │
        │                                                      ▼
        └──────────────────────────────────────────────► [24 features + RF pred = 25 inputs]
                                                               │
                                                     Dense 64, ReLU
                                                               │
                                                     Dense 32, ReLU
                                                               │
                                                     Output: log price
```

```python
# Shared preprocessing, with StandardScaler on numeric columns
prep_ann = make_preprocessor(scale=True)
Xtr_p = prep_ann.fit_transform(X_train, y_train)
Xte_p = prep_ann.transform(X_test)

# Base model
rf_base = RandomForestRegressor(n_estimators=300, max_features=0.5,
                                min_samples_leaf=2, n_jobs=-1, random_state=42)

# Out-of-fold predictions for the training set
rf_oof = cross_val_predict(rf_base, Xtr_p, y_train,
                           cv=KFold(5, shuffle=True, random_state=42))

# Full RF for the test set
rf_base.fit(Xtr_p, y_train)
rf_test = rf_base.predict(Xte_p)

# Meta-model
hybrid_ann = MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True,
                          max_iter=500, random_state=42)
hybrid_ann.fit(np.column_stack([Xtr_p, rf_oof]), y_train)
p_hybrid = hybrid_ann.predict(np.column_stack([Xte_p, rf_test]))
```

#### 4.6.3 Why out-of-fold predictions are essential

If we trained RF on the full training set and then fed its predictions **on that same training set** to the ANN, those predictions would be almost perfect, because RF has already seen (and partly memorised) those rows. The ANN would learn to **trust the RF prediction completely**. At test time, RF's predictions on new cars are much less accurate, so the ANN would fail. This is **leakage between stacking layers**.

**Out-of-fold (OOF) solution with 5-fold cross-validation:**

1. Split the training set into 5 folds.
2. For each fold: train an RF on the other 4 folds and predict the held-out fold.
3. Combine the 5 sets of held-out predictions. Every training car now has an RF prediction from a model that **never saw it**.

These OOF predictions have **realistic error levels**, the same kind of error RF makes on unseen test cars. The ANN therefore learns **how much to trust RF and when to correct it**. For the test set, a final RF trained on all training data makes the predictions.

#### 4.6.4 How the meta-model combines representations

The ANN has two kinds of information:
- **RF prediction:** a strong, already-good estimate, built from hard thresholds (piecewise-constant).
- **Raw features:** let the ANN learn **smooth adjustments** to that estimate, for example where RF systematically under-predicts expensive cars or over-predicts certain brands.

Because the RF prediction is already close to the answer, the ANN essentially learns `final ≈ RF prediction + small correction(features)`, which is a much easier task than learning the whole price function from scratch. That is why a **smaller network (64 → 32)** is enough for the meta-model.

#### 4.6.5 Result

| Model | MAE (lakhs) | R² |
|---|---|---|
| ANN alone | 7.72 | 0.835 |
| Random Forest alone | 5.39 | 0.901 |
| **Hybrid RF + ANN** | **5.69** | **0.900** |
| Stacking (4 base models → Ridge), for comparison | 5.45 | 0.913 |

- The hybrid **reduced the ANN's error by 26%**, a large improvement.
- It reached almost the same R² as RF, but its MAE is slightly **worse** than RF alone.
- **Conclusion:** on this tabular dataset, RF already captures nearly all the learnable signal. The ANN adds complexity and training time (33.8 s, the slowest model) without adding accuracy over RF.

**Why RF + ARIMA was not attempted:** ARIMA models a single variable over time (a time series). This dataset is a **cross-sectional snapshot** of ads from a two-month window, so there is no meaningful time series to model.

---

## 5. Evaluation metrics and head-to-head comparison

### 5.1 Metrics explained (in the context of car prices)

All metrics are computed on the **1,908 test cars**, **after converting predictions back to lakhs** with `expm1`. Let *yᵢ* be the actual price, *ŷᵢ* the predicted price and *n* the number of test cars.

#### MAE: Mean Absolute Error

```
MAE = (1/n) Σ |yᵢ − ŷᵢ|
```

- **Meaning:** the average size of the mistake, in lakhs.
- **Example:** MAE 5.39 means that, on average, the predicted price is **about 5.4 lakhs (Rs 539,000) away** from the advertised price.
- **Why it is the primary metric:** it is in the same unit as the price, easy to explain to a buyer, and not dominated by a few extreme luxury cars.

#### RMSE: Root Mean Squared Error

```
RMSE = √[ (1/n) Σ (yᵢ − ŷᵢ)² ]
```

- **Meaning:** also in lakhs, but errors are **squared before averaging**, so large mistakes are penalised much more.
- **Here:** RMSE (≈ 14 to 19 lakhs) is about **2.5x larger than MAE** (≈ 5 to 8 lakhs). That gap shows the errors are **uneven**: most cars are predicted well, but a few expensive cars have large errors that inflate RMSE.

#### R²: Coefficient of determination

```
R² = 1 − Σ (yᵢ − ŷᵢ)² / Σ (yᵢ − ȳ)²
```

- **Meaning:** the fraction of the variation in car prices that the model explains. 1.0 is perfect, 0 is no better than always guessing the average price.
- **Here:** R² 0.912 means the model explains **91.2% of the variation** in prices.
- **Note:** R² uses squared errors (like RMSE), so it is also sensitive to the large errors on expensive cars.

#### MAPE: Mean Absolute Percentage Error

```
MAPE = (100 / n) Σ |yᵢ − ŷᵢ| / yᵢ
```

- **Meaning:** the average error **as a percentage of the car's price**.
- **Example:** MAPE 13.4% means predictions are on average 13.4% off. A 5-lakh error matters more for a 20-lakh car (25%) than for a 200-lakh car (2.5%), and MAPE captures that.
- **Why it matters here:** because training was done on the log scale (which optimises relative error), MAPE is closely aligned with what the models actually optimised.

### 5.2 Complete comparison table

Results from the final Colab run, sorted by MAE (lower is better). Best value in each column is in **bold**.

| Rank | Model | Family | R² | MAE (lakhs) | RMSE (lakhs) | MAPE (%) | Train time (s) |
|---|---|---|---|---|---|---|---|
| 1 | Random Forest | Bagging | 0.901 | **5.39** | 14.59 | **13.41** | 11.17 |
| 2 | Stacking | Stacking | **0.913** | 5.45 | **13.73** | 13.43 | ≈27.2† |
| 3 | Bagging | Bagging | 0.900 | 5.50 | 14.68 | 13.87 | 11.51 |
| 4 | Hybrid RF + ANN | Hybrid | 0.900 | 5.69 | 14.69 | 13.71 | 33.84 |
| 5 | XGBoost | Boosting | 0.889 | 5.87 | 15.44 | 14.02 | 1.47 |
| 6 | LightGBM | Boosting | 0.908 | 5.88 | 14.12 | 14.40 | 1.49 |
| 7 | CatBoost | Boosting | 0.912 | 5.98 | 13.73 | 14.32 | 3.55 |
| 8 | Decision Tree (baseline) | Single model | 0.870 | 6.96 | 16.72 | 17.33 | **0.18** |
| 9 | AdaBoost | Boosting | 0.878 | 7.10 | 16.21 | 16.05 | 9.97 |
| 10 | ANN (alone) | Neural network | 0.835 | 7.72 | 18.87 | 16.71 | n/a* |

\* The ANN's training time was not recorded separately in the Colab run. It is included in the hybrid's 33.84 s. (The updated notebook now records it.)

† Stacking was trained on a different machine with the same split and library version, so its time is approximate. Stacking's R² is only marginally above CatBoost's (0.91252 vs 0.91246), so the two are effectively tied on R² and RMSE.

Training times were measured on Colab's CPU runtime and will vary between runs and machines. Relative differences are more meaningful than exact values.

**Family averages:**

| Family | Average MAE (lakhs) | Average R² |
|---|---|---|
| Bagging (Bagging, RF) | 5.45 | 0.901 |
| Boosting (AdaBoost, XGBoost, LightGBM, CatBoost) | 6.21 | 0.897 |
| Boosting excluding AdaBoost | 5.91 | 0.903 |
| Stacking | 5.45 | 0.913 |

### 5.3 Results interpretation

#### Which model won on what metric?

| Metric | Winner | Why |
|---|---|---|
| MAE, MAPE (typical error) | **Random Forest** | Averaging 300 deep, decorrelated trees gives very stable predictions for the large majority of ordinary cars. |
| R², RMSE (squared error) | **Stacking**, with **CatBoost** best among single models | CatBoost's worst misses on a few extreme cars are smaller than RF's. Stacking blends RF with the boosting models and gets the best of both. |
| Training speed (ensembles) | **XGBoost / LightGBM** | Histogram-based split finding and highly optimised C++ code. |

The **metrics disagree** because they weight errors differently. MAE treats every lakh of error equally. R² and RMSE square the errors, so a **handful of very large errors** dominate them.

It is tempting to conclude "boosting handles expensive cars better", but the data does **not** support that. On **average** error, RF beats CatBoost even for cars over 100 lakhs (19.81 vs 21.39 lakhs, see the table below). What differs is the **few worst misses**:

| Cars over 100 lakhs (181 test cars) | Random Forest | CatBoost |
|---|---|---|
| Average error (MAE) | **19.8** | 21.4 |
| RMSE | 40.0 | **37.1** |
| Single worst miss | 290.5 | **261.2** |
| 5 worst misses | 115, 133, 147, 240, 291 | **86, 96, 128, 221, 261** |

So:
- **Random Forest** has the lowest typical error in every price range among single models.
- **CatBoost** has smaller **extreme** errors on a few unusual cars, and because R² and RMSE square errors, those few cars are enough to give it the higher R².
- **Stacking** combines both strengths.

#### Error by price range (all main models)

Average absolute error (lakhs) in each price band:

| Price range | Test cars | Decision Tree | Random Forest | XGBoost | LightGBM | CatBoost | Stacking |
|---|---|---|---|---|---|---|---|
| Under 25 lakhs | 432 | 5.06 | 3.77 | 3.76 | 3.88 | 3.76 | **3.66** |
| 25 to 50 lakhs | 629 | 4.23 | **3.28** | 3.55 | 3.64 | 3.66 | 3.36 |
| 50 to 100 lakhs | 666 | 6.83 | **4.52** | 4.90 | 5.13 | 5.42 | 4.70 |
| Over 100 lakhs | 181 | 21.45 | 19.81 | 22.61 | 21.25 | 21.39 | **19.73** |

- For cheap and mid-range cars, errors are small (3 to 5 lakhs) and the ensembles are close.
- For cars **over 100 lakhs**, errors jump to about **20 lakhs**. There are few luxury cars (only about 9% of the test set), their prices vary hugely, and model-specific factors (trim level, condition, import status) are not in the data.
- Among single models, **RF has the lowest or tied-lowest error in every band**. RF beats XGBoost in every band above 25 lakhs.
- **Stacking** is lowest of all in the cheapest and the most expensive bands.

#### How significant are these differences?

All results come from **one** 80/20 split. Differences of a few tenths of a lakh in MAE (for example RF 5.39 vs Stacking 5.45, or XGBoost 5.87 vs LightGBM 5.88) or 0.01 in R² (0.901 vs 0.912) are **small** and could change order on a different split. The robust conclusions are the large gaps: all good ensembles vs the single tree (about 1 to 1.5 lakhs), and tree ensembles vs the ANN (about 2.3 lakhs). Repeated k-fold cross-validation would be needed to say whether the small gaps are statistically significant (see Section 8).

#### Accuracy vs training speed trade-off

- RF is the most accurate on MAE but took **11.2 s**, about **7.6x slower** than XGBoost (1.47 s).
- The accuracy difference is small: **0.48 lakhs** (Rs 48,000) of MAE.
- On a 9,500-row dataset, 11 seconds is trivial. On millions of rows or with frequent retraining, the faster boosting libraries would be the practical choice.
- The **hybrid** is the slowest (33.8 s) and does not beat RF, so it is not worth the extra complexity here.

#### Key conclusions

1. **Bagging, gradient boosting and stacking beat the single tree.** The single tree's MAE of 6.96 dropped to 5.39 to 5.98. This confirms the central claim of ensemble learning: combining many models reduces error.
2. **AdaBoost is the exception.** It was worse than the single tree, because its sample re-weighting over-focuses on noisy, extreme ads.
3. **Bagging and gradient boosting perform similarly overall.** RF wins on typical error, CatBoost on squared error (because of a few extreme cars). Neither dominates.
4. **Stacking combines them.** Blending RF with three boosting models gives the best R² and RMSE while keeping MAE close to RF.
5. **Untuned defaults favour Random Forest.** RF is famously robust with default settings. Gradient boosting models usually need tuning (learning rate, depth, number of trees, early stopping) to reach their best. With tuning, they would likely close the gap or overtake RF.
6. **Neural networks underperform on this tabular data**, even when helped by stacking (the RF + ANN hybrid).

### 5.4 Prediction example (from the live dashboard)

For a **2010 Toyota Vitz**, 1,000 cc, automatic, petrol, Colombo, used, all equipment available:

| Model | Predicted (lakhs) |
|---|---|
| AdaBoost | 67.8 |
| LightGBM | 63.4 |
| XGBoost | 62.3 |
| CatBoost | 59.9 |
| Random Forest | 59.0 |
| Hybrid RF + ANN | 58.3 |
| Bagging | 58.2 |
| **Average** | **61.3** |
| **Actual median of 344 similar ads (Toyota Vitz)** | **60.1** |

(Measured before the stacking model was added. With stacking included, the dashboard's average becomes 61.2 lakhs.)

The models agree closely, and the average is within about 1.2 lakhs of the real market median. Random Forest and Bagging are closest. AdaBoost overestimates, consistent with its weaker test performance.

### 5.5 Feature importance: what drives car prices in Sri Lanka

#### 5.5.1 Built-in (impurity / gain) importance

This is the importance shown in the notebook and dashboard. One-hot columns are summed back to their original feature, and values are normalised to sum to 1.

| Feature | Random Forest | XGBoost |
|---|---|---|
| Gear | 0.362 | 0.863 |
| Model | 0.236 | 0.019 |
| YOM | 0.132 | 0.032 |
| Age | 0.131 | 0.014 |
| Brand | 0.071 | 0.008 |
| Engine (cc) | 0.027 | 0.006 |
| Town | 0.012 | 0.002 |
| Power Mirror | 0.011 | 0.008 |
| Power Window | 0.009 | 0.009 |
| Fuel Type | 0.003 | 0.011 |
| Power Steering | 0.003 | 0.008 |
| Leasing | 0.002 | 0.009 |
| Air Condition | 0.001 | 0.005 |
| Condition | 0.001 | 0.009 |

- **RF importance** = total reduction in squared error from all splits on that feature, averaged over trees.
- **XGBoost importance** = average **gain** from splits on that feature.

**Why Gear looks so dominant (especially for XGBoost).** Gear is strongly linked to both price and age:

| Gear | Median price (lakhs) | Median year |
|---|---|---|
| Automatic | 61.0 | 2012 |
| Manual | 22.9 | 2003 |

The single split "manual or automatic" divides the data into old/cheap and newer/expensive cars, producing a very large error reduction early on. Gain-based importance credits that whole reduction to Gear. In XGBoost especially, the first trees capture this big split, and later trees only make small corrections, so their gain is small. Built-in importance therefore **over-credits** features that are good early "shortcuts", and is known to be biased.

#### 5.5.2 Permutation importance (more reliable)

To check, **permutation importance** was computed on the test set. Each feature's values are randomly shuffled (5 repeats), and we measure how much the model's R² (on log price) drops. A big drop means the model truly relies on that feature for accurate predictions on unseen data.

| Rank | Random Forest | Drop in R² | XGBoost | Drop in R² |
|---|---|---|---|---|
| 1 | Model | 0.227 | YOM | 0.432 |
| 2 | YOM | 0.095 | Model | 0.205 |
| 3 | Age | 0.094 | Brand | 0.101 |
| 4 | Brand | 0.075 | Engine (cc) | 0.050 |
| 5 | Gear | 0.052 | Gear | 0.045 |
| 6 | Engine (cc) | 0.022 | Age | 0.005 |
| 7 | Power Mirror | 0.004 | Fuel Type | 0.004 |
| 8 | Fuel Type | 0.002 | Power Mirror | 0.003 |

> Permutation importance was computed as an extra validation step for this document. It is not part of the notebook or dashboard.

**Why YOM/Age look different between models:** `Age` and `YOM` contain exactly the same information. RF spreads its reliance across both (0.095 and 0.094). XGBoost happened to use `YOM` almost exclusively (0.432 vs 0.005). Shuffling one of them hurts less when the model also uses the other, so for RF their **combined** importance is what matters.

#### 5.5.3 What drives car prices in Sri Lanka

Combining both methods:

1. **Model (and Brand):** the single most important factor. *What car it is* (a BMW 5 Series vs a Suzuki Alto) sets the price range.
2. **Year of manufacture / age:** newer cars cost much more, and post-2015 cars are especially expensive given import restrictions.
3. **Gear:** automatic cars are much more expensive, partly because they are newer, partly because buyers value them.
4. **Engine size:** matters **once age is accounted for** (see the confounder in 2.7).
5. **Fuel type, equipment, town, leasing and condition:** minor effects. Location barely matters, consistent with the overlapping town box plots.

---

## 6. Streamlit dashboard architecture

### 6.1 Overview

The dashboard (`app.py`) is a single-file Streamlit app deployed on Streamlit Community Cloud. It **does not train models**. It loads the artifacts produced by the Colab notebook and displays them.

```
car-price-ensemble/
├── app.py                        # Streamlit dashboard
├── ensemble_car_price.ipynb      # Colab training notebook
├── requirements.txt              # Dependencies (scikit-learn pinned to 1.6.1)
├── README.md
└── artifacts/                    # Produced by the notebook
    ├── random_forest.pkl         # Full pipeline: preprocessing + model
    ├── bagging.pkl
    ├── xgboost.pkl
    ├── lightgbm.pkl
    ├── catboost.pkl
    ├── adaboost.pkl
    ├── stacking.pkl              # Full pipeline: preprocessing + StackingRegressor
    ├── stacking_weights.json     # Ridge meta-model weights
    ├── hybrid_rf_ann.pkl         # Dict: {prep, rf, ann}
    ├── results.csv               # Metrics for all 10 models
    ├── feature_importance.csv    # RF and XGBoost importance
    ├── test_predictions.csv      # Actual vs predicted for 1,908 test cars
    ├── clean_data.csv            # Cleaned dataset (9,536 rows)
    └── meta.json                 # Dropdown options for the predictor
```

### 6.2 Data flow

```
 Google Colab                                  Streamlit Cloud
 ┌───────────────────────────┐                ┌──────────────────────────────┐
 │ Notebook                  │                │ app.py                       │
 │ 1. Load CSV               │                │                              │
 │ 2. EDA                    │  artifacts/    │ load_tables()  (cache_data)  │
 │ 3. Preprocess             │ ─────────────► │ load_models()  (cache_resource)
 │ 4. Train 10 models        │  via GitHub    │                              │
 │ 5. Evaluate               │                │ 7 tabs render charts/tables  │
 │ 6. Save .pkl/.csv/.json   │                │ Predictor runs live inference│
 └───────────────────────────┘                └──────────────────────────────┘
```

### 6.3 Loading and caching

Streamlit **re-runs the whole script from top to bottom** every time the user interacts (changes a dropdown, switches a tab). Without caching, every click would reload ~60 MB of models from disk. Two caching decorators prevent this:

```python
@st.cache_data
def load_tables():
    results = pd.read_csv(ART / "results.csv")
    fi      = pd.read_csv(ART / "feature_importance.csv", index_col=0)
    preds   = pd.read_csv(ART / "test_predictions.csv")
    data    = pd.read_csv(ART / "clean_data.csv")
    meta    = json.load(open(ART / "meta.json"))
    return results, fi, preds, data, meta

@st.cache_resource
def load_models():
    models = {name: joblib.load(ART / f) for name, f in MODEL_FILES.items()}
    models["Hybrid RF + ANN"] = joblib.load(ART / "hybrid_rf_ann.pkl")
    return models
```

| Decorator | Used for | Behaviour |
|---|---|---|
| `st.cache_data` | DataFrames and JSON | Runs once. Returns a **copy** of the cached data on each call, so accidental changes can't corrupt the cache. Best for serialisable data. |
| `st.cache_resource` | Trained models | Runs once per server process. Returns the **same object** to every user and every rerun (no copying). Best for large, read-only objects like ML models. |

After the first load, interactions are near-instant.

**State management:** the app does not need `st.session_state`. Each widget (selectbox, slider, toggle, radio) holds its own value across reruns, and all derived values (filtered tables, predictions) are recomputed from widget values on each rerun, which is cheap once data and models are cached.

**Version pinning:** `.pkl` files are pickled Python objects tied to the library version that created them. The models were trained with **scikit-learn 1.6.1** in Colab, so `requirements.txt` pins `scikit-learn==1.6.1`. With a different version, unpickling can fail (classes renamed or moved internally) or behave inconsistently.

### 6.4 The 7 tabs

The header shows four summary cards computed from `results.csv`: cars in the dataset, lowest-error model, best-R² model and fastest-training ensemble (bagging or boosting family only, so the baseline tree is excluded).

| # | Tab | Contents | Data source |
|---|---|---|---|
| 1 | 📊 **Dataset** | Brand/model/median-price cards; price histogram with log-scale toggle; median price by brand (brands with 20+ ads); median price by year; the mileage data quality finding; engine cc vs price scatter (coloured by fuel type, log scale); price by town box plots (top 12 towns); browsable cleaned data table | `clean_data.csv` |
| 2 | 🏆 **Model comparison** | Family filter (multi-select); metric switch (MAE / RMSE / R² / MAPE / train time); bar chart coloured by family; full results table with best values highlighted; bagging vs boosting average; **average error by price band for the main models**, with the explanation of why RF and CatBoost win different metrics and a note on statistical significance | `results.csv`, `test_predictions.csv` |
| 3 | ⚔️ **RF vs XGBoost** | Side-by-side metric cards; actual vs predicted scatter plots for each (log scales, with a perfect-prediction line); average error by price band; explanation of variance vs bias reduction | `results.csv`, `test_predictions.csv` |
| 4 | 🔍 **Feature importance** | Grouped horizontal bar chart of normalised importance for RF and XGBoost, with an explanation of the two importance types | `feature_importance.csv` |
| 5 | 🧮 **Price predictor** | Input form for every feature; live predictions from all 8 saved models (including stacking); average prediction in lakhs and rupees; count and median of similar real ads | `*.pkl`, `meta.json`, `clean_data.csv` |
| 6 | 🧬 **Stacking & hybrid** | **Stacking:** how it works, architecture diagram, learned meta-model weights, comparison with RF and CatBoost. **Hybrid RF + ANN:** explanation of OOF predictions; Graphviz architecture diagram; error comparison of ANN alone vs RF vs hybrid; computed improvement percentage; note on why RF + ARIMA was not applicable | `results.csv` |
| 7 | 📚 **Research** | Comparison table of RF, XGBoost, LightGBM and CatBoost; tabular vs image/text discussion; key findings computed from the live results | `results.csv` |

### 6.5 Live price predictor pipeline

**Step 1: Build the input form from `meta.json`.**
`meta.json` stores, from the training data: a mapping of each brand to its models, the list of towns, the categories of each low-cardinality column, the year range and the exact feature order.

- **Brand** dropdown (defaults to TOYOTA).
- **Model** dropdown, filtered to the selected brand, defaulting to that brand's most common model (for example VITZ for Toyota).
- **Year** slider and **Engine** input default to the median values of real ads for that brand and model, so the form starts on a realistic car.
- Gear, fuel, town, leasing, condition and the four equipment fields have their own dropdowns.

**Step 2: Assemble a one-row DataFrame** with exactly the same columns, names and order as training:

```python
row = pd.DataFrame([{
    "Brand": brand, "Model": model_name, "Town": town,
    "Gear": gear, "Fuel Type": fuel, "Leasing": leasing, "Condition": cond,
    "AIR CONDITION": ..., "POWER STEERING": ..., "POWER MIRROR": ..., "POWER WINDOW": ...,
    "YOM": yom, "Age": 2025 - yom, "Engine (cc)": engine,
}])[meta["features"]]
```

`Age` is derived automatically, exactly as in training.

**Step 3: Predict with each model.**

```python
def predict(models, name, X):
    m = models[name]
    if name == "Hybrid RF + ANN":
        Xp = m["prep"].transform(X)                    # encode + scale
        rf_pred = m["rf"].predict(Xp)                  # base model
        return np.expm1(m["ann"].predict(np.column_stack([Xp, rf_pred])))
    return np.expm1(m.predict(X))                      # pipeline does everything
```

- For the 6 pipeline models, one `predict()` call runs the **whole chain**: target encoding (using training-set category means), one-hot encoding, model prediction.
- For the hybrid, the three saved parts are applied in order: preprocessing, RF, then ANN with the RF prediction appended.
- `np.expm1` converts log price back to lakhs.

**Step 4: Display.** The average of all 8 predictions is shown in lakhs and rupees, with a bar chart of each model's prediction (coloured by family, with an average line). The app also shows how many real ads exist for that exact brand and model and their median price, as a reality check. If there are no ads for that exact model, it warns that predictions are less reliable.

**Handling unseen inputs:** the dropdowns only offer values seen in training, so encoders always receive known categories. Even so, the pipeline is safe: `TargetEncoder` maps unknown categories to the global mean, and `OneHotEncoder(handle_unknown="ignore")` maps them to all zeros.

### 6.6 Deployment

1. Code and artifacts pushed to GitHub (`harshanacz/car-price-ensemble`). The largest file (~27 MB) is well under GitHub's 100 MB per-file limit.
2. Streamlit Community Cloud connected to the repository, main file `app.py`.
3. Streamlit Cloud installs `requirements.txt` and serves the app at https://car-price-ensemble.streamlit.app/.
4. Every `git push` to `main` automatically redeploys the app.

---

## 7. Research analysis: tabular vs unstructured data

### 7.1 Comparing Random Forest and gradient boosting

| | **Random Forest** | **XGBoost** | **LightGBM** | **CatBoost** |
|---|---|---|---|---|
| Ensemble type | Bagging | Gradient boosting | Gradient boosting | Gradient boosting |
| Tree building | Independent, parallel, bootstrap samples, deep trees | Sequential, level-wise | Sequential, leaf-wise, histogram-based | Sequential, symmetric (oblivious) trees |
| Mainly reduces | Variance | Bias | Bias | Bias |
| Key innovation | Feature subsampling to decorrelate trees | 2nd-order gradients + regularised objective | Leaf-wise growth, GOSS, EFB | Ordered boosting, native categorical handling |
| Tuning effort | Low, strong defaults | Medium to high | Medium to high | Low to medium |
| Overfitting risk | Low (more trees never hurts) | Medium, needs regularisation / early stopping | Higher on small data | Low |
| Training speed | Parallel but deep trees | Fast | Fastest | Slower |
| Categorical features | Need encoding | Need encoding (basic native support) | Native support | Best native support |
| Sensitivity to noise | Robust | Moderate | Moderate | Robust |
| **This project** | **MAE 5.39** (best) | MAE 5.87, fastest | MAE 5.88, R² 0.908 | R² 0.912 (best single model) |

**When to choose which:**
- **Random Forest:** a strong first model, little tuning, robust to noise, hard to overfit. Good for quick, reliable baselines and smaller datasets.
- **XGBoost / LightGBM:** usually the most accurate on tabular data **after tuning**, and very fast. LightGBM is ideal for very large datasets.
- **CatBoost:** best when there are many categorical features, and gives strong results with little tuning.

### 7.2 Why tree ensembles dominate tabular data

Tabular data means rows and columns where **each column has its own meaning** (brand, year, engine size). Tree ensembles fit this structure naturally:

1. **Mixed feature types.** Trees handle numbers, categories and binary flags in the same model. There's no need to put them on a common scale.
2. **No scaling needed.** Splits depend only on the order of values, so features measured in cc, years or 0/1 can be used as they are.
3. **Irregular, non-smooth patterns.** Real tabular relationships are often step-like: a car made after 2015 jumps in price, one brand is a different market segment. Trees model these sharp thresholds directly. Research (Grinsztajn et al., 2022, *"Why do tree-based models still outperform deep learning on tabular data?"*) shows that neural networks are biased towards **overly smooth** functions and struggle with such irregular targets.
4. **Robust to uninformative features.** Tabular datasets often include weak features (like `Condition` and `AIR CONDITION` here). Trees simply don't split on them. Neural networks must learn to ignore them, which is harder and needs more data.
5. **Automatic interactions.** A tree path such as "Brand = Toyota → Year > 2015 → Engine > 1,500" captures an interaction between three features without anyone defining it. This is exactly how trees resolved the **year vs engine confounder** in this project.
6. **Not rotation invariant (a good thing here).** Each tabular column is meaningful on its own. Trees respect this, splitting on one original column at a time. Neural networks mix all inputs in every neuron, which helps for images but blurs the meaning of individual columns in tables.
7. **Efficient with modest data.** Tens of thousands of rows are plenty for strong tree ensembles. Neural networks usually need much more data to reach the same level.
8. **Robust to outliers in features.** A threshold split is unaffected by how extreme a value is, only by which side of the threshold it falls.

### 7.3 Why neural networks struggle on tabular data without extensive tuning

This project shows it directly: the **ANN alone had the highest error of all models** (MAE 7.72 vs RF's 5.39).

1. **Smooth inductive bias.** MLPs with ReLU learn smooth functions, while car prices change in steps across brands, models and years.
2. **Many hyperparameters.** Layers, units, learning rate, batch size, regularisation, dropout, early stopping and initialisation all matter, and the results are sensitive to them. Trees work well with defaults.
3. **Preprocessing sensitivity.** Neural networks require careful scaling and encoding. Poor choices hurt performance a lot.
4. **Data hunger.** With 7,628 training rows, a network with thousands of weights can easily overfit, or stop early before learning the full pattern.
5. **Categorical data is awkward.** Neural networks need embeddings or one-hot vectors, and 1,555 car models with many rare ones are hard to learn from limited data. Here the target encoding had to do most of the work.
6. **Optimisation is noisy.** Gradient descent can land in different solutions depending on the random seed. Trees are more deterministic and stable.

Specialised deep tabular architectures (such as TabNet, FT-Transformer and recent tabular foundation models) narrow this gap, but they need significantly more effort, tuning and computation. Well-tuned gradient boosting remains the standard strong baseline for tabular problems.

### 7.4 Why tree ensembles fail on images and text (unstructured data)

On unstructured data, the situation reverses.

**Images:**
- An image is a grid of pixels. A single pixel means nothing by itself. Meaning comes from **spatial patterns**: edges, textures, shapes, objects.
- A tree treats each pixel as an independent column and splits on one pixel at a time ("is pixel (120, 45) brighter than 0.6?"). It has no idea that neighbouring pixels are related.
- A small shift (the object moves 5 pixels) changes all the pixel values, so trees cannot recognise the same object in a different position.
- **Convolutional neural networks (CNNs)** and vision transformers are built for this: shared filters scan the whole image, detecting local patterns regardless of position, and build them up into higher-level features (edges → parts → objects).

**Text:**
- Meaning depends on **word order, context and semantics**. "Not good" and "good" differ by one word. "Bank" means different things in different sentences.
- Trees need text converted to fixed columns (bag-of-words or TF-IDF). That loses word order and context, and creates huge, sparse feature spaces.
- **Transformers** (BERT, GPT-style models) model context and long-range relationships between words with self-attention, and benefit from pre-training on huge text corpora.

**Where trees still help with unstructured data:** a common practical pattern is to use a deep network to **extract features** (for example CNN image embeddings or sentence embeddings), then train a gradient boosting model on those features combined with regular tabular columns. For example, to price a car from both its photos and its specifications, a CNN could summarise the photos into a vector, and XGBoost could combine that vector with year, brand and engine size.

### 7.5 Summary

| Data type | Best approach | Why |
|---|---|---|
| **Tabular** (this project) | Tree ensembles (RF, XGBoost, LightGBM, CatBoost) | Mixed feature types, sharp thresholds, meaningful columns, modest data size |
| **Images** | CNNs, vision transformers | Spatial structure, local patterns, translation invariance |
| **Text** | Transformers | Word order, context, semantics, pre-training |
| **Mixed** (tabular + images/text) | Deep feature extraction + gradient boosting | Combines the strengths of both |

---

## 8. Limitations and future work

### 8.1 Limitations

1. **No hyperparameter tuning.** All models used reasonable fixed settings. Gradient boosting usually benefits most from tuning, so the RF vs boosting comparison favours RF somewhat.
2. **Single train/test split.** Results come from one 80/20 split. A different split could reorder close models (for example XGBoost vs LightGBM, which differ by only 0.01 lakhs of MAE). K-fold cross-validation would give more reliable averages and confidence intervals.
3. **Asking price, not sale price.** The target is the price in the advertisement. Actual sale prices are often lower after negotiation.
4. **Missing important attributes.** Real mileage (synthetic in this data), service history, accident history, trim level, colour, number of owners and registration status are not available. These limit accuracy, especially for expensive cars.
5. **Short time window.** All ads are from December 2024 to February 2025. The Sri Lankan car market changes quickly with import policy (vehicle imports reopened in 2025), so the model may go out of date.
6. **Few luxury cars.** Only about 9% of test cars cost over 100 lakhs, and errors there are about 20 lakhs on average.
7. **Built-in feature importance is biased.** As shown in 5.5, gain/impurity importance over-credits Gear. Permutation importance or SHAP values are more reliable.
8. **Redundant feature.** `Age` duplicates `YOM`. It doesn't hurt accuracy but splits importance.
9. **LightGBM row subsampling not active.** `subsample=0.8` had no effect without `subsample_freq > 0`.

### 8.2 Future work

- Hyperparameter tuning with `RandomizedSearchCV` or Optuna, plus **early stopping** for boosting models.
- **5-fold cross-validation** for all models, reporting mean ± standard deviation.
- Use **CatBoost's native categorical handling** instead of shared target encoding.
- **SHAP values** for model explanations, including per-prediction explanations in the dashboard.
- Try **CatBoost or LightGBM as the stacking meta-model**, or pass the original features through (`passthrough=True`), to see if stacking can beat RF on MAE as well.
- **Prediction intervals** (for example quantile regression) so the dashboard shows a price range, not one number.
- Collect **multi-period data** to enable time-series analysis (and the RF + ARIMA hybrid).
- Collect **real mileage** and **car photos** for a multimodal model (Section 7.4).

---

## 9. How to reproduce

### 9.1 Train the models (Google Colab)

1. Open https://colab.research.google.com → **File → Upload notebook** → `ensemble_car_price.ipynb`.
2. **Runtime → Run all**.
3. When prompted, upload `car_price_dataset.csv` (from Kaggle).
4. Training takes about one minute on CPU. At the end, `artifacts.zip` downloads automatically.

### 9.2 Run the dashboard locally

```bash
git clone https://github.com/harshanacz/car-price-ensemble.git
cd car-price-ensemble
# (If retrained: unzip artifacts.zip so the artifacts/ folder sits next to app.py)
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

### 9.3 Requirements

```
streamlit
scikit-learn==1.6.1
pandas
numpy
joblib
plotly
xgboost
lightgbm
catboost
```

`scikit-learn` must match the version used for training (1.6.1 in Colab at the time of training). If the notebook is re-run in a Colab environment with a newer scikit-learn, update the pin in `requirements.txt` to that version.

### 9.4 Reproducibility

- `random_state=42` (or `random_seed=42`) is set for the split, every model and the encoders.
- Small differences may still appear between runs because of multithreading and library versions, but rankings and conclusions should hold.

---

*End of documentation.*
