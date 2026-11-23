# Sri Lankan Used Car Price Prediction with Ensemble Learning

Predicts used car prices in Sri Lanka and compares bagging, boosting, stacking and a hybrid model.

**Live dashboard:** https://car-price-ensemble.streamlit.app/
**Dataset:** [SriLankan Vehicle (Car) Price Dataset](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset) (ikman.lk & riyasewana.lk)

## Models
| Model | Type | MAE (lakhs) | R² |
|---|---|---|---|
| Random Forest | Bagging | **5.39** | 0.901 |
| Stacking (RF + XGBoost → Ridge) | Stacking | 5.44 | **0.904** |
| Hybrid RF + ANN | Hybrid | 5.69 | 0.900 |
| XGBoost | Boosting | 5.87 | 0.889 |
| Decision Tree | Baseline | 6.96 | 0.870 |
| ANN | Reference | 7.72 | 0.835 |

## Key findings
- All ensembles beat the single decision tree.
- Random Forest had the lowest average error; stacking had the best R².
- XGBoost trained about 8x faster than Random Forest.
- The hybrid RF + ANN improved the neural network by 26% but did not beat Random Forest.
- Mileage was dropped: it was calculated from the year, not real readings.

## Files
- `ensemble_car_price.ipynb`: training notebook (Google Colab)
- `app.py`: Streamlit dashboard
- `artifacts/`: trained models and results

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
`scikit-learn==1.6.1` must match the version used for training.
