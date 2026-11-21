# Ensemble Learning Dashboard: Sri Lankan Used Car Prices

Compares bagging (Random Forest, Bagging), boosting (AdaBoost, XGBoost, LightGBM, CatBoost) and a hybrid RF + ANN model for predicting used car prices in Sri Lanka.

**Dataset:** [SriLankan Vehicle (Car) Price Dataset](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset) (ikman.lk & riyasewana.lk)

## Files
- `ensemble_car_price.ipynb`: training notebook (run in Google Colab)
- `app.py`: Streamlit dashboard
- `artifacts/`: trained models and results produced by the notebook

## Run the dashboard
```bash
pip install -r requirements.txt
streamlit run app.py
```
Note: `scikit-learn==1.6.1` must match the version the models were trained with in Colab.
