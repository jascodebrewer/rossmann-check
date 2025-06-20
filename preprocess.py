import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import os

# Load data
df_train = pd.read_csv('train.csv', dtype={'StateHoliday': str})

# --- SARIMAX ---
# Data preparation
store_df = df_train[df_train['Store'] == 1].copy()
store_df['Date'] = pd.to_datetime(store_df['Date'])
store_df = store_df.sort_values('Date')
store_df.set_index('Date', inplace=True)
y_sarimax = store_df['Sales']
exog = store_df[['Promo', 'SchoolHoliday']].astype(int)

# Train SARIMAX 
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 7)  # weekly seasonality
sarimax_model = SARIMAX(y_sarimax, exog=exog, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
sarimax_results = sarimax_model.fit(disp=False)

# Create a directory to store the model if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save SARIMAX model
pickle.dump(sarimax_results, open('models/sarimax_model.pkl', 'wb'))

print("SARIMAX model trained and saved!")