import csv
from datetime import timedelta
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import pickle
import os
import requests  # Import the requests library
from io import StringIO  # To read string as file
from io import BytesIO

# Cloud storage URL for train.csv 
#TRAIN_CSV_URL = "YOUR_PUBLIC_LINK_TO_TRAIN_CSV"
TRAIN_CSV_URL = "https://drive.google.com/uc?export=download&id=18JnM4YQl9covb_g43znFSlZOoOmWWCyL"

# Google Drive URL for sarimax model 
SARIMAX_MODEL_URL = "https://drive.google.com/uc?export=download&id=1wHJ_NQSfkCGe_KCIUAZB8QMdtcDadOBV"  

# Function to load the SARIMAX model
@st.cache_resource  # Cache to load only once
def load_model(url):
    """Downloads the model from the given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Load the model from the downloaded content
        model = pickle.load(BytesIO(response.content))
        return model
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading the model: {e}")
        return None
    except pickle.PickleError as e:
        st.error(f"Error unpickling the model: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Function to load data with caching
@st.cache_data # Cache the dataframe
def load_data(url):
    """Downloads the data from the given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        csv_data = StringIO(response.text)  # Treat response content as a file

        # Attempt to read the data using pandas, trying different options
        try:
            df = pd.read_csv(csv_data, sep=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, encoding='utf-8', dtype={'StateHoliday': str})
            return df
        except Exception as e:
            st.write(f"Attempt 1 failed: {e}") # Helpful debug
            try:
               df = pd.read_csv(csv_data, sep=';', quotechar='"', quoting=csv.QUOTE_MINIMAL, encoding='utf-8', dtype={'StateHoliday': str})
               return df
            except Exception as e2:
                st.write(f"Attempt 2 failed: {e2}") # Helpful debug
                st.error(f"Could not read the data. Check the file and URL.")
                return None

    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading the data: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Function for SARIMAX prediction
def predict_sarimax(model, days_to_forecast, df_train):
    store_df = df_train[df_train['Store'] == 1].copy()
    store_df['Date'] = pd.to_datetime(store_df['Date'])
    store_df = store_df.sort_values('Date')
    store_df.set_index('Date', inplace=True)

    # Create a DataFrame with future dates
    last_date = store_df.index.max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_forecast + 1)]  # Forecast the next 30 days
    future_df = pd.DataFrame(index=future_dates)

    # Add exogenous variables for the future dates
    future_df['Promo'] = [1] * days_to_forecast  # You might want to get this from an external source
    future_df['SchoolHoliday'] = [0] * days_to_forecast

    # Ensure they are the correct type
    exog_future = future_df[['Promo', 'SchoolHoliday']].astype(int)

    # Generate the forecast
    forecast = model.get_forecast(steps=days_to_forecast, exog=exog_future)

    # Extract the forecast values
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Prepare output
    forecast_df = pd.DataFrame({'Forecast': forecast_mean, 'Lower': conf_int.iloc[:, 0], 'Upper': conf_int.iloc[:, 1]})
    forecast_df.index.name = 'Date'  # Set the index name for clarity
    return forecast_df

# Function to plot forecast
def plot_forecast(forecast_df, model_type):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(forecast_df.index, forecast_df['Forecast'], label=f'{model_type} Forecast', color='red')
    ax.fill_between(forecast_df.index, forecast_df['Lower'], forecast_df['Upper'], color='pink', alpha=0.3)

    ax.set_title(f'{model_type} Sales Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# Load data
# df_train = pd.read_csv('train.csv', dtype={'StateHoliday': str}) # Local load
df_train = load_data(TRAIN_CSV_URL)  # Load from the cloud

# Check if data loading was successful
if df_train is None:
    st.stop()  # Stop the app if data loading failed

# Load SARIMAX model
try:
    sarimax_model = load_model(SARIMAX_MODEL_URL) # From Cloud now!
except Exception as e:
    st.error(f"Model file not found. {e} Make sure to train the SARIMAX model and upload it to Google Drive!")
    st.stop()

# --- Streamlit App ---
st.title("Retail Sales Forecasting App")

st.write("Generate a sales forecast using the SARIMAX model.")

# Number of days to forecast
days_to_forecast = st.slider("Select the number of days to forecast:", 1, 30, 7)

# SARIMAX prediction button
if st.button("Generate SARIMAX Forecast"):
    with st.spinner('Generating SARIMAX forecast...'):
        sarimax_forecast = predict_sarimax(sarimax_model, days_to_forecast, df_train)
        st.subheader("SARIMAX Forecast")
        st.dataframe(sarimax_forecast)
        plot_forecast(sarimax_forecast, 'SARIMAX')
