import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import yfinance as yf
import datetime

# Define the function to download data and cache it
@st.cache_data
def get_historical_data(symbol, start):
    data = yf.download(symbol, start=start)
    data.reset_index(drop=False, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Title of the app
st.title('Peak and Trough Detection')

# Define default values
symbol = 'SPY'
start = datetime.datetime(1995, 1, 1)

# Select box for symbol selection
sidebar = st.sidebar
symbol = sidebar.selectbox('Select a symbol', ['SPY', 'BTC-USD', 'ETH-USD','SOL-USD','TSLA', 'GOOG', 'MSFT','QQQ','AMZN','IBM','INTC'])

# Download historical data and cache it
historical_data = get_historical_data(symbol, start)
log_data = np.log(historical_data['Close'])

# Sliders for distance and prominence
distance = sidebar.slider('Distance', 10, 1000, 100)
prominence = sidebar.slider('Prominence', 0.1, 1.0, 0.2)

# Find peaks and troughs
peaks = find_peaks(log_data, distance=distance, prominence=prominence)[0]
troughs = find_peaks(-log_data, distance=distance, prominence=prominence)[0]

# Copy original peaks and troughs
original_peaks = peaks.copy()
original_troughs = troughs.copy()

# Filter peaks to ensure each peak is higher than all previous peaks
max_peak_value = -np.inf  # Initialize to negative infinity
retained_peaks = []
retained_troughs = []

for i,peak in enumerate(peaks):
    previous_peaks = peaks[:i]
    current_peak_date = historical_data['Date'][peak]
    current_peak_value = log_data[peak]
    #immediate_prices = historical_data['Close'][current_peak_date:(current_peak_date + datetime.timedelta(days=distance))]
    immediate_prices = historical_data['Close'].iloc[peak+1:peak+distance]
    next_max_price = immediate_prices.max()
    if current_peak_value < np.log(next_max_price):
        continue
    else:
        if any(log_data[peak] < log_data[p] for p in previous_peaks):
            continue
        else:
            retained_peaks.append(peak)

for i, trough in enumerate(troughs):
    future_troughs = troughs[i+1:]
    if any(log_data[trough] > log_data[t] for t in future_troughs):
        continue
    retained_troughs.append(trough)

retained_peak_prices = np.array(log_data[retained_peaks])
retained_trough_prices = np.array(log_data[retained_troughs])


st.write(f'peaks: {retained_peaks}')
st.write(f'peak prices: {retained_peak_prices}')

st.write(f'troughs: {retained_troughs}')
st.write(f'trough prices: {retained_trough_prices}')

# st.write(f'second peak {retained_peak_prices[1]}')
# st.write(f'first trough {retained_trough_prices[0]}')
# Determine the range of prices and divide into four segments
divisions = sidebar.slider('Divisions', 2, 10, 4)
peak_price_range = np.linspace(retained_peak_prices.min(), retained_peak_prices.max(), num=divisions)
trough_price_range = np.linspace(retained_trough_prices.min(), retained_trough_prices.max(), num=divisions)



def segment_extremes(indices, prices, price_ranges, keep='lowest'):
    # Determine which segment each price falls into
    segments = np.digitize(prices, price_ranges) - 1

    # We will map each segment to its indices and then choose the appropriate one
    segment_dict = {}
    for idx, seg in enumerate(segments):
        if seg not in segment_dict:
            segment_dict[seg] = []
        segment_dict[seg].append(indices[idx])

    # For each segment, keep the lowest or highest index
    retained_indices = []
    for segment in sorted(segment_dict.keys()):
        if keep == 'lowest':
            retained_indices.append(min(segment_dict[segment]))
        elif keep == 'highest':
            retained_indices.append(max(segment_dict[segment]))

    return retained_indices

# Get the retained peaks and troughs
retained_peaks = segment_extremes(retained_peaks, retained_peak_prices, peak_price_range, keep='lowest')
retained_troughs = segment_extremes(retained_troughs, retained_trough_prices, trough_price_range, keep='highest')

st.write("Retained Peak Indices:", retained_peaks)
st.write("Retained Trough Indices:", retained_troughs)









# Plot the data
fig, ax = plt.subplots()
ax.plot(historical_data.index, log_data, label='Log Price')
ax.plot(retained_peaks, log_data[retained_peaks], "^", color='green', alpha=0.8, label='New Peaks')
ax.plot(retained_troughs, log_data[retained_troughs], "^", color='red', alpha=0.8, label='New Troughs')
ax.plot(original_peaks, log_data[original_peaks], ".", color='green', alpha=0.5, label='Original Peaks')
ax.plot(original_troughs, log_data[original_troughs], ".", color='red', alpha=0.5, label='Original Troughs')
ax.set_title(f'{symbol} Log Price')
ax.set_xlabel('Date')
ax.set_ylabel('Log Price')
ax.set_xticks(historical_data.index[::400])
ax.set_xticklabels(historical_data['Date'].dt.year[::400])
ax.legend()

st.pyplot(fig)
