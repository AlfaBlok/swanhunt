# %%
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import yfinance as yf
import datetime
from matplotlib.animation import FuncAnimation, PillowWriter

def get_peaks_troughs(log_data, distance, prominence, divisions, min_date_model_index=None):
    # Find peaks and troughs
    # st.dataframe(log_data)
    print('Barlam')
    print('Distance:', distance)
    peaks = find_peaks(log_data, distance=distance, prominence=prominence)[0]
    troughs = find_peaks(-log_data, distance=distance, prominence=prominence)[0]

    # Copy original peaks and troughs
    original_peaks = peaks.copy()
    original_troughs = troughs.copy()
    

    # if any peaks or troughs are found, we filter them to ensure they are above min_date_model_index
    if min_date_model_index is not None:
        peaks = [p for p in peaks if p >= min_date_model_index]
        troughs = [t for t in troughs if t >= min_date_model_index]


    # st.text(f'Peaks: {peaks}')
    # st.text(f'Troughs: {troughs}')

    # Filter peaks to ensure each peak is higher than all previous peaks
    max_peak_value = -np.inf  # Initialize to negative infinity
    retained_peaks = []
    retained_troughs = []

    for i,peak in enumerate(peaks):
        previous_peaks = peaks[:i]
        current_peak_date = historical_data['date'][peak]
        current_peak_value = log_data[peak]
        #immediate_prices = historical_data['Close'][current_peak_date:(current_peak_date + datetime.timedelta(days=distance))]
        immediate_prices = historical_data['close_price'].iloc[peak+1:peak+distance]
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

    # do the final filtering pass
    if len(retained_peaks) < 2:
        return [], []
    
    peak_price_range = np.linspace(retained_peak_prices.min(), retained_peak_prices.max(), num=divisions)
    trough_price_range = np.linspace(retained_trough_prices.min(), retained_trough_prices.max(), num=divisions)

    retained_peaks = segment_extremes(retained_peaks, retained_peak_prices, peak_price_range, keep='lowest')
    retained_troughs = segment_extremes(retained_troughs, retained_trough_prices, trough_price_range, keep='highest')

    

    return retained_peaks, retained_troughs

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

@st.cache_data
def load_data(symbol, start_date=None, final_year_model_prediction=None):

    if symbol == 'BTC-USD':
        historical_data_btc = pd.read_csv('historical_btc.csv')
        historical_data_btc.columns = ['date', 'close_price']
        # st.dataframe(historical_data_btc)
        historical_data_btc['date'] = pd.to_datetime(historical_data_btc['date'])
        last_date = historical_data_btc['date'].max()
        new_data_btc = yf.download(symbol, start=start_date) if start_date else yf.download(symbol, start='1970-01-01')
        new_data_btc = new_data_btc.reset_index()
        new_data_btc = new_data_btc[['Date', 'Close']]
        new_data_btc.columns = ['date', 'close_price']
        full_data = pd.concat([historical_data_btc, new_data_btc], axis=0)
        full_data = full_data.sort_values('date')
        full_data = full_data[full_data['date'] >= start_date] if start_date else full_data
        full_data['date'] = pd.to_datetime(full_data['date'])
        
        full_data = full_data.reset_index(drop=False)
        max_date = full_data['date'].max()
        # print('Max date loaded:', max_date)
    else:
        new_data_btc = yf.download(symbol, start=start_date)
        new_data_btc = new_data_btc.reset_index()
        new_data_btc = new_data_btc[['Date', 'Close']]
        new_data_btc.columns = ['date', 'close_price']
        full_data = new_data_btc
        full_data = full_data.sort_values('date')
        full_data = full_data.reset_index(drop=False)
        full_data['date'] = pd.to_datetime(full_data['date'])
        max_date = full_data['date'].max()
        print('Min date loaded:', full_data['date'].min())
        print('Max date loaded:', max_date)

    if symbol == 'SPY':
        full_data['close_price'] = full_data['close_price'] * 10   

    # st.dataframe(full_data)
    full_data = full_data.drop_duplicates('date')
    full_data = full_data[['date', 'close_price']]

    # st.dataframe(full_data)
    full_data = full_data.dropna(subset=['close_price'])
    full_data = full_data.reset_index(drop=False)
    # st.dataframe(full_data)

    return full_data

def get_frame_plot(full_data, clean_peaks, clean_troughs, peak_model, trough_model, average_model_prices,min_x, max_x, min_y, max_y, symbol, peak_model_params, trough_model_params, average_model_params, peak_r2, trough_r2, current_date=None, alpha=1, transparency = False, shaded = False, ax=None):
    # fig, axs = plt.subplots(1, 3, figsize=(14, 7.3))
    print('Here alpha:', alpha)

    if ax is None:
        fig, axs = plt.subplots(1, 3, figsize=(16, 11))
    else:
        fig, axs = plt.gcf(), ax

    if transparency == False:
        for ax_i in axs:
            ax_i.clear()

    if min_x is not None:
        min_x = pd.to_datetime(min_x)
        if min_x < full_data['date'].min():
            min_x = full_data['date'].min()

    if max_x is not None:
        max_x = pd.to_datetime(max_x)
        if max_x > full_data['date'].max():
            max_x = full_data['date'].max()


    axs[0].plot(full_data['date'], full_data['close_price'], label='BTC Price', color='blue')
    axs[0].scatter(full_data['date'][clean_peaks], full_data['close_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    axs[0].scatter(full_data['date'][clean_troughs], full_data['close_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[0].plot(full_data['date'], peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[0].plot(full_data['date'], trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[0].plot(full_data['date'], average_model_prices, color='orange', label='Model Average', alpha=alpha)


    axs[0].set_title(f'{symbol} price')
    # axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    # axs[0].legend()
    # full_data['date'].min() if min_x is None else pd.datatime(min_x)
    min_x = full_data['date'].min() if min_x is None else min_x
    max_x = full_data['date'].max() if max_x is None else max_x

    min_x_index = full_data[full_data['date'] >= min_x].index[0]
    max_x_index = full_data[full_data['date'] <= max_x].index[-1]

        
    # Calculate x-ticks
    num_ticks = 5
    x_ticks = pd.date_range(start=min_x, end=max_x, periods=num_ticks).to_pydatetime()
    axs[0].set_xticks(x_ticks)
    axs[0].set_xticklabels([tick.year for tick in x_ticks])



    axs[0].set_xlim(min_x, max_x)
    # axs[0].set_ylim(full_data['close_price'].min(), full_data['close_price'].max())
    # axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].yaxis.set_major_formatter('${:,.0f}'.format)
    # axs[0].legend()
    axs[0].set_ylim([0, max_y])  
    
    # we show only 5 ticks on the x axis


    
    # Log scale
    axs[1].plot(full_data['date'], full_data['close_price'], label='BTC Price', color='blue')
    axs[1].scatter(full_data['date'][clean_peaks], full_data['close_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    axs[1].scatter(full_data['date'][clean_troughs], full_data['close_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[1].plot(full_data['date'], peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[1].plot(full_data['date'], trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[1].plot(full_data['date'], average_model_prices, color='orange', label='Model Average', alpha=alpha)
    axs[1].set_xlim(full_data['date'].min() if min_x is None else min_x, full_data['date'].max() if max_x is None else max_x)
    # axs[1].set_ylim(full_data['close_price'].min(), full_data['close_price'].max())
    axs[1].set_title(f'{symbol} price (log)')
    # axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Price')
    axs[1].set_yscale('log')
    axs[1].set_ylim([min_y, max_y])  # Set log y range
    axs[1].yaxis.set_major_formatter('${:,.0f}'.format)
    axs[1].set_xticks(x_ticks)
    axs[1].set_xticklabels([tick.year for tick in x_ticks])

    # # Log-log scale
    axs[2].plot(full_data.index, full_data['close_price'], label='BTC Price', color='blue')
    if len(clean_peaks) > 0:
        axs[2].scatter(full_data.index[clean_peaks], full_data['close_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    if len(clean_troughs) > 0:
        axs[2].scatter(full_data.index[clean_troughs], full_data['close_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[2].plot(full_data.index, peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[2].plot(full_data.index, trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[2].plot(full_data.index, average_model_prices, color='orange', label='Model Average', alpha=alpha)
    axs[2].set_title(f'{symbol} price (log-log)')
    # axs[2].set_xlabel('Date')
    axs[2].set_ylabel('Price')
    axs[2].set_yscale('log')
    axs[2].set_xscale('log')
    if max_x is None:
        max_index = len(full_data) - 1
    else:
        max_index = full_data[full_data['date'] <= max_x].index[-1]
    if max_index < 1000:
        axs[2].set_xticks([1, 10, 100, 1000])
    else:
        axs[2].set_xticks([1, 10, 100, 1000, max_index])
    axs[2].set_xticklabels([full_data['date'][1].year, full_data['date'][10].year, full_data['date'][100].year, full_data['date'][1000].year, full_data['date'][max_index].year])
    # we get 
    
    # axs[2].set_xticks(x_ticks)
    # axs[2].set_xticklabels([tick.year for tick in x_ticks])


    axs[2].yaxis.set_major_formatter('${:,.0f}'.format)
    axs[2].set_ylim([min_y, max_y])
    if min_x is not None:
        # we look for the index of the min_x date
        min_x_index = full_data[full_data['date'] >= min_x].index[0]
        axs[2].set_xlim([min_x_index, max_index])  # Set min for the third axis




    subtitle = f"{symbol} Current Date: {current_date.strftime('%Y-%m-%d') if current_date else 'N/A'}\n"
    if peak_model_params is not None:
        subtitle += f"Peak Model Params: a={peak_model_params[0]:.4f}, b={peak_model_params[1]:.4f}, R²={peak_r2:.4f}\n"
    else:
        subtitle += "Peak Model Params: waiting for peaks\n"
    if trough_model_params is not None:
        subtitle += f"Trough Model Params: a={trough_model_params[0]:.4f}, b={trough_model_params[1]:.4f}, R²={trough_r2:.4f}\n"
    else:
        subtitle += "Trough Model Params: waiting for troughs\n"

    if average_model_params is not None:
        subtitle += f"Mid Model Params: a={average_model_params[0]:.4f}, b={average_model_params[1]:.4f}\n"
    else:
        subtitle += "Mid Model Params: waiting for peaks and troughs\n"
    
    if shaded == True:
        # Adding shaded area between peak_model and trough_model
        axs[0].fill_between(full_data['date'], peak_model, trough_model, color='yellow', alpha=alpha, label='Power Law Model') 
        axs[1].fill_between(full_data['date'], peak_model, trough_model, color='yellow', alpha=alpha, label='Power Law Model')
        axs[2].fill_between(full_data.index, peak_model, trough_model, color='yellow', alpha=alpha, label='Power Law Model')



    fig.suptitle(subtitle)

    plt.subplots_adjust(top=0.8, wspace=alpha)    

    return fig, axs

                        # full_data, clean_peaks, clean_troughs, peak_model_prices, trough_model_prices, average_model_prices, min_x, max_x, min_y, max_y, symbol, peak_model_params, trough_model_params, average_model_params, peak_r2, trough_r2, current_date, alpha, transparency, shaded, prediction_2030_peak, prediction_2040_peak
def get_frame_radarplot(full_data, clean_peaks, clean_troughs, peak_model, trough_model, average_model_prices, min_x, max_x, min_y, max_y, symbol, peak_model_params, trough_model_params, average_model_params, peak_r2, trough_r2, current_date=None, alpha=1, transparency=False, shaded=False, max_2030_peak=None, max_2040_peak=None, ax=None):
    if ax is None:
        fig, axs = plt.subplots(2, 3, figsize=(16, 11))
    else:
        fig, axs = plt.gcf(), ax

    if not transparency:
        for row in axs:
            for ax_i in row:
                ax_i.clear()

    if min_x is not None:
        min_x = pd.to_datetime(min_x)
        if min_x < full_data['date'].min():
            min_x = full_data['date'].min()

    if max_x is not None:
        max_x = pd.to_datetime(max_x)
        if max_x > full_data['date'].max():
            max_x = full_data['date'].max()
    # print('Here is full data:', full_data)
    axs[0, 0].plot(full_data['date'], full_data['close_price'], label='BTC Price', color='blue')
    axs[0, 0].scatter(full_data['date'][clean_peaks], full_data['close_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    axs[0, 0].scatter(full_data['date'][clean_troughs], full_data['close_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[0, 0].plot(full_data['date'], peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[0, 0].plot(full_data['date'], trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[0, 0].plot(full_data['date'], average_model_prices, color='orange', label='Model Average', alpha=alpha)

    axs[0, 0].set_title(f'{symbol} price')
    # axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Price')
    axs[0, 0].set_xlim(min_x, max_x)
    axs[0, 0].set_ylim([0, max_y])
    axs[0, 0].yaxis.set_major_formatter('${:,.0f}'.format)

    num_ticks = 5
    x_ticks = pd.date_range(start=min_x, end=max_x, periods=num_ticks).to_pydatetime()
    axs[0, 0].set_xticks(x_ticks)
    axs[0, 0].set_xticklabels([tick.year for tick in x_ticks])

    axs[0, 1].plot(full_data['date'], full_data['close_price'], label='BTC Price', color='blue')
    axs[0, 1].scatter(full_data['date'][clean_peaks], full_data['close_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    axs[0, 1].scatter(full_data['date'][clean_troughs], full_data['close_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[0, 1].plot(full_data['date'], peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[0, 1].plot(full_data['date'], trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[0, 1].plot(full_data['date'], average_model_prices, color='orange', label='Model Average', alpha=alpha)
    axs[0, 1].set_xlim(min_x, max_x)
    axs[0, 1].set_title(f'{symbol} price (log)')
    # axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Price')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_ylim([min_y, max_y])
    axs[0, 1].yaxis.set_major_formatter('${:,.0f}'.format)
    axs[0, 1].set_xticks(x_ticks)
    axs[0, 1].set_xticklabels([tick.year for tick in x_ticks])

    axs[0, 2].plot(full_data.index, full_data['close_price'], label='BTC Price', color='blue')
    if len(clean_peaks) > 0:
        axs[0, 2].scatter(full_data.index[clean_peaks], full_data['close_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    if len(clean_troughs) > 0:
        axs[0, 2].scatter(full_data.index[clean_troughs], full_data['close_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[0, 2].plot(full_data.index, peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[0, 2].plot(full_data.index, trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[0, 2].plot(full_data.index, average_model_prices, color='orange', label='Model Average', alpha=alpha)
    axs[0, 2].set_title(f'{symbol} price (log-log)')
    # axs[0, 2].set_xlabel('Date')
    axs[0, 2].set_ylabel('Price')
    axs[0, 2].set_yscale('log')
    axs[0, 2].set_xscale('log')
    axs[0, 2].set_ylim([min_y, max_y])
    axs[0, 2].yaxis.set_major_formatter('${:,.0f}'.format)

    if min_x is not None:
        min_x_index = full_data[full_data['date'] >= min_x].index[0]
    if max_x is not None:
        max_x_index = full_data[full_data['date'] <= max_x].index[-1]
        axs[0, 2].set_xlim([min_x_index, max_x_index])

    # we label the x axis with the year of the date
    if max_x_index < 1000:
        axs[0, 2].set_xticks([1, 10, 100, 1000])
    else:
        axs[0, 2].set_xticks([1, 10, 100, 1000, max_x_index])
    axs[0, 2].set_xticklabels([full_data['date'][1].year, full_data['date'][10].year, full_data['date'][100].year, full_data['date'][1000].year, full_data['date'][max_x_index].year])

    # Radar plot
    if average_model_prices is not None:
        # st.dataframe(full_data)
        date_index_2030 = full_data[full_data['date'] == pd.to_datetime('2030-01-01')].index[0]
        prediction_2030_peak = peak_model[date_index_2030]
        prediction_2030_trough = trough_model[date_index_2030]
        prediction_2030_mid = average_model_prices[date_index_2030]
        try:
            current_date_index = full_data[full_data['date'] == current_date].index[0]
        except IndexError:
            current_date_index = None
            print('Current date not in data')
        try:
            mid_r2 = (peak_r2 + trough_r2) / 2
        except:
            mid_r2 = np.nan


    axs[1, 0].scatter([peak_r2], [prediction_2030_peak], color='red', label='Peaks', alpha=alpha)
    axs[1, 0].scatter([trough_r2], [prediction_2030_trough], color='green', label='Troughs', alpha=alpha)
    axs[1, 0].scatter(mid_r2, [prediction_2030_mid], color='orange', label='Mid', alpha=alpha)

    axs[1, 0].set_title('Radar plot')
    axs[1, 0].set_xlabel('R²')
    axs[1, 0].set_ylabel('Price in 2030')
    axs[1, 0].set_xlim([-1, 1.1])
    max_y_2030 = max_2030_peak*1.1
    axs[1, 0].set_ylim([0, max_y_2030])
    axs[1, 0].yaxis.set_major_formatter('${:,.0f}'.format)

    # we create an array that is null for every date except for current_date, and in that date we put the 2030 prediction
    # we create a scatter plot with dates on x and price predictions on y. It will be a single dot in the current date
    peak_price_predictions = np.full(len(full_data), np.nan)
    trough_price_predictions = np.full(len(full_data), np.nan)
    mid_price_predictions = np.full(len(full_data), np.nan)

    peak_price_predictions_2040 = np.full(len(full_data), np.nan)
    trough_price_predictions_2040 = np.full(len(full_data), np.nan)
    mid_price_predictions_2040 = np.full(len(full_data), np.nan)
    
    
    if current_date_index is not None and full_data.loc[current_date_index, 'date'].strftime("%Y-%m-%d") <= datetime.datetime.now().strftime("%Y-%m-%d"):
        print('!!!!!!! PREDICTION 2030 !!!!!!!!!!!!!')
        print('Current date:', current_date)
        print('Current date index:', current_date_index)
        print('2030 prediction peak:', prediction_2030_peak)
        print('2030 prediction trough:', prediction_2030_trough)
        print('2030 prediction mid:', prediction_2030_mid)
        current_date_index = full_data[full_data['date'] == current_date].index[0]
        date_index_2030 = full_data[full_data['date'] == pd.to_datetime('2030-01-01')].index[0]
        prediction_2030_mid = average_model_prices[date_index_2030]
        prediction_2030_peak = peak_model[date_index_2030]
        prediction_2030_trough = trough_model[date_index_2030]
        if current_date_index is not None:
            peak_price_predictions[current_date_index] = prediction_2030_peak
            trough_price_predictions[current_date_index] = prediction_2030_trough
            mid_price_predictions[current_date_index] = prediction_2030_mid
        else:
            print('Current date not in data')

        print('!!!!! PREDICTION 2040 !!!!!!')
        date_index_2040 = full_data[full_data['date'] == pd.to_datetime('2040-01-01')].index[0]
        prediction_2040_peak = peak_model[date_index_2040]
        prediction_2040_trough = trough_model[date_index_2040]
        prediction_2040_mid = average_model_prices[date_index_2040]
        print('2040 prediction peak:', prediction_2040_peak)
        print('2040 prediction trough:', prediction_2040_trough)
        print('2040 prediction mid:', prediction_2040_mid)
        if current_date_index is not None:
            peak_price_predictions_2040[current_date_index] = prediction_2040_peak
            trough_price_predictions_2040[current_date_index] = prediction_2040_trough
            mid_price_predictions_2040[current_date_index] = prediction_2040_mid





        # we calculate the mid price prediction for 1st january 2030


    axs[1, 1].scatter(full_data['date'], peak_price_predictions, color='red', label='2030 Prediction', alpha=1)
    axs[1, 1].scatter(full_data['date'], trough_price_predictions, color='green', label='2030 Prediction', alpha=1)
    axs[1, 1].scatter(full_data['date'], mid_price_predictions, color='orange', label='2030 Prediction', alpha=1)

    axs[1, 1].set_title('2030 Prediction')
    # axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Price in 2030')
    # we define max x as the last date where price_close is not null
    max_x_date = pd.to_datetime('2030-01-01')
    print('XXXXXXXXXXXXXXXXXXXXX  Max x date:', max_x_date)
    # we add 3 months to the max_x_date
    # max_x_date = max_x_date + pd.DateOffset(months=12)
    axs[1, 1].set_xlim(min_x, max_x_date)    
    max_y_2030 = max_2030_peak*2
    axs[1, 1].set_ylim([0, max_y_2030])
    axs[1, 1].yaxis.set_major_formatter('${:,.0f}'.format)
    x_ticks = pd.date_range(start=min_x, end=max_x_date, periods=num_ticks).to_pydatetime()
    axs[1, 1].set_xticks(x_ticks)
    axs[1, 1].set_xticklabels([tick.year for tick in x_ticks])

    axs[1, 2].scatter(full_data['date'], peak_price_predictions_2040, color='red', label='2040 Prediction', alpha=1)
    axs[1, 2].scatter(full_data['date'], trough_price_predictions_2040, color='green', label='2040 Prediction', alpha=1)
    axs[1, 2].scatter(full_data['date'], mid_price_predictions_2040, color='orange', label='2040 Prediction', alpha=1)

    axs[1, 2].set_title('2040 Prediction')
    # axs[1, 2].set_xlabel('Date')
    axs[1, 2].set_ylabel('Price in 2040')
    max_x_date = pd.to_datetime('2040-01-01')
    
    axs[1, 2].set_xlim(min_x, max_x_date)
    
    max_y_2040 = max_2040_peak*2

    axs[1, 2].set_ylim([0, max_y_2040])
    axs[1, 2].yaxis.set_major_formatter('${:,.0f}'.format)
    x_ticks = pd.date_range(start=min_x, end=max_x_date, periods=num_ticks).to_pydatetime()
    axs[1, 2].set_xticks(x_ticks)
    axs[1, 2].set_xticklabels([tick.year for tick in x_ticks])

   




    subtitle = f"{symbol} Current Date: {current_date.strftime('%Y-%m-%d') if current_date else 'N/A'}\n"
    if peak_model_params is not None:
        subtitle += f"Peak Model Params: a={peak_model_params[0]:.4f}, b={peak_model_params[1]:.4f}, R²={peak_r2:.4f}\n"
    else:
        subtitle += "Peak Model Params: waiting for peaks\n"
    if trough_model_params is not None:
        subtitle += f"Trough Model Params: a={trough_model_params[0]:.4f}, b={trough_model_params[1]:.4f}, R²={trough_r2:.4f}\n"
    else:
        subtitle += "Trough Model Params: waiting for troughs\n"

    
    if shaded:
        try:
            axs[0, 0].fill_between(full_data['date'], peak_model, trough_model, color='yellow', alpha=alpha, label='Power Law Model') 
            axs[0, 1].fill_between(full_data['date'], peak_model, trough_model, color='yellow', alpha=alpha, label='Power Law Model')
            axs[0, 2].fill_between(full_data.index, peak_model, trough_model, color='yellow', alpha=alpha, label='Power Law Model')
        except:
            print('Error in shaded area, continuing')
            pass

    fig.suptitle(subtitle)

    plt.subplots_adjust(top=0.83, wspace=0.3)    

    return fig, axs

def get_power_law_model(full_data, samples, text):
    historical_index = full_data.index
    if len(samples) < 2:
        model_price = [None] * len(historical_index)
        return model_price, None, None, None, None
    
    
    samples_index = full_data.index[samples]
    samples_price = full_data['close_price'][samples]
    log_samples_index = np.log(samples_index)
    log_samples_price = np.log(samples_price)

    # Fit a linear model to the log-log data

    
    model_params = np.polyfit(log_samples_index, log_samples_price, 1)
    model_price = np.exp(model_params[1]) * historical_index**model_params[0]
    model_price_samples = np.exp(model_params[1]) * samples_index**model_params[0]

    # Calculate R^2
    residuals = samples_price - model_price_samples
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((samples_price - np.mean(samples_price))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f'{text} R2 = ', r_squared)

    # Calculate the price in 2030
    # we get the index for closest to 1st January 2030
    index_2030 = full_data[full_data['date'] <= '2030-01-01'].index[-1]

    prediction_2030 = np.exp(model_params[1]) * (full_data.index[index_2030])**model_params[0]
    print(f'{text} 2030 prediction = ', prediction_2030)

    # work out what the index would be for 1st Jan 2040 (considering the data may not reach that date - we just want to extrapolate the index to that future date outside the list)
    last_date = full_data['date'].max()
    last_index = full_data[full_data['date'] == last_date].index[0]
    # future_date = last_date + (pd.to_datetime('2040-01-01') - last_date)
    index_2040 = last_index + (pd.to_datetime('2040-01-01') - last_date).days

    prediction_2040 = np.exp(model_params[1]) * (index_2040)**model_params[0]

    return model_price, model_params, r_squared, prediction_2030, prediction_2040  

def extend_data(full_data,final_year_model_prediction = 2040):
    full_data = full_data[['date', 'close_price']]
    final_date = pd.to_datetime(f'01-01-{final_year_model_prediction}')
    # st.text(f'Final date: {final_date}')
    # we extend full_data to finish at final_date, creating a new row for each day. BTC price is NaN for these rows
    full_data = full_data.set_index('date')
    original_max_date = full_data.index.max() 
    # st.dataframe(full_data)
    full_data = full_data.reindex(pd.date_range(full_data.index.min(), final_date, freq='D'))
    full_data = full_data.reset_index(drop=False)
    full_data.columns = ['date', 'close_price']

    # Drop NaN values only for rows with date less than or equal to the original maximum date
    mask = (full_data['date'] <= original_max_date) & (full_data['close_price'].isna())
    full_data = full_data[~mask]  # Use ~mask to keep rows not matching the condition

    # Reset index again after dropping rows
    full_data.reset_index(drop=True, inplace=True)

    return full_data

def animate(i, full_data, distance, prominence, divisions,min_x, max_x,min_y, max_y, symbol, min_date_model_index, min_year_ani,axs, alpha, transparency, shaded = False, style = 'Simple', max_2030_peak=None,max_2040_peak=None):
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    current_date = pd.Timestamp(f'{min_year_ani}-01-01') + pd.DateOffset(months=i)
    print('Current date:', current_date)
    temp_data = full_data.copy()
    # print(temp_data['date'])
    # temp_data['close_price'][temp_data['date'] > current_date] = np.nan
    print('Hi')
    temp_data.loc[temp_data['date'] > current_date, 'close_price'] = np.nan
    print('Hello')

    clean_peaks, clean_troughs = get_peaks_troughs(temp_data['close_price'], distance, prominence, divisions, min_date_model_index)
    print('Bimoland!')
    # Plot the data
    peak_model_prices, peak_model_params, peak_r2, prediction_2030_peak, prediction_2040_peak = get_power_law_model(temp_data, clean_peaks, 'Peak Model')
    trough_model_prices, trough_model_params, trough_r2, prediction_2030_trough, prediction_2040_trough = get_power_law_model(temp_data, clean_troughs, 'Trough Model')
    average_model_params = (peak_model_params + trough_model_params) / 2 if peak_model_params is not None and trough_model_params is not None else None
    average_model_prices = np.exp(average_model_params[1]) * full_data.index**average_model_params[0] if average_model_params is not None else [None] * len(full_data)
    print('Bimo!')

    if style == 'Simple':
        axs = get_frame_plot(temp_data, clean_peaks, clean_troughs, peak_model_prices, trough_model_prices, average_model_prices, min_x, max_x, min_y, max_y, symbol, peak_model_params, trough_model_params, average_model_params, peak_r2, trough_r2, current_date, shaded=shaded, alpha=alpha, transparency=transparency)
    elif style == 'Radar':
        axs = get_frame_radarplot(temp_data, clean_peaks, clean_troughs, peak_model_prices, trough_model_prices, average_model_prices, min_x, max_x, min_y, max_y, symbol, peak_model_params, trough_model_params, average_model_params, peak_r2, trough_r2, current_date, alpha, transparency, shaded, ax=axs, max_2030_peak= max_2030_peak, max_2040_peak= max_2040_peak)


    
    return axs


# %%
# Title of the app
st.title('Price Modelling')

# Define default values
symbol = 'BTC-USD'
start = datetime.datetime(1995, 1, 1)

# Select box for symbol selection
sidebar = st.sidebar
symbol = sidebar.selectbox('Select a symbol', ['SPY', 'BTC-USD', 'ETH-USD','SOL-USD','TSLA', 'GOOG', 'MSFT','QQQ','AMZN','IBM','INTC'])
max_x_date = sidebar.slider('Final Prediction', 2025, 2050, 2030,5)
final_year_model_prediction = 2050

# Download historical data and cache it
historical_data = load_data(symbol, start)
full_data = extend_data(historical_data, final_year_model_prediction)
current_date = historical_data['date'].max()

log_data = np.log(full_data['close_price'])
log10_hist = np.log10(full_data['close_price'])

# Sliders for distance and prominence
sidebar.header('Peak and Trough Detection')
distance = sidebar.slider('Distance', 10, 1000, 100)
default_prominence = (np.ceil(log10_hist.max()) - np.floor(log10_hist.min()))/20
# st.text(f'Default prominence: {default_prominence}')

prominence = sidebar.slider('Prominence', 0.1, 1.0, 0.2,0.1)
divisions = sidebar.slider('Divisions', 2, 10, 4)

# to calc the defualt min_model date, we want to get the index in the middle of the log index of dates. 
# we get log_indeces of dates, and we get the index in the middle
# print('original index:', full_data.index)
log_index_dates = np.log10(full_data.index)
# print('log index:', log_index_dates)
print('INDEX!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('max index:', historical_data.index.max())
print('max date:', historical_data['date'].max())
print('min date:', historical_data['date'].min())


print('max log index:', log_index_dates.max())
print('min log index:', log_index_dates.min())
log_middle_index = log_index_dates.max()// 2
print('middle index:', 10**log_middle_index)

def_middle_date = full_data['date'][10**log_middle_index]
print('default middle date:', def_middle_date)


# min_date_model_default = 
min_date_model = pd.to_datetime(sidebar.date_input('Min Date Model', value=def_middle_date))
min_date_model_index = full_data[full_data['date'] > min_date_model].index.min()

clean_peaks, clean_troughs = get_peaks_troughs(log_data, distance, prominence, divisions, min_date_model_index)

retained_peak_prices = np.array(log_data[clean_peaks])
retained_trough_prices = np.array(log_data[clean_troughs])

min_x = sidebar.date_input('Min Date', value=full_data['date'].min())
max_x = pd.to_datetime(f'01-01-{max_x_date}')
# st.text(f'Max date: {max_x}')
# we do a base 10 log scale of historical data

default_min_y = np.floor(log10_hist.min())
default_max_y = np.ceil(log10_hist.max())
min_y = 10**sidebar.number_input('Min Y (1e)', value=default_min_y)
max_y = 10**sidebar.number_input('Max Y (1e exponent)', value=default_max_y)

# Plot the data
peak_model_prices, peak_model_params, peak_r2, prediction_2030_peak, prediction_2040_peak = get_power_law_model(full_data, clean_peaks, 'Peak Model')
print('peak prediction 2030:', prediction_2030_peak)
print('peak prediction 2040:', prediction_2040_peak)
trough_model_prices, trough_model_params, trough_r2, prediction_2030_trough, prediction_2040_trough = get_power_law_model(full_data, clean_troughs, 'Trough Model')
average_model_params = (peak_model_params + trough_model_params) / 2 if peak_model_params is not None and trough_model_params is not None else None
average_model_prices = np.exp(average_model_params[1]) * full_data.index**average_model_params[0] if average_model_params is not None else [None] * len(full_data)

style = sidebar.radio('Choose chart style', ['Simple','Radar'], index=1)
alpha = sidebar.slider('Alpha', 0.0, 1.0, 0.5)
transparency = sidebar.checkbox('Trailing')
shaded = sidebar.checkbox('Shaded Area')
months_step = sidebar.slider('Months step', 1, 60, 12)

if style == 'Simple':
    fig, ax = get_frame_plot(full_data, clean_peaks, clean_troughs, peak_model_prices, trough_model_prices, average_model_prices, min_x, max_x, min_y, max_y, symbol, peak_model_params, trough_model_params, average_model_params, peak_r2, trough_r2, current_date, shaded=shaded, alpha=alpha, transparency=transparency)
elif style == 'Radar':
    fig, ax = get_frame_radarplot(full_data, clean_peaks, clean_troughs, peak_model_prices, trough_model_prices, average_model_prices, min_x, max_x, min_y, max_y, symbol, peak_model_params, trough_model_params, average_model_params, peak_r2, trough_r2, current_date, alpha, transparency, shaded, prediction_2030_peak, prediction_2040_peak)


min_year_ani = sidebar.slider('Min Year Animation', 1995, 2025, 2020)
max_year_ani = sidebar.slider('Max Year Animation', 2025, 2050, 2030)

if st.button('Animate'):
    fig, axs = plt.subplots(1 if style == 'Simple' else 2, 3, figsize=(14, 9))
    f_args = (full_data, distance, prominence, divisions, min_x, max_x, min_y, max_y, symbol, min_date_model_index, min_year_ani,axs, alpha, transparency, shaded, style, prediction_2030_peak,prediction_2040_peak)
    ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)
    ani.save('temp_animation.gif', writer=PillowWriter(fps=2))
    plt.close(fig)
    st.image('temp_animation.gif')
    # st.pyplot(fig)
else:
    st.pyplot(fig)


st.write('Min date loaded: ', full_data['date'].min())
st.write('Max date loaded: ', full_data['date'].max())