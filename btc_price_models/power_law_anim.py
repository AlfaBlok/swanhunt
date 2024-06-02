# %%

# WE TRY NOW TO SUPPORT ITEMS OTHER THAN BTC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import find_peaks
import yfinance as yf
import datetime


def load_data(symbol, start_date=None):

    if symbol == 'BTC-USD':
        historical_data_btc = pd.read_csv('historical_btc.csv')
        historical_data_btc['date'] = pd.to_datetime(historical_data_btc['date'])
        last_date = historical_data_btc['date'].max()
        new_data_btc = yf.download(symbol, start=start_date) if start_date else yf.download(symbol, start='1970-01-01')
        new_data_btc = new_data_btc.reset_index()
        new_data_btc = new_data_btc[['Date', 'Close']]
        new_data_btc.columns = ['date', 'btc_price']
        full_data = pd.concat([historical_data_btc, new_data_btc], axis=0)
        full_data = full_data.sort_values('date')
        full_data = full_data[full_data['date'] >= start_date] if start_date else full_data
        full_data['date'] = pd.to_datetime(full_data['date'])
        # we remove dates before start_date

        full_data = full_data.reset_index(drop=False)
        max_date = full_data['date'].max()
        print('Max date loaded:', max_date)
    else:
        new_data_btc = yf.download(symbol, start=start_date)
        new_data_btc = new_data_btc.reset_index()
        new_data_btc = new_data_btc[['Date', 'Close']]
        new_data_btc.columns = ['date', 'btc_price']
        full_data = new_data_btc
        full_data = full_data.sort_values('date')
        full_data = full_data.reset_index(drop=False)
        full_data['date'] = pd.to_datetime(full_data['date'])
        max_date = full_data['date'].max()
        print('Min date loaded:', full_data['date'].min())
        print('Max date loaded:', max_date)

    if symbol == 'SPY':
        full_data['btc_price'] = full_data['btc_price'] * 10   

    return full_data



def get_peaks(full_data, d_days, threshold, d_days2, min_date=None):
    full_data = full_data[['date', 'btc_price']].copy()
    full_data.reset_index(drop=True, inplace=True)
    # print('Getting peaks')

    # Find initial peaks with a specified distance
    peaks, _ = find_peaks(full_data['btc_price'], distance=d_days)
    
    # First pass to identify initial peaks
    clean_peaks = []
    btc_prices = full_data['btc_price'].values
    
    for peak in peaks:
        if not any(btc_prices[peak] < btc_prices[past_peak] for past_peak in clean_peaks):
            if peak + 1 < len(btc_prices):
                right_side = btc_prices[peak + 1:peak + 1 + d_days2]
                if len(right_side) > 0 and (right_side < btc_prices[peak] * (1 - threshold)).any():
                    if (right_side >= btc_prices[peak]).any():
                        pass
                    else:
                        clean_peaks.append(peak)

    clean_peaks = np.array(clean_peaks)
    
    # Enforce one peak per d_days using a backwards window
    final_peaks = []
    if len(clean_peaks) > 0:
        final_peaks.append(clean_peaks[0])
        for i in range(1, len(clean_peaks)):
            if clean_peaks[i] - final_peaks[-1] >= d_days:
                final_peaks.append(clean_peaks[i])
            elif btc_prices[clean_peaks[i]] > btc_prices[final_peaks[-1]]:
                final_peaks[-1] = clean_peaks[i]

    final_peaks = np.array(final_peaks)
    verified_peaks = final_peaks

    clean_peaks_after_year = verified_peaks

    if min_date:
        # if min_date < full_data['date'].min():
        # we convert min_date to datetime
        min_date = pd.to_datetime(min_date)
        min_date_index = full_data[full_data['date'] >= min_date].index[0]
        clean_peaks_after_year = clean_peaks_after_year[clean_peaks_after_year >= min_date_index]
    
    return peaks, clean_peaks_after_year

def get_peaks_old(full_data, d_days, threshold, d_days2, min_date=None):
    full_data = full_data[['date', 'btc_price']].copy()
    full_data.reset_index(drop=True, inplace=True)
    # print('Getting peaks')

    # Find initial peaks with a specified distance
    peaks, _ = find_peaks(full_data['btc_price'], distance=d_days)
    
    # First pass to identify initial peaks
    clean_peaks = []
    btc_prices = full_data['btc_price'].values
    
    for peak in peaks:
        if not any(btc_prices[peak] < btc_prices[past_peak] for past_peak in clean_peaks):
            if peak + 1 < len(btc_prices):
                right_side = btc_prices[peak + 1:peak + 1 + d_days2]
                if len(right_side) > 0 and (right_side < btc_prices[peak] * (1 - threshold)).any():
                    clean_peaks.append(peak)

    clean_peaks = np.array(clean_peaks)
    
    # Enforce one peak per d_days using a backwards window
    final_peaks = []
    if len(clean_peaks) > 0:
        final_peaks.append(clean_peaks[0])
        for i in range(1, len(clean_peaks)):
            if clean_peaks[i] - final_peaks[-1] >= d_days:
                final_peaks.append(clean_peaks[i])
            elif btc_prices[clean_peaks[i]] > btc_prices[final_peaks[-1]]:
                final_peaks[-1] = clean_peaks[i]

    final_peaks = np.array(final_peaks)
    verified_peaks = final_peaks

    clean_peaks_after_year = verified_peaks

    if min_date:
        # if min_date < full_data['date'].min():
        # we convert min_date to datetime
        min_date = pd.to_datetime(min_date)
        min_date_index = full_data[full_data['date'] >= min_date].index[0]
        clean_peaks_after_year = clean_peaks_after_year[clean_peaks_after_year >= min_date_index]
    
    return peaks, clean_peaks_after_year

def get_troughs(full_data, d_days, threshold, min_date):
    full_data = full_data[['date', 'btc_price']].copy()
    full_data.reset_index(drop=True, inplace=True)
    troughs, _ = find_peaks(-full_data['btc_price'], distance=d_days)
    
    # print('Troughs:', troughs   )
    # print('Full data:', full_data)
    clean_troughs = []
    if len(troughs) > 0:
        for i, trough in enumerate(troughs):
            if not any(full_data['btc_price'][trough] > full_data['btc_price'][future_trough] for future_trough in troughs[i:]):
                clean_troughs.append(trough)

    clean_troughs = np.array(clean_troughs)
    # Enforce one peak per d_days using a backwards window
    final_troughs = []
    if len(clean_troughs) > 0:
        final_troughs.append(clean_troughs[0])
        for i in range(1, len(clean_troughs)):
            if clean_troughs[i] - final_troughs[-1] >= d_days:
                final_troughs.append(clean_troughs[i])
            elif full_data['btc_price'][clean_troughs[i]] > full_data['btc_price'][final_troughs[-1]]:
                final_troughs[-1] = clean_troughs[i]
    
    final_troughs = np.array(final_troughs)

    clean_troughs_after_year = final_troughs


    clean_troughs_after_year = clean_troughs_after_year[:-1]

    if min_date:
        min_date = pd.to_datetime(min_date)
        min_date_index = full_data[full_data['date'] >= min_date].index[0]
        clean_troughs_after_year = clean_troughs_after_year[clean_troughs_after_year >= min_date_index]


    return troughs, clean_troughs_after_year

def get_frame_plot(full_data, clean_peaks, clean_troughs, peak_model, trough_model, average_model_prices,min_x, max_x, min_y, max_y, symbol, peak_model_params, trough_model_params, average_model_params, peak_r2, trough_r2, current_date=None, alpha=1, transparency = False, shaded = False, ax=None):
    # fig, axs = plt.subplots(1, 3, figsize=(14, 7.3))
    print('Here alpha:', alpha)

    if ax is None:
        fig, axs = plt.subplots(1, 3, figsize=(14, 7.25))
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


    axs[0].plot(full_data['date'], full_data['btc_price'], label='BTC Price', color='blue')
    axs[0].scatter(full_data['date'][clean_peaks], full_data['btc_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    axs[0].scatter(full_data['date'][clean_troughs], full_data['btc_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[0].plot(full_data['date'], peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[0].plot(full_data['date'], trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[0].plot(full_data['date'], average_model_prices, color='orange', label='Model Average', alpha=alpha)


    axs[0].set_title(f'{symbol} price')
    axs[0].set_xlabel('Date')
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
    # axs[0].set_ylim(full_data['btc_price'].min(), full_data['btc_price'].max())
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].yaxis.set_major_formatter('${:,.0f}'.format)
    # axs[0].legend()
    axs[0].set_ylim([0, max_y])  
    
    # we show only 5 ticks on the x axis


    
    # Log scale
    axs[1].plot(full_data['date'], full_data['btc_price'], label='BTC Price', color='blue')
    axs[1].scatter(full_data['date'][clean_peaks], full_data['btc_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    axs[1].scatter(full_data['date'][clean_troughs], full_data['btc_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[1].plot(full_data['date'], peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[1].plot(full_data['date'], trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[1].plot(full_data['date'], average_model_prices, color='orange', label='Model Average', alpha=alpha)
    axs[1].set_xlim(full_data['date'].min() if min_x is None else min_x, full_data['date'].max() if max_x is None else max_x)
    # axs[1].set_ylim(full_data['btc_price'].min(), full_data['btc_price'].max())
    axs[1].set_title(f'{symbol} price (log)')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Price')
    axs[1].set_yscale('log')
    axs[1].set_ylim([min_y, max_y])  # Set log y range
    axs[1].yaxis.set_major_formatter('${:,.0f}'.format)
    axs[1].set_xticks(x_ticks)
    axs[1].set_xticklabels([tick.year for tick in x_ticks])

    # # Log-log scale
    axs[2].plot(full_data.index, full_data['btc_price'], label='BTC Price', color='blue')
    if len(clean_peaks) > 0:
        axs[2].scatter(full_data.index[clean_peaks], full_data['btc_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    if len(clean_troughs) > 0:
        axs[2].scatter(full_data.index[clean_troughs], full_data['btc_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[2].plot(full_data.index, peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[2].plot(full_data.index, trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[2].plot(full_data.index, average_model_prices, color='orange', label='Model Average', alpha=alpha)
    axs[2].set_title(f'{symbol} price (log-log)')
    axs[2].set_xlabel('Date')
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

def get_frame_radarplot(full_data, clean_peaks, clean_troughs, peak_model, trough_model, average_model_prices, min_x, max_x, min_y, max_y, symbol, peak_model_params, trough_model_params, average_model_params, peak_r2, trough_r2, current_date=None, alpha=1, transparency=False, shaded=False, ax=None, prediction_2030_peak=None, prediction_2030_trough=None):
    if ax is None:
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
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

    axs[0, 0].plot(full_data['date'], full_data['btc_price'], label='BTC Price', color='blue')
    axs[0, 0].scatter(full_data['date'][clean_peaks], full_data['btc_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    axs[0, 0].scatter(full_data['date'][clean_troughs], full_data['btc_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[0, 0].plot(full_data['date'], peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[0, 0].plot(full_data['date'], trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[0, 0].plot(full_data['date'], average_model_prices, color='orange', label='Model Average', alpha=alpha)

    axs[0, 0].set_title(f'{symbol} price')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Price')
    axs[0, 0].set_xlim(min_x, max_x)
    axs[0, 0].set_ylim([0, max_y])
    axs[0, 0].yaxis.set_major_formatter('${:,.0f}'.format)

    num_ticks = 5
    x_ticks = pd.date_range(start=min_x, end=max_x, periods=num_ticks).to_pydatetime()
    axs[0, 0].set_xticks(x_ticks)
    axs[0, 0].set_xticklabels([tick.year for tick in x_ticks])

    axs[0, 1].plot(full_data['date'], full_data['btc_price'], label='BTC Price', color='blue')
    axs[0, 1].scatter(full_data['date'][clean_peaks], full_data['btc_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    axs[0, 1].scatter(full_data['date'][clean_troughs], full_data['btc_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[0, 1].plot(full_data['date'], peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[0, 1].plot(full_data['date'], trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[0, 1].plot(full_data['date'], average_model_prices, color='orange', label='Model Average', alpha=alpha)
    axs[0, 1].set_xlim(min_x, max_x)
    axs[0, 1].set_title(f'{symbol} price (log)')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Price')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_ylim([min_y, max_y])
    axs[0, 1].yaxis.set_major_formatter('${:,.0f}'.format)
    axs[0, 1].set_xticks(x_ticks)
    axs[0, 1].set_xticklabels([tick.year for tick in x_ticks])

    axs[0, 2].plot(full_data.index, full_data['btc_price'], label='BTC Price', color='blue')
    if len(clean_peaks) > 0:
        axs[0, 2].scatter(full_data.index[clean_peaks], full_data['btc_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    if len(clean_troughs) > 0:
        axs[0, 2].scatter(full_data.index[clean_troughs], full_data['btc_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[0, 2].plot(full_data.index, peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[0, 2].plot(full_data.index, trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[0, 2].plot(full_data.index, average_model_prices, color='orange', label='Model Average', alpha=alpha)
    axs[0, 2].set_title(f'{symbol} price (log-log)')
    axs[0, 2].set_xlabel('Date')
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


    # Radar plot
    if average_model_prices is not None:
        date_index_2030 = full_data[full_data['date'] == pd.to_datetime('2030-01-01')].index[0]
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
    axs[1, 0].set_ylim([0, max_y])
    axs[1, 0].yaxis.set_major_formatter('${:,.0f}'.format)

    # we create an array that is null for every date except for current_date, and in that date we put the 2030 prediction
    # we create a scatter plot with dates on x and price predictions on y. It will be a single dot in the current date
    peak_price_predictions = np.full(len(full_data), np.nan)
    trough_price_predictions = np.full(len(full_data), np.nan)
    mid_price_predictions = np.full(len(full_data), np.nan)
    
    
    if current_date_index is not None and full_data.loc[current_date_index, 'date'].strftime("%Y-%m-%d") < datetime.datetime.now().strftime("%Y-%m-%d"):
        current_date_index = full_data[full_data['date'] == current_date].index[0]
        date_index_2030 = full_data[full_data['date'] == pd.to_datetime('2030-01-01')].index[0]
        prediction_2030_mid = average_model_prices[date_index_2030]
        if current_date_index is not None:
            peak_price_predictions[current_date_index] = prediction_2030_peak
            trough_price_predictions[current_date_index] = prediction_2030_trough
            mid_price_predictions[current_date_index] = prediction_2030_mid
        else:
            print('Current date not in data')


        # we calculate the mid price prediction for 1st january 2030


    axs[1, 1].scatter(full_data['date'], peak_price_predictions, color='red', label='2030 Prediction', alpha=1)
    axs[1, 1].scatter(full_data['date'], trough_price_predictions, color='green', label='2030 Prediction', alpha=1)
    axs[1, 1].scatter(full_data['date'], mid_price_predictions, color='orange', label='2030 Prediction', alpha=1)

    axs[1, 1].set_title('2030 Prediction')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Price in 2030')
    axs[1, 1].set_xlim(min_x, max_x)    
    axs[1, 1].set_ylim([0, max_y])
    axs[1, 1].yaxis.set_major_formatter('${:,.0f}'.format)
    axs[1, 1].set_xticks(x_ticks)
    axs[1, 1].set_xticklabels([tick.year for tick in x_ticks])






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

def get_frame_radarplot_old(full_data, clean_peaks, clean_troughs, peak_model, trough_model, average_model_prices,min_x, max_x, min_y, max_y, symbol, peak_model_params, trough_model_params, average_model_params, peak_r2, trough_r2, current_date=None, alpha=1, transparency = False, shaded = False, ax=None, prediction_2030_peak = None, prediction_2030_trough = None):
    # fig, axs = plt.subplots(1, 3, figsize=(14, 7.3))

    if ax is None:
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
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


    axs[0,0].plot(full_data['date'], full_data['btc_price'], label='BTC Price', color='blue')
    axs[0,0].scatter(full_data['date'][clean_peaks], full_data['btc_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    axs[0,0].scatter(full_data['date'][clean_troughs], full_data['btc_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[0,0].plot(full_data['date'], peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[0,0].plot(full_data['date'], trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[0,0].plot(full_data['date'], average_model_prices, color='orange', label='Model Average', alpha=alpha)


    axs[0,0].set_title(f'{symbol} price')
    axs[0,0].set_xlabel('Date')
    axs[0,0].set_ylabel('Price')
    # axs[0].legend()
    # full_data['date'].min() if min_x is None else pd.datatime(min_x)
    min_x = full_data['date'].min() if min_x is None else min_x
    max_x = full_data['date'].max() if max_x is None else max_x

    min_x_index = full_data[full_data['date'] >= min_x].index[0]
    max_x_index = full_data[full_data['date'] <= max_x].index[-1]

        
    # Calculate x-ticks
    num_ticks = 5
    x_ticks = pd.date_range(start=min_x, end=max_x, periods=num_ticks).to_pydatetime()
    axs[0,0].set_xticks(x_ticks)
    axs[0,0].set_xticklabels([tick.year for tick in x_ticks])



    axs[0,0].set_xlim(min_x, max_x)
    # axs[0].set_ylim(full_data['btc_price'].min(), full_data['btc_price'].max())
    axs[0,0].set_xlabel('Date')
    axs[0,0].set_ylabel('Price')
    axs[0,0].yaxis.set_major_formatter('${:,.0f}'.format)
    # axs[0].legend()
    axs[0,0].set_ylim([0, max_y])  
    
    # we show only 5 ticks on the x axis


    
    # Log scale
    axs[0,1].plot(full_data['date'], full_data['btc_price'], label='BTC Price', color='blue')
    axs[0,1].scatter(full_data['date'][clean_peaks], full_data['btc_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    axs[0,1].scatter(full_data['date'][clean_troughs], full_data['btc_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[0,1].plot(full_data['date'], peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[0,1].plot(full_data['date'], trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[0,1].plot(full_data['date'], average_model_prices, color='orange', label='Model Average', alpha=alpha)
    axs[0,1].set_xlim(full_data['date'].min() if min_x is None else min_x, full_data['date'].max() if max_x is None else max_x)
    # axs[0,1].set_ylim(full_data['btc_price'].min(), full_data['btc_price'].max())
    axs[0,1].set_title(f'{symbol} price (log)')
    axs[0,1].set_xlabel('Date')
    axs[0,1].set_ylabel('Price')
    axs[0,1].set_yscale('log')
    axs[0,1].set_ylim([min_y, max_y])  # Set log y range
    axs[0,1].yaxis.set_major_formatter('${:,.0f}'.format)
    axs[0,1].set_xticks(x_ticks)
    axs[0,1].set_xticklabels([tick.year for tick in x_ticks])

    # # Log-log scale
    axs[0,2].plot(full_data.index, full_data['btc_price'], label='BTC Price', color='blue')
    if len(clean_peaks) > 0:
        axs[0,2].scatter(full_data.index[clean_peaks], full_data['btc_price'][clean_peaks], color='red', label='Peaks', alpha=alpha)
    if len(clean_troughs) > 0:
        axs[0,2].scatter(full_data.index[clean_troughs], full_data['btc_price'][clean_troughs], color='green', label='Troughs', alpha=alpha)
    axs[0,2].plot(full_data.index, peak_model, color='red', label='Model Peaks', alpha=alpha)
    axs[0,2].plot(full_data.index, trough_model, color='green', label='Model Troughs', alpha=alpha)
    axs[0,2].plot(full_data.index, average_model_prices, color='orange', label='Model Average', alpha=alpha)
    axs[0,2].set_title(f'{symbol} price (log-log)')
    axs[0,2].set_xlabel('Date')
    axs[0,2].set_ylabel('Price')
    axs[0,2].set_yscale('log')
    axs[0,2].set_xscale('log')
    if max_x is None:
        max_index = len(full_data) - 1
    else:
        max_index = full_data[full_data['date'] <= max_x].index[-1]
    if max_index < 1000:
        axs[0,2].set_xticks([1, 10, 100, 1000])
    else:
        axs[0,2].set_xticks([1, 10, 100, 1000, max_index])
    axs[0,2].set_xticklabels([full_data['date'][1].year, full_data['date'][10].year, full_data['date'][100].year, full_data['date'][1000].year, full_data['date'][max_index].year])
    # we get 
    
    # axs[0,2].set_xticks(x_ticks)
    # axs[0,2].set_xticklabels([tick.year for tick in x_ticks])


    axs[0,2].yaxis.set_major_formatter('${:,.0f}'.format)
    axs[0,2].set_ylim([min_y, max_y])
    if min_x is not None:
        # we look for the index of the min_x date
        min_x_index = full_data[full_data['date'] >= min_x].index[0]
        axs[0,2].set_xlim([min_x_index, max_index])  # Set min for the third axis


    # Radar plot
    # we want a scatter plot with the prediction for 2030 on y axis and r2 on x axis
    axs[1][0].scatter([peak_r2], [prediction_2030_peak], color='red', label='Peaks', alpha=alpha)
    axs[1][0].scatter([trough_r2], [prediction_2030_trough], color='green', label='Troughs', alpha=alpha)
    axs[1][0].set_title('Radar plot')
    axs[1][0].set_xlabel('R²')
    axs[1][0].set_ylabel('Price in 2030')
    axs[1][0].set_xlim([-1, 1])
    # axs[1][0].set_ylim([0, max_y])


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
        axs[0,0].fill_between(full_data['date'], peak_model, trough_model, color='yellow', alpha=alpha, label='Power Law Model') 
        axs[0,1].fill_between(full_data['date'], peak_model, trough_model, color='yellow', alpha=alpha, label='Power Law Model')
        axs[0,2].fill_between(full_data.index, peak_model, trough_model, color='yellow', alpha=alpha, label='Power Law Model')



    fig.suptitle(subtitle)

    plt.subplots_adjust(top=0.8, wspace=alpha)    

    return fig, axs

def get_power_law_model(full_data, samples, text):
    historical_index = full_data.index
    if len(samples) < 2:
        model_price = [None] * len(historical_index)
        return model_price, None, None, None
    
    
    samples_index = full_data.index[samples]
    samples_price = full_data['btc_price'][samples]
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

    return model_price, model_params, r_squared, prediction_2030

def extend_data(full_data,final_year_model_prediction = 2040):
    final_date = pd.to_datetime(f'01-01-{final_year_model_prediction}')
    # we extend full_data to finish at final_date, creating a new row for each day. BTC price is NaN for these rows
    full_data = full_data.set_index('date')
    full_data = full_data.reindex(pd.date_range(full_data.index.min(), final_date, freq='D'))
    # we convert the index to a column
    full_data = full_data.reset_index()
    full_data.columns = ['date', 'btc_price']
    # full_data = full_data.reset_index(drop=False, inplace=True)
    return full_data

def animate(i, full_data, d_days, threshold, d_days2,min_x, max_x,min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency, shaded = False, radar = False):
    current_date = pd.Timestamp(f'{min_year_ani}-01-01') + pd.DateOffset(months=i)
    temp_data = full_data.copy()
    # print(temp_data['date'])
    # temp_data['btc_price'][temp_data['date'] > current_date] = np.nan
    temp_data.loc[temp_data['date'] > current_date, 'btc_price'] = np.nan
    peaks, clean_peaks = get_peaks(temp_data, d_days, threshold, d_days2, min_date_model)
    troughs, clean_troughs = get_troughs(temp_data, d_days, threshold, min_date_model)
    peak_model_prices, peak_model_params, peak_r2, peak_prediction_2030 = get_power_law_model(temp_data, clean_peaks, 'Peak Model')
    trough_model_prices, trough_model_params, trough_r2, trough_prediction_2030 = get_power_law_model(temp_data, clean_troughs, 'Trough Model')
    average_model_params = (peak_model_params + trough_model_params) / 2 if peak_model_params is not None and trough_model_params is not None else None
    average_model_prices = np.exp(average_model_params[1]) * full_data.index**average_model_params[0] if average_model_params is not None else [None] * len(full_data)
    print('Alpha:', alpha)
    if radar == True:
        axs = get_frame_radarplot(temp_data, clean_peaks, clean_troughs, peak_model_prices, trough_model_prices, average_model_prices, min_x, max_x, min_y, max_y, symbol, peak_model_params, trough_model_params, average_model_params, peak_r2, trough_r2, current_date, alpha, transparency, shaded, ax=axs, prediction_2030_peak = peak_prediction_2030, prediction_2030_trough = trough_prediction_2030)
    else:
        axs = get_frame_plot(temp_data, clean_peaks, clean_troughs, peak_model_prices, trough_model_prices, average_model_prices, min_x, max_x, min_y, max_y, symbol, peak_model_params, trough_model_params, average_model_params, peak_r2, trough_r2, current_date, alpha, transparency, shaded, ax=axs)

    return axs

# %%

symbol = 'BTC-USD'
historical_data = load_data(symbol)
min_date_model = '2010-01-01'
final_year_model_prediction = 2030
d_days = 300
d_days2 = 300
threshold = 0.45
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 1e-4
max_y = 6e5
min_x = '2010-01-01'
max_x = '2030-01-01'

current_date = full_data['date'].max()
   

transparency = False
alpha = 1
shaded = False

peaks, clean_peaks = get_peaks(full_data, d_days, threshold, d_days2, min_date_model)
troughs, clean_troughs = get_troughs(full_data, d_days, threshold, min_date_model)
peak_model_prices, peak_model_params, peak_r2, prediction_2030_peak = get_power_law_model(full_data, clean_peaks, 'Peak Model')
trough_model_prices, trough_model_params, trough_r2, prediction_2030_trough = get_power_law_model(full_data, clean_troughs, 'Trough Model')
average_model_params = (peak_model_params + trough_model_params) / 2
average_model_prices = np.exp(average_model_params[1]) * full_data.index**average_model_params[0]

fig, ax = get_frame_plot(full_data, clean_peaks, clean_troughs, peak_model_prices, trough_model_prices, average_model_prices, min_x, max_x, min_y, max_y, symbol, peak_model_params, trough_model_params, average_model_params, peak_r2, trough_r2, current_date)


plt.show()



symbol = 'BTC-USD'
historical_data = load_data(symbol)

final_year_model_prediction = 2030
d_days = 365
d_days2 = 365
threshold = 0.5
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 5e-4
max_y = 6e5

min_date_model = '2010-01-01'
min_x = '2009-09-01'
max_x = '2030-01-01'
min_year_ani = 2015
max_year_ani = 2030

months_step = 60

transparency = False
alpha = 1
shaded = False
radar = False

fig, axs = plt.subplots(1, 3, figsize=(14, 7))
f_args = (full_data, d_days, threshold, d_days2, min_x, max_x, min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency, shaded, radar)


ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)

ani.save('new_flow_TEST.gif', writer=PillowWriter(fps=2))

plt.show()

# %%

symbol = 'GOOG'
historical_data = load_data(symbol)

final_year_model_prediction = 2030
d_days = 365
d_days2 = 365
threshold = 0.3
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 1e0
max_y = 3e2

min_date_model = '2006-01-01'
min_x = '2004-09-01'
max_x = '2030-01-01'
min_year_ani = 2010
max_year_ani = 2030

months_step = 12

transparency = True
alpha = 0.15

fig, axs = plt.subplots(1, 3, figsize=(14, 7))
f_args = (full_data, d_days, threshold, d_days2, min_x, max_x, min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency)

ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)

ani.save('new_flow_TEST.gif', writer=PillowWriter(fps=2))

plt.show()


# %%


symbol = 'MSFT'
historical_data = load_data(symbol)

final_year_model_prediction = 2030
d_days = 60
d_days2 = 50
threshold = 0.1
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 1e1
max_y = 1e3

min_date_model = '2005-01-01'
min_x = '2000-01-01'
max_x = '2030-01-01'
min_year_ani = 1995
max_year_ani = 2040

months_step = 60

transparency = False
alpha = 1

fig, axs = plt.subplots(1, 3, figsize=(14, 7))
f_args = (full_data, d_days, threshold, d_days2, min_x, max_x, min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency)

ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)

ani.save('new_flow_TEST.gif', writer=PillowWriter(fps=2))

plt.show()

# %%

symbol = 'SPY'
historical_data = load_data(symbol)

final_year_model_prediction = 2030
d_days = 200
d_days2 = 200
threshold = 0.2
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 1e2
max_y = 1e4

min_date_model = '2000-01-01'
min_x = '1990-01-01'
max_x = '2030-01-01'
min_year_ani = 1995
max_year_ani = 2040

months_step = 60

transparency = False
alpha = 1

fig, axs = plt.subplots(1, 3, figsize=(14, 7))
f_args = (full_data, d_days, threshold, d_days2, min_x, max_x, min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency)

ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)

ani.save('new_flow_TEST.gif', writer=PillowWriter(fps=2))

plt.show()

# %%


symbol = 'SPY'
historical_data = load_data(symbol)

final_year_model_prediction = 2030
d_days = 900
d_days2 = 300
threshold = 0.2
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 1e2
max_y = 1e4

min_date_model = '2005-01-01'
min_x = '1994-01-01'
max_x = '2030-01-01'
min_year_ani = 1994
max_year_ani = 2040

months_step = 24

transparency = True
alpha = 0.2

fig, axs = plt.subplots(1, 3, figsize=(14, 7))
f_args = (full_data, d_days, threshold, d_days2, min_x, max_x, min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency)

ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)

ani.save('new_flow_TEST.gif', writer=PillowWriter(fps=2))

plt.show()

# %%

symbol = 'META'
historical_data = load_data(symbol)

final_year_model_prediction = 2030
d_days = 100
d_days2 = 180
threshold = 0.35
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 1e1
max_y = 1e3

min_date_model = '2015-01-01'
min_x = '2013-01-01'
max_x = '2030-01-01'
min_year_ani = 2000
max_year_ani = 2040

months_step = 12

transparency = True
alpha = 0.3

fig, axs = plt.subplots(1, 3, figsize=(14, 7))
f_args = (full_data, d_days, threshold, d_days2, min_x, max_x, min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency)

ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)

ani.save('new_flow_TEST.gif', writer=PillowWriter(fps=2))

plt.show()


# %%

symbol = 'BTC-USD'
historical_data = load_data(symbol)

final_year_model_prediction = 2030
d_days = 365
d_days2 = 365
threshold = 0.5
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 5e-4
max_y = 6e5

min_date_model = '2010-01-01'
min_x = '2009-09-01'
max_x = '2030-01-01'
min_year_ani = 2010
max_year_ani = 2030

months_step = 12

transparency = True
alpha = 0.1

fig, axs = plt.subplots(1, 3, figsize=(14, 7))
f_args = (full_data, d_days, threshold, d_days2, min_x, max_x, min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency)

ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)

ani.save('new_flow_TEST.gif', writer=PillowWriter(fps=2))

plt.show()

# %%
# WHY TRY TO DO shaded between the two models

symbol = 'BTC-USD'
historical_data = load_data(symbol)

final_year_model_prediction = 2030
d_days = 365
d_days2 = 365
threshold = 0.5
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 5e-4
max_y = 6e5

min_date_model = '2010-01-01'
min_x = '2009-09-01'
max_x = '2030-01-01'
min_year_ani = 2015
max_year_ani = 2030

months_step = 30

transparency = False
alpha = 0.2
shaded = True

fig, axs = plt.subplots(1, 3, figsize=(14, 7))
f_args = (full_data, d_days, threshold, d_days2, min_x, max_x, min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency, shaded)

ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)

ani.save('new_flow_TEST.gif', writer=PillowWriter(fps=2))

plt.show()

# %%
# WHY TRY TO DO shaded between the two models

symbol = 'BTC-USD'
historical_data = load_data(symbol)

final_year_model_prediction = 2030
d_days = 365
d_days2 = 365
threshold = 0.5
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 5e-4
max_y = 6e5

min_date_model = '2010-01-01'
min_x = '2009-09-01'
max_x = '2030-01-01'
min_year_ani = 2015
max_year_ani = 2030

months_step = 5

transparency = True
alpha = 0.03
shaded = True

fig, axs = plt.subplots(1, 3, figsize=(14, 7))
f_args = (full_data, d_days, threshold, d_days2, min_x, max_x, min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency, shaded)

ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)

ani.save('new_flow_TEST.gif', writer=PillowWriter(fps=2))

plt.show()

# %%


symbol = 'BTC-USD'
historical_data = load_data(symbol)

final_year_model_prediction = 2030
d_days = 365
d_days2 = 365
threshold = 0.5
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 5e-4
max_y = 1e6
min_date_model = '2010-01-01'
min_x = '2009-09-01'
max_x = '2030-01-01'
min_year_ani = 2013
max_year_ani = 2030

months_step = 12

transparency = True
alpha = 0.1
shaded = True
radar = True

fig, axs = plt.subplots(1 if radar == False else 2, 3, figsize=(14, 7))
f_args = (full_data, d_days, threshold, d_days2, min_x, max_x, min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency, shaded, radar)

ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)

ani.save('radar_BTC.gif', writer=PillowWriter(fps=2))

plt.show()


# %%


symbol = 'ETH-USD'
historical_data = load_data(symbol)

final_year_model_prediction = 2030
d_days = 365
d_days2 = 365
threshold = 0.5
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 1e1
max_y = 1e4
min_date_model = '2018-01-01'
min_x = '2009-09-01'
max_x = '2030-01-01'
min_year_ani = 2013
max_year_ani = 2030

months_step = 6

transparency = True
alpha = 0.05
shaded = True
radar = True

fig, axs = plt.subplots(1 if radar == False else 2, 3, figsize=(14, 7))
f_args = (full_data, d_days, threshold, d_days2, min_x, max_x, min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency, shaded, radar)

ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)

ani.save('radar_ETH.gif', writer=PillowWriter(fps=2))

plt.show()




# %%
symbol = 'GOOG'
historical_data = load_data(symbol)

final_year_model_prediction = 2030
d_days = 365
d_days2 = 365
threshold = 0.3
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 1e0
max_y = 3e2
min_date_model = '2006-01-01'
min_x = '2004-09-01'
max_x = '2030-01-01'
min_year_ani = 2010
max_year_ani = 2030
months_step = 24

transparency = True
alpha = 0.1
shaded = True
radar = True

fig, axs = plt.subplots(1 if radar == False else 2, 3, figsize=(14, 7))
f_args = (full_data, d_days, threshold, d_days2, min_x, max_x, min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency, shaded, radar)
ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)
ani.save('4d_flow_TEST.gif', writer=PillowWriter(fps=2))
plt.show()




# %%
symbol = 'MSFT'
historical_data = load_data(symbol)

final_year_model_prediction = 2030
d_days = 60
d_days2 = 50
threshold = 0.1
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 1e1
max_y = 1e3

min_date_model = '2005-01-01'
min_x = '2000-01-01'
max_x = '2030-01-01'
min_year_ani = 1995
max_year_ani = 2040
months_step = 36

transparency = True
alpha = 0.1
shaded = True
radar = True

fig, axs = plt.subplots(1 if radar == False else 2, 3, figsize=(14, 7))
f_args = (full_data, d_days, threshold, d_days2, min_x, max_x, min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency, shaded, radar)
ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)
ani.save('4d_flow_TEST.gif', writer=PillowWriter(fps=2))
plt.show()


# %%

symbol = 'SPY'
historical_data = load_data(symbol)

final_year_model_prediction = 2030
d_days = 365
d_days2 = 100
threshold = 0.30
full_data = historical_data.copy()
full_data = full_data.drop_duplicates('date')
full_data = full_data[['date', 'btc_price']]
full_data = extend_data(full_data, final_year_model_prediction) if final_year_model_prediction else full_data

min_y = 1e2
max_y = 1e4

min_date_model = '2005-01-01'
min_x = '1990-01-01'
max_x = '2030-01-01'
min_year_ani = 1995
max_year_ani = 2040

months_step = 60

transparency = True
alpha = 0.1
shaded = True
radar = True

fig, axs = plt.subplots(1 if radar == False else 2, 3, figsize=(14, 7))
f_args = (full_data, d_days, threshold, d_days2, min_x, max_x, min_y, max_y, symbol, min_date_model, min_year_ani,axs, alpha, transparency, shaded, radar)
ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=f_args, interval=200)
ani.save('4d_flow_TEST.gif', writer=PillowWriter(fps=2))
plt.show()