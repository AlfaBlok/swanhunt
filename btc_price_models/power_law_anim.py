# %%

# WE TRY NOW TO SUPPORT ITEMS OTHER THAN BTC



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import find_peaks
import yfinance as yf



def get_peaks(full_data, d_days, threshold):
    full_data = full_data[['date', 'btc_price']].copy()
    full_data.reset_index(drop=True, inplace=True)
    print('Getting peaks')
    # print(full_data)
    peaks, _ = find_peaks(full_data['btc_price'])
    clean_peaks = []
    
    # First pass to identify initial peaks
    if len(peaks) > 0:
        for i, peak in enumerate(peaks):
            if not any(full_data['btc_price'][peak] < full_data['btc_price'][past_peak] for past_peak in peaks[:i]):
                if peak + 1 < len(full_data):
                    right_side = full_data['btc_price'][peak+1:]
                    if len(right_side) > 0 and (right_side < (full_data['btc_price'][peak] * (1 - threshold))).any():
                        clean_peaks.append(peak)
    
    # Convert to numpy array for easier manipulation
    clean_peaks = np.array(clean_peaks)
    
    # Enforce one peak per d_days using a backwards window d_days
    final_peaks = []
    if len(clean_peaks) > 0:
        final_peaks.append(clean_peaks[0])
        for i in range(1, len(clean_peaks)):
            if clean_peaks[i] - final_peaks[-1] >= d_days:
                final_peaks.append(clean_peaks[i])
            elif full_data['btc_price'][clean_peaks[i]] > full_data['btc_price'][final_peaks[-1]]:
                final_peaks[-1] = clean_peaks[i]
    
    final_peaks = np.array(final_peaks)

    # Third pass to ensure each peak is followed by a significant drop within drop_window
    # verified_peaks = []
    # for peak in final_peaks:
    #     drop_occurred = False
    #     for j in range(peak + 1, min(peak + drop_window + 1, len(full_data))):
    #         if full_data['btc_price'][j] <= full_data['btc_price'][peak] * (1 - drop_pct):
    #             drop_occurred = True
    #             break
    #     if drop_occurred:
    #         verified_peaks.append(peak)

    # verified_peaks = np.array(verified_peaks)
    verified_peaks = final_peaks

    clean_peaks_after_year = verified_peaks
    
    return clean_peaks_after_year

def get_troughs(full_data, d_days):
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
    return clean_troughs_after_year

def get_params_and_r2(full_data_after_year, clean_peaks_after_year, clean_troughs_after_year):
    print('Getting params and r2')
    print(clean_peaks_after_year)
    peak_dates = full_data_after_year.index[clean_peaks_after_year]
    peak_prices = full_data_after_year['btc_price'].iloc[clean_peaks_after_year]

    trough_dates = full_data_after_year.index[clean_troughs_after_year]
    trough_prices = full_data_after_year['btc_price'].iloc[clean_troughs_after_year]

    if len(peak_dates) < 2 or len(trough_dates) < 2:
        full_data_after_year['model_price_peaks'] = np.nan
        full_data_after_year['model_price_troughs'] = np.nan
        return None, None, None, None, peak_prices, trough_prices

    log_peak_dates = np.log(peak_dates)
    log_peak_prices = np.log(peak_prices)
    log_trough_dates = np.log(trough_dates)
    log_trough_prices = np.log(trough_prices)

    peak_model_params = np.polyfit(log_peak_dates, log_peak_prices, 1)
    trough_model_params = np.polyfit(log_trough_dates, log_trough_prices, 1)

    # full_data_after_year['model_price_peaks'] = np.exp(peak_model_params[1]) * full_data_after_year.index ** peak_model_params[0]
    # full_data_after_year['model_price_troughs'] = np.exp(trough_model_params[1]) * full_data_after_year.index ** trough_model_params[0]
    full_data_after_year.loc[:, 'model_price_peaks'] = np.exp(peak_model_params[1]) * full_data_after_year.index ** peak_model_params[0]
    full_data_after_year.loc[:, 'model_price_troughs'] = np.exp(trough_model_params[1]) * full_data_after_year.index ** trough_model_params[0]


    peak_predicted = np.exp(peak_model_params[1]) * peak_dates ** peak_model_params[0]
    SS_res_peaks = np.sum((peak_prices - peak_predicted) ** 2)
    SS_tot_peaks = np.sum((peak_prices - np.mean(peak_prices)) ** 2)
    r2_peaks = 1 - (SS_res_peaks / SS_tot_peaks)

    trough_predicted = np.exp(trough_model_params[1]) * trough_dates ** trough_model_params[0]
    SS_res_troughs = np.sum((trough_prices - trough_predicted) ** 2)
    SS_tot_troughs = np.sum((trough_prices - np.mean(trough_prices)) ** 2)
    r2_troughs = 1 - (SS_res_troughs / SS_tot_troughs)
    return peak_model_params, trough_model_params, r2_peaks, r2_troughs, peak_prices, trough_prices

def get_figure(full_data_after_year, clean_peaks_after_year, clean_troughs_after_year, max_year, current_date, min_y, max_y,symbol, ax=None):
    print('Getting figure')
    print('Clean peaks:', clean_peaks_after_year)
    print('Clean troughs:', clean_troughs_after_year)
    if len(clean_peaks_after_year)>0 and len(clean_troughs_after_year)>0 :
        print('Getting params')
        params = get_params_and_r2(full_data_after_year, clean_peaks_after_year, clean_troughs_after_year)
        
        if params[0] is None or params[1] is None:
            peak_model_params, trough_model_params, r2_peaks, r2_troughs, peak_prices, trough_prices = [None] * 6
        else:
            peak_model_params, trough_model_params, r2_peaks, r2_troughs, peak_prices, trough_prices = params
        
        if peak_model_params is not None and trough_model_params is not None:
            mid_model_params = [(peak_model_params[0] + trough_model_params[0]) / 2, (peak_model_params[1] + trough_model_params[1]) / 2]
        else:
            mid_model_params = [None, None]
    else:
        peak_model_params, trough_model_params, r2_peaks, r2_troughs, mid_model_params, peak_prices, trough_prices = [None] * 7
        mid_model_params = [None, None]
    print('first date: ',full_data_after_year['date'])
    future_dates = pd.date_range(start=full_data_after_year['date'].iloc[-1], end=str(max_year)+'-12-31', freq='D')
    future_dates = future_dates[~future_dates.isin(full_data_after_year['date'])]

    full_dates = pd.concat([full_data_after_year['date'], pd.Series(future_dates)])
    full_index = np.arange(len(full_dates))

    combined_model_price_peaks = np.exp(peak_model_params[1]) * np.array(full_index) ** peak_model_params[0] if peak_model_params is not None else None
    combined_model_price_troughs = np.exp(trough_model_params[1]) * np.array(full_index) ** trough_model_params[0] if trough_model_params is not None else None
    combined_model_price_mid = np.exp(mid_model_params[1]) * np.array(full_index) ** mid_model_params[0] if mid_model_params[0] is not None else None

    combined_dates = sorted(pd.concat([full_data_after_year['date'], pd.Series(future_dates)]))
    
    # combined_model_price_peaks = np.concatenate([full_data_after_year['model_price_peaks'], future_model_price_peaks]) if future_model_price_peaks is not None else None
    # combined_model_price_troughs = np.concatenate([full_data_after_year['model_price_troughs'], future_model_price_troughs]) if future_model_price_troughs is not None else None
    # combined_model_price_mid = np.concatenate([np.exp(mid_model_params[1]) * full_data_after_year.index ** mid_model_params[0], future_model_price_mid]) if mid_model_params[0] is not None else None
    combined_btc_price = pd.concat([full_data_after_year['btc_price'], pd.Series(np.nan, index=future_dates)])
    print('combined_dates:', combined_dates)    



    if ax is None:
        fig, axs = plt.subplots(1, 3, figsize=(15, 8))
    else:
        fig, axs = plt.gcf(), ax

    for ax_i in axs:
        ax_i.clear()

    axs[0].plot(combined_dates, combined_btc_price, label=f'{symbol} price')
    if peak_prices is not None:
        axs[0].plot(full_data_after_year['date'].iloc[clean_peaks_after_year], peak_prices, "x", color='red')
    if trough_prices is not None:
        axs[0].plot(full_data_after_year['date'].iloc[clean_troughs_after_year], trough_prices, "x", color='green')
    if combined_model_price_peaks is not None:
        axs[0].plot(combined_dates, combined_model_price_peaks, color='red', label='Peak Model')
    if combined_model_price_troughs is not None:
        axs[0].plot(combined_dates, combined_model_price_troughs, color='green', label='Trough Model')
    if combined_model_price_mid is not None:
        axs[0].plot(combined_dates, combined_model_price_mid, color='orange', label='Mid Model')
    axs[0].set_title(f'{symbol} price')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].yaxis.set_major_formatter('${:,.0f}'.format)
    axs[0].legend()
    axs[0].set_ylim([0, max_y])  # Set max for the first axis
    axs[0].set_xlim([combined_dates[0], combined_dates[-1]])

    axs[1].plot(combined_dates, combined_btc_price)
    if peak_prices is not None:
        axs[1].plot(full_data_after_year['date'].iloc[clean_peaks_after_year], peak_prices, "x", color='red')
    if trough_prices is not None:
        axs[1].plot(full_data_after_year['date'].iloc[clean_troughs_after_year], trough_prices, "x", color='green')
    if combined_model_price_peaks is not None:
        axs[1].plot(combined_dates, combined_model_price_peaks, color='red')
    if combined_model_price_troughs is not None:
        axs[1].plot(combined_dates, combined_model_price_troughs, color='green')
    if combined_model_price_mid is not None:
        axs[1].plot(combined_dates, combined_model_price_mid, color='orange')
    axs[1].set_title(f'{symbol} price (log)')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Price')
    axs[1].set_yscale('log')
    axs[1].set_ylim([min_y, max_y])  # Set log y range
    axs[1].set_xlim([combined_dates[0], combined_dates[-1]])

    axs[2].plot(range(len(combined_dates)), combined_btc_price)
    if peak_prices is not None:
        axs[2].plot(full_data_after_year.index[clean_peaks_after_year], peak_prices, "x", color='red')
    if trough_prices is not None:
        axs[2].plot(full_data_after_year.index[clean_troughs_after_year], trough_prices, "x", color='green')
    if combined_model_price_peaks is not None:
        axs[2].plot(range(len(combined_dates)), combined_model_price_peaks, color='red')
    if combined_model_price_troughs is not None:
        axs[2].plot(range(len(combined_dates)), combined_model_price_troughs, color='green')
    if combined_model_price_mid is not None:
        axs[2].plot(range(len(combined_dates)), combined_model_price_mid, color='orange')
    axs[2].set_title(f'{symbol} price (log-log)')
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('Price')
    axs[2].set_yscale('log')
    axs[2].set_xscale('log')
    axs[2].set_ylim([min_y, max_y])  # Set log y range
    max_index = len(combined_dates) - 1
    axs[2].set_xticks([1, 10, 100, 1000, max_index])
    axs[2].set_xticklabels([combined_dates[1].year,combined_dates[10].year, combined_dates[100].year, combined_dates[1000].year, combined_dates[max_index].year])
    axs[2].yaxis.set_major_formatter('${:,.0f}'.format)
    axs[2].set_xlim([1, len(combined_dates) - 1])

    subtitle = f"Current Date: {current_date.strftime('%Y-%m-%d') if current_date else 'N/A'}\n"
    if peak_model_params is not None:
        subtitle += f"Peak Model Params: a={peak_model_params[0]:.4f}, b={peak_model_params[1]:.4f}, R²={r2_peaks:.4f}\n"
    else:
        subtitle += "Peak Model Params: waiting for peaks\n"
    if trough_model_params is not None:
        subtitle += f"Trough Model Params: a={trough_model_params[0]:.4f}, b={trough_model_params[1]:.4f}, R²={r2_troughs:.4f}\n"
    else:
        subtitle += "Trough Model Params: waiting for troughs\n"

    if mid_model_params[0] is not None:
        subtitle += f"Mid Model Params: a={mid_model_params[0]:.4f}, b={mid_model_params[1]:.4f}\n"
    else:
        subtitle += "Mid Model Params: waiting for peaks and troughs\n"
    
    fig.suptitle(subtitle)

    plt.subplots_adjust(top=0.8, wspace=0.3)    

    return fig, axs


def load_data(symbol, start_date):

    if symbol == 'BTC-USD':
        historical_data_btc = pd.read_csv('historical_btc.csv')
        historical_data_btc['date'] = pd.to_datetime(historical_data_btc['date'])
        last_date = historical_data_btc['date'].max()
        new_data_btc = yf.download(symbol, start=start_date)
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
        print('Max date loaded:', max_date)

    if symbol == 'SPY':
        full_data['btc_price'] = full_data['btc_price'] * 10   

    return full_data

def animate(i, full_data, min_peak_date, d_days, max_year, axs, min_year_ani, threshold, min_y,max_y, symbol):
    print('Animating frame', i)
    print('full data min:', full_data['date'].min())
    print('full data max:', full_data['date'].max())
    print('min_year_ani:', min_year_ani)
    min_data_year = full_data['date'].min().year
    if min_year_ani <= min_data_year:
        min_year_ani = min_data_year+1

    current_date = pd.Timestamp(f'{min_year_ani}-01-01') + pd.DateOffset(months=i)
    print('Current date:', current_date)
    full_data_until_now = full_data[full_data['date'] <= current_date].copy()
    full_data_peak_trough = full_data_until_now[full_data_until_now['date'] > min_peak_date].copy() if min_peak_date else full_data_until_now.copy()
   
    

    max_d = full_data_peak_trough['date'].max()
    min_d = full_data_peak_trough['date'].min()
    print(f'Full data max: {max_d}, min: {min_d}')
    removed_dates = full_data_until_now[full_data_until_now['date'] <= min_peak_date].shape[0] if min_peak_date else 0
    print('Removed dates:', removed_dates)

    

    clean_peaks_after_date = get_peaks(full_data_peak_trough, d_days, threshold)
    print('Peaks after year:', clean_peaks_after_date)
    clean_peaks = clean_peaks_after_date + removed_dates
    print('Clean peaks:', clean_peaks)


    clean_troughs_after_date = get_troughs(full_data_peak_trough, d_days)
    print('Troughs after year:', clean_troughs_after_date)
    clean_troughs = clean_troughs_after_date + removed_dates
    print('Clean troughs:', clean_troughs)


    axs = get_figure(full_data_until_now, clean_peaks, clean_troughs, max_year, current_date, min_y, max_y, symbol, ax=axs)
    return axs


# min_year_params = 2011
min_year_ani = None #2015
max_year_ani = 2027
d_days = 365


months_step = 24
threshold=0.3
# drop_pct=0.25
# drop_window=30
min_y = 1e0
max_y = 1e3
max_year = 2035
yahoo_start_date = None #'2010-01-01'
min_peak_date = None
symbol = 'NVDA'


full_data = load_data(symbol, yahoo_start_date)
if min_year_ani is None:
    min_year_ani = full_data['date'].min().year

fig, axs = plt.subplots(1, 3, figsize=(14, 7))
ani = FuncAnimation(fig, animate, frames=range(0, (max_year_ani-min_year_ani)*12 + 1,months_step), fargs=(full_data, min_peak_date, d_days, max_year, axs, min_year_ani, threshold, min_y, max_y, symbol), interval=200)

ani.save('testing_NVDA_1.gif', writer=PillowWriter(fps=1))

plt.show()
