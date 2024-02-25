import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick
# we import funcanimation from the matplotlib library
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from IPython.display import HTML
import ccxt
import datetime
import pandas as pd
from pycoingecko import CoinGeckoAPI
import pandas as pd
from datetime import datetime, timedelta
# we hide warnings in the notebook
import warnings
warnings.filterwarnings('ignore')
import time
from scipy.ndimage import gaussian_filter1d
import imageio
import matplotlib.dates as mdates


def convert_unix_to_datetime(unix_timestamp):
    # Convert Unix timestamp to datetime object
    dt = datetime.utcfromtimestamp(unix_timestamp)
    converted_datetime_str = dt.strftime('%Y-%m-%d')
    return dt


def unix_timestamp_to_date(unix_timestamp):
    """
    Convert a Unix timestamp to a date (without time).

    Parameters:
    - unix_timestamp: An integer or float representing the Unix timestamp.

    Returns:
    - A datetime.date object representing the date.
    """
    # Convert the Unix timestamp to a datetime object
    dt_object = datetime.utcfromtimestamp(unix_timestamp)
    
    # Extract and return just the date part

    

def fetch_historical_data(coin_id, days='max', interval='daily'):
    """
    Fetch historical market data for a specific coin from CoinGecko.
    
    :param coin_id: The CoinGecko ID of the coin (e.g., 'bitcoin').
    :param days: The number of days of historical data to fetch ('max' for as much as possible).
    :param interval: The interval of the data ('daily').
    :return: A DataFrame with the historical data.
    """
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days, interval=interval)
    # print(data['prices'])
    # print(data['prices'][0])
    dates = [unix_timestamp_to_date(ms[0]/1000) for ms in data['prices']]
    # print(dates)
    prices = [price[1] for price in data['prices']]
    # print(prices)
    # we convert timestamp 1550707200000 to datetime
    # datetime.fromtimestamp(1708128000000/1000.0)
    # print(datetime.fromtimestamp(1550707200000/1000.0))
    # print(pd.to_datetime(1708128000000/1000).date().strftime('%Y-%m-%d'))
    
    df = pd.DataFrame(data={'date': dates, 'price': prices})
    return df

def get_daily_returns(returns_df,coin):
    daily_returns = returns_df.copy()
    # for each row we calculate how many days since date of previous row
    daily_returns['days_since'] = daily_returns['date'].diff().dt.days
    # we calculate the daily return by undoing the cumulative product
    # for each row after the first one, we calculate the daily return as the nth root of the cumulative (1 + return) product
    daily_returns['gross_return'] = daily_returns[coin] / daily_returns[coin].shift(1)
    daily_returns['daily_return'] = daily_returns['gross_return'] ** (1 / daily_returns['days_since'])-1
    # we check if the daily return is correct by calculating the cumulative product doing (1 + daily_return	) ** days_since
    # daily_returns['gross_check'] = ( daily_returns['daily_return']) ** daily_returns['days_since']
    # we fill the first row with 1
    # daily_returns['daily_return'].iloc[0] = 1
    
    # return daily_returns
    return daily_returns['daily_return'].to_list()


# get_daily_returns(merged_df[['Date','spy_close']]).head()
# print(get_daily_returns(merged_df[['Date','spy_close']]))
    

    

def get_pdf(data, bins, min_x=-0.4, max_x=0.4):
    bin_edges = np.linspace(min_x, max_x, bins + 1)  # bins + 1 because np.linspace includes both ends
    # data = np.clip(data, min_x, max_x)  # Clip the data to the range
    counts, _ = np.histogram(data, bins=bin_edges, density=True)  # Use density=True for PDF
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_width = bin_edges[1] - bin_edges[0]  # Width of each bin
    counts = counts / counts.sum()

    return bin_centers, counts, bin_width

def find_month_starts(df, num_months=24):
    end_date = df['Date'].max()
    start_date = end_date - pd.DateOffset(months=num_months)
    month_starts = pd.date_range(start=start_date, end=end_date, freq='MS')
    # make sure the date is a datetime and not a timestamp
    month_starts = month_starts.to_pydatetime()
    # print(month_starts)
    return month_starts

def percentage_formatter(x, pos):
    return '{:0.0f}%'.format(x*100)



# we try the new method of calculating annual returns using the daily returns and a monter carlo simulation

def generate_random_daily_returns_from_pdf(bin_centers, bin_counts, size=(500, 365)):
    # print('bin_centers:', bin_centers)
    # print('99 bin_counts:', bin_counts)
    probabilities = bin_counts / np.sum(bin_counts)
    daily_returns = np.random.choice(bin_centers, size=size, p=probabilities)
    return daily_returns

def simulate_annual_returns(daily_returns):
    annual_returns = np.prod(1 + daily_returns, axis=1) - 1
    return annual_returns

def get_annual_returns_pdf(annual_returns, bins=30):
    # Calculate the histogram of annual returns
    counts, bin_edges = np.histogram(annual_returns, bins=bins, density=True)
    # Calculate bin centers from edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0] 
    return bin_centers, counts, bin_width


def get_stochastic_annual_median_return(merged_df, montecarlo_simulations, iterations_in_a_year, min_x, max_x, min_x_daily, max_x_daily, bins_daily=600, bins_yearly=88):
    # print(merged_df.tail(tail))
    bin_centers_btc, bin_counts_btc, bin_width_btc = get_pdf(merged_df, bins=bins_daily, min_x=min_x_daily, max_x=max_x_daily)
    daily_montecarlo_returns_btc = generate_random_daily_returns_from_pdf(bin_centers_btc, bin_counts_btc, size=(montecarlo_simulations, iterations_in_a_year))
    annual_returns_btc = simulate_annual_returns(daily_montecarlo_returns_btc)
    annual_returns_btc_clipped = np.clip(annual_returns_btc, min_x, max_x)
    annual_centers_btc, annual_counts_btc, annual_bin_width = get_annual_returns_pdf(annual_returns_btc_clipped, bins=bins_yearly)
    # we normalize the counts
    annual_counts_btc = annual_counts_btc / annual_counts_btc.sum()
    median = np.percentile(annual_returns_btc, 50)

    annual_centers_log = np.log1p(annual_centers_btc)
    log_mean = np.dot(annual_centers_log, annual_counts_btc)
    log_mean_e = np.expm1(log_mean)

    return median, annual_centers_btc, annual_counts_btc, annual_bin_width, log_mean_e
# we clip the annual_centers between -1 and 2


def add_annual_stochastic_returns(date,merged_df, annualization_window_tail_days,information_window_tail_days, montecarlo_simulations, coins, iterations_in_a_year, min_x, max_x, min_x_daily, max_x_daily, bins_daily, bins_yearly):
    date = pd.to_datetime(date)
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df=merged_df[merged_df['Date'] <= date]
    min_tail_date = date - pd.DateOffset(days=annualization_window_tail_days+information_window_tail_days)
    merged_df=merged_df[(merged_df['Date'] >= min_tail_date)]
   

    for coin in coins:

        # print(tail_date)

        # for row in merged_df[(merged_df['Date'] >= tail_date) & (merged_df['Date'] <= date)].itertuples():
        for row in merged_df.itertuples():
            date = row.Date
            median_yearly, _, _,_, _ = get_stochastic_annual_median_return(merged_df[coin+'_daily_return'], montecarlo_simulations, iterations_in_a_year, min_x, max_x, min_x_daily, max_x_daily, bins_daily, bins_yearly)
            merged_df.loc[merged_df['Date'] == date, coin+'_yearly_return'] = median_yearly
        
    return merged_df


# Define the find_nearest function
def find_nearest(all_dates, target_date):
    # Calculate the absolute difference between each date in all_dates and the target_date
    differences = [abs(date - target_date) for date in all_dates]
    # Find the index of the smallest difference
    index_nearest = differences.index(min(differences))
    return index_nearest



def add_swan(bin_centers, counts, swan_size, swan_frequency, swan_f = 1):
    bin_centers = bin_centers.tolist()
    counts = counts.tolist()
    if swan_frequency > 0:
        # swan_size = -(1-1e-6) if swan_size == -1 else swan_size
        bin_centers.append(swan_size*swan_f)
        counts.append(swan_frequency)
        # we normalize the array_2
        counts = np.array(counts) / np.sum(counts)
    # bin_centers_spy = np.sort(bin_centers_spy)
    bin_centers = np.array(bin_centers)
    counts = np.array(counts)
    return bin_centers, counts

def add_log_swan(bin_centers, counts, swan_size, swan_frequency, swan_f=1):
    bin_centers = bin_centers.tolist()
    counts = counts.tolist()
    
    if swan_frequency > 0:
        bin_centers.append(np.log1p(swan_size*swan_f))
        counts.append(swan_frequency)
        counts = np.array(counts) / np.sum(counts)
        
    return bin_centers, counts



def process_kelly(bin_kelly_fractions, annual_bin_centers, annual_counts):
    annual_bin_centers_swan, annual_counts_swan = add_swan(annual_bin_centers, annual_counts, swan_size, swan_frequency)

    exp_g_array = []
    fr = np.linspace(0,1,bin_kelly_fractions)
    for f in fr:
        annual_bin_centers_swan_f = annual_bin_centers_swan * f
        annual_bin_centers_swan_log_f = np.log1p(annual_bin_centers_swan_f)
        exp_g = np.dot(annual_bin_centers_swan_log_f, annual_counts_swan)
        exp_g = np.expm1(exp_g)
        exp_g_array.append(exp_g)
    max_f = fr[np.argmax(exp_g_array)]
    max_g = np.max(exp_g_array)
    return max_f, exp_g_array, fr, max_g


def get_ax1_chart(ax1, daily_bin_centers, daily_counts, daily_bin_width, min_x_daily, max_x_daily, y_max_ax1,date):
    ax1.bar(daily_bin_centers, daily_counts, width=daily_bin_width, align='center') 
    ax1.set_title(f"Daily Returns Distribution {information_window_tail_days} days ending in {date.strftime('%Y-%m-%d')}")
    ax1.set_xlabel('Daily Return')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(min_x_daily*0.5, max_x_daily*0.2)
    ax1.set_ylim(0.001, y_max_ax1)
    ax1.xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax1.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    return ax1


def get_ax2_chart(ax2, annual_centers, annual_counts, annual_bin_width, min_x, max_x, y_max_ax2, log_mean_e, coin, log_mean, index_for):
    ax2.bar(annual_centers, annual_counts, width=annual_bin_width, align='center')  # align='center' to center the bars
    ax2.axvline(x=log_mean_e if log_mean_e < max_x else max_x, color='g' if log_mean_e > 0 else 'r', linestyle='--', label='Mean return')
    ax2.set_title(f"Annual Returns Distribution last {information_window_tail_days} days")
    ax2.set_xlabel('Annual Return')
    ax2.set_ylabel('Frequency')
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(0.001, y_max_ax2)
    ax2.text(log_mean_e, y_max_ax2*0.9*(1-0.1*index_for), f'{coin}: {log_mean_e*100:.0f}%', rotation=0, verticalalignment='bottom', horizontalalignment='left' if log_mean < 0 else 'right', color='g' if log_mean_e > 0 else 'r', fontsize=6)
    ax2.xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax2.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    return ax2
        
def get_ax3_chart(ax3, max_f, exp_g_array, fr, ax3_ymax, coin):
    ax3.plot(fr, exp_g_array)
    max_g = np.max(exp_g_array)
    ax3.axvline(x=max_f if max_f < 1 else 1, color='g', linestyle='--', label='Max G')
    ax3.text(max_f, max_g if max_g < ax3_ymax else ax3_ymax, f"{coin}: {max_f*100:.0f}% ", rotation=0, verticalalignment='bottom', horizontalalignment='right' if max_f > 0.5 else 'left', color='g', fontsize=6)
    ax3.set_title(f"Expected G as function of fraction")
    ax3.set_xlabel('Fraction')
    ax3.set_ylabel('Expected G')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-1,ax3_ymax)
    ax3.xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax3.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    
    return ax3


# we make titles smaller size
def set_fontsize_params(ax, title_size=10, label_size=8, tick_size=8):
    ax.title.set_size(10)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    return ax



def get_3charts(date,merged_df, coins, information_window_tail_days, min_x_daily, max_x_daily, min_x, max_x, y_max_ax2,y_max_ax1, bins_daily, bins_yearly, swan_size, swan_frequency,bin_kelly_fractions,ax3_ymax,frame_prefix=None):
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    date = pd.to_datetime(date)

    information_window_df = merged_df[merged_df['Date'] < date]
    information_window_df = information_window_df.tail(information_window_tail_days)
    fig, (ax1, ax2,  ax3) = plt.subplots(3, 1, figsize=(8, 6))

    for coin in coins:
        index_for = coins.index(coin)
        daily_bin_centers, daily_counts, daily_bin_width = get_pdf(information_window_df[coin+'_daily_return'], bins=bins_daily, min_x=min_x_daily, max_x=max_x_daily)
        _, annual_centers, annual_counts, annual_bin_width, log_mean_e = get_stochastic_annual_median_return(information_window_df[coin+'_daily_return'],montecarlo_simulations, iterations_in_a_year, min_x, max_x, min_x_daily, max_x_daily, bins_daily, bins_yearly)
        max_f, exp_g_array, fr, max_g = process_kelly(bin_kelly_fractions, annual_centers, annual_counts)

        ax1 = get_ax1_chart(ax1, daily_bin_centers, daily_counts, daily_bin_width, min_x_daily, max_x_daily, y_max_ax1)
        ax2 = get_ax2_chart(ax2, annual_centers, annual_counts, annual_bin_width, min_x, max_x, y_max_ax2, log_mean_e, coin, log_mean, index_for)
        ax3 = get_ax3_chart(ax3, max_f, exp_g_array, fr, ax3_ymax, coin)

        title_size=10
        label_size=8
        tick_size=8
        args = (title_size, label_size, tick_size) 
        ax1 = set_fontsize_params(ax1, *args)
        ax2 = set_fontsize_params(ax2, *args)
        ax3 = set_fontsize_params(ax3, *args)

    plt.tight_layout() 
    plt.subplots_adjust(hspace=0.8)
    fig.savefig(f"{frame_prefix}_ax1.png")
    return fig, ax1, ax2, ax3
    
def get_ax4_chart(ax4, max_f, coin, max_g, ax4_ymax):
    max_g = max_g if max_g < ax4_ymax else ax4_ymax
    ax4.scatter(max_f, max_g, label='Max G')
    ax4.set_title(f"Max G and Max F")
    ax4.set_xlabel('Max Fraction')
    ax4.set_ylabel('Max G')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0,ax4_ymax)

    ax4.xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax4.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    ax4.text(max_f+0.05, max_g+0.05, f"{coin}", rotation=0, verticalalignment='bottom', horizontalalignment='right' if max_f > 0.5 else 'left', color='g', fontsize=6)
    return ax4


def get_4charts(date,merged_df, coins, information_window_tail_days, min_x_daily, max_x_daily, min_x, max_x, y_max_ax2,y_max_ax1, bins_daily, bins_yearly, swan_size, swan_frequency,bin_kelly_fractions,ax3_ymax,ax4_ymax,frame_prefix=None):
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    date = pd.to_datetime(date)

    information_window_df = merged_df[merged_df['Date'] < date]
    information_window_df = information_window_df.tail(information_window_tail_days)
    fig, ((ax1,ax1r),(ax2,ax2r),(ax3,ax3r)) = plt.subplots(3, 2, figsize=(8, 6))
    max_f_array = []

    for coin in coins:
        index_for = coins.index(coin)
        daily_bin_centers, daily_counts, daily_bin_width = get_pdf(information_window_df[coin+'_daily_return'], bins=bins_daily, min_x=min_x_daily, max_x=max_x_daily)
        _, annual_centers, annual_counts, annual_bin_width, log_mean_e = get_stochastic_annual_median_return(information_window_df[coin+'_daily_return'],montecarlo_simulations, iterations_in_a_year, min_x, max_x, min_x_daily, max_x_daily, bins_daily, bins_yearly)
        max_f, exp_g_array, fr, max_g = process_kelly(bin_kelly_fractions, annual_centers, annual_counts)
        max_f_array.append(max_f)

        ax1 = get_ax1_chart(ax1, daily_bin_centers, daily_counts, daily_bin_width, min_x_daily, max_x_daily, y_max_ax1,date)
        ax2 = get_ax2_chart(ax2, annual_centers, annual_counts, annual_bin_width, min_x, max_x, y_max_ax2, log_mean_e, coin, log_mean, index_for)
        ax3 = get_ax3_chart(ax3, max_f, exp_g_array, fr, ax3_ymax, coin)
        ax1r = get_ax4_chart(ax1r, max_f, coin, max_g,ax4_ymax)

        title_size=10
        label_size=8
        tick_size=8
        args = (title_size, label_size, tick_size) 
        ax1 = set_fontsize_params(ax1, *args)
        ax2 = set_fontsize_params(ax2, *args)
        ax3 = set_fontsize_params(ax3, *args)
        ax1r = set_fontsize_params(ax1r, *args)

    plt.tight_layout() 
    plt.subplots_adjust(hspace=0.8)
    fig.savefig(f"{frame_prefix}_ax1.png")
    return fig #, ax1, ax2, ax3, ax4
    
def get_price_timeline_chart(ax2r, merged_df, information_window_df, coin, date, information_window_tail_days, min_price_date, max_price_date):
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    
    max_price_date = pd.to_datetime(max_price_date)
    min_price_date = pd.to_datetime(min_price_date)
    plot_data_df = merged_df[merged_df['Date'] <= max_price_date]
    plot_data_df = merged_df[merged_df['Date'] >= min_price_date]
    dates = plot_data_df['Date']
    # we normalize coin column by the first value
    plot_data_df[coin] = plot_data_df[coin] / plot_data_df[coin].iloc[0]
    ax2r.plot(dates, plot_data_df[coin], label=coin)
    ax2r.set_title(f"Price Timeline")
    ax2r.set_xlabel('Date')
    ax2r.set_ylabel('Price')
    ax2r.set_xlim(min_price_date, max_price_date)
    ax2r.set_ylim(0,3)
    ax2r.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax2r.xaxis.set_major_locator(mdates.WeekdayLocator(interval=30))
    # we set x axis range from min_price_date to max_price_date
    


    # we add vertical line for the date
    ax2r.axvline(x=date, color='r', linestyle='--', label='Date')

    # we add a shaded area for the information window
    ax2r.axvspan(date - pd.DateOffset(days=information_window_tail_days), date, alpha=0.1, color='green')


    # ax2r.legend()
    return ax2r



def get_5charts(date,merged_df, coins, information_window_tail_days, min_x_daily, max_x_daily, min_x, max_x, y_max_ax2,y_max_ax1, bins_daily, bins_yearly, swan_size, swan_frequency,bin_kelly_fractions,ax3_ymax,ax4_ymax,min_price_date, max_price_date,frame_prefix=None):
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    date = pd.to_datetime(date)

    information_window_df = merged_df[merged_df['Date'] < date]
    information_window_df = information_window_df.tail(information_window_tail_days)
    fig, ((ax1,ax1r),(ax2,ax2r),(ax3,ax3r)) = plt.subplots(3, 2, figsize=(8, 6))
    max_f_array = []

    for coin in coins:
        index_for = coins.index(coin)
        daily_bin_centers, daily_counts, daily_bin_width = get_pdf(information_window_df[coin+'_daily_return'], bins=bins_daily, min_x=min_x_daily, max_x=max_x_daily)
        _, annual_centers, annual_counts, annual_bin_width, log_mean_e = get_stochastic_annual_median_return(information_window_df[coin+'_daily_return'],montecarlo_simulations, iterations_in_a_year, min_x, max_x, min_x_daily, max_x_daily, bins_daily, bins_yearly)
        max_f, exp_g_array, fr, max_g = process_kelly(bin_kelly_fractions, annual_centers, annual_counts)
        max_f_array.append(max_f)
        
        ax1 = get_ax1_chart(ax1, daily_bin_centers, daily_counts, daily_bin_width, min_x_daily, max_x_daily, y_max_ax1,date)
        ax2 = get_ax2_chart(ax2, annual_centers, annual_counts, annual_bin_width, min_x, max_x, y_max_ax2, log_mean_e, coin, log_mean, index_for)
        ax3 = get_ax3_chart(ax3, max_f, exp_g_array, fr, ax3_ymax, coin)
        ax1r = get_ax4_chart(ax1r, max_f, coin, max_g,ax4_ymax)
        ax2r = get_price_timeline_chart(ax2r, merged_df, information_window_df, coin, date, information_window_tail_days, min_price_date, max_price_date)

        title_size=10
        label_size=8
        tick_size=8
        args = (title_size, label_size, tick_size) 
        ax1 = set_fontsize_params(ax1, *args)
        ax2 = set_fontsize_params(ax2, *args)
        ax3 = set_fontsize_params(ax3, *args)
        ax1r = set_fontsize_params(ax1r, *args)

    plt.tight_layout() 
    plt.subplots_adjust(hspace=0.8)
    fig.savefig(f"{frame_prefix}_ax1.png")
    return fig #, ax1, ax2, ax3, ax4
    

def get_price_data(coins, days=3000):
    prices_df_0 = pd.DataFrame(columns=['date', 'price', 'coin'])
    for coin in coins:
        historical_data = fetch_historical_data(coin, days)
        # we get rid of the last reading as it is today's price and it is not complete
        historical_data = historical_data.iloc[:-1]
        historical_data['coin'] = coin
        prices_df_0 = pd.concat([prices_df_0, historical_data])
        # we sleep for 1 second to avoid hitting the CoinGecko API rate limits
        time.sleep(1)
    prices_pivot = prices_df_0.pivot(index='date', columns='coin', values='price')
    prices_pivot.reset_index(inplace=True)
    for coin in coins:
        coin = coin 
        prices_pivot[f'{coin}_daily_return'] = get_daily_returns(prices_pivot[['date', coin]],coin)
        # we also add the daily log1p return which we take from the daily return that we have just calculated
        prices_pivot[f'{coin}_daily_log1p_return'] = np.log1p(prices_pivot[f'{coin}_daily_return'])
    merged_df = prices_pivot
    merged_df = merged_df.rename(columns={"date": "Date"})
    return merged_df


cg = CoinGeckoAPI()