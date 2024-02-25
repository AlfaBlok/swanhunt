import multiprocessing
from datetime import timedelta
import pandas as pd
import imageio
import time
from utils import *

# Assuming get_5charts is a function that generates and saves charts
def get_5charts_wrapper(args):
    return get_5charts(*args)

if __name__ == "__main__":
    montecarlo_simulations = 5000
    coins = ['bitcoin', 'ethereum', 'solana', 'uniswap', 'chainlink']
    min_x = -1
    max_x = 10
    bins_yearly = 120
    min_x_daily = -0.3
    max_x_daily = 1.3
    bins_daily = 600
    bin_kelly_fractions = 500
    y_max_ax1 = 0.05
    y_max_ax2 = 1/bins_yearly*10#0.05
    y_max_ax3 = 1
    ax4_ymax = y_max_ax3

    swan_size = -1
    swan_frequency = 1/10

    coins = ['bitcoin', 'ethereum', 'solana', 'uniswap', 'chainlink']
    min_price_date = pd.to_datetime('2022-01-01').date()
    max_price_date = pd.to_datetime('2023-02-15').date()
    start_date = pd.to_datetime('2022-01-20').date()
    information_window_tail_days = 30
    sequence_length = 60
    interval_days = 1
    

    merged_df = get_price_data(coins, 3000)
    dates = [start_date + timedelta(days=i*interval_days) for i in range(sequence_length)]
    args_list = [(date, merged_df, coins, information_window_tail_days, min_x_daily, max_x_daily, min_x, max_x, y_max_ax2, y_max_ax1, bins_daily, bins_yearly, swan_size, swan_frequency, bin_kelly_fractions, y_max_ax3, ax4_ymax, min_price_date, max_price_date, date.strftime("%Y-%m-%d")) for date in dates]

    pool = multiprocessing.Pool(processes=13)
    pool.map(get_5charts_wrapper, args_list)
    pool.close()
    pool.join()

    filenames = [f"{date.strftime('%Y-%m-%d')}_ax1.png" for date in dates]
    
    with imageio.get_writer('charts5_log_99_BLING.mp4', fps=1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
