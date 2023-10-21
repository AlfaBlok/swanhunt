import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
# import icecream
# from icecream import ic

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

N = norm.cdf

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)

st.title('Put Option Price as a Function of Volatility')




# Parameters
T = 1  # Time to expiration (in years)
r = 0.05  # Risk-free interest rate
S = 100  # Current stock price

# Volatility range
volatilities = np.linspace(0, 1, 100)

# Strike prices for different moneyness levels
strike_prices = [60,80,100,120,140] #np.linspace(50, 150, 10)

# Plot
fig3, ax3 = plt.subplots()

for K in strike_prices:
    put_prices = [black_scholes_put(S, K, T, r, sigma) for sigma in volatilities]
    # normalization_factor = black_scholes_put(S, K, T, r, 0.10)
    # normalization_factor = 1
    # normalized_prices = [price / normalization_factor for price in put_prices]
    ax3.plot(volatilities, put_prices, label=f'K={K:.0f}')

ax3.set_xlabel('Volatility (σ)')
ax3.set_ylabel('Put Option Price ($) (log scale)')
ax3.set_ylim(0.001, 100)
ax3.set_title('Put Price as function of vol (σ) and Strike (K)')
ax3.grid(False)
for K in strike_prices:
    put_prices = [black_scholes_put(S, K, T, r, sigma) for sigma in volatilities]
    ax3.text(volatilities[-1], put_prices[-1], f'  K={K:.0f}', horizontalalignment='right', verticalalignment='bottom', fontsize=6)
ax3.set_yscale('log')

# we use streamlit cache to show the plot to avoid delays
# @st.cache()
st.pyplot(fig3)



# SECOND PLOT 
fig3, ax3 = plt.subplots()

# st.write("Volatilities: ", volatilities)
factor = st.sidebar.slider('Volatility entry point', 0.1, 0.9, 0.1, 0.1, key='2')
for K in strike_prices:
    put_prices = [black_scholes_put(S, K, T, r, sigma) for sigma in volatilities]
    # st.write(put_prices)
    normalization_factor = black_scholes_put(S, K, T, r, factor)
    # st.write(f"Normalization factor for K= {K}: ", normalization_factor)
    # normalization_factor = 1
    normalized_prices = [price / normalization_factor for price in put_prices]
    ax3.plot(volatilities, normalized_prices, label=f'K={K:.0f}')
    # if text is above 10000, we don't show it
    if normalized_prices[-1] < 10000:
        ax3.text(volatilities[-1], normalized_prices[-1], f'  K={K:.0f}', horizontalalignment='right', verticalalignment='bottom', fontsize=6)
    else:
        ax3.text(volatilities[12], normalized_prices[12], f'  K={K:.0f}', horizontalalignment='right', verticalalignment='top', fontsize=6)

ax3.set_xlabel('Volatility (σ)')
ax3.set_ylabel('Normalized Put Option Price')
ax3.set_ylim(0.1, 10000)
ax3.set_title(f'Put Price as a Function of σ (self-normalized to {factor} σ)')
# ax3.legend()
ax3.grid(False)
# we make Y axis logarithmic
ax3.set_yscale('log')
st.pyplot(fig3)


# THIRD PLOT
# in this plot we show Put price as a function of Time to expiry and K
# Parameters
# T = 1  # Time to expiration (in years)
r = 0.05  # Risk-free interest rate
S = 100  # Current stock price

# expiry range (backwards)
# we use a text input box to get the start time in the sidebar
# start_time =

start_time = st.sidebar.number_input("Time to expiry (days): ", 21)/252 # 1
end_time = 0.0
# st.sidebar.input_box("Start time") start_time = 1
# end_time = 0.0
expiries = np.linspace(end_time, start_time, 100)[::-1]
# we reverse the array

# st.write("Expiries: ", expiries)

# Strike prices for different moneyness levels
strike_prices = [60,80,100,120,140] #np.linspace(50, 150, 10)

sigma = 0.2

# Plot
fig3, ax3 = plt.subplots()

for K in strike_prices:
    put_prices = [black_scholes_put(S, K, T, r, sigma) for T in expiries]
    # normalization_factor = black_scholes_put(S, K, T, r, 0.10)
    # normalization_factor = 1
    # normalized_prices = [price / normalization_factor for price in put_prices]
    ax3.plot(expiries, put_prices, label=f'K={K:.0f}')

ax3.set_xlabel('Time to expiry (T)')
# set reverse X axis
ax3.set_xlim(start_time, end_time)
ax3.set_ylabel('Put Option Price ($) (log scale)')
ax3.set_ylim(0.000001, 100)
ax3.set_title('Put Price as function of time to expiry (T) and Strike (K)')
ax3.grid(False)

for K in strike_prices:
    put_prices = [black_scholes_put(S, K, T, r, sigma) for T in expiries]
    if put_prices[20] > 0.0001:
        ax3.text(expiries[20], put_prices[20], f'  K={K:.0f}', horizontalalignment='right', verticalalignment='top', fontsize=6)
    else:
        pass
        # ax3.text(expiries[10], put_prices[10], f'  K={K:.0f}', horizontalalignment='right', verticalalignment='top', fontsize=6)
ax3.set_yscale('log')
st.pyplot(fig3)

# PLOT 4 SELF NORMALIZED TIME DECAY

fig3, ax3 = plt.subplots()



for K in strike_prices:
    put_prices = [black_scholes_put(S, K, T, r, sigma) for T in expiries]
    normalization_factor = black_scholes_put(S, K, start_time, r, sigma)
    # normalization_factor = 1
    normalized_prices = [price / normalization_factor for price in put_prices]
    ax3.plot(expiries, normalized_prices, label=f'K={K:.0f}')
    if normalized_prices[-1] > 0.0001:
        ax3.text(expiries[-1], normalized_prices[-1], f'  K={K:.0f}', horizontalalignment='right', verticalalignment='top', fontsize=6)
    else:
        ax3.text(expiries[50], normalized_prices[50], f'  K={K:.0f}', horizontalalignment='right', verticalalignment='bottom', fontsize=6)


ax3.set_xlabel('Time to expiry (T)')
# set reverse X axis
ax3.set_xlim(start_time, end_time)
ax3.set_ylabel('Put Option Price ($) (log scale)')
ax3.set_ylim(0.0001, 1.2)
ax3.set_title('Put Price (self normalized) as function of time to expiry (T) and Strike (K)')
ax3.grid(False)

# ax3.set_yscale('log')
st.pyplot(fig3)


# PLOT 5 SELF NORMALIZED TIME DECAY AND VOLATILITY - 2 INDENPENDENT VARIABLES PLOT
fig3, ax3 = plt.subplots()
st.write("")
st.title('Simulation with VIX')
#  we load VIX_History.csv into a dataframe
import pandas as pd
df = pd.read_csv('VIX_History.csv')
#  we only keep the columns we need: Date and Close
df = df[['DATE', 'CLOSE']]

# we only keep the last X months of time, per an input on sidebar
months = st.sidebar.number_input("Months to display: ", 1, 240, 12)
# we convert months to days
days = months * 30
# we only keep the last X days of time
df = df.tail(days)

# we divide the Close column by 100 to get the VIX in %
df['CLOSE'] = df['CLOSE']/100
# we make dates into date format YYYY-MM-DD
df['DATE'] = pd.to_datetime(df['DATE'])


# st.dataframe(df)

#  we read SP500_historical.csv into a dataframe
df2 = pd.read_csv('SP500_historical.csv')
# we rename columns to DATE and CLOSE
df2 = df2.rename(columns={'Date': 'DATE', 'Close': 'CLOSE'})
# we make dates into date format
df2['DATE'] = pd.to_datetime(df2['DATE'])
# we do a left join on the two dataframes, using the DATE column as key, starting with the VIX dataframe
df3 = pd.merge(df, df2, on='DATE', how='left')
# we drop the rows with NaN values
df3 = df3.dropna()
# we rename the columns CLOSE_x and CLOSE_y to VIX and SP500
df3 = df3.rename(columns={'CLOSE_x': 'VIX', 'CLOSE_y': 'SP500'})
# for column SP500, we remove any "," in values and convert to float
df3['SP500'] = df3['SP500'].str.replace(',', '').astype(float)




# we plot both VIX and SP500 on the same chart (2 Y axis)
fig3, ax3 = plt.subplots()
ax3.plot(df3['DATE'], df3['VIX'], label='VIX')
ax3.set_xlabel('Date')
ax3.set_ylabel('VIX')
ax3.set_title('VIX')
ax3.grid(False)
ax3.legend(loc='upper left')
# we add a second Y axis
ax4 = ax3.twinx()
ax4.plot(df3['DATE'], df3['SP500'], label='SP500', color='red')
# ax4.yaxis.set_ticks(np.arange(0, 5000, 500))

# ax4.yaxis.set_ticks(np.arange(min_value, max_value, step_value))
from matplotlib.ticker import FuncFormatter

def thousands_formatter(x, pos):
    return f'{x * 1e-3}K'

ax4.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

ax4.set_ylabel('SP500')
ax4.legend(loc='upper right')
ax4.grid(False)

st.pyplot(fig3)


#  we add a column to the dataframe named "PUT_PRICE"
df3['STRIKE']    = 0.0
df3['PUT_PRICE_FULL_TERM'] = 0.0


# we add to sidebar number input for Strike price
st.sidebar.write("Simulation parameters")
K = st.sidebar.number_input("Strike price: ", 50, 150, 90)/100

# we make STRIKE column = to K * SP500
df3['STRIKE'] = K * df3['SP500']

#  for each date in the dataframe, we calculate the put price and store it in the PUT_PRICE column
for index, row in df3.iterrows():
    # st.write(row['DATE'], row['SP500'], row['VIX'])
    # we calculate the put price using the current SP500 and VIX values
    put_price = black_scholes_put(row['SP500'], row['STRIKE'], start_time, 0.05, row['VIX'])
    # we store the put price in the PUT_PRICE column
    df3.loc[index, 'PUT_PRICE_FULL_TERM'] = put_price


cycle_budget = st.sidebar.number_input("Cycle budget: ", 1, 10000,1000)

# we make the first date in the datframe have TRADE_SIGNAL = "BUY"
df3['TRADE_SIGNAL'] = '0'
df3['CASH'] = 0.0
df3['PUT_POSITION_ENTRY_PRICE'] = 0.0
# df3['EXIT_PRICE'] = 0.0
df3['PUT_POSITION_ENTRY_STRIKE'] = 0.0
# df3['P&L'] = 0.0
df3['PUT_POSITION_TIME_TO_EXPIRY'] = 0.0
df3['PUT_POSITION_DECAYED_UNIT_PRICE'] = 0.0
# df3['PUT_PRICE_T0'] = 0.0
df3['PUT_POSITION_QTY'] = 0.0
df3['PUT_POSITION_VALUE'] = 0.0
df3['TRADE_PL'] = 0.0

df3.loc[1, 'TRADE_SIGNAL'] = 'BUY'
df3.loc[1, 'CASH'] = cycle_budget

# we reset df3 index 
df3 = df3.reset_index(drop=True)

target_profit = st.sidebar.number_input("Profit target: ", 0, 20, 1)

# THIS IS THE MAIN STRATEGY LOOP
# WE SKIP FIRST DATE 
# FOR EACH OTHER DAY, IF PUT POSITION of previous day close = 0 AND TRADE SIGNAL of previous day = BUY, 
#       WE BUY PUTS, AND WE SET ENTRY PRICE = PUT PRICE of previous day, AND WE SET TRADE SIGNAL = HOLD



for index,row in df3.iterrows():
    # FOR EACH OTHER DAY, IF PUT POSITION of previous day close = 0 AND TRADE SIGNAL of previous day = BUY, 
    # ic(index)
    # ic(row['TRADE_SIGNAL'])
    try: 
        if index > 0 and (
             df3.loc[index-1, 'TRADE_SIGNAL'] == 'BUY' or 
             df3.loc[index-1, 'TRADE_SIGNAL'] == 'SOLD' or 
             df3.loc[index-1, 'TRADE_SIGNAL'] == 'EXPIRED'
             ):
            # WE BUY PUTS, AND WE SET ENTRY PRICE = PUT PRICE of previous day, AND WE SET TRADE SIGNAL = HOLD
            df3.loc[index, 'PUT_POSITION_QTY'] =  cycle_budget / df3.loc[index-1, 'PUT_PRICE_FULL_TERM'] / 100
            df3.loc[index, 'PUT_POSITION_ENTRY_PRICE'] = df3.loc[index-1, 'PUT_PRICE_FULL_TERM']
            df3.loc[index, 'PUT_POSITION_ENTRY_STRIKE'] = df3.loc[index-1, 'STRIKE']
            df3.loc[index, 'TRADE_SIGNAL'] = 'HOLD'
            df3.loc[index, 'CASH'] = df3.loc[index-1, 'CASH'] - cycle_budget
            df3.loc[index, 'PUT_POSITION_TIME_TO_EXPIRY'] = start_time*252 -1
            # df3.loc[index, 'PUT_PRICE_T0'] = df3.loc[index-1, 'PUT_PRICE']
            S = df3.loc[index, 'SP500']
            K = df3.loc[index, 'PUT_POSITION_ENTRY_STRIKE']
            T = df3.loc[index, 'PUT_POSITION_TIME_TO_EXPIRY']/252
            r = 0.05
            sigma = df3.loc[index, 'VIX']
            decayed_value = black_scholes_put(S, K, T, r, sigma)
            df3.loc[index, 'PUT_POSITION_DECAYED_UNIT_PRICE'] = decayed_value
            df3.loc[index, 'PUT_POSITION_VALUE'] = df3.loc[index, 'PUT_POSITION_QTY'] * decayed_value * 100
            
        elif index > 0 and df3.loc[index-1, 'TRADE_SIGNAL'] == 'HOLD':
            # WE HOLD PUTS, AND WE SET TRADE SIGNAL = HOLD
            # print("ENTERING HOLD")
            df3.loc[index, 'PUT_POSITION_QTY'] = df3.loc[index-1, 'PUT_POSITION_QTY']
            df3.loc[index, 'PUT_POSITION_ENTRY_PRICE'] = df3.loc[index-1, 'PUT_POSITION_ENTRY_PRICE']
            df3.loc[index, 'PUT_POSITION_ENTRY_STRIKE'] = df3.loc[index-1, 'PUT_POSITION_ENTRY_STRIKE']
            
            df3.loc[index, 'CASH'] = df3.loc[index-1, 'CASH']
            df3.loc[index, 'PUT_POSITION_TIME_TO_EXPIRY'] = df3.loc[index-1, 'PUT_POSITION_TIME_TO_EXPIRY'] - 1
            # df3.loc[index, 'PUT_PRICE_T0'] = df3.loc[index-1, 'PUT_PRICE']
            S = df3.loc[index, 'SP500']
            K = df3.loc[index, 'PUT_POSITION_ENTRY_STRIKE']
            T = df3.loc[index, 'PUT_POSITION_TIME_TO_EXPIRY']/252
            r = 0.05
            sigma = df3.loc[index, 'VIX']
            decayed_value = black_scholes_put(S, K, T, r, sigma)
            df3.loc[index, 'PUT_POSITION_DECAYED_UNIT_PRICE'] = decayed_value
            df3.loc[index, 'PUT_POSITION_VALUE'] = df3.loc[index, 'PUT_POSITION_QTY'] * decayed_value * 100
            df3.loc[index, 'TRADE_PL'] = decayed_value / df3.loc[index-1, 'PUT_POSITION_ENTRY_PRICE'] - 1
            if df3.loc[index, 'TRADE_PL'] >  target_profit:
                df3.loc[index, 'CASH'] = df3.loc[index, 'CASH'] + df3.loc[index, 'PUT_POSITION_VALUE']
                df3.loc[index, 'PUT_POSITION_VALUE'] = 0
                df3.loc[index, 'PUT_POSITION_QTY'] = 0
                df3.loc[index, 'TRADE_SIGNAL'] = 'SOLD'
            elif df3.loc[index, 'PUT_POSITION_TIME_TO_EXPIRY'] <= 0:
                # df3.loc[index, 'CASH'] = df3.loc[index, 'PUT_POSITION_VALUE']
                df3.loc[index, 'PUT_POSITION_VALUE'] = 0
                df3.loc[index, 'PUT_POSITION_QTY'] = 0
                df3.loc[index, 'TRADE_SIGNAL'] = 'EXPIRED'
            else:
                df3.loc[index, 'TRADE_SIGNAL'] = 'HOLD'
        else:
            pass
    except Exception as e:
        print(f'Error: {e}')


# we make CASH column into integers
df3['CASH'] = df3['CASH'].astype(int)

st.dataframe(df3)

def normalize_series(series, min_value, max_value):
    scale = series.max()/max_value
    return series / scale


# we plot the put position value
fig3, ax3 = plt.subplots()
ax3.plot(df3['DATE'], df3['VIX'], label='VIX')
ax3.set_xlabel('Date')
ax3.set_ylabel('VIX')
ax3.legend(loc='upper left')   
ax3.set_title('VIX')
ax3.grid(False)
ax3.legend(loc='upper left')

# we add a second Y axis
ax4 = ax3.twinx()
min = df3['CASH'].min()
max = df3['CASH'].max()
ax4.plot(df3['DATE'], (df3['CASH']), label='CASH', color='orange')
# ax4.plot(df3['DATE'], normalize_series(df3['PUT_POSITION_VALUE'],min,max), label='PUT_POSITION_VALUE', color='red')
ax4.plot(df3['DATE'], normalize_series(df3['SP500'],min,max), label='SP500', color='green')

ax4.set_ylabel('$')
ax4.legend(loc='upper right')   
ax4.grid(False)


st.pyplot(fig3)



