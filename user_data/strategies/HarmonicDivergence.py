# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
from typing import List
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from collections import deque

class HarmonicDivergence(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 100
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.07

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            "pivot_lows": {
                "pivot_lows": {'type': 'scatter', 'color': 'red'}
            },
            "pivot_highs": {
                "pivot_highs": {'type': 'triangle', 'color': 'green'}
            }
        },
        'subplots': {            
            "pivot_highs": {
                "pivot_highs": {'mode': 'markers', 'type': 'scatter', 'color': 'green'}
            },
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        
        # Momentum Indicators
        # ------------------------------------

        # Awesome Oscillator
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        # Ultimate Oscillator
        dataframe['uo'] = ta.ULTOSC(dataframe)
        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # Keltner Channel
        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upperband"] = keltner["upper"]
        dataframe["kc_middleband"] = keltner["mid"]
        dataframe["kc_lowerband"] = keltner["lower"]
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        # EMA - Exponential Moving Average
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        pivots = pivot_points(dataframe)
        dataframe['pivot_lows'] = pivots['pivot_lows']
        dataframe['pivot_highs'] = pivots['pivot_highs']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # Signal: RSI crosses above 30
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], 70)) &  # Signal: RSI crosses above 70
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe

def divergence_finder_dataframe(dataframe: DataFrame, indicator_source: str) -> pd.Series:
    lastClose = 0
    lastIndicator = 0
    lastDate = 0
    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        close = row.close
        date = row.date
        indicator = getattr(row, indicator_source)
        divergence_finder([close, lastClose], [indicator, lastIndicator], [date, lastDate], index)
        lastClose = close
        lastIndicator = indicator
        lastDate = date
    return None

def divergence_finder(close: List[float], indicator: List[float], date: List[float], index: int):
    dontconfirm = False # Should we wait 1 extra bar to confirm the divergence
    divlen = 0 #
    pivot_period = 5 # Between 1 and 50. How many bars to use for a pivot.

    if dontconfirm or indicator[0] > indicator[1] or close > close[1]:
        startpoint = 0 if dontconfirm else 1
        #for x in range(0, 10):
        #    len = bar_index - array.get(pl_positions, x) + prd

    return True

def pivot_points(dataframe: DataFrame, window: int = 5) -> DataFrame:
    pivot_points_lows = np.empty(len(dataframe['close'])) * np.nan
    pivot_points_highs = np.empty(len(dataframe['close'])) * np.nan
    last_values = deque()
    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        last_values.append(row)
        if len(last_values) >= window * 2 + 1:
            last_values.popleft()
            current_value = last_values[window - 1]
            is_greater = True
            is_less = True
            for index in range(0, window):
                left = last_values[index]
                right = last_values[window - index]
                if current_value.high < left.high or current_value.high < right.high:
                    is_greater = False
                if current_value.low > left.low or current_value.low > right.low:
                    is_less = False
            if is_greater:
                pivot_points_lows[index] = row.high
            if is_less:
                pivot_points_lows[index] = row.low
    return pd.DataFrame(index=dataframe.date, data={
        'pivot_lows': pivot_points_lows,
        'pivot_highs': pivot_points_highs
    })