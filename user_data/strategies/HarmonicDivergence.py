# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
from typing import List, Tuple
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
                "plotly": {
                    'mode': 'markers',
                    'marker': {
                        'symbol': 'diamond-open',
                        'size': 11,
                        'line': {
                            'width': 2
                        },
                        'color': 'olive'
                    }
            }},
            "pivot_highs": {
                "plotly": {
                    'mode': 'markers',
                    'marker': {
                        'symbol': 'diamond-open',
                        'size': 11,
                        'line': {
                            'width': 2
                        },
                        'color': 'violet'
                    }
            }},
        },
        'subplots': {
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
        dataframe['divergence_ema'] = divergence_finder_dataframe(dataframe, 'ema9')

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
    divergences = np.empty(len(dataframe['close'])) * np.nan
    low_iterator = []
    high_iterator = []
    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        if row['pivot_lows'] != np.nan:
            low_iterator.append(index)
        else:
            low_iterator.append(low_iterator[-1])
        if row['pivot_highs'] != np.nan:
            high_iterator.append(index)
        else:
            high_iterator.append(high_iterator[-1])

    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        if divergence_finder(dataframe,
            dataframe[indicator_source],
            low_iterator,
            high_iterator,
            index
        ):
            divergences[index] = row["close"]
    return divergences

def divergence_finder(dataframe, indicator, low_iterator, high_iterator, index):
    if high_iterator[index] == index:
        current_pivot = high_iterator[index]
        prev_pivot = high_iterator[index - 1]
        if ((dataframe['pivot_highs'][current_pivot] < high_iterator['pivot_highs'][prev_pivot] and indicator[current_pivot] > indicator[prev_pivot])
        or (dataframe['pivot_highs'][current_pivot] > high_iterator['pivot_highs'][prev_pivot] and indicator[current_pivot] < indicator[prev_pivot])):
            slope1 = (dataframe['pivot_highs'][current_pivot] - high_iterator['pivot_highs'][prev_pivot]) / (current_pivot - prev_pivot)
            slope2 = (indicator[current_pivot] - indicator[prev_pivot]) / (current_pivot - prev_pivot)
            return True

    return False

from enum import Enum

class PivotSource(Enum):
    HighLow = 0
    Close = 1

def check_if_pivot_is_greater_or_less(current_value, high_source: str, low_source: str, left, right) -> Tuple[bool, bool]:
    is_greater = True
    is_less = True
    if (getattr(current_value, high_source) < getattr(left, high_source) or
        getattr(current_value, high_source) < getattr(right, high_source)):
        is_greater = False
    if (getattr(current_value, low_source) > getattr(left, low_source) or
        getattr(current_value, low_source) > getattr(right, low_source)):
        is_less = False
    return (is_greater, is_less)

def pivot_points(dataframe: DataFrame, window: int = 5, pivot_source: PivotSource = PivotSource.Close) -> DataFrame:
    high_source = None
    low_source = None
    if pivot_source == PivotSource.Close:
        high_source = 'close'
        low_source = 'close'
    elif pivot_source == PivotSource.HighLow:
        high_source = 'high'
        low_source = 'low'
    pivot_points_lows = np.empty(len(dataframe['close'])) * np.nan
    pivot_points_highs = np.empty(len(dataframe['close'])) * np.nan
    last_values = deque()
    # find pivot points
    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        last_values.append(row)
        if len(last_values) >= window * 2 + 1:
            current_value = last_values[window]
            is_greater = True
            is_less = True
            for window_index in range(0, window):
                left = last_values[window_index]
                right = last_values[2 * window - window_index]
                local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(current_value, high_source, low_source, left, right)
                is_greater &= local_is_greater
                is_less &= local_is_less
            if is_greater:
                pivot_points_highs[index - window] = getattr(current_value, high_source)
            if is_less:
                pivot_points_lows[index - window] = getattr(current_value, low_source)
            last_values.popleft()
    # find last one
    if len(last_values) >= window + 2:
        current_value = last_values[-2]
        is_greater = True
        is_less = True
        for window_index in range(0, window):
            left = last_values[-2 - window_index - 1]
            right = last_values[-1]
            local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(current_value, high_source, low_source, left, right)
            is_greater &= local_is_greater
            is_less &= local_is_less
        if is_greater:
            pivot_points_highs[index - 1] = getattr(current_value, high_source)
        if is_less:
            pivot_points_lows[index - 1] = getattr(current_value, low_source)
        print((is_greater, is_less))
    return pd.DataFrame(index=dataframe.index, data={
        'pivot_lows': pivot_points_lows,
        'pivot_highs': pivot_points_highs
    })