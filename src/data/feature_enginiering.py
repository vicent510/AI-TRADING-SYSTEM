# Libraries Imports
import pandas as pd
import numpy as np
import pandas_ta as ta
import os
import sys
import warnings

# Local imports
from src.utils.utils import getTime, bcolors

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

def rolling_hurst(series, window, lag=10):
    sigma_1 = series.diff(1).rolling(window=window, min_periods=window).std()
    sigma_n = series.diff(lag).rolling(window=window, min_periods=window).std()
    return np.log(safe_divide(sigma_n, sigma_1 + 1e-9)) / np.log(lag)

def get_mtf(base_df, multiplier, suffix):
    group_id = np.arange(len(base_df)) // multiplier

    agg = base_df.groupby(group_id).agg({
        'close': 'last',
        'high':  'max',
        'low':   'min'
    })

    agg_shifted = agg.shift(1)

    if agg_shifted['close'].dropna().shape[0] < 55:
        res = pd.DataFrame(index=base_df.index)
        res[f"trend_{suffix}"] = np.nan
        res[f"rsi_{suffix}"] = np.nan
        res[f"hurst_{suffix}"] = np.nan
        return res

    ema = ta.ema(agg_shifted['close'], length=50)
    trend = (agg_shifted['close'] - ema) / ema
    rsi = ta.rsi(agg_shifted['close'], length=14)

    sigma_1 = agg_shifted['close'].diff(1).rolling(50, min_periods=50).std()
    sigma_n = agg_shifted['close'].diff(5).rolling(50, min_periods=50).std()
    hurst = np.log(safe_divide(sigma_n, sigma_1 + 1e-9)) / np.log(5)

    res = pd.DataFrame(index=base_df.index)
    feats = {'trend': trend, 'rsi': rsi, 'hurst': hurst}

    for name, s in feats.items():
        vals = s.values
        expanded = np.repeat(vals, multiplier)

        if len(expanded) > len(base_df):
            expanded = expanded[:len(base_df)]
        elif len(expanded) < len(base_df):
            pad = np.full(len(base_df) - len(expanded), np.nan)
            expanded = np.concatenate([expanded, pad])

        res[f'{name}_{suffix}'] = expanded

    return res

INITIAL_FEATURES = [
    'atr_pct', 'atr_ratio', 'vol_of_vol', 'tr_norm',
    'z_score_price', 'ema_dislocation', 'autocorr_1', 'autocorr_3',
    'skew', 'kurt', 'entropy_proxy',
    'hurst', 'fractal_dim', 'regime_trend',
    'tick_rate', 'rtick_rel', 'rtick_mom',
    'dominant_freq', 'wavelet_energy',
    'trend_mtf10', 'rsi_mtf10', 'hurst_mtf10',
    'trend_mtf40', 'rsi_mtf40', 'hurst_mtf40',
    'hour_sin', 'hour_cos'
]

def feature_engineering(INPUT_FILE: str, OUTPUT_FILE: str, TICKS_PER_BAR: int, W_SHORT: int, W_LONG: int, W_TREND: int, correlation_threshold: float):
    print(f"\n{getTime()}{bcolors.ENDC} 🔄 STARTING FEATURE ENGINEERING...")

    if not os.path.exists(INPUT_FILE):
        print(f"\n{getTime()}{bcolors.FAIL} ERROR: Input file '{INPUT_FILE}' does not exist.{bcolors.ENDC}\n")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE, index_col='time', parse_dates=['time'])
    data = df.copy()

    data['bar_seconds'] = data.index.to_series().diff().dt.total_seconds()
    data['bar_seconds'] = data['bar_seconds'].bfill()

    median_bar_sec = data['bar_seconds'].median()
    if median_bar_sec <= 0 or np.isnan(median_bar_sec):
        median_bar_sec = 1.0
    data.loc[data['bar_seconds'] <= 0, 'bar_seconds'] = median_bar_sec

    data['tick_rate'] = TICKS_PER_BAR / data['bar_seconds']
    tick_ma = data['tick_rate'].rolling(window=W_LONG, min_periods=W_LONG).mean()
    data['rtick_rel'] = safe_divide(data['tick_rate'], tick_ma)
    data['rtick_mom'] = data['tick_rate'].pct_change(5, fill_method=None).rolling(window=5, min_periods=5).mean()

    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))

    data['atr_val'] = ta.atr(data['high'], data['low'], data['close'], length=W_SHORT)
    data['atr_pct'] = data['atr_val'] / data['close']

    atr_long = ta.atr(data['high'], data['low'], data['close'], length=W_LONG)
    data['atr_ratio'] = safe_divide(data['atr_val'], atr_long)

    data['vol_of_vol'] = (
        data['atr_pct'].rolling(window=W_LONG, min_periods=W_LONG).std() /
        data['atr_pct'].rolling(window=W_LONG, min_periods=W_LONG).mean()
    )

    tr = ta.true_range(data['high'], data['low'], data['close'])
    data['tr_norm'] = safe_divide(tr, data['atr_val'])

    ma_short = data['close'].shift(1).rolling(window=W_LONG, min_periods=W_LONG).mean()
    std_short = data['close'].shift(1).rolling(window=W_LONG, min_periods=W_LONG).std()
    data['z_score_price'] = (data['close'] - ma_short) / std_short

    ema_trend = ta.ema(data['close'], length=W_TREND)
    data['ema_dislocation'] = (data['close'] - ema_trend) / ema_trend

    for lag in [1, 3]:
        data[f'autocorr_{lag}'] = data['close'].rolling(window=W_LONG, min_periods=W_LONG).corr(data['close'].shift(lag))

    rolling_window = data['log_ret'].rolling(window=W_LONG, min_periods=W_LONG)
    data['skew'] = rolling_window.skew()
    data['kurt'] = rolling_window.kurt()

    change = np.abs(data['close'] - data['close'].shift(W_SHORT))
    path = np.abs(data['close'] - data['close'].shift(1)).rolling(window=W_SHORT, min_periods=W_SHORT).sum()
    data['entropy_proxy'] = safe_divide(change, path)

    data['fractal_dim'] = ta.chop(data['high'], data['low'], data['close'], length=W_SHORT) / 100.0
    data['hurst'] = rolling_hurst(data['close'], window=W_TREND)

    adx = ta.adx(data['high'], data['low'], data['close'], length=14)
    col_adx = [c for c in adx.columns if c.startswith('ADX')][0]
    data['regime_trend'] = adx[col_adx] / 100.0

    zcr = ((data['log_ret'] * data['log_ret'].shift(1)) < 0).rolling(window=W_SHORT, min_periods=W_SHORT).mean()
    data['dominant_freq'] = zcr

    hp_filter = data['close'] - data['close'].rolling(window=5, min_periods=5).mean()
    data['wavelet_energy'] = (hp_filter ** 2).rolling(window=W_SHORT, min_periods=W_SHORT).mean()

    mtf_10 = get_mtf(data, 10, 'mtf10')
    mtf_40 = get_mtf(data, 40, 'mtf40')
    data = pd.concat([data, mtf_10, mtf_40], axis=1)

    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    print(f"{getTime()}{bcolors.ENDC} Performing feature selection...")
    df_feats = data[INITIAL_FEATURES]
    corr_matrix = df_feats.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    print(f"{getTime()}{bcolors.ENDC} Dropping {len(to_drop)} highly correlated features: {to_drop}")

    FINAL_FEATURES = [f for f in INITIAL_FEATURES if f not in to_drop]

    COLS_TO_KEEP = FINAL_FEATURES + ['open', 'high', 'low', 'close', 'atr_val']
    df_final = data[COLS_TO_KEEP].copy()

    df_final.to_csv(OUTPUT_FILE)

    print(f"{getTime()}{bcolors.OKGREEN} ✅ FEATURE ENGINEERING COMPLETED. Processed data saved to '{OUTPUT_FILE}'.{bcolors.ENDC}")
    print(f"{getTime()}{bcolors.ENDC} Final Features Used ({len(FINAL_FEATURES)}): {FINAL_FEATURES}")
