# Library Imports
import os
import pandas as pd
import numpy as np

# Local Imports
from src.utils.utils import getTime, bcolors

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): 
        return iterable

def create_tick_bars(df_ticks, bar_size):
    print(f"{getTime()}{bcolors.ENDC} Creating Tick Bars...")

    mid_prices = ((df_ticks['bid'] + df_ticks['ask']) / 2).values
    times = df_ticks.index.values

    opens, highs, lows, closes, datetimes, volumes = [], [], [], [], [], []

    for i in tqdm(range(0, len(mid_prices), bar_size)):
        chunk = mid_prices[i: i + bar_size]
        if len(chunk) < bar_size:
            break

        opens.append(chunk[0])
        highs.append(np.max(chunk))
        lows.append(np.min(chunk))
        closes.append(chunk[-1])
        datetimes.append(times[i + bar_size - 1])
        volumes.append(bar_size)

    return pd.DataFrame(
        {'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes},
        index=datetimes
    )

def data_wrangling(RAW_FILE: str, OUTPUT_FILE: str, TICK_BAR_SIZE: float):
    print(f"{getTime()} 🔄 STARTING DATA WRANGLING")

    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"{getTime()}{bcolors.FAIL} ERROR: Raw data file '{RAW_FILE}' not found.{bcolors.ENDC}")

    df = None
    is_tick_data = False

    configs = [
        {'sep': ';', 'encoding': 'utf-16'},
        {'sep': ';', 'encoding': 'utf-8'},
        {'sep': ',', 'encoding': 'utf-8'},
        {'sep': '\t', 'encoding': 'utf-16'}
    ]

    for cfg in configs:
        try:
            preview = pd.read_csv(RAW_FILE, sep=cfg['sep'], encoding=cfg['encoding'], nrows=2)
            num_cols = preview.shape[1]

            cols_lower = [str(c).strip().lower() for c in preview.columns]

            if 'bid' in cols_lower and 'ask' in cols_lower:
                is_tick_data = True
                expected_cols = ['time', 'bid', 'ask']
            elif all(c in cols_lower for c in ['open', 'high', 'low', 'close']):
                is_tick_data = False
                expected_cols = ['time', 'open', 'high', 'low', 'close']
                if 'volume' in cols_lower:
                    expected_cols.append('volume')
            else:
                if num_cols == 3:
                    is_tick_data = True
                    expected_cols = ['time', 'bid', 'ask']
                elif num_cols >= 5:
                    is_tick_data = False
                    expected_cols = ['time', 'open', 'high', 'low', 'close']
                    if num_cols >= 6:
                        expected_cols.append('volume')
                else:
                    continue

            print(f"{getTime()}{bcolors.ENDC} Format detected: Separator='{cfg['sep']}' | Encoding='{cfg['encoding']}' | Type={'TICKS' if is_tick_data else 'OHLC'}")

            df = pd.read_csv(
                RAW_FILE,
                sep=cfg['sep'],
                encoding=cfg['encoding'],
                header=0,
                low_memory=False
            )
            df.columns = [str(c).strip().lower() for c in df.columns]

            if 'time' not in df.columns:
                df.rename(columns={df.columns[0]: 'time'}, inplace=True)

            break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"{getTime()}{bcolors.FAIL} ERROR: Unable to read raw data file with available configurations.{bcolors.ENDC}")

    df['time'] = pd.to_datetime(df['time'], format="%Y.%m.%d %H:%M:%S", errors='coerce')
    df = df[df['time'].notna()].copy()
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    print(f"{getTime()}{bcolors.OKGREEN} ✅ Data loaded. Number of rows: {len(df)}{bcolors.ENDC}")

    if is_tick_data:
        print(f"{getTime()}{bcolors.ENDC} 🚀 Processing real tick data...")

        if 'bid' not in df.columns or 'ask' not in df.columns:
            raise ValueError(f"{getTime()}{bcolors.FAIL} ERROR: Tick data requires 'bid' and 'ask' columns.{bcolors.ENDC}")

        df['bid'] = pd.to_numeric(df['bid'], errors='coerce')
        df['ask'] = pd.to_numeric(df['ask'], errors='coerce')
        df.dropna(subset=['bid', 'ask'], inplace=True)

        df_final = create_tick_bars(df, int(TICK_BAR_SIZE))
        df_final.index.name = 'time'

    else:
        print(f"{getTime()}{bcolors.WARNING} ⚠️  Processing OHLC data...{bcolors.ENDC}")

        for c in ['open', 'high', 'low', 'close']:
            if c not in df.columns:
                raise ValueError(f"{getTime()}{bcolors.FAIL} ERROR: Missing required OHLC column '{c}'.{bcolors.ENDC}")
            df[c] = pd.to_numeric(df[c], errors='coerce')

        if 'volume' not in df.columns:
            df['volume'] = 100
        else:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(100)

        df_final = df.copy()
        df_final.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"{getTime()}{bcolors.ENDC} Saving processed data to '{OUTPUT_FILE}'...{bcolors.ENDC}")
    df_final.to_csv(OUTPUT_FILE, index=True)

    print(f"{getTime()}{bcolors.OKGREEN} ✅ DATA WRANGLING COMPLETED. Final number of rows: {len(df_final)}{bcolors.ENDC}")
