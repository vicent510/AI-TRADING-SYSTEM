# Dependencies Imports
import pandas as pd
import numpy as np
import os
import sys

# Local Imports
from src.utils.utils import getTime, bcolors

def data_split(INPUT_FILE: str, OUTPUT_FILE: str, START_DATE: str, END_DATE: str, TRAIN_RATIO: float, VAL_RATIO: float):

    print(f"\n{getTime()} 🔄 Starting data split...{bcolors.ENDC}")
    print(f"{getTime()} 📅 Loading data from {START_DATE} to {END_DATE}.")

    if not os.path.exists(INPUT_FILE):
        print(f"\n{getTime()} {bcolors.FAIL}ERROR: Input file '{INPUT_FILE}' does not exist.{bcolors.ENDC}\n")
        sys.exit(1)
    
    os.makedirs(OUTPUT_FILE, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, index_col='time', parse_dates=['time'])
    df.sort_index(inplace=True)

    # Data Filtering by Date
    try:
        start = START_DATE if START_DATE else df.index.min()
        end = END_DATE if END_DATE else df.index.max()
        
        df_filtered = df.loc[start:end]
        
        if df_filtered.empty:
            print(f"{getTime()}{bcolors.FAIL}❌ ERROR: The range from {START_DATE} to {END_DATE} results in an empty dataset.{bcolors.ENDC}")
            sys.exit()
            
        print(f"{getTime()}{bcolors.OKGREEN} ✅ Data after Filtering: {len(df_filtered):,} rows.")

    except Exception as e:
        print(f"{getTime()}{bcolors.FAIL}❌ ERROR: Error during date filtering - {e}{bcolors.ENDC}")
        sys.exit()

    # Cross point indices for splitting
    n = len(df_filtered)
    train_end_idx = int(n * TRAIN_RATIO)
    val_end_idx = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df_filtered.iloc[:train_end_idx]
    val_df = df_filtered.iloc[train_end_idx:val_end_idx]
    test_df = df_filtered.iloc[val_end_idx:]

    # Save splits
    train_path = os.path.join(OUTPUT_FILE, 'train_raw.csv')
    val_path = os.path.join(OUTPUT_FILE, 'val_raw.csv')
    test_path = os.path.join(OUTPUT_FILE, 'test_raw.csv')

    train_df.to_csv(train_path)
    val_df.to_csv(val_path)
    test_df.to_csv(test_path)

    print(f"{getTime()}{bcolors.OKGREEN} ✅ Data Split Completed:{bcolors.ENDC}")
    print(f"{getTime()}   - Training Set: {len(train_df):,} rows saved to {train_path}")
    print(f"{getTime()}   - Validation Set: {len(val_df):,} rows saved to {val_path}")
    print(f"{getTime()}   - Testing Set: {len(test_df):,} rows saved to {test_path}")