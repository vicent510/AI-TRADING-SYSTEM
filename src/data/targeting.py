# Libraries Imports
import pandas as pd
import numpy as np
import pandas_ta as ta
import optuna
import os
import sys

# Local Imports
from src.utils.utils import bcolors, getTime

MANUAL_PARAMS = {
    'HORIZON': 10,
    'RR_TP': 3.0,
    'RR_SL': 1.0,
    'RR_MIN': 1.0
}

def calculate_targets(df_in, horizon, rr_tp, rr_sl, rr_min, return_df=False):
    df = df_in.copy()
    
    # 1. WINDOW LOOK-AHEAD
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
    future_high_max = df['high'].rolling(window=indexer).max()
    future_low_min  = df['low'].rolling(window=indexer).min()
    future_close    = df['close'].shift(-horizon)
    
    # 2. DYNAMIC BARRIERS
    barrier_up_tp = df['close'] + (df['atr_val'] * rr_tp)
    barrier_dn_sl = df['close'] - (df['atr_val'] * rr_sl)
    
    barrier_dn_tp_short = df['close'] - (df['atr_val'] * rr_tp)
    barrier_up_sl_short = df['close'] + (df['atr_val'] * rr_sl)
    
    # 3. EVENT DETECTION
    long_hit_tp  = future_high_max >= barrier_up_tp
    long_hit_sl  = future_low_min <= barrier_dn_sl
    short_hit_tp = future_low_min <= barrier_dn_tp_short
    short_hit_sl = future_high_max >= barrier_up_sl_short
    
    # 4. TARGET ASSIGNMENT
    target_class = np.zeros(len(df), dtype=int)
    target_r_val = np.zeros(len(df), dtype=float)
    
    # Strong Wins (+2 / -2)
    cond_long_tp  = (long_hit_tp & ~long_hit_sl)
    cond_short_tp = (short_hit_tp & ~short_hit_sl)
    
    target_class[cond_long_tp]  = 2
    target_r_val[cond_long_tp]  = rr_tp
    target_class[cond_short_tp] = -2
    target_r_val[cond_short_tp] = -rr_tp
    
    # Failures (SL hit)
    cond_long_fail  = long_hit_sl
    cond_short_fail = short_hit_sl
    
    target_r_val[cond_long_fail]  = -rr_sl
    target_r_val[cond_short_fail] = -rr_sl 
    
    # Weak / Neutral Logic
    mask_pending = (target_class == 0) & (~cond_long_fail) & (~cond_short_fail)
    r_close = (future_close - df['close']) / df['atr_val']
    
    mask_weak_long  = (r_close >= rr_min)
    mask_weak_short = (r_close <= -rr_min)
    
    mask_wl = mask_pending & mask_weak_long
    target_class[mask_wl] = 1
    target_r_val[mask_wl] = r_close[mask_wl]
    
    mask_ws = mask_pending & mask_weak_short
    target_class[mask_ws] = -1
    target_r_val[mask_ws] = r_close[mask_ws]
    
    # 5. FINAL TARGET CALCULATION
    scale_factor = 0.5
    df['target'] = np.tanh(target_r_val * scale_factor)
    df['target_class'] = target_class
    df['target_R'] = target_r_val
    df['meta_y'] = (np.abs(target_class) == 2).astype(int)
    df['future_close'] = future_close
    
    df.dropna(subset=['target', 'future_close'], inplace=True)
    
    if return_df:
        return df
    else:
        # TQS CALCULATOR
        if len(df) == 0: return -999
        
        mean_val    = df['target'].mean()
        neutral_pct = (df['target_class'] == 0).mean()
        meta_wr     = df['meta_y'].mean()
        
        
        penalty_bias    = abs(mean_val) * 25.0          
        penalty_noise   = abs(neutral_pct - 0.60) * 10.0
        penalty_alpha   = abs(meta_wr - 0.15) * 10.0
        
        penalty_safety = 0.0
        if neutral_pct < 0.40:
            penalty_safety = 10.0
            
        tqs = 100.0 - penalty_bias - penalty_noise - penalty_alpha - penalty_safety
        return tqs

def targeting(INPUT_FILE: str, OUTPUT_FILE: str, ATR_PERIOD: int, ENABLE_OPTIMIZATION: bool, N_TRIALS: int):
    print(f"\n{getTime()} 🔄 Starting Targeting Process...{bcolors.ENDC}")

    # Load Data
    if not os.path.exists(INPUT_FILE):
        sys.exit(f"{bcolors.FAIL}❌ FAIL: {INPUT_FILE} impossible to find.{bcolors.ENDC}")

    df_raw = pd.read_csv(INPUT_FILE, index_col='time', parse_dates=['time'])

    if 'atr_val' not in df_raw.columns:
        df_raw['atr_val'] = ta.atr(df_raw['high'], df_raw['low'], df_raw['close'], length=ATR_PERIOD)
        df_raw['atr_val'].fillna(method='bfill', inplace=True)

    df_raw.dropna(subset=['atr_val'], inplace=True)
    print(f"{getTime()}{bcolors.OKGREEN} ✅ Data loaded: {len(df_raw)} rows.{bcolors.ENDC}")

    # Optimitzation
    final_params = MANUAL_PARAMS

    if ENABLE_OPTIMIZATION:
        print(f"{getTime()} 🔬 STARTING AUTO-TUNE ({N_TRIALS} trials)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            h   = trial.suggest_int('HORIZON', 6, 20)           
            tp  = trial.suggest_float('RR_TP', 2.0, 5.0, step=0.1)
            sl  = trial.suggest_float('RR_SL', 0.8, 2.0, step=0.05) 
            rm  = trial.suggest_float('RR_MIN', 0.5, 1.5, step=0.1)
            
            return calculate_targets(df_raw, h, tp, sl, rm, return_df=False)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS)
        
        print(f"{getTime()}{bcolors.OKGREEN} ✨ BEST TQS: {study.best_value:.4f}")
        print(f"{getTime()}{bcolors.OKGREEN} 🏆 BEST Parameters: {study.best_params}")
        final_params = study.best_params
    else:
        print(f"{getTime()}{bcolors.WARNING} ⚙️ Using Manual Parameters: {final_params}{bcolors.ENDC}")

    # Calculate Targets
    print(f"\n{getTime()} 🔢 Calculating Targets with parameters: {final_params}...")

    df_final = calculate_targets(
        df_raw, 
        final_params['HORIZON'], 
        final_params['RR_TP'], 
        final_params['RR_SL'], 
        final_params['RR_MIN'], 
        return_df=True
    )
    
    # Final Report
    print(f"\n{getTime()}{bcolors.HEADER} ====== TARGETING REPORT ====== {bcolors.ENDC}")
    print(f"{getTime()} Total Samples: {len(df_final)}")
    dist = df_final['target_class'].value_counts(normalize=True).sort_index()
    for cls, pct in dist.items():
        print(f"{getTime()} CLASS {cls:2d}: {pct*100:5.2f}%")

    print(f"\n{getTime()} META-TARGET WR: {df_final['meta_y'].mean()*100:.2f}")
    print(f"{getTime()} TARGET MEAN: {df_final['target'].mean():.4f}")

    # Save
    df_final.to_csv(OUTPUT_FILE)
    print(f"\n{getTime()}{bcolors.OKGREEN} ✅ DATASET COMPLETED: {OUTPUT_FILE}")
    