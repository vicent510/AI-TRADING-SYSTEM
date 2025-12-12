# Libraries
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import yaml

# Local Dependencies
from src.utils.utils import *
from src.utils.errors import *

from src.data.data_wrangling import data_wrangling
from src.data.feature_enginiering import feature_engineering
from src.data.targeting import targeting
from src.data.data_split import data_split
from src.data.preprocess import preprocess
from src.train.main_model import main_model
from src.data.meta_dataset import meta_dataset
from src.train.meta_model import meta_model
from src.optimitzation.model_optimitzation import model_optimitzation
from src.backtest.backtester import backtester

CONFIG_PATH = "config/config.yaml"
REPORTS_PATH = "reports/"

def main():
    # Load Config
    config = load_config(CONFIG_PATH)

    # Config Parameters
    name = config.get('name', 'AI TRADING SYSTEM')
    version = config.get('version', 'Unknown')
    report_title = config.get('report_title', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    data_source = config.get('data_source', None)
    tick_bar_size = config.get('tick_bar_size', 200)

    wshort = config.get('W-SHORT', 20)
    wlong = config.get('W-LONG', 100)
    wtrend = config.get('W-TREND', 200)
    correlation_threshold = config.get('correlation_threshold', 0.95)

    atr_period = config.get('atr_period', 14)
    enableOptimization = config.get('enableOptimization', True)
    n_trials = config.get('N_TRIALS', 500)

    start_date = config.get('START_DATE', '2020-01-01')
    final_date = config.get('END_DATE', '2023-01-01')
    train_ratio = config.get('TRAIN_RATIO', 0.7)
    val_ratio = config.get('VAL_RATIO', 0.15)

    seq_len = config.get('SEQ_LEN', 60)
    batch_size = config.get('BATCH_SIZE', 2048)
    epochs = config.get('EPOCHS', 50)
    dropout = config.get('DROPOUT', 0.2)
    learning_rate = config.get('LEARNING_RATE', 0.001)

    threshold = config.get('THRESHOLD', 0.05)

    random_state = config.get('RANDOM_STATE', 42)
    test_size = config.get('TEST_SIZE', 0.2)

    op_n_trials = config.get('OP_N_TRIALS', 500)
    initial_capital = config.get('INITIAL CAPITAL', 10000)
    risk_per_trade = config.get('RISK_PER_TRADE', 0.0025)
    n_min_trades = config.get('N_MIN_TRADES', 40)
    w_r = config.get('W_R', 3.0)
    w_pf = config.get('W_PF', 1.5)
    w_wr = config.get('W_WR', 1.0)
    w_stab = config.get('W_Stab', 2.0)
    w_act = config.get('W_Act', 0.1)
    w_dd = config.get('W_DD', 4.0)

    weights = {
        'W_R': w_r,
        'W_PF': w_pf,
        'W_WR': w_wr,
        'W_Stab': w_stab,
        'W_Act': w_act,
        'W_DD': w_dd
    }

    # Define Paths
    data_wrangling_path = f"sample_data/{report_title}/step1_data.csv"
    feature_enginiering_path = f"sample_data/{report_title}/step2_data.csv"
    targeting_path = f"sample_data/{report_title}/step3_data.csv"
    data_split_path = f"sample_data/{report_title}/splits"
    preprocess_path = f"sample_data/{report_title}/processed_data"
    main_model_path = f"models/{report_title}"
    meta_dataset_path = f"sample_data/{report_title}/processed_data/meta_dataset.npz"
    meta_model_path = f"models/{report_title}//meta_model_xgb.pkl"
    model_optimitzation_path = f"sample_data/{report_title}/config"
    backtest_path  = f"reports/{report_title}"

    # Welcome Message
    welcome_message(name, version, report_title, CONFIG_PATH, REPORTS_PATH)

    # Data Wrangling
    verificate_data_source(data_source, tick_bar_size, CONFIG_PATH)
    #data_wrangling(data_source, data_wrangling_path, tick_bar_size)

    # Feature Enginiering
    verificate_feature_source(wlong, wshort, wtrend)
    #feature_engineering(data_wrangling_path, feature_enginiering_path, tick_bar_size, wshort, wlong, wtrend, correlation_threshold)
    
    # Targeting
    verificate_target_source(atr_period, enableOptimization, n_trials)
    #targeting(feature_enginiering_path, targeting_path, atr_period, enableOptimization, n_trials)

    # Data Split
    verificate_data_split_source(start_date, final_date, train_ratio, val_ratio)
    #data_split(targeting_path, data_split_path, start_date, final_date, train_ratio, val_ratio)
    
    # Preprocess
    #preprocess(data_split_path, preprocess_path, train_ratio, val_ratio)

    # Main Model Train
    verificate_main_train_source(seq_len, batch_size, epochs, dropout, learning_rate)
    #main_model(preprocess_path, main_model_path, seq_len, batch_size, epochs, dropout, learning_rate)

    # Meta Dataset
    verificate_meta_dataset(threshold)
    #meta_dataset(preprocess_path, meta_dataset_path, main_model_path, seq_len, threshold)
    
    # Meta Model Train
    verificate_meta_train(random_state, test_size)
    #meta_model(meta_dataset_path, meta_model_path, random_state, test_size)

    # Optimitzation
    #model_optimitzation(main_model_path, meta_model_path, preprocess_path, data_split_path, preprocess_path, data_split_path, model_optimitzation_path, op_n_trials, initial_capital, risk_per_trade, seq_len, n_min_trades, weights)

    # Backtest
    backtester(main_model_path, meta_model_path, model_optimitzation_path, preprocess_path, data_split_path, preprocess_path, data_split_path, seq_len, initial_capital, risk_per_trade, backtest_path)

if __name__ == "__main__":
    main()