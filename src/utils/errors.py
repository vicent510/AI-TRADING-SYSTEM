import os
from src.utils.utils import getTime, bcolors


def verificate_data_source(data_source: str, tick_bar_size: int, config_path: str):
    if(data_source is None):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: Data Source is not defined in {config_path}.{bcolors.ENDC}\n")
    if(isinstance(data_source, str) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: Data Source must be a string. Make sure it is enclosed in double quotes in {config_path}.{bcolors.ENDC}\n")
    if(os.path.exists(data_source) == False):
        raise FileNotFoundError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: Data Source file '{data_source}' not found. Check the path in {config_path}.{bcolors.ENDC}\n")
    if(isinstance(tick_bar_size, int) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: Tick Bar Size must be a number (int). Check the value in {config_path}.{bcolors.ENDC}\n")
    if(tick_bar_size <= 0):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: Tick Bar Size must be greater than 0. Check the value in {config_path}.{bcolors.ENDC}\n")
    else:
        print(f"{getTime()} Data Source '{data_source}' found and verified.\n")

def verificate_feature_source( wlong: int, wshort: int, wtrend: int):
    if(isinstance(wlong, int) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: W_LONG must be a number (int). Check the value in config.{bcolors.ENDC}\n")
    if(wlong <= 0):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: W_LONG must be greater than 0. Check the value in config.{bcolors.ENDC}\n")
    if(wlong.is_integer() == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: W_LONG must be an integer. Check the value in config.{bcolors.ENDC}\n")
    if(wshort <= 0):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: W_SHORT must be greater than 0. Check the value in config.{bcolors.ENDC}\n")
    if(wshort.is_integer() == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: W_SHORT must be an integer. Check the value in config.{bcolors.ENDC}\n")
    if(wtrend <= 0):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: W_TREND must be greater than 0. Check the value in config.{bcolors.ENDC}\n")
    if(wtrend.is_integer() == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: W_TREND must be an integer. Check the value in config.{bcolors.ENDC}\n")
    
def verificate_target_source(atr_period: int, ENABLE_OPTIMIZATION: bool, N_TRIALS: int):
    if(isinstance(atr_period, int) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: ATR_PERIOD must be a number (int). Check the value in config.{bcolors.ENDC}\n")
    if(atr_period <= 0):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: ATR_PERIOD must be greater than 0. Check the value in config.{bcolors.ENDC}\n")
    if(ENABLE_OPTIMIZATION not in [True, False]):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: ENABLE_OPTIMIZATION must be a boolean (True/False). Check the value in config.{bcolors.ENDC}\n")
    if(isinstance(N_TRIALS, int) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: N_TRIALS must be a number (int). Check the value in config.{bcolors.ENDC}\n")
    if(N_TRIALS <= 0):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: N_TRIALS must be greater than 0. Check the value in config.{bcolors.ENDC}\n")

def verificate_data_split_source(start_date: str, final_date: str, train_ratio: float, val_ratio: float):
    if(isinstance(start_date, str) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: START_DATE must be a string. Make sure it is enclosed in double quotes in config.{bcolors.ENDC}\n")
    if(isinstance(final_date, str) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: END_DATE must be a string. Make sure it is enclosed in double quotes in config.{bcolors.ENDC}\n")
    if(train_ratio <= 0 or train_ratio >= 1):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: TRAIN_RATIO must be between 0 and 1. Check the value in config.{bcolors.ENDC}\n")
    if(val_ratio <= 0 or val_ratio >= 1):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: VAL_RATIO must be between 0 and 1. Check the value in config.{bcolors.ENDC}\n")
    if(train_ratio.is_integer() == False and isinstance(train_ratio, float) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: TRAIN_RATIO must be an integer. Check the value in config.{bcolors.ENDC}\n")
    if(val_ratio.is_integer() == False and isinstance(val_ratio, float) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: VAL_RATIO must be an integer. Check the value in config.{bcolors.ENDC}\n")

def verificate_main_train_source(SQLEN: int, BATCH_SIZE: int, EPOCHS: int, DROPOUT: float, LEARNING_RATE: float):
    if(isinstance(SQLEN, int) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: SEQ_LEN must be a number (int). Check the value in config.{bcolors.ENDC}\n")
    if(SQLEN <= 0):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: SEQ_LEN must be greater than 0. Check the value in config.{bcolors.ENDC}\n")
    if(isinstance(BATCH_SIZE, int) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: BATCH_SIZE must be a number (int). Check the value in config.{bcolors.ENDC}\n")
    if(BATCH_SIZE <= 0):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: BATCH_SIZE must be greater than 0. Check the value in config.{bcolors.ENDC}\n")
    if(isinstance(EPOCHS, int) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: EPOCHS must be a number (int). Check the value in config.{bcolors.ENDC}\n")
    if(EPOCHS <= 0):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: EPOCHS must be greater than 0. Check the value in config.{bcolors.ENDC}\n")
    if(isinstance(DROPOUT, float) == False and isinstance(DROPOUT, int) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: DROPOUT must be a number (float). Check the value in config.{bcolors.ENDC}\n")
    if(DROPOUT < 0 or DROPOUT >= 1):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: DROPOUT must be between 0 and 1. Check the value in config.{bcolors.ENDC}\n")
    if(isinstance(LEARNING_RATE, float) == False and isinstance(LEARNING_RATE, int) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: LEARNING_RATE must be a number (float). Check the value in config.{bcolors.ENDC}\n")
    if(LEARNING_RATE <= 0):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: LEARNING_RATE must be greater than 0. Check the value in config.{bcolors.ENDC}\n")
    pass

def verificate_meta_dataset(threshold: float):
    if(isinstance(threshold, float) == False and isinstance(threshold, int) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: THRESHOLD must be a number (float). Check the value in config.{bcolors.ENDC}\n")
    if(threshold < 0):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATA ERROR: THRESHOLD must be higher than 0. Check the value in config.{bcolors.ENDC}")
    
def verificate_meta_train(random_state: int, test_size: float):
    if(isinstance(random_state, int) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: RANDOM STATE must be a number (int). Check the value in config.{bcolors.ENDC}\n")
    if(random_state <= 0):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATA ERROR: RANDOM STATE must be higher than 0. Check the value in config.{bcolors.ENDC}")
    if(isinstance(test_size, float) == False and isinstance(test_size, int) == False):
        raise TypeError(f"\n{getTime()} {bcolors.FAIL}FATAL ERROR: TEST SIZE must be a number (float). Check the value in config.{bcolors.ENDC}\n")
    if(test_size <= 0):
        raise ValueError(f"\n{getTime()} {bcolors.FAIL}FATA ERROR: TEST SIZE must be higher than 0. Check the value in config.{bcolors.ENDC}")
    
