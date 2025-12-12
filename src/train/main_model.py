# Libraries imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import sys
import matplotlib.pyplot as plt

# Local Imports
from src.utils.utils import getTime, bcolors

def main_model(INPUT_FILE: str, OUTPUT_PATH: str, SEQ_LEN: int, BATCH_SIZE: int, EPOCHS: int, DROPOUT: float, LEARNING_RATE: float):
    print(f"\n{getTime()}{bcolors.HEADER} ====== Main Model Training Started ====== {bcolors.ENDC}")
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        sys.exit(f"{bcolors.FAIL}❌ FAIL: {INPUT_FILE} impossible to find.{bcolors.ENDC}")
    
    train_data = np.load(os.path.join(INPUT_FILE, 'train_data.npz'))
    val_data = np.load(os.path.join(INPUT_FILE, 'test_data.npz'))

    X_train = train_data['X']
    y_train = train_data['y']
    X_val = val_data['X']
    y_val = val_data['y']

    print(f"{getTime()}📥 Training data:   {X_train.shape} samples.")
    print(f"{getTime()}📥 Validation data: {X_val.shape} samples.")

    # Data Generators
    train_ds = tf.keras.utils.timeseries_dataset_from_array(X_train, y_train, sequence_length=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = tf.keras.utils.timeseries_dataset_from_array(X_val, y_val, sequence_length=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=False)
    
    # Final Architecture

    n_features = X_train.shape[1]

    model = Sequential([
        Input(shape=(SEQ_LEN, n_features)),
        
        LSTM(128, return_sequences=True),
        Dropout(DROPOUT),
        
        LSTM(64, return_sequences=False),
        Dropout(DROPOUT),
        
        Dense(32, activation='relu'),
        
        Dense(1, activation='tanh') 
    ])

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    model.summary()

    checkpoint_path = os.path.join(OUTPUT_PATH, 'best_model.keras')

    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    # Train the Model
    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks, verbose=1)

    print(f"{getTime()}{bcolors.OKGREEN}✅ SUCCESS: Main Model Trained and Saved in {checkpoint_path}{bcolors.ENDC}")
