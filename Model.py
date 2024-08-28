'''

LSTM Model building, training and evaluation

'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json

# Define Stochastic ReLU
class StochasticReLU(Layer):
    def __init__(self, p=0.1):
        super(StochasticReLU, self).__init__()
        self.p = p

    def call(self, inputs, training=None):
        if training:
            mask = tf.random.uniform(tf.shape(inputs)) < (1 - self.p)
            return tf.where(mask, tf.nn.relu(inputs), tf.nn.relu(-inputs))
        else:
            return (1 - self.p) * tf.nn.relu(inputs) + self.p * tf.nn.relu(-inputs)

'''
Different preprocessing sequence tentatives
'''

def create_sequences(X, y, time_steps):
    X_seq, y_seq = [], []
    unique_ids = train_df['ChassisId_encoded'].unique()
    for uid in unique_ids:
        X_sub = X[train_df['ChassisId_encoded'] == uid]
        y_sub = y[train_df['ChassisId_encoded'] == uid]
        if len(X_sub) >= time_steps:
            # Use the last `time_steps` timesteps
            X_seq.append(X_sub[-time_steps:])
            y_seq.append(y_sub[-time_steps:])
        else:
            # Duplicate the first row to fill the sequence
            num_missing = time_steps - len(X_sub)
            X_seq.append(np.vstack([X_sub[0]] * num_missing + [X_sub]))
            y_seq.append(np.hstack([y_sub[0]] * num_missing + [y_sub]))
    return np.array(X_seq), np.array(y_seq)

# Define the sequence length (e.g., 10 timesteps)
time_steps = 10
X_seq, y_seq = create_sequences(X_scaled, y, time_steps)
print(f"Shape of X_seq: {X_seq.shape}")
print(f"Shape of y_seq: {y_seq.shape}")

def create_sequences(X, y, time_steps):
    X_seq, y_seq = [], []
    unique_ids = train_df['ChassisId_encoded'].unique()
    for uid in unique_ids:
        X_sub = X[train_df['ChassisId_encoded'] == uid]
        y_sub = y[train_df['ChassisId_encoded'] == uid]
        for i in range(len(X_sub) - time_steps + 1):  # Sliding window
            X_seq.append(X_sub[i:i + time_steps])
            y_seq.append(y_sub[i:i + time_steps])
    return np.array(X_seq), np.array(y_seq)


def create_sequences_test(X, y, time_steps):
    X_seq, y_seq = [], []
    unique_ids = train_df['ChassisId_encoded'].unique()
    for uid in unique_ids:
        X_sub = X[train_df['ChassisId_encoded'] == uid]
        y_sub = y[train_df['ChassisId_encoded'] == uid]
        for i in range(len(X_sub) - time_steps + 1):
            sequence_x = X_sub[i:i + time_steps]
            sequence_y = y_sub[i:i + time_steps]
            # Check if there is at least one minority label in the sequence
            if np.any(np.isin(sequence_y, [3])):  # Assuming labels 1 and 2 are the minority classes
                for _ in range(5):  # Upsample by replicating 20 times
                    X_seq.append(sequence_x)
                    y_seq.append(sequence_y)
            else:
                X_seq.append(sequence_x)
                y_seq.append(sequence_y)
    return np.array(X_seq), np.array(y_seq)

def create_sequences_test_1(X, y, time_steps):
    X_seq, y_seq = [], []
    unique_ids = train_df['ChassisId_encoded'].unique()
    for uid in unique_ids:
        X_sub = X[train_df['ChassisId_encoded'] == uid]
        y_sub = y[train_df['ChassisId_encoded'] == uid]
        for i in range(len(X_sub) - time_steps + 1):  # Sliding window
            X_seq.append(X_sub[i:i + time_steps])
            y_seq.append(y_sub[i:i + time_steps])
    return np.array(X_seq), np.array(y_seq)

'''
Best Model configuration, just run the following code to obtain the best step-0 model, then follow the next paragraph to add also the pseudo label and instantiate the boosting cycle
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from collections import Counter
from tensorflow.keras.optimizers import Adam
import random

# Preprocess the data
# Assuming 'risk level' is the target and other columns are features
X = train_df.drop(columns=['risk_level', 'Timesteps', 'ChassisId_encoded', 'gen'])
y = train_df['risk_level']
print(f"Shape of X after dropping columns: {X.shape}")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f"Shape of X_scaled: {X.shape}")


def create_sequences(X, y, time_steps):
    X_seq, y_seq = [], []
    unique_ids = train_df['ChassisId_encoded'].unique()

    for uid in unique_ids:
        X_sub = X[train_df['ChassisId_encoded'] == uid]
        y_sub = y[train_df['ChassisId_encoded'] == uid]

        # Skip sequences shorter than x time steps
        if len(X_sub) < 10:
            continue

        else:
            # Handle longer sequences based on the label of the last time step
            if y_sub[-1] == 1:
                # Random window from the sequence
                for i in range(2):
                  start_idx = random.randint(0, len(X_sub) - time_steps)
                  X_seq.append(X_sub[start_idx:start_idx + time_steps])
                  y_seq.append(y_sub[start_idx:start_idx + time_steps])
                  #y_seq.append(y_sub[start_idx + time_steps - 1])
            elif y_sub[-1] == 2:
                continue
            elif y_sub[-1] == 0:
                if len(y_sub) >= 10:  # Ensure there are at least 11 elements in the series
                  for i in range(2):
                    # Find indices where the label is 0 and there are at least 10 preceding elements
                    zero_label_indices = np.where(y_sub == 0)[0]
                    # Filter indices to ensure there are at most 9 preceding zeros in the last 10 timesteps
                    valid_indices = [idx for idx in zero_label_indices if idx >= time_steps and
                             np.sum(y_sub[idx - time_steps:idx] == 0) <= 8]
                    if valid_indices:
                      selected_idx = random.choice(valid_indices)
                      # Take the 10 timesteps ending just before the selected index
                      start_idx = selected_idx - time_steps+1
                      end_idx = selected_idx+1
                      X_seq.append(X_sub[start_idx:end_idx])
                      y_seq.append(y_sub[start_idx:end_idx])
                      #y_seq.append(y_sub[selected_idx])

    return np.array(X_seq), np.array(y_seq)


# Reshape the data back into sequences
time_steps = 10

# Update your script to use the new sequence generation function
X_seq, y_seq = create_sequences(X, y, time_steps)
print(f"Shape of X_seq: {X_seq.shape}")
print(f"Shape of y_seq: {y_seq.shape}")

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")


# Standardize the features using only the training data
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_val = scaler.transform(X_val)

from collections import Counter

# Ensure y_train is a flat list
label_counts_train = Counter(y_train.flatten().tolist())  # Flatten and convert to list if not already
print("Label counts in y_train:", label_counts_train)

# Ensure y_train is a flat list
label_counts_val = Counter(y_val.flatten().tolist())  # Flatten and convert to list if not already
print("Label counts in y_val:", label_counts_val)



dropout = 0.5
recurrent_dropout = 0.0
ret = True
nodes = 400
reg = 0.0002

# Define the LSTM model
model = Sequential()
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))

model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Custom F1-score metric for Keras
def macro_f1_score(y_true, y_pred):
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.cast(y_true, 'int32')
    y_pred = K.cast(y_pred, 'int32')

    f1_scores = []
    for i in range(len(label_encoder.classes_)):
        true_pos = K.sum(K.cast(y_true == i, 'float32') * K.cast(y_pred == i, 'float32'))
        false_pos = K.sum(K.cast(y_true != i, 'float32') * K.cast(y_pred == i, 'float32'))
        false_neg = K.sum(K.cast(y_true == i, 'float32') * K.cast(y_pred != i, 'float32'))

        precision = true_pos / (true_pos + false_pos + K.epsilon())
        recall = true_pos / (true_pos + false_neg + K.epsilon())

        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        f1_scores.append(f1)

    return K.mean(K.stack(f1_scores))

class MacroF1Score(Callback):
    def __init__(self, model, validation_data):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        self.best_score = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict(self.validation_data[0])
        # Assuming your model's output and y_val are shaped as (samples, timesteps, features) and you want to collapse the timesteps
        #val_pred = np.argmax(val_pred, axis=1)  # Changed axis since no longer sequence
        val_pred = np.argmax(val_pred, axis=-1).flatten()  # Flatten predictions
        val_true = self.validation_data[1].flatten()  # Flatten true labels
        #print(f'Predicted shape: {val_pred.shape}, True shape: {val_true.shape}')
        _val_f1 = f1_score(val_true, val_pred, average='macro')
        logs['val_macro_f1'] = _val_f1  # Ensure this metric is logged
        print(f" - val_macro_f1: {_val_f1:.4f}")

        # Check if the current F1 is better than the best and manually save
        if _val_f1 > self.best_score:
            self.best_score = _val_f1
            self.model.save('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_f1_model.h5')
            print("Saved improved model with F1 score:", _val_f1)

# Define the Adam optimizer with weight decay
#optimizer = Adam(learning_rate=0.001)
optimizer = 'adam'

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#model.compile(loss=CustomLoss(batch_size=2048, label_encoder=label_encoder), optimizer=optimizer, metrics=['accuracy'])


# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model.h5', monitor='val_macro_f1', mode='max', save_best_only=True, verbose=1)
# checkpoint = ModelCheckpoint('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

#Initialize the custom F1 score callback
f1_score_callback = MacroF1Score(model, validation_data=(X_val, y_val))

# Train the model with the checkpoint callback
history = model.fit(X_train, y_train, epochs=300, batch_size=1024, validation_data=(X_val, y_val), callbacks=[f1_score_callback])

# Load the best model
# model = tf.keras.models.load_model('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model.h5', custom_objects={'macro_f1_score': macro_f1_score})

model = tf.keras.models.load_model('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_f1_model.h5')

# Evaluate the best model
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=2)

# Flatten the arrays for evaluation
y_val_flat = y_val.flatten()
y_val_pred_flat = y_val_pred_classes.flatten()

# Print the value counts of the flattened arrays
print("Value counts of y_val_flat:", Counter(y_val_flat))
print("Value counts of y_val_pred_flat:", Counter(y_val_pred_flat))

# Calculate and print the macro F1 score
f1 = f1_score(y_val_flat, y_val_pred_flat, average='macro')
print(f"Macro F1 Score: {f1}")

from sklearn.metrics import f1_score, classification_report

# Calculate and print F1 score for each class
print("Classification Report:")
print(classification_report(y_val_flat, y_val_pred_flat, target_names=label_encoder.classes_))


---------------------------------------------------------------------------------------------------


# adding also the pseudo label of the test set

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from collections import Counter
from tensorflow.keras.optimizers import Adam
import random



# Load the original training data
train_df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/train_gen1.csv')

# Preprocess the training data
columns_with_nan_train = train_df.columns[train_df.isna().any()]
train_df = train_df.drop(columns=columns_with_nan_train)
#zero_col_train = ['af2__5', 'af2__6', 'af2__13', 'af2__18', 'af2__19', 'af2__20', 'af2__22']
zero_col_train = ['af2__5', 'af2__18']
train_df = train_df.drop(columns=zero_col_train)



# Load the test data
df_test = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/public_X_test.csv')

# Preprocess the test data
columns_with_nan_test = df_test.columns[df_test.isna().any()]
df_test_cleaned = df_test.drop(columns=columns_with_nan_test)
#zero_col_test = ['af2__5', 'af2__6', 'af2__13', 'af2__18', 'af2__19', 'af2__20', 'af2__22']
zero_col_test = ['af2__5', 'af2__18']
df_test_cleaned = df_test_cleaned.drop(columns=zero_col_test)

# Load the 'risk_level' predictions
df_predictions = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction_best.csv')

# Add the 'risk_level' column to the cleaned test data
df_test_cleaned['risk_level'] = df_predictions['pred']

# Append the modified test data to the original training data
combined_df = pd.concat([train_df, df_test_cleaned], ignore_index=True)


# Preprocess the data
# Assuming 'risk level' is the target and other columns are features
X = combined_df.drop(columns=['risk_level', 'Timesteps', 'ChassisId_encoded', 'gen'])
y = combined_df['risk_level']
print(f"Shape of X after dropping columns: {X.shape}")
print(f"Shape of y after dropping columns: {y.shape}")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f"Shape of X_scaled: {X.shape}")


# Assuming 'ChassisId_encoded' is still present in 'combined_df'
unique_ids = combined_df['ChassisId_encoded'].unique()

def create_sequences(X, y, ids, time_steps):
    X_seq, y_seq = [], []

    for uid in unique_ids:
        mask = ids == uid
        X_sub = X[mask]
        y_sub = y[mask]

        # Skip sequences shorter than x time steps
        if len(X_sub) < 10:
            continue

        else:
            # Handle longer sequences based on the label of the last time step
            if y_sub[-1] == 1:
                # Random window from the sequence
                for i in range(2):
                  start_idx = random.randint(0, len(X_sub) - time_steps)
                  X_seq.append(X_sub[start_idx:start_idx + time_steps])
                  y_seq.append(y_sub[start_idx:start_idx + time_steps])
                  #y_seq.append(y_sub[start_idx + time_steps - 1])
            elif y_sub[-1] == 2:
                continue
            elif y_sub[-1] == 0:
                if len(y_sub) > 15:  # Ensure there are at least 11 elements in the series
                  for i in range(2):
                    # Find indices where the label is 0 and there are at least 10 preceding elements
                    zero_label_indices = np.where(y_sub == 0)[0]
                    # Filter indices to ensure there are at most 9 preceding zeros in the last 10 timesteps
                    valid_indices = [idx for idx in zero_label_indices if idx >= time_steps and
                             np.sum(y_sub[idx - time_steps:idx] == 0) <= 8]
                    if valid_indices:
                      selected_idx = random.choice(valid_indices)
                      # Take the 10 timesteps ending just before the selected index
                      start_idx = selected_idx - time_steps+1
                      end_idx = selected_idx+1
                      X_seq.append(X_sub[start_idx:end_idx])
                      y_seq.append(y_sub[start_idx:end_idx])
                      #y_seq.append(y_sub[selected_idx])
                elif len(y_sub) == 10:
                  for i in range(2):
                    X_seq.append(X_sub)
                    y_seq.append(y_sub)


    return np.array(X_seq), np.array(y_seq)


# Reshape the data back into sequences
time_steps = 10

# Update your script to use the new sequence generation function
X_seq, y_seq = create_sequences(X, y, combined_df['ChassisId_encoded'].values, time_steps)
print(f"Shape of X_seq: {X_seq.shape}")
print(f"Shape of y_seq: {y_seq.shape}")

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")


# Standardize the features using only the training data
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_val = scaler.transform(X_val)

from collections import Counter

# Ensure y_train is a flat list
label_counts_train = Counter(y_train.flatten().tolist())  # Flatten and convert to list if not already
print("Label counts in y_train:", label_counts_train)

# Ensure y_train is a flat list
label_counts_val = Counter(y_val.flatten().tolist())  # Flatten and convert to list if not already
print("Label counts in y_val:", label_counts_val)



dropout = 0.5
recurrent_dropout = 0.0
ret = True
nodes = 300
reg = 0.0002

# Define the LSTM model
model = Sequential()
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))

model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Custom F1-score metric for Keras
def macro_f1_score(y_true, y_pred):
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.cast(y_true, 'int32')
    y_pred = K.cast(y_pred, 'int32')

    f1_scores = []
    for i in range(len(label_encoder.classes_)):
        true_pos = K.sum(K.cast(y_true == i, 'float32') * K.cast(y_pred == i, 'float32'))
        false_pos = K.sum(K.cast(y_true != i, 'float32') * K.cast(y_pred == i, 'float32'))
        false_neg = K.sum(K.cast(y_true == i, 'float32') * K.cast(y_pred != i, 'float32'))

        precision = true_pos / (true_pos + false_pos + K.epsilon())
        recall = true_pos / (true_pos + false_neg + K.epsilon())

        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        f1_scores.append(f1)

    return K.mean(K.stack(f1_scores))

class MacroF1Score(Callback):
    def __init__(self, model, validation_data):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        self.best_score = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict(self.validation_data[0])
        # Assuming your model's output and y_val are shaped as (samples, timesteps, features) and you want to collapse the timesteps
        #val_pred = np.argmax(val_pred, axis=1)  # Changed axis since no longer sequence
        val_pred = np.argmax(val_pred, axis=-1).flatten()  # Flatten predictions
        val_true = self.validation_data[1].flatten()  # Flatten true labels
        #print(f'Predicted shape: {val_pred.shape}, True shape: {val_true.shape}')
        _val_f1 = f1_score(val_true, val_pred, average='macro')
        logs['val_macro_f1'] = _val_f1  # Ensure this metric is logged
        print(f" - val_macro_f1: {_val_f1:.4f}")

        # Check if the current F1 is better than the best and manually save
        if _val_f1 > self.best_score:
            self.best_score = _val_f1
            self.model.save('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_f1_model.h5')
            print("Saved improved model with F1 score:", _val_f1)

# Define the Adam optimizer with weight decay
#optimizer = Adam(learning_rate=0.001)
optimizer = 'adam'

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#model.compile(loss=CustomLoss(batch_size=2048, label_encoder=label_encoder), optimizer=optimizer, metrics=['accuracy'])


# Define the ModelCheckpoint callback
# checkpoint = ModelCheckpoint('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model.h5', monitor='val_macro_f1', mode='max', save_best_only=True, verbose=1)
# checkpoint = ModelCheckpoint('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

#Initialize the custom F1 score callback
f1_score_callback = MacroF1Score(model, validation_data=(X_val, y_val))

# Train the model with the checkpoint callback
history = model.fit(X_train, y_train, epochs=200, batch_size=1024, validation_data=(X_val, y_val), callbacks=[f1_score_callback])

# Load the best model
# model = tf.keras.models.load_model('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model.h5', custom_objects={'macro_f1_score': macro_f1_score})

model = tf.keras.models.load_model('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_f1_model.h5')

# Evaluate the best model
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=2)

# Flatten the arrays for evaluation
y_val_flat = y_val.flatten()
y_val_pred_flat = y_val_pred_classes.flatten()

# Print the value counts of the flattened arrays
print("Value counts of y_val_flat:", Counter(y_val_flat))
print("Value counts of y_val_pred_flat:", Counter(y_val_pred_flat))

# Calculate and print the macro F1 score
f1 = f1_score(y_val_flat, y_val_pred_flat, average='macro')
print(f"Macro F1 Score: {f1}")

from sklearn.metrics import f1_score, classification_report

# Calculate and print F1 score for each class
print("Classification Report:")
print(classification_report(y_val_flat, y_val_pred_flat, target_names=label_encoder.classes_))


-------------------------------------------------------------------------------------------

#SMOTE

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from collections import Counter
from tensorflow.keras.optimizers import Adam
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/train_gen1.csv')

# Identify columns with at least one NaN value
columns_with_nan = train_df.columns[train_df.isna().any()]
train_df = train_df.drop(columns=columns_with_nan)
zero_col = ['af2__5', 'af2__6', 'af2__13', 'af2__18', 'af2__19', 'af2__20', 'af2__22']
train_df = train_df.drop(columns=zero_col)

# Preprocess the data
# Assuming 'risk level' is the target and other columns are features
X = train_df.drop(columns=['risk_level', 'Timesteps', 'ChassisId_encoded', 'gen'])
y = train_df['risk_level']
print(f"Shape of X after dropping columns: {X.shape}")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f"Shape of X_scaled: {X.shape}")


def create_sequences(X, y, time_steps, jitter_intensity=0.001):
    X_seq, y_seq = [], []
    unique_ids = train_df['ChassisId_encoded'].unique()

    for uid in unique_ids:
        X_sub = X[train_df['ChassisId_encoded'] == uid]
        y_sub = y[train_df['ChassisId_encoded'] == uid]

        # Skip sequences shorter than x time steps
        if len(X_sub) < 10:
            continue

        else:
            # Handle longer sequences based on the label of the last time step
            if y_sub[-1] == 1:
                # Random window from the sequence
                for i in range(1):
                  start_idx = random.randint(0, len(X_sub) - time_steps)
                  X_seq.append(X_sub[start_idx:start_idx + time_steps])
                  y_seq.append(y_sub[start_idx:start_idx + time_steps])
                  #y_seq.append(y_sub[start_idx + time_steps - 1])
                  #for i in range(10):
                    # Create and add jittered sequence
                    #jitter = np.random.normal(0, jitter_intensity, X_sub[start_idx:start_idx + time_steps].shape)
                    #X_jittered = X_sub[start_idx:start_idx + time_steps] + jitter
                    #X_seq.append(X_jittered)
                    #y_seq.append(y_sub[start_idx:start_idx + time_steps])  # Jittering doesn't change labels

            elif y_sub[-1] == 2:
                continue
            elif y_sub[-1] == 0:
                if len(y_sub) >= 10:  # Ensure there are at least 11 elements in the series
                  for i in range(1):
                    # Find indices where the label is 0 and there are at least 10 preceding elements
                    zero_label_indices = np.where(y_sub == 0)[0]
                    # Filter indices to ensure there are at most 9 preceding zeros in the last 10 timesteps
                    valid_indices = [idx for idx in zero_label_indices if idx >= time_steps and
                             np.sum(y_sub[idx - time_steps:idx] == 0) <= 8]
                    if valid_indices:
                      selected_idx = random.choice(valid_indices)
                      # Take the 10 timesteps ending just before the selected index
                      start_idx = selected_idx - time_steps+1
                      end_idx = selected_idx+1
                      X_seq.append(X_sub[start_idx:end_idx])
                      y_seq.append(y_sub[start_idx:end_idx])
                      #y_seq.append(y_sub[selected_idx])
                      #for i in range(10):
                        #jitter = np.random.normal(0, jitter_intensity, X_sub[start_idx:end_idx].shape)
                        #X_jittered = X_sub[start_idx:end_idx] + jitter
                        #X_seq.append(X_jittered)
                        #y_seq.append(y_sub[start_idx:end_idx])  # Jittering doesn't change labels

    return np.array(X_seq), np.array(y_seq)

def augment_sequences(X_seq, y_seq, jitter_intensity=0.01, num_augments=1):

    # Initialize lists to hold the augmented sequences and labels
    X_augmented = []
    y_augmented = []

    # Iterate through each sequence and its corresponding label
    for i in range(len(X_seq)):
        original_sequence = X_seq[i]
        label = y_seq[i]

        # Append the original sequence and label to the augmented lists
        X_augmented.append(original_sequence)
        y_augmented.append(label)

        # Generate jittered versions of the sequence
        for _ in range(num_augments):
            # Create Gaussian noise
            jitter = np.random.normal(loc=0, scale=jitter_intensity, size=original_sequence.shape)
            # Add the jitter to the original sequence to create a new jittered sequence
            jittered_sequence = original_sequence + jitter

            # Append the jittered sequence and the original label to the augmented lists
            X_augmented.append(jittered_sequence)
            y_augmented.append(label)

    # Convert the lists to numpy arrays before returning
    return np.array(X_augmented), np.array(y_augmented)

# Reshape the data back into sequences
time_steps = 10

# Now apply the create_sequences function to the training data only
X_train_seq, y_train_seq = create_sequences(X, y, time_steps)
print(f"Shape of X_seq: {X_train_seq.shape}")
print(f"Shape of y_seq: {y_train_seq.shape}")

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_seq, y_train_seq, test_size=0.2, random_state=42)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")

# Apply jittering to training data only
X_train, y_train = augment_sequences(X_train, y_train, time_steps)

# Standardize the features using only the training data
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_val = scaler.transform(X_val)

from collections import Counter

# Ensure y_train is a flat list
label_counts_train = Counter(y_train.flatten().tolist())  # Flatten and convert to list if not already
print("Label counts in y_train:", label_counts_train)

# Ensure y_train is a flat list
label_counts_val = Counter(y_val.flatten().tolist())  # Flatten and convert to list if not already
print("Label counts in y_val:", label_counts_val)

dropout = 0.5
recurrent_dropout = 0.0
ret = True
reg = 0.0002
nodes = 300

# Define the LSTM model
model = Sequential()
model.add(LSTM(nodes, return_sequences=ret, input_shape=(time_steps, X_train.shape[2]), recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))

model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Custom F1-score metric for Keras
def macro_f1_score(y_true, y_pred):
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.cast(y_true, 'int32')
    y_pred = K.cast(y_pred, 'int32')

    f1_scores = []
    for i in range(len(label_encoder.classes_)):
        true_pos = K.sum(K.cast(y_true == i, 'float32') * K.cast(y_pred == i, 'float32'))
        false_pos = K.sum(K.cast(y_true != i, 'float32') * K.cast(y_pred == i, 'float32'))
        false_neg = K.sum(K.cast(y_true == i, 'float32') * K.cast(y_pred != i, 'float32'))

        precision = true_pos / (true_pos + false_pos + K.epsilon())
        recall = true_pos / (true_pos + false_neg + K.epsilon())

        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        f1_scores.append(f1)

    return K.mean(K.stack(f1_scores))

class MacroF1Score(Callback):
    def __init__(self, model, validation_data):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        self.best_score = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict(self.validation_data[0])
        # Assuming your model's output and y_val are shaped as (samples, timesteps, features) and you want to collapse the timesteps
        #val_pred = np.argmax(val_pred, axis=1)  # Changed axis since no longer sequence
        val_pred = np.argmax(val_pred, axis=-1).flatten()  # Flatten predictions
        val_true = self.validation_data[1].flatten()  # Flatten true labels
        #print(f'Predicted shape: {val_pred.shape}, True shape: {val_true.shape}')
        _val_f1 = f1_score(val_true, val_pred, average='macro')
        logs['val_macro_f1'] = _val_f1  # Ensure this metric is logged
        print(f" - val_macro_f1: {_val_f1:.4f}")

        # Check if the current F1 is better than the best and manually save
        if _val_f1 > self.best_score:
            self.best_score = _val_f1
            self.model.save('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_f1_model.h5')
            print("Saved improved model with F1 score:", _val_f1)

# Define the Adam optimizer with weight decay
#optimizer = Adam(learning_rate=0.001)
optimizer = 'adam'

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#model.compile(loss=CustomLoss(batch_size=2048, label_encoder=label_encoder), optimizer=optimizer, metrics=['accuracy'])


# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model.h5', monitor='val_macro_f1', mode='max', save_best_only=True, verbose=1)
# checkpoint = ModelCheckpoint('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

#Initialize the custom F1 score callback
f1_score_callback = MacroF1Score(model, validation_data=(X_val, y_val))

# Train the model with the checkpoint callback
history = model.fit(X_train, y_train, epochs=200, batch_size=1024, validation_data=(X_val, y_val), callbacks=[f1_score_callback])

# Load the best model
# best_model = tf.keras.models.load_model('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model.h5', custom_objects={'macro_f1_score': macro_f1_score})

model = tf.keras.models.load_model('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_f1_model.h5')

# Evaluate the best model
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=2)

# Flatten the arrays for evaluation
y_val_flat = y_val.flatten()
y_val_pred_flat = y_val_pred_classes.flatten()

# Print the value counts of the flattened arrays
print("Value counts of y_val_flat:", Counter(y_val_flat))
print("Value counts of y_val_pred_flat:", Counter(y_val_pred_flat))

# Calculate and print the macro F1 score
f1 = f1_score(y_val_flat, y_val_pred_flat, average='macro')
print(f"Macro F1 Score: {f1}")

from sklearn.metrics import f1_score, classification_report

# Calculate and print F1 score for each class
print("Classification Report:")
print(classification_report(y_val_flat, y_val_pred_flat, target_names=label_encoder.classes_))



'''
Tentative with LSTM+CNN
'''

#CNN - LSTM

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from collections import Counter
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout, TimeDistributed
from tensorflow.keras.regularizers import l2


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from collections import Counter
from tensorflow.keras.optimizers import Adam
import random


# Preprocess the data
# Assuming 'risk level' is the target and other columns are features
X = train_df.drop(columns=['risk_level', 'Timesteps', 'ChassisId_encoded', 'gen'])
y = train_df['risk_level']
print(f"Shape of X after dropping columns: {X.shape}")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f"Shape of X_scaled: {X.shape}")


def create_sequences(X, y, time_steps):
    X_seq, y_seq = [], []
    unique_ids = train_df['ChassisId_encoded'].unique()

    for uid in unique_ids:
        X_sub = X[train_df['ChassisId_encoded'] == uid]
        y_sub = y[train_df['ChassisId_encoded'] == uid]

        # Skip sequences shorter than x time steps
        if len(X_sub) < 10:
            continue

        else:
            # Handle longer sequences based on the label of the last time step
            if y_sub[-1] == 1:
                # Random window from the sequence
                for i in range(2):
                  start_idx = random.randint(0, len(X_sub) - time_steps)
                  X_seq.append(X_sub[start_idx:start_idx + time_steps])
                  y_seq.append(y_sub[start_idx:start_idx + time_steps])
                  #y_seq.append(y_sub[start_idx + time_steps - 1])
            elif y_sub[-1] == 2:
                continue
            elif y_sub[-1] == 0:
                if len(y_sub) >= 18:  # Ensure there are at least 11 elements in the series
                  for i in range(2):
                    # Find indices where the label is 0 and there are at least 10 preceding elements
                    zero_label_indices = np.where(y_sub == 0)[0]
                    # Filter indices to ensure there are at most 9 preceding zeros in the last 10 timesteps
                    valid_indices = [idx for idx in zero_label_indices if idx >= time_steps and
                             np.sum(y_sub[idx - time_steps:idx] == 0) <= 8]
                    if valid_indices:
                      selected_idx = random.choice(valid_indices)
                      # Take the 10 timesteps ending just before the selected index
                      start_idx = selected_idx - time_steps+1
                      end_idx = selected_idx+1
                      X_seq.append(X_sub[start_idx:end_idx])
                      y_seq.append(y_sub[start_idx:end_idx])
                      #y_seq.append(y_sub[selected_idx])

    return np.array(X_seq), np.array(y_seq)


# Reshape the data back into sequences
time_steps = 10

# Update your script to use the new sequence generation function
X_seq, y_seq = create_sequences(X, y, time_steps)
print(f"Shape of X_seq: {X_seq.shape}")
print(f"Shape of y_seq: {y_seq.shape}")

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")


# Standardize the features using only the training data
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_val = scaler.transform(X_val)

from collections import Counter

# Ensure y_train is a flat list
label_counts_train = Counter(y_train.flatten().tolist())  # Flatten and convert to list if not already
print("Label counts in y_train:", label_counts_train)

# Ensure y_train is a flat list
label_counts_val = Counter(y_val.flatten().tolist())  # Flatten and convert to list if not already
print("Label counts in y_val:", label_counts_val)

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Assuming input_shape is (batch, time_steps, features)
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        # Apply attention mechanism
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return output  # Return shape (batch, time_steps, features)

dropout = 0.5
recurrent_dropout = 0.0
ret = True
reg = 0.0002
nodes = 300


# Define the LSTM model

model = Sequential()

# Add convolutional layer which can be beneficial for extracting features
model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002), input_shape=(time_steps, X_train.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002), input_shape=(time_steps, X_train.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002), input_shape=(time_steps, X_train.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(dropout))

model.add(LSTM(nodes, return_sequences=ret, input_shape=(time_steps, X_train.shape[2]), recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))
model.add(LSTM(nodes, return_sequences=ret, recurrent_dropout=recurrent_dropout, kernel_regularizer=l2(reg)))
model.add(BatchNormalization())  # Add Batch Normalization after LSTM layer
model.add(Dropout(dropout))

model.add(TimeDistributed(Dense(150, activation='relu')))
model.add(Dropout(dropout))
model.add(TimeDistributed(Dense(75, activation='relu')))
model.add(Dropout(dropout))
model.add(TimeDistributed(Dense(25, activation='relu')))
model.add(Dropout(dropout))

# Output layer that predicts for each time step
model.add(TimeDistributed(Dense(len(label_encoder.classes_), activation='softmax')))
# model.add(Dense(len(label_encoder.classes_), activation='softmax'))

class MacroF1Score(Callback):
    def __init__(self, model, validation_data):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        self.best_score = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict(self.validation_data[0])
        val_pred = np.argmax(val_pred, axis=-1)  # Do not flatten yet
        val_true = self.validation_data[1]

        # Check if the predictions need reshaping
        if val_pred.ndim > 1:
            val_pred = val_pred.flatten()
        if val_true.ndim > 1:
            val_true = val_true.flatten()

        _val_f1 = f1_score(val_true, val_pred, average='macro')
        logs['val_macro_f1'] = _val_f1  # Ensure this metric is logged
        print(f" - val_macro_f1: {_val_f1:.4f}")

        # Save the model if F1 score improved
        if _val_f1 > self.best_score:
            self.best_score = _val_f1
            self.model.save('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_f1_model.h5')
            print("Saved improved model with F1 score:", _val_f1)

# Define the Adam optimizer with weight decay
#optimizer = Adam(learning_rate=0.001)
optimizer = 'adam'

'''
def custom_loss(y_true, y_pred):
    # Assuming 'Low' is 0, 'Medium' is 1, and 'High' is 2
    low = tf.constant(1, dtype=tf.int64)
    medium = tf.constant(2, dtype=tf.int64)
    high = tf.constant(0, dtype=tf.int64)

    # Convert predictions to discrete labels
    y_pred_discrete = tf.argmax(y_pred, axis=-1)

    # Check for 'Low' followed by 'Medium' or 'High' or vice versa
    presence_high_medium = tf.logical_or(tf.equal(y_pred_discrete, medium), tf.equal(y_pred_discrete, high))
    presence_low = tf.equal(y_pred_discrete, low)

    # Cumulative sums to track presence of each type
    cumsum_high_medium = tf.cumsum(tf.cast(presence_high_medium, tf.int32), axis=1)
    cumsum_low = tf.cumsum(tf.cast(presence_low, tf.int32), axis=1)

    # Identify invalid transitions
    invalid_low_after_high_medium = tf.logical_and(presence_low, cumsum_high_medium > 0)
    invalid_high_medium_after_low = tf.logical_and(presence_high_medium, cumsum_low > 0)

    # Combine invalid cases
    invalid_cases = tf.logical_or(invalid_low_after_high_medium, invalid_high_medium_after_low)

    # Apply penalties
    penalties = tf.cast(invalid_cases, tf.float32) * 10.0  # Penalty scale factor

    # Calculate the standard loss (e.g., sparse categorical crossentropy)
    standard_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # Combine the standard loss with penalties
    total_loss = standard_loss + penalties
    return total_loss
'''

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, batch_size, label_encoder, lambda_uniformity=0.5, name="custom_loss"):
        super(CustomLoss, self).__init__(name=name)
        self.batch_size = batch_size
        self.lambda_uniformity = lambda_uniformity
        self.label_encoder = label_encoder

    def call(self, y_true, y_pred):
        # Standard sparse categorical crossentropy for classification
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        classification_loss = scce(y_true, y_pred)

        # Calculate the number of 'High' labels in each series
        high_label_id = tf.constant(self.label_encoder.transform(['High'])[0], dtype=tf.int64)
        high_counts = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), high_label_id), tf.int32)
        high_counts = tf.reduce_sum(high_counts, axis=1)

        # Filter out series with zero 'High' labels to focus on 1 to 9
        high_counts = tf.boolean_mask(high_counts, high_counts > 0)

        # Calculate the distribution of 'High' counts
        count_occurrences = tf.math.bincount(high_counts, minlength=10, maxlength=10, dtype=tf.float32)
        count_distribution = count_occurrences / tf.reduce_sum(count_occurrences)

        # Target uniform distribution from 1 to 9 (ignoring the zero count)
        target_distribution = tf.constant([0.] + [1./9] * 9, dtype=tf.float32)

        # Calculate variance as the uniformity penalty
        uniformity_penalty = tf.reduce_mean(tf.square(count_distribution - target_distribution))

        # Combine losses
        return classification_loss + self.lambda_uniformity * uniformity_penalty

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# model.compile(loss=CustomLoss(batch_size=2048, label_encoder=label_encoder), optimizer=optimizer, metrics=['accuracy'])

model.summary()

# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model.h5', monitor='val_macro_f1', mode='max', save_best_only=True, verbose=1)
# checkpoint = ModelCheckpoint('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

#Initialize the custom F1 score callback
f1_score_callback = MacroF1Score(model, validation_data=(X_val, y_val))

# Train the model with the checkpoint callback
history = model.fit(X_train, y_train, epochs=250, batch_size=1024, validation_data=(X_val, y_val), callbacks=[f1_score_callback])

# Load the best model
# best_model = tf.keras.models.load_model('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model.h5', custom_objects={'macro_f1_score': macro_f1_score})

best_model = tf.keras.models.load_model('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_f1_model.h5')

# Evaluate the best model
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=2)

# Flatten the arrays for evaluation
y_val_flat = y_val.flatten()
y_val_pred_flat = y_val_pred_classes.flatten()

# Print the value counts of the flattened arrays
print("Value counts of y_val_flat:", Counter(y_val_flat))
print("Value counts of y_val_pred_flat:", Counter(y_val_pred_flat))

# Calculate and print the macro F1 score
f1 = f1_score(y_val_flat, y_val_pred_flat, average='macro')
print(f"Macro F1 Score: {f1}")

from sklearn.metrics import f1_score, classification_report

# Calculate and print F1 score for each class
print("Classification Report:")
print(classification_report(y_val_flat, y_val_pred_flat, target_names=label_encoder.classes_))


'''
Tentative: predict just the final label
'''


# predict just the final label

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from collections import Counter
from tensorflow.keras.optimizers import Adam
import random

# Load the data
train_df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/train_gen1.csv')
print(f"Initial shape of train_df: {train_df.shape}")

# Fill NaN values with 0
# train_df = train_df.fillna(0)

# Identify columns with at least one NaN value
columns_with_nan = train_df.columns[train_df.isna().any()]
train_df_cleaned = train_df.drop(columns=columns_with_nan)

spec_df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/variants.csv')

# Select relevant columns (from the 2nd to the 13th)
spec_features = spec_df.iloc[:, 1:13]  # Adjust the indices as per your actual data

# Map spec_df by ChassisId_encoded to retrieve one row per ID
spec_features_unique = spec_features.groupby(spec_df['ChassisId_encoded']).first()

# Convert these categories to one-hot encoding
spec_features_one_hot = pd.get_dummies(spec_features_unique, columns=spec_features_unique.columns, dtype=int, prefix=[f"spec_{col}" for col in spec_features_unique.columns])

# Merge these features back into train_df
train_df_cleaned = train_df_cleaned.join(spec_features_one_hot, on='ChassisId_encoded')

# Preprocess the data
# Assuming 'risk level' is the target and other columns are features
X = train_df_cleaned.drop(columns=['risk_level', 'Timesteps', 'ChassisId_encoded', 'gen'])
y = train_df['risk_level']
print(f"Shape of X after dropping columns: {X.shape}")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Shape of X_scaled: {X_scaled.shape}")


def create_sequences(X, y, time_steps):
    X_seq, y_seq = [], []
    unique_ids = train_df['ChassisId_encoded'].unique()

    for uid in unique_ids:
        X_sub = X[train_df['ChassisId_encoded'] == uid]
        y_sub = y[train_df['ChassisId_encoded'] == uid]

        # Skip sequences shorter than x time steps
        if len(X_sub) < 10:
            continue

        else:
            # Handle longer sequences based on the label of the last time step
            if y_sub[-1] == 1:
                # Random window from the sequence
                for i in range(1):
                  start_idx = random.randint(0, len(X_sub) - time_steps)
                  X_seq.append(X_sub[start_idx:start_idx + time_steps])
                  y_seq.append(y_sub[start_idx + time_steps - 1])  # Only add the last label of the window
            elif y_sub[-1] == 2:
              continue
            elif y_sub[-1] == 0:
              for i in range(1):
                # Find a random index with label 0 and take a window ending at this index
                zero_label_indices = np.where(y_sub == 0)[0]
                # Ensure index has at most 8 preceding zeros and is large enough for a sequence
                valid_indices = [idx for idx in zero_label_indices if idx >= time_steps - 1 and
                                 np.sum(y_sub[idx - time_steps:idx] == 0) <= 8]
                if len(valid_indices) > 0:
                    selected_idx = random.choice(valid_indices)
                    X_seq.append(X_sub[selected_idx - time_steps + 1:selected_idx + 1])
                    y_seq.append(y_sub[selected_idx])  # Only add the label at the selected index

    return np.array(X_seq), np.array(y_seq)

# Reshape the data back into sequences
time_steps = 10

# Update your script to use the new sequence generation function
X_seq, y_seq = create_sequences(X_scaled, y, time_steps)
print(f"Shape of X_seq: {X_seq.shape}")
print(f"Shape of y_seq: {y_seq.shape}")

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")

from collections import Counter

# Ensure y_train is a flat list
label_counts_train = Counter(y_train.flatten().tolist())  # Flatten and convert to list if not already
print("Label counts in y_train:", label_counts_train)

# Ensure y_train is a flat list
label_counts_val = Counter(y_val.flatten().tolist())  # Flatten and convert to list if not already
print("Label counts in y_val:", label_counts_val)



from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(time_steps, X_train.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv1D(64, 3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Custom F1-score metric for Keras
def macro_f1_score(y_true, y_pred):
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.cast(y_true, 'int32')
    y_pred = K.cast(y_pred, 'int32')

    f1_scores = []
    for i in range(len(label_encoder.classes_)):
        true_pos = K.sum(K.cast(y_true == i, 'float32') * K.cast(y_pred == i, 'float32'))
        false_pos = K.sum(K.cast(y_true != i, 'float32') * K.cast(y_pred == i, 'float32'))
        false_neg = K.sum(K.cast(y_true == i, 'float32') * K.cast(y_pred != i, 'float32'))

        precision = true_pos / (true_pos + false_pos + K.epsilon())
        recall = true_pos / (true_pos + false_neg + K.epsilon())

        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        f1_scores.append(f1)

    return K.mean(K.stack(f1_scores))

# Define the Adam optimizer with weight decay
# optimizer = Adam(learning_rate=0.001, weight_decay=1e-4)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=[macro_f1_score])
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model_finallabel.h5', monitor='val_macro_f1_score', mode='max', save_best_only=True, verbose=1)
# checkpoint = ModelCheckpoint('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

# Train the model with the checkpoint callback
history = model.fit(X_train, y_train, epochs=200, batch_size=1024, validation_data=(X_val, y_val), callbacks=[checkpoint], class_weight=class_weight_dict)

# Load the best model
best_model = tf.keras.models.load_model('/content/gdrive/My Drive/Colab Notebooks/Volvo/lstm_best_model_finallabel.h5', custom_objects={'macro_f1_score': macro_f1_score})

# Evaluate the best model
y_val_pred = best_model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

# Flatten the arrays for evaluation
y_val_flat = y_val.flatten()
y_val_pred_flat = y_val_pred_classes.flatten()

# Print the value counts of the flattened arrays
print("Value counts of y_val_flat:", Counter(y_val_flat))
print("Value counts of y_val_pred_flat:", Counter(y_val_pred_flat))

# Calculate and print the macro F1 score
f1 = f1_score(y_val_flat, y_val_pred_flat, average='macro')
print(f"Macro F1 Score: {f1}")


