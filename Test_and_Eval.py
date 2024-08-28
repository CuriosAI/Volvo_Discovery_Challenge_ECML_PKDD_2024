'''
Test and evaluation
'''


import matplotlib.pyplot as plt

def count_high(sequence):
    # Count the occurrences of "High" using numpy
    return np.count_nonzero(sequence == 0)

# Apply the function to count "High" in each sequence
high_counts = [count_high(seq) for seq in y_train]

# Filter out sequences with zero "High"
high_counts = [count for count in high_counts if count > 0]

# Plotting the histogram
plt.hist(high_counts, bins=range(1,12), edgecolor='black')
plt.title('Histogram of "High" Counts (At least 1 "High")')
plt.xlabel('Number of "High" in Sequence')
plt.ylabel('Frequency')
plt.xticks(range(min(high_counts), max(high_counts) + 1))  # Ensure all bins are labeled
plt.show()

def count_high(sequence):
    # Count the occurrences of "High" using numpy
    return np.count_nonzero(sequence == 0)

# Apply the function to count "High" in each sequence
high_counts = [count_high(seq) for seq in y_val]

# Filter out sequences with zero "High"
high_counts = [count for count in high_counts if count > 0]

# Plotting the histogram
plt.hist(high_counts, bins=range(1,12), edgecolor='black')
plt.title('Histogram of "High" Counts (At least 1 "High")')
plt.xlabel('Number of "High" in Sequence')
plt.ylabel('Frequency')
plt.xticks(range(min(high_counts), max(high_counts) + 1))  # Ensure all bins are labeled
plt.show()

def count_high(sequence):
    # Count the occurrences of "High" using numpy
    return np.count_nonzero(sequence == 0)

# Apply the function to count "High" in each sequence
high_counts = [count_high(seq) for seq in y_val_pred_classes]

# Filter out sequences with zero "High"
high_counts = [count for count in high_counts if count > 0]

# Plotting the histogram
plt.hist(high_counts, bins=range(1,12), edgecolor='black')
plt.title('Histogram of "High" Counts (At least 1 "High")')
plt.xlabel('Number of "High" in Sequence')
plt.ylabel('Frequency')
plt.xticks(range(min(high_counts), max(high_counts) + 1))  # Ensure all bins are labeled
plt.show()





import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the test data
df_test = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/public_X_test.csv')

# Fill NaN values with 0
# df_test  = df_test.fillna(0)

# Identify columns with at least one NaN value
columns_with_nan = df_test.columns[df_test.isna().any()]
df_test_cleaned = df_test.drop(columns=columns_with_nan)
zero_col = ['af2__5', 'af2__6', 'af2__13', 'af2__18', 'af2__19', 'af2__20', 'af2__22']
df_test_cleaned = df_test_cleaned.drop(columns=zero_col)

'''
spec_df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/variants.csv')

# Select relevant columns (from the 2nd to the 13th)
spec_features = spec_df.iloc[:, 1:13]  # Adjust the indices as per your actual data

# Map spec_df by ChassisId_encoded to retrieve one row per ID
spec_features_unique = spec_features.groupby(spec_df['ChassisId_encoded']).first()

# Convert these categories to one-hot encoding
spec_features_one_hot = pd.get_dummies(spec_features_unique, columns=spec_features_unique.columns, dtype=int, prefix=[f"spec_{col}" for col in spec_features_unique.columns])

# Merge these features back into train_df
df_test_cleaned = df_test_cleaned.join(spec_features_one_hot, on='ChassisId_encoded')
'''

# Preprocess the test data
X_test = df_test_cleaned.drop(columns=['Timesteps', 'ChassisId_encoded', 'gen'])
#X_test = X_test[significant_features]

# X_test_scaled = scaler.transform(X_test)

# Standardize the features
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)


# Define the sequence length (e.g., 10 timesteps)
time_steps = 10
num_features = X_test_scaled.shape[1]  # Number of features after scaling

def create_test_sequences(X, time_steps):
    X_seq = []
    unique_ids = df_test['ChassisId_encoded'].unique()
    for uid in unique_ids:
        X_sub = X[df_test['ChassisId_encoded'] == uid]
        for i in range(0, len(X_sub), time_steps):
            X_seq.append(X_sub[i:i + time_steps])
    return np.array(X_seq)

X_test_seq = create_test_sequences(X_test_scaled, time_steps)

# Pad sequences to ensure uniform length
X_test_seq_padded = pad_sequences(X_test_seq, maxlen=time_steps, dtype='float32', padding='post', truncating='post')

# Ensure input shape matches the model's expected input shape
# Check the shape of the padded sequences
print("Shape of X_test_seq_padded:", X_test_seq_padded.shape)  # Should be (num_sequences, time_steps, num_features)

# Load the trained model
# model = load_model('lstm_model.h5')

# Predict the risk levels for the test data
y_test_pred = model.predict(X_test_seq_padded)
y_test_pred_classes = np.argmax(y_test_pred, axis=2).flatten()

# Inverse mapping to convert numerical labels back to original categorical labels
label_mapping = {0: 'High', 1: 'Low', 2: 'Medium'}
y_test_pred_labels = [label_mapping[pred] for pred in y_test_pred_classes]

# Create a DataFrame with the prediction vector
pred_df = pd.DataFrame(y_test_pred_labels, columns=["pred"])

# Insert the header row
pred_df.loc[-1] = ["pred"]  # Add a row at the beginning for the header
pred_df.index = pred_df.index + 1  # Shift index
pred_df = pred_df.sort_index()  # Sort by index to ensure the header is first

# Save the DataFrame to a CSV file
pred_df.to_csv("/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction.csv", index=False, header=False)

print("Prediction vector saved to prediction.csv")




#splitting gen1 and gen2

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the test data
df_test = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/public_X_test.csv')

# Drop columns with NaNs as previously done
columns_with_nan = df_test.columns[df_test.isna().any()]
df_test_cleaned = df_test.drop(columns=columns_with_nan)
zero_col = ['af2__5', 'af2__6', 'af2__13', 'af2__18', 'af2__19', 'af2__20', 'af2__22']
#zero_col = ['af2__5', 'af2__18']
df_test_cleaned = df_test_cleaned.drop(columns=zero_col)

# Split data into gen1 and gen2
gen1_data = df_test_cleaned[df_test_cleaned['gen'] == 'gen1']
gen2_data = df_test_cleaned[df_test_cleaned['gen'] == 'gen2']

# Remove 'gen' column before scaling
gen1_data = gen1_data.drop(columns=['gen'])
gen2_data = gen2_data.drop(columns=['gen'])

# Apply StandardScaler independently
scaler_gen1 = StandardScaler()
gen1_scaled = scaler_gen1.fit_transform(gen1_data)

scaler_gen2 = StandardScaler()
gen2_scaled = scaler_gen2.fit_transform(gen2_data)

# Convert scaled arrays back to DataFrames
gen1_df = pd.DataFrame(gen1_scaled, columns=gen1_data.columns, index=gen1_data.index)
gen2_df = pd.DataFrame(gen2_scaled, columns=gen2_data.columns, index=gen2_data.index)

# Concatenate back together
X_test_scaled = pd.concat([gen1_df, gen2_df]).sort_index()

# Preprocess the test data
X_test_scaled = X_test_scaled.drop(columns=['Timesteps', 'ChassisId_encoded'])

# Define the sequence length (e.g., 10 timesteps)
time_steps = 10
num_features = X_test_scaled.shape[1]  # Number of features after scaling

def create_test_sequences(X, time_steps):
    X_seq = []
    unique_ids = df_test['ChassisId_encoded'].unique()
    for uid in unique_ids:
        X_sub = X[df_test['ChassisId_encoded'] == uid]
        for i in range(0, len(X_sub), time_steps):
            X_seq.append(X_sub[i:i + time_steps])
    return np.array(X_seq)

X_test_seq = create_test_sequences(X_test_scaled, time_steps)

# Pad sequences to ensure uniform length
X_test_seq_padded = pad_sequences(X_test_seq, maxlen=time_steps, dtype='float32', padding='post', truncating='post')

# Ensure input shape matches the model's expected input shape
# Check the shape of the padded sequences
print("Shape of X_test_seq_padded:", X_test_seq_padded.shape)  # Should be (num_sequences, time_steps, num_features)

model = tf.keras.models.load_model('/content/gdrive/My Drive/Colab Notebooks/Volvo/trans_best_f1_model_16_timeseries_timesteps_False.h5')

# Load the trained model
# model = load_model('lstm_model.h5')

# Predict the risk levels for the test data
y_test_pred = model.predict(X_test_seq_padded)
y_test_pred_classes = np.argmax(y_test_pred, axis=2).flatten()

# Inverse mapping to convert numerical labels back to original categorical labels
label_mapping = {0: 'High', 1: 'Low', 2: 'Medium'}
y_test_pred_labels = [label_mapping[pred] for pred in y_test_pred_classes]

# Create a DataFrame with the prediction vector
pred_df = pd.DataFrame(y_test_pred_labels, columns=["pred"])

# Insert the header row
pred_df.loc[-1] = ["pred"]  # Add a row at the beginning for the header
pred_df.index = pred_df.index + 1  # Shift index
pred_df = pred_df.sort_index()  # Sort by index to ensure the header is first

# Save the DataFrame to a CSV file
pred_df.to_csv("/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction.csv", index=False, header=False)

print("Prediction vector saved to prediction.csv")


'''
Consistency checks
'''

# Load the prediction counts CSV, skipping the first row
file_path = '/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction_counts.csv'
df_counts = pd.read_csv(file_path, header=None, skiprows=1)

# Prepare a list to collect the expanded values
expanded_predictions = ["pred"]

# Define the rules for expansion
for index, row in df_counts.iterrows():
    count = int(row[0])
    if count == 0:
        expanded_predictions.extend(["Low"] * 10)
    else:
        expanded_predictions.extend(["Medium"] * count + ["High"] * (10 - count))

# Create a new DataFrame
df_expanded = pd.DataFrame(expanded_predictions)

# Save the DataFrame to a new CSV file
output_path = '/content/gdrive/My Drive/Colab Notebooks/Volvo/expanded_predictions.csv'
df_expanded.to_csv(output_path, index=False, header=False)

print("Expanded prediction vector saved to expanded_predictions.csv")

# Load the data, skipping the first row as it is just a label
file_path = '/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction.csv'
pred_df = pd.read_csv(file_path, header=None, skiprows=1)  # Skip the first row, which is 'pred'

# Reshape the data into sequences of 10, as we now have 33590 entries, divisible by 10
sequences = pred_df[0].values.reshape(-1, 10)

def is_sequence_valid(sequence):
    # Define the correct order for the terms
    valid_order = ["Low", "Medium", "High"]
    order_dict = {k: i for i, k in enumerate(valid_order)}  # Map values to their positions
    filtered_seq = [x for x in sequence if x in valid_order]  # Filter valid terms
    indices = [order_dict[x] for x in filtered_seq]  # Convert to indices based on order
    return indices == sorted(indices)  # Check if sorted indices match the original, which means they are in order

# Apply the function to check each sequence
results = [is_sequence_valid(seq) for seq in sequences]

# Count and report inconsistencies
inconsistent_count = sum(not x for x in results)

print(f"Number of inconsistent sequences: {inconsistent_count}")
print("Indices of inconsistent sequences:", [i for i, x in enumerate(results) if not x])

# Load the data, skipping the first row as it is just a label
file_path = '/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction.csv'
pred_df = pd.read_csv(file_path, header=None, skiprows=1)  # Skip the first row, which is 'pred'

# Reshape the data into sequences of 10, as we now have 33590 entries, divisible by 10
sequences = pred_df[0].values.reshape(-1, 10)

def contains_high_and_low(sequence):
    # Check if both "High" and "Low" are present in the sequence
    return "Medium" in sequence and "Low" in sequence

# Apply the function to find sequences containing both "High" and "Low"
sequences_with_high_and_low = [contains_high_and_low(seq) for seq in sequences]

# Count and report sequences containing both "High" and "Low"
count_high_and_low = sum(sequences_with_high_and_low)

print(f"Number of sequences containing both 'High' and 'Low': {count_high_and_low}")
print("Indices of sequences containing both 'High' and 'Low':", [i for i, x in enumerate(sequences_with_high_and_low) if x])

# Load the data, skipping the first row as it is just a label
file_path = '/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction.csv'
pred_df = pd.read_csv(file_path, header=None, skiprows=1)  # Skip the first row, which is 'pred'

# Reshape the data into sequences of 10, as we now have 33590 entries, divisible by 10
sequences = pred_df[0].values.reshape(-1, 10)

def contains_high_and_low(sequence):
    # Check if both "High" and "Low" are present in the sequence
    return "High" in sequence and "Low" in sequence

# Apply the function to find sequences containing both "High" and "Low"
sequences_with_high_and_low = [contains_high_and_low(seq) for seq in sequences]

# Count and report sequences containing both "High" and "Low"
count_high_and_low = sum(sequences_with_high_and_low)

print(f"Number of sequences containing both 'High' and 'Low': {count_high_and_low}")
print("Indices of sequences containing both 'High' and 'Low':", [i for i, x in enumerate(sequences_with_high_and_low) if x])

# Load the CSV file
df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/public_X_test.csv')

# Find the indexes where 'gen' column is 'gen1'
gen1_indexes = df.index[df['gen'] == 'gen1'].tolist()
# Find the indexes where 'gen' column is 'gen1'
gen2_indexes = df.index[df['gen'] == 'gen2'].tolist()

# Load the data, skipping the first row as it is just a label
file_path = '/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction.csv'
pred_df = pd.read_csv(file_path, header=None, skiprows=1)  # Skip the first row, which is 'pred'
#pred_df = pred_df.iloc[0:15100]
# Reshape the data into sequences of 10, as we now have 33590 entries, divisible by 10
sequences = pred_df[0].values.reshape(-1, 10)

def count_high(sequence):
    # Count the occurrences of "High" using numpy
    return np.count_nonzero(sequence == "High")

# Apply the function to count "High" in each sequence
high_counts = [count_high(seq) for seq in sequences]

# Filter out sequences with zero "High"
high_counts = [count for count in high_counts if count > 0]

# Plotting the histogram
plt.hist(high_counts, bins=range(1,11), edgecolor='black')
plt.title('Histogram of "High" Counts')
plt.xlabel('Number of "High" in Sequence')
plt.ylabel('Frequency')
plt.xticks(range(min(high_counts), max(high_counts) + 1))  # Ensure all bins are labeled
plt.show()

# Load the data, skipping the first row as it is just a label
file_path = '/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction.csv'
pred_df = pd.read_csv(file_path, header=None, skiprows=1)  # Skip the first row, which is 'pred'

# Define the function to count "High"
def count_high(sequence):
    return np.count_nonzero(sequence == "High")

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# First plot for "gen1"
pred_df_gen1 = pred_df.iloc[gen1_indexes]
sequences_gen1 = pred_df_gen1[0].values.reshape(-1, 10)
high_counts_gen1 = [count_high(seq) for seq in sequences_gen1]
high_counts_gen1 = [count for count in high_counts_gen1 if count > 0]
axes[0].hist(high_counts_gen1, bins=range(1, 11), edgecolor='black', color='blue')
axes[0].set_title('Histogram of "High" Counts for gen1')
axes[0].set_xlabel('Number of "High" in Sequence')
axes[0].set_ylabel('Frequency')
axes[0].set_ylim(0,150)
axes[0].set_xticks(range(min(high_counts_gen1), max(high_counts_gen1) + 1))

# Second plot for "gen2"
pred_df_gen2 = pred_df.iloc[gen2_indexes]
sequences_gen2 = pred_df_gen2[0].values.reshape(-1, 10)
high_counts_gen2 = [count_high(seq) for seq in sequences_gen2]
high_counts_gen2 = [count for count in high_counts_gen2 if count > 0]
axes[1].hist(high_counts_gen2, bins=range(1, 11), edgecolor='black', color='red')
axes[1].set_title('Histogram of "High" Counts for gen2')
axes[1].set_xlabel('Number of "High" in Sequence')
axes[1].set_ylabel('Frequency')
axes[1].set_ylim(0,200)
axes[1].set_xticks(range(min(high_counts_gen2), max(high_counts_gen2) + 1))

# Show the plots
plt.tight_layout()
plt.show()

def count_high(sequence):
    # Count the occurrences of "High" using numpy
    return np.count_nonzero(sequence == "Medium")

# Apply the function to count "High" in each sequence
high_counts = [count_high(seq) for seq in sequences]

# Filter out sequences with zero "High"
high_counts = [count for count in high_counts if count > 0]

# Plotting the histogram
plt.hist(high_counts, bins=range(1,11), edgecolor='black')
plt.title('Histogram of "High" Counts')
plt.xlabel('Number of "High" in Sequence')
plt.ylabel('Frequency')
plt.xticks(range(min(high_counts), max(high_counts) + 1))  # Ensure all bins are labeled
plt.show()

# Load the data from CSV files
prediction_df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction.csv', skiprows=1, header=None, names=['pred'])
prediction_final_df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction_final.csv', skiprows=1, header=None, names=['pred_final'])

# Initialize the list that will hold the final values
final_values = ['pred']  # Starting with 'pred' as the first row

# Get the number of complete 10-length windows
num_windows = len(prediction_df) // 10

for i in range(num_windows):
    start_idx = i * 10
    end_idx = start_idx + 10

    # Get the current window from prediction.csv
    prediction_window = prediction_df.iloc[start_idx:end_idx]

    # Check the last value in the current prediction window
    if prediction_window.iloc[-1, 0] == 'Low':
        # If the last value is 'Low', use this window for the final.csv
        final_values.extend(prediction_window['pred'].tolist())
    elif prediction_window.iloc[-1, 0] == 'High':
        # If the last value is 'High', use the corresponding window from prediction_final.csv
        final_window = prediction_final_df.iloc[start_idx:end_idx]
        final_values.extend(final_window['pred_final'].tolist())

# If the length of prediction_df is not a multiple of 10, handle the remaining rows
if len(prediction_df) % 10 != 0:
    remaining_rows = prediction_df.iloc[num_windows * 10:]
    final_values.extend(remaining_rows['pred'].tolist())

# Convert the final values list into a DataFrame
final_df = pd.DataFrame(final_values, columns=['pred'])

### Step 4: Write to CSV
final_df.to_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/final.csv', index=False, header=False)

pred_df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction.csv')

# Count the number of predictions for each label
label_counts = pred_df['pred'].value_counts()

print("Ensemble prediction vector saved to prediction.csv")
print("Label counts in the final ensemble predictions:")
print(label_counts)
print(len(pred_df))

# Extracting the last label of each sequence for y_train and y_val
y_train_last = np.array([seq[-1] for seq in y_train])
y_val_last = np.array([seq[-1] for seq in y_val])

def plot_label_histogram(y, title):
    # Counting occurrences of each label
    labels, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(8, 4))
    plt.bar(labels, counts, color=['red', 'green', 'blue'])
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(labels, ['Label 0', 'Label 1', 'Label 2'])
    plt.show()

# Plotting histograms for y_train_last and y_val_last
plot_label_histogram(y_train_last, 'Label Distribution at the End of Series in y_train')
plot_label_histogram(y_val_last, 'Label Distribution at the End of Series in y_val')

def plot_original_label_histogram(y, title):
    # Counting occurrences of each label
    labels, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(8, 4))
    plt.bar(labels, counts, color=['red', 'green', 'blue'])
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(labels, ['Label 0', 'Label 1', 'Label 2'])
    plt.show()

# Plotting histogram for original y
plot_original_label_histogram(y, 'Label Distribution in Original Dataset')

