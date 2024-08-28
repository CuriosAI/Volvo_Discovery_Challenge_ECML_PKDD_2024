'''

Exploratory Data Analysis and Data Inspection

Collection of all the script and piece of code to inspect the dataset
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/public_X_test.csv')

# Replace 'gen1' with 1 and 'gen2' with 2 in the 'gen' column
df['gen'].replace({'gen1': 1, 'gen2': 2}, inplace=True)

# Extract the 'gen' column as a DataSeries
gen_series = df['gen']

# Save the DataSeries to a new CSV file
gen_series.to_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/gen_series.csv', index=False)




# Load the CSV files
df_prediction = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction.csv')
df_prediction_binary = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction_binary.csv')

# Initialize the DataFrame for the final predictions
prediction_final = []

# Iterate over each label in prediction_binary.csv
for index, binary_label in enumerate(df_prediction_binary['pred']):
    # Get the corresponding block of 10 labels from prediction.csv
    block_start = index * 10
    block_end = block_start + 10
    block_labels = df_prediction['pred'][block_start:block_end]

    # Apply rules based on the value in prediction_binary.csv
    if binary_label == 'Low':
        # Rule 1: If binary label is 'Low', all corresponding labels are 'Low'
        prediction_final.extend(['Low'] * 10)
    elif binary_label == 'High':
        # Rule 2 and 3: If binary label is 'High'
        if 'High' in block_labels.values:
            # Copy the 10 values if there is at least one 'High'
            prediction_final.extend(block_labels.values)
        else:
            # If there is no 'High', write 5 'Medium' and then 5 'High'
            prediction_final.extend(['Medium'] * 5 + ['High'] * 5)

# Create a new DataFrame for the final predictions
df_prediction_final = pd.DataFrame(prediction_final, columns=['pred'])

# Save the DataFrame to a new CSV file
df_prediction_final.to_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction_final.csv', index=False)




# Load the CSV file
df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/public_X_test.csv')

# Check if rows are ordered by the 'gen' column with 'gen1' before 'gen2'
gen1_rows = df['gen'] == 'gen1'
gen2_rows = df['gen'] == 'gen2'

# Find the index of the last 'gen1' and the first 'gen2'
last_gen1_index = df[gen1_rows].index.max()
first_gen2_index = df[gen2_rows].index.min()

# Verify ordering
if last_gen1_index < first_gen2_index:
    print("Rows are correctly ordered: All 'gen1' rows come before 'gen2' rows.")
else:
    print("Ordering error: Some 'gen2' rows appear before 'gen1' rows.")




import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/public_X_test.csv')

# Prepare a simple count for each row to represent in the bar
df['Count'] = range(1, len(df) + 1)

# Assign colors based on 'gen' values
colors = df['gen'].map({'gen1': 'blue', 'gen2': 'red'})

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(df.index, df['Count'], color=colors)
plt.xlabel('Index')
plt.ylabel('Count')
plt.title('Bar Plot by Gen Type')
plt.show()




# Load the CSV files
df_test = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/public_X_test.csv')
df_gen1 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction_gen1.csv', skiprows=1, header=None, names=['pred'])
df_gen2 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction_gen2.csv', skiprows=1, header=None, names=['pred'])

# Check data alignment
if len(df_gen1) != len(df_test) or len(df_gen2) != len(df_test):
    raise ValueError("The length of the label files does not match the test file.")

# Initialize the list for combined predictions
combined_predictions = []

# Iterate over the rows in public_X_test.csv using its index for accurate mapping
for index, row in df_test.iterrows():
    try:
        if row['gen'] == 'gen1':
            # Copy the label from prediction_gen1.csv
            combined_predictions.append(df_gen1['pred'].iloc[index])
        elif row['gen'] == 'gen2':
            # Copy the label from prediction_gen2.csv
            combined_predictions.append(df_gen2['pred'].iloc[index])
    except IndexError:
        print(f"Failed at index {index}: Row does not exist in one of the label files.")

# Create a new DataFrame for the combined predictions
df_combined = pd.DataFrame(combined_predictions, columns=['pred'])

# Save the DataFrame to a new CSV file
df_combined.to_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction_combined.csv', index=False)




# Load the CSV files
df_test = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/public_X_test.csv')
df_gen1 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction_gen1.csv', skiprows=1, header=None, names=['pred'])
df_gen2 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction_gen2.csv', skiprows=1, header=None, names=['pred'])

df_final = pd.DataFrame()

df_final['pred1'] =  df_gen1['pred']
df_final['pred2'] =  df_gen2['pred']
df_final['gen'] =  df_test['gen']

# Define the conditions to set values in the new fourth column
# Check the value in the third column (index 2 since indexing starts at 0)
df_final['comb'] = df_final.apply(lambda row: row[0] if row[2] == 'gen1' else row[1], axis=1)

# Save the updated DataFrame to a new CSV file, if you need to keep the changes
df_final.to_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/prediction_combined.csv', index=False, header=False)

df_final['comb'].to_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/predictions.csv', index=False, header=False)



# Load the CSV file
df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/public_X_test.csv')

# Find the indexes where 'gen' column is 'gen1'
gen1_indexes = df.index[df['gen'] == 'gen1'].tolist()
# Find the indexes where 'gen' column is 'gen1'
gen2_indexes = df.index[df['gen'] == 'gen2'].tolist()

df_1 = df.iloc[gen1_indexes]
df_2 = df.iloc[gen2_indexes]

# Check each column to see if all values are zero
zero_columns = [col for col in X_test.columns if (X_test_1[col] == 0).all()]

# Output the result
if zero_columns:
    print("Columns where all values are zero:", zero_columns)
else:
    print("There are no columns where all values are zero.")

# Check each column to see if all values are zero
zero_columns = [col for col in X_test.columns if (X_test_2[col] == 0).all()]

# Output the result
if zero_columns:
    print("Columns where all values are zero:", zero_columns)
else:
    print("There are no columns where all values are zero.")

# Check each column to see if all values are zero
zero_columns = [col for col in X_train.columns if (X_train[col] == 0).all()]

# Output the result
if zero_columns:
    print("Columns where all values are zero:", zero_columns)
else:
    print("There are no columns where all values are zero.")

# Load the test data
df_test = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/public_X_test.csv')

# Fill NaN values with 0
# df_test  = df_test.fillna(0)

# Identify columns with at least one NaN value
columns_with_nan = df_test.columns[df_test.isna().any()]
df_test_cleaned = df_test.drop(columns=columns_with_nan)
zero_col = ['af2__5', 'af2__6', 'af2__13', 'af2__18', 'af2__19', 'af2__20', 'af2__22']
df_test_cleaned = df_test_cleaned.drop(columns=zero_col)
df_test_cleaned = df_test_cleaned.drop(columns=['Timesteps', 'ChassisId_encoded', 'gen'])

df_1 = df_test_cleaned.iloc[gen1_indexes]
df_2 = df_test_cleaned.iloc[gen2_indexes]

df_2.describe()
df_2.describe().loc['mean'].plot()




def check_non_decreasing(series):
    mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    numeric_risk_levels = series.map(mapping)
    return all(numeric_risk_levels.iloc[i] <= numeric_risk_levels.iloc[i + 1] for i in range(len(numeric_risk_levels) - 1))

# Filter the DataFrame for series with at least 10 time steps
filtered_df = train_df.groupby('ChassisId_encoded').filter(lambda x: len(x) >= 10)

# Ensure that within each series, the risk level is non-decreasing
filtered_df = filtered_df.groupby('ChassisId_encoded').filter(lambda x: check_non_decreasing(x['risk_level']))

# Further filter to include only those series that end with a 'High' and have at least 9 'High's and 9 'Medium's
def check_highs_and_mediums(group):
    if group['risk_level'].iloc[-1] == 'High':
        counts = group['risk_level'].value_counts()
        return counts.get('High', 0) >= 9 and counts.get('Medium', 0) >= 9
    return True

train_df = filtered_df.groupby('ChassisId_encoded').filter(check_highs_and_mediums)

# Print the filtered DataFrame
train_df

columns_with_nan = train_df.columns[train_df.isna().any()]
columns_with_nan

columns_with_nan = train_df.columns[train_df.isna().any()]
train_df_cleaned = train_df.drop(columns=columns_with_nan)

feature_columns = [col for col in train_df_cleaned.columns if col not in ['risk_level', 'Timesteps', 'ChassisId_encoded', 'gen']]

def is_monotonous(series):
    return all(series.iloc[i] <= series.iloc[i + 1] for i in range(len(series) - 1))

monotonous_results = {}
unique_chassis_ids = train_df_cleaned['ChassisId_encoded'].unique()

for chassis_id in unique_chassis_ids:
    monotonous_results[chassis_id] = {}
    chassis_data = train_df_cleaned[train_df['ChassisId_encoded'] == chassis_id]
    for feature in feature_columns:
        monotonous_results[chassis_id][feature] = is_monotonous(chassis_data[feature])

monotonous_results





# Count the number of occurrences for each unique ChassisId_encoded
series_lengths = train_df['ChassisId_encoded'].value_counts()

# Plotting the histogram of the series lengths
series_lengths.hist(bins=35, figsize=(10, 6))


min(series_lengths), max(series_lengths), np.mean(series_lengths)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'risk_level' is at the last timestep and already contains 'Low', 'Medium', 'High' as categories
# Group by 'ChassisId_encoded' to determine the length and last risk_level of each series
grouped = train_df.groupby('ChassisId_encoded').agg(length=('risk_level', 'size'), last_label=('risk_level', 'last'))

# Count occurrences of lengths and labels
length_label_counts = grouped.groupby(['length', 'last_label']).size().unstack(fill_value=0)

# Calculate proportions
length_label_props = length_label_counts.divide(length_label_counts.sum(axis=1), axis=0)

# Plot
fig, ax = plt.subplots()
length_label_props.plot(kind='bar', stacked=True, ax=ax, color=['red', 'yellow', 'green'])
ax.set_xlabel('Length of Time Series')
ax.set_ylabel('Proportion of Labels')
ax.set_title('Proportions of Last Labels in Time Series of Different Lengths')
plt.show()



# Group by 'ChassisId_encoded' assuming each group is a full series
# Adjust this if your data is structured differently
grouped = train_df.groupby('ChassisId_encoded')

# Initialize a list to store DataFrames
high_risk_series_list = []

for name, group in grouped:
    if group['risk_level'].iloc[-1] == 'High':  # Check if the last entry in the group is 'High'
        high_risk_series_list.append(group)

# Concatenate all the DataFrames in the list into one DataFrame
high_risk_series = pd.concat(high_risk_series_list, ignore_index=True)

ids = high_risk_series['ChassisId_encoded'].unique()

i = 3
index = ids[i]
high_risk_series[high_risk_series['ChassisId_encoded']==index]

# Dictionary to store the counts for 'Medium' and 'High' for each ChassisId_encoded
counts_dict = {}

for index in ids:
    # Filter the series for the current ChassisId_encoded
    series_data = high_risk_series[high_risk_series['ChassisId_encoded'] == index]

    # Count 'Medium' and 'High'
    counts = series_data['risk_level'].value_counts()

    # Extract counts for 'Medium' and 'High', handling cases where they might not exist
    low_count = counts.get('Low', 0)
    medium_count = counts.get('Medium', 0)
    high_count = counts.get('High', 0)

    # Store in dictionary
    counts_dict[index] = {'Low': low_count, 'Medium': medium_count, 'High': high_count}

# Print the counts
for chassis_id, counts in counts_dict.items():
    print(f"Chassis ID: {chassis_id}, Low: {counts['Low']}, Medium: {counts['Medium']}, High: {counts['High']}")

for chassis_id, counts in counts_dict.items():
  if counts['High']<9:
    print(f"Chassis ID: {chassis_id}, Low: {counts['Low']}, Medium: {counts['Medium']}, High: {counts['High']}")



import pandas as pd
import matplotlib.pyplot as plt

# Dictionary to store the counts of series with specific counts of 'High'
high_counts_distribution = {}

# Aggregate data to count the number of series with each count of 'High' from 1 to 10
for chassis_id, counts in counts_dict.items():
    high_count = counts['High']
    if 1 <= high_count <= 10:  # Only consider counts from 1 to 10
        if high_count in high_counts_distribution:
            high_counts_distribution[high_count] += 1
        else:
            high_counts_distribution[high_count] = 1

# Ensure all counts from 1 to 10 are represented in the dictionary
for i in range(1, 11):
    if i not in high_counts_distribution:
        high_counts_distribution[i] = 0

# Data for plotting
high_counts = sorted(high_counts_distribution.items())  # Sort items to plot in order
counts = [count for _, count in high_counts]
labels = [str(i) for i, _ in high_counts]

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color='blue')
plt.xlabel('Number of "High" in Series')
plt.ylabel('Number of Series')
plt.title('Distribution of Series Ending with Various Counts of "High"')
plt.xticks(labels)  # Ensure that all labels are shown
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



import pandas as pd

# Initialize counts for series ending with 0 to 10 "High"
high_end_counts = {i: 0 for i in range(1,11)}  # from 0 to 10

# Define function to count the consecutive 'High' at the end of a series
def count_high_at_end(series):
    count = 0
    # Ensure that the series is indeed a Series, not a DataFrame column
    for value in reversed(series.values):
        if value == 'High':
            count += 1
        else:
            break
    return count

# Calculate the counts
for _, group in train_df.groupby('ChassisId_encoded'):
    n_high = count_high_at_end(group['risk_level'])
    if 1 <= n_high <= 10:  # We only care about 1 to 10
        high_end_counts[n_high] += 1

# Determine the maximum count
M = max(high_end_counts.values())




# Prepare to add duplicated series
augmented_series = []

# Duplicate each series appropriately
for id, group in train_df.groupby('ChassisId_encoded'):
    n_high = count_high_at_end(group['risk_level'])
    if 1 <= n_high <= 10 and high_end_counts[n_high]!=523:
        num_duplicates = M // high_end_counts[n_high] # Subtract 1 because we already have one instance
        for _ in range(num_duplicates):
            new_group = group.copy()
            new_group['ChassisId_encoded'] = f"{id}_dup_{_}"  # Create a new unique ID for each duplicate
            augmented_series.append(new_group)

# Concatenate the original DataFrame with the augmented data
augmented_data = pd.concat(augmented_series, ignore_index=True)
train_df = pd.concat([train_df, augmented_data], ignore_index=True)


# Quick check of the new balance
new_counts = {i: 0 for i in range(1, 11)}
for _, group in train_df.groupby('ChassisId_encoded'):
    n_high = count_high_at_end(group['risk_level'])
    if 1 <= n_high <= 10:
        new_counts[n_high] += 1

for n in range(1, 11):
    print(f"Series ending with exactly {n} 'High': {new_counts[n]}")



# Initialize dictionary to count endings with 1 to 10 "High"
high_end_counts = {i: 0 for i in range(1, 11)}

# Group by 'ChassisId_encoded' and analyze the ending of each group
for id, group in train_df.groupby('ChassisId_encoded'):
    # Check for endings with 1 to 10 "High"
    for n in range(1, 11):
        if len(group) >= n and (group['risk_level'].iloc[-n:] == 'High').all() and (len(group) == n or group['risk_level'].iloc[-n-1] != 'High'):
            high_end_counts[n] += 1

# Output the results
for n in range(1, 11):
    print(f"Number of series ending with exactly {n} 'High': {high_end_counts[n]}")


# Dictionary to store the counts for 'Medium' and 'High' for each ChassisId_encoded
counts_dict = {}
chassis_ids_few_medium = []  # List to store IDs with fewer than 9 'Medium'
chassis_ids_many_high = []   # List to store IDs with more than 9 'High'

for index in ids:
    # Filter the series for the current ChassisId_encoded
    series_data = high_risk_series[high_risk_series['ChassisId_encoded'] == index]

    # Count 'Medium' and 'High'
    counts = series_data['risk_level'].value_counts()

    # Extract counts for 'Medium' and 'High', handling cases where they might not exist
    medium_count = counts.get('Medium', 0)
    high_count = counts.get('High', 0)

    # Store in dictionary
    counts_dict[index] = {'Medium': medium_count, 'High': high_count}

    # Check conditions and store IDs accordingly
    if medium_count < 4:
        chassis_ids_few_medium.append(index)
    if high_count > 9:
        chassis_ids_many_high.append(index)

# Print the results
print("Chassis IDs with fewer than 9 'Medium':", chassis_ids_few_medium)
print("Chassis IDs with more than 9 'High':", chassis_ids_many_high)

# Sum the 'High' counts for the IDs in chassis_ids_many_high
total_high_counts = sum(counts_dict[id]['High'] for id in chassis_ids_many_high)

# Print the total count of 'High'
print("Total 'High' counts for Chassis IDs with more than 9 'High':", total_high_counts)



# Load the test data
df_test = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/public_X_test.csv')

# Fill NaN values with 0
# df_test  = df_test.fillna(0)

# Identify columns with at least one NaN value
columns_with_nan = df_test.columns[df_test.isna().any()]
df_test_cleaned = df_test.drop(columns=columns_with_nan)

# Preprocess the test data
#X_test = df_test_cleaned.drop(columns=['Timesteps', 'ChassisId_encoded', 'gen'])

import pandas as pd
from scipy.stats import mannwhitneyu

# Assuming test_df is your dataframe and it includes a column 'gen' that denotes generation either 'gen1' or 'gen2'
test_gen1 = df_test_cleaned[df_test_cleaned['gen'] == 'gen1']
test_gen2 = df_test_cleaned[df_test_cleaned['gen'] == 'gen2']

significant_features = []

for feature in df_test_cleaned.columns.difference(['gen', 'ChassisId_encoded', 'Timesteps']):
    # Perform Mann-Whitney U Test assuming non-normal distributions
    stat, p_value = mannwhitneyu(test_gen1[feature], test_gen2[feature], alternative='two-sided')

    # You can choose a significance level
    if p_value < 0.01:
        significant_features.append(feature)
        print(f"Feature {feature} is significantly different with p-value: {p_value}")

print("Significant features based on statistical testing:", significant_features)

# Load the data
train_df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Volvo/train_gen1.csv')
print(f"Initial shape of train_df: {train_df.shape}")

# Fill NaN values with 0
# train_df = train_df.fillna(0)

# Identify columns with at least one NaN value
columns_with_nan = train_df.columns[train_df.isna().any()]
train_df_cleaned = train_df.drop(columns=columns_with_nan)

# Preprocess the data
# Assuming 'risk level' is the target and other columns are features
X_train = train_df_cleaned.drop(columns=['risk_level', 'Timesteps', 'ChassisId_encoded', 'gen'])

# First, let's check the counts of 'gen1' vs 'gen2' in the test set.
gen_counts = df_test['gen'].value_counts()

# Plotting the count of 'gen1' vs 'gen2'
plt.figure(figsize=(8, 6))
gen_counts.plot(kind='bar', color=['blue', 'green'])
plt.title('Count of gen1 vs gen2 in Test Set')
plt.xlabel('Gen Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()




# Now, prepare the data for the distribution analysis of a specific feature column, for example the first feature column.
feature_column = df_test_cleaned.columns[8]  # Assume the first column after dropping the non-feature columns

# Split the data into gen1 and gen2 based on the 'gen' column for the chosen feature
gen1_values = df_test_cleaned[df_test['gen'] == 'gen1'][feature_column]
gen2_values = df_test_cleaned[df_test['gen'] == 'gen2'][feature_column]
train_values = df_test_cleaned[feature_column]

weights_gen1 = np.ones_like(gen1_values) / len(gen1_values)
weights_gen2 = np.ones_like(gen2_values) / len(gen2_values)
weights_train = np.ones_like(train_values) / len(train_values)

# Create a figure with two horizontally placed subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))

axes[0].hist(train_values, bins=20, alpha=0.7, color='red', weights=weights_train)
axes[0].set_title(f'Distribution of {feature_column} for Train')
axes[0].set_xlabel('Feature Values')
axes[0].set_ylim([0,1])
axes[0].set_ylabel('Frequency')

# Plotting the distribution for gen1
axes[1].hist(gen1_values, bins=20, alpha=0.7, color='blue', weights=weights_gen1)
axes[1].set_title(f'Distribution of {feature_column} for Gen1')
axes[1].set_xlabel('Feature Values')
axes[1].set_ylim([0,1])
axes[1].set_ylabel('Frequency')

# Plotting the distribution for gen2
axes[2].hist(gen2_values, bins=20, alpha=0.7, color='green', weights=weights_gen2)
axes[2].set_title(f'Distribution of {feature_column} for Gen2')
axes[2].set_xlabel('Feature Values')
axes[2].set_ylim([0,1])
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Create normalized histograms
hist_train, bin_edges = np.histogram(train_values, bins=20, weights=weights_train)
hist_gen1, _ = np.histogram(gen1_values, bins=bin_edges, weights=weights_gen1)
hist_gen2, _ = np.histogram(gen2_values, bins=bin_edges, weights=weights_gen2)

from scipy.stats import wasserstein_distance
from scipy.special import kl_div

# Calculate the Earth Mover's Distance
emd_gen1 = wasserstein_distance(hist_train, hist_gen1)
emd_gen2 = wasserstein_distance(hist_train, hist_gen2)

# Calculate Kullback-Leibler Divergence (ensure no zero values)
kl_gen1 = np.sum(kl_div(hist_train + 1e-10, hist_gen1 + 1e-10))
kl_gen2 = np.sum(kl_div(hist_train + 1e-10, hist_gen2 + 1e-10))

print(f'EMD with Gen1: {emd_gen1}, EMD with Gen2: {emd_gen2}')
print(f'KL Divergence with Gen1: {kl_gen1}, KL Divergence with Gen2: {kl_gen2}')

from scipy.stats import wasserstein_distance
from scipy.special import kl_div


EMD_1 = []
EMD_2 = []
KL_1 = []
KL_2 = []

for i in range(297):
  feature_column = df_test_cleaned.columns[3+i]

  gen1_values = df_test_cleaned[df_test['gen'] == 'gen1'][feature_column]
  gen2_values = df_test_cleaned[df_test['gen'] == 'gen2'][feature_column]
  train_values = df_test_cleaned[feature_column]

  weights_gen1 = np.ones_like(gen1_values) / len(gen1_values)
  weights_gen2 = np.ones_like(gen2_values) / len(gen2_values)
  weights_train = np.ones_like(train_values) / len(train_values)

  hist_train, bin_edges = np.histogram(train_values, bins=20, weights=weights_train)
  hist_gen1, _ = np.histogram(gen1_values, bins=bin_edges, weights=weights_gen1)
  hist_gen2, _ = np.histogram(gen2_values, bins=bin_edges, weights=weights_gen2)

  emd_gen1 = wasserstein_distance(hist_train, hist_gen1)
  emd_gen2 = wasserstein_distance(hist_train, hist_gen2)

  kl_gen1 = np.sum(kl_div(hist_train + 1e-10, hist_gen1 + 1e-10))
  kl_gen2 = np.sum(kl_div(hist_train + 1e-10, hist_gen2 + 1e-10))

  EMD_1.append(emd_gen1)
  EMD_2.append(emd_gen2)
  KL_1.append(kl_gen1)
  KL_2.append(kl_gen2)

  print(f'Feature processed: {i+1}')


#plt.plot(EMD_1, color='red')
#plt.plot(EMD_2, color='blue')
#plt.legend(['EMD_1', 'EMD_2'])
#plt.show()

# Create a figure with two horizontally placed subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

axes[0].plot(EMD_1, color='red')
axes[0].set_title('EMD: Train vs gen1 Test')
axes[0].set_xlabel('Features')
axes[0].set_ylim([0,0.05])
axes[0].set_ylabel('EMD')

# Plotting the distribution for gen1
axes[1].plot(EMD_2, color='blue')
axes[1].set_title('EMD: Train vs gen2 Test')
axes[1].set_xlabel('Features')
axes[1].set_ylim([0,0.05])
axes[1].set_ylabel('EMD')

plt.tight_layout()
plt.show()

# Create a figure with two horizontally placed subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

axes[0].plot(KL_1, color='red')
axes[0].axhline(2, color='orange')
axes[0].axhline(0.5, color='yellow')
axes[0].set_title('KL: Train vs gen1 Test')
axes[0].set_xlabel('Features')
axes[0].set_ylim([0,10])
axes[0].set_ylabel('KL')

# Plotting the distribution for gen1
axes[1].plot(KL_2, color='blue')
axes[1].axhline(2, color='orange')
axes[1].axhline(0.5, color='yellow')
axes[1].set_title('KL: Train vs gen2 Test')
axes[1].set_xlabel('Features')
axes[1].set_ylim([0,10])
axes[1].set_ylabel('KL')

plt.tight_layout()
plt.show()







