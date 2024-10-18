import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

"""# **1. Datasets Uploading and Merging**"""

# File paths
dataset1_path = 'dataset1.csv'
dataset2_path = 'dataset2.csv'
dataset3_path = 'dataset3.csv'

# 1. Loading dataset1.csv
try:
    df1 = pd.read_csv(dataset1_path)
    print(f"dataset1.csv loaded successfully with {df1.shape[0]} records and {df1.shape[1]} columns.")
except FileNotFoundError:
    print(f"Error: {dataset1_path} not found. Please check the file path.")

# 2. Loading dataset2.csv
try:
    df2 = pd.read_csv(dataset2_path)
    print(f"dataset2.csv loaded successfully with {df2.shape[0]} records and {df2.shape[1]} columns.")
except FileNotFoundError:
    print(f"Error: {dataset2_path} not found. Please check the file path.")

# 3. Loading dataset3.csv
try:
    df3 = pd.read_csv(dataset3_path)
    print(f"dataset3.csv loaded successfully with {df3.shape[0]} records and {df3.shape[1]} columns.")
except FileNotFoundError:
    print(f"Error: {dataset3_path} not found. Please check the file path.")

# 4. Merging dataset1 and dataset2 on 'ID'
df_merged_1_2 = pd.merge(df1, df2, on='ID', how='inner')
print(f"Merged dataset1 and dataset2: {df_merged_1_2.shape[0]} records and {df_merged_1_2.shape[1]} columns.")

# 5. Merging the above result with dataset3 on 'ID'
df_final_merged = pd.merge(df_merged_1_2, df3, on='ID', how='inner')
print(f"Final merged dataset: {df_final_merged.shape[0]} records and {df_final_merged.shape[1]} columns.")

# 6. Saving the merged dataset to a new CSV file (optional)
output_path = 'merged_dataset.csv'
df_final_merged.to_csv(output_path, index=False)
print(f"Merged dataset saved to {output_path}.")

# 7. the first few rows of the merged dataset
print("\nFirst 5 records of the merged dataset:")
print(df_final_merged.head())

# Set visual aesthetics
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. Load the Merged Dataset
merged_dataset_path = 'merged_dataset.csv'

try:
    df = pd.read_csv(merged_dataset_path)
    print(f"Merged dataset loaded successfully with {df.shape[0]} records and {df.shape[1]} columns.")
except FileNotFoundError:
    print(f"Error: {merged_dataset_path} not found. Please check the file path.")

# Display the first few rows
print("\nFirst 5 records of the merged dataset:")
print(df.head())

"""# **2. Data Cleaning and Preprocessing**"""

## 2.1 Handling Missing Values
print("\nMissing values per column:")
print(df.isnull().sum())

# Option 1: Remove rows with missing values
df = df.dropna()

# For simplicity, rows with missing well-being indicators are dropped
well_being_cols = [
    'Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr',
    'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind',
    'Loved', 'Intthg', 'Cheer'
]

# Checking how many rows have missing well-being indicators
missing_wb = df[well_being_cols].isnull().sum().sum()
print(f"\nTotal missing values in well-being indicators: {missing_wb}")

# Dropping rows with any missing well-being indicators
df_clean = df.dropna(subset=well_being_cols)
print(f"After dropping missing well-being indicators: {df_clean.shape[0]} records.")

## 2.2 Data Type Checking
print("\nData types before conversion:")
print(df_clean.dtypes)

# screen time variables are numeric
screen_time_cols = [
    'C_we', 'C_wk', 'G_we', 'G_wk',
    'S_we', 'S_wk', 'T_we', 'T_wk'
]

# Converting to numeric if not already
for col in screen_time_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# conversion
print("\nData types after conversion:")
print(df_clean.dtypes)

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)], lower_bound, upper_bound

# Example: Detect outliers in total smartphone usage
for col in screen_time_cols:
    outliers, lower_bound, upper_bound = detect_outliers_iqr(df_clean, col)
    print(f"\nNumber of outliers in {col}: {outliers.shape[0]}")

# Removing outliers
df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

"""# **3. Feature Engineering**"""

## 3.1 Calculating Total and Average Screen Time

# Define a function to calculate average daily screen time
def calculate_average_screen_time(we_hours, wk_hours):
    # Assuming weekend has 2 days and weekdays have 5 days
    return (we_hours * 2 + wk_hours * 5) / 7

# Calculate average daily screen time for each device
df_clean['Avg_Computers'] = calculate_average_screen_time(df_clean['C_we'], df_clean['C_wk'])
df_clean['Avg_Video_Games'] = calculate_average_screen_time(df_clean['G_we'], df_clean['G_wk'])
df_clean['Avg_Smartphones'] = calculate_average_screen_time(df_clean['S_we'], df_clean['S_wk'])
df_clean['Avg_TV'] = calculate_average_screen_time(df_clean['T_we'], df_clean['T_wk'])

# Calculate total screen time per day
df_clean['Total_Screen_Time'] = df_clean['Avg_Computers'] + df_clean['Avg_Video_Games'] + df_clean['Avg_Smartphones'] + df_clean['Avg_TV']

print("\nSample of newly engineered features:")
print(df_clean[['Avg_Computers', 'Avg_Video_Games', 'Avg_Smartphones', 'Avg_TV', 'Total_Screen_Time']].head())

## 3.2 Creating Screen Time Categories
# Categorize total screen time into Low, Medium, High based on quartiles
df_clean['Screen_Time_Category'] = pd.qcut(df_clean['Total_Screen_Time'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

print("\nScreen Time Category distribution:")
print(df_clean['Screen_Time_Category'].value_counts())

"""# **4. Exploratory Data Analysis (EDA)**"""

## Descriptive Statistics
print("\nDescriptive statistics for screen time:")
print(df_clean[screen_time_cols + ['Total_Screen_Time']].describe())

print("\nDescriptive statistics for well-being indicators:")
print(df_clean[well_being_cols].describe())

### Distribution of Total Screen Time
plt.figure(figsize=(10,6))
sns.histplot(df_clean['Total_Screen_Time'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Total Daily Screen Time')
plt.xlabel('Total Screen Time (hours)')
plt.ylabel('Number of Respondents')
plt.tight_layout()
plt.show()

### Box Plot: Well-Being Score vs. Screen Time Category
plt.figure(figsize=(12,8))
sns.boxplot(x='Screen_Time_Category', y='Goodme', data=df_clean, palette='Set3')
plt.title('Goodme Score Across Screen Time Categories')
plt.xlabel('Screen Time Category')
plt.ylabel('Goodme Score')
plt.tight_layout()
plt.show()

### Correlation Heatmap
# Compute correlation matrix for screen time and well-being indicators
corr_cols = screen_time_cols + ['Total_Screen_Time'] + well_being_cols
corr_matrix = df_clean[corr_cols].corr()

plt.figure(figsize=(16,12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap: Screen Time and Well-Being Indicators')
plt.tight_layout()
plt.show()

### Scatter Plot: Total Screen Time vs. Goodme Score
plt.figure(figsize=(10,6))
sns.scatterplot(x='Total_Screen_Time', y='Goodme', data=df_clean, alpha=0.3, color='purple')
sns.regplot(x='Total_Screen_Time', y='Goodme', data=df_clean, scatter=False, color='red')
plt.title('Total Screen Time vs. Goodme Score')
plt.xlabel('Total Screen Time (hours)')
plt.ylabel('Goodme Score')
plt.tight_layout()
plt.show()

### Screen Time by Gender
plt.figure(figsize=(10,6))
sns.boxplot(x='gender', y='Total_Screen_Time', data=df_clean, palette='Set2')
plt.title('Total Screen Time by Gender')
plt.xlabel('Gender (1=Male, 0=Other)')
plt.ylabel('Total Screen Time (hours)')
plt.tight_layout()
plt.show()

# Set a style for the plots
sns.set(style="whitegrid")

# Plot 1: Screen Time Category vs Minority
plt.figure(figsize=(10, 6))
sns.countplot(data=df_clean, x='Screen_Time_Category', hue='minority', palette='Set2')
plt.title('Screen Time Category by Minority Status')
plt.xlabel('Screen Time Category')
plt.ylabel('Count')
plt.legend(title='Minority Status', loc='upper right')
plt.tight_layout()
plt.show()

# Plot 2: Screen Time Category vs Gender
plt.figure(figsize=(10, 6))
sns.countplot(data=df_clean, x='Screen_Time_Category', hue='gender', palette='Set1')
plt.title('Screen Time Category by Gender')
plt.xlabel('Screen Time Category')
plt.ylabel('Count')
plt.legend(title='Gender', loc='upper right')
plt.tight_layout()
plt.show()

# Calculate mean and standard error
screen_time_mean = df_clean.groupby('minority')['Total_Screen_Time'].mean().reset_index()
screen_time_se = df_clean.groupby('minority')['Total_Screen_Time'].sem().reset_index()

# Merge mean and SE
screen_time_stats = pd.merge(screen_time_mean, screen_time_se, on='minority', suffixes=('_mean', '_se'))

# Bar Plot with Error Bars
plt.figure(figsize=(10,6))
sns.barplot(x='minority', y='Total_Screen_Time_mean', data=screen_time_stats, palette='Set1', ci=None)
plt.errorbar(x=range(len(screen_time_stats)),
             y=screen_time_stats['Total_Screen_Time_mean'],
             yerr=screen_time_stats['Total_Screen_Time_se'],
             fmt='none', c='black', capsize=5)
plt.title('Average Total Screen Time by Minority Status')
plt.xlabel('Minority Status (1=Minority, 0=Majority)')
plt.ylabel('Average Total Screen Time (hours)')
plt.tight_layout()
plt.show()

# Calculate mean and standard error
screen_time_mean_deprived = df_clean.groupby('deprived')['Total_Screen_Time'].mean().reset_index()
screen_time_se_deprived = df_clean.groupby('deprived')['Total_Screen_Time'].sem().reset_index()

# Merge mean and SE
screen_time_stats_deprived = pd.merge(screen_time_mean_deprived, screen_time_se_deprived, on='deprived', suffixes=('_mean', '_se'))

# Bar Plot with Error Bars
plt.figure(figsize=(10,6))
sns.barplot(x='deprived', y='Total_Screen_Time_mean', data=screen_time_stats_deprived, palette='Set3', ci=None)
plt.errorbar(x=range(len(screen_time_stats_deprived)),
             y=screen_time_stats_deprived['Total_Screen_Time_mean'],
             yerr=screen_time_stats_deprived['Total_Screen_Time_se'],
             fmt='none', c='black', capsize=5)
plt.title('Average Total Screen Time by Deprivation Status')
plt.xlabel('Deprived (1=Yes, 0=No)')
plt.ylabel('Average Total Screen Time (hours)')
plt.tight_layout()
plt.show()

"""# **5. Modeling: Linear Regression**"""

## 5.1 Selecting the Dependent and Independent Variables
dependent_var = 'Goodme'
independent_vars = ['', 'gender', 'minority', 'deprived']

independent_vars = ['Total_Screen_Time','Avg_Computers','Avg_Video_Games','Avg_Smartphones','Avg_TV','gender','minority','deprived']

X = df_clean[independent_vars]
y = df_clean[dependent_var]

## 5.2 Splitting the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} records")
print(f"Testing set size: {X_test.shape[0]} records")

## 5.3 Building the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

## 5.4 Making Predictions
y_pred = model.predict(X_test)

## 5.5 Evaluating the Model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nLinear Regression Model Evaluation:")
print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

## 5.7 Residual Analysis
residuals = y_test - y_pred

plt.figure(figsize=(10,6))
sns.histplot(residuals, bins=30, kde=True, color='teal')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.3, color='orange')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.tight_layout()
plt.show()

