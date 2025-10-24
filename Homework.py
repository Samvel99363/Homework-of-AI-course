import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.float_format', lambda x: '%.2f' % x)

data = pd.read_csv('tmdb_5000_movies.csv')
print("✅ Dataset loaded successfully!")
print("Shape:", data.shape)
data.head()

# --- Data Diagnosis ---
print("\n--- Basic Info ---")
data.info()

print("\n--- Missing Values ---")
print(data.isnull().sum())

print("\n--- Basic Statistics ---")
print(data[['budget', 'revenue', 'popularity', 'runtime']].describe())

# --- Data Cleaning ---
# Convert to numeric and handle zeros
data['budget'] = pd.to_numeric(data['budget'], errors='coerce')
data['revenue'] = pd.to_numeric(data['revenue'], errors='coerce')

# Replace zeros with NaN
data.loc[data['budget'] == 0, 'budget'] = np.nan
data.loc[data['revenue'] == 0, 'revenue'] = np.nan

# Drop rows with missing budget or revenue
df = data[['budget', 'revenue']].dropna()
print("\nRemaining rows after cleaning:", len(df))

# --- Optional: Apply log transformation ---
df['log_budget'] = np.log1p(df['budget'])
df['log_revenue'] = np.log1p(df['revenue'])

# --- Exploratory Data Analysis  ---
plt.figure(figsize=(6, 5))
plt.scatter(df['log_budget'], df['log_revenue'], alpha=0.5)
plt.title('Log(Budget) vs Log(Revenue)')
plt.xlabel('Log(Budget)')
plt.ylabel('Log(Revenue)')
plt.grid(True)
plt.show()

# --- Model Building ---
# Define features and target
X = df[['log_budget']]  # Feature
y = df['log_revenue']   # Target

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Initialize and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Display coefficients
print("\n--- Model Parameters ---")
print("Intercept:", model.intercept_)
print("Coefficient (log_budget):", model.coef_[0])

# --- Model Evaluation ---
# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"Mean Squared Error (log scale): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# --- Plot actual vs predicted ---
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Log(Revenue)")
plt.ylabel("Predicted Log(Revenue)")
plt.title("Actual vs Predicted Revenue (Log Scale)")
plt.grid(True)
plt.show()

# --- Inverse transform to original scale ---
df_result = pd.DataFrame({
    'Budget ($)': np.expm1(X_test['log_budget']),
    'Actual Revenue ($)': np.expm1(y_test),
    'Predicted Revenue ($)': np.expm1(y_pred)
})

print("\n--- Sample Predictions (Original Scale) ---")
print(df_result.head(10))