import numpy as np #mathmatical calculation & array
import pandas as pd #for handle to dataset
import matplotlib.pyplot as plt  #for graphs

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from linear_regression import LinearRegressionScratch

housing = fetch_california_housing() #for loading datset

df=pd.DataFrame(
    housing.data,  #dataset input feature
    columns=housing.feature_names  #feature columns name
)
df["HouseValue"]=housing.target  #target value for pridiction
print(df.head())   #head means first 5 row print

print("\nDataset Shape:")
print(df.shape)   # find the rows & column

print("\nDataset Information:")
df.info()   #find the column name , data types , non null values

print("\nStatistical Summary:")
print(df.describe())   # count ,means, SD, min, max, quartiles 

print("\nMissing Values:")
print(df.isnull().sum())   # find missing values every columns  if =0 then dataset clean 

X = df.drop("HouseValue", axis=1)
y = df["HouseValue"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Features Shape:")
print(X_train.shape)     #model training

print("\nTesting Features Shape:")
print(X_test.shape)      #model testing

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)

X_train = X_train.to_numpy()  # convert pandas to numpy
X_test = X_test.to_numpy()

y_train = y_train.to_numpy()  #target to numpy arrays
y_test = y_test.to_numpy()

mean = np.mean(X_train, axis=0)  #calculate mean & SD from training data
std = np.std(X_train, axis=0)

X_train_scaled = (X_train - mean) / std  # SD Ttraining & testing data
X_test_scaled = (X_test - mean) / std


print("\nFeature Standardization Complete!")

print("\nFirst 3 rows of Scaled Training Data:")
print(X_train_scaled[:3])

model = LinearRegressionScratch(  # create the scratch model
    learning_rate=0.01,
    iterations=1000
)
model.fit(X_train_scaled, y_train)    #train model

print("\nModel Training Complete!")

print("\nLearned Theta:")  #display learnd parameters
print(model.theta)

print("\nFinal Training Cost:")  # display final cost
print(model.cost_history[-1])

y_pred = model.predict(X_test_scaled)  #pridiction on test data

print("\nFirst 10 Predictions:")
print(y_pred[:10])

print("\nFirst 10 Actual Values:")
print(y_test[:10])

errors = y_pred - y_test  # calculate test MSE (prediction-actual)

test_mse = np.mean(errors ** 2)  # error square

print("\nTest MSE:")  # error avg=test MSE
print(test_mse)

# Calculate R² Score manually

ss_res = np.sum((y_test - y_pred) ** 2)

ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

r2_score = 1 - (ss_res / ss_tot)

print("\nTest R² Score:")
print(r2_score)

# Plot Cost Function
plt.figure(figsize=(8, 5))

plt.plot(model.cost_history)

plt.xlabel("Iterations")
plt.ylabel("MSE Cost")
plt.title("Cost Function During Gradient Descent")

plt.grid(True)
plt.show()

from sklearn.linear_model import LinearRegression

# Create sklearn Linear Regression model
sklearn_model = LinearRegression()

# Train sklearn model
sklearn_model.fit(X_train_scaled, y_train)

# Make predictions
sklearn_pred = sklearn_model.predict(X_test_scaled)

# Calculate sklearn MSE
sklearn_errors = sklearn_pred - y_test
sklearn_mse = np.mean(sklearn_errors ** 2)

# Calculate sklearn R²
sklearn_ss_res = np.sum((y_test - sklearn_pred) ** 2)
sklearn_ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

sklearn_r2 = 1 - (sklearn_ss_res / sklearn_ss_tot)

print("\n===== MODEL COMPARISON =====")

print("\nScratch Linear Regression:")
print("Test MSE:", test_mse)
print("Test R²:", r2_score)

print("\nScikit-learn Linear Regression:")
print("Test MSE:", sklearn_mse)
print("Test R²:", sklearn_r2)

X_single_train = X_train[:, 0].reshape(-1, 1)
X_single_test = X_test[:, 0].reshape(-1, 1)

# Calculate mean and standard deviation using training data
single_mean = np.mean(X_single_train, axis=0)
single_std = np.std(X_single_train, axis=0)

# Standardize single feature
X_single_train_scaled = (X_single_train - single_mean) / single_std
X_single_test_scaled = (X_single_test - single_mean) / single_std

# Create single-feature scratch model
single_model = LinearRegressionScratch(
    learning_rate=0.01,
    iterations=1000
)

# Train
single_model.fit(X_single_train_scaled, y_train)

# Predict
single_pred = single_model.predict(X_single_test_scaled)

# Calculate MSE
single_errors = single_pred - y_test
single_mse = np.mean(single_errors ** 2)

# Calculate R²
single_ss_res = np.sum((y_test - single_pred) ** 2)
single_ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

single_r2 = 1 - (single_ss_res / single_ss_tot)

print("\n===== SINGLE-FEATURE MODEL =====")
print("Feature Used: MedInc")

print("\nLearned Theta:")
print(single_model.theta)

print("\nTest MSE:")
print(single_mse)

print("\nTest R²:")
print(single_r2)