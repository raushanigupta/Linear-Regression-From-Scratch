# Linear Regression From Scratch Using NumPy

A complete implementation of **Linear Regression from scratch using NumPy**, without using `sklearn.LinearRegression` for the actual model implementation.

The project demonstrates how Linear Regression works internally using mathematical equations, matrix operations, Mean Squared Error (MSE), and Gradient Descent.

---

## 📌 Project Overview

This project implements a Linear Regression model from scratch using NumPy.

The model supports:

* Single-feature Linear Regression
* Multi-feature Linear Regression
* Hypothesis calculation
* Mean Squared Error (MSE)
* Gradient Descent optimization
* Feature standardization
* Train/Test split
* Manual MSE calculation
* Manual R² score calculation
* Cost function visualization
* Comparison with Scikit-learn Linear Regression

---

## 📊 Dataset

The project uses the **California Housing Dataset**.

The dataset contains information about California districts and their housing values.

### Dataset Size

* Total samples: **20,640**
* Features: **8**
* Target: **HouseValue**

### Features

| Feature    | Description                 |
| ---------- | --------------------------- |
| MedInc     | Median income               |
| HouseAge   | Median house age            |
| AveRooms   | Average number of rooms     |
| AveBedrms  | Average number of bedrooms  |
| Population | Population of the district  |
| AveOccup   | Average number of occupants |
| Latitude   | Latitude                    |
| Longitude  | Longitude                   |

### Target

`HouseValue` represents the median house value.

---

# 🧮 Mathematical Explanation

## 1. Hypothesis Function

Linear Regression predicts the output using:

**ŷ = Xθ**

Where:

* `ŷ` = predicted value
* `X` = input feature matrix
* `θ` = model parameters

For multiple features:

**ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ**

The implementation uses NumPy matrix multiplication:

```python
np.dot(X, self.theta)
```

---

## 2. Intercept

An intercept column containing ones is added to the feature matrix:

```python
X = np.c_[np.ones((X.shape[0], 1)), X]
```

For example, with 8 features:

```text
8 features + 1 intercept = 9 parameters
```

---

## 3. Mean Squared Error

The model uses Mean Squared Error to measure prediction error.

### Formula

**MSE = (1/n) Σ(y - ŷ)²**

Where:

* `y` = actual values
* `ŷ` = predicted values
* `n` = number of observations

Implementation:

```python
errors = predictions - y
cost = np.mean(errors ** 2)
```

---

## 4. Gradient Descent

Gradient Descent is used to minimize the MSE cost.

The gradient is calculated using matrix operations:

**Gradient = (2/n) Xᵀ(Xθ - y)**

Implementation:

```python
gradients = (2 / n) * np.dot(X.T, errors)
```

---

## 5. Parameter Update

The parameters are updated using:

**θ = θ - α × Gradient**

Where:

* `θ` = model parameters
* `α` = learning rate
* `Gradient` = calculated gradient

Implementation:

```python
self.theta = self.theta - self.learning_rate * gradients
```

---

# ⚙️ Data Preprocessing

The dataset is divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Split

* Training samples: **16,512**
* Testing samples: **4,128**

Feature standardization is performed manually using the training data mean and standard deviation.

```python
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std
```

The same training mean and standard deviation are used for the test data to avoid data leakage.

---

# 🚀 Model Training

The scratch model is initialized with:

```python
model = LinearRegressionScratch(
    learning_rate=0.01,
    iterations=1000
)
```

The model is then trained using:

```python
model.fit(X_train_scaled, y_train)
```

---

# 📉 Cost Function Visualization

The project stores the cost after every iteration:

```python
self.cost_history.append(cost)
```

This allows us to visualize how the MSE decreases during Gradient Descent.

The graph demonstrates whether the model is converging during training.

---

# 📈 Model Evaluation

The trained model is evaluated on unseen test data.

## Test MSE

```text
0.55458
```

## Test R² Score

```text
0.57679
```

The R² score indicates that the model explains approximately **57.7% of the variation** in the target values on the test dataset.

---

# 🔬 Scratch Model vs Scikit-learn

The implementation is compared with Scikit-learn's Linear Regression model as a final sanity check.

| Model                          | Test MSE | Test R² |
| ------------------------------ | -------: | ------: |
| Scratch Linear Regression      |  0.55458 | 0.57679 |
| Scikit-learn Linear Regression |  0.55589 | 0.57579 |

The results are very close, demonstrating that the NumPy implementation is working correctly.

---

# 🔹 Single-Feature Regression

The implementation was also tested using only one feature:

```text
Feature: MedInc
```

Results:

```text
Test MSE: 0.70912
Test R²: 0.45886
```

This demonstrates that the same implementation can handle both **single-feature and multi-feature Linear Regression**.

---

# 📁 Project Structure

```text
Linear-Regression-From-Scratch/
│
├── data/
│
├── venv/
│
├── linear_regression.py
├── main.py
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

### Scikit-learn Usage

Scikit-learn is used only for:

* `train_test_split`
* California Housing dataset loading
* Final Linear Regression comparison

The actual Linear Regression implementation is written from scratch using NumPy.

---

# ▶️ How to Run

## 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

## 2. Navigate to the project

```bash
cd Linear-Regression-From-Scratch
```

## 3. Create a virtual environment

```bash
python -m venv venv
```

## 4. Activate the virtual environment

### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

## 5. Install dependencies

```bash
pip install -r requirements.txt
```

## 6. Run the project

```bash
python main.py
```

---

# 🎯 Key Learning Outcomes

Through this project, the following concepts were implemented and understood:

* How Linear Regression works internally
* Matrix multiplication using NumPy
* Hypothesis function
* MSE cost function
* Gradient Descent
* Parameter optimization
* Intercept handling
* Feature standardization
* Single-feature regression
* Multi-feature regression
* Manual MSE calculation
* Manual R² calculation
* Model convergence visualization
* Comparison with a standard ML library

---

# 👩‍💻 Author

**Raushani Gupta**

B.Tech — Machine Learning

GitHub: `https://github.com/raushanigupta`
