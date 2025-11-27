import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# --- LOAD AND PREPROCESS DATA ---

# Load the data
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')

# Define the target variable and features
target = 'default'
features = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']

# Separate features and target
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better performance with Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- TRAIN THE MODEL ---

# Initialize and train the Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- CREATE PREDICTIVE FUNCTION ---

def predict_loan_default(credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score):
    """
    Predicts the probability of default for a loan borrower.

    Args:
        credit_lines_outstanding (float): Number of credit lines outstanding.
        loan_amt_outstanding (float): Amount of the loan outstanding.
        total_debt_outstanding (float): Total debt outstanding.
        income (float): Borrower's annual income.
        years_employed (int): Number of years employed.
        fico_score (int): Borrower's FICO credit score.

    Returns:
        dict: A dictionary containing the predicted probability of default (PD),
              the expected loss (EL), and the predicted class (0 or 1).
    """
    # Create a DataFrame for the single prediction
    input_data = pd.DataFrame({
        'credit_lines_outstanding': [credit_lines_outstanding],
        'loan_amt_outstanding': [loan_amt_outstanding],
        'total_debt_outstanding': [total_debt_outstanding],
        'income': [income],
        'years_employed': [years_employed],
        'fico_score': [fico_score]
    })

    # Scale the input data using the same scaler
    input_data_scaled = scaler.transform(input_data)

    # Predict the probability of default (class 1)
    pd_prob = model.predict_proba(input_data_scaled)[0][1]  # Probability of class 1 (default)

    # Calculate Expected Loss (EL)
    # EL = PD * LGD * EAD
    # LGD = 1 - Recovery Rate = 1 - 0.10 = 0.9
    # EAD = loan_amt_outstanding
    lgd = 0.9
    ead = loan_amt_outstanding
    expected_loss = pd_prob * lgd * ead

    # Predict the class (0 or 1)
    predicted_class = model.predict(input_data_scaled)[0]

    return {
        "probability_of_default": round(pd_prob, 4),
        "expected_loss": round(expected_loss, 2),
        "predicted_default": int(predicted_class)
    }

# --- EXAMPLE USAGE ---

# Example: Predict for a borrower with specific characteristics
example_prediction = predict_loan_default(
    credit_lines_outstanding=5,
    loan_amt_outstanding=10000,
    total_debt_outstanding=15000,
    income=60000,
    years_employed=5,
    fico_score=650
)

print("\n--- Example Prediction ---")
for key, value in example_prediction.items():
    print(f"{key}: {value}")