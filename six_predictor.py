# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pymc as pm
import numpy as np

# Load the dataset
data = pd.read_csv("C:\\Users\\hp\\Desktop\\SixPredictorProject\\ipl_2022_deliveries.csv.zip")

# Preprocess the data
data['is_six'] = (data['runs_of_bat'] == 6).astype(int)  # Create is_six column
data_encoded = pd.get_dummies(data, columns=['striker', 'bowler'], drop_first=True)  # One-hot encoding

# Select features and target
X = data_encoded.drop(columns=['is_six', 'match_id', 'season', 'match_no', 'date', 'venue', 
                               'batting_team', 'bowling_team', 'innings', 'player_dismissed', 
                               'wicket_type', 'fielder'])
y = data_encoded['is_six']
X = X.select_dtypes(include=np.number)  # Keep numeric features only

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to numpy arrays
X_train_np = X_train.values
y_train_np = y_train.values

# Create the Bayesian Logistic Regression model
with pm.Model() as logistic_model:
    # Priors for weights and intercept
    betas = pm.Normal('betas', mu=0, sigma=10, shape=X_train_np.shape[1])
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    
    # Logistic regression equation
    logits = intercept + pm.math.dot(X_train_np, betas)
    p = pm.math.sigmoid(logits)
    
    # Likelihood
    y_obs = pm.Bernoulli('y_obs', p=p, observed=y_train_np)
    
    # Sampling from posterior
    trace = pm.sample(1000, tune=500, cores=2, target_accept=0.9)

# Prediction and Evaluation
with logistic_model:
    pm.set_data({"X": X_test.values})
    posterior_predictive = pm.sample_posterior_predictive(trace)
    y_pred = posterior_predictive['y_obs'].mean(axis=0).round()

# Print evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the results to a file
with open('evaluation_results.txt', 'w') as f:
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred))

# Create a Streamlit App for Prediction
streamlit_code = '''
import streamlit as st
import numpy as np

st.title("T20 Cricket Six Predictor")
runs = st.slider("Runs scored by the batter in this over so far:", 0, 36)
overs = st.slider("Current over number:", 1, 20)

if st.button("Predict"):
    # Use your trained model to predict (dummy example)
    pred_prob = np.random.rand()  # Replace with actual model prediction
    if pred_prob > 0.5:
        st.write("Prediction: Batter is likely to hit a six!")
    else:
        st.write("Prediction: Batter is unlikely to hit a six!")
'''

# Save Streamlit app code
with open('six_predictor.py', 'w') as f:
    f.write(streamlit_code)
