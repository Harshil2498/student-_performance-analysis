import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load and preprocess data
data = pd.read_csv('stud.csv')

# Encode categorical variables
le_gender = LabelEncoder()
le_lunch = LabelEncoder()
le_test_prep = LabelEncoder()
data['gender'] = le_gender.fit_transform(data['gender'])
data['lunch'] = le_lunch.fit_transform(data['lunch'])
data['test_preparation_course'] = le_test_prep.fit_transform(data['test_preparation_course'])
data = pd.get_dummies(data, columns=['race_ethnicity', 'parental_level_of_education'], drop_first=True)

# Features and target
X = data.drop(['math_score', 'reading_score', 'writing_score'], axis=1)
y = data['math_score']

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Save model and encoders
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_lunch, 'le_lunch.pkl')
joblib.dump(le_test_prep, 'le_test_prep.pkl')