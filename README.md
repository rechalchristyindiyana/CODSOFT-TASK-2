# CODSOFT-TASK-2
TASK-2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
# Load the dataset with the appropriate encoding
data = pd.read_csv('/content/IMDb Movies India.csv', encoding='latin1')
data
# Update the column names as needed
data.rename(columns={'Director': 'director', 'Actor 1': 'actors', 'Genre': 'genre', 'Rating': 'rating'}, inplace=True)
# Data Preprocessing
# Handle missing values in the 'rating' column
data.dropna(subset=['rating'], inplace=True)
# Convert 'rating' column to a numerical data type (e.g., float)
data['rating'] = data['rating'].astype(float)
data['rating']
# Encode categorical features into numerical values
label_encoder = LabelEncoder()
data['director'] = label_encoder.fit_transform(data['director'])
data['director']
data['actors'] = label_encoder.fit_transform(data['actors'])
data['actors']
data['genre'] = label_encoder.fit_transform(data['genre'])
data['genre']
# Filter out rows with large or invalid 'rating' values (e.g., consider a reasonable range)
data = data[(data['rating'] >= 0) & (data['rating'] <= 10)]
data

# Split the data into features (X) and target (y)
X = data[['director', 'actors', 'genre']]
X
y = data['rating']
y
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Build and train the regression model (XGBoost in this case)
model = XGBRegressor()
model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred
# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
