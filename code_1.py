
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from IPython.display import display, clear_output
import ipywidgets as widgets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import keras
import tensorflow as tf

ipl = pd.read_csv('ipl_data.csv')
ipl.head()

# Dropping certain features
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5',
              'wickets_last_5', 'mid', 'striker', 'non-striker'], axis=1)

X = df.drop(['total'], axis=1)
y = df['total']

# Label Encoding


# Create a LabelEncoder object for each categorical feature
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

# Fit and transform the categorical features with label encoding
X['venue'] = venue_encoder.fit_transform(X['venue'])
X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
X['batsman'] = striker_encoder.fit_transform(X['batsman'])
X['bowler'] = bowler_encoder.fit_transform(X['bowler'])


# Train test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


scaler = MinMaxScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define the neural network model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),  # Input layer
    # Hidden layer with 512 units and ReLU activation
    keras.layers.Dense(512, activation='relu'),
    # Hidden layer with 216 units and ReLU activation
    keras.layers.Dense(216, activation='relu'),
    # Output layer with linear activation for regression
    keras.layers.Dense(1, activation='linear')
])

# Compile the model with Huber loss
# You can adjust the 'delta' parameter as needed
huber_loss = tf.keras.losses.Huber(delta=1.0)
# Use Huber loss for regression
model.compile(optimizer='adam', loss=huber_loss)


# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=64,
          validation_data=(X_test_scaled, y_test))

model_losses = pd.DataFrame(model.history.history)
model_losses.plot()

# Make predictions
predictions = model.predict(X_test_scaled)

mean_absolute_error(y_test, predictions)


warnings.filterwarnings("ignore")

venue = widgets.Dropdown(
    options=df['venue'].unique().tolist(), description='Select Venue:')
batting_team = widgets.Dropdown(
    options=df['bat_team'].unique().tolist(),  description='Select Batting Team:')
bowling_team = widgets.Dropdown(
    options=df['bowl_team'].unique().tolist(),  description='Select Batting Team:')
striker = widgets.Dropdown(
    options=df['batsman'].unique().tolist(), description='Select Striker:')
bowler = widgets.Dropdown(
    options=df['bowler'].unique().tolist(), description='Select Bowler:')

predict_button = widgets.Button(description="Predict Score")

output = widgets.Output()


def predict_score(b):
    with output:
        clear_output()  # Clear the previous output

        # Decode the encoded values back to their original values
        decoded_venue = venue_encoder.transform([venue.value])
        decoded_batting_team = batting_team_encoder.transform(
            [batting_team.value])
        decoded_bowling_team = bowling_team_encoder.transform(
            [bowling_team.value])
        decoded_striker = striker_encoder.transform([striker.value])
        decoded_bowler = bowler_encoder.transform([bowler.value])

        input = np.array([decoded_venue,  decoded_batting_team,
                         decoded_bowling_team, decoded_striker, decoded_bowler])
        input = input.reshape(1, 5)
        input = scaler.transform(input)
        # print(input)
        predicted_score = model.predict(input)
        predicted_score = int(predicted_score[0, 0])

        print(predicted_score)
