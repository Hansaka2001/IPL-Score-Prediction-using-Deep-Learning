from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Loading the dataset

ipl = pd.read_csv('ipl_data.csv')
ipl.head()

# Data Preprocessing

# Droping unimportant features
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5',
              'wickets_last_5', 'mid', 'striker', 'non-striker'], aixs=1)

# furter preprocessing
x = df.drop(['total'], axis=1)
y = df['total']

# label encoding


# create a labelEncoder object for each categorial feature
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()

# fit and transform the categorial features with label encoding
X['venue'] = venue_encoder.fit_transform(X['venue'])
X['batting_team'] = batting_team_encoder.fit_transform(X['batting_team'])
X['bowling_team'] = bowling_team_encoder.fit_transform(X['bowling_team'])
X['batsman'] = striker_encoder.fit_transform(X['batsman'])
X['bowler'] = bowler_encoder.fit_transform(X['bowler'])

# Train test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# feature scaling

scaler = MinMaxScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(216, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

huber_loss = tf.keras.losses.Huber(dekta=1.0)
model.compile(optimizer='adam', loss=huber_loss)
