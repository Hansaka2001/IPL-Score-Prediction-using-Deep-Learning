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
print(ipl.head())
