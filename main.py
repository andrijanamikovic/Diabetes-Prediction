import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv('data/diabetes_prediction_dataset.csv');

data.info()
