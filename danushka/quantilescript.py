import pandas as pd
import numpy as np
'''
calculate the quantiles from given csv column
'''
CSV_FILE_PATH = "training.csv"
COLUMN_NAME = "arousal"
df = pd.read_csv(CSV_FILE_PATH)
valence_column = df[COLUMN_NAME]
arr = list(valence_column)
print(np.quantile(arr, .33)) 
print(np.quantile(arr, .66))
