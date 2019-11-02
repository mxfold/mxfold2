import pandas as pd
import sys

df = pd.read_csv(sys.argv[1], names=("name", "length", "elapsed_time", "score", "TP", "TN", "FP", "FN", "SEN", "PPV", "F", "MCC"))
print(df[["length", "elapsed_time", "SEN", "PPV", "F"]].describe())
