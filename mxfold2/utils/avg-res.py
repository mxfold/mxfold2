import pandas as pd
import sys

columns = ['name', 'length', 'elapsed_time', 'score', 'TP', 'TN', 'FP', 'FN',
           'SEN', 'PPV', 'F', 'MCC']

# Accept both the historical headered format and files produced by the
# upstream writer before the compatibility fix.  In the latter case, using
# pandas' default header would discard the first sequence.
with open(sys.argv[1]) as f:
    first_line = f.readline().strip()
headered = first_line.split(',')[0].strip().lower() == 'name'

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

df = pd.read_csv(sys.argv[1], header=0 if headered else None,
                 names=None if headered else columns)

# すべての数値列に対して統計量を出す
print(df.describe())
