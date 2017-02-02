import pandas as pd
import ifmm
import numpy as np
df = pd.read_csv('user_000007.csv')
record = df.item_index[range(0,df.item_index.size-1)]

x_list = np.sort(np.unique(record))
N = 2
fileNames = glob.glob('*.csv')