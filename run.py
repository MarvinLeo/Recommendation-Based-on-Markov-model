import pandas as pd
import ifmm
import numpy as np
import glob
df = pd.read_csv('user_000007.csv')
record = df.item_index[range(0,df.item_index.size-1)]

x_list = np.sort(np.unique(record))
N = 2
fileNames = glob.glob('*.csv')

for files in fileNames:
    df = pd.read_csv(files)
    if df.shape[0] < 1200:
        total -= 1
        continue
    #record = df.item_index[range(df.item_index.size-1100,df.item_index.size-99)]
    record = df.label6[range(df.item_index.size-1100,df.item_index.size-99)]
    record = record.as_matrix()
    x_list = np.unique(record)
    aim = record[-1]
    record = record[:-1]

    prob = predict(theta1, x_list, record)
    rank = np.argsort(prob)[::-1]
    top_N = np.argsort(prob)[::-1][:N]
    aim_index = np.where(x_list == aim)[0][0]
    rank_n = np.where(rank == aim_index)[0][0]
    rank_list[index] = rank_n + 1
    if aim_index in top_N:
        print "yes"
        acu[index] = 1
    else:
        print "No"
    index += 1