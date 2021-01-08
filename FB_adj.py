import pandas as pd
import numpy as np
import glob

FILES = glob.glob('ego-Facebook/*.edges')
dataFrames = [pd.read_table(file,sep=' ',header=None) for file in FILES]
df = pd.concat(dataFrames, ignore_index=True)
df = df.rename(columns={0: "from", 1: "to"})

adj_matrix = np.zeros([df['from'].max()+1,df['to'].max()+1])
adj_matrix.shape

node_id = []
node_adj = []
count = 0
tmp = df.groupby('from')
for group_idx, group_data in tmp:
    count += 1
    node_id.append(group_idx)
    node_adj.append(group_data['to'].unique())

for i in range(count):
    for j in range(node_adj[i].shape[0]):
        adj_matrix[node_id[i],node_adj[i][j]] = 1
        adj_matrix[node_adj[i][j],node_id[i]] = 1

ego_list = [0,107,1684,1912,3437,348,3980,414,686,698]
for i in range(10):
    tb = dataFrames[i]
    ego = ego_list[i]
    node_list = tb[tb.columns[0]].unique()
    print(ego)
    for j in range(node_list.shape[0]):
        adj_matrix[ego,node_list[j]] = 1
        adj_matrix[node_list[j],ego] = 1

np.save('Adj_Matrix_FB', adj_matrix.astype(np.bool_))