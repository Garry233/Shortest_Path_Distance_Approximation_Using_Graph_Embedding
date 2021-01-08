import pandas as pd
import numpy as np
import time
import networkx as nx
import node2vec
from node2vec import Node2Vec

graph = nx.Graph(np.load('Adj_Matrix_FB.npy'))

# train node2vec model
start = time.clock()
node2vec = Node2Vec(graph, dimensions=32, walk_length=16, num_walks=10, workers = 2)
model = node2vec.fit()
elapsed = (time.clock() - start)
print("Training node2vec model : {:.2f} (s)".format(elapsed))

# load node2vec model
#from gensim.models import KeyedVectors
#model = KeyedVectors.load('node2vec_FB.model', mmap='r')

from sklearn.utils import resample, shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def vectors_transform(v1,v2,method):
    if(method=='sub_2'):
        v3 = v1-v2
        if(v3.sum()<0):
            v3 = v3*-1
    elif(method=='add'):
        v3 = v1+v2
    elif(method=='concat'):
        if(v1.sum()<v2.sum()):
            v3 = np.append(v1,v2)
        else:
            v3 = np.append(v2,v1)
    elif(method=='mul'):
        v3 = np.multiply(v1,v2)
    return v3

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost

record = []
for t in range(10):
    print("Round : ",t)
    num = 40000
    tr_query = np.random.choice(a=len(model.wv.vocab), size=num)
    tr_query = tr_query.reshape(-1,2)

    start = time.clock()
    tr_ans = []
    for i in range(int(num/2)):
        try:
            n=nx.shortest_path_length(graph,tr_query[i,0],tr_query[i,1])
            tr_ans.append(n)
        except nx.NetworkXNoPath:
            tr_ans.append(0)
    elapsed = (time.clock() - start)

    print("    Prepare training data : {:.2f} (s)".format(elapsed))

    num = 200000
    ts_query = np.random.choice(a=len(model.wv.vocab), size=num)
    ts_query = ts_query.reshape(-1,2)

    start = time.clock()
    ts_ans = []
    for i in range(int(num/2)):
        try:
            n=nx.shortest_path_length(graph,ts_query[i,0],ts_query[i,1])
            ts_ans.append(n)
        except nx.NetworkXNoPath:
            ts_ans.append(0)
    elapsed = (time.clock() - start)
    print("    Prepare testing data : {:.2f} (s)".format(elapsed))

    num = 40000
    method = 'concat'
    start = time.clock()
    tr_x = []
    tr_y = []
    for i in range(int(num/2)):
        x1 = model.wv[str(tr_query[i,0])]
        x2 = model.wv[str(tr_query[i,1])]
        y = tr_ans[i]
        tr_x.append(vectors_transform(x1,x2,method))
        tr_y.append(y)  
    tr_x = np.array(tr_x).reshape(int(num/2),-1)
    tr_y = np.array(tr_y).reshape(int(num/2),)
    #r_model = LinearRegression()
    #r_model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,8,4),
    #                        learning_rate_init=0.005,max_iter=500))
    #r_model = RandomForestRegressor(max_depth=10,n_estimators=20,criterion='mse')
    #r_model = xgboost.XGBRegressor(max_depth=10,n_estimators=80,learning_rate=0.1,
    #                               colsample_bytree=0.9,objective='reg:squarederror',n_jobs=3)
    r_model = xgboost.XGBClassifier(max_depth=10,n_estimators=80,learning_rate=0.1,
                                   colsample_bytree=0.9,num_class=6,nthread=3)
    
    r_model.fit(tr_x, tr_y)
    elapsed = (time.clock() - start)
    print("    Training Predictive model : {:.2f} (s)".format(elapsed))

    num = 200000
    method = 'concat'
    start = time.clock()
    ts_x = []
    ts_y = []
    for i in range(int(num/2)):
        x1 = model.wv[str(ts_query[i,0])]
        x2 = model.wv[str(ts_query[i,1])]
        y = ts_ans[i]
        ts_x.append(vectors_transform(x1,x2,method))
        ts_y.append(y)
    ts_x = np.array(ts_x).reshape(int(num/2),-1)
    ts_y = np.array(ts_y).reshape(int(num/2),)
    pred_y = np.rint(r_model.predict(ts_x))
    elapsed = (time.clock() - start)
    print("    Predict Distance : {:.2f} (s)".format(elapsed))
    ts_loss = mean_absolute_error(pred_y,ts_y)
    record.append(ts_loss)
    print("    Testing MAE loss : {:.2f}".format(ts_loss))
print("Average MAE loss : {:.2f}".format(sum(record)/10))
