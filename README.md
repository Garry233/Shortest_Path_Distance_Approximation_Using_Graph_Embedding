# Shortest_Path_Distance_Approximation_Using_Graph_Embedding
Estimate shortest path distance on social network by using graph embedding technique  
file description :  
 - main.py : It can be divided into 5 steps  
 reading graph -> training node2vec model -> preparing training and testing set -> training predictive model -> prediction 
 - FB_adj.py : create adjacency matrix from ego-net edges data.
 - Adj_Matrix_FB.npy : An adjacency matrix of ego-Facebook data. You can download original dataset from SNAP website.  
 dataset link : [https://snap.stanford.edu/data/egonets-Facebook.html]
 - node2vec_FB.model : pre-trained node2vec model for Facebook dataset.  

We have tried different graph embedding method such as GraphSAGE and MUSAE, but node2vec has best performance.
Therefore, we will just use node2vec as embedding method in code.
