from neighbourhood import Neighbourhood
import numpy as np
from tqdm import tqdm

class BipartiteGraph:
    def __init__(self,features,bilinks,links):
        self.initdim = features.shape[1]
        self.Features = features
        self.links = links
        self.bilinks = bilinks
        self.N = Neighbourhood(bilinks,links)
        
    def random_subgraph(self,nbr_of_nodes):
        """ 
        Generates random Node IDs for random subgraph of given size.
        """
        NodeIDs = np.random.choice(self.bilinks.Id.unique(),size=nbr_of_nodes)
        return NodeIDs
    
    def pooling(self,NodeIDs,pool_size): 
        """ 
        Neighbourhood function that find the most imporant neighbours - the pool.
        Also calculates their importances.
        """
        NeighbourIDs = [] # (15, 4)
        Importances = [] # (15, 4)
        for Node in tqdm(NodeIDs,
                         desc=f"Finding neighbours and importances"):
            Pool = self.N.importance_pooling(Node,T=pool_size)
            Neighs = list(Pool.keys())
            Imps = list(Pool.values())
            if len(Neighs) < pool_size:
                for i in range(pool_size-len(Neighs)):
                    Neighs.append(0)
                    Imps.append(0)
            NeighbourIDs.append(Neighs)
            Importances.append(Imps)
        return NeighbourIDs, Importances
    
    def init_embedding(self,NodeIDs):
        """ 
        Fetches the initial embeddings for nodes - features of nodes.
        """
        NodeFeatures = []
        for id in NodeIDs:
            x = self.Features[self.Features.Id==id]
            if x.shape[0] != 0:
                NodeFeatures.append(np.array(x))
            else: NodeFeatures.append(np.zeros((1,self.initdim)))
        return np.array(NodeFeatures).reshape(len(NodeIDs),self.initdim)
    
