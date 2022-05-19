
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Layer, Dense, Concatenate
import numpy as np

class Aggregator(Layer):
    def __init__(self, units):
        super(Aggregator, self).__init__()
        self.units = units

    def call(self, inputs):
        Neighbours = inputs[0]
        Alphas = inputs[1]
        Neighbourhood = Dense(self.units,activation='relu',dtype='float32')(Neighbours)
        return tf.matmul(Alphas,Neighbourhood)

class Convolve(Layer):
    def __init__(self, units):
        super(Convolve, self).__init__()
        self.units = units

    def call(self, inputs):
        Node, Neighbours, Alpha = inputs
        NodeEmb = Dense(self.units,activation='relu',dtype='float32')(Node)
        NeighboursEmb = Aggregator(self.units)([Neighbours,Alpha])
        AggEmb = Concatenate()([NodeEmb,NeighboursEmb])
        Emb = Dense(self.units,activation='relu',dtype='float32')(AggEmb)
        return tf.divide(Emb,tf.norm(Emb,ord=2))

class Sampler(Layer):
    def __init__(self, graph, depth_K, pool_T):
        super(Sampler, self).__init__()
        self.depth_K = depth_K
        self.G = graph
        self.pool_T = pool_T
    
    def call(self,inputs):
        node_ids = inputs #tf.constant(inputs,dtype="int32")
        stacks, alphas = [],[]
        for k in range(self.depth_K): # k = 0,...,K-1
            neigh_ids = tf.constant(self.G.pooling(node_ids,self.pool_T)[0]) # get neighbours
            alpha = tf.constant(self.G.pooling(node_ids,self.pool_T)[1]) # get importances
            node_ids = tf.constant(node_ids,shape=(node_ids.numpy().size,1),dtype="int32") # reshape nodes
            stack = tf.concat([node_ids,neigh_ids],axis=-1) # stack nodes with their neighbours
            stacks.append(stack)
            alphas.append(alpha)
            if k > 0:
                stacks[k] = tf.concat([stack,stacks[k-1]],axis=0)
                alphas[k] = tf.concat([alpha,alphas[k-1]],axis=0)
            node_ids = tf.concat(tf.unstack(neigh_ids, axis=1),axis=0) # flatten neighbours for next round nodes
        return stacks[-1], alphas

class Embedder(Layer):
    def __init__(self, graph, depth_K, pool_T, units):
        super(Embedder, self).__init__()
        self.G = graph
        self.depth_K = depth_K
        self.pool_T = pool_T
        self.init_dim = self.G.initdim
        self.units = units
    
    def call(self, inputs):
        stack = stack_back(inputs[0],self.pool_T,self.depth_K)
        alphas = stack_back(inputs[1],self.pool_T,self.depth_K,alphas_tensor=True)
        Sn = alphas.shape[0]
        a1 = int(Sn*(self.pool_T-1)/(self.pool_T**(self.depth_K)-1))
        sizes = [0]+list(np.cumsum([a1*(self.pool_T**i) for i in range(self.depth_K)]))
        # get first stack with initial embeddings
        stack_np = tf.concat(tf.unstack(stack,axis=1),axis=0).numpy() # flatten and numpy
        stack_emb = tf.reshape(tf.constant(self.G.init_embedding(stack_np)), [-1, self.pool_T+1, self.init_dim])
        alphas = tf.expand_dims(alphas,axis=1)
        # layers:
        for k in range(self.depth_K-1,-1,-1): # k = K-1,...,0
            # nodes, neighbours, alphas -> CONV -> new embeddings
            node_emb = tf.expand_dims(tf.unstack(stack_emb,axis=1)[0],axis=1)
            neigh_emb = tf.reshape(tf.concat(tf.unstack(stack_emb,axis=1)[1:],axis=0), [-1, self.pool_T, stack_emb.shape[-1]])
            new_emb = Convolve(self.units)([node_emb, neigh_emb, alphas])
            if k > 0: # k = K-1,...,1
                # new embeddings -> COMBINE -> next stack
                splits,combined = [],[]
                for i in range(1,len(sizes[:k+2])):
                    split = tf.concat(tf.unstack(new_emb,axis=0)[sizes[i-1]:sizes[i]],axis=0)
                    splits.append(tf.expand_dims(split,axis=1))
                    if i > 1: 
                        reshape_split = tf.reshape(split,[-1,self.pool_T,split.shape[-1]])
                        combined.append(tf.concat([splits[i-2],reshape_split],axis=1))
                stack_emb = tf.concat(combined,axis=0)
                alphas = tf.unstack(alphas,axis=0)[:sizes[k]]
        return new_emb

def stack_back(stack, pool_T, depth_K, alphas_tensor=False):
    if alphas_tensor: y = pool_T
    else: y = pool_T + 1
    pieces = [tf.concat(
        [tf.reshape(tf.unstack(stack,axis=1)[i],shape=(stack.shape[0],1)) for i in range((y)*j, (y)*(j+1))]
        ,axis=1) for j in range(int(stack.shape[1]/(y)))]
    s = [pieces[0]]
    for i in range((pool_T+1)):
        for j in range(depth_K+1):
            s.append(pieces[1+i+(pool_T+1)*j])
    return tf.concat(s,axis=0)
    
class PinSAGE(Model):
    def __init__(self, bip_graph, pool_T, depth_K, emb_dim):
        super().__init__()
        self.I = bip_graph
        self.pool_T = pool_T
        self.depth_K = depth_K
        self.init_dim = self.I.initdim
        self.emb_dim = emb_dim
    
    def sampling(self, inputs):
        stacks, alphas = [],[]
        node_ids = tf.constant(inputs,dtype="int32")
        for k in range(self.depth_K): # k = 0,...,K-1
            neigh_ids = tf.constant(self.I.pooling(node_ids,self.pool_T)[0]) # get neighbours
            alpha = tf.constant(self.I.pooling(node_ids,self.pool_T)[1]) # get importances
            node_ids = tf.constant(node_ids,shape=(node_ids.numpy().size,1),dtype="int32") # reshape nodes
            stack = tf.concat([node_ids,neigh_ids],axis=-1) # stack nodes with their neighbours
            stacks.append(stack)
            alphas.append(alpha)
            if k > 0:
                stacks[k] = tf.concat([stack,stacks[k-1]],axis=0)
                alphas[k] = tf.concat([alpha,alphas[k-1]],axis=0)
            node_ids = tf.concat(tf.unstack(neigh_ids, axis=1),axis=0) # flatten neighbours for next round nodes
        return stacks[-1], alphas
    
    def embedding(self, stack, importances):
        stack_sizes = [0]+[importances[i].shape[0] for i in range(len(importances))]
        # get first stack with initial embeddings
        stack_np = tf.concat(tf.unstack(stack,axis=1),axis=0).numpy() # flatten and numpy
        stack_emb = tf.reshape(tf.constant(self.I.init_embedding(stack_np)), [-1, self.pool_T+1, self.init_dim])
        # layers:
        for k in range(self.depth_K-1,-1,-1): # k = K-1,...,0
            # nodes, neighbours, alphas -> CONV -> new embeddings
            node_emb = tf.expand_dims(tf.unstack(stack_emb,axis=1)[0],axis=1)
            neigh_emb = tf.reshape(tf.concat(tf.unstack(stack_emb,axis=1)[1:],axis=0), [-1, self.pool_T, stack_emb.shape[-1]])
            alphas = tf.expand_dims(importances[k],axis=1)
            new_emb = Convolve(self.units)([node_emb, neigh_emb, alphas])
            if k > 0: # k = K-1,...,1
                # new embeddings -> COMBINE -> next stack
                splits,combined = [],[]
                for i in range(1,len(stack_sizes[:k+2])):
                    split = tf.concat(tf.unstack(new_emb,axis=0)[stack_sizes[i-1]:stack_sizes[i]],axis=0)
                    splits.append(tf.expand_dims(split,axis=1))
                    if i > 1: 
                        reshape_split = tf.reshape(split,[-1,self.pool_T,split.shape[-1]])
                        combined.append(tf.concat([splits[i-2],reshape_split],axis=1))
                stack_emb = tf.concat(combined,axis=0)
        return new_emb

    def call(self, inputs):
        inputs = Input(shape=(1,),dtype="int32")
        stack, alphas = self.sampling(inputs)
        embeddings = self.embedding(stack, alphas)
        output = Dense(self.emb_dim,activation='relu',dtype='float32')(embeddings)
        return output

