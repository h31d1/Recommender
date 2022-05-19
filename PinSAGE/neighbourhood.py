import numpy as np

""" Neighbourhood in bipartite graph """

class Neighbourhood:
    
    def __init__(self, bilinks, links):
        self.bilinks = bilinks
        self.links = links

    def find_neighbourhood(self, Product):
        """
        Finds 1-hop neighbours through common customer.
        """
        #if Product not in bilinks.Id.values: return
        bilinks = self.bilinks
        Neighbourhood, Similarities = [],[]
        ProdRating = bilinks[bilinks.Id==Product]["Rating"].mean()
        related_customers = list(bilinks[bilinks.Id==Product]["CId"])
        for customer in related_customers:
            df = bilinks[(bilinks.CId==customer) & (bilinks.Id != Product)]
            Neighbourhood += list(df.Id.values)
            Similarities += list(4.0-np.abs(df.Rating.values-ProdRating))
        return Neighbourhood, Similarities


    def get_neighbourhood(self, Product):
        """
        Gets all 1-hop neighbours (through common customer and direct neighbours).
        """
        links = self.links
        Neighbourhood, Similarities = [],[]
        N, S = self.find_neighbourhood(Product)
        Neighbourhood += N
        Similarities += S
        Neighbourhood += list(links[links["from"]==Product]["to"]) 
        Neighbourhood += list(links[links["to"]==Product]["from"])
        n = len(Neighbourhood)-len(Similarities)
        if n>0: Similarities += n*[4.0]
        summa = np.maximum(sum(Similarities),0.1)
        Probabilities = list(np.array(Similarities)/summa)
        return Neighbourhood, Probabilities


    def get_receptive_field(self, Product, K=2):
        """
        Chooses K neighbours from Neighbourhood distribution.
        """
        Neighbourhood,Probabilities = self.get_neighbourhood(Product)
        Probabilities = Probabilities[:-1].append(1.-np.sum(Probabilities[:-1]))
        if len(Neighbourhood)>0: 
            Field = list(np.random.choice(
                Neighbourhood,size=K,replace=True,p=Probabilities))
            return Field
        else:
            return None
        

    def generate_random_walk(self,Product, H=5):
        """
        Generates H-hop random walk from node Product
        """
        random_walk = [Product]
        for hop in range(H):
            step = self.get_receptive_field(random_walk[hop],1)
            if step != None: random_walk.append(step[0])
            else: break
        return random_walk[1:]
    
    
    def importance_pooling(self, Product,T=10):
        """ 
        Selects T most important neighbours of Product 
        by generating T random walks from T-hop neighbourhood of Product.
        """
        Neighbourhood = {}
        for times in range(T):
            collect = self.generate_random_walk(Product,H=T)
            for neighbour in collect:
                if neighbour in Neighbourhood: Neighbourhood[neighbour] += 1
                else: Neighbourhood[neighbour] = 1
        Neighbourhood = L1_norm(Neighbourhood)
        Neighbourhood = dict(list(Neighbourhood.items())[:T])
        return Neighbourhood

def L1_norm(dictionary):
    """ 
    Calculates L1 norm for dictionary values and sort them in descending order.
    """
    s = sum(list(dictionary.values()))
    N = {k: v/s for k, v in sorted(dictionary.items(), key=lambda item: item[1],reverse=True)}
    return N