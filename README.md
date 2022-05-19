# Network Science project
## Graph Neural Networks for Recommender Systems

### Related work
We have read several materials about recommendation systems and graph neural networks. Our main goal is to train graph neural network to build a recommender system.
- [Recommender systems based on graph embedding techniques: A comprehensive review](https://arxiv.org/pdf/2109.09587.pdf)
- [Analysis of Product Purchase Patterns in a Co-purchase Network](https://ieeexplore.ieee.org/document/7052071)
- [Rank the Top-N Products in Co-Purchasing Network through Discovering Overlapping Communities Using (LC- BDL) Algorithm](http://www.jmest.org/wp-content/uploads/JMESTN42352389.pdf)
- [entity2rec: Property-specific knowledge graph embeddings for item recommendation](https://www.sciencedirect.com/science/article/abs/pii/S0957417420300610)
- [Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/3219819.3219890)
- [Knowledge Graph Convolutional Networks for Recommender Systems](https://arxiv.org/pdf/1904.12575.pdf)

### Dataset
As our main goal is to build a recommender system, we needed the data of products and customers and the relationships between them. We found the open data of Amazon co-purchased products from the Stanford University homepage.

The link of source dataset: https://snap.stanford.edu/data/amazon-meta.html

### Methodology
The basic roadmap from data to recommender systems would look roughtly like this:

```
Dataset -> Clean -> Graph -> Graph Analyze -> Neural Network -> Recommender System
```

### Files
There are 3 main files: 
1. `DataPreparation.ipynb` which consists
- the **e**xtraction of data from source file (text to pandas dataframe) (**done**)
  - additional file `text2dictionary.py` including the method for collecting the data from text file
- the **t**ransformation of initial dataframe for cleansing purposes (**done**)
- initial analysis on data to filter out the most relevant, accurate and interesting part (**done**)
- the **l**oading to alinks, bilinks and nodes dataframes (**done**)
2. `GraphAnalysis.ipynb` which contains
- graph creation (**done**)
  - product-product graph with links of Amazon's similarities `alinks`
  - product-customer bipartite graph with links gained from customer reviews `bilinks`
  - product-product graph with links connecting products of same customer `clinks`
    - clinks creation can be found in `preprocessor.py`
- graph analysis (**done**)
  - graph methods provided in additional file `graphmethods.py`
3. `Recommender.ipynb` consisting of
- making Stellargraph object of `nodes` and `alinks`
- embedding the graph with shallow net `Attri2vec` which uses note attributes to create node embeddings, and deep graph convolutional neural network `GraphSAGE` which uses the node and it's neighbours attributes embedding them, aggregating them, etc, containing more structural information than Attri2vec should do
- Using embeddings in `recommender` which uses unsupervised Nearest Neighbour algorithm. We used different distance metrics as `minkowski`, `manhattan` and `canberra`.
- Finding the recommendations using similarity measures as `adamic-adar` (networkx actually has a bug in their algorithm and we cant use it), `jaccard similarity` and `preferential attachment`.
  - all needed functions are saved into `recommender.py`.
4. Working notebooks created during the project are in Drafts folder to save the history.
5. Additionally, needed packages can be installed to your 3.8 python environment from `requirements.txt` file.

```
conda create --name NS
conda activate NS
conda install python=3.8
conda install pip
pip install -r requirements.txt
```