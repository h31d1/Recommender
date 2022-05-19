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
Dataset -> Clean -> Graph -> Graph Analyze -> Neural Network building -> Recommender System
```

### Files
Main file is `Notebook.ipynb` which consists
- the **e**xtraction data from source file (text to pandas dataframe) (**done**)
  - additional file `text2dictionary.py` including the method for collecting the data from text file
- the **t**ransformation of initial dataframe for cleansing purposes (**done**)
- initial analysis on data to filter out the most relevant, accurate and interesting part (**done**)
- the **l**oading to links, bilinks and nodes dataframes (**done**)
- graph creation (*in progress*)
- graph analysis (*in progress*)
  - graph methods provided in additional file `graphmethods.py`
