import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stop_words = stopwords.words('english')
punct = string.punctuation
stemmer = PorterStemmer()

def cleaner(text):
    # all charaters to lowercase
    text = text.lower().split()
    # removing URLs
    text = [re.sub(r"^(http).*",'',word) for word in text]
    # moving non-letter symbols: numbers, symbols .,:;_- etc and replace with white space
    # but removing leading and trailing spaces
    text = [re.sub(r"[^a-zA-Z]",' ',word).strip() for word in text]
    # removing punctuation and stopwords and numbers
    text = [word for word in text if (word not in stop_words) and (word not in punct)]
    # join words to one sentence again
    text=' '.join(text)
    return text

def get_features(nodes,columns):
    """ 
    Define the nodes you want to turn into numerical format
    columns = {"ID": [], "numerical": [], "textual": [], "categorical": []}
    """
    IDs = nodes[columns.get("ID")]
    Numerical = nodes[columns.get("numerical")]
    Textual = nodes[columns.get("textual")]
    if columns.get("textual"):
        text_col = Textual.apply(
            lambda row: ' '.join([cleaner(row[col]) for col in Textual.columns]),axis=1)
        converter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stop_words)
        Textual = pd.DataFrame(converter.fit_transform(text_col).toarray())
    Categorical = nodes[columns.get("categorical")]
    Categorical = Categorical.astype("category")
    Categorical = Categorical.apply(lambda x: x.cat.codes)
    Features = pd.concat([IDs,Numerical,Textual,Categorical],axis=1)
    return Features

def create_clinks(bilinks,nodes):
    # only the best ratings are considered for recommendation:
    good_ratings = bilinks[bilinks.Rating>4]

    # join the products through THE SAME CUSTOMER:
    one = good_ratings.set_index("CId")
    two = good_ratings.set_index("CId")
    all_links = one.join(two,lsuffix="_1",rsuffix="_2")
    diff_nodes = all_links[all_links.Id_1 != all_links.Id_2]

    # select only the source and target node id columns:
    clinks = diff_nodes[["Id_1","Id_2"]]
    clinks = clinks.drop_duplicates()
    clinks = clinks.reset_index(drop=True)

    # Check that all nodes exists in nodes
    clinks = clinks[(clinks["Id_1"].isin(nodes.index)) & 
                    (clinks["Id_2"].isin(nodes.index))]
    
    # rename columns for stellargraph
    clinks.columns = ["source","target"]
    return clinks