import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


import nltk


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r"\w+")

def preprocess_text(text):
    """
    Text preprocessing pipeline for Project 2 (Search Engine):
    - Lowercasing
    - Removing punctuation
    - Removing numbers
    - Tokenization
    - Stopword removal
    - Lemmatization
    """

    if not isinstance(text, str):
        return []

    # 1. Lowercase
    text = text.lower()

    # 2. Remove numbers
    text = re.sub(r"\d+", "", text)

    # 3. Tokenization (no punkt, no errors)
    tokens = tokenizer.tokenize(text)

    # 4. Remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]

    # 5. Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


# Test
text = "The cars are running very fast!"
print(preprocess_text(text))


def preprocess_documents(documents):
    return [preprocess_text(doc) for doc in documents]

