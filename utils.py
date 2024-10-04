import spacy
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
import torch
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as tfidf_cosine_similarity

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

model = SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_data
def load_data(file_path):
    """Load and preprocess the academic papers dataset."""
    # st.write("Loading data...")
    data = pd.read_csv(file_path)
    # st.write("Data loaded successfully!")

    required_columns = ['title', 'abstract', 'comments', 'journal-ref', 'doi', 'report-no', 'license', 'authors_parsed', 'categories']
    for column in required_columns:
        if column not in data.columns:
            st.error(f"Missing required column: {column}")
            return None, None, None  

    data['comments'].fillna('No comments', inplace=True)
    data['journal-ref'].fillna('No journal reference', inplace=True)
    data['doi'].fillna('No DOI', inplace=True)
    data['report-no'].fillna('No report number', inplace=True)
    data['license'].fillna('No license', inplace=True)

    data['title_normalized'] = data['title'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)
    data['abstract_normalized'] = data['abstract'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)

    data['authors_extracted'] = data['authors_parsed'].apply(eval).apply(lambda x: ', '.join([f"{author[1]} {author[0]}" for author in x]))
    data['categories_list'] = data['categories'].str.split()

    data['title_embedding'] = list(model.encode(data['title_normalized'].tolist(), convert_to_tensor=False))
    data['abstract_embedding'] = list(model.encode(data['abstract_normalized'].tolist(), convert_to_tensor=False))

    combined_text = data['title_normalized'] + " " + data['abstract_normalized']
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)

    data['title_entities'] = data['title_normalized'].apply(extract_entities)
    data['abstract_entities'] = data['abstract_normalized'].apply(extract_entities)

    return data, tfidf_matrix, tfidf_vectorizer 


@st.cache_data
def compute_tfidf(data):
    """Compute TF-IDF vectors for title and abstract."""
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    title_tfidf = tfidf_vectorizer.fit_transform(data['title_normalized'])
    abstract_tfidf = tfidf_vectorizer.fit_transform(data['abstract_normalized'])
    return title_tfidf, abstract_tfidf, tfidf_vectorizer

def extract_entities(text):
    """Extract named entities from text."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def get_unique_authors(data):
    """Return a list of unique authors from the DataFrame."""
    authors = set()
    for author_list in data['authors_extracted']:  
        authors.update(author.strip() for author in author_list.split(',') if author)
    return list(authors)

def get_unique_categories(data):
    """Return a list of unique categories from the DataFrame."""
    categories = set()
    for category_list in data['categories']:  
        split_categories = re.split(r'[,\s]+', category_list) 
        categories.update(category.strip() for category in split_categories if category)
    return list(categories)

def get_synonyms(word):
    """Get synonyms from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def extract_entities(text):
    """Extract named entities from text, focusing on authors and institutions."""
    doc = nlp(text)
    entities = {'authors': [], 'organizations': []}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities['authors'].append(ent.text)
        elif ent.label_ in ["ORG", "GPE"]: 
            entities['organizations'].append(ent.text)
    return entities


def expand_query(original_query):
    """Expand the user query with synonyms."""
    tokens = nltk.word_tokenize(original_query.lower())
    expanded_terms = []
    for token in tokens:
        synonyms = get_synonyms(token)
        expanded_terms.append(token)
        expanded_terms.extend(synonyms)
    return ' '.join(set(expanded_terms))

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    device = a.device if isinstance(a, torch.Tensor) else 'cpu'
    
    if isinstance(a, np.ndarray):
        a = torch.tensor(a, device=device)
    if isinstance(b, np.ndarray):
        b = torch.tensor(b, device=device)

    a = a.to(device)
    b = b.to(device)
    
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def search(data, query, tfidf_matrix, tfidf_vectorizer, author_filter=None, cat_filter=None):
    """Search for academic papers based on a user query with optional entity filters."""

    expanded_query = expand_query(query)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    query_embedding = model.encode(expanded_query, convert_to_tensor=True, device=device)

    query_tfidf = tfidf_vectorizer.transform([expanded_query])

    data['title_similarity'] = data['title_embedding'].apply(lambda x: cosine_similarity(query_embedding, x))
    data['abstract_similarity'] = data['abstract_embedding'].apply(lambda x: cosine_similarity(query_embedding, x))

    tfidf_similarity = tfidf_cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    data['exact_match'] = data.apply(
        lambda row: 1 if (query.lower() in row['title_normalized'] or query.lower() in row['abstract_normalized']) else 0,
        axis=1
    )
    data['exact_match_score'] = data['exact_match'] * 1  

    data['final_similarity'] = (
        0.4 * data['title_similarity'] +
        0.4 * data['abstract_similarity'] +
        0.3 * tfidf_similarity +
        data['exact_match_score'] 
    )

    if author_filter:
        data = data[data['authors_extracted'].str.contains(author_filter, case=False, na=False)]
    if cat_filter:
        data = data[data['categories_list'].apply(lambda categories: any(cat_filter in category for category in categories))]


    sorted_data = data.sort_values(by=['exact_match', 'final_similarity'], ascending=[False, False])

    top_n = 10
    return sorted_data.head(top_n)[['title', 'authors_extracted', 'final_similarity', 'categories','abstract']]

