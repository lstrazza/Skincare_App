# skincare_app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel
import nltk

# Only download once
nltk.download('punkt')
nltk.download('stopwords')

# ========== LOAD & CLEAN DATA ==========
@st.cache_data
def load_data():
    df = pd.read_excel("Sephora_Description1.1.xlsx")
    df['Description'] = df['Description'].fillna('')
    return df

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]

@st.cache_data
def prepare_lda(df):
    texts = df['Description'].apply(preprocess).tolist()
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary, passes=10, random_state=42)
    topic_dist = [lda_model.get_document_topics(doc) for doc in corpus]
    main_topic = [max(doc, key=lambda x: x[1])[0] if doc else -1 for doc in topic_dist]
    df['main_topic'] = main_topic
    return lda_model, df, corpus, dictionary

KEY_TAGS = [
    'moisturizing', 'hydrating', 'soothing', 'anti-inflammatory', 'brightening',
    'firming', 'anti-aging', 'nourishing', 'acne', 'oily', 'dry', 'sensitive',
    'eye', 'lip', 'body', 'face', 'redness', 'softening', 'elasticity', 'stabilizes', 'moisture', 'moisturizes',
    'enhances', 'protects', 'texture', 'ph', 'hydrates', 'improves',
   'provides', 'prevents', 'adds', 'product', 'softens', 'formulations', 'hydration', 'agent', 'balances', 'draws', 'into'
]

KEY_TOPICS = [
    {
        "Topic 0": [
            ("skin", 0.099), ("ph", 0.034), ("product", 0.025), ("adds", 0.024), ("moisture", 0.022),
            ("agent", 0.020), ("scent", 0.018), ("cleansing", 0.017), ("prevents", 0.017), ("fragrance", 0.016)
        ]
    },
    {
        "Topic 1": [
            ("skin", 0.158), ("moisture", 0.034), ("moisturizes", 0.033), ("hydrates", 0.032), ("protects", 0.023),
            ("stabilizes", 0.022), ("texture", 0.021), ("improves", 0.020), ("helps", 0.019), ("reduces", 0.019)
        ]
    },
    {
        "Topic 2": [
            ("skin", 0.092), ("ph", 0.066), ("product", 0.041), ("stabilizes", 0.040), ("texture", 0.030),
            ("balances", 0.030), ("moisture", 0.028), ("formulations", 0.026), ("adjusts", 0.025), ("hydrates", 0.022)
        ]
    },
    {
        "Topic 3": [
            ("skin", 0.124), ("protects", 0.047), ("adds", 0.032), ("provides", 0.029), ("moisturizes", 0.029),
            ("scent", 0.028), ("fragrance", 0.024), ("softens", 0.023), ("moisture", 0.021), ("damage", 0.018)
        ]
    },
    {
        "Topic 4": [
            ("skin", 0.093), ("texture", 0.044), ("stabilizes", 0.035), ("moisture", 0.031), ("enhances", 0.028),
            ("helps", 0.027), ("product", 0.025), ("moisturizes", 0.021), ("improves", 0.021), ("provides", 0.020)
        ]
    }
]

def match_description_to_topic(user_description, key_tags, key_topics):
    user_words = user_description.lower().split()
    matched_tags = [tag for tag in key_tags if tag in user_words]

    topic_scores = []
    for i, topic in enumerate(key_topics):
        _, keywords = list(topic.items())[0]
        score = sum(weight for word, weight in keywords if word in matched_tags)
        topic_scores.append((i, score))

    topic_scores.sort(key=lambda x: x[1], reverse=True)
    best_topic_index = topic_scores[0][0] if topic_scores else None
    return best_topic_index, matched_tags

def recommend_products(user_input, df, key_tags, key_topics, top_n=5):
    topic_index, matched = match_description_to_topic(user_input, key_tags, key_topics)
    
    if topic_index is None:
        return pd.DataFrame(), matched

    recommended = df[df['main_topic'] == topic_index].copy()
    recommended = recommended[['Brand', 'Product_Name', 'Description']].head(top_n)
    return recommended, matched

# ========== STREAMLIT APP ==========
st.title("🧴 Skincare Recommendation App")
st.markdown("Tell me what your skin needs, and I'll recommend a few products!")

user_input = st.text_area("Describe your skincare needs:", 
    "I want something moisturizing that improves skin texture and hydrates deeply.")

if st.button("Get Recommendations"):
    with st.spinner("Finding the best skincare products for you..."):
        sephora_data = load_data()
        lda_model, sephora_data, corpus, dictionary = prepare_lda(sephora_data)
        recs, tags = recommend_products(user_input, sephora_data, KEY_TAGS, KEY_TOPICS)
    
    if recs.empty:
        st.warning("Sorry, I couldn't find any matching products.")
    else:
        st.success(f"Matched tags: {', '.join(tags)}")
        st.subheader("🧼 Recommended Products:")
        st.dataframe(recs)


#connect to github ! and then run