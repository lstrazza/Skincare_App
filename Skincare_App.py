from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Flask setup
app = Flask(__name__)

# Load data
sephora_data = pd.read_excel("Sephora_Description1.1.xlsx")
sephora_data['Description'] = sephora_data['Description'].fillna('')

# Preprocess for LDA
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]

texts = sephora_data['Description'].apply(preprocess).tolist()
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# LDA model
lda_model = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary, passes=10, random_state=42)

# Topic distribution
sephora_data['topic_dist'] = [lda_model.get_document_topics(c) for c in corpus]
sephora_data['main_topic'] = sephora_data['topic_dist'].apply(lambda x: max(x, key=lambda tup: tup[1])[0])

# Clean description for later matching
def preprocess_description(desc):
    return re.sub(r'[^a-z\s]', '', desc.lower())

sephora_data['desc_clean'] = sephora_data['Description'].apply(preprocess_description)

# Tags and topics
KEY_TAGS = [
    'moisturizing', 'hydrating', 'soothing', 'anti-inflammatory', 'brightening',
    'firming', 'anti-aging', 'nourishing', 'acne', 'oily', 'dry', 'sensitive',
    'eye', 'lip', 'body', 'face', 'redness', 'softening', 'elasticity', 'stabilizes',
    'moisture', 'moisturizes', 'enhances', 'protects', 'texture', 'ph', 'hydrates',
    'improves', 'provides', 'prevents', 'adds', 'product', 'softens', 'formulations',
    'hydration', 'agent', 'balances', 'draws', 'into'
]

key_topics = [
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

# Match user input to topic
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

# Recommendation function
def recommend_products(user_input, df, key_tags, key_topics, top_n=5):
    topic_index, matched = match_description_to_topic(user_input, key_tags, key_topics)

    if topic_index is None:
        return [], matched

    recommended = df[df['main_topic'] == topic_index].copy()
    recommended = recommended[['Brand', 'Product_Name', 'Description']].head(top_n)
    return recommended, matched

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Skincare Recommender</title>
    <style>
        body {
            background-color: pink;
            font-family: Arial, sans-serif;
            margin: 20px;
        }

        .container {
            display: flex;
            align-items: flex-start;
            gap: 40px;
        }

        .left-panel {
            width: 200px;
        }

        .right-panel {
            flex: 1;
        }

        textarea {
            width: 100%;
            max-width: 500px;
        }

        .recommendations {
            margin-top: 20px;
            background-color: #fff0f5;
            padding: 10px;
            border-radius: 8px;
        }

        h3, h4 {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h2>🧴 Skincare Recommendation App</h2>

    <form method="POST">
        <div class="container">
            <!-- LEFT SIDE: Radio + Dropdown -->
            <div class="left-panel">
                <strong>Type:</strong><br>
                <input type="radio" id="skin" name="type" value="Skin">
                <label for="skin">Skin</label><br>

                <input type="radio" id="lips" name="type" value="Lips">
                <label for="lips">Lips</label><br>

                <input type="radio" id="eyes" name="type" value="Eyes">
                <label for="eyes">Eyes</label><br>

                <input type="radio" id="bath" name="type" value="Bath">
                <label for="bath">Bath</label><br>

                <input type="radio" id="body" name="type" value="Body">
                <label for="body">Body</label><br><br>

                <label for="skin_type"><strong>Skin Type:</strong></label><br>
                <select name="skin_type" id="skin_type">
                    <option value="Sensitive">Sensitive</option>
                    <option value="Dry">Dry</option>
                    <option value="Oily">Oily</option>
                    <option value="Combination">Combination</option>
                </select><br><br>
            </div>

            <!-- RIGHT SIDE: Textarea + Recommendations -->
            <div class="right-panel">
                <label for="skincare_input"><strong>Description:</strong></label><br>
                <textarea name="skincare_input" rows="4" placeholder="Describe your skincare needs...">I want something moisturizing that improves skin texture and hydrates deeply.</textarea><br><br>

                <input type="submit" value="Get Recommendations">

                {% if tags %}
                    <h4>Matched Tags: {{ tags | join(", ") }}</h4>
                {% endif %}

                {% if results %}
                    <div class="recommendations">
                        <h3>Recommended Products:</h3>
                        <ul>
                            {% for product in results %}
                                <li>
                                    <strong>{{ product.Product_Name }}</strong><br>
                                    Rating: {{ product.Rating }}<br>
                                    Ingredients: {{ product.Ingredients }}
                                </li>
                            {% endfor %}
                        </ul>
                    </div>
                {% endif %}
            </div>
        </div>
    </form>
</body>
</html>


"""

# Routes
@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    matched_tags = []
    if request.method == "POST":
        user_input = request.form["skincare_input"]
        recs, matched_tags = recommend_products(user_input, sephora_data, KEY_TAGS, key_topics)
        recommendations = recs.to_dict(orient='records')
    return render_template_string(HTML_TEMPLATE, tags=matched_tags, results=recommendations)

# Run app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT isn't set
    app.run(host='0.0.0.0', port=port)
