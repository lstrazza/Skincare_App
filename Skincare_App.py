import dash
from dash import dcc, html, Input, Output, Dash
import pandas as pd
import numpy as np
import re
from collections import Counter
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os
import dash_bootstrap_components as dbc

nltk.download('punkt')
nltk.download('stopwords')


# Dash Setup
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# Load Data
sephora_data = pd.read_excel("Sephora_Description1.1.xlsx")
sephora_data['Description'] = sephora_data['Description'].fillna('')

# Preprocess for LDA
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]

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
    user_tokens = set(word_tokenize(user_description.lower()))
    matched_tags = [tag for tag in key_tags if tag in user_tokens]

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
    print(f"\n[recommend_products] Called with user_input: '{user_input}'")
    
    topic_index, matched = match_description_to_topic(user_input, key_tags, key_topics)
    
    print(f"[recommend_products] Topic index: {topic_index}")
    print(f"[recommend_products] Matched tags: {matched}")

    if topic_index is None:
        print("[recommend_products] No matching topic found. Returning empty list.")
        return [], matched

    recommended = df[df['main_topic'] == topic_index].copy()
    print(f"[recommend_products] Number of products matched with topic {topic_index}: {len(recommended)}")

    print("Columns in recommended:", recommended.columns.tolist())
    recommended = recommended[['brand_name', 'product_name', 'Description']].head(top_n)
    print(f"[recommend_products] Returning top {len(recommended)} recommendations")

    return recommended, matched


# App Layout
app.layout = html.Div(
    style={
        'backgroundColor': '#f5f9f6',  # sage-inspired soft green background
        'fontFamily': 'Helvetica Neue, sans-serif',
        'padding': '40px'
    },
    children=[
        html.Div([
            html.H1("🌿 Skincare Matchmaker", style={
                'textAlign': 'center',
                'color': '#3a5a40',
                'fontSize': '3em',
                'fontWeight': 'bold',
                'marginBottom': '20px',
                'fontFamily': 'Georgia, serif'
            }),
            html.H5("Get matched with your glow-up essentials", style={
                'textAlign': 'center',
                'color': '#6c757d',
                'marginBottom': '40px'
            })
        ]),

        dbc.Row([
            # Left panel
            dbc.Col([
                html.Div([
                    html.Label("Choose Your Category", style={'fontWeight': 'bold', 'color': '#3a5a40'}),
                    dcc.RadioItems(
                        id="type-radio",
                        options=[
                            {"label": "Skin", "value": "Skin"},
                            {"label": "Lips", "value": "Lips"},
                            {"label": "Eyes", "value": "Eyes"},
                            {"label": "Bath", "value": "Bath"},
                            {"label": "Body", "value": "Body"}
                        ],
                        value="Skin",
                        labelStyle={'display': 'block', 'margin': '5px 0'},
                        inputStyle={'marginRight': '10px'}
                    ),

                    html.Label("Your Skin Type", style={'fontWeight': 'bold', 'marginTop': '20px', 'color': '#3a5a40'}),
                    dcc.Dropdown(
                        id="skin-type-dropdown",
                        options=[
                            {"label": "Sensitive", "value": "Sensitive"},
                            {"label": "Dry", "value": "Dry"},
                            {"label": "Oily", "value": "Oily"},
                            {"label": "Combination", "value": "Combination"}
                        ],
                        value="Sensitive",
                        style={"marginBottom": "30px"}
                    )
                ], style={
                    'padding': '30px',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '15px',
                    'boxShadow': '0 4px 10px rgba(0,0,0,0.05)'
                })
            ], width=4),

            # Right panel
            dbc.Col([
                html.Div([
                    html.Label("Tell us what you're looking for", style={'fontWeight': 'bold', 'color': '#3a5a40'}),
                    dcc.Textarea(
                        id="skincare-input",
                        value="I want something moisturizing that improves skin texture and hydrates deeply.",
                        style={"width": "100%", "height": 120, 'marginBottom': '20px'}
                    ),
                    dbc.Button("✨ Get Recommendations", id="submit-btn", n_clicks=0, color="success", className="mb-3"),

                    html.Div(id="tags-output", style={'marginTop': '20px'}),
                    html.Div(id="recommendations-output", style={
                        'marginTop': '20px',
                        'backgroundColor': '#ffffff',
                        'padding': '20px',
                        'borderRadius': '15px',
                        'boxShadow': '0 4px 10px rgba(0,0,0,0.05)'
                    })
                ], style={
                    'padding': '30px',
                    'backgroundColor': '#e9f5f1',
                    'borderRadius': '15px',
                    'boxShadow': 'inset 0 0 6px rgba(0,0,0,0.03)'
                })
            ], width=8)
        ], justify="center")
    ]
)
# Callback for generating recommendations
@app.callback(
    [Output("tags-output", "children"),
     Output("recommendations-output", "children")],
    [Input("submit-btn", "n_clicks")],
    [dash.dependencies.State("skincare-input", "value")]
)
def update_recommendations(n_clicks, user_input):
    print(f"Function called with n_clicks={n_clicks} and user_input='{user_input}'")
    
    if n_clicks > 0:
        print("n_clicks > 0, generating recommendations...")
        recommendations, matched_tags = recommend_products(user_input, sephora_data, KEY_TAGS, key_topics)
        
        print(f"Matched tags: {matched_tags}")
        print(f"Number of recommendations: {0 if recommendations is None or recommendations.empty else len(recommendations)}")
        
        tag_output = f"Matched Tags: {', '.join(matched_tags)}" if matched_tags else "No matching tags found."
        
        if recommendations is not None and not recommendations.empty:
            recs = [
                html.Li([
                    html.Strong(row.get('product_name', 'Unnamed Product')),
                    html.Details([
                        html.Summary("Description"),
                        html.P(row.get('Description', 'No description available'))
                    ])
                ])
                for _, row in recommendations.iterrows()
            ]

            recommendations_output = html.Ul(recs)
        else:
            recommendations_output = "No recommendations available."
        
        print("Returning tag_output and recommendations_output")
        return tag_output, recommendations_output

    print("n_clicks <= 0, returning empty outputs")
    return "", ""



server = app.server
# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
