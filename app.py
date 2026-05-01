import streamlit as st
import pandas as pd
import numpy as np
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import json

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Financial Intelligence", page_icon="📈", layout="wide")

# --- UI STYLING ---
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1e24 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Title Styling */
    .main-title {
        font-size: 3rem !important;
        font-weight: 900 !important;
        background: -webkit-linear-gradient(45deg, #fbc2eb 0%, #a6c1ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Metric / KPI Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Cards and Glassmorphism */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Setup API Keys ---
# Using Streamlit Secrets for a cleaner UI and better security
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- NLTK Downloads ---
@st.cache_resource
def download_nltk_data():
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_data()

# Initialize tools
sia = SentimentIntensityAnalyzer()
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# 1. SMART NEWS FETCHING MODULE
def fetch_news(query="finance OR stock OR banking OR economy", num_articles=15):
    """Fetches real-time financial news using NewsAPI."""
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_KEY":
        st.warning("NewsAPI Key is not set. Using sample data for preview.")
        return pd.DataFrame([
            {"title": "Global Tech Stocks Surge Amid AI Boom", "description": "Technology shares hit record highs as artificial intelligence continues to drive global market indices upward. Investors are highly optimistic."},
            {"title": "Central Bank Warns of Impending Inflation Risks", "description": "In a surprising move, the central bank has warned about rising inflation, suggesting potential interest rate hikes that could plunge the market."},
            {"title": "Oil Prices Stabilize After Weeks of Volatility", "description": "Global crude oil prices have reached a stable point as demand forecasts align with current supply metrics, easing market fears."},
            {"title": "Retail Sales Plummet as Consumer Confidence Drops", "description": "FMCG sectors report massive losses as retail sales decline significantly. Economists fear a consumer-led recession."},
            {"title": "Green Energy Startups See Massive Investment Influx", "description": "Renewable energy sector experiences a massive surge in investments, showcasing strong growth potential and optimism among venture capitalists."}
        ])
    
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize={num_articles}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        
        news_list = []
        seen_titles = set()
        for article in articles:
            title = article.get('title', '')
            desc = article.get('description', '')
            if title and desc and title not in seen_titles:
                seen_titles.add(title)
                news_list.append({'title': title, 'description': desc})
                
        return pd.DataFrame(news_list)
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return pd.DataFrame()

# 2. TEXT PREPROCESSING MODULE
def preprocess_text(text):
    """Cleans and processes text using NLP."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    return " ".join(tokens)

# 3. ADVANCED SENTIMENT ANALYSIS MODULE
def analyze_sentiment(text):
    """Analyzes sentiment and categorizes it."""
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.5:
        category = "Strong Positive"
    elif 0.1 <= compound < 0.5:
        category = "Mild Positive"
    elif -0.1 < compound < 0.1:
        category = "Neutral"
    elif -0.5 < compound <= -0.1:
        category = "Mild Negative"
    else:
        category = "Strong Negative"
        
    return compound, category

# 4. SECTOR CLASSIFICATION MODULE
def classify_sector(text):
    """Rule-based multi-label classification."""
    text = text.lower()
    sectors = {
        'Banking': ['bank', 'loan', 'rbi', 'fed', 'interest', 'rate'],
        'Technology': ['ai', 'software', 'tech', 'cloud', 'cybersecurity'],
        'Energy': ['oil', 'gas', 'power', 'renewable', 'energy', 'green'],
        'FMCG': ['retail', 'consumer', 'goods', 'fmcg', 'sales'],
        'Stock Market': ['shares', 'market', 'stock', 'equity', 'nasdaq', 'nifty', 'indices']
    }
    
    matched_sectors = []
    for sector, keywords in sectors.items():
        if any(kw in text for kw in keywords):
            matched_sectors.append(sector)
            
    if not matched_sectors:
        matched_sectors.append("General")
        
    sector_impact = "Low"
    if "Stock Market" in matched_sectors or "Banking" in matched_sectors:
        sector_impact = "High"
    elif "Technology" in matched_sectors or "Energy" in matched_sectors:
        sector_impact = "Medium"
        
    return matched_sectors, sector_impact

# 5. GEMINI AI INSIGHT ENGINE
def gemini_batch_analysis(texts):
    """Prompts Gemini API for deep insights on multiple articles at once in a JSON array."""
    default_res = [{
        "summary": "Sample summary generated locally.",
        "emotion": "Neutral",
        "business_impact": "Requires Gemini API Key for detailed impact.",
        "risk_level": "Medium",
        "opportunity_level": "Medium",
        "recommendation": "Monitor"
    } for _ in texts]
    
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_KEY" or not texts:
        return default_res
        
    prompt = """
    Analyze the following financial news articles.
    Return ONLY a valid JSON ARRAY of objects, where each object corresponds to an article in the exact same order.
    Do not use markdown wrappers like ```json.
    
    Required keys for EACH object:
    - "summary": (1-line news summary)
    - "emotion": (Choose from: Fear, Optimism, Growth, Risk, Neutral)
    - "business_impact": (short explanation)
    - "risk_level": (Low, Medium, or High)
    - "opportunity_level": (Low, Medium, or High)
    - "recommendation": (Invest, Hold, Avoid, or Monitor)
    
    Articles:
    """
    for i, t in enumerate(texts):
        prompt += f"[{i}] {t}\n\n"
        
    try:
        response = model.generate_content(prompt)
        res_text = response.text.strip()
        if res_text.startswith("```json"):
            res_text = res_text[7:-3]
        elif res_text.startswith("```"):
            res_text = res_text[3:-3]
            
        parsed = json.loads(res_text)
        if isinstance(parsed, list) and len(parsed) == len(texts):
            return parsed
        else:
            default_res[0]["summary"] = "AI Error: Mismatched output length from Gemini."
            return default_res
    except Exception as e:
        for res in default_res:
            res["summary"] = f"AI Error: {str(e)}"
            res["business_impact"] = "Failed to parse API response."
        return default_res

# 6. MARKET IMPACT PREDICTION MODULE
def predict_market(sentiment_score, text):
    """Predicts market impact (Bullish/Bearish/Stable)."""
    text = text.lower()
    bullish_kw = ['growth', 'surge', 'rally', 'record', 'high', 'jump', 'gain', 'optimistic', 'boom']
    bearish_kw = ['crash', 'decline', 'plunge', 'drop', 'low', 'fear', 'recession', 'losses', 'plummet']
    
    bullish_count = sum(1 for kw in bullish_kw if kw in text)
    bearish_count = sum(1 for kw in bearish_kw if kw in text)
    
    if sentiment_score > 0.2 or bullish_count > bearish_count:
        return "Bullish 📈"
    elif sentiment_score < -0.2 or bearish_count > bullish_count:
        return "Bearish 📉"
    else:
        return "Stable ⚖️"

# 7. IMPACT SCORING MODULE
def calculate_impact(sentiment_score, sector_impact, text):
    """Calculates impact score and label."""
    base_score = abs(sentiment_score) * 10
    sector_weight = {"High": 1.5, "Medium": 1.0, "Low": 0.5}.get(sector_impact, 1.0)
    
    strong_kw = ['surge', 'crash', 'record', 'plunge', 'crisis', 'boom', 'massive']
    kw_count = sum(1 for kw in strong_kw if kw in text.lower())
    kw_multiplier = 1 + (0.1 * kw_count)
    
    final_score = base_score * sector_weight * kw_multiplier
    
    if final_score > 6:
        impact_label = "High Positive Impact" if sentiment_score > 0 else "High Negative Impact"
    elif final_score > 3:
        impact_label = "Moderate Impact"
    else:
        impact_label = "Low Impact"
        
    return impact_label, round(final_score, 2)

# Data Processing Wrapper
def process_data(df):
    if df.empty:
        return df
        
    results = []
    progress_bar = st.progress(0)
    total = len(df)
    
    # Send all texts to Gemini at once to avoid rate limits
    full_texts = [f"{row['title']}. {row['description']}" for _, row in df.iterrows()]
    ai_insights_batch = gemini_batch_analysis(full_texts)
    
    for i in range(total):
        row = df.iloc[i]
        full_text = full_texts[i]
        cleaned_text = preprocess_text(full_text)
        
        sent_score, sent_cat = analyze_sentiment(cleaned_text)
        sectors, sec_impact = classify_sector(cleaned_text)
        
        market_pred = predict_market(sent_score, cleaned_text)
        impact_label, impact_score = calculate_impact(sent_score, sec_impact, cleaned_text)
        
        # Safely grab the insight for this article
        ai_insights = ai_insights_batch[i] if i < len(ai_insights_batch) else ai_insights_batch[0]
        
        results.append({
            'title': row['title'],
            'cleaned_text': cleaned_text,
            'sentiment': sent_cat,
            'sentiment_score': sent_score,
            'sector': sectors,
            'sector_impact': sec_impact,
            'emotion': ai_insights.get('emotion', 'Neutral'),
            'summary': ai_insights.get('summary', row['description']),
            'business_impact': ai_insights.get('business_impact', 'N/A'),
            'risk_level': ai_insights.get('risk_level', 'Medium'),
            'opportunity_level': ai_insights.get('opportunity_level', 'Medium'),
            'recommendation': ai_insights.get('recommendation', 'Monitor'),
            'market_prediction': market_pred,
            'impact_score': impact_score,
            'impact_label': impact_label
        })
        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    return pd.DataFrame(results)

# --- STREAMLIT DASHBOARD UI ---

st.markdown("<h1 class='main-title'>💡 AI Financial Intelligence & Market Impact Predictor</h1>", unsafe_allow_html=True)
st.markdown("An advanced AI-powered dashboard that fetches, analyzes, and predicts financial news impact")
st.markdown("---")

# Sidebar
st.sidebar.header("🧭 Navigation & Filters")
if st.sidebar.button("🔄 Refresh News"):
    st.session_state['news_df'] = None
    st.rerun()

st.sidebar.markdown("---")

# Load & Process Data
if 'news_df' not in st.session_state or st.session_state['news_df'] is None:
    with st.spinner("Fetching latest financial news..."):
        raw_df = fetch_news()
        if not raw_df.empty:
            st.session_state['news_df'] = process_data(raw_df)
        else:
            st.session_state['news_df'] = pd.DataFrame()

df = st.session_state['news_df']

if not df.empty:
    # Sidebar Filters
    all_sectors = sorted(list(set([s for sublist in df['sector'] for s in sublist])))
    selected_sectors = st.sidebar.multiselect("Sector Filter", options=all_sectors, default=all_sectors)
    
    sentiments = sorted(list(set(df['sentiment'])))
    selected_sentiments = st.sidebar.multiselect("Sentiment Filter", options=sentiments, default=sentiments)
    
    impacts = sorted(list(set(df['impact_label'])))
    selected_impacts = st.sidebar.multiselect("Impact Filter", options=impacts, default=impacts)
    
    # Apply Filters
    mask = (
        df['sector'].apply(lambda x: any(s in selected_sectors for s in x)) &
        df['sentiment'].isin(selected_sentiments) &
        df['impact_label'].isin(selected_impacts)
    )
    filtered_df = df[mask]
    
    # 🔹 KPIs
    st.subheader("🔹 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    total_news = len(filtered_df)
    positive_news = len(filtered_df[filtered_df['sentiment_score'] > 0])
    pct_positive = (positive_news / total_news * 100) if total_news > 0 else 0
    
    sector_counts = {}
    for sectors in filtered_df['sector']:
        for s in sectors:
            if s in selected_sectors:
                sector_counts[s] = sector_counts.get(s, 0) + 1
    most_impacted = max(sector_counts, key=sector_counts.get) if sector_counts else "N/A"
    
    avg_impact = filtered_df['impact_score'].mean() if total_news > 0 else 0
    
    col1.metric("Total News", total_news)
    col2.metric("% Positive News", f"{pct_positive:.1f}%")
    col3.metric("Most Active Sector", most_impacted)
    col4.metric("Avg Impact Score", f"{avg_impact:.2f}")
    
    st.markdown("---")
    
    # 📊 VISUALIZATIONS
    st.subheader("📊 Trend Analysis")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        if sector_counts:
            fig = px.bar(
                x=list(sector_counts.keys()), 
                y=list(sector_counts.values()), 
                color=list(sector_counts.keys()),
                color_discrete_sequence=px.colors.qualitative.Pastel,
                labels={'x': 'Sector', 'y': 'Count'},
                title="📈 News Count per Sector"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                showlegend=False,
                font=dict(color="white")
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with chart_col2:
        sentiment_dist = filtered_df['sentiment'].value_counts()
        if not sentiment_dist.empty:
            fig = px.pie(
                names=sentiment_dist.index, 
                values=sentiment_dist.values, 
                hole=0.4, 
                title="🎭 Sentiment Distribution",
                color=sentiment_dist.index,
                color_discrete_map={
                    "Strong Positive": "#00CC96",
                    "Mild Positive": "#636EFA",
                    "Neutral": "#AB63FA",
                    "Mild Negative": "#FFA15A",
                    "Strong Negative": "#EF553B"
                }
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )
            fig.update_traces(textfont_color='white')
            st.plotly_chart(fig, use_container_width=True)
            
    st.markdown("---")
    
    # 🔍 NEWS EXPLORER
    st.subheader("📰 News Explorer & Insights")
    search_query = st.text_input("🔍 Search News by keywords...")
    
    if search_query:
        filtered_df = filtered_df[
            filtered_df['title'].str.contains(search_query, case=False, na=False) | 
            filtered_df['summary'].str.contains(search_query, case=False, na=False)
        ]
        
    for idx, row in filtered_df.iterrows():
        pred_color = "green" if "Bullish" in row['market_prediction'] else "red" if "Bearish" in row['market_prediction'] else "gray"
        
        with st.expander(f"📌 {row['title']} | {row['market_prediction']}"):
            cols = st.columns([2, 1])
            with cols[0]:
                st.markdown(f"**Sector:** {', '.join(row['sector'])}")
                st.markdown(f"**Sentiment:** {row['sentiment']} (Score: {row['sentiment_score']:.2f})")
                st.markdown(f"**Impact:** {row['impact_label']} (Score: {row['impact_score']})")
                st.markdown(f"**Recommendation:** `{row['recommendation']}`")
            with cols[1]:
                st.metric("Market Prediction", row['market_prediction'])
            
            st.markdown("### 🤖 AI Insight Panel")
            st.info(f"**Summary:** {row['summary']}")
            st.warning(f"**Business Impact:** {row['business_impact']}")
            
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Emotion:** {row['emotion']}")
            c2.write(f"**Risk Level:** {row['risk_level']}")
            c3.write(f"**Opportunity Level:** {row['opportunity_level']}")

else:
    st.info("No news data available to display. Try refreshing or updating filters.")
