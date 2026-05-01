# AI Financial Intelligence & Market Impact Predictor

## 🎯 1. Problem Statement
In today's fast-paced financial ecosystem, traders, analysts, and investors are overwhelmed by the sheer volume of news articles, reports, and global events published every second. Manually reading and interpreting how a specific news event might impact a particular market sector is virtually impossible. This creates an information bottleneck where critical market-moving insights are missed or analyzed too late, leading to suboptimal investment decisions or unmitigated risks.

## 🚀 2. Project Objective
The objective of this project is to build an automated, intelligent, and highly responsive dashboard that fetches real-time financial news, processes it using Natural Language Processing (NLP), and leverages advanced Generative AI to instantly predict market impacts. It aims to transform unstructured text into actionable intelligence, empowering users to make data-driven financial decisions rapidly.

## ⚙️ 3. Tech Stack
- **Frontend & UI Framework:** Streamlit (with custom CSS/Glassmorphism design)
- **Data Visualization:** Plotly Express & Plotly Graph Objects
- **Data Manipulation:** Pandas, NumPy
- **Natural Language Processing (NLP):** NLTK (VADER Sentiment Analysis, Tokenization, Stopwords)
- **Generative AI:** Google Gemini Pro (`gemini-2.5-flash`)
- **External Data Source:** NewsAPI

## 🔄 4. System Workflow
1. **Data Ingestion:** The app triggers an HTTP request to **NewsAPI** to fetch the latest global articles related to finance, tech, banking, and the stock market.
2. **Text Preprocessing:** The raw text (titles and descriptions) is cleaned using **NLTK**. Punctuation is stripped, text is lowercased, and English stopwords are removed to reduce noise.
3. **Sentiment & Sector Analysis:** 
   - **VADER** analyzes the text to generate a raw sentiment compound score and categorizes it (e.g., "Strong Positive", "Mild Negative").
   - A rule-based classification engine maps keywords to specific sectors (Banking, Tech, Energy, FMCG, Stock Market).
4. **AI Deep Insight Generation:** The cleaned texts are sent in a batched prompt to **Google Gemini**. Gemini acts as an expert financial analyst, returning structured JSON containing a concise summary, underlying emotion, business impact, risk/opportunity levels, and an investment recommendation (Invest/Hold/Avoid/Monitor).
5. **Market Impact Calculation:** The app calculates an impact score combining the sentiment score, the weight of the affected sector, and the density of market-moving keywords (e.g., "surge", "crash", "plunge").
6. **UI Rendering:** The processed DataFrame is pushed to the **Streamlit** frontend. The data is visually broken down into KPIs, interactive Plotly charts, and an expandable "News Explorer".

## ✨ 5. Key Features
- **Real-Time News Fetching:** Live data stream ensures the dashboard reflects current market conditions.
- **Multidimensional Filtering:** Users can dynamically filter the dashboard by specific Sectors, Sentiments, and Impact Levels.
- **AI-Powered Deep Summaries:** Bypasses clickbait by providing a 1-line AI summary alongside the business context and emotional tone.
- **Automated Market Prediction:** Tags articles as "Bullish 📈", "Bearish 📉", or "Stable ⚖️" based on sentiment and financial terminology density.
- **Interactive Visualizations:** Premium Plotly donut charts and bar charts provide a bird's-eye view of market trends and sentiment distribution.

## 📊 6. Results and Insights
By using this tool, an analyst can immediately deduce:
- **Which sector is currently the most active** (e.g., if a new AI chip is announced, "Technology" dominates the trend chart).
- **The general market mood** (e.g., identifying if the market is fearful or optimistic based on the Sentiment Distribution pie chart).
- **High-Risk Events:** Filtering by "High Negative Impact" immediately isolates the 1 or 2 articles that require urgent attention (e.g., central bank rate hikes), ignoring the noise of neutral daily updates.

## 🔮 7. Future Scope
- **Live Stock Price Integration:** Connecting the dashboard to a financial API like `yfinance` to overlay actual stock price charts next to the news events, proving the AI's impact prediction.
- **User Portfolios:** Allowing users to input their specific stock portfolio so the news feed only fetches and analyzes articles directly impacting their holdings.
- **Historical Backtesting:** Storing daily AI predictions in a database to measure how accurate the Gemini API's "Bullish" or "Bearish" tags were against actual historical market movements.
- **Alert System:** Adding email or SMS webhooks to automatically notify the user when a "Strong Negative" or "High Impact" news article drops for a selected sector.
