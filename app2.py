import streamlit as st
st.set_page_config(page_title="City Cost AI Agent", layout="wide")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

# --- CONFIGURATION & API SETUP ---
GROQ_API_KEY = "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"
DATA_PATH = "feature_engineered_us_cities.csv"

# --- DATA & MODEL ENGINE ---
@st.cache_data
def load_and_engineer_data():
    df = pd.read_csv(DATA_PATH)
    
    # Identify cost metrics (exclude metadata and state dummies)
    exclude = ['city', 'slug', 'lng', 'lat', 'rank', 'population', 'pop2026', 'pop2025', 
               'pop2020', 'pop2010', 'growth', 'densityMi', 'areaMi', 'pop_growth_2010_2020', 
               'pop_growth_2020_2025', 'density_calc', 'avg_cost_of_living', 'max_cost_metric']
    cost_cols = [c for c in df.columns if c not in exclude and not c.startswith('state_')]
    
    # Create deltas for features
    for col in cost_cols:
        df[f'delta_{col}'] = df[col] - df[col].mean()
    
    # Create Targets
    df['overall_cost_diff'] = df['avg_cost_of_living'] - df['avg_cost_of_living'].mean()
    df['max_cost_diff'] = df['max_cost_metric'] - df['max_cost_metric'].mean()
    
    return df, cost_cols

@st.cache_resource
def train_agent_model(data):
    X = data[[c for c in data.columns if c.startswith("delta_")]]
    y = data[["overall_cost_diff", "max_cost_diff"]]
    
    model = MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    ))
    model.fit(X, y)
    
    importances = model.estimators_[0].feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(5)
    return model, X.columns, feat_imp

# Initialize Data and Model
try:
    df, cost_metrics = load_and_engineer_data()
    model, feature_names, top_features = train_agent_model(df)
except Exception as e:
    st.error(f"Error loading data: {e}. Ensure '{DATA_PATH}' is in the folder.")
    st.stop()

# --- STREAMLIT UI SETUP ---
st.title("🏙️ City Analytics & Agentic Inference Engine")
st.markdown("---")

tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "💬 AI Agent Chat"])

# --- TAB 1: 10 CHARTS DASHBOARD ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.scatter(df, x="population", y="growth", size="densityMi", color="city", title="1. Growth vs Population (Density Bubble)"), use_container_width=True)
        
        fig2 = go.Figure(data=[
            go.Bar(name='City Centre', x=df['city'], y=df['Apartment (1 bedroom) in City Centre']),
            go.Bar(name='Outskirts', x=df['city'], y=df['Apartment (1 bedroom) Outside of Centre'])
        ])
        fig2.update_layout(title="2. Rent Comparison", barmode='group')
        st.plotly_chart(fig2, use_container_width=True)
        
        st.plotly_chart(px.scatter(df, x="Average Monthly Net Salary (After Tax)", y="avg_cost_of_living", trendline="ols", title="3. Purchasing Power Index"), use_container_width=True)
        st.plotly_chart(px.box(df, y="International Primary School, Yearly for 1 Child", title="4. Private Education Cost Variance"), use_container_width=True)
        st.plotly_chart(px.imshow(df[['population', 'avg_cost_of_living', 'growth', 'densityMi']].corr(), text_auto=True, title="5. Metric Correlation Heatmap"), use_container_width=True)

    with col2:
        st.plotly_chart(px.bar(df.nlargest(10, 'avg_cost_of_living'), x='avg_cost_of_living', y='city', orientation='h', title="6. Top 10 Most Expensive Cities"), use_container_width=True)
        
        df['Grocery_Basket'] = df['Apples (1kg)'] + df['Banana (1kg)'] + df['Eggs (regular) (12)']
        st.plotly_chart(px.violin(df, y="Grocery_Basket", box=True, title="7. Basic Grocery Basket Costs"), use_container_width=True)
        st.plotly_chart(px.scatter(df, x="Gasoline (1 liter)", y="Monthly Pass (Regular Price)", color="city", title="8. Transport: Gas vs Transit Pass"), use_container_width=True)
        st.plotly_chart(px.histogram(df, x="overall_cost_diff", title="9. Model Target: Cost Deviations"), use_container_width=True)
        st.plotly_chart(px.scatter_mapbox(df, lat="lat", lon="lng", color="avg_cost_of_living", size="population", mapbox_style="carto-positron", zoom=3, title="10. Geographic Cost Mapping"), use_container_width=True)

# --- TAB 2: GROQ CHATBOT AGENT ---
with tab2:
    st.header("🤖 City Intelligence Agent")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ===== CHAT INPUT (TOP LEVEL ONLY) =====
prompt = st.chat_input("Ask me about city economics...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    context = df[['city', 'avg_cost_of_living', 'population']].head(10).to_string()
    agent_prompt = (
        f"Data Context:\n{context}\n\n"
        f"Top Features for Model:\n{top_features.to_string()}\n\n"
        f"User: {prompt}"
    )

    client = Groq(api_key=GROQ_API_KEY)
    full_response = ""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": agent_prompt}],
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        stream=True
    )

    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        full_response += content

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

    st.rerun()
