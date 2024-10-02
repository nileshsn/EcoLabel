import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
import requests

# Load environment variables
load_dotenv()

# Set up API key from .env file
groq_api_key = os.getenv("GROQ_API_KEY")
pixabay_api_key = os.getenv("PIXABAY_API_KEY")

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# Load the dataset
df = pd.read_csv('ProductStates.csv')

# Normalize product names to lower case
product_names = [
    "Almonds", "Apples", "Avocados", "Bananas", "BlackTea",
    "Broccoli", "Butter", "Carrots", "Cashews", "Cauliflower",
    "Cheese", "Chickpeas", "Coffee", "Coke", "Conditioner", "Cornflakes",
    "Cucumbers", "Deodorant", "FaceWash", "Garlic", "Grapes", "GreenTea", "IcedTea",
    "KidneyBeans", "Lotion", "Mangoes", "Milk", "Oatmeal", "Onion",
    "Oranges", "Pineapple", "Popcorn", "PotatoChips", "Potatoes",
    "Rice", "Shampoo", "Soda", "Spinach", "Sprite", "Sunscreen", 
    "Tea", "Tomatoes", "Toothpaste", "Wheat", "Yogurt", "Blueberries", 
    "Eggs"
]

# Function to fetch image from Pixabay
def fetch_image_from_pixabay(query):
    url = f"https://pixabay.com/api/?key={pixabay_api_key}&q={query}&image_type=photo&per_page=3"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['hits']:
            return data['hits'][0]['webformatURL']
    return None

# Function to display an image for a product
def show_product_image(product_name):
    """Shows an image for the specified product."""
    image_url = fetch_image_from_pixabay(product_name)
    if image_url:
        st.image(image_url, caption=product_name, use_column_width=True)
    else:
        placeholder_url = f"https://via.placeholder.com/300x200.png?text={product_name}"
        st.image(placeholder_url, caption=f"No image found for {product_name}", use_column_width=True)

# Function to generate content using Groq API
def generate_content(prompt, max_tokens=100):
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=max_tokens
        )
        return response
    except Exception as e:
        st.error(f"Error while connecting to the API: {e}")
        return None

# Function to create a pie chart for product's state distribution
def create_pie_chart(product_name):
    """Creates a pie chart of the state distribution for a given product."""
    product_data = df[df['ProductName'].str.lower() == product_name.lower()]
    state_distribution = {}
    
    for _, row in product_data.iterrows():
        for i in range(1, 5):
            state_column = f"State{i}"
            percentage_column = f"Percentage{i}"
            state = row[state_column]
            percentage = row[percentage_column]
            state_distribution[state] = state_distribution.get(state, 0) + float(percentage[:-1]) / 100

    fig = go.Figure(data=[go.Pie(labels=list(state_distribution.keys()), values=list(state_distribution.values()))])
    fig.update_layout(title=f"State Distribution for {product_name.capitalize()}")
    st.plotly_chart(fig, use_container_width=True)

# Function to create a bar chart for product comparison
def create_bar_chart(product_names):
    """Creates a bar chart comparing the state distribution of two products."""
    state_distributions = {product: {} for product in product_names}

    for product_name in product_names:
        product_data = df[df['ProductName'].str.lower() == product_name.lower()]
        
        if product_data.empty:
            st.error(f"No data found for {product_name}")
            return

        for _, row in product_data.iterrows():
            for i in range(1, 5):
                state_column = f"State{i}"
                percentage_column = f"Percentage{i}"
                state = row[state_column]
                percentage = row[percentage_column]
                if state and percentage:
                    state_distributions[product_name][state] = float(percentage.rstrip('%')) / 100

    # Get all unique states
    all_states = sorted(set(state for product in state_distributions.values() for state in product.keys()))

    # Prepare data for plotting
    data = []
    for product_name in product_names:
        values = [state_distributions[product_name].get(state, 0) for state in all_states]
        data.append(go.Bar(name=product_name.capitalize(), x=all_states, y=values))

    # Create the figure
    fig = go.Figure(data=data)
    fig.update_layout(
        title=f"State Distribution Comparison: {product_names[0].capitalize()} vs {product_names[1].capitalize()}",
        xaxis_title="States",
        yaxis_title="Percentage",
        barmode='group',
        yaxis=dict(tickformat='.0%')
    )
    st.plotly_chart(fig, use_container_width=True)

    # Generate descriptions for both products
    for product_name in product_names:
        description_prompt = f"Generate a detailed description for {product_name}."
        description = generate_content(description_prompt, max_tokens=1500)
        if description:
            st.write(f"**Description for {product_name.capitalize()}:** {description.choices[0].message.content}")
        else:
            st.write(f"**Description for {product_name.capitalize()}:** Unable to generate description.")

# Chatbot functionality
def chat_with_bot():
    st.title("Chat with AI Assistant")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for speaker, message in st.session_state.chat_history:
        with st.chat_message(speaker):
            st.write(message)

    user_input = st.chat_input("Ask me anything about food products!")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_content(user_input, max_tokens=150)
                if response:
                    st.write(response.choices[0].message.content)
                    st.session_state.chat_history.append(("assistant", response.choices[0].message.content))
                else:
                    st.write("Sorry, I couldn't generate a response.")
                    st.session_state.chat_history.append(("assistant", "Sorry, I couldn't generate a response."))

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()  

# Streamlit UI for user input
# Update the show_product_info function
def show_product_info():
    st.title("Product Information")
    
    option = st.radio("Select an option:", ("View a single product", "Compare two products"))

    if option == "View a single product":
        product_name = st.selectbox("Select a product:", product_names)
        
        if product_name:
            col1, col2 = st.columns([1, 2])
            with col1:
                show_product_image(product_name)
            with col2:
                st.subheader(f"{product_name.capitalize()} Information")
                create_pie_chart(product_name.lower())
            
            st.markdown("### Product Description")
            with st.spinner("Generating description..."):
                prompt = f"Generate a detailed description for {product_name}. Provide more in-depth information."
                output = generate_content(prompt, max_tokens=1500)
                if output:
                    st.write(output.choices[0].message.content)

    elif option == "Compare two products":
        col1, col2 = st.columns(2)
        with col1:
            product_name1 = st.selectbox("Select first product:", product_names)
        with col2:
            product_name2 = st.selectbox("Select second product:", product_names)

        if product_name1 and product_name2:
            create_bar_chart([product_name1, product_name2])

def show_home():
    st.image("img.jpeg", width=200)
    st.write("""
        Welcome to **EcoLabel**, a platform that provides clear and personalized insights into the health, environmental, and societal impacts of everyday products. By verifying sustainability metrics and origins, EcoLabel helps you make informed choices that align with your values.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 View product statistics")
    with col2:
        st.info("🔍 Compare different products")
    with col3:
        st.info("💬 Chat with our AI assistant")

    st.markdown("### Featured Products")
    featured_products = random.sample(product_names, 4)
    
    cols = st.columns(4)
    for i, product in enumerate(featured_products):
        with cols[i]:
            show_product_image(product)

def main():
    st.set_page_config(page_title="EcoLabel", page_icon="🌱", layout="wide")

    st.markdown("""
    <style>
    /* Global styles */
    body {
        color: #333333;
        background-color: #e6e9ef;  /* Slightly darker background */
    }
    .stApp {
        background-color: #e6e9ef;
    }
    
    /* Content area styles */
    # .stMarkdown, .stText {
    #     background-color: #ffffff;
    #     padding: 1rem;
    #     border-radius: 0.5rem;
    #     box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    #     margin-bottom: 1rem;
    # }
    
    /* Text styles */
    p, li {
        color: #1f2937;
        line-height: 1.6;
    }
                
    # h1, h2, h3, h4, h5, h6 {
    #     color: #1e3a8a;
    # }
                
    /* Sidebar styles */
    [data-testid="stSidebar"] {
        background-color: #1b1d21;
    }
    [data-testid="stSidebar"] .sidebar-content {
        background-color: #1b1d21;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        color: #ffffff !important;
    }
    
    /* Header styles */
    h1 {
        color: #0c339f;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    h2, h3 {
        color: #0c339f;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Paragraph styles */
    p {
        color: #1f2937;
        line-height: 1.6;
    }
    
    /* Info box styles */
    .stInfo {
        background-color: #e0f2fe;
        color: #0c4a6e;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #7dd3fc;
        margin-bottom: 1rem;
    }
    
    /* Button styles */
    .stButton>button {
        width: 100%;
        background-color: #1e40af;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1e3a8a;
    }
    
    /* Input field styles */
    .stTextInput>div>div>input {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 0.25rem;
    }
    
    /* Image styles */
    .stImage > img {
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Error message styles */
    .stError {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.875rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Product Info", "Chat"])

    if page == "Home":
        show_home()
    elif page == "Product Info":
        show_product_info()
    elif page == "Chat":
        chat_with_bot()

if __name__ == "__main__":
    main()