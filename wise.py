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
    "Badam", "Apples", "Asparagus", "Avocado", "Bacon",
    "Bag", "Bagels", "Backpack", "Batteries", "Basketball",
    "Beet", "Belt", "Bike", "Blanket", "Blender",
    "Blueberries", "Bottle", "Bracelet", "Bread", "Broccoli",
    "Broom", "Brownies", "Burger", "Butter", "Cabbage",
    "Camera", "Carrots", "Calendar", "Cap", "Cereal",
    "Celery", "Chair", "Charger", "Cheese", "Chicken",
    "Chickpeas", "Chips", "Chocolate", "Clock", "Coconut",
    "Coffee", "Cookies", "Conditioner", "Coke", "Cornflakes",
    "Cupcake", "Dates", "Deodorant", "Detergent", "Doughnut",
    "Drill", "Earrings", "Eggplant", "Eggs", "Fan",
    "Fish", "Flashlight", "Football", "Fork", "Freezer",
    "Garlic", "Glasses", "Glue", "Gloves", "Grapes",
    "Guitar", "Hammer", "Hat", "Headphones", "Helmet",
    "Honey", "Hotdog", "Jam", "Juice", "Ketchup",
    "Knife", "Lamp", "Laptop", "Lemon", "Lightbulb",
    "Lime", "Lotion", "Mangoes", "Marker", "Mask",
    "Mayonnaise", "Milk", "Mop", "Mug", "Muffin",
    "Mushrooms", "Mustard", "Nail Clippers", "Necklace", "Notebook",
    "Noodles", "Olives", "Onion", "Oranges", "Oven",
    "Oysters", "Papaya", "Pancakes", "Peaches", "Peanuts",
    "Pears", "Pen", "Pencil", "Perfume", "Pineapple",
    "Pizza", "Plant", "Plate", "Plums", "Popcorn",
    "Potatoes", "Quinoa", "Radish", "Razor", "Ring",
    "Rice", "Ruler", "Salmon", "Salt", "Scarf",
    "Scissors", "Shrimp", "Shoes", "Shampoo", "Soap",
    "Soda", "Socks", "Spoon", "Spinach", "Stove",
    "Suitcase", "Sunscreen", "Sugar", "Table", "Tape",
    "Tea", "Tent", "Tissues", "Tomatoes", "Toaster",
    "Toothbrush", "Toothpaste", "Tripod", "Tuna", "Turkey",
    "Umbrella", "Violin", "Vase", "Wallet", "Watch",
    "Water", "Wrench", "Yogurt", "Zucchini"
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
    
    # Option to show previous chat history
    if st.button("Show Previous Chat History"):
        st.subheader("Chat History")
        for speaker, message in st.session_state.chat_history:
            st.write(f"**{speaker.capitalize()}:** {message}")
    
    for speaker, message in st.session_state.chat_history:
        with st.chat_message(speaker):
            st.write(message)
    
    user_input = st.chat_input("Ask me anything about food / products!")
    
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.write(user_input)
        
        # Include previous chat history in the prompt
        previous_chat = "\n".join([f"{speaker}: {message}" for speaker, message in st.session_state.chat_history])
        prompt = f"{previous_chat}\nAssistant:"
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_content(prompt, max_tokens=1500)
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
        color: #2e3a24;  /* Dark green text */
        background-color: #e8f5e9;  /* Light green background */
    }
    .stApp {
        background-color: #e8f5e9;
    }
    
    /* Text styles */
    p, li {
        color: #2e3a24;  /* Dark green text */
        line-height: 1.6;
    }
                
    /* Sidebar styles */
    [data-testid="stSidebar"] {
        background-color: #4caf50;  /* Green sidebar */
    }
    [data-testid="stSidebar"] .sidebar-content {
        background-color: #4caf50;  /* Green sidebar */
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #ffffff !important;  /* White text in sidebar */
    }
    
    /* Header styles */
    h1 {
        color: #1b5e20;  /* Darker green for headers */
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    h2, h3 {
        color: #1b5e20;  /* Darker green for subheaders */
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Info box styles */
    .stInfo {
        background-color: #c8e6c9;  /* Light green info box */
        color: #1b5e20;  /* Dark green text */
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #a5d6a7;  /* Slightly darker green border */
        margin-bottom: 1rem;
    }
    
    /* Button styles */
    .stButton>button {
        width: 100%;
        background-color: #388e3c;  /* Green button */
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2e7d32;  /* Darker green on hover */
    }
    
    /* Input field styles */
    .stTextInput>div>div>input {
        background-color: #ffffff;  /* White input fields */
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
        background-color: #ffebee;  /* Light red for errors */
        color: #c62828;  /* Dark red text */
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