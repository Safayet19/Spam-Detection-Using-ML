import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # for converting text to numeric value
from sklearn.naive_bayes import MultinomialNB
import streamlit as st  # Streamlit: turns Python code into a web app

# Load data
data = pd.read_csv("spam.csv")

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Replace 'ham' with 'Not Spam' and 'spam' with 'Spam' in the 'Category' column
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Define input and output
X = data['Message']  # input text data
y = data['Category']  # target output

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit on training data and transform both train and test data
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Create the Naive Bayes model
model = MultinomialNB()

# Train the model on training data
model.fit(X_train_counts, y_train)

# Streamlit app UI
st.set_page_config(page_title="Spam Detector", page_icon="🔍")
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #1f77b4;'>💬 Spam Detector App</h1>
        <h4 style='color: grey;'>🔐 Detect whether your message is <span style='color:green'>Not Spam</span> or <span style='color:red'>Spam</span></h4>
    </div>
    <hr style='border: 1px solid #f0f0f0;'/>
    """, unsafe_allow_html=True
)

# Predict function
def predict(message):
    input_message = vectorizer.transform([message])
    result = model.predict(input_message)
    return result

# Input box
user_input = st.text_input("📥 Type your suspicious message below:")

# Check button
if st.button("🚀 Check Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to check.")
    else:
        prediction = predict(user_input)
        if prediction[0] == "Spam":
            st.error(f"📛 Prediction: **{prediction[0]}** ❌")
        else:
            st.success(f"✅ Prediction: **{prediction[0]}** 🎉")
