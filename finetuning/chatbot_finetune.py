import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')

st.title("Fine-Tuned GPT-2 Text Generator")

# Input for user prompt
user_input = st.text_area("Enter your prompt:", "Once upon a time")

# Generate text on button click
if st.button("Generate"):
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    st.write("Generated Text:")
    st.write(generated_text)


import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')

# Set up the page title and icon
st.set_page_config(page_title="GPT-2 Text Generator", page_icon="📝", layout="wide")

# Add a custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    footer {
        visibility: hidden;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        margin: 10px;
        border: none;
        cursor: pointer;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Set up the header of the app
st.title("GPT-2 Text Generator")
st.write("Fine-tuned GPT-2 Model for Text Generation. Enter your prompt and generate creative text!")

# User input for text generation
user_input = st.text_area("Enter your prompt:", "Once upon a time", height=100)

# Slider for controlling text generation length
length = st.slider("Select the maximum length of the generated text:", min_value=50, max_value=300, value=100)

# Generate text on button click
if st.button("Generate Text"):
    with st.spinner("Generating text..."):
        inputs = tokenizer.encode(user_input, return_tensors="pt")
        outputs = model.generate(inputs, max_length=length, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.subheader("Generated Text:")
    st.write(generated_text)

# Additional customization for footer or side notes
st.sidebar.header("About")
st.sidebar.info("This app demonstrates text generation using a fine-tuned GPT-2 model. Adjust the length slider to control the length of the generated output.")

st.sidebar.subheader("Fine-Tuned Model Info")
st.sidebar.write("This model was fine-tuned on a custom corpus using the GPT-2 architecture.")

# Add some social media icons or a footer message if needed
st.sidebar.write("Developed with ❤️ using Streamlit and Transformers.")
