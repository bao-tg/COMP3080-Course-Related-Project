import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import HfApi, HfFolder

# Store model in your repository, put False if you don't want to store your model
repository_id = ""
store_model = False

# App title
st.set_page_config(page_title="🤖💬 VinUni's CS chatbot")

# Load the model and tokenizer
@st.cache_resource
def load_model(model_name):
    if model_name == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer
# Sidebar settings

with st.sidebar:
    st.title("🤖💬 VinUni's CS chatbot")
    st.subheader('Model selection')
    model_name = st.selectbox('Choose a model', ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'meta-llama/Meta-Llama-3.1-8B-Instruct'])
    model, tokenizer = load_model(model_name)
    
    st.subheader('Model parameters')
    temperature = st.slider('Temperature', min_value=0.01, max_value=1.0, value=0.7, step=0.01)
    max_length = st.slider('Max Length', min_value=32, max_value=128, value=120, step=8)
    top_k = st.slider('Top-k', min_value=1, max_value=50, value=50, step=1)
    top_p = st.slider('Top-p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Generate a response using GPT-2
def generate_gpt2_response(prompt_input):
    inputs = tokenizer.encode(prompt_input + tokenizer.eos_token, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,    
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if store_model:
    # Use HfApi to create the repository (optional, you can also do this on the website)
    api = HfApi()
    api.create_repo(repo_id=repository_id)

    # Push the model and tokenizer to the Hub
    model.push_to_hub(repository_id)
    tokenizer.push_to_hub(repository_id)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            prompt_input = f"{prompt}\nAssistant:"
            response = generate_gpt2_response(prompt_input)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
