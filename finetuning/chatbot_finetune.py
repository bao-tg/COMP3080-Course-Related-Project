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

