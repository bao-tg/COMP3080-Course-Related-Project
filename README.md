# Chatbot Demonstration

## 1. Introduction

This repository contains the files for a demo chatbot application that can be run using Streamlit.

### How to Install and Run
- Install necessary packages using the command: 
pip install -r requirements.txt
- To run the chatbot, use the following command:
streamlit run chatbot.py
## 2. Fine-tuning Folder

This folder contains the scripts and data necessary for fine-tuning the chatbot model.

### Contents
- `chatbot_finetuning.py`: The user interface for interacting with the chatbot.
- `train.py`: Contains the fine-tuning logic for the chatbot model.
- `corpus.jsonl`: The dataset used for fine-tuning the model.

## 3. Remarks

- The fine-tuned model is available but not included in this repository due to its size and conflicts with the LFS package.
- The user interface is currently under maintenance.

## 4. References

1. Dataset used for fine-tuning: [Hugging Face Dataset](https://huggingface.co/datasets/ArtifactAI/arxiv-beir-cs-ml-generated-queries/tree/main)
2. Streamlit demo tutorial: [How to Build a Chatbot with Streamlit](https://bl
