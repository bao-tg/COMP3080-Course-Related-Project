import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct", ignore_patterns=["*.pth"])

model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import code; code.interact(local=locals())
