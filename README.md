# Chatbot for Computer Science Students at VinUni

This project develops a specialized chatbot to assist Computer Science students at VinUni. The chatbot leverages fine-tuned language models to provide accurate and useful responses to various academic queries.

## Getting Started

These instructions will help you set up a copy of the project on your local machine for development and testing purposes.

### Prerequisites

- Git
- Python 3.6+
- pip

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/bao-tg/COMP3080-Course-Related-Project
2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
3. **Navigate to the Main Directory**
   ```bash
   cd main
4. **Run the chatbot**
   ```bash
   streamlit run chatbot.py

### Finetuning the model
To fine-tune the model with custom datasets:
1. **Navigate to the Main Directory**
   ```bash
   cd main
2. **Run the Fine-tuning script**
   ```bash
   torchrun --nproc_per_node=6 --master_port=29500 finetuning.py

### Load the fine-tuned model
To use your fine-tuned model in the chatbot:
+ Modify the *`chatbot.py`* file and change the *`modelname`* variable to the path of your fine-tuned model.

### Using the Llama-8b Model
1. **Get Access Permission**
Request access at Meta-Llama 3.1-8B Instruct.
2. **Login to Hugging Face Hub**
   Use your Hugging Face key to authenticate
   ```bash
   huggingface-cli login
3. **Run the Chatbot with the Llama-8b Model**
Select the Llama-8b model in the interface after running chatbot.py.

## Built With

* [Streamlit](https://streamlit.app/) - UI and Deployment
* [Hugging Face](https://huggingface.co/) - LLMs Model Handling

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


