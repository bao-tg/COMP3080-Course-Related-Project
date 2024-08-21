import json
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load the corpus.jsonl file
corpus = []
with open('corpus.jsonl', 'r') as f:
    for line in f:
        corpus.append(json.loads(line))

# Create a list of texts from the corpus
texts = [doc['text'] for doc in corpus]

# Limit the number of texts for faster training
texts = texts[:1000]

# Create a dataset from the list of texts
dataset = Dataset.from_dict({"text": texts})

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function with labels
def tokenize_function(examples):
    # Tokenize the text
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True)
    # Use input IDs as labels (shifted)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and evaluation sets
train_size = int(0.8 * len(tokenized_datasets))
train_dataset = tokenized_datasets.select(range(train_size))
eval_dataset = tokenized_datasets.select(range(train_size, len(tokenized_datasets)))

# Set up training arguments for distributed training
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,  # Batch size per GPU
    save_steps=10_000,
    save_total_limit=2,
    gradient_accumulation_steps=2,  # Accumulate gradients to reduce memory footprint
    fp16=True,  # Enable mixed precision training for faster computations
    eval_strategy="epoch",  # Updated from evaluation_strategy
    logging_dir="./logs",
    logging_steps=500,
    report_to="tensorboard",  # Enable TensorBoard logging
    dataloader_num_workers=4,  # Number of data loading workers
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
