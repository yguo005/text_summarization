# In this assignment, you will compare several pre-trained models from the Hugging Face
# model hub on text summarization and then fine-tune one of them on a custom dataset. The
# dataset you will use for this task is the SAMSum developed by Samsung, which contains
# about 16,000 messenger-like dialogues and their corresponding summaries.

# This script will guide you through the steps.

# 0. Installation
# Make sure you have the required libraries installed.
# You can install them using pip:
# pip install transformers datasets torch matplotlib wordcloud nltk evaluate rouge_score

import torch
import nltk
from datasets import load_dataset
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# 1. Read the Hugging Face summarization task guide.
# The guide provides a comprehensive overview of how to use transformers for summarization.
# You can find it here: https://huggingface.co/docs/transformers/tasks/summarization

# 2. Exploratory Data Analysis (EDA)

# (a) Conduct an initial exploration of the SAMSum dataset.
print("Loading SAMSum dataset...")
samsum_dataset = load_dataset("samsum")
print("Dataset loaded.")

print("\n--- Initial Exploration ---")
print(samsum_dataset)

# Display a few examples from the training set
print("\n--- Sample Dialogues and Summaries ---")
for i in range(2):
    print(f"\n--- Example {i+1} ---")
    print(f"ID: {samsum_dataset['train'][i]['id']}")
    print(f"Dialogue:\n{samsum_dataset['train'][i]['dialogue']}")
    print(f"\nSummary:\n{samsum_dataset['train'][i]['summary']}")

# (b) Plot the length distribution of dialogues and summaries in the training set.
train_dialogues = samsum_dataset['train']['dialogue']
train_summaries = samsum_dataset['train']['summary']

dialogue_lengths = [len(dialogue.split()) for dialogue in train_dialogues]
summary_lengths = [len(summary.split()) for summary in train_summaries]

def plot_length_distribution(lengths, title):
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, alpha=0.75)
    plt.title(title)
    plt.xlabel("Length (number of words)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

print("\nPlotting length distributions...")
plot_length_distribution(dialogue_lengths, "Distribution of Dialogue Lengths in SAMSum Training Set")
plot_length_distribution(summary_lengths, "Distribution of Summary Lengths in SAMSum Training Set")
print("Plots displayed.")

# (c) Display the 20 most common words in the dialogues and their frequencies.
print("\n--- Top 20 Most Common Words in Dialogues ---")
all_dialogues_text = " ".join(train_dialogues)

# Tokenize, remove stopwords and non-alphabetic characters
words = re.findall(r'\b\w+\b', all_dialogues_text.lower())
filtered_words = [word for word in words if word not in stop_words and word.isalpha()]

word_counts = Counter(filtered_words)
most_common_words = word_counts.most_common(20)

print("Top 20 most common words and their frequencies:")
for word, freq in most_common_words:
    print(f"{word}: {freq}")

# Optional: Generate and display a word cloud
print("\nGenerating word cloud for dialogues...")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Dialogues in SAMSum Training Set")
plt.show()
print("Word cloud displayed.")


# 3. Using pre-trained models with Hugging Face pipeline
print("\n--- Text Summarization using Pre-trained Models ---")

# We will use the 'summarization' pipeline from Hugging Face.
# It simplifies the process of using models for this task.

# Let's pick a sample dialogue to summarize
sample_dialogue = samsum_dataset['test'][0]['dialogue']
reference_summary = samsum_dataset['test'][0]['summary']

print("\n--- Sample for Summarization ---")
print(f"Dialogue:\n{sample_dialogue}")
print(f"\nReference Summary:\n{reference_summary}")

# Model 1: 't5-small'
print("\n--- Summarizing with 't5-small' ---")
# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1
summarizer_t5 = pipeline("summarization", model="t5-small", device=device)
t5_summary = summarizer_t5(sample_dialogue, max_length=50, min_length=10, do_sample=False)
print("\nGenerated Summary (t5-small):")
print(t5_summary[0]['summary_text'])

# Model 2: 'sshleifer/distilbart-cnn-12-6' (a popular and efficient model for summarization)
print("\n--- Summarizing with 'sshleifer/distilbart-cnn-12-6' ---")
summarizer_bart = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
bart_summary = summarizer_bart(sample_dialogue, max_length=50, min_length=10, do_sample=False)
print("\nGenerated Summary (distilbart-cnn-12-6):")
print(bart_summary[0]['summary_text'])

# 4. Fine-tuning a model on SAMSum (Roadmap)
# Below is a commented-out example of how you would set up to fine-tune a model.
# This part is for your reference to complete the assignment.

"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import numpy as np
import evaluate

# 1. Load tokenizer and model
model_checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# 2. Preprocess the dataset
max_input_length = 1024
max_target_length = 128
prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = samsum_dataset.map(preprocess_function, batched=True)

# 3. Set up training arguments
batch_size = 4 # Adjust based on your GPU memory
args = Seq2SeqTrainingArguments(
    output_dir="samsum-t5-small",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3, # Start with a small number of epochs
    predict_with_generate=True,
    fp16=torch.cuda.is_available(), # Use mixed precision if a GPU is available
    push_to_hub=False, # Set to True to upload your model to Hugging Face Hub
)

# 4. Define metrics
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}
    
    # Add generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

# 5. Create Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 6. Train the model
# To start training, uncomment the following line:
# trainer.train()
"""

print("\n\nScript finished. You have now explored the SAMSum dataset and used pre-trained models for summarization.")
print("The script also provides a commented-out section to guide you in fine-tuning your own model.")
