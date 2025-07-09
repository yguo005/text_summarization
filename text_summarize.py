"""
Text Summarization using Hugging Face Transformers
============================================================

This assignment focuses on text summarization using the SAMSum dataset,
which contains messenger-like conversations with human-written summaries.

Goals:
1. Explore the SAMSum dataset characteristics
2. Compare multiple pre-trained summarization models
3. Fine-tune a model on the dataset
4. Evaluate and compare performance

Dataset: https://huggingface.co/datasets/knkarthick/samsum

"""


# Install required packages
!pip install transformers>=4.21.0
!pip install datasets>=2.0.0
!pip install torch>=1.12.0
!pip install evaluate>=0.4.0
!pip install rouge-score>=0.1.2
!pip install nltk>=3.7
!pip install accelerate>=0.20.0
!pip install sentencepiece>=0.1.97
!pip install protobuf>=3.20.0
!pip install --upgrade fsspec>=2023.1.0

# Download NLTK data: A pre-trained model that helps NLTK tokenize text into sentences. The ROUGE metric, which is used to evaluate the quality of a summary, works best when it compares summaries sentence by sentence.
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

print("All packages installed successfully!")


import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import gc

# Hugging Face
from datasets import Dataset, DatasetDict
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

"""
A comprehensive analyzer for the SAMSum dataset and text summarization models.
"""
class SAMSumSummarizer:
    
    def __init__(self):
        """Initialize the analyzer and load the dataset from CSV files."""
        print("Loading SAMSum dataset from local CSV files...")
        
        # Load CSV files from the same directory as the script
        try:
            # Load the CSV files
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')
            validation_df = pd.read_csv('validation.csv')
            
            # Convert pandas DataFrames to Hugging Face Dataset format
            train_dataset = Dataset.from_pandas(train_df)
            test_dataset = Dataset.from_pandas(test_df)
            validation_dataset = Dataset.from_pandas(validation_df)
            
            # Create DatasetDict
            self.dataset = DatasetDict({
                'train': train_dataset,
                'test': test_dataset,
                'validation': validation_dataset
            })
            
            print(f"Successfully loaded dataset from CSV files!")
            print(f"   Training samples: {len(self.dataset['train'])}")
            print(f"   Validation samples: {len(self.dataset['validation'])}")
            print(f"   Test samples: {len(self.dataset['test'])}")
            
        except FileNotFoundError as e:
            print(f"Error: Could not find CSV files in the current directory.")
            print(f"   Error details: {e}")
            raise
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
        self.stop_words = set(stopwords.words('english'))
        print(f"Dataset structure: {self.dataset}")
        
        # Display sample data to verify format
        print(f"Sample data format:")
        sample = self.dataset['train'][0]
        print(f"   Columns: {list(sample.keys())}")
        if 'dialogue' in sample and 'summary' in sample:
            print(f"   Required columns found: 'dialogue' and 'summary'")
        else:
            print(f"   Warning: Expected 'dialogue' and 'summary' columns")
    
    """
    Part 2a: Conduct initial exploration of the SAMSum dataset.
    """
    def explore_data_analysis1(self):
        print(" Exploratory data analysis")
        print("="*60)

        print(f"  Training damples: {len(self.dataset['train'])}")
        print(f"  Validation sample: {len(self.dataset['validation'])}")
        print(f"  Test sample: {len(self.dataset['test'])}")

    """
    Part 2b: Plot length distribution of dialogues and summaries.
    """    
    def analyze_lengths(self):
        print(f"\n Analyze text lengths")

        # Extract texts
        train_data = self.dataset['train']
        
        # Filter out None values and convert to strings
        dialogues = [str(item['dialogue']) for item in train_data if item['dialogue'] is not None]
        summaries = [str(item['summary']) for item in train_data if item['summary'] is not None]

        # Calculate lengths (in words)
        dialogue_lengths = [len(text.split()) for text in dialogues if text and text != 'None']
        summary_lengths = [len(text.split()) for text in summaries if text and text != 'None']

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Dialogue lengths
        ax1.hist(dialogue_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Distribution of Dialogue Lengths', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Words')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(dialogue_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(dialogue_lengths):.1f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Summary lengths
        ax2.hist(summary_lengths, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_title('Distribution of Summary Lengths', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Words')
        ax2.set_ylabel('Frequency')
        ax2.axvline(np.mean(summary_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(summary_lengths):.1f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return dialogue_lengths, summary_lengths
    
    """
    Part 2c: Display the 20 most common words in dialogues.
    """
    def analyze_vocabulary(self):
        print("\n Analyze vocabulary")

        # Join all dialogues into one string for analysis, filtering out None values
        dialogues = [str(item['dialogue']) for item in self.dataset['train'] if item['dialogue'] is not None]
        all_dialogues = ' '.join(dialogues)

        # Clean and tokenize
        # Remove speaker names (text before colon and the colon itself) from each line
        cleaned_text = re.sub(r'^[^:]+:', '', all_dialogues, flags=re.MULTILINE)
        
        # Extract only alphabetic words (no numbers, punctuation, etc.)
        words = re.findall(r'\b[a-zA-Z]+\b', cleaned_text.lower())

        # Filter out stopwords
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        most_common = word_counts.most_common(20)

        print("\n Top 20 Most Common Words in Dialogues:")
        for i, (word, count) in enumerate(most_common, 1):
            print(f" {i:2d} | {word:11s} | {count:,}")
        # - enumerate(most_common, 1): Creates pairs of (index, item) starting from 1
        # - (word, count): Unpacks each tuple from most_common list into word and count
        # - f" {i:2d}": Formats index as 2-digit number
        # - f"{word:11s}": Formats word as string with 11 characters
        # - f"{count:,}": Formats count with comma separators for thousands

        return most_common
    
    """
    Compare multiple pre-trained summarization models on sample data.
    """
    def compare_pretrained_models(self):
        print(" Compare Pre-trained Models")

        # .select([0]) creates a new dataset containing only the item at index 0
        # [0] then extracts that single item as a dictionary with 'dialogue' and 'summary' keys
        # Example: If the test set has items like [{'dialogue': 'Hi John...', 'summary': 'Brief chat'}, ...]
        # then sample will be {'dialogue': 'Hi John...', 'summary': 'Brief chat'}
        sample = self.dataset['test'].select([0])[0]
        dialogue = sample['dialogue']
        reference_summary = sample['summary']

        print("\n Test Dialogue:")
        print(f"{dialogue}")
        print("\n Reference Summary:")
        print(f"{reference_summary}")

        # Define models to compare (as specified in requirements)
        models = [
            "t5-large",  # As specified in requirement 3(a)
            "facebook/bart-large-cnn", 
            "google/pegasus-xsum"
        ]

        # Check for GPU
        device = 0 if torch.cuda.is_available() else -1
        print(f"\n Using device: {'GPU' if device == 0 else 'CPU'}")

        results = {}

        """
        Requirement 3(b): Use the models to generate summaries for a few randomly selected dialogues
        source: https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text2text_generation.py#L242
        https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
        """
        for model_name in models:
            print(f"\n Testing {model_name}")
            try:
                summarizer = pipeline("summarization", model=model_name, device=device)

                # Generate summary
                summary = summarizer(
                    dialogue,
                    max_length=50,
                    min_length=10,
                    do_sample=False # greedy decoding: always picks the token with highest prob, faster: no random sampling computation 
                    )[0]['summary_text']
                
                # Store the generated summary in results dictionary
                # results is a dict where keys are model names and values are generated summaries
                # Example: results = {"t5-large": "John and Mary discussed...", "facebook/bart-large-cnn": "The conversation was about..."}
                results[model_name] = summary
                print(f" Generated: {summary}")
                
                # Clean up memory after each model
                del summarizer
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f" Error with {model_name}: {str(e)}")
                results[model_name] = "Error during generation"

        # Display comparison
        print(f"\n Model Comparison Results:")
        print("="*60)
        print(f"Reference Summary: {reference_summary}")
        print("-"*60)
        for model, summary in results.items():
            print(f"{model:25s}: {summary}")
        
        # Part 3(b): Analyze the quality of summaries
        print(f"\n QUALITY ANALYSIS:")
        print("="*60)
        print(" Analyzing coherence and essential point capture:")
        
        for model, summary in results.items():
            if summary != "Error during generation":
                print(f"\n {model}:")
                print(f"   Summary: {summary}")
                
                # Basic quality checks
                # Example: if summary = "John and Mary discussed dinner plans for tonight."
                # len(summary.split()) = 8 words (> 5) and summary.count('.') = 1 (<= 3) → "High"
                # if summary = "Yes." then len = 1 word (< 5) → "Medium"
                coherence_score = "High" if len(summary.split()) > 5 and summary.count('.') <= 3 else "Medium"
                
                # Example: if dialogue starts with "John: Hey Mary, want to grab dinner tonight?"
                # dialogue.lower().split()[:10] = ["john:", "hey", "mary,", "want", "to", "grab", "dinner", "tonight?"]
                # if summary contains "dinner" → essential_points = "Good"
                # if summary = "They talked." and doesn't contain key words → "Needs improvement"
                essential_points = "Good" if any(word in summary.lower() for word in dialogue.lower().split()[:10]) else "Needs improvement"
                
                print(f"    Coherence: {coherence_score}")
                print(f"    Essential Points: {essential_points}")
            
        return results
    
    """
    Part 4. Define a baseline summarization model.
    """
    def create_lead3_baseline(self, text):
        """
        Part 4a: Create a lead-3 baseline summary by taking the first 3 sentences.
        """
        # Remove speaker names (everything before colon) to get clean sentences
        lines = text.split('\n')
        clean_lines = []

        for line in lines:
            if ':' in line:
                # Split the line at the first colon to separate speaker name from message
                # split(':', 1) splits only at the first colon occurrence, returning a list of 2 parts
                # [1] gets the second part (the message after the colon)
                # .strip() removes leading/trailing whitespace from the message
                speaker_removed = line.split(':', 1)[1].strip()
                if speaker_removed:
                    clean_lines.append(speaker_removed)
            else:
                clean_lines.append(line.strip())
        
        # Join all clean lines into one text
        clean_text = ' '.join(clean_lines)

        # Use NLTK punkt tokenizer to split into sentences
        sentences = sent_tokenize(clean_text)

        # Take the first 3 sentences 
        lead3_sentences = sentences[:3]

        # Join them back into a summary
        baseline_summary = ' '.join(lead3_sentences)

        return baseline_summary
    
    """
    Part 4b: Evaluate summaries using ROUGE.
    """
    def evaluate_with_rouge(self, predictions, references):

        # Load ROUGE metric
        rouge = evaluate.load("rouge")

        # ROUGE used to evaluate automatic summarization and machine translation systems.
        # It measures the overlap between generated summaries and reference summaries.
        # 
        # ROUGE metrics include:
        # - ROUGE-1: Overlap of unigrams (single words)
        # - ROUGE-2: Overlap of bigrams (two consecutive words)
        # - ROUGE-L: Longest Common Subsequence based overlap
        # - ROUGE-Lsum: ROUGE-L applied to summary-level
        #
        # Source: Hugging Face Evaluate library
        # Documentation: https://huggingface.co/spaces/evaluate-metric/rouge
    
        # Compute ROUGE scores
        results = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True  # Enable Porter stemming to reduce words to their root form for better matching
        )
        return results
    
    """
    Compare lead-3 baseline with pre-trained models using ROUGE.
    """
    def compare_baseline_with_models(self, num_samples=10):
        print(" Baseline vs Models Comparison")
        print("="*60)

        # Get test samples using select method instead of slicing
        test_samples = self.dataset['test'].select(range(num_samples))

        # Extract dialogues and reference summaries
        dialogues = [sample['dialogue'] for sample in test_samples]
        reference_summaries = [sample['summary'] for sample in test_samples]
        
        # Generate lead-3 baseline summaries
        print(" Generate lead-3 baseline summaries")
        baseline_summaries = []
        for dialogue in dialogues:
            baseline_summary = self.create_lead3_baseline(dialogue)
            baseline_summaries.append(baseline_summary)
        
        # Generate summaries from pre-trained models
        models_to_test = [
            "t5-large",
            "facebook/bart-large-cnn",
            "google/pegasus-xsum"
        ]

        device = 0 if torch.cuda.is_available() else -1
        model_summaries = {}

        for model_name in models_to_test:
            print(f" Generate summaries with {model_name}")
            try:
                summarizer = pipeline("summarization", model=model_name, device=device)
                summaries = []

                for dialogue in dialogues:
                    summary = summarizer(
                        dialogue,
                        max_length=50,
                        min_length=10,
                        do_sample=False
                    )[0]['summary_text']
                    summaries.append(summary)

                model_summaries[model_name] = summaries
                print(f" Completed {model_name}")
                
                # Clean up memory after each model
                del summarizer
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f" Error with {model_name}: {str(e)}")
        
        # Evaluate all methods with ROUGE
        print("\nEvaluating with ROUGE scores")

        # Evaluate baseline with ROUGE
        baseline_rouge = self.evaluate_with_rouge(baseline_summaries, reference_summaries)

        # Evaluate each model
        model_rouge_scores = {}
        for model_name, summaries in model_summaries.items():
            rouge_scores = self.evaluate_with_rouge(summaries, reference_summaries)
            model_rouge_scores[model_name] = rouge_scores
        
        # Display results
        print("\n" + "="*60)
        print("ROUGE SCORE COMPARISON")
        print("="*60)
        
        # Create a comparison table
        print(f"{'Method':<25} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10}")
        print("-" * 55)
        
        # Baseline scores
        print(f"{'Lead-3 Baseline':<25} {baseline_rouge['rouge1']:<10.4f} {baseline_rouge['rouge2']:<10.4f} {baseline_rouge['rougeL']:<10.4f}")

        # Model scores
        for model_name, scores in model_rouge_scores.items():
            short_name = model_name.split('/')[-1]
            print(f"{short_name:<25} {scores['rouge1']:<10.4f} {scores['rouge2']:<10.4f} {scores['rougeL']:<10.4f}")

        return {
            'baseline_rouge': baseline_rouge,
            'model_rouge_scores': model_rouge_scores,
            'baseline_summaries': baseline_summaries,
            'model_summaries': model_summaries
        }


    """
    Part 5: Fine-Tuning a Pre-trained Model
    """
    
    """"
    Part 5a & 5b: Choose a model and preprocess the dataset for fine-tuning.
    """
    # use t5-small for efficiency in fine tuning
    def setup_fine_tuning(self, model_name='t5-small', max_input_length=256, max_target_length=64):
        print("\n" + "="*60)
        print("="*60)
        print(f" Chosen model: {model_name}")

        # Clear any existing models from memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Load tokenizer and model
        print(f" Loading tokenizer and model")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Store parameters
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.model_name = model_name

        print(f" Model loaded successfully!")
        print(f" Max input length: {max_input_length}")
        print(f" Max target length: {max_target_length}")

        return self.tokenizer, self.model
    
    """
    Part 5b: Preprocess the dataset to fit the T5-small model's input format.
    """
    def preprocess_dataset(self):
        print(f"\n Preprocessing dataset for {self.model_name}")

        # Filter out None values from the dataset first
        def filter_none_values(example):
            return (example['dialogue'] is not None and 
                    example['summary'] is not None and
                    str(example['dialogue']).strip() != '' and
                    str(example['summary']).strip() != '')

        print("   Filtering out None and empty values")
        self.dataset['train'] = self.dataset['train'].filter(filter_none_values)
        self.dataset['validation'] = self.dataset['validation'].filter(filter_none_values)
        self.dataset['test'] = self.dataset['test'].filter(filter_none_values)

        # The "summarize: " prefix tells the T5 model that this is a text summarization task
        prefix = "summarize: "

        """
        Tokenize dialogues and summaries for T5-small training.
        """
        def preprocess_function(examples):
            # Add T5 prefix to dialogues, handling None values
            inputs = [prefix + (dialogue if dialogue is not None else "") for dialogue in examples["dialogue"]]
            
            # Filter out None summaries and convert to strings
            summaries = [summary if summary is not None else "" for summary in examples["summary"]]

            # Tokenize inputs
            # Source: https://huggingface.co/docs/transformers/tasks/summarization#preprocess
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_input_length,
                truncation=True,
                padding=False # Pad dynamically during training
            )
            
            # Tokenize targets (summaries)
            labels = self.tokenizer(
                text_target=summaries,
                max_length=self.max_target_length,
                truncation=True,
                padding=False
            )

            # Replace padding token id with -100 so it's ignored in loss calculation
            labels["input_ids"] = [
                [-100 if token == self.tokenizer.pad_token_id else token for token in label]
                for label in labels["input_ids"] 
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Apply preprocessing function to all splits
        print("   Tokenizing training data")
        tokenized_train = self.dataset["train"].map(
            preprocess_function, 
            batched=True,
            desc="Tokenizing train"
            )

        print("   Tokenizing validation data")
        tokenized_validation = self.dataset["validation"].map(
                preprocess_function,
                batched=True,
                desc="Tokenizing validation"
            )
        
        print(" Tokenizing test data")
        tokenized_test = self.dataset["test"].map(
            preprocess_function,
            batched=True,
            desc="Tokenizing test"
        )

        # Store tokenized datasets
        self.tokenized_datasets = {
            "train": tokenized_train,
            "validation": tokenized_validation,
            "test": tokenized_test
        }

        print(f"   Dataset preprocessing completed")
        print(f"   Train samples: {len(tokenized_train)}")
        print(f"   Validation samples: {len(tokenized_validation)}")
        print(f"   Test samples: {len(tokenized_test)}")

        return self.tokenized_datasets

    """
    Part 5c: Set up training arguments optimized for T5-small 
    """
    def setup_training_arguments(self, output_dir="./samsum-t5-finetuned", num_epochs=1, batch_size=4):
        """
        Optimized training arguments for efficiency
        """
        print(f"\n Setting up training arguments for T5-small ")

        # Create training arguments optimized for T5-small 
        # source Seq2SeqTrainingArguments: set_inital_training_value :https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            
            # Training hyperparameters 
            num_train_epochs=num_epochs,  # Reduced from 3 to 1 for faster training
            learning_rate=3e-4,  # Slightly higher learning rate for faster convergence
            weight_decay=0.01,
            warmup_steps=100,  # Reduced warmup steps

            # Batch sizes (optimized for memory)
            per_device_train_batch_size=batch_size,  # Increased from 2 to 4
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,  # Reduced from 4 to 2
            
            # Evaluation and saving (optimized for speed)
            eval_strategy="steps",  # Changed from "epoch" to "steps"
            eval_steps=500,  # Evaluate every 500 steps
            save_strategy="steps",
            save_steps=500,
            save_total_limit=1,  # Keep only 1 checkpoint to save space
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            greater_is_better=True,

            # Generation settings
            # Source: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments.predict_with_generate
            predict_with_generate=True,
            generation_max_length=self.max_target_length,
            
            # Performance optimization for T5-large
            # Source: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.fp16
            fp16=torch.cuda.is_available(), # Mixed precision training - https://arxiv.org/abs/1710.03740
            dataloader_pin_memory=False,
            remove_unused_columns=True,
            dataloader_num_workers=0,  # Avoid multiprocessing issues in Colab
            
            # Logging (reduced frequency)
            logging_steps=100,
            report_to=None,  # Disable wandb/tensorboard
            
            # Reproducibility
            # Source: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.seed
            seed=42,
            
            # Memory optimization
            max_grad_norm=1.0,
          
        )

        print(f" Training arguments configured for T5-small:")
        print(f"   Output directory: {output_dir}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size} (effective: {batch_size * 2} with gradient accumulation)")
        print(f"   Learning rate: {self.training_args.learning_rate}")
        print(f"   Mixed precision: {self.training_args.fp16}")
        print(f"   Evaluation strategy: every {self.training_args.eval_steps} steps")

        return self.training_args
    
    """
    Part 5d: Set up ROUGE metrics for T5-small evaluation during training.
    """
    def setup_metrics(self):
        print(f"\n Setting up evaluation metrics for T5-small")

        # Load ROUGE metric
        self.rouge_metric = evaluate.load("rouge")

        """
        Compute ROUGE scores during evaluation.
        """
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            
            # Handle predictions - they might be logits, so take argmax if needed
            if predictions.ndim == 3:
                predictions = np.argmax(predictions, axis=-1)
            
            # Clean predictions: remove padding tokens and invalid values
            cleaned_predictions = []
            for pred in predictions:
                # Filter out pad tokens and invalid values
                cleaned_pred = [token for token in pred if token >= 0 and token < self.tokenizer.vocab_size]
                cleaned_predictions.append(cleaned_pred)
            
            # Decode predictions
            decoded_preds = self.tokenizer.batch_decode(cleaned_predictions, skip_special_tokens=True)
            
            # Clean labels: replace -100 with pad token and filter invalid values
            cleaned_labels = []
            for label in labels:
                # Replace -100 with pad token ID and filter out invalid values
                cleaned_label = [self.tokenizer.pad_token_id if token == -100 else token for token in label]
                cleaned_label = [token for token in cleaned_label if token >= 0 and token < self.tokenizer.vocab_size]
                cleaned_labels.append(cleaned_label)
            
            # Decode labels
            decoded_labels = self.tokenizer.batch_decode(cleaned_labels, skip_special_tokens=True)
            
            # ROUGE expects newline-separated sentences for proper evaluation
            decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds if pred.strip()]
            decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels if label.strip()]
            
            # Make sure we have equal length lists
            min_length = min(len(decoded_preds), len(decoded_labels))
            decoded_preds = decoded_preds[:min_length]
            decoded_labels = decoded_labels[:min_length]
            
            # Skip ROUGE computation if no valid predictions
            if not decoded_preds or not decoded_labels:
                return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            
            # Compute ROUGE scores
            # Source: https://huggingface.co/spaces/evaluate-metric/rouge
            result = self.rouge_metric.compute(
                predictions=decoded_preds, 
                references=decoded_labels, 
                use_stemmer=True
            )
            
            return result

        self.compute_metrics = compute_metrics
        print(" Metrics setup completed!")

        return compute_metrics
    
    """
    Part 5c: Train T5-small on the SAMSum corpus .
    """
    def fine_tune_model(self):
        print(f"\nStarting T5-small fine-tuning process ")

        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )

        # Create Trainer
        
        # Source: https://huggingface.co/docs/transformers/tasks/summarization#train
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        # Start training
        print(f"\n Beginning T5-small training")
        train_result = self.trainer.train()

        # Save the final model
        print(f"\n Saving fine-tuned T5-small model...")
        self.trainer.save_model()
        self.trainer.save_state()

        print(f" T5-small fine-tuning completed!")
        print(f"   Final training loss: {train_result.training_loss:.4f}")
        print(f"   Model saved to: {self.training_args.output_dir}")

        return train_result
    
    """
    Part 6a: Evaluate the fine-tuned T5-large model on the test set.
    """
    def evaluate_finetuned_model(self, num_samples=50):
        print(f"\n EVALUATING FINE-TUNED MODEL")
        print("="*60)

        # Evaluate on test set
        print(f" Running evaluation on {num_samples} test samples")

        # Get test samples
        test_samples = self.dataset['test'].select(range(num_samples))
        test_dialogues = [sample['dialogue'] for sample in test_samples]
        test_references = [sample['summary'] for sample in test_samples]

        # Generate summaries with fine-tuned model
        print(f"   Generating summaries with fine-tuned model")

        # Create pipeline with fine-tuned model
        fine_tuned_pipeline = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        fine_tuned_summaries = []
        for dialogue in test_dialogues:
            # Add T5 prefix 
            input_text = f"summarize: {dialogue}"

            summary = fine_tuned_pipeline(
                input_text,
                max_length=self.max_target_length,
                min_length=10,
                do_sample=False
            )[0]['summary_text']

            fine_tuned_summaries.append(summary)

        # Calculate ROUGE scores
        fine_tuned_rouge = self.evaluate_with_rouge(fine_tuned_summaries, test_references)

        print(f"\n FINE-TUNED MODEL PERFORMANCE:")
        print(f"   ROUGE-1: {fine_tuned_rouge['rouge1']:.4f}")
        print(f"   ROUGE-2: {fine_tuned_rouge['rouge2']:.4f}")
        print(f"   ROUGE-L: {fine_tuned_rouge['rougeL']:.4f}")

        return {
            'rouge_scores': fine_tuned_rouge,
            'summaries': fine_tuned_summaries,
            'references': test_references,
            'dialogues': test_dialogues
        }
    
    """
    Part 6b: Compare pre-trained vs fine-tuned model performance.
    """
    def compare_before_after_finetuning(self, num_samples=20):
        print(f"\n COMPARING: BEFORE vs AFTER FINE-TUNING")
        print("="*60)

        # Get test samples
        test_samples = self.dataset['test'].select(range(num_samples))
        test_dialogues = [sample['dialogue'] for sample in test_samples]
        test_references = [sample['summary'] for sample in test_samples]

        # 1. Pre-fine-tuning performance (original model)
        print(f"\n Generating summaries with PRE-TRAINED {self.model_name}")
        original_pipeline = pipeline(
            'summarization',
            model=self.model_name,
            device=0 if torch.cuda.is_available() else -1
        )

        original_summaries = []
        for dialogue in test_dialogues:
            # T5 needs the prefix
            input_text = f"summarize: {dialogue}"
            summary = original_pipeline(
                input_text,
                max_length=self.max_target_length,
                min_length=10,
                do_sample=False
            )[0]['summary_text']
            original_summaries.append(summary)

        original_rouge = self.evaluate_with_rouge(original_summaries, test_references)

        # 2. Post-fine-tuning performance 
        print(f"\n Generating summaries with FINE-TUNED {self.model_name}")

        fine_tuned_pipeline = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        fine_tuned_summaries = []
        for dialogue in test_dialogues:
            input_text = f"summarize: {dialogue}"

            summary = fine_tuned_pipeline(
                input_text,
                max_length=self.max_target_length,
                min_length=10,
                do_sample=False
            )[0]['summary_text']
            fine_tuned_summaries.append(summary)

        fine_tuned_rouge = self.evaluate_with_rouge(fine_tuned_summaries, test_references)

        # 3. Display comparison
        print(f"\n MODEL PERFORMANCE COMPARISON")
        print("="*60)
        print(f"{'Metric':<15} {'Pre-trained':<12} {'Fine-tuned':<12} {'Improvement':<12}")
        print("-" * 55)
        
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            original_score = original_rouge[metric]
            finetuned_score = fine_tuned_rouge[metric]
            improvement = finetuned_score - original_score
            improvement_percent = (improvement / original_score) * 100 if original_score > 0 else 0
            print(f"{metric.upper():<15} {original_score:.4f}      {finetuned_score:.4f}      {improvement:+.4f} ({improvement_percent:+.1f}%)")

        # 4. Show example comparisons
        print(f"\n EXAMPLE COMPARISONS (First 3 samples)")
        print("="*60)

        for i in range(min(3, num_samples)):
            print(f"\n Example {i+1}")
            print(f"Dialogue: {test_dialogues[i][:100]}...")
            print(f"Reference: {test_references[i]}")
            print(f"Pre-trained: {original_summaries[i]}")
            print(f"Fine-tuned: {fine_tuned_summaries[i]}")
            print("-" * 40)
        
        return {
            'original_rouge': original_rouge,
            'finetuned_rouge': fine_tuned_rouge,
            'original_summaries': original_summaries,
            'finetuned_summaries': fine_tuned_summaries
        }
    
    """
    Run the complete assignment pipeline including fine-tuning.
    """
    def run_complete_analysis(self):
        print("\n" + "="*60)
        print(" STARTING COMPLETE SAMSum ANALYSIS PIPELINE")
        print("="*60)
        
        try:
            # Part 2: Exploratory Data Analysis
            print("\n PART 2: EXPLORATORY DATA ANALYSIS")
            self.explore_data_analysis1()
            self.analyze_lengths()
            self.analyze_vocabulary()
            
            # Part 3: Compare Pre-trained Models
            print("\n PART 3: COMPARING PRE-TRAINED MODELS")
            model_comparison = self.compare_pretrained_models()
            
            # Part 4: Baseline and Comparison
            print("\n PART 4: BASELINE COMPARISON")
            baseline_comparison = self.compare_baseline_with_models()
            
            # Part 5: Fine-tuning 
            print("\n PART 5: FINE-TUNING ")
           
            
            # Step 1: Setup fine-tuning with t5-small for efficiency
            self.setup_fine_tuning(model_name="t5-small")
            
            # Step 2: Preprocess dataset
            self.preprocess_dataset()
            
            # Step 3: Setup training arguments 
            self.setup_training_arguments(batch_size=2, num_epochs=1)
            
            # Step 4: Setup metrics
            self.setup_metrics()
            
            # Step 5: Fine-tune the model
            train_result = self.fine_tune_model()
            
            # Part 6: Evaluation
            print("\n PART 6: EVALUATION")
            evaluation_results = self.evaluate_finetuned_model()
            comparison_results = self.compare_before_after_finetuning()
            
            print(f"\n ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            return {
                'model_comparison': model_comparison,
                'baseline_comparison': baseline_comparison,
                'train_result': train_result,
                'evaluation': evaluation_results,
                'comparison': comparison_results
            }
            
        except Exception as e:
            print(f"\n Error during analysis: {str(e)}")
            raise e

def main():
    
    print(" Starting SAMSum Text Summarization Assignment")
    print("=" * 60)
    
    # Initialize the analyzer
    analyzer = SAMSumSummarizer()
    
    # Run the complete analysis
    results = analyzer.run_complete_analysis()
    
    print("\n Assignment completed successfully!")
    print("\n SUMMARY OF RESULTS:")
    print("- Exploratory Data Analysis:  Complete")
    print("- Pre-trained Model Comparison:  Complete") 
    print("- Baseline vs Models Comparison:  Complete")
    print("- Fine-tuning:  Complete")
    print("- Evaluation:  Complete")
    
    return results

if __name__ == "__main__":
    
    main()
        










        



    





       
    






        

