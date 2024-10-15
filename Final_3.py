# -*- coding: utf-8 -*-
"""
Final Team Project: Build a Chatbot
Priscilla Marquez, Johnathan Long, Greg Moore
Applied Artificial Intelligence (AAI), University of San Diego
AAI-520: Natural Language Processing and GenAI
Professor Kahila Mokhtari, PhD
October 21, 2024
"""

####################################################################

# Project Overview:
#
# Goal: Build a chatbot that can carry out multi-turn conversations, adapt
# to context, and handle a variety of topics.
# Output: A web or app interface where users can converse with the chatbot.
#
# Pre-requisites:
# Basic understanding of deep learning and neural networks.
# Familiarity with a deep learning framework (e.g., TensorFlow, PyTorch).
# Basic knowledge of web development (for the interface).
#
# Phases:
#   
# Research and Study Phase:
# Study generative-based chatbot architectures like Seq2Seq, Transformers,
# and GPT and deep learning. Understand the challenges of chatbot design:
# context management, coherency, handling ambiguous queries, etc.
#
# Data Collection and Preprocessing:
# We chose to use the Ubuntu Dialogue Corpus Dataset
# https://www.kaggle.com/datasets/rtatman/ubuntu-dialogue-corpus
#
# Model Design and Training:
# Choose an architecture (e.g., Transformer-based models or deep learning models).
# Implement or leverage existing implementations to train the model with the dataset.
#
# Evaluation:
# Implement evaluation metrics.

##################################################

# Load the libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
# stop the future warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# We will use the text column in the Ubuntu Dialogue Corpus
# Read in entire Ubuntu dialogueText.csv dataset
ubuntu = pd.read_csv('C:/Users/gregm/.spyder-py3/AAI_520/FINAL/dialogueText.csv')
# Output Consumer Complaints Dataset
print('\nDisplay Ubuntu Dataset:\n', ubuntu)

##################################################

# Text Preprocessing
# Extract the first 50 text entries to debug, taking around 8 minutes/Epoch
text = ubuntu['text'].head(50)
# Output the first 5 texts
print('\nHead of First 5 texts:\n', text.head(5))

# Convert text to a standard list for processing
text = text.tolist()
# Output the text lists
print('\nFirst of Ubuntu Text List:\n', text[:3])

##################################################

# Data Preparation and Pre-processing
class DialogueDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_length=512):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        encoded = self.tokenizer.encode_plus(
            dialogue,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }


# Pre-process data
dialogues = text
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
dataset = DialogueDataset(dialogues, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print(dialogues)

