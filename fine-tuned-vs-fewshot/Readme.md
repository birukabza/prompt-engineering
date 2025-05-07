# Emotion Classification with Transformers

This project uses a pre-trained DistilBERT model to classify text into six emotion categories: **joy**, **sadness**, **anger**, **fear**, **surprise**, and **love**.
[View the full documentation here](https://docs.google.com/document/d/1u91-Fvl3wPGp0M60FdjzTcgC1ZsiqT0-vWtYece8q60/edit?usp=sharing)

## Features

- Fine-tunes a `distilbert-base-uncased` model on the [SetFit/emotion](https://huggingface.co/datasets/SetFit/emotion) dataset.
- Uses sentence embeddings and cosine similarity for a prototype-based few-shot classification.
- Evaluates the model using test data and prints classification results.
- Shows correctly classified and misclassified examples.

## Requirements

- Python 3.7+
- Install dependencies using:

```bash
pip install torch transformers datasets scikit-learn
