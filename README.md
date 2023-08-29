# Text Sumarization

This project implements an abstractive text summarization model using the T5 transformer architecture from Hugging Face. The model is trained to generate a short, fluent summary given a longer text input


## Introduction

Text summarization is the task of producing a concise summary while preserving key information content and overall meaning. Abstractive methods aim to generate novel phrases and sentences, while extractive methods copy key snippets verbatim from the original text.

This project focuses on abstractive summarization using an encoder-decoder architecture. The encoder reads and builds representations of the input text. The decoder condense these representations into a short, relevant summary.

## Model Architecture

The model architecture used is the T5 transformer developed by Google AI. T5 converts all text processing tasks into a common text-to-text format. For summarization, the input text is prefixed with "summarize: " and the target output is the summary.

The base T5 model has roughly 220 million parameters. The encoder and decoder each have 12 transformer layers with model dimension of 768 and feed forward dimension of 3072.

## Data

The model is trained on the CNN/Daily Mail dataset containing ~300k news article and summary pairs. The data is split into 80% train, 10% validation, and 10% test.

The torch.utils.data.Dataset class is used to load the data and preprocess text into input IDs and target IDs. The DataLoader handles batching, shuffling, and sampling.

## Training

The model is trained for 3 epochs with a batch size of 128. The AdamW optimizer is used with a peak learning rate of 3e-4 and linear decay.

The loss function is label smoothed cross entropy. Validation loss is monitored for early stopping and model checkpointing.

Multi-GPU training is enabled for faster training. The best model on validation data is saved.

## Evaluation

The trained model is evaluated on the test set for ROUGE scores. This measures overlap between generated and reference summaries in terms of n-gram, word, and phrase overlap.

Human evaluation is also critical for assessing fluency and information coverage.

## Usage

The summarize.py script loads a pretrained T5 model and tokenizer and summarizes input text.

Model checkpoints from training can be loaded for domain-specific summarization.

## References

- T5 Paper: [link](https://arxiv.org/abs/1910.10683)
- Hugging Face Transformers: [link](https://huggingface.co/transformers/)

Pretrained T5 models of various sizes are available from Hugging Face. Fine-tuning on domain-specific data can further improve performance.
