# **Transformer From Scratch using PyTorch**

A full, educational implementation of a Transformer model in PyTorch following the architecture presented in the "Attention is All You Need" paper. The implementation includes:

* Tokenization

* Positional Encoding

* Multi-Head Self-Attention

* Cross-Attention

* Encoder & Decoder stacks

* Full Transformer model for sequence-to-sequence tasks

This repository is intended as a learning resource for understanding the internal mechanics of Transformers without relying on high-level PyTorch modules.

## **Features**

* Character-level or word-level tokenization (configurable)

* Custom Positional Encoding

* Mask generation that performs:

    1. Encoder padding mask

    2. Decoder padding + look-ahead mask

    3. Cross-attention encoder-padding mask

* Multi-Head Self-Attention and Cross-Attention

* Layer Normalization

* Position-wise Feedforward Network

* Full encoder–decoder architecture

* Works on GPU or CPU automatically


## **Model Overview**

This implementation follows the original Attention Is All You Need architecture:

### 🔵 Encoder

For each encoder layer:

* Multi-Head Self-Attention

* Add & Norm

* Feed-Forward Network

* Add & Norm

### 🔴 Decoder

For each decoder layer:

* Masked Multi-Head Self-Attention

* Add & Norm

* Cross Attention over encoder outputs

* Add & Norm

* Feed-Forward Network

* Add & Norm

## 🟡 Masking

* The project includes a dedicated Masks class that generates:

* Encoder padding masks

* Decoder padding + look-ahead masks

* Cross-attention masks

Masks are expanded to the correct 4-D shape for multi-head attention.

##  **Main Classes**
Module	Description
| Module | Description |
|--------|-------------|
| `PositionalEncoding` | Computes sinusoidal positional encodings. |
| `SentenceEmbedding`  | Tokenizes, embeds, and adds positional encoding. |
| `Masks`              | Builds encoder/decoder attention masks. |
| `MultiHeadAttention` | Self-attention layer with multiple heads. |
| `MultiHeadCrossAttention` | Encoder→decoder attention. |
| `EncoderLayer`       | A single encoder block. |
| `DecoderLayer`       | A single decoder block. |
| `Transformer`        | Full encoder–decoder model. |
