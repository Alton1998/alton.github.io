---
draft: false
date: 2025-01-09
comments: true
authors:
  - alton
categories:
  - Artificial Neural Networks
  - Artificial Intelligence
  - LLMs
  - VLMs
  - LLaVA
---

# LLaVA
## Overview
LLaVA (Large Language and Vision Assistant) was first introduced in the paper "Visual Instruction Tuning".

## What is Visual Instruction Tuning?
Visual instruction tuning is a method used to fine-tune a large language model, enabling it to interpret and respond to instructions derived from visual inputs.

One example is to ask a machine learning model to describe an image.

## LLaVA
As already established LLaVA is a multimodal model. LLaVA was trained on a small dataset. Despite this it can perform image analysis and respond to questions.

### Architecture
The LLaVA has the following components:
1. Language model
2. Vision Encoder
3. Projection

We use the Llama as the language model, which is a family of autoregressive LLMs released by Meta AI.

The vision encoder is implemented by CLIP visual encoder ViT-L/14. The encoder extracts visual features and connects them to language embeddings through a projection matrix. The projection component translates visual features into language embedding tokens, thereby bridgin the gap between text and images.

### Training 
Two stages of training:

1. Pre-training for Feature Alignment: LLaVA aligns visual and language features to ensure compatibility in this initial stage.
2. Fine-tune end-to-end: The second training stage focuses on fine-tuning the entire model. At this stage the vision encoder's weights remain fixed

## LLaVA-1.5
In LLaVA-1.5 there are two significant changes:
1. MLP vision-language connector
2. Trained for academic task-oriented data.

The linear projection layer is replaced with a 2 layer MLP.

## LLaVA 1.6 (LLaVA-NeXT)
n addition to LLaVA 1.5, which uses the Vicuna-1.5 (7B and 13B) LLM backbone, LLaVA 1.6 considers more LLMs, including Mistral-7B and Nous-Hermes-2-Yi-34B. These LLMs possess nice properties, flexible commercial use terms, strong bilingual support, and a larger language model capacity. It allows LLaVA to support a broader spectrum of users and more scenarios in the community. The LLaVA recipe works well with various LLMs and scales up smoothly with the LLM up to 34B.

Here are the performance improvements LLaVA-NeXT has over LLaVA-1.5:

Increasing the input image resolution to 4x more pixels. This allows it to grasp more visual details. It supports three aspect ratios, up to 672x672, 336x1344, and 1344x336 resolution.
Better visual reasoning and zero-shot OCR capability with multimodal document and chart data.
Improved visual instruction tuning data mixture with a higher diversity of task instructions and optimizing for responses that solicit favorable user feedback.
Better visual conversation for more scenarios covering different applications. Better world knowledge and logical reasoning.
Efficient deployment and inference with SGLang.

Other variants of LLaVA:
1. LLaVA-Med
2. LLaVA-Interactive

## Reference
1. A. Acharya, “LLAVA, LLAVA-1.5, and LLAVA-NeXT(1.6) explained,” Nov. 04, 2024. https://encord.com/blog/llava-large-language-vision-assistant/
2. Wikipedia contributors, “Llama (language model),” Wikipedia, Jan. 01, 2025. https://en.wikipedia.org/wiki/Llama_(language_model)