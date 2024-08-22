---
draft: false 
date: 2024-08-22
authors:
  - alton
categories:
  - Computer Vision
  - Transformers
  - Image Classification
  - Vision Transformers
---

# Understanding Vision Transformers

## Overview
I was taking part in the ISIC 2024 challenge when I got stuck training a ResNet50 model that it started overfitting. My score at this point was 0.142. To be at the the top I had to beat the score 0.188. While scouring the internet for any new model I came across Vision Transformers. I was honestly surprised that transformer architecture could be applied to images. I came across this interesting paper called "[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://doi.org/10.48550/arXiv.2010.11929)"

## About the Paper
We know that the Transformer architecture has become the norm for Natural Language Processing(NLP) tasks. Unlike in NLP tasks in conjunction with attention we also have convoluitonal networks. However, this paper demonstrates **that convolitonal networks need not be applied** and a pure transformer architecture applied on a sequence of image patches can perform image classification tasks really well provided that its pre-trained on large amounts of data 
