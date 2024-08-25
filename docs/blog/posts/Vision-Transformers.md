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
We know that the Transformer architecture has become the norm for Natural Language Processing(NLP) tasks. Unlike in NLP tasks in conjunction with attention we also have convoluitonal networks. However, this paper demonstrates **that convolitonal networks need not be applied** and a pure transformer architecture applied on a sequence of image patches can perform image classification tasks really well provided that its pre-trained on large amounts of data.

## Basic Theory of Vision Transformers

The Vision Transformer architecture has been inspired by the successes of the Transformer successes in NLP. The first step to create a Vision Transformer is to split an image into patches. We now generate the position of these patches and then generate embeddings for them. Let us consider dimensional tranformation that is taking place here. 

Our original image X had the dimenstion HxWxC. Where H is height and W is the width of the images and C is the channel. Since, we are dealing with RGB images the C will be 3. 

After fetching the patches, we get the following dimensions NxPxPxC.

Where N is the number of patches in an image. 

To calculate it N = $\frac{H * W}{P*P}$

Now, we flatten the aforementioned patches and project them via a dense layer to have a dimension D whic his known as the **constant latent vector size D**. Then we add the patch embeddings and the positional embeddings to retain some of the position information. The postional information is in 1D and not 2D since no performance gain was observed.

This output is forwarded through the some layers of the transformer blocks.

The transformer enocder block is composed of alternating layers of multiheaded self attention and MLP blocks. Layer Norm is applied before every block i.e. before an attention or MLP block and a residual connection is created after every block.

It is to be noted that Vision Transformers have much less inductive bias than CNNs. Inductive biases are assumptions we make about a data set. For example we can assume the marks of students in a given subject to follow a gaussian distribution. CNN architectures inherently will have some biases due to the way they are structured. CNNs are structured to capture the local relationship between the pixels of an image. As CNNS get deeper the local feature extractors help tp extract the global features. In Vit only the MLP layers are local and translationally equivariant while the self attention layers are global. An hybrid version of ViT also exists where CNN is applied to extract the feature maps and then forward to the Transformer encoder block.

 

