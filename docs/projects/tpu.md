# Tensor Processing Unit
## Overview
<div style='text-align: justify;'>
The purpose of this project is to implement the
Multiply and Accumulate(MAC) Unit that can be used in the
Tensor Processor Unit that optimizes matrix multiplication by
integrating the computation unit as close to the memory as
possible and reducing the read and write times to memory. The
multiply and accumulate unit includes a multiplier and adder.
In this project we have chosen to go with the Carry Look Ahead
Adder for the adder and for the multiplier we implement a design
with scan flip flops. In our design we use static CMOS logic to
build all out gates i.e AND, OR, half adders. The result of our
project is a schematic and physical design of the MAC unit that
can operate at a frequency of 0.167GHz.
</div>

## Introduction
<div style='text-align: justify;'>
Deep learning models are like a ”Swiss Army Knife”
which are revolutionizing various fields. One example that
comes to mind where is Healthcare. With the break through
in image recognition in AI, we can create models that can
help in chest X-ray or MRI scan diagnosis. Also, we have
various machine learning models that play a vital role today
in software engineering and research. Some problems that AI
helps us solve today are image recognition, natural language
processing and recommendation systems. In addition to this
AI has a new frontier, generative AI [1].
Generative Adversarial Networks(GANs) are known to have
high computational and memory requirements. The operations
performed by GANs are convolution and deconvolution. These
operations are not as compatible with conventional accelerators that are designed for convolution operations. There
is a need for for customized accelerators or Application
Specific Integrated Circuits like TPUs [2]. For the sake of this
project we will consider neural network equations as shown
in equation (1),(2),(3) and (4) where W are the weights, X are
the inputs, b is the bias and f is the activation function. At the
end of the day GANs are made up of two networks i.e. the
generator and discriminator.
</div>

![Equations](img/Screenshot%202024-05-17%20224230.png)
<div style='text-align: justify;'>
While there are several different hardware architectures for
DNN acceleration, systolic array based implementations are
</div>