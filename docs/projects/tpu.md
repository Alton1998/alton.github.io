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
DNN acceleration, systolic array based implementations areshown to be most promising.The advantage with using systolic array based implementations that there need for buffering
the inputs and routing is less complex. Theoretically, this
should be energy efficient because we reducing the frequent
reads for the weights and inputs. A general architecture for
systolic based hardware architectures is shown in Fig. 1. MAC
unit implementation is shown in Fig. 2 which is what we try
to design in this project
</div>

![Systolic array General Architecture](./img/Screenshot%202024-06-05%20222645.png)

![MAC Unit Design](./img/Screenshot%202024-06-05%20222800.png)

## Literature Review


<div style='text-align: justify;'>
[4] Kuan-Chieh Hsu et al. proposes a General Purpose
Computing architecture built on Edge Tensor Processing Units.This is an open source framework which allows researchers
to easily use Neural Network accelerators for various applications. It was found that the proposed architecture is 2.46
times faster than CPU and the energy consumption is reduced
by 40%. The aforementioned Edge Tensor Processing used is a
trimmed down version of Google Cloud TPU i.e it has smaller
data memory .
</div>
<div style='text-align: justify;'>
[5] Adam G. M. Lewis et al. in their paper have shown how
to repurpose for large-scale scientific computation. They speed
up matrix multiply calculations for QR decomposition and
linear systems by distributing these in the matrix multiplication
units in Google’s Tensor Processing Units.
</div>
<div style='text-align: justify;'>
[6] Pramesh Pandey et al. proposes a low power near threshold TPU design without affecting the inference accuracies. The
way they achieve this is by identifying error-causing activation
sequences in the systolic array and preventing timing errors
from the same sequence by booting the operating voltage
for specific multiply and accumulate (MAC) units. The paper
improves the performance of a TPU by 2-3 times without
compromising the inference accuracies.
</div>
<div style='text-align: justify;'>
[7] Pramesh Pandey et al. proposes a way to solve the
problem of underutilization of TPU systolic arrays. In their
work they create of profile for idleness of the MAC units for
different batch sizes. Also, they come up with an approach
“UPTPU”, a low overhead power gating solution that can adapt
to various batch sizes and zero weight computations
</div>
<div style='text-align: justify;'>
[8] Norman P. Jouppi et al. evaluate Google’s Tensor
Processing Unit (TPU), a custom ASIC accelerator for neural
network inference deployed in their data centers since 2015.
At the heart of the TPU is a 65,536 8-bit multiply-accumulate
(MAC) matrix unit offering 92 TeraOps/s peak throughput
and a large 28MB software-managed on-chip memory. The
TPU’s deterministic execution model better matches the 99th
percentile response time requirements compared to the varying optimizations of CPUs/GPUs aimed at boosting average
throughput. The TPU’s relatively small size and low power are
attributed to the lack of such complex features. Benchmarking
using production neural nets representing 95% of datacenter
inference demand, the TPU demonstrated 15X-30X higher
performance and 30X-80X better TOPS/Watt compared to
contemporary Haswell CPUs and K80 GPUs. Using the GPU’s
GDDR5 memory could potentially triple the TPU’s TOPS and
boost TOPS/Watt to 70X the GPU and 200X the CPU.
</div>
<div style='text-align: justify;'>
[9] Yang Ni et al. perform comprehensive characterization
of the performance and power consumption of Google’s Edge
TPU accelerator for deep learning inference. They generate
over 10,000 neural network models and measure their execution time and power on the Edge TPU. Key findings reveal non-linear relationships between metrics like the number
of MACs and performance/power. Critical factors like onchip/off-chip memory usage are identified as having significant
impact. Based on this characterization, the authors propose
PETET, a machine learning-based modeling framework that
can accurately predict Edge TPU performance and power
consumption online with less than 10% error for new models.
</div>
<div style='text-align: justify;'>
[10] Kiran Seshadri et.al provide an in-depth analysis of the
Fig. 3. Binary multiplication for 8 bits
microarchitecture and performance characteristics of Google’s
Edge TPU accelerators for low-power edge devices. The
authors first discuss the key microarchitectural details of three
different classes of Edge TPUs spanning different computing
ecosystems. They then present an extensive evaluation across
423K unique convolutional neural network (CNN) models to
comprehensively study how these accelerators perform with
varying CNN structures.
</div>

## Functional Requirements

The proposed MAC unit should meet the following requirements:
- Implement high-performance multiplication and addition circuits capable of performing parallel multiply accumulate operations.
- Support configurable precision data formats to accommodate different neural network models and applications.
- Ensure low latenct and high throughput for the core matrix multiplication operations
- Implement strategies for efficient accumulation and storage of partial results.
- The operating frequency of atleast 1.2GHz.
- The MAC unit should satisfy equation (5).
- The inputs to the weights and inputs that the MAC unit accepts is 8 bits each.
- The final output is 24 bits.
- The multiplier should produce a 16 bit output as shown in Fig. 3.


![Tensor Processor Unit](./img/Screenshot%202024-06-05%20224412.png)