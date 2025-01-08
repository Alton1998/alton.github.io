---
comments: true
---
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

The proposed MAC unit should meet the following requirements:
<div style='text-align: justify;'>
• Implement high-performance multiplication and addition circuits capable of performing parallel multiplyaccumulate operations.
</div>
<div style='text-align: justify;'>
• Support configurable precision data formats to accommodate different neural network models and applications.
</div>
<div style='text-align: justify;'>
• Ensure low latenct and high throughput for the core
matrix multiplication operations
</div>
<div style='text-align: justify;'>
• Implement strategies for efficient accumulation and storage of partial results.
</div>
<div style='text-align: justify;'>
• The operating frequency of atleast 1.2GHz.
</div>
<div style='text-align: justify;'>
• The MAC unit should satisfy equation (5).
</div>
<div style='text-align: justify;'>
• The inputs to the weights and inputs that the MAC unit
accepts is 8 bits each.
</div>
<div style='text-align: justify;'>
• The final output is 24 bits.
</div>
<div style='text-align: justify;'>
• The multiplier should produce a 16 bit output as shown
in Fig. 3.
</div>


![Tensor Processor Unit](./img/Screenshot%202024-06-05%20224412.png)

## Design
In this section we discuss the design we wish to implement in our project.

### Multiplier

<div style='text-align: justify;'>
In the design of the multiplier as shown in Fig. 4we make
use of scan flip flops that allows us to load the values and
shift them. Load bit stays high for one bit to allow us to load
the values in the scan registers and in the next clock cycle the load bit is low which allows us to shift the values. Fig. 5
shows how a scan flip flop is designed using multiplexer and
D flip-flops.
</div>

![Multiplier](./img/Screenshot%202024-06-05%20230828.png)

### Adder
<div style='text-align: justify;'>
We use a carry look ahead adder(CLA) in the MAC unit
as shown in Fig. 7. Table I has the truth table for the carry
lookahead adder. Using 3 8 bit CLA Adders we create a 24
bit adder as show in Fig.
</div>

![Adder Equations](./img/Screenshot%202024-06-05%20231225.png)

## Design Alternatives
An alternative to the MAC unit is discussed in this section.

### Multiplier
The hardware needed here if N was the number of bits we
would need 8x8 hardware as shown in Fig. 8 and is much
faster.

![CLA Design](./img/Screenshot%202024-06-05%20232338.png)

![CLA Design 2](./img/Screenshot%202024-06-05%20232851.png)

![CLA Truth Table](./img/Screenshot%202024-06-05%20232953.png)

### Adder
An alternative adder would be the Carry Select Adder which
is one of the fastest adders as shown in Fig. 9

## Design Calculations

### Determining NMOS/PMOS ratio
This ratio helps us size opir pmos given a nmos width. It is
common for us to make use of equation 10. However, in reality we consider the ratio to be √
2
To find this ratio we find the delays of 1-0 and 0-1
transitions and the rise and fall times. Ideally we want all
these times to be equal, but its not possible.

### Determining Fastest Clock Period

To determine the fastest clock cycle we need to run our
simulations is Fast-Fast process variation we use the equations
(11), (12) and (13). But, for our implementation we will lean
towards equation (13).

### Determining Power Consumption
For practical purposes we will calculate the power consumption we will use equation (14).

![Equations](./img/Screenshot%202024-06-06%20003007.png)



![Multiplier Design](./img/Screenshot%202024-06-05%20234303.png)

![Alternative Adder Design](./img/Screenshot%202024-06-05%20235542.png)

### Floor Plan and Area Calculations
In this section before implementing the design we draw out
the floor plan for each of circuits. Fig. 10, 11 and 12 show
the floor plan design and area calculated.

### Input Output Signals and Power
From Fig. 11 the input signals are
I0−I7(inputs), W0−W7(weights), A0−A23(partial products)

Output Signals from Fig. 11 are
O0 − O2

Power signal from the same diagram is
VDD

GND -
VSS

### SCHEMATIC DESIGN
In all of the schematic design we have used static CMOS
logic. In Table II the sizes and timing information of the
gates used in building the multipliers and adders has been
summarized.

![floor Plan](./img/Screenshot%202024-06-06%20005143.png)

## Physical Layout Design
<div style='text-align: justify;'>
The physical layout was made using the sticks diagram such
that we tried to use merged contacts as much as possible. The
design approach for the layouts was as follows:
</div>
<div style='text-align: justify;'>
• First a graph representation of the schematics for all our
circuits was created.
</div>
<div style='text-align: justify;'>
• We tried to create a Eulerian path such the number of
diffusion regions was reduced.
</div>
<div style='text-align: justify;'>
• Create short wires but using higher level of metal. In our
implementation upto Metal 3 was used.
</div>
<div style='text-align: justify;'>
• The VDD and VSS signals were created with Metal 1
layers
</div>


## SUMMARY OF DATA FLOW

Two 8bit numbers are loaded into the multiplier in the MAC
unit, The values are loaded by driving the LOAD bit high for
atleast one clock cycle and remains low for the rest. Also,
the partial products are loaded into the adder. The multiplier
performs shifting and ANDs the outputs of the scan registers
and sends the output to the adder.

## Results

We made a full custom MAC unit whose area is 250 x 173
µm
and can operate at frequency of 0.167 GHz

![Multiplier Floor Plan](./img/Screenshot%202024-06-06%20010155.png)

![Tables](./img/Screenshot%202024-06-06%20010316.png)

![AND Design](./img/Screenshot%202024-06-06%20010446.png)

![All Designs](./img/Screenshot%202024-06-06%20010610.png)

## Future Scope

For the future scope of this project re-design the MAC unit
with a smaller pmos and nmos ratio. Also, we could redesign
our circuits with adiabatic logic.

## Conclusion
Circuit sizes are much larger thatn the 45nm standard cell
library. We should use an nmos size that is ≤ 1um, since we
did not do that our circuits are much larger. The MAC unit so
designed in this project is not suitalbe for scalar multiplication
since we need speeds ≥ 1GHz. Scan Flip Flop Multiplier uses
less hardware but we need to synchronize when we load and
shift patterns, so clocking is more complicated.
## References
[1] R. Ferenc, T. Viszkok, T. Aladics, J. J ´ asz, and P. Heged ´ us, “Deep- ˝
water framework: The Swiss army knife of humans working with
machine learning models,” SoftwareX, vol. 12, p. 100551, Jul. 2020,
doi: 10.1016/j.softx.2020.100551.
[2] N. Shrivastava, M. A. Hanif, S. Mittal, S. R. Sarangi, and M. Shafique,
“A survey of hardware architectures for generative adversarial networks,”
Journal of Systems Architecture, vol. 118, p. 102227, Sep. 2021, doi:
10.1016/j.sysarc.2021.102227.
[3] J. Zhang, K. Rangineni, Z. Ghodsi, and S. Garg, “Thundervolt,” Research Gate, Jun. 2018, doi: 10.1145/3195970.3196129.
[4] K.-C. Hsu and H.-W. Tseng, “Accelerating applications using edge
tensor processing units,” SC ’21: Proceedings of the International
Conference for High Performance Computing, Networking, Storage and
Analysis, pp. 1–14, Nov. 2021, doi: 10.1145/3458817.3476177.
[5] A. G. M. Lewis, J. Beall, M. Ganahl, M. Hauru, S. B. Mallick, and G. Vidal, “Large-scale distributed linear algebra with tensor processing units,”
Proceedings of the National Academy of Sciences of the United States
of America, vol. 119, no. 33, Aug. 2022, doi: 10.1073/pnas.2122762119.
[6] P. Pandey, P. Basu, K. Chakraborty, and S. Roy, “GreenTPU,” DAC ’19:
Proceedings of the 56th Annual Design Automation Conference 2019,
pp. 1–6, Jun. 2019, doi: 10.1145/3316781.3317835.
[7] P. Pandey, N. D. Gundi, K. Chakraborty and S. Roy, ”UPTPU: Improving
Energy Efficiency of a Tensor Processing Unit through Underutilization Based Power-Gating,” 2021 58th ACM/IEEE Design Automation
Conference (DAC), San Francisco, CA, USA, 2021, pp. 325-330, doi:
10.1109/DAC18074.2021.9586224.
[8] Norman P. Jouppi, Cliff Young, Nishant Patil, David Patterson, et
al., Google, Inc., Mountain View, CA USA 2017. In-Datacenter
Performance Analysis of a Tensor Processing Unit. In Proceedings
of ISCA ’17, Toronto, ON, Canada, June 24-28, 2017, 12 pages.
https://doi.org/10.1145/3079856.308024.
[9] Ni, Y., Kim Y., Rosing, T., Imani, M. (2022). Online performance
and power Prediction for Edge TPU via comprehensive characterization. 2022 Design, Automation ; Test in Europe Conference .
https://doi.org/10.23919/date54114.2022.9774764.
[10] Yazdanbakhsh, A., Seshadri, K., Akin, B., Laudon, J., Narayanaswami,
R. (2022). An Evaluation of Edge TPU Accelerators for Convolutional
Neural Networks. IEEE International Symposium on Workload Characterization (IISWC). https://doi.org/10.1109/iiswc55918.2022.00017.






