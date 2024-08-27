---
draft: false 
date: 2024-08-26
authors:
  - alton
categories:
  - LLMs
  - Tensorflow
  - Pytorch
  - Computational Graphs
  - Partial Differentiation
  - DAG
---

# Computational Graphs

These are Directed Graphs that helps map out dependencies for mathematical computations. For Example let us consider the following set of equations:

1. Y=(a-b)*(a+b)
2. Let d=(a-b) and e=(a+b)

Our dependency graph will look as follows:

![Graph Example](./pics/Graph.png)

The lower nodes are evaluated first then the higher nodes are evaluated.

Let us consider how this works when performing chain differentiation when it comes to neural networks. 

To review chain differentiation consider the following equation:

1. y = $u^4$
2. u = 3x + 2 

Performing chain rule differentiation with respect to x we would get the follolwing:

$$\frac{\partial y(u)}{\partial x } = \frac{\partial (u^4)}{\partial x}$$

$$\frac{\partial u}{\partial x} = \frac{\partial (3x+2)}{\partial x} $$

$$\frac{\partial u}{\partial x} = 3 + 0 $$

$$\frac{\partial u}{\partial x} = 3 $$

$$\frac{\partial( \partial y(u))}{\partial x \partial u} = \frac{\partial (\partial (u^4))}{\partial x \partial u}$$

$$\frac{\partial y(u)} {\partial x } =  \frac{\partial (4u^3)}{\partial x} $$

$$\frac{\partial y(u)} {\partial x } = \frac{4*3 u^2 \partial u}{\partial x}$$

$$\frac{\partial y(u)} {\partial x } = 12*(3x+2)^2 * 3 $$

Representing the above steps in a computational graph we get the following: 

![Chained Computational Graph](./pics/Chained%20Equation.png)

How do we implement this? Luckily this has already been implemented for us in Tensorflow and Pytorch.

There are 2 implementations of Computational Graphs:

1. Static Computational Graphs - Graphs are constructed once befor the execution of the model.
2. Dynamic Computational Graphs - Graphs are constructed on the fly.

## Tensorflow Computation Graph implementation.

```python
import tensorflow as tf
```

    2024-08-27 18:45:09.326809: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-08-27 18:45:09.357051: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-08-27 18:45:09.365983: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-08-27 18:45:09.395484: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.



```python
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    u = 3*x + 2
    y = u ** 4
```

    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1724784312.756873    1081 cuda_executor.cc:1001] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
    Your kernel may have been built without NUMA support.
    I0000 00:00:1724784312.767002    1081 cuda_executor.cc:1001] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
    Your kernel may have been built without NUMA support.
    I0000 00:00:1724784312.767097    1081 cuda_executor.cc:1001] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
    Your kernel may have been built without NUMA support.
    I0000 00:00:1724784312.777530    1081 cuda_executor.cc:1001] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
    Your kernel may have been built without NUMA support.
    I0000 00:00:1724784312.777763    1081 cuda_executor.cc:1001] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
    Your kernel may have been built without NUMA support.
    I0000 00:00:1724784312.777895    1081 cuda_executor.cc:1001] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
    Your kernel may have been built without NUMA support.
    I0000 00:00:1724784313.020475    1081 cuda_executor.cc:1001] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
    Your kernel may have been built without NUMA support.
    I0000 00:00:1724784313.020614    1081 cuda_executor.cc:1001] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
    Your kernel may have been built without NUMA support.
    2024-08-27 18:45:13.020636: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2112] Could not identify NUMA node of platform GPU id 0, defaulting to 0.  Your kernel may not have been built with NUMA support.
    I0000 00:00:1724784313.020750    1081 cuda_executor.cc:1001] could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
    Your kernel may have been built without NUMA support.
    2024-08-27 18:45:13.020821: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2021] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 1767 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3050 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6



```python
g = tape.gradient(y,x)
```


```python
g
```




    <tf.Tensor: shape=(), dtype=float32, numpy=15972.0>




```python

```


```python

```


```python

```


```python

```


```python

```




## Pytorch Computation Graph Implementation.
```python
import torch
```


```python
x = torch.tensor(3.0, requires_grad=True)
u = 3*x +2
y = u**4
```


```python
y.backward()
```


```python
x.grad
```




    tensor(15972.)




```python

```


```python

```