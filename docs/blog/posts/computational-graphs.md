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