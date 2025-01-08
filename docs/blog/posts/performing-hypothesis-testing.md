---
draft: false 
comments: true
date: 2024-08-23
authors:
  - alton
categories:
  - Statistics
  - Hypothesis Testing
  - Interview Question 
---
# Hypothesis Testing

It's a statistical method used to determine whether a hypothesis about a population is true or not. It involves collection data, analyzing it, and making a decision based on a the evidence

## Steps

### Step 1: State your null and alternate hypothesis

The null hypothesis is a prediction of no relationship between variables you are interested in. The alternate hypothesis on the other hand is your hypothesis that predicts a relationship between variables.

#### Examples

1. You want to test whether there is relationship between gender and height. Based on your knowledge of human physiology, taller than women. To test this hypothesis you restate it as:

1. H<sub>0</sub> : Men are, on average, not taller than women
2. H<sub>a</sub>: Men are, on average, taller than women.

#### Some Guidelines when using mathematical symbols

|   H<sub>0</sub>  | H<sub>a</sub> |
| ----------- | ----------- |
| Equal (=)   | Not equal ($\neq$)       |
| Greater Than or equal to ($\geq$)   | Less than ($\lt$)        |
| Less than or equal to ($\leq$)                 | Greater than ($\gt$)             |

##### Examples 
We want to test whether the mean GPA of students in American colleges is different from 2.0.

The null and alternative hypothese are 

H<sub>0</sub>: $\mu$ = 2.0

H<sub>a</sub>: $\mu$ $\ne$ 2.0

### Steps 2 : Perform an appropriate statistical test

For this step we perform something known as the t-test. A t-test is any statistical hypothesis test in which the test statistic follows  a t-distribution under the null hypothesis.

A t-test is most commonly applied when the test statistic would follow a normal distribution if the value of a scaling term in the test statistic were known.

The t-test can be used to determine if the means of two sets of data are significantly different from each other.

An independent Samples t-test compares the means for two groups. 

A paired sample t-test compares means from the same  group at different times 

A one sample t-test test the mean of a single group against a known mean.

T-score is a ration between 2 groups and the difference within the groups.

The larger the t score, the more difference there is between groups. 

The smaller the t score, the more similarity there is between groups.

Every t-score has a p-value to go with it. 

A p-value is th probability that results that your sample data occured by chance.

P-values are from 0% to 100%

Low p-values are good. They indicate that your data did not occur by chance.

### Step 3: Decide Whether to reject or accept your null hypothesis.

To understand this step let us solve a problem:

Suppose a sample of n students were give a diagnostic test before studying a particular module and then again after completing the module. We want to find out if in general teaching leads to improvements in students knowledge/skills. We can use the results from our sample of students to draw concludsion about the impact of this module in general.

So since we are calculating the mean of the same sample at different points in time we will be using the Pairesd t-test.

Null hypothesis - There is no difference after completing the module.
Alternate Hypothesis - There is a difference after completing the module.

Calculate the difference between the two observations i.e. di = yi - xi.

Calculate the mean difference d

Calcualte the standard deviation of the differences, S<sub>d</sub> and use this to calculate the standard error of the mean difference, SE(d) = $\frac{Sd}{\sqrt{n}}$

Calculate the T value

t = $\frac{d}{SE(d)}$

then use a table of t value to look up the p-values for the paired t-test.


