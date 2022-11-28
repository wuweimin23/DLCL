# <div align="center">Pre-Parameterization And Differentiable Layer For Constrained Lasso</div>

## Introduction

<p align="justify">Constrained Lasso has been a widely used problem in convex optimization. Most existing works concentrate on constructing fast algorithm, without considering the learned method. Nowadays, the learned method has been applied as an efficient and effective framework and it can bring various benefits. This work introduces pre-parameterization method to constrained Lasso problem, which embeds the optimization problem into a larger space by parameterizing the problem data, and it provides a learned framework. Take the precision matrix estimation in graphical model learning as an example, existing algorithms focused on solving the optimization problem derived from the maximum likelihood estimation. However, there was no guarantee that such a definition of the optimization problem was the best one. Thus this method can be trained to search for the best optimization problem. To train the pre-parameterization framework efficiently, we also develop a GPU-based differentiable layer for constrained Lasso (DLCL) based on ADMM and implicit differentiation of KKT condition. Finally, we evaluate our methods on precision matrix estimation, and we implement numerical experiments based on synthetic data to demonstrate the capability of our methods. 

## Constrained Lasso Problem

$$
    \begin{aligned}
    \underset{x}{\operatorname{minimize}} &  \frac{1}{2} x^{T} \Sigma x + b^{T}x + \lambda ||x||_{1} \\
    \text{subject to } &  Bx = c \\
      &   Dx \leq g \\
    \end{aligned}
$$

where $x \in \mathbb{R}^{n}$ is the variable, $\Sigma \in \mathbb{R}^{n \times n}$ (a positive semi-definite matrix), $b \in \mathbb{R}^{n}$, $B \in \mathbb{R}^{m \times n}$, $c \in \mathbb{R}^{m}$, $D \in \mathbb{R}^{k \times n}$, $g \in \mathbb{R}^{k}$ are the problem data, and $\lambda$ is a scalar.

## Pre-Parameterization Method

$$
    \begin{aligned}
    \Sigma & = LL^{T} \\
    \hat{L} & = L + W_{1}L + W_{2} \\
    \hat{\Sigma} &  = \hat{L}\hat{L}^{T} \\
    \hat{b} & =b + W_{3}b + W_{4}
    \end{aligned}
$$

where $W_{1}, W_{2}, W_{3} \in \mathbb{R}^{n \times n}, W_{4} \in \mathbb{R}^{n}$ are the trainable parameters.

## Training
The code implements the numerical experiments for one of the three settings in paper 'Pre-Parameterization And Differentiable Layer For Constrained Lasso'

## Results

Pre-parameterization method brings 9.0% to 70.7% improvement on the precision matrix estimation. And DLCL runs highly faster than existing differentiable layers for large-scale optimization problems (nearly 10 times faster than OptNet, and nearly 100 times faster than CvxpyLayer). </p>
