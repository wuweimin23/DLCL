## Introduction
Constrained Lasso has been a widely used problem in convex optimization. Most existing works concentrate on constructing fast algorithm, without considering the learned method. Nowadays, the learned method has been applied as an efficient and effective framework and it can bring various benefits. This work introduces pre-parameterization method to constrained Lasso problem, which embeds the optimization problem into a larger space by parameterizing the problem data, and it provides a leaned framework. Take the precision matrix estimation as an example, this method can be trained to construct a better optimization problem to estimate the precision matrix. To train the pre-parameterization framework efficiently, we also develop a GPU-based differentiable layer for constrained Lasso (DLCL) based on ADMM and implicit differentiation of KKT condition. Finally, we evaluate our methods on precision matrix estimation, and we implement numerical experiments based on synthetic data to demonstrate the capability of our methods. Pre-parameterization method brings 9.0% to 70.7% improvement on the precision matrix estimation. And DLCL runs highly faster than existing differentiable layers for large-scale optimization problems (nearly 10 times faster than OptNet, and nearly 100 times faster than CvxpyLayer). 

## Constrained Lasso Problem
$$
\underset{x}{\operatorname{minimize}}  \frac{1}{2} x^{T} \Sigma x + b^{T}x + \lambda ||x||_{1} 
$$

$$
\text{subject to } Bx = c 
$$

$$
Dx \leq g 
$$

## Parameterized Constrained Lasso Problem

\begin{equation}
    \begin{aligned}
    & \Sigma =  LL^{T} \\
    &\hat{L} = L + W_{2} \\
    &\hat{\Sigma}  =  \hat{L}\hat{L}^{T} \\
    &\hat{b}  =  b + W_{3}b \\
    \underset{x}{\operatorname{minimize}} & \frac{1}{2} x^{T} \hat{\Sigma} x + \hat{b}^{T}x + \lambda ||x||_{1} \\
    \text{subject to } &  Bx = c \\
      &  Dx \leq g \\
    \end{aligned}
\end{equation}

## Training
The code implements the numerical experiments for one of the three settings in paper "Pre-Parameterization And Differentiable Layer For Constrained Lasso"