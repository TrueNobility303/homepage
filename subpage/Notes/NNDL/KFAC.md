# KFAC

 论文阅读笔记：[Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://arxiv.org/pdf/1503.05671v7.pdf)

KFAC(Kronecker-factoredApproximate Curvature) 是一种基于Kronecker-分解的二阶优化算法，一般神经网络的额优化不使用二阶优化算法，因为计算Hesson矩阵什么的实在太慢了，但KFAC利用一些近似实现了该算法。



## 自然梯度与Fisher信息矩阵

首先，该优化算法定义在自然梯度上，自然梯度即为使用Fisher信息矩阵定义的梯度。


$$
F = E[\nabla \log p(x)  \nabla\log p(x)^T]
$$


注意，为了表示方便，上式做了简化，其实表达的意思是，


$$
\nabla \log p(x) := \nabla_{\theta} log(p|\theta) \\
E[p(x)] = E_{x \sim p(x|\theta)}
$$


首先，我们知道，


$$
\begin{align}
E[\nabla \log p(x)] 
&= \int p(x) \frac{\nabla p(x)}{p(x)} \\ 
&= \int \nabla p(x) \\
&= \nabla \int p(x) \\
& = \nabla 1\\\
& = 0
\end{align}
$$


且当损失为负对数似然概率时，


$$
L = -\log p(x)
$$


对$L$ 求二阶导，


$$
\begin{align}
H &= \nabla^2 L \\
&= -\nabla^2 \log p(x) \\
&= -\nabla \frac{\nabla p(x)}{p(x)} \\
&= \frac{\nabla p(x)}{p(x)}^T \frac{\nabla p(x)}{p(x)} - \frac{p(x)^2}{p(x)}  \\
&= \nabla \log p(x)^T {\nabla \log p(x)} - \frac{\nabla^2 p(x)}{p(x)}
\end{align}
$$


对上式取期望，


$$
\begin{align}
E[H] 
&= E[\nabla \log p(x)^T {\nabla \log p(x)}] - E[\frac{\nabla ^2p(x)}{p(x)}] \\
&= F - \int p(x)  \frac{\nabla ^2p(x)}{p(x)} \\
&= F - \int {\nabla ^2p(x)} \\
&= F - \nabla^2 \int p(x) \\
& = F - \nabla^2 1 \\
&= F
\end{align}
$$


回顾牛顿法的更新公式，


$$
x_{t+1} = x_t - H^{-1} \nabla L(x) 
$$


那么，可以用$F$ 代替$H$ ,相当于用$H$的期望代替了$H$,该更新方式就是自然梯度下降。

其实，自然梯度也和KL散度在流形上的东东等相关，这里就不多说了。

## Fisher信息矩阵的近似及误差分析

为了加速计算，对Fisher信息矩阵的信息尤其关键。



首先，回顾$F$, 


$$
F = E[\nabla \log p(x)  \nabla\log p(x)^T] = E[\nabla L(x) \nabla L(x)^T]
$$



由于$\theta$ 本质为所有层$W$ 的拼接，


$$
\begin{align}
d \theta &:= \nabla_{\theta} L(x) \\
F &= E[\nabla L(x) \nabla L(x)^T] = E[d\theta \ d\theta^T] \\
\theta &= [vec(W_0)^T,vec(W_1)^T,...vec(W_n)^T]^T
\end{align}
$$


代入展开得到，


$$
\begin{align}
F_{ij} = E[vec(dW_i) vec(dW_j)^T] 
\end{align}
$$


令$a_i,g_i$ 分别为第$i$层的前向输入和反向传播梯度，由反向传播算法，其实就是链式求导法则，


$$
d W_i = g_i a_i^T 
$$
 代入，


$$
vec(dW_i) = vec(g_i a_i^T) = a_i \otimes g_i
$$


那么，


$$
\begin{align}
F_{ij} &= E[vec(dW_i) vec(dW_j)^T] \\
&= E[(a_i \otimes g_i) (a_j \otimes g_j)^T] \\
&= E[(a_i a_j^T) \otimes (g_i g_j^T)] \\
& \approx E[a_i a_j^T] E[g_i g_j^T] 

\end{align}
$$


最后一个$\approx$ 为什么成立呢，他等价于说，


$$
E[a^{(1)} a^{(2)} g^{(1)} g^{(2)}] \approx E[a^{(1)} a^{(2)}] E[g^{(1)} g^{(2)}]
$$


其中$a^{(1)}$ 表示向量$a$​的一个分量，



这个近似的误差其实可以用公式写出来，也即我们要计算下式，


$$
E[a^{(1)} a^{(2)} g^{(1)} g^{(2)}] - E[a^{(1)} a^{(2)}] E[g^{(1)} g^{(2)}]
$$



首先，由累积量和矩的关系，可以将$E[a^{(1)} a^{(2)} g^{(1)} g^{(2)}] $展开为一系列累积量之和，由于我们知道，


$$
\begin{align}
E[g^{(i)})] &= E[-\nabla_g \log p(x)] \\
&= -\int p(x) \frac{\nabla_g p(x)}{p(x)} \\
&= -\nabla_g \int p(x) \\
& = \nabla_g 1 \\
&=0
\end{align}
$$


那么，


$$
\begin{align}
Cov(a^{(i)}, g^{(j)}) &= E[ (a^{(i)} - E[a^{(i)}])(g^{(i)}- E[g^{(i)}])] \\
&= E[(a^{(i)} - E[a^{(i)}]) g^{(i)}] \\
&= E[a^{(i)} g^{(i)}]- E[a^{(i)}]E[g^{(i)}] \\
&=  E[a^{(i)} g^{(i)}]
\end{align}
$$


又类似地，


$$
\begin{align}
E[a^{(i)} g^{(i)}] E[g^{(i)})] &= E[- a \nabla_g \log p(x)] \\
&= -\int a \ p(x) \frac{\nabla_g p(x)}{p(x)} \\
&= -\nabla_g \int a \ p(x) \\
& = \nabla_g a \\
&=0
\end{align}
$$


又由于一阶或二阶累积量和矩是等价的，代入可以消去很多项为0的累积量，为了简便，下面用下标代替，


$$
\begin{align}
E[a_1,a_2,g_1,g_2] 
= &\kappa(a_1,a_2,g_1,g_2) \\
+ &\kappa(a_1) \kappa(a_2,g_1,g_2) + \kappa(a_2) \kappa(a_1,g_1,g_2) \\
+ &\kappa(a_1,a_2) \kappa(g_1,g_2) + \kappa(a_1) \kappa(a_2) \kappa(g_1,g_2)

\end{align}
$$


又，


$$
\begin{align}
\kappa(a_1,a_2) \kappa(g_1,g_2) + \kappa(a_1) \kappa(a_2) \kappa(g_1,g_2) &= Cov(a_1,a_2) Cov(g_1,g_2) +E(a_1)E(a_2)Cov(g_1,g_2))  \\
&=(Cov(a_1,a_2)+E(a_1)E(a_2)) Cov(g_1,g_2) \\
&= E(a_1,a_2)E(g_1,g_2)
\end{align}
$$


移项就得到了，


$$
\begin{align}
&E[a_1,a_2,g_1,g_2]  - E(a_1,a_2)E(g_1,g_2)=\kappa(a_1,a_2,g_1,g_2) + \kappa(a_1) \kappa(a_2,g_1,g_2) + \kappa(a_2) \kappa(a_1,g_1,g_2) \\
\end{align}
$$



只要假设所有的右端的累积量都较小，那么我们的近似误差是较小的，


$$
E[a^{(1)} a^{(2)} g^{(1)} g^{(2)}] \approx E[a^{(1)} a^{(2)}] E[g^{(1)} g^{(2)}]
$$



那么，只需要不断地计算$E[a^{(1)} a^{(2)}] E[g^{(1)} g^{(2)}]$ 就可以估计出Fisher矩阵，下面解决$F^{-1}$不存在的情况，

其实也很简单，虽然$F$ 不一定可逆，但$F$ 作为协方差矩阵，是半正定的，加上一个小量就好啦。

$$
F^{-1} \approx (F+\epsilon)^{-1}
$$

嘿嘿！
