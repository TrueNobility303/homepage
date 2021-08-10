# 图卷积神经网络GCN

论文阅读笔记：[Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907v4)

GCN（图卷积神经网络）是图神经网络中的杰出代表，文本主要推到其理论来源。



首先，GCN依托于ChebConv（切比雪夫图卷积），因此要先理解GCN，应该先理解ChebConv。而要理解ChebConv必须先从图傅里叶变换出发，因此本文由以下部分组成：

* 图卷积与图傅里叶变换
* ChebConv
* GCN



## 图卷积与图傅里叶变换

首先，由图傅里叶变换出发。



给定图$G$,定义拉普拉斯矩阵$L=D-W$, 其中$W$为边权，对角阵$D$表示节点的度数，$D_{ii}=\sum_j W_{ij}$，给定图的结点特征$X$,

有$E = tr(X^TLX)=\frac{1}{2}\sum_{i,j}w_{ij}\Vert x_i - x_j\Vert_2$,$E$可以视作衡量$G$的能量,同时也衡量了图的平滑度等，当相邻两个结点的特征差距$\Vert x_i-x_j\Vert_2$越大时，图的能量$E$越大，

当边无边权的时候，边权可由邻接矩阵定义，即$W=A$,

由于$L$为正定对称矩阵，$L$存在谱分解，$L=U^T \Lambda U$, $E=tr(X^T U^T \Lambda U X)$, 令$Y = UX$ ,则$E=tr(Y^T \Lambda Y)$, 

 ，将信号$X$转换为另一个空间的信号$Y$,且$Y$对于衡量信号$X$的变化，也即频率有很大的作用，如果将$X$所在的子空间视为空间域，$Y$所在的空间视为频率域，则变换算子$U$,起到了类似于傅里叶变换的效果，将该变换方式称为图傅里叶变换。

由于$U$为正交矩阵，$U^TU=I$, 则傅里叶变换$U$对应的反傅里叶变换为$U^{-1}=U^T$.

对于在$U$对应的像空间中的向量$y$, $y$可以表示为一组标准正交基$\{u_i\}$的线性组合$y = \sum_i x_i u_i$, 则$y^T \Lambda y = \sum_i \lambda_i x_i^2$，如果$U$选择部分特征向量$\{u_k\}$对应的子空间，则获得的$y$只得到了$\{u_k\}$对应的特征$x$ 的信号,比如选取$U = [u_1,0,...,0]$ , 则$y = \lambda_1 x_1^2$ , 如果将小特征值$\lambda_1$视作低频信号的表征，则上述例子中选取的$U$相当于对$x$进行了一次低通滤波。



由傅里叶变换的卷积定理，在空间域的卷积等价于在频率域的$Hadama$乘积$\odot$,用该性质定义图卷积，


$$
U(f *g) := (Uf) \odot (Ug)
$$


用可学习的卷积核

$$
g_{\theta} (\Lambda)=diag(g(\theta)) = g(diag(\theta))
$$

对空间域的$X$做卷积得到同样在空间域的另一信号$Y$。

上述卷积也等价于使用$Hadama$乘积对$X$对应的频率域信号$\tilde X$与卷积核进行逐点相乘得到同样在频率域的信号$\tilde Y$ ,用矩阵表示如下，

$$
\begin{align}
g_{\theta}(\Lambda) X &=  \Theta \odot X \\
\tilde Y &= U Y \\
\tilde X &= U X
\end{align}
$$


代入得，

$$
\begin{align}
\tilde Y &= g_{\theta} (\Lambda) \tilde X \\
UY &= g_{\theta} (\Lambda) U X \\
Y &= U^T g_{\theta} (\Lambda) U X \\
\end{align}
$$

其中卷积核的参数$\theta$ 为可学习参数，而该多项式本身有应该和图的特征值$\Lambda$ 相关，才能得到图的频率特征等。比如，令$g_{\theta} (\Lambda) = \Lambda^{\frac{1}{2}}$ ,则 

$$
\begin{align}
Y &= U^T\Lambda^{\frac{1}{2}} U X = L^{\frac{1}{2}} X \\
E &=tr(Y^TY) = tr(X^T L X)
\end{align}
$$

此时的$Y$可以认为储存了能量信息。



## ChebConv

ChebConv是一种可学习的图卷积操作：



用多项式表示卷积核，也即

$$
g_{\theta} (\Lambda) = \sum \theta_k \Lambda^k
$$

则结点分类任务可以转化为选取特定的$\theta$ ,使得图上的结点特征（原信号），经过上述图卷积运算后得到标签特征（目标信号），而$\theta$ 可以通过梯度下降算法迭代计算得到。同时，在实际中，可以使用多次图卷积操作堆叠得到最终的信号，而不一定只能进行一次图卷积。

但计算该多项式的复杂度很高，如果使用切比雪夫多项式定义,

$$
\begin{align}
&g_{\theta}(\Lambda) = \sum_k \theta_kT_k(\tilde \Lambda) \\
&\tilde \Lambda = 2 \Lambda / \lambda_{max} - I 
\end{align}
$$

其中$T_k$为k阶切比雪夫多项式，上述使用$\tilde \Lambda$ 定义的原因是使得满足切比雪夫多项式的定义域$[-1,1]$,

由于$T_k$可以递归计算，

$$
\begin{align}
T_0(x) &= 1 \\
T_1(x) &= x \\
T_k(x) &= 2xT_{k-1}(x) - T_{k-2}(x) \\
\end{align}
$$

所以用切比雪夫多项式定义的卷积核可以大大加速计算效率。

由此，定义了ChebConv  ， 本质可以理解为一种可学习的图信号处理的手段。



## GCN

有了ChebConv的基础之后，我们终于可以进入GCN的最终公式了。



利用ChebConv中的结论，我们来回顾一下，

首先归一化拉普拉斯矩阵$L$ ,并不影响结果，

$$
L = I - D^{-\frac{1}{2} } A D^{-\frac{1}{2}}
$$

又由$Lx = \Lambda x$,

$$
\begin{align}
&\tilde Y = g_{\theta} (\tilde \Lambda) U X  = g_{\theta} (\tilde L) X \\
& \tilde L = 2L / \lambda_{max} - I
\end{align}
$$

由于$L$的最大特征值不超过2，该结论可由随机矩阵的性质，或用反证法等推出，此处从略。

假设$\lambda_{max} \approx 2$ ,代入得

$$
\tilde L = L - I = - D^{-\frac{1}{2} } A D^{-\frac{1}{2}}
$$

取$k=2$,

$$
g_{\theta} (\tilde L) X = \sum_k \theta_kT_k(\tilde L) = \theta_0 I - \theta_1 D^{-\frac{1}{2} } A D^{-\frac{1}{2}}
$$

假设$\theta_0 = -\theta_1 =\theta$ ,则

$$
g_{\theta} (\tilde L) X =  \theta (I + D^{-\frac{1}{2} } A D^{-\frac{1}{2}}) X
$$

为了计算方便，改动该公式，相当于对邻接矩阵先添加自环$I$后再用度数$D$进行归一化操作，

改动的原因是，如果使用

$$
\tilde A :=I + D^{-\frac{1}{2} } A D^{-\frac{1}{2}}
$$

则该矩阵的特征值范围为$[0,2]$, 如果多次迭代进行会导致数值不稳定的情况发生，若改为，

$$
\tilde A := D^{-\frac{1}{2} } (A+I) D^{-\frac{1}{2}}
$$

则每次归一化保证矩阵的特征值属于$[0,1]$ ,避免了上述情况，

$$
g_{\theta} (\tilde L) X =   \tilde A  X \Theta  \\
\tilde A := D^{-\frac{1}{2} } (A+I) D^{-\frac{1}{2}}
$$

使用两层GCN的话，为了增加神经网络拟合函数的非线性性，在每一层之间加入激活函数Relu，而最后一层采用Softmax函数使得输出为结点属于每一个类别的概率，得到最终的公式，

$$
Z = f(X,\Theta) = Softmax(\tilde A \ Relu(\tilde A XW_0) \ W_1) 
$$


其实从ChebConv到GCN的推导中做了很多近似，但GCN的效果很好，笔者理解是利用了GCN的强大的拟合能力去减小这些近似的影响，或者可以认为神经网络可以自动地学习到这些近似成立的条件。
