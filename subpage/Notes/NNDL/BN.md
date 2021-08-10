# 神奇的BatchNorm

论文阅读笔记：[How Does Batch Normalization Help Optimization? NIPS'18](https://arxiv.org/pdf/1805.11604v5.pdf)

Batch Normalization (BatchNorm,BN,中文名批标准化)自问世以来，已经基本成为神经网络的标配。很多不收敛的网络，使用BN都可以化腐朽为神奇，即刻收敛。但人们对于BN的原理一直不甚了然。该文章给出了一系列BN的性质，对BN的神奇性做出解释。



### 2.5 Proof

该章节分为几个部分分析BN

* gradient in BN network: 理论基础, 讨论BN在反向传播中如何求导
* loss landscape：证明BN使得损失函数关于输入的导数更为平滑
* gradient predictive：证明BN使得损失函数关于神经网络权重的导数更为平滑
* smoothness measured by second order ：证明BN网络的Hesson矩阵在某种度量意义下也更为平滑
* better initialization given by BN：BN为神经网路的训练提供更接近最优解的初值



从BN的导数出发，文章证明了加入BN的网络在一阶意义上、二阶意义上都更为平滑，因此更加易于学习和训练；同时，BN的尺度不变性质也使得网络训练通常具有更好的初始值。



#### 2.5.1 gradient in BN network



首先，回顾BN的公式，


$$
\begin{align}
\mu &= \frac{1}{m} \sum_i x_i \\
\sigma^2 &= \frac{1}{m} \sum_i (x_i - u)^2 \\
\hat{x_i} &= \frac{x_i-u}{\sigma} \\
y_i &=  \gamma \hat{x_i} + \beta
\end{align}
$$



为了求BN中需要用到的梯度，我们关心$\frac{\partial f}{\partial x_i}$,但再次之前我们先求几个量，


$$
\begin{align}
\frac{\partial f}{\partial \mu} 
&= \sum_i\frac{\partial f}{\partial \hat{x_i}} \frac{\partial \hat{x_i}}{\partial \mu} + \frac{\partial f}{\partial \sigma^2} \frac{\partial \sigma^2}{\partial \mu} \\
&=\sum_i \frac{\partial f}{\partial \hat{x_i}}(-\frac{1}{\sigma}) +  \frac{\partial f}{\partial \sigma^2} (-\frac{2}{m} (x_i-u)) \\
&= \sum_i \frac{\partial f}{\partial \hat{x_i}}(-\frac{1}{\sigma}) \\
 
\frac{\partial f}{\partial \sigma^2}
&= \sum_i\frac{\partial f}{\partial \hat{x_i}}\frac{\partial \hat{x_i}}{\partial \sigma^2} \\
&= \sum_i\frac{\partial f}{\partial \hat{x_i}} (-\frac{x_i-\mu}{2 \sigma^3})
\end{align}
$$

最后求，


$$
\begin{align}
\frac{\partial f}{\partial x_i}  
&= \gamma(\frac{\partial f}{\partial \hat{x_i}} \frac{\partial \hat{x_i}}{\partial x_i} + \frac{\partial f}{\partial \mu} \frac{\partial \mu}{\partial x_i} + \frac{\partial f}{\partial \sigma^2} \frac{\partial \sigma^2}{\partial x_i} )\\
&= \gamma(\frac{\partial f}{\partial \hat{x_i}} \frac{1}{\sigma} + \sum_i \frac{\partial f}{\partial \hat{x_i}}(-\frac{1}{\sigma}) \frac{1}{m} + \sum_i\frac{\partial f}{\partial \hat{x_i}} (-\frac{x_i-\mu}{2 \sigma^3})(\frac{2(x_i-\mu)}{m})) \\
&= \frac{\gamma}{m\sigma} (m \frac{\partial f}{\partial \hat{x_i}} - \sum_j \frac{\partial f}{\partial \hat{x_j}} + \hat{x_i} \sum_j \hat{x_j} \frac{\partial f}{\partial \hat{x_j}})
\end{align}
$$



#### 2.5.2 loss landscape

对上式两端同时取2范数，


$$
\begin{align}
\Vert  \frac{\partial f}{\partial x}\Vert  _2 
&= (\frac{\gamma}{m \sigma})^2 \Vert  m \frac{\partial f}{\partial \hat{x_i}} - \sum_j \frac{\partial f}{\partial \hat{x_j}} + \hat{x_i} \sum_j \hat{x_j} \frac{\partial f}{\partial \hat{x_j}} \Vert  _2 \\
&= (\frac{\gamma}{m \sigma})^2 \Vert   m \frac{\partial f}{\partial \hat{x}} - e\langle e,\frac{\partial f}{\partial \hat{x}}\rangle + \hat{x} \langle\hat{x}, \frac{\partial f}{\partial \hat{x}}\rangle \Vert  _2
\end{align}
$$



由于$\hat{x}$ 具有标准均值和方差，



$$
\begin{align}
\langle e,x \rangle &= 0 \\
\langle x,x \rangle &= m \\
\langle e,e \rangle &= m
\end{align}
$$



利用上式可化简，


$$
\begin{align}
\Vert  \frac{\partial \hat{f}}{\partial x}\Vert  _2 
&= (\frac{\gamma}{m \sigma})^2 \Vert   m \frac{\partial f}{\partial \hat{x}} - e\langle e,\frac{\partial f}{\partial \hat{x}} \rangle + \hat{x} \langle \hat{x}, \frac{\partial f}{\partial \hat{x}}\rangle \Vert  _2 \\
&= (\frac{\gamma}{m \sigma})^2(m^2 \Vert  \nabla_x L\Vert  ^2_2 -m \langle e,\nabla_x L\rangle^2 -m\langle x,\nabla_x L \rangle^2)
\end{align}
$$



其中，$\nabla_x L$ 即为有BN网络的梯度，而左端为有BN网络的梯度$\nabla_x \hat{L}$，


$$
\Vert  \nabla_x{\hat{L}}\Vert  _2 = (\frac{\gamma}{\sigma})^2(\Vert  \nabla_x L\Vert  ^2_2 -\frac{1}{m} \langle e,\nabla _xL\rangle^2 -\frac{1}{m}\langle x,\nabla_x L \rangle^2 )
$$



当$\frac{\gamma}{\sigma}\le 1$ 的前提下，由于$\gamma$ 作为可学习参数，我们可以假设我们的网络可以学到其作为较小的值，而$\sigma$ 作为数据的标准差通常也是一个较大的值，所以该假设其实并不难达到。

上面曾经提及猜想，BN在训练的后期的平滑作用更为明显，此处也可以作为一个简要的补充。因为在训练后期，网络更有可能学到更小的$\gamma$ 值，使得网络的权重空间非常平滑。



在上述假设之下，显然有，


$$
\Vert  \nabla_x \hat{L}\Vert  _2 \le \Vert  \nabla_x L\Vert  _2
$$



#### 2.5.3 gradient predictive

由链式法则，$\nabla_x L = X^T \nabla_WL$


$$
\hat{g} := \max_{\Vert  X\Vert  _2 \le \lambda} \Vert  \nabla_W \hat{L}\Vert  _2 = \max_{\Vert  X\Vert  _2 \le \lambda} \nabla_x \hat{L} X^TX \nabla_x \hat{L} = \lambda^2 \Vert  \nabla_x \hat{L}\Vert  _2
$$



类似地，


$$
g:= \lambda^2 \Vert  \nabla_xL\Vert  _2
$$

$$
\hat{g} = (\frac{\gamma}{\sigma})^2(g -\frac{\lambda^2}{m} \langle e,\nabla _xL\rangle^2 -\frac{\lambda^2}{m}\langle x,\nabla_x L \rangle^2 )
$$



同理我们知道，在大多是情况下，


$$
\hat{g} \le g
$$





#### 2.5.4 smoothness measured by second order  

上面本质上是证明了BN网络的Lipschitz性质，类似地还可以证明BN网络的平滑性，由于证明较为繁琐，此处从略。

结论为，


$$
\hat{\nabla}^T \hat{H}\hat\nabla \le (\frac{\gamma}{\sigma})^2 \nabla^T H \nabla - \frac{\gamma}{m \sigma^2} \langle\nabla,x \rangle \Vert  \nabla\Vert  _2^2 \\
$$


类似地，定义左式在条件$\Vert    X\Vert    _2 \le \lambda$ 的最大值，也可以得到类似的最大值不等式，此处从略。



#### 2.5.5 better initialization given by BN

还可以证明，使用BN的网络，可以使得初始化的权重$W_0$ 离最优点的位置更靠近，本质是BN具有尺度不变性。

设无BN网络的最优权重为$W^*$ , 则由于BN的尺度不变性，$\forall k, \hat{W^*} = k W^*$ 也是BN网络的最优解。



$$
\begin{align}
\Vert  W_0 -\hat{W^*}\Vert  _2^2 - \Vert  W_0 - W^*\Vert  _2^2 = \Vert  W_0 -k W^*\Vert  _2^2 - \Vert  W_0 - W^*\Vert  _2^2 = (k^2-1 )\Vert  W^*\Vert  _2^2 - 2(k-1)\langle W_0,W^*\rangle
\end{align}
$$



取$k = \frac{\langle W_0, W^* \rangle}{\Vert   W^*\Vert   _2^2}$,则


$$
\begin{align}
\Vert  W_0 -\hat{W^*}\Vert  _2^2 - \Vert  W_0 - W^*\Vert  _2^2 &= (k^2 -1) \Vert  W^*\Vert  _2^2 -2k(k-1)\Vert  W^*\Vert  _2^2 
\\&= -(k-1)^2\Vert  W^*\Vert  _2^2 \\
& \le 0
\end{align}
$$


那么换一个角度，我们可以认为BN网络有更优的权重初始化。
