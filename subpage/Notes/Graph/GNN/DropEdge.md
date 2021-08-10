# DropEdge

论文阅读笔记：[DropEdge: Towards Deep Graph Convolutional Networks on Node Classification](https://arxiv.org/pdf/1907.10903v4.pdf)



DropEdge旨在解决GCN的过平滑化的缺点。

论文首先阐述了GCN过平滑化的缺点，并且给出理论证明，然后证明了DropEdge的手段，也即每两层GCN之间随机Drop掉一些边可以减缓GCN的过平滑化的缺点。

关于GCN，可以参见: [图卷积网络GCN](https://truenobility303.github.io/subpage/Notes/Graph/GNN/GCN)

## GCN的过平滑化

首先我们证明一下，GCN过平滑化（OverSmoothing）的缺陷。



#### 定理1.1 GCN的卷积公式

首先，回归GCN的卷积公式

首先回顾GCN的定义，单层GCN的计算公式，


$$
f(X) = \sigma(PXW)
$$


其中，$P$表示和邻接矩阵相关的线性变换，$W$表示一个可学习的线性变换，$\sigma$为激活函数，



对比GCN的公式，


$$
f(X) =  Relu(\tilde A  X \Theta)  \\
$$


也即$\Theta =W$, $\tilde A = P$, $\sigma(x)=Relu(x) = max(x,0)$,



DopEdge论文中定义了$P$ 为$L = I - D^{-\frac{1}{2} } A D^{-\frac{1}{2}}$ 的多项式函数，可以注意到，没有经过自环trick之前的GCN公式，可以表示为，


$$
\tilde A X W = (2I-L) X W = g(L) XW \\
\tilde A :=I + D^{-\frac{1}{2} } A D^{-\frac{1}{2}} = 2I-L
$$
 

也即原GCN的公式也满足该定义，下面使用该公式作为GCN的公式分析。

#### 定理1.2 空间距离的定义

要定义什么是过平滑化，本质是某个距离度量趋近于0，此时也即收敛到了某个空间上，即过平滑化。

因此，首先应该定义一个空间距离度量。



首先注意到，$U$为$L$的零空间，同时也是其最小特征值对应的特征子空间,因为


$$
Le = (I - D^{-\frac{1}{2} } A D^{-\frac{1}{2}})e = 0
$$


其中$e=[1,1,...,1]^T$



此时满足$U$为线性变换$P$的不变子空间


$$
\forall x \in U, Lx=0
$$


可以直接验证下式成立，


$$
Lx = 0 \Rightarrow L P x =0
$$



同时，由于$L$为对称阵，$P$也为对称阵，则$U$的正交补空间$U^C$ 也为$P$的不变子空间，


$$
\forall u \in U, v \in U^C \\
<Pv,u> = <v,P^Tu> = <v,Pu> = 0
$$


因为$Pu \in U$ ,



定义子空间$X$相对子空间$M$之间的距离

$$
d_M(X) := inf\{\Vert X-Y \Vert_F  :  Y \in M\} \\
M = :U \otimes R^C = \{\sum_m^M e_m \otimes w_m : w_m \in R^C\}
$$


其中$R^C$表示$C$维实数集，其中假设$U$属于$M$维子空间$R^M$，而$\{e_m\}$ 为$R^N$ 空间中的标准正交基， 而其中的前$M$个为$U$ 的标准正交基，



#### 定理1.4 过平滑化的理论证明

下面需要证明，GCN可以使得$d_M(X)$逐层递减，

证明分为两步，第一步为消息传递和聚合的过程使得距离递减，


$$
d_M(PXW) \le d_M(X)
$$


第二步为证明激活函数使得距离也递减，


$$
d_M(\sigma(X)) \le d_M(X)
$$


综合两式就可以得到，每一层GCN使得$d_M(X)$递减，


$$
d_M(f(X)) \le d_M(X)
$$



由于$X \in R^N \times R^C$, $X$可以表示为
$$
X = \sum_m^N e_m  \otimes w_m, \exist w_m
$$




#### 定理1.4.1 消息传递的过平滑化

由Krocker积的性质等可以推出，该结论也可以直观理解，

$$
d_M(X) = \sum_{m=M+1}^{N} \Vert w_m\Vert_2
$$


推导如下，首先有,

$$
d_M(X) = \Vert \sum_{m=M+1}^N e_m \otimes w_m\Vert_F
$$

对该式进行化简，

$$
\begin{align}
d_M(X) 
&= \sum_{m=M+1}^N (e_m \otimes w_m)^T (e_m\otimes w_m) \\
&=  \sum_{m=M+1}^N (e_m^T e_m) \otimes(w_m^T w_m) \\
&=  \sum_{m=M+1}^N (w_m\otimes w_m) \\
&= \sum_{m=M+1}^{N} \Vert w_m \Vert_2
\end{align}
$$


同理，

$$
\begin{align}
d_M(PXW) 
&= \sum_{m=M+1}^N (Pe_m) \otimes (W^T w_m) \\
&= \sum_{m=M+1}^N (\lambda_m e_m) \otimes  (W^T w_m) \\
&=\sum_{m=M+1}^N e_m \otimes  ( \lambda_mW^T w_m) \\
&= \sum_{m=M+1}^N \Vert \lambda_m W^Tw_m \Vert_2 \\
&\le \lambda s \sum_{m=M+1}^N \Vert w_m \Vert_2 \\
&\le d_M(X)
\end{align}
$$


其中，定义$(\lambda_m,e_m)$ 为 $P$的一个特征对，而$\lambda,s$ 分别为$P$的最大特征值和$W$的最大奇异值， 

在$\lambda s \le 1$的假设下，得到最后一个不等式，从而证明了消息传递的过程使得距离递减的性质。



#### 定理1.4.2 激活函数的过平滑化

下面需要证明激活函数也使得$d_M(X)$递减，

个人觉得这个结论是直观的，因为$Relu$激活函数只保留了正部，所以$d_M(X)$不可能会增大，当然文中的证明非常严谨，此处从略，大致证明思路也是从$Relu$函数入手，具体证明的过程中用到了$e_m$正交性质以及$\Vert.|Vert_F$ 的排列不变性质，此处从略。



 有了以上的结论，可以定义$\epsilon-soomth$的定义，其实就是说，GCN堆叠到了一定的层数之后，$d_M(f^{(l)}(X)) \to 0$

用分析的语言表达为，
$$
\exist l^\*, \forall l \ge l^\*,d_M(f^{(l)}(X)) \le \epsilon
$$


若要求
$$
d_M(f^{(l)}(X)) \le (s\lambda)^l d_M(X) \le \epsilon 
$$
则只需要，
$$
l \ge \frac{\log\frac{\epsilon}{d_M(X)}}{\log s\lambda}
$$
也即当GCN的层数达到一定数目后，就会陷入过平滑化的危险之中。



搞了这么久，好像还没有进入DropEdge，DropEdge为什么有效呢，原因是DropEdge可以减缓过平滑化的趋势，那么又是如何减缓的呢，可以证明使用DropEdge的技术，可以提高上面推出的$l$的下界，也就是说GCN需要更多的层数才会达到同样的平滑化效果。

但在证明DropEdge的神奇作用之前，还得先证明一些也很神奇的公式，

好像又有点说来话长了，首先要从图上的随机游走说起.....

首先我们需要引入一些些概念....



## 从随机游走到DropEdge

#### 定理2.1 图上的随机游走

图上的随机游走定义为每个结点随机地选择一个邻居转移，也即转移矩阵由如下公式表示


$$
M = D^{-1} A
$$



且有


$$
M e = e ,e_i=1,\\
M^T \pi = \pi, \pi_i = \frac{d_i}{2m}
$$


上述得到的两个向量即为转移矩阵$M$的左右特征向量，直接代入验算可得。



但$M$不是实对称矩阵，使用起来不方便，使用


$$
N := D^{-\frac{1}{2}} A D^{-\frac{1}{2}} = D^{\frac{1}{2}} M D^{-\frac{1}{2}}
$$


则$N$存在正交谱分解，


$$
N = \sum_i  \lambda_i v_i v_i^T \\
v_1 = (\frac{D}{2m})^{\frac{1}{2}} e
$$


代入得，


$$
M = D^{-\frac{1}{2}} N D^{\frac{1}{2}} = Q + \sum_{i=2}^n D^{-\frac{1}{2}} \lambda_i v_i v_i^TD^{\frac{1}{2}} \\
M^t = D^{-\frac{1}{2}} N^t D^{\frac{1}{2}} = Q + \sum_{i=2}^n D^{-\frac{1}{2}} \lambda_i^t v_i v_i^TD^{\frac{1}{2}}
$$


其中$Q$为$\pi$组成的矩阵，由于$\forall i >1, \lambda_i<1$ ，右式的第二项收敛至0，


$$
M \to Q, t\to \infin
$$


#### 定理2.2 结点的平均距离

下面定义结点之间的平均距离，$H(i,j)$定义为结点$i$走到结点$j$的期望时间，由于结点$i$到结点$j$的期望时间等于其邻居结点到$j$的平均时间加1，用矩阵表示为，其中$J$为全1矩阵，当$i \neq j $ 时有下式成立，


$$
H = J + M H
$$


所以已知


$$
F = J + MH-H
$$


为对角阵，下面求其对角元素，利用下式，


$$
F^T \pi = J^T \pi + H^T(M^T-I) \pi = J \pi = e
$$


则计算得到，


$$
F_{ii} = \frac{2m}{d_i}\\
F = 2m D^{-1} \\
(M-I)H = 2m D^{-1} -J
$$


但由于根据Perron定理，$M$的最大特征值为1，因此$M-I$不可逆，无法通过直接求逆的方式求得$H$,但注意到，


$$
(M-I) e = 0
$$


因此，如果$H$为该方程的解，则$\forall a, H + e a^T$ 也为该方程的解, 因此可以通过选取合适的$a$，使得该方程可解，

取$a=\pi$, 则代入后可以求出$H$，此时$e a^T = Q$,



这里采用另一种方式的证明，



$$
\begin{align}
N &= Q^T \Lambda Q \\
L &=  D- A\\
L^+ &= D^{-\frac{1}{2}} Q(I-\Lambda)^{+}  D^{-\frac{1}{2}}
\end{align}
$$



根据上式，


$$
\begin{align}
(I-M) H &= J - 2m D^{-1} \\
LH &= DJ-2mI \\
\end{align}
$$
又由，


$$
L^+ L = I - \frac{1}{n} e e^T,e=[1,1,...1,]^T
$$


两边同乘$L^+$并移项，


$$
H = L^+ J - 2m L^+ +e u^T, \exist u
$$


展开每个元素得，


$$
H_{ij} = \sum_k L^+_{ik} d_k - 2m L^+_{ij} + u_j
$$


我们知道$H_{ii}=0$,因此代入可以求出$u_j$,


$$
u_j = \sum_k L^+_{jk}d_k + 2mL^+_{jj}
$$

#### 定理2.3 结点的通讯距离

结点的通讯距离根据定理2.2，结点的平均距离定义：




$$
\kappa(i,j) = H(i,j) +H(j,i) = 2m(L^+_{ii}+L^+_{jj}- 2L^+_{ij})
$$


再代入之间得到的，


$$
L^+ = D^{-\frac{1}{2}} Q(I-\Lambda)^{+}  D^{-\frac{1}{2}}
$$


可以得到最终的公式，其中$v$表示$Q$的列向量，


$$
\kappa(i,j)= 2m\sum_{k=2} \frac{1}{1-\lambda_k} (\frac{v_{ik}}{\sqrt{d_i}} - \frac{v_{jk}}{\sqrt{d_j}})^2
$$


#### 定理2.4 通讯距离与特征值的关系

由$\lambda = \lambda_2 \ge \lambda_k , \forall k\ge2$, 以及$Q$为正交矩阵，$v_{ik}^2 +v_{jk}^2 \le 1$,

以及显然上式的最大值不超过$v_{ik}^2 +v_{jk}^2 = 1$ 的情况，

但在$v_{ik}^2 +v_{jk}^2 = 1$ 的情况下可以通过不等式放缩证明，


$$
(\frac{v_{ik}}{\sqrt{d_i}} - \frac{v_{jk}}{\sqrt{d_j}})^2 \le \frac{1}{d_i} + \frac{1}{d_j}
$$


故


$$
m (\frac{1}{d_i} + \frac{1}{d_j}) \le \kappa(i,j) \le \frac{2m}{1-\lambda} (\frac{1}{d_i} + \frac{1}{d_j})
$$



#### 定理2.5 随机游走与电路的联系

下面的证明，建立起了图上的随机游走和电路之间的关系，个人觉得非常美妙，先看如下公式，


$$
\phi(v) = \frac{1}{d_v} \sum_{u \in \Gamma(v)} \phi(u)
$$

其中，$\Gamma(v)$ 表示$v$的出度邻居集合，



考虑图上的两个结点$s，t$ 。

如果$\phi(u)$ 定义为一个从$u$出发的随机游走，在到达$t$之间经过了结点$s$的概率，显然这样定义的$\phi$ 满足上式，由于要经过一个结点必先到达其某个入度邻居。

如果将图视作一个电路，图中的每一条边视作一个单位电阻，考虑一个从结点$s$流向$t$的电流，定义$\phi(u)$ 为结点$u$上的电压，显然由伏安定律，$\phi$也满足上式。

所以上述两个定义本质上是等价的，同时，注意到在上述两个定义中均满足$\phi(s)=1, \phi(t)=0$,



#### 定理2.6 结点的通讯距离与图的电阻

在定理2.5的基础上，由伏安定律，$s$与$t$之间大的电阻表示为，


$$
R_{st} = \frac{1}{ \sum_{u \in \Gamma(t)} \phi(u)} = \frac{1}{\phi(t) d_t}
$$


此时$\phi(t)$的含义是，从$t$出发的随机游走，在返回$t$之前经过了$s$的概率。



在证明最终结论之前，需要再看一下图上随机游走的性质，代入易验证上述平稳分布实际上为细致平稳分布，


$$
p_{ij} \pi_i = p_{ji} \pi_j = \frac{1}{2m}
$$


该式子实际上说明了，到达平稳分布后，停留在每一条边都是等概率的。

那么，如果我们从一条边出发，回到该边的期望步数为$2m$,

同理，停留在每一个结点的概率等于$\pi_i$, 从该节点出发，重新回到该节点的期望步数为$\frac{1}{\pi_i} = \frac{2m}{d_i}$,



$$
\begin{align}
&E(\sigma) :=E(t \to s\to t) =  H(t,s) +H(s,t) = \kappa(s,t)\\
&E(\tau):=E(t \to t) = \frac{2m}{d_t}
\end{align}
$$


其中，$E(s \to t)$ 定义为从$s$出发到达$t$的期望步数，又有


$$
E(\sigma) - E(\tau) = (1-q) E(\sigma)
$$


因为$P(\sigma=\tau)=q$, 而若以$1-q$的概率该事件没有发生，那么需要走的期望步数为，


$$
(1-q) E(\sigma)
$$



又因为该事件没有发生正好意味着$\tau < \sigma$, 那么此时已经走了$E(\tau)$步，只需要再走$E(\sigma) - E(\tau) $ 步就可以使得事件$\sigma$发生，也即期望步数为


$$
E(\sigma) - E(\tau)
$$



故


$$
q = \phi(t) = \frac{E(\tau)}{E(\sigma)} = \frac{1}{\pi_t \ \kappa(s,t)}
$$


则有，


$$
R_{st} = 2m \kappa(s,t)
$$



## DropEdge的威力

从上面的一系列精巧的理论出发，我们与DropEdge建立联系，可以看见DropEdge的强大威力。



总结一下上面推出的几个公式，

$$
\lambda \ge 1 - \frac{1}{\kappa(i,j)} (\frac{1}{d_i} + \frac{1}{d_j}) \\
 \kappa(i,j) = \frac{1}{2m R_{ij}} \\
 l \ge \frac{\log\frac{\epsilon}{d_M(X)}}{\log s\lambda}
$$


那么DropEdge的效果是什么呢？

我们从电路的角度看整张图，将边看作单位电阻，DropEdge等价于删去一个电阻（将其断路），电路中的总电阻值应该增大，

由，

$$
\lambda \ge 1 - \frac{2m}{\kappa(i,j)} (\frac{1}{d_i} + \frac{1}{d_j}) =1 - \frac{1}{R_{ij}} (\frac{1}{d_i} + \frac{1}{d_j}) \\
$$

我们知道这相当于增加了$\lambda$的下界，

而且，在极限情况下，不断执行DropEdge操作将会得到完全的断路，也即，

$$
\exist s,t, R_{st} = \infin
$$

此时有$\lambda \ge 1$ 但我们又已知$\lambda \le 1$ ,故极限情况下一定会达到$\lambda=1$ , 故在未达到$\lambda=1$ 之间$\lambda$ 的确会增加而不可能一直保持不变状态。

其实该结论等价于，增加了1特征值的重数。

那么，又由于，

$$
 l \ge \frac{\log\frac{\epsilon}{d_M(X)}}{\log s\lambda} \\
 when \ \log\frac{\epsilon}{d_M(X)} <0
$$

$l$与$\lambda$成正相关关系，故$l$的值也会增加，

也即使用DropEdge的确可以减缓$\epsilon -smooth$的速度，



同时，从另一个角度，当图达到不连通状态时，空间$M$的维数至少增加1，那么$dim(M)$也增大了，相当于可以在更高维的空间中进行特征处理。从信息论的角度上，允许我们在更高维的空间中考虑问题，可能可以获得更多的信息，从而有助于GNN的学习。

