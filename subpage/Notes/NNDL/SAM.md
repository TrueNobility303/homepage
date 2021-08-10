# SAM

论文阅读笔记：[Sharpness-Aware Minimization for Efficiently Improving Generalization](https://paperswithcode.com/paper/sharpness-aware-minimization-for-efficiently-1)

SAM优化器（Sharpness-Aware Minimization)，笔者亲切地称其为山姆大叔，虽然并不十分出名，但SAM简单、直接、而又极其高效，是笔者个人在训练神经网络的时候非常喜欢的一个优化器，且屡试不爽。因此笔者非常想整理SAM的细节，并且希望SAM能得到更多人的关注。



## SAM问题：最小化泛化误差

SAM尝试减小在训练集$S$上训练到在整个数据分布$D$上的泛化误差，使用PAC-贝叶斯泛化误差上界理论，并且在给定条件下可以得到以下不等式以高概率近似成立，其中$h$为一通过计算得到的单调函数，由于该部分证明非常分析学且繁琐，函数表达式此处从略，感兴趣的读者详见论文链接，


$$
L_D(w) \le \max_{\left\|\epsilon\right\|_2 \le \rho} L_S(w+\epsilon) + h(\frac{\left\|w\right\|_2^2}{\rho^2})
$$



使用正则化项$\lambda \left\|w\right\|_2$ 近似替代$h(\frac{\left\|w\right\|_2^2}{\rho^2})$ ，可以得到优化目标，


$$
min \{\max_{\left\|\epsilon\right\|_2 \le \rho} L_S(w+\epsilon) + \lambda \left\|w\right\|_2\}
$$

## SAM迭代公式

SAM优化算法使用如下两步迭代法解决该优化问题，


$$
\epsilon_t = \rho \frac{\nabla L(w_t)}{ \left\|\nabla L(w_t) \right\|_2} \\
w_{t+1} = w_t - \alpha(\nabla L(w_t + \epsilon_t) + \lambda w)
$$



而第二步实际与带权重衰减的SGD更新方法相同，SAM迭代法的推导简要如下，


$$
\begin{align}
\epsilon 
&= \{\epsilon |  \{\max_{\left\|\epsilon\right\|_2 \le \rho} 
L_S(w+\epsilon)\} \\
&\approx \{\epsilon | \max_{\left\|\epsilon\right\|_2 \le \rho}\epsilon^T \nabla L(w)\} \\
&= \rho \text{ sign } \{\nabla L(w)\} \frac{
|\nabla L(w)|^{q-1}}{(\left\|\nabla L(w)\right\|_q^q)^{\frac{1}{p}}}
\end{align}
$$


满足$\frac{1}{p} +\frac{1}{q}=1$,其中第一个近似由$\epsilon$ 为小量得到，第二个等式由Holder不等式和对偶范数与对偶问题的性质得到，当$p= q=2$时，也即取2范数时,得到$\epsilon$的更新公式


$$
\epsilon_t = \rho \frac{\nabla L(w_t)}{ \left\|\nabla L(w_t) \right\|_2} \\
$$


忽略二阶小量的前提下，可以得到$w$较为简单的更新公式，该结果较为直观，推导过程略，

$$
w_{t+1} = w_t - \alpha(\nabla L(w_t + \epsilon_t) + \lambda w)
$$

SAM,yyds!
