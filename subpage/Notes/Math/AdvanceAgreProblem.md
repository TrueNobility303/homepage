# 高等线性代数精选题目

记录一些精妙的题目，帮助自己记忆。

## 题目1

求证： 下面的式子成立（其中＞号表示正定）

![img](AdvanceAgreProblem.assets/clip_image002.gif)



证明：利用特征值与矩阵放缩

![img](AdvanceAgreProblem.assets/clip_image004.gif)



下面的大于号同样表示矩阵正定的大于关系

![img](AdvanceAgreProblem.assets/clip_image006.gif)

运用到的性质总结：

* 正定阵的主子阵正定，因此矩阵$A,C$均为正定阵
* 对于对称正定阵，特征值均为正值，且存在谱分解，特征值和奇异值等价
* 对称矩阵的合同变换，并不改变其正定性质，且用到了常见的合同变换，将矩阵化为Schur补的形式
* 正定阵的偏序关系（大于关系）和普通的大小关系的相似性质，利用其进行放缩
* 利用特征值将矩阵放缩为常量阵，$ \lambda_1(A) I_n \prec  A \prec \lambda_n(A) I_n$， 可以用对角化/上三角化直观证明
* 奇异值、二范数与特征值的关系



## 题目2

求证：

![img](AdvanceAgreProblem.assets/clip_image008.gif)

证明：一方面， 不等号成立，且相似变换不改变特征值

![img](AdvanceAgreProblem.assets/clip_image010.gif)

另一方面， 取等在极限情况可取到， 只需取一个由特殊的小量构成的对角阵

![img](AdvanceAgreProblem.assets/clip_image012.gif)

运用到的性质总结：

* 上三角化以及上三角矩阵的F范数
* 任意上三角阵经过由某个小量$\epsilon$ 构成对角阵的相似变换后可以趋近于（无限接近）于对角阵
* 对角阵的F范数



## 题目3

证明： 若非负矩阵A的每个元素都大于B， 那么A的谱大于B的谱

证法1： 根据 Perron 定理， 利用左右特征向量，取出对应矩阵的特征值

![img](AdvanceAgreProblem.assets/clip_image014.gif)

证法2： 利用 Weyl 不等式， 拆分矩阵为对角元和非对角元

![img](AdvanceAgreProblem.assets/clip_image016.gif)

矩阵 C为正矩阵， D为其对角元素 E为其非对角元素

![img](AdvanceAgreProblem.assets/clip_image018.gif)

运用到的性质总结：

* 非负矩阵的Perron定理，Perron向量的元素均为正数，左右Perron向量
* 使用Weyl不等式进行特征值放缩，Weyl不等式实际上与特征子空间相关



## 题目4

已知矩阵A，特征值：

![img](AdvanceAgreProblem.assets/clip_image020.gif)

求证：

![img](AdvanceAgreProblem.assets/clip_image022.gif)

证明： 利用 Hoffman 不等式

![img](AdvanceAgreProblem.assets/clip_image024.gif)

再证等号可以取到，取最小特征值所对应的特征向量

![img](AdvanceAgreProblem.assets/clip_image026.gif)

该秩1扰动只改变最小特征值，根据 Schur分解

![img](AdvanceAgreProblem.assets/clip_image028.gif)

故等号可取到，证毕



运用到的性质总结：

* Hoffman不等式的F范数版本
* 选取特定的特征向量，秩1扰动仅仅扰动在对应的特征值上，而其他特征值不发生变化
