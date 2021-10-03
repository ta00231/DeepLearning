# 二、浅层神经网络（SNN）1002

## 1. 神经网络概述

你应该记得逻辑回归中，有一些从后向前的计算用来计算导数𝑑𝑎、𝑑𝑧。同样，在神经网
络中我们也有从后向前的计算，看起来就像这样，最后会计算𝑑𝑎
[2] 、𝑑𝑧
[2]，计算出来之后，
然后计算计算𝑑𝑊[2]、𝑑𝑏
[2] 等，

## 2. 神经网络的表示

![1002_01.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_01.png)

### 输入层 隐藏层 输出层

我们有输入特征𝑥1、𝑥2、𝑥3，它们被竖直地堆叠起来，这叫做神经网络的输入层。   
另外一层我们称之为隐藏层。  
最后一层只由一个结点构成，而这个只有一个结点的层被称为输出层，它负责产生预测值。

### 隐藏的含义

在训练集中，这些中间结点的准确值我们是不知道到的，你能看见输入的值，你也能看见输出的值，但是隐藏层中的东西，在训练集中你是无法看到的。所以这也解释了词语隐藏层，只是表示你无法在训练集中看到他们。

![1002_02.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_02.png)

𝑎表示激活的意思，它意味着网络中不同层的值会传递到它们
后面的层中，输入层将𝑥传递给隐藏层，所以我们将输入层的激活值称为𝑎
[0]；下一层即隐藏
层也同样会产生一些激活值，那么我将其记作𝑎
[1]，所以具体地，这里的第一个单元或结点
我们将其表示为𝑎1
[1]，第二个结点的值我们记为𝑎2
[1]以此类推。

#### 惯例

当我们计算网络的层数时，输入层是不算入总层数内，所以隐藏层是第一层，输出层是第二层。

最后，我们要看到的隐藏层以及最后的输出层是带有参数的，这里的隐藏层将拥有两个
参数𝑊和𝑏，我将给它们加上上标 [1]
(𝑊[1]
,𝑏
[1]
)，表示这些参数是和第一层这个隐藏层有关
系的。之后在这个例子中我们会看到𝑊是一个 4x3 的矩阵，而𝑏是一个 4x1 的向量，第一个
数字 4 源自于我们有四个结点或隐藏层单元，然后数字 3 源自于这里有三个输入特征，相似的输出层也有一
些与之关联的参数𝑊[2]以及𝑏
[2]。从维数上来看，它们的规模分别是 1x4 以及 1x1。1x4为隐藏层有四个隐藏层单元而输出层只有一个单元。

## 3. 计算一个神经网络输出

用圆圈表
示神经网络的计算单元，逻辑回归的计算有两个步骤，首先你按步骤计算出𝑧，然后在第二
步中你以 sigmoid 函数为激活函数计算𝑧（得出𝑎），一个神经网络只是这样子做了好多次重
复计算。


![1002_03.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_03.png)

第一步，计算
$$
\mathrm{z}_{1}^{\left[ 1 \right]}=\mathrm{w}_{1}^{\left[ 1 \right] \mathrm{T}}\mathrm{x}+\mathrm{b}_{1}^{\left[ 1 \right]}
$$

  
  第二步，通过激活函数计算
  $$
\mathrm{a}_{1}^{\left[ 1 \right]}=\mathrm{\sigma}\left( \mathrm{z}_{1}^{\left[ 1 \right]} \right) 
$$


隐藏层的第二个以及后面两个神经元的计算过程一样，只是注意符号表示不同，最终分
别得到𝑎2
[1]、𝑎3
[1]、𝑎4
[1]

同样为了避免低效的for循环，向量化很重要

![1002_04.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_04.png)

## 4.多样本向量化

$$
\mathrm{a}^{\left[ \mathrm{i} \right] \left[ \mathrm{j} \right]}
$$


i 表示第几层神经网络  
j 表示第几个样本

未向量化：

![1002_05.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_05.png)

向量化：

横向表示不同样本同一属性，纵向表示同一样本不同属性。（特征、隐藏单元节点等）

![1002_06.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_06.png)

## 5. 向量化实现的解释

![1002_07.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_07.png)

如图片展示

## 6.激活函数（Activation functions）

使用一个神经网络时，需要决定使用哪种激活函数用隐藏层上，哪种用在输出节点上。

#### sigmoid 函数

$$
\mathrm{a}=\mathrm{\sigma}\left( \mathrm{z} \right) =\frac{1}{1+\mathrm{e}^{-\mathrm{z}}}
$$


#### tanh函数

$$
\mathrm{a}=\tanh \left( \mathrm{z} \right) =\frac{\mathrm{e}^{\mathrm{z}}-\mathrm{e}^{-\mathrm{z}}}{\mathrm{e}^{\mathrm{z}}+\mathrm{e}^{-\mathrm{z}}}
$$


事实上，tanh 函数是 sigmoid 的向下平移和伸缩后的结果。对它进行了变形后，穿过了
(0,0)点，并且值域介于+1 和-1 之间。

结果表明，如果在隐藏层和输出层上使用函数tanh效果优于sigmoid，但对于二分类问题例外，需要y的值处于 0 或 1 之间时，输出层使用sigmoid，隐藏层使用tanh。

#### 缺点

sigmoid 函数和 tanh 函数两者共同的缺点是，在𝑧特别大或者特别小的情况下，导数的
梯度或者函数的斜率会变得特别小，最后就会接近于 0，导致降低梯度下降的速度。

在机器学习另一个很流行的函数是：修正线性单元的函数（ReLu），ReLu 函数图像是如
下图。

#### ReLu函数

$$
\mathrm{a}=\max \left( 0,\mathrm{z} \right) 
$$

只要𝑧是正值的情况下，导数恒等于 1，当𝑧是负值的时候，导数恒等于 0。

![1002_08.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_08.png)

#### 激活函数选择经验法则

如果输出是 0、1 值（二分类问题），则输出层选择 sigmoid 函数，然后其它的所有单
元都选择 Relu 函数。
这是很多激活函数的默认选择，如果在隐藏层上不确定使用哪个激活函数，那么通常会
使用 Relu 激活函数。有时，也会使用 tanh 激活函数，但 Relu 的一个优点是：当𝑧是负值的
时候，导数等于 0。

这里也有另一个版本的 Relu 被称为 Leaky Relu。

#### Leaky ReLu函数

$$
\mathrm{a}=\max \left( 0.01\mathrm{z},\mathrm{z} \right) 
$$

快速概括一下不同激活函数的过程和结论。

![1002_09.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_09.png)

#### 优缺点对比

第一，在𝑧的区间变动很大的情况下，激活函数的导数或者激活函数的斜率都会远大于
0，在程序实现就是一个 if-else 语句，而 sigmoid 函数需要进行浮点四则运算，在实践中，
使用 ReLu 激活函数神经网络通常会比使用 sigmoid 或者 tanh 激活函数学习的更快。

第二，sigmoid 和 tanh 函数的导数在正负饱和区的梯度都会接近于 0，这会造成梯度弥散，而 Relu 和 Leaky ReLu 函数大于 0 部分都为常数，不会产生梯度弥散现象。(同时应该注
意到的是，Relu 进入负半区的时候，梯度为 0，神经元此时不会训练，产生所谓的稀疏性，
而 Leaky ReLu 不会有这问题)。  
𝑧在 ReLu 的梯度一半都是 0，但是，有足够的隐藏层使得 z 值大于 0，所以对大多数的
训练数据来说学习过程仍然可以很快。


## 7. 为什么需要非线性激活函数？

事实证明，如果你使用
线性激活函数或者没有使用一个激活函数，那么无论你的神经网络有多少层一直在做的只是
计算线性函数，所以不如直接去掉全部隐藏层。在我们的简明案例中，事实证明如果你在隐
藏层用线性激活函数，在输出层用 sigmoid 函数，那么这个模型的复杂度和没有任何隐藏层
的标准 Logistic 回归是一样的。

唯一可以用线性激活函数的通常就是输出层；除了这种情况，会
在隐层用线性函数的，除了一些特殊情况，比如与压缩有关的。

## 8. 激活函数的导数

![1002_10.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_10.png)

![1002_11.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_11.png)

![1002_12.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_12.png)

![1002_13.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_13.png)

![1002_14.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_14.png)

## 9. 神 经 网 络 的 梯 度 下 降

![1002_15.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_15.png)

![1002_16.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_16.png)

## 10. 随机初始化

当你训练神经网络时，权重随机初始化是很重要的。对于逻辑回归，把权重初始化为 0
当然也是可以的。但是对于一个神经网络，如果你把权重或者参数都初始化为 0，那么梯度
下降将不会起作用。

#### 原因

对称性：导致多次计算仍是计算同一方程式，神经网络的多层性质形同虚设。

#### 解决方法

初始化时用 np.random.randn()

对于b修正偏移量可以用 np.zeros()

![1002_17.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/1002_17.png)

对于w后的乘的常数，为什么是 0.01，而不是 100 或者 1000。我们通常
倾向于初始化为很小的随机数。因为如果你用 tanh 或者 sigmoid 激活函数，或者说只在输
出层有一个 Sigmoid，如果（数值）波动太大，当你计算激活值时𝑧
[1] = 𝑊[1]𝑥 + 𝑏
[1]
, 𝑎
[1] =
𝜎(𝑧
[1]
) = 𝑔
[1]
(𝑧
[1]
)如果𝑊很大，𝑧就会很大。𝑧的一些值𝑎就会很大或者很小，因此这种情况
下你很可能停在 tanh/sigmoid 函数的平坦的地方，这些地方梯度很小也就意味着
梯度下降会很慢，因此学习也就很慢。

