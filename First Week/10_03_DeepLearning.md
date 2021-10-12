# 三、深层神经网络 DNN 1003

## 1、 总览

![1004_01.png](attachment:1004_01.png)

![1004_02.png](attachment:1004_02.png)

上图是一个四层的神经网络，有三个隐藏层。我们可以看到，第一层（即左边数过去第
二层，因为输入层是第 0 层）有 5 个神经元数目，第二层 5 个，第三层 3 个。

$$
\mathrm{L}
$$
表示层数；
$$
\mathrm{n}^{\left[ \mathrm{i} \right]}
$$
表示i层索引；

上图：𝐿 = 4，输入层的索引为“0”，第一个隐藏层𝑛
[1] = 5,表示有 5
个隐藏神经元，同理𝑛
[2] = 5，𝑛
[3] = 3，𝑛
[4]=𝑛
[𝐿] = 1（输出单元为 1）。而输入层，𝑛
[0] =
𝑛𝑥 = 3。

在不同层所拥有的神经元的数目，对于每层 l 都用𝑎[𝑙]来记作 l 层激活后结果，我们会在后面看到在正向传播时，最终能你会计算出𝑎[𝑙]。  
通过用激活函数 𝑔 计算𝑧[𝑙]，激活函数也被索引为层数𝑙，然后我们用𝑤[𝑙]来记作在 l 层计算𝑧[𝑙]值的权重。  
类似的，𝑧
[𝑙]里的方程𝑏
[𝑙]也一样。  
最后总结下符号约定：
输入的特征记作𝑥，但是𝑥同样也是 0 层的激活函数，所以𝑥 = 𝑎
[0]。
最后一层的激活函数，所以𝑎
[𝐿]是等于这个神经网络所预测的输出结果。


## 2、前向传播和反向传播 propagation

### 前向传播

$$
\mathrm{Z}^{\left[ \mathrm{L} \right]}=\mathrm{W}^{\left[ \mathrm{L} \right]}\mathrm{A}^{\left[ \mathrm{L}-1 \right]}+\mathrm{b}^{\left[ \mathrm{L} \right]}
\\
\mathrm{A}^{\left[ \mathrm{L} \right]}=\mathrm{g}^{\left[ \mathrm{L} \right]}\left( \mathrm{Z}^{\left[ \mathrm{L} \right]} \right) 
$$


其中

$$
\mathrm{A}^{\left[ 0 \right]}=\mathrm{X}
$$


#### 关于向量的维数

$$
\mathrm{A}^{\left[ \mathrm{L} \right]}:\left( \mathrm{n}^{\left[ \mathrm{L}-1 \right]},\mathrm{m} \right) 
\\
\mathrm{W}^{\left[ \mathrm{L} \right]}:\left( \mathrm{n}^{\left[ \mathrm{L} \right]},\mathrm{n}^{\left[ \mathrm{L}-1 \right]} \right) 
\\
\mathrm{b}^{\left[ \mathrm{L} \right]}:\left( \mathrm{n}^{\left[ \mathrm{L} \right]},1 \right) 
\\
\mathrm{n}^{\left[ 0 \right]}=\mathrm{n}_{\mathrm{x}}
\\
\mathrm{A}^{\left[ 0 \right]}=\mathrm{X}
$$


dZ 与 dA 相同与A  
dW 相同于 W  
db 相同于 b

### 反向传播

$$
\mathrm{dZ}^{\left[ \mathrm{L} \right]}=\mathrm{dA}^{\left[ \mathrm{L} \right]}*\mathrm{g}^{\left[ \mathrm{L} \right] \mathrm{`}}\left( \mathrm{Z}^{\left[ \mathrm{L} \right]} \right) 
\\
\mathrm{dW}^{\left[ \mathrm{L} \right]}=\frac{1}{\mathrm{m}}\mathrm{dZ}^{\left[ \mathrm{L} \right]}\cdot \mathrm{A}^{\left[ \mathrm{L}-1 \right] \mathrm{T}}
\\
\mathrm{db}=\frac{1}{\mathrm{m}}\mathrm{np}.\mathrm{sum}\left( \mathrm{dZ}^{\left[ \mathrm{L} \right]},\mathrm{axis}=1,\mathrm{keepdims}=\mathrm{True} \right) 
\\
\mathrm{dA}^{\left[ \mathrm{L}-1 \right]}=\mathrm{W}^{\left[ \mathrm{L} \right] \mathrm{T}}\mathrm{dZ}^{\left[ \mathrm{L} \right]}
$$


### 总览图

![1004_03.png](attachment:1004_03.png)

![1004_04.png](attachment:1004_04.png)

## 3、参数和超参数 Parameters & Hyperparameters

参数：W，b

超参数：学习率α、梯度下降循环次数、神经网络层数、隐藏层的深度等影响参数的数值设置
