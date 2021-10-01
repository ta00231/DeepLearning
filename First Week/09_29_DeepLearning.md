# 一、神经网络编程基础（监督学习）0929

## 1. 二分类（Binary Classification）问题

![929_01.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/929_01.png)

举例：识别图片内的猫

结果为1（猫），0（不是猫）

RGB三色像素通道，从3个64 X 64的像素矩阵，提取特征值向量x，则形成了64 X 64 X 3 的x特征向量，即n_x或（n） = 64 x 64 x 3 =12288。

所以二分类问题中，目标是训练出一个 分类器 ，以图片的特征向量 x 作为输入，预测输出结果标签 y （1 or 0）

### 符号定义 ：
𝑥：表示一个𝑛𝑥维数据，为输入数据，维度为(𝑛𝑥, 1);  
  
𝑦：表示输出结果，取值为(0,1)；  
  
(𝑥(𝑖), 𝑦(𝑖))：表示第𝑖组数据，可能是训练数据，也可能是测试数据，此处默认为训练数据；  

𝑋 = [𝑥(1), 𝑥(2), . . . , 𝑥(𝑚)]：表示所有的训练数据集的输入值，放在一个 𝑛𝑥 × 𝑚的矩阵中，其中𝑚表示样本数目;  
  
𝑌 = [𝑦(1), 𝑦(2), . . . , 𝑦(𝑚)]：对应表示所有训练数据集的输出值，维度为1 × 𝑚。

## 2. 逻辑回归（Logistic Regression）算法

解决二分类问题基础算法

### Hypothesis Function（假设函数）：

𝑤：实际上是特征权重；  
𝑏：实数（表示偏差）；  
预测值𝑦^： 
$$
\mathrm{\hat{y}} =\,\,\mathrm{w}^{\mathrm{T}}\mathrm{x} +\,\,\mathrm{b}
$$

但二分类要输出在0 ~ 1之间，所以𝑦^ = 𝑤𝑇𝑥 + 𝑏单纯的线性函数不是很好

引入sigmoid（z）：  
$$
\mathrm{\sigma}\left( \mathrm{z} \right) =\frac{1}{1+\mathrm{e}^{-\mathrm{z}}}
$$
预测值𝑦^： 
$$
\mathrm{\hat{y}} =\,\,\mathrm{\sigma}\left( \mathrm{w}^{\mathrm{T}}\mathrm{x} +\,\,\mathrm{b} \right) 
$$

据图像可知，横轴两端扩展，大正数为1，小负数为0。  

![929_02.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/929_02.png)

分类器训练落在了寻找好的w和b上。  
两种参数训练，分开w、b 或者 w、b合为θ，定义
$$
\mathrm{\hat{y}} =\,\,\mathrm{\sigma}\left( \mathrm{\theta}^{\mathrm{T}}\mathrm{x} \right) 
$$
。

### Logistic Regression Cost Function  (代价函数)

![929_03.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/929_03.png)

#### 损失函数：
损失函数又叫做误差函数，  
用来衡量算法的运行情况，Loss function:  𝐿(𝑦^ , 𝑦)  
  
衡量预测输出值和实际值有多接近。  
一般我们用预测值和实际值的平方差或者它们平方差的一半，但是通常在逻辑回归中我们不这么做，因为当我们在学习逻辑回归参数的时候，会发现我们的优化目标不是凸优化，只能找到多个局部最优值，梯度下降法很可能找不到全局最优值。  
  
损失函数是在单个训练样本中定义的，它衡量的是算法在单个训练样本中表现如何，为了衡量算法在全部训练样本上的表现如何，我们需要定义一个算法的代价函数，即  通过损失函数推出总样本的代价函数。

我们在逻辑回归中用到的损失函数是：
$$
\mathrm{L}\left( \mathrm{\hat{y}},\mathrm{y} \right) \,\,=\,\,-\mathrm{y}log\left( \mathrm{\hat{y}} \right) \,\,-\,\,\left( 1-\mathrm{y} \right) log\left( 1-\mathrm{\hat{y}} \right) 
$$


当𝑦 = 1时损失函数𝐿 = −log(𝑦^)，如果想要损失函数𝐿尽可能得小，那么𝑦^就要尽可能大，因为 sigmoid 函数取值[0,1]，所以𝑦^会无限接近于 1。  
当𝑦 = 0时损失函数𝐿 = −log(1 − 𝑦^)，如果想要损失函数𝐿尽可能得小，那么𝑦^就要尽可能小，因为 sigmoid 函数取值[0,1]，所以𝑦^会无限接近于 0。  
预测结果也就跟实际值更近似。

#### 代价函数推出：

算法的代价函数是对𝑚个样本的损失函数求和然后除以𝑚:
$$
\mathrm{J}\left( \mathrm{w},\mathrm{b} \right) \,\,=\,\,\frac{1}{\mathrm{m}}\sum_{\mathrm{i}=1}^{\mathrm{m}}{\mathrm{L}\left( \mathrm{\hat{y}}^{\left( \mathrm{i} \right)},\mathrm{y}^{\left( \mathrm{i} \right)} \right)}\,\,=\,\,\frac{1}{\mathrm{m}}\sum_{\mathrm{i}=1}^{\mathrm{m}}{\left( -\mathrm{y}^{\left( \mathrm{i} \right)}log\left( \mathrm{\hat{y}}^{\left( \mathrm{i} \right)} \right) \,\,-\,\,\left( 1-\mathrm{y}^{\left( \mathrm{i} \right)} \right) log\left( 1-\mathrm{\hat{y}}^{\left( \mathrm{i} \right)} \right) \right)}
$$


## 3. 梯度下降法（Gradient Descent）

在你测试集上，通过最小化代价函数（成本函数）𝐽(𝑤, 𝑏)来训练的参数𝑤和𝑏。

![929_04.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/929_04.png)

在代价函数是一个凸函数(convex function)的时候，可以通过梯度下降找到让代价最小的w和b，非凸函数（non-convex function）有众多局部最小值，所以一定确保代价函数为凸函数。

$$
\mathrm{w} :=\,\,\mathrm{w} -\,\,\mathrm{\alpha}\frac{\partial \mathrm{J}\left( \mathrm{w},\mathrm{b} \right)}{\partial \mathrm{w}}
\\
\mathrm{b} :=\,\,\mathrm{b} -\,\,\mathrm{\alpha}\frac{\partial \mathrm{J}\left( \mathrm{w},\mathrm{b} \right)}{\partial \mathrm{b}}
$$


$$
\mathrm{\alpha}
$$
表示学习率（learning rate），用来控制步长（step）；  
$$
\frac{\partial \mathrm{J}\left( \mathrm{w},\mathrm{b} \right)}{\partial \mathrm{w}}
$$
即w向下走一步的长度就是函数𝐽(w,b)对𝑤 求导（derivative）,在代码中我们会使用𝑑𝑤表示这个结果；  
$$
\frac{\partial \mathrm{J}\left( \mathrm{w},\mathrm{b} \right)}{\partial \mathrm{b}}
$$
同理，且用db表示。

  
且由于步长正负跟斜率有关，即对参数的导数结果有关：  
k > 0 时，参数缩小，靠近最低点；  
k < 0 时，参数放大，也靠近最低点。

## 4. 计算图（Computation Graph）

可以说，一个神经网络的计算，都是按照前向或反向传播过程组织的。首先我们计算出
一个新的网络的输出（前向过程），紧接着进行一个反向传输操作。后者我们用来计算出对
应的梯度或导数。计算图解释了为什么我们用这种方式组织这些计算过程。

简单例子  
![929_05.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/929_05.png)

### 使用计算图求导数（反向过程）

![929_06.png](https://github.com/ta00231/DeepLearning/blob/main/Pictures/929_06.png)

所以这是一个计算流程图，就是正向或者说从左到右的计算来计算成本函数𝐽，你可能
需要优化的函数，然后反向从右到左计算导数。
