# 一、神经网络编程基础（监督学习）0930

## 1. 逻辑回归中的梯度下降

## 最初的式子

## $$\mathrm{z} =\,\,\mathrm{w}^{\mathrm{T}}\mathrm{x} +\,\,\mathrm{b}$$

## $$
\mathrm{\hat{y}} =\,\,\mathrm{a} =\,\,\mathrm{\sigma}\left( \mathrm{z} \right) \,\,=\,\,\frac{1}{1+\,\,\mathrm{e}^{-\mathrm{z}}}
$$

## $$
\mathrm{L}\left( \mathrm{a},\mathrm{y} \right) \,\,=\,\,-\left( \mathrm{y}\log \left( \mathrm{a} \right) \,\,+\,\,\left( 1 -\,\,\mathrm{y} \right) \log \left( 1 -\,\,\mathrm{a} \right) \right) 
$$


## 求解过程

假设样本只有两个特征𝑥1和𝑥2，为了计算𝑧，我们需要输入参数𝑤1、𝑤2 和𝑏，除此之外
还有特征值𝑥1和𝑥2。因此𝑧的计算公式为： 𝑧 = 𝑤1𝑥1 + 𝑤2𝑥2 + b。

假设现在只考虑单个样本的情况，单个样本的代价函数定义如下：

$$
\mathrm{L}\left( \mathrm{a},\mathrm{y} \right) \,\,=\,\,-\left( \mathrm{y}\log \left( \mathrm{a} \right) \,\,+\,\,\left( 1 -\,\,\mathrm{y} \right) \log \left( 1 -\,\,\mathrm{a} \right) \right) 
$$

其中𝑎是逻辑回归的输出，𝑦是样本的标签值。

𝑤和𝑏的修正量可以表达如下：

$$
\mathrm{w} :=\,\,\mathrm{w} -\,\,\mathrm{\alpha}\frac{\partial \mathrm{L}\left( \mathrm{w},\mathrm{b} \right)}{\partial \mathrm{w}}
\\
\mathrm{b} :=\,\,\mathrm{b} -\,\,\mathrm{\alpha}\frac{\partial \mathrm{L}\left( \mathrm{w},\mathrm{b} \right)}{\partial \mathrm{b}}
$$

则要计算相应的微积分：

![930_01.png](attachment:930_01.png)

则有链式法则反向推到可知：

$$
\mathrm{dw}_1\mathrm{} =\,\,\frac{\partial \mathrm{L}}{\partial \mathrm{w}_1}\,\,=\,\,\frac{\mathrm{dL}}{\mathrm{dz}}\,\,. \frac{\mathrm{dz}}{\mathrm{dw}_1}
$$
  
$$
\mathrm{dw}_2\mathrm{} =\,\,\frac{\partial \mathrm{L}}{\partial \mathrm{w}_2}\,\,=\,\,\frac{\mathrm{dL}}{\mathrm{dz}}\,\,. \frac{\mathrm{dz}}{\mathrm{dw}_2}
$$
  
$$
\mathrm{db} =\,\,\frac{\partial \mathrm{L}}{\partial \mathrm{b}}\,\,=\,\,\frac{\mathrm{dL}}{\mathrm{dz}}\,\,. \frac{\mathrm{dz}}{\mathrm{db}}
$$

同理可知：

$$
\mathrm{dz} =\,\,\frac{\mathrm{dL}\left( \mathrm{a},\mathrm{y} \right)}{\mathrm{dz}}\,\,=\,\,\frac{\mathrm{dL}}{\mathrm{da}}\,\,. \frac{\mathrm{da}}{\mathrm{dz}}
$$


故先求：

$$
\frac{\mathrm{dL}}{\mathrm{da}}\,\,=\,\,-\frac{\mathrm{y}}{\mathrm{a}}\,\,+\,\,\frac{\left( 1-\mathrm{y} \right)}{1-\mathrm{a}}
$$
  
$$
\frac{\mathrm{da}}{\mathrm{dz}}\,\,=\,\,\frac{\mathrm{e}^{-\mathrm{z}}}{\left( 1+\mathrm{e}^{-\mathrm{z}} \right)}\,\,=\,\,\mathrm{a}\left( 1-\mathrm{a} \right) 
$$

得出：

$$
\mathrm{dz} =\,\,\mathrm{a}-\mathrm{y}
$$


进而得出：

$$
\mathrm{dw}_1\mathrm{} =\,\,\mathrm{x}1 \cdot \,\,\mathrm{dz} 
\\
\mathrm{dw}_2\mathrm{} =\,\,\mathrm{x}2 \cdot \,\,\mathrm{dz} 
\\
\mathrm{db} =\,\, \mathrm{dz} 
$$


进而更新单个样本的：

$$
\mathrm{w}_1\,\,=\,\,\mathrm{w}_1\,\,-\,\,\mathrm{\alpha dw}_1
\\
\mathrm{w}_2\,\,=\,\,\mathrm{w}_2\,\,-\,\,\mathrm{\alpha dw}_2
\\
\mathrm{b} =\,\,\mathrm{b} -\,\,\mathrm{\alpha db}
$$

## 2. 现在应用到m个样本上 

代价函数已知为：

$$
\mathrm{J}\left( \mathrm{w},\mathrm{b} \right) \,\,=\,\,\frac{1}{\mathrm{m}}\sum_{\mathrm{i}=1}^{\mathrm{m}}{\mathrm{L}\left( \mathrm{\hat{y}}^{\left( \mathrm{i} \right)},\mathrm{y}^{\left( \mathrm{i} \right)} \right)}\,\,=\,\,\frac{1}{\mathrm{m}}\sum_{\mathrm{i}=1}^{\mathrm{m}}{\left( -\mathrm{y}^{\left( \mathrm{i} \right)}log\left( \mathrm{\hat{y}}^{\left( \mathrm{i} \right)} \right) \,\,-\,\,\left( 1-\mathrm{y}^{\left( \mathrm{i} \right)} \right) log\left( 1-\mathrm{\hat{y}}^{\left( \mathrm{i} \right)} \right) \right)}
$$

修正量：

$$
\mathrm{w} :=\,\,\mathrm{w} -\,\,\mathrm{\alpha}\frac{\partial \mathrm{J}\left( \mathrm{w},\mathrm{b} \right)}{\partial \mathrm{w}}
\\
\mathrm{b} :=\,\,\mathrm{b} -\,\,\mathrm{\alpha}\frac{\partial \mathrm{J}\left( \mathrm{w},\mathrm{b} \right)}{\partial \mathrm{b}}
$$

则具体得：

$$
\mathrm{dw}_1\,\,=\,\,\frac{\partial \mathrm{J}}{\partial \mathrm{w}_1}\,\,=\,\,\frac{1}{\mathrm{m}}\sum_{i=1}^{\mathrm{m}}{\frac{\partial \mathrm{L}}{\partial \mathrm{w}_1}}
\\
\mathrm{dw}_2\,\,=\,\,\frac{\partial \mathrm{J}}{\partial \mathrm{w}_2}\,\,=\,\,\frac{1}{\mathrm{m}}\sum_{i=1}^{\mathrm{m}}{\frac{\partial \mathrm{L}}{\partial \mathrm{w}_2}}
\\
\mathrm{db} =\,\,\frac{\partial \mathrm{J}}{\partial \mathrm{b}}\,\,=\,\,\frac{1}{\mathrm{m}}\sum_{i=1}^{\mathrm{m}}{\frac{\partial \mathrm{L}}{\partial \mathrm{b}}}
$$


更新修正量：

$$
\mathrm{w}_1\,\,=\,\,\mathrm{w}_1\,\,-\,\,\mathrm{\alpha dw}_1
\\
\mathrm{w}_2\,\,=\,\,\mathrm{w}_2\,\,-\,\,\mathrm{\alpha dw}_2
\\
\mathrm{b} =\,\,\mathrm{b} -\,\,\mathrm{\alpha db}
$$

代码流程：

![930_02.png](attachment:930_02.png)

![930_03.png](attachment:930_03.png)
