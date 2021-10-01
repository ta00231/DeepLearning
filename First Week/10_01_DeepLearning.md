# 向量化 Vectorization

向量化是非常基础的去除代码中 for 循环的艺术。

## 1. 简单例子 


```python
import numpy as np
```


```python
a = np.array([1,2,3,4])
print(a)
```

    [1 2 3 4]
    


```python
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(c)
print("Vectorized version:" + str(1000*(toc - tic)) + 'ms')

c = 0
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()

print(c)
print('For loop:' + str(1000*(toc - tic)) + 'ms')
```

    249686.54167691586
    Vectorized version:0.9982585906982422ms
    249686.54167690442
    For loop:691.1487579345703ms
    

## 2.向量化讨论

经验提醒我，当我们在写神经网络程序时，或者在写逻辑(logistic)回归，或者其他神经
网络模型时，应该避免写循环(loop)语句。虽然有时写循环(loop)是不可避免的，但是我们可
以使用比如 numpy 的内置函数或者其他办法去计算。当你这样使用后，程序效率总是快于
循环(loop)。

## 3. 向量化逻辑回归(Vectorizing Logistic Regression)

前向传播过程，计算预测值：

$$
\mathrm{z} =  \mathrm{w}^{\mathrm{T}}\mathrm{x} +  \mathrm{b}
$$  

$$
\mathrm{a} =  \mathrm{\sigma}\left( \mathrm{z} \right) 
$$


如果你有 𝑚 个训练样本，你可能需要这样做 𝑚 次。

向量化后：

$$
\mathrm{Z} =  \left[ \mathrm{z}^{\left( 1 \right)}, \mathrm{z}^{\left( 2 \right)},\mathrm{z}^{\left( 3 \right)}....,\mathrm{z}^{\left( \mathrm{m} \right)} \right]   =  \mathrm{w}^{\mathrm{T}}\mathrm{X} +  \left[ \mathrm{b},\mathrm{b},\mathrm{b}....,\mathrm{b} \right] 
$$  

$$
\mathrm{A} =  \left[ \mathrm{a}^{\left( 1 \right)}, \mathrm{a}^{\left( 2 \right)},\mathrm{a}^{\left( 3 \right)}....,\mathrm{a}^{\left( \mathrm{m} \right)} \right]   =  \mathrm{\sigma}\left( \mathrm{Z} \right) 
$$


w，b均为 1 x m 的向量，X为n_x x m 的向量（n_x为特征点的数量）。


```python
Z = np.dot(w.T,x) + b
A = sigmoid(Z)
```

反向传播过程，计算修正值：

$$
\mathrm{dz} =  \mathrm{a}-\mathrm{y}
$$
$$
\mathrm{dw}_1\mathrm{} =  \mathrm{x}1 \cdot   \mathrm{dz} 
$$  

$$
\mathrm{dw}_2\mathrm{} =  \mathrm{x}2 \cdot   \mathrm{dz} 
$$  

$$
\mathrm{db} =   \mathrm{dz} 
$$


向量化后：

$$
\mathrm{Y} =  \left[ \mathrm{y}^{\left( 1 \right)}, \mathrm{y}^{\left( 2 \right)},\mathrm{y}^{\left( 3 \right)}....,\mathrm{y}^{\left( \mathrm{m} \right)} \right] 
$$

$$
\mathrm{dz} =  \mathrm{A} -  \mathrm{Y} =  \left[ \mathrm{a}^{\left( 1 \right)}-\mathrm{y}^{\left( 1 \right)}, \mathrm{a}^{\left( 2 \right)}-\mathrm{y}^{\left( 2 \right)},\mathrm{a}^{\left( 3 \right)}-\mathrm{y}^{\left( 3 \right)}....,\mathrm{a}^{\left( \mathrm{m} \right)}-\mathrm{y}^{\left( \mathrm{m} \right)} \right] 
$$


$$
\mathrm{dw} =  \frac{1}{\mathrm{m}}  \mathrm{X} \mathrm{dz}^{\mathrm{T}}
$$


$$
\mathrm{db} =  \frac{1}{\mathrm{m}}\sum_{\mathrm{i}=1}^{\mathrm{m}}{\mathrm{dz}^{\left( \mathrm{i} \right)}}  =  \frac{1}{\mathrm{m}}  \mathrm{np}.\mathrm{sum}\left( \mathrm{dz} \right) 
$$


$$
\mathrm{w}  :=  \mathrm{w}  -  \mathrm{\alpha dw}
$$  

$$
\mathrm{b} :=  \mathrm{b} -  \mathrm{\alpha db}
$$
