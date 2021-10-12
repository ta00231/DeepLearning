# 四、改善深层神经网络 1011

## 1、归一化输入（Normalizing inputs）


### 归一化的原因

但如果特征值在不同
范围，假如𝑥1取值范围从 1 到 1000，特征𝑥2的取值范围从 0 到 1，结果是参数𝑤1和𝑤2值的
范围或比率将会非常不同，这些数据轴应该是𝑤1和𝑤2，但直观理解，我标记为𝑤和𝑏，代价
函数就有点像狭长的碗一样，如果你能画出该函数的部分轮廓，它会是这样一个狭长的函数。


梯度下降法可
能需要多次迭代过程，直到最后找到最小值。但如果函数是一个更圆的球形轮廓，那么不论
从哪个位置开始，梯度下降法都能够更直接地找到最小值，你可以在梯度下降法中使用较大
步长，而不需要像在左图中那样反复执行。

![1011_01.png](attachment:1011_01.png)

### 归一化的步骤

![1011_02.png](attachment:1011_02.png)

#### 1.零均值化

$$
\mathrm{\mu}=\frac{1}{\mathrm{m}}\sum_{\mathrm{i}=1}^{\mathrm{m}}{\mathrm{x}^{\left( \mathrm{i} \right)}}
$$


$$
\mathrm{X}=\mathrm{X}-\mathrm{\mu}
$$


#### 2.归一化方差

我们已经完成零值均化，(𝑥
(𝑖)
)
2元素𝑦
2就是方差

$$
\mathrm{\sigma}^2=\frac{1}{\mathrm{m}}\sum_{\mathrm{i}=1}^{\mathrm{m}}{\left( \mathrm{x}^{\left( \mathrm{i} \right)} \right) ^2}
$$


$$
\mathrm{X}=\frac{\mathrm{X}}{\mathrm{\sigma}^2}
$$


## 2、梯度消失/梯度爆炸（Vanishing/Exploding gradients）

训练神经网络，尤其是深度神经所面临的一个问题就是梯度消失或梯度爆炸，也就是你
训练神经网络的时候，导数或坡度有时会变得非常大，或者非常小，甚至于以指数方式变小，
这加大了训练的难度。


![1011_04.png](attachment:1011_04.png)

随机初始化权重选取，导致的指数型效果，不完整的解决方案如下：

## 3、神经网络的权重初始化

![1011_03.png](attachment:1011_03.png)

先从一个神经元看起：

𝑧 = 𝑤1𝑥1 + 𝑤2𝑥2 + ⋯ + 𝑤𝑛𝑥𝑛，𝑏 = 0，暂时忽略𝑏，为了预防𝑧值过大或过小，你可以看
到𝑛越大，你希望𝑤𝑖越小，因为𝑧是𝑤𝑖𝑥𝑖的和，如果你把很多此类项相加，希望每项值更小，
最合理的方法就是设置𝑤𝑖 =
1/𝑛，𝑛表示神经元的输入特征数量。

则深层网络可以有：

$$
\mathrm{w}^{\left[ \mathrm{i} \right]}=\mathrm{np}.\mathrm{random}.\mathrm{randn}\left( \mathrm{shape} \right) *\mathrm{np}.\mathrm{sqrt}\left( \frac{1}{\mathrm{n}^{\left[ \mathrm{l}-1 \right]}} \right) 
$$


n_[l-1]就是喂给L层的神经单元的数量，即L-1层的神经元数量。

如果你是用的是 Relu 激活函数，而不是1/𝑛，方差设置为2/𝑛，效果会更好，即$$
\mathrm{np}.\mathrm{random}.\mathrm{randn}\left( \mathrm{shape} \right) *\mathrm{np}.\mathrm{sqrt}\left( \frac{2}{\mathrm{n}^{\left[ \mathrm{l}-1 \right]}} \right) 
$$
  
如果使用tanh函数，可以用$$
\mathrm{np}.\mathrm{random}.\mathrm{randn}\left( \mathrm{shape} \right) *\mathrm{np}.\mathrm{sqrt}\left( \mathrm{np}.\mathrm{sqrt}\left( \frac{1}{\mathrm{n}^{\left[ \mathrm{l}-1 \right]}} \right) \right) 
$$



### 4、梯度的数值逼近

梯度检验，确保backprop反向传播正确实施。

在了解梯度检验的前提，我们先了解梯度的数值逼近：  
其实类似于导数的官方定义的双边检测
![1011_05.png](attachment:1011_05.png)

$$
f\prime\left( \mathrm{\theta} \right) =\lim_{\mathrm{\varepsilon}\rightarrow 0} \frac{f\left( \mathrm{\theta}+\mathrm{\varepsilon} \right) -f\left( \mathrm{\theta}-\mathrm{\varepsilon} \right)}{2\mathrm{\epsilon}}
$$


### 5、梯度检验

实际上就是利用W和b，形成θ，然后因为是对代价函数进行梯度下降，所以函数就为J（θ），对齐进行双边误差检验求解dθ。

![1011_06.png](attachment:1011_06.png)

接着看得出的结果与自己的dθ比较（欧几里得范数）

$$
\frac{||\mathrm{d\theta}_{\mathrm{approx}}-\mathrm{d\theta}||_2}{||\mathrm{d\theta}_{\mathrm{approx}}||_2+||\mathrm{d\theta}||_2}
$$


它是误差平方之和，然后求平方根，得到欧式距离，然后用向量长度归一化，使用向量长度的欧几里得范数。

分母只是用于预防这些向量太小或太大，分母使得这个方程式变成比率，我们实际执行这个方程式，，𝜀可能为10^−7，使用这个取值范围内的𝜀，如果你发现计算方程式得到的值为10^−7或更小，这就很好，这就意味着导数逼近很有可能是正确的，它的值非常小。

如果它的值在10^−5范围内，我就要小心了，也许这个值没问题，但我会再次检查这个向量的所有项，确保没有一项误差过大，可能这里有 bug。

在实施神经网络时，我经常需要执行 foreprop 和 backprop，然后我可能发现这个梯度
检验有一个相对较大的值，我会怀疑存在 bug，然后开始调试，调试，调试，调试一段时间
后，我得到一个很小的梯度检验值，现在我可以很自信的说，神经网络实施是正确的。

##### 使用注意事项

![1011_07.png](attachment:1011_07.png)
