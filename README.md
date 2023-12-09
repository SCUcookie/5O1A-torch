# Tensor

### 简介

`Tensor`模块实现了一个名为 `Tensor` 的张量类，用于进行张量操作和自动微分。该模块具有一些基本的数学运算和张量操作，同时支持自动微分以便进行梯度计算。

### 类方法

- `__add__(self, other)`：实现张量的加法。
- `__mul__(self, other)`：实现张量的乘法。
- `__matmul__(self, other)`：实现矩阵乘法。
- `__copy__(self)`：复制张量的 `grad`、`data`、`requires_grad` 属性。
- `numpy(self)`：返回张量的数据的副本。
- `__len__(self)`：返回张量的长度。
- `size`：返回张量的大小。
- `ndim`：返回张量的维度。
- `shape`：返回张量的形状。
- `swapaxes(self, axis1, axis2)`：转置张量的两个轴。
- `reshape(self, *new_shape)`：重新塑造张量的形状。



# Optim

## SGD（随机梯度下降）

### 简介

随机梯度下降是一种简单而有效的优化算法，广泛应用于机器学习和深度学习。其关键思想是沿着损失函数相对于参数的梯度的相反方向更新模型参数。

### 参数

```python
def __init__(self, param_list: list, learning_rate=0.01, momentum=0., decay=0.)
```

- **学习率 (`lr`)**：用于更新参数的步长。
- **动量 (`momentum`)**： 在相关方向上加速 SGD 并减缓振荡的参数。
- **衰减 (`decay`)** ：每次更新的学习率衰减。

## Adam

### 简介

Adam（自适应矩估计）是一种流行的优化算法，结合了动量和RMSprop的思想。它为每个参数维护自适应学习率，并指数衰减过去梯度的平均值。

### 参数

```python
def __init__(self, param_list: list, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
```

- **学习率 (`lr`):** 用于更新参数的步长。
- **Beta1 (`beta1`):** 第一矩估计的指数衰减率。
- **Beta2 (`beta2`):** 第二矩估计的指数衰减率。
- **Epsilon (`epsilon`):** 添加到分母的小常数，防止除以零。

## 优化器 API

两个优化器（`SGD` 和 `Adam`）共享在 `Optimizer` 基类中定义的公共 API。

### 类方法

- `optimizer.zero_grad()`：将所有优化参数的梯度重置为零。

- `optimizer.step()`：根据优化算法更新参数。



# Autograd

### 简介

`Autograd`模块提供对张量操作的自动微分功能。该模块能够定义和执行张量操作，同时利用反向模式自动微分来自动计算梯度。

### Context 类

`Context` 类提供了在前向传播期间跟踪数据的上下文，这些数据在后向传播期间可以用于计算梯度。

### Autograd 类

`Autograd` 类是一个抽象基类，作为定义自定义自动微分操作的基础。

#### 类方法

- **`forward(ctx, \*tensor_list)`**：定义操作的前向传播的抽象方法。
- **`backward(ctx, grad)`**：定义计算梯度的反向传播的抽象方法。



# Dataset

## Dataloader

### 简介

`Dataloader` 类用于加载数据集并提供数据迭代器。

### 类方法

- **__init__(self, dataset)**：初始化方法，接受一个数据集作为参数。
- **get_single_iterable(self) -> SingleDataset**：返回单样本数据集迭代器的实例。
- **get_batch_iterable(self, batch_size) -> BatchDataset**：返回批量数据集迭代器的实例，可以指定批量大小。
- **get_all_batches(self, shuffle=False)**：返回所有数据批次，可选择是否打乱顺序。

## SingleDataset

### 简介

`SingleDataset` 类表示单样本数据集，继承自抽象类 `Dataset`。

### 类方法

- **__init__(self, dataset)**：初始化方法，接受一个数据集作为参数，并创建索引。
- **shuffle(self)**：将数据集索引打乱。
- **__len__(self)**：返回数据集的长度。
- **__getitem__(self, idx)**：获取单个样本数据。

## BatchDataset

### 简介

`BatchDataset` 类表示批量数据集，继承自抽象类 `Dataset`。

### 类方法

- **__init__(self, dataset, batch_size)**：初始化方法，接受一个数据集和批量大小作为参数，并创建索引。
- **shuffle(self)**：将数据集索引打乱。
- **__len__(self)**：返回数据集批次的数量。
- **__getitem__(self, idx)**：获取指定批次的数据。

## Dataset

### 简介

`Dataset` 类是一个抽象基类，定义了数据集的基本接口。

### 类方法

- **__getitem__(self, item)**：抽象方法，根据索引获取数据。
- **__len__(self)**：抽象方法，返回数据集的长度。

## MNIST

### 简介

`MNIST` 类表示手写数字数据集 MNIST，继承自抽象类 `Dataset`。

### 类方法

- **__init__(self, images_path, labels_path, train, flatten_input, one_hot_output, input_normalization)**：初始化方法，接受一系列参数用于加载 MNIST 数据集。
- **__getitem__(self, idx)**：根据索引获取数据，返回经过预处理的输入和输出。
- **__len__(self)**：返回数据集的长度。
- **download_if_not_exists(url, path)**：静态方法，用于下载数据集文件。
- **_load_mnist(images_path, labels_path, flatten_input, one_hot_output)**：静态方法，加载 MNIST 数据集的图像和标签。



# Module

## activition

### 简介

`activition`模块实现了不同激活函数的前向计算以及反向传播功能。

### 函数实现

- `ReLU` 类表示修正线性单元激活函数。
- `Sigmoid` 类表示 Sigmoid 激活函数。
- `Softmax` 类表示 Softmax 激活函数。
- `Softplus` 类表示 Softplus 激活函数。
- `Softsign` 类表示 Softsign 激活函数。
- `ArcTan` 类表示反正切激活函数。
- `Tanh` 类表示双曲正切激活函数。

## container

### 简介

`containe`中的`Sequential` 类是继承自 `Module` 类的深度学习模块，用于按顺序组合其他模块。

### 类方法

- **__init__(self, \*sequences)**：初始化方法，接受多个模块作为参数，并将它们按顺序存储在 `module_list` 中。同时，尝试获取每个模块的状态字典，并将其注册为参数。
- **forward(self, \*input)**：正向传播方法，接受多个输入参数，通过按顺序调用每个模块的正向传播方法实现整个模型的正向传播。

## conv

### 简介

`conv`模块通过`Conv2`类实现卷积核的初始化以及前向计算功能。

### 类方法

- **__init__(self, in_channels, out_channels, kernel_size, stride, padding=0, add_bias=True)**：初始化方法，接受输入通道数（`in_channels`）、输出通道数（`out_channels`）、卷积核大小（ `kernel_size`）、步长（`stride`）、填充（`padding`）、是否偏置（`add_bias`）参数。根据参数设置卷积层的基本属性，并初始化权重和偏置。
- **reset_parameters(self)**：重新初始化权重和偏置参数，使用 `Kaiming` 初始化权重。
- **forward(self, input: Tensor) -> Tensor**：正向传播方法，接受输入张量，通过卷积操作计算输出。内部使用 `Img2Col` 类将输入转换为矩阵，然后进行矩阵乘法计算。

## creation

### 简介

`creation`模块包含了一些用于初始化张量的方法。

### 函数实现

- **emty(shape, dtype=np.float32, requires_grad=False)**：创建一个给定形状和数据类型的空张量。
- **empty_like(other, dtype=None, requires_grad=False)**：创建一个与目标形状和类型一致的空张量。
- **ones(shape, dtype=np.float32, requires_grad=False)**：创建一个元素全为 1 的张量。
- **ones_like(other, dtype=None, requires_grad=False)**：创建一个与目标形状和类型一致的元素全为 1 的张量。
- **zeros(shape, dtype=np.float32, requires_grad=False)**：创建一个元素全为 0 的张量。
- **zeros_like(other, dtype=None, requires_grad=False)**：创建一个与目标形状和类型一致的元素全为 0 的张量。
- **rands(shape, requires_grad=False)**：创建一个元素服从标准正态分布的张量。

## flatten

### 简介

`flatten`模块中的`Flatten`类实现了将输入张量展平的功能。

### 类方法

- **forward(self, ctx, input)**：正向传播方法，将输入张量展平，并保存展平前的形状用于反向传播。
- **backward(self, ctx, grad)**：反向传播方法，根据保存的形状信息，将梯度重新恢复成原始形状。

## img2col

### 简介

`Img2Col` 用于实现卷积操作中的图像转换为矩阵的过程。

### 类方法

- **__init__(self, kernel_size, stride: Union[int, Tuple[int, int]] = 1)**：初始化方法，接受卷积核大小（`kernel_size`）和步幅（`stride`）参数。
- **img2col_forward(kernel_size, stride, merge_channels, image)**：静态方法，实现图像转换为矩阵的正向传播。根据卷积核大小和步幅，将输入图像转换为二维矩阵。
- **img2col_backward(kernel_size, stride, back_shape, grad)**：静态方法，实现图像转换为矩阵的反向传播。根据卷积核大小和步幅，将梯度矩阵还原为与输入图像相同形状的梯度。
- **forward(self, ctx: Context, image: np.array) -> np.array**：正向传播方法，将输入图像转换为矩阵，并保存输入图像形状信息用于反向传播。
- **backward(self, ctx: Context, grad: np.array = None) -> np.array**：反向传播方法，根据保存的形状信息和梯度矩阵，还原为输入图像的梯度。

## init

### 简介

初始化模块。

### 函数实现

- `_calculate_fan_in_and_fan_out(tensor)`：计算张量的输入和输出通道数量。
- ` _calculate_correct_fan(tensor, mode)`：计算正确的输入或输出通道数量。
- `calculate_gain(nonlinearity, param=None)`：返回给定非线性函数的建议增益值。
- `kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')`：使用均匀分布方法填充输入张量，实现 Kaiming 初始化。
- `uniform_(tensor, low, high)`：使用均匀分布方法填充输入张量。
- `ones_(tensor)`：使用常数 1.0 填充输入张量。
- `zeros_(tensor)`：使用常数 0.0 填充输入张量。

## linear

### 简介

`linear`模块中的`Linear` 类用于实现全连接层的初始化以及前向计算。

### 类方法

- **__init__(self, in_features, out_features, bias=True)**：初始化方法，接受输入特征数、输出特征数和是否包含偏置的参数。初始化权重和偏置参数。
- **reset_parameters(self)**：重新初始化权重和偏置参数，使用 Kaiming 初始化权重。
- **forward(self, input)**：正向传播方法，计算线性层的输出。

## loss

### 简介

`loss`模块用于实现不同损失函数的计算以及反向传播。

### MSELoss

`MSELoss` 类用于实现均方误差损失函数。

- **__init__(self)**：初始化方法，继承自 `Autograd` 类。
- **forward(self, ctx, target, input)**：正向传播方法，计算均方误差损失。
- **backward(self, ctx, grad)**：反向传播方法，计算均方误差损失的梯度。

### CrossEntropyLoss

`CrossEntropyLoss` 类用于实现交叉熵损失函数。

- **__init__(self)**：初始化方法，继承自 `Autograd` 类。

- **forward(self, ctx, target, input)**：正向传播方法，计算交叉熵损失。

- **backward(self, ctx, grad)**：反向传播方法，计算交叉熵损失的梯度。

## manipulation

### 简介

`manipulation`模块用于在计算图中进行数据操作，包括交换数组轴 (`SwapAxes`)、改变数组形状 (`Reshape`) 以及获取数组特定项 (`GetItem`)。

### SwapAxes

`SwapAxes` 类用于交换数组的两个轴。

- **__init__(self, axis1, axis2)**：初始化方法，接受两个轴的索引。
- **forward(self, ctx, input)**：正向传播方法，交换输入数组的两个轴。
- **backward(self, ctx, grad)**：反向传播方法，交换梯度数组的两个轴。

### Reshape

`Reshape` 类用于改变数组的形状。

- **__init__(self, \*new_shape)**：初始化方法，接受新的形状参数。
- **forward(self, ctx, input)**：正向传播方法，将输入数组 reshape 为新的形状，并保存原始形状用于反向传播。
- **backward(self, ctx, grad)**：反向传播方法，将梯度数组 reshape 为原始形状。

### GetItem

`GetItem` 类用于获取数组的特定项。

- **__init__(self, item)**：初始化方法，接受要获取的项的索引。
- **forward(self, ctx, input)**：正向传播方法，获取输入数组的特定项，并保存原始形状用于反向传播。
- **backward(self, ctx, grad)**：反向传播方法，将梯度赋值给原始形状的对应项。

## mathematical

### 简介

`mathematical`实现了各种常用的数学运算功能。

### Add

`Add` 类用于执行两个数组的加法。主要方法如下：

- **forward(self, ctx, x, y)**:
  - 正向传播方法，返回两个输入数组的和。
- **backward(self, ctx, grad)**:
  - 反向传播方法，返回梯度数组。

### Subtract

`Subtract` 类用于执行两个数组的减法。主要方法如下：

- **forward(self, ctx, x, y)**:
  - 正向传播方法，返回两个输入数组的差。
- **backward(self, ctx, grad)**:
  - 反向传播方法，返回梯度数组。

### MatMul

`MatMul` 类用于执行两个数组的矩阵乘法。

- **forward(self, ctx, x, y)**：正向传播方法，返回两个输入数组的矩阵乘法结果。
- **backward(self, ctx, grad: np.array)**：反向传播方法，返回梯度数组。

### Multiply

`Multiply` 类用于执行两个数组的元素乘法。

- **forward(self, ctx, x, y)**：正向传播方法，返回两个输入数组的元素乘法结果。
- **backward(self, ctx, grad: np.array)**：反向传播方法，返回梯度数组。

### Assign

`Assign` 类用于返回输入数组。

- **forward(self, ctx, x)**：正向传播方法，返回输入数组。
- **backward(self, ctx, grad)**：反向传播方法，返回空。

### Divide

`Divide` 类用于执行两个数组的元素除法。

- **forward(self, ctx, x, y)**：正向传播方法，返回两个输入数组的元素除法结果。
- **backward(self, ctx, grad)**：反向传播方法，返回梯度数组。

### Negative

`Negative` 类用于对输入数组进行取负操作。

- **forward(self, ctx, x)**：正向传播方法，返回输入数组的负值。
- **backward(self, ctx, grad)**：反向传播方法，返回负梯度数组。

### Positive

`Positive` 类用于对输入数组进行取正操作。

- **forward(self, ctx, x)**：正向传播方法，返回输入数组。
- **backward(self, ctx, grad)**：反向传播方法，返回梯度数组。

### Power

`Power` 类用于执行两个数组的指数幂运算。

- **forward(self, ctx, x, y)**：正向传播方法，返回两个输入数组的指数幂运算结果。
- **backward(self, ctx, grad)**：反向传播方法，返回梯度数组。

### Exp

`Exp` 类用于对输入数组进行指数函数运算。

- **forward(self, ctx, x)**：正向传播方法，返回输入数组的指数函数运算结果。
- **backward(self, ctx, grad)**：反向传播方法，返回梯度数组。

### Log

`Log` 类用于对输入数组进行对数函数运算。

- **forward(self, ctx, x)**：正向传播方法，返回输入数组的对数函数运算结果。

- **backward(self, ctx, grad)**：反向传播方法，返回梯度数组。

## module

### 简介

`Module` 类是深度学习模型中所有模块的基类。

### 类方法

- `__init__(self)`：类的初始化方法，初始化模块的参数字典 `_parameters`。
- `register_parameter(self, \*var_iterable)`：注册模块参数的方法，接受一个或多个变量名和对应变量的键值对，并将其存储在 `_parameters` 中。
- `parameters(self) -> list`：获取模块所有参数的方法，返回参数列表。
- `get_state_dict(self) -> OrderedDict`：获取模块状态字典的方法，返回包含模块所有参数及其对应值的有序字典。
- `load_state_dict(self, state_dict: OrderedDict)`：加载模块状态字典的方法，接受一个状态字典，并将其中的值赋给模块的对应参数。
- `forward(self, \*input) -> Tensor`：正向传播方法，需要在子类中实现。接受输入并返回输出的张量。
- `__call__(self, \*input) -> Tensor`：通过调用实例来执行正向传播，等效于调用 `forward` 方法。

## pooling

### BasePool 类

`BasePool` 类是所有池化模块的基类，继承自 `Autograd` 类和 `ABC` 抽象基类。

- `__init__(self, kernel_size, stride=1)`：初始化池化模块的方法，接受池化核大小 `kernel_size` 和步幅 `stride`。
- `_fill_col(to_fill, new_shape)`：填充列向量的方法，用于将列向量沿指定维度复制多次，以满足指定的形状。

### MaxPool2d 类

`MaxPool2d` 类用于实现最大池化模块。

- `orward(self, ctx: Context, input)`：前向传播方法，对输入进行最大池化操作。保存相关信息以备后向传播使用。
- `backward(self, ctx: Context, grad: np.array = None)`：反向传播方法，计算最大池化的梯度。

### AvgPool2d 类

`AvgPool2d` 类用于实现平均池化模块。

- `forward(self, ctx: Context, input)`：前向传播方法，对输入进行平均池化操作。保存相关信息以备后向传播使用。
- `backward(self, ctx, grad)`：反向传播方法，计算平均池化的梯度。

## summary

###简介

`summary` 函数用来打印显示网络结构和参数。

###使用

调用`summary` 函数示例：`summary(model, input_size)`

- `model`为需要打印的模型
- `input_size`为网络结构，如`input_size = (1, 3, 32, 32)`

