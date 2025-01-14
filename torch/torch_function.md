# torch_function

记录常用的torch函数

## torch.randn

- `torch.randn` 函数生成一个张量，其元素是从标准正态分布 $N(0,1)$ 中随机采样的。这种分布的均值为0，标准差为1。在深度学习中，使用标准正态分布初始化模型参数可以帮助模型更快地收敛。

- ``` python
  torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
  ```

- 示例

  - ```python
    import torch
    
    # 生成一个形状为 (2, 3) 的张量，元素从标准正态分布中随机采样
    x = torch.randn(2, 3)
    print(x)
    
    # 生成一个形状为 (2, 3) 的张量，数据类型为 torch.float64
    y = torch.randn(2, 3, dtype=torch.float64)
    print(y)
    
    # 生成一个形状为 (2, 3) 的张量，并存储在 GPU 上（如果可用）
    if torch.cuda.is_available():
        z = torch.randn(2, 3, device='cuda')
        print(z)
    ```

  - ``` pyyhon
    tensor([[ 0.1234, -0.5678,  1.2345],
            [-0.9876,  0.4321, -0.3456]])
    
    tensor([[ 0.6789, -1.2345,  0.5678],
            [ 0.9876, -0.4321,  0.3456]])
    
    tensor([[ 0.1234, -0.5678,  1.2345],
            [-0.9876,  0.4321, -0.3456]], device='cuda:0')
    ```



## torch.randperm

- 作用

- 函数签名

  - ```python
    torch.randperm(n, *, generator=None, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False) -> Tensor
    ```

- 参数说明

  - **n**（int）：生成随机排列的长度，即生成一个从 0 到 n-1 的随机排列。
  - **generator**（torch.Generator，可选）：用于生成随机数的生成器。如果指定，则使用该生成器来生成随机排列。
  - **out**（Tensor，可选）：输出张量。如果指定，则将结果存储在该张量中，而不是创建新的张量。
  - **dtype**（torch.dtype，可选）：输出张量的数据类型，默认为 `torch.int64`。
  - **layout**（torch.layout，可选）：输出张量的布局，默认为 `torch.strided`。
  - **device**（torch.device，可选）：输出张量所在的设备，默认为 CPU。
  - **requires_grad**（bool，可选）：是否为输出张量启用梯度计算，默认为 `False`。

- 示例

  - ``` python
    import torch
    
    # 生成一个长度为 5 的随机排列
    rand_perm = torch.randperm(5)
    print(rand_perm)  # 输出类似于 tensor([2, 4, 0, 1, 3])
    
    # 指定设备为 GPU（如果有）
    rand_perm_gpu = torch.randperm(5, device='cuda')
    print(rand_perm_gpu)  # 输出类似于 tensor([1, 3, 0, 4, 2], device='cuda:0')
    
    # 指定数据类型为 torch.int32
    rand_perm_int32 = torch.randperm(5, dtype=torch.int32)
    print(rand_perm_int32)  # 输出类似于 tensor([4, 2, 0, 3, 1], dtype=torch.int32)
    ```



## torch.randint

- `torch.randint` 是 PyTorch 中用于生成随机整数张量的函数

- 签名

  - ```python
    torch.randint(low, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    ```

- 参数说明

  - **low**：生成随机整数的最小值（包含），默认为0。如果只传入一个参数，则该参数被视为`high`，`low`默认为0。
  - **high**：生成随机整数的最大值（不包含）。例如，`torch.randint(0, 5, (3,))` 会生成一个形状为 `(3,)` 的张量，其元素值在 `[0, 5)` 范围内。
  - **size**：生成的张量的形状，可以是一个整数或整数元组。例如，`(3,)` 表示生成一个一维张量，长度为3；`(2, 3)` 表示生成一个二维张量，形状为2行3列。
  - **generator**：一个`torch.Generator`对象，用于指定随机数生成器。如果不指定，则使用默认的全局随机数生成器。
  - **out**：一个可选的输出张量，如果指定，则将结果存储在该张量中，而不是创建一个新的张量。
  - **dtype**：生成的张量的数据类型，默认为`torch.int64`。可以指定为`torch.int32`、`torch.int16`等。
  - **layout**：张量的布局，默认为`torch.strided`，表示常规的张量布局。
  - **device**：生成的张量所在的设备，默认为CPU。可以指定为`'cuda'`或其他设备。
  - **requires_grad**：是否为张量启用梯度计算，默认为`False`。

- 返回值

  - 返回一个指定形状和数据类型的张量，其元素值为在 `[low, high)` 范围内的随机整数。

- 示例

  - ```python
    import torch
    
    # 生成一个形状为 (3,) 的张量，元素值在 [0, 5) 范围内
    tensor1 = torch.randint(0, 5, (3,))
    print(tensor1)
    
    # 生成一个形状为 (2, 3) 的张量，元素值在 [10, 20) 范围内
    tensor2 = torch.randint(10, 20, (2, 3))
    print(tensor2)
    
    # 指定数据类型为 torch.int32
    tensor3 = torch.randint(0, 5, (3,), dtype=torch.int32)
    print(tensor3)
    
    # 指定设备为 cuda
    tensor4 = torch.randint(0, 5, (3,), device='cuda')
    print(tensor4)
    ```



## torch.allclose

- `torch.allclose` 是 PyTorch 中用于检查两个张量是否在某个容忍度范围内近似相等的函数

- 签名

  - ```python
    torch.allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) → bool
    ```

- 参数说明

  - **input** (Tensor)：第一个输入张量。
  - **other** (Tensor)：第二个输入张量。
  - **rtol** (float, 可选)：相对容忍度，默认为 1e-05。
  - **atol** (float, 可选)：绝对容忍度，默认为 1e-08。
  - **equal_nan** (bool, 可选)：如果为 `True`，则两个 `NaN` 值将被视为相等。默认为 `False`。

- 返回值

  - 一个布尔值，如果两个张量在容忍度范围内是近似的，则返回 `True`，否则返回 `False`。

- 示例

  - ```python
    import torch
    
    # 创建两个张量
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.001, 3.0])
    
    # 比较两个张量是否近似相等
    print(torch.allclose(a, b, rtol=1e-03))  # 输出：True
    print(torch.allclose(a, b, rtol=1e-05))  # 输出：False
    
    # 比较包含 NaN 值的张量
    c = torch.tensor([1.0, float('nan')])
    d = torch.tensor([1.0, float('nan')])
    print(torch.allclose(c, d))  # 输出：False
    print(torch.allclose(c, d, equal_nan=True))  # 输出：True
    ```



## torch.sqeeze

- `squeeze` 方法用于去除张量中大小为 1 的维度

- 签名

  - ```python
    torch.squeeze(input, dim=None) → Tensor
    ```

- 参数说明

  - **input** (Tensor)：输入张量。
  - **dim** (int, 可选)：指定要去除的维度。如果指定，则只去除该维度上的大小为 1 的维度。如果未指定，则去除所有大小为 1 的维度。

- 返回

  - 返回一个新的张量，其形状与输入张量相同，但所有大小为 1 的维度都被去除。

- 示例

  - ```python
    import torch
    
    # 创建一个形状为 (1, 3, 1, 4) 的张量
    x = torch.randn(1, 3, 1, 4)
    print("原始张量形状:", x.shape)
    
    # 去除所有大小为 1 的维度
    y = torch.squeeze(x)
    print("去除所有大小为 1 的维度后:", y.shape)
    
    # 只去除第 0 维度上的大小为 1 的维度
    z = torch.squeeze(x, dim=0)
    print("只去除第 0 维度上的大小为 1 的维度后:", z.shape)
    ```



## torch.unsqueeze

- `unsqueeze` 方法用于在张量的指定位置插入一个大小为 1 的维度

- 签名

  - ```python
    torch.unsqueeze(input, dim) → Tensor
    ```

- 参数说明

  - **input** (Tensor)：输入张量。
  - **dim** (int)：指定插入大小为 1 的维度的位置。

- 返回值

  - 返回一个新的张量，其形状与输入张量相同，但在指定位置插入了一个大小为 1 的维度。

- 示例

  - ```python
    import torch
    
    # 创建一个形状为 (3, 4) 的张量
    x = torch.randn(3, 4)
    print("原始张量形状:", x.shape)
    
    # 在第 0 维度前插入一个大小为 1 的维度
    y = torch.unsqueeze(x, dim=0)
    print("在第 0 维度前插入一个大小为 1 的维度后:", y.shape)
    
    # 在第 1 维度前插入一个大小为 1 的维度
    z = torch.unsqueeze(x, dim=1)
    print("在第 1 维度前插入一个大小为 1 的维度后:", z.shape)
    
    # 在第 2 维度前插入一个大小为 1 的维度
    w = torch.unsqueeze(x, dim=2)
    print("在第 2 维度前插入一个大小为 1 的维度后:", w.shape)
    ```






## torch.split

- 在 PyTorch 中，`split` 方法用于将一个张量分割成多个较小的张量。这个方法非常有用，特别是在需要将一个大的张量分成多个部分进行处理时。`split` 方法可以沿着指定的维度进行分割，并且可以指定每个部分的大小。

- 签名

  - ```python
    torch.split(tensor, split_size_or_sections, dim=0)
    ```

- 参数说明

  - **tensor** (`Tensor`): 要分割的输入张量。
  - split_size_or_sections(int或list):
    - 如果是 `int`，表示每个分割块的大小。输入张量将被分割成多个大小为 `split_size_or_sections` 的块，最后一个块的大小可能小于 `split_size_or_sections`。
    - 如果是 `list`，表示每个分割块的具体大小。输入张量将被分割成多个大小分别为 `list` 中指定的块。
  - **dim** (`int`, 默认值为 0): 沿着哪个维度进行分割。



## torch  ：：

- 切片操作 `::` 用于选择张量中的特定元素。切片操作的语法是 `tensor[start:stop:step]`
  - `start`：起始索引（包含）。默认0
  - `stop`：结束索引（不包含）。默认张量长度
  - `step`：步长。默认1



## 广播机制

- 广播机制遵循以下三个主要规则：
  1. **对齐维度**：如果两个张量的维度数不同，形状较小的张量会在前面补1，直到两个张量的维度数相同。
  2. **扩展维度**：如果两个张量在某个维度上的大小不同，较小的张量会在该维度上进行扩展，使其大小与较大的张量相同。
  3. **兼容性**：如果两个张量在某个维度上的大小相同，或者其中一个张量在该维度上的大小为1，则这两个张量在该维度上是兼容的。如果两个张量在所有维度上都是兼容的，则它们可以进行广播运算。



## torch.size

- `.size()` 方法可以接受一个参数，这个参数指定了要获取的特定维度的大小。这个参数是一个整数，表示张量的某个维度的索引。

- 签名

  - ```python
    tensor.size(dim) → int
    ```

- 返回值

  - 返回指定维度的大小，类型为 `int`。

- **无参数**：`tensor.size()` 返回一个 `torch.Size` 对象，表示张量的形状。

- **有参数**：`tensor.size(dim)` 返回一个 `int`，表示张量在指定维度 `dim` 上的大小。





## torch.argsort

- `argsort` 方法用于返回输入张量中元素的排序索引

- 签名

  - ```python
    torch.argsort(input, dim=None, descending=False) → Tensor
    ```

- 参数说明

  - **input** (Tensor)：输入张量。
  - **dim** (int, 可选)：指定排序的维度。如果未指定，则将输入张量视为一维张量进行排序。默认为 `None`。
  - **descending** (bool, 可选)：是否按降序排序。默认为 `False`，表示按升序排序。

- 返回值

  - 返回一个新的张量，包含输入张量中元素的排序索引。

- 示例

  - ```python
    import torch
    
    # 创建一个一维张量
    x = torch.tensor([4, 1, 3, 2])
    print("原始张量:", x)
    
    # 获取排序索引（升序）
    indices = torch.argsort(x)
    print("排序索引（升序）:", indices)
    
    # 使用排序索引重新排列张量
    sorted_x = x[indices]
    print("排序后的张量（升序）:", sorted_x)
    
    # 获取排序索引（降序）
    indices_desc = torch.argsort(x, descending=True)
    print("排序索引（降序）:", indices_desc)
    
    # 使用排序索引重新排列张量
    sorted_x_desc = x[indices_desc]
    print("排序后的张量（降序）:", sorted_x_desc)
    
    # 创建一个二维张量
    y = torch.tensor([[4, 1, 3], [2, 3, 1]])
    print("原始二维张量:\n", y)
    
    # 按最后一维排序（升序）
    indices_y = torch.argsort(y, dim=-1)
    print("按最后一维排序索引（升序）:\n", indices_y)
    
    # 使用排序索引重新排列张量
    sorted_y = y.gather(dim=-1, index=indices_y)
    print("按最后一维排序后的张量（升序）:\n", sorted_y)
    ```



## torch.arange

- 用于生成等差数列

- 签名

  - ```python
    torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    ```

- 参数

  - **start** (int, 可选)：生成序列的起始值，默认为0。
  - **end** (int)：生成序列的结束值（不包含）。
  - **step** (int, 可选)：生成序列的步长，默认为1。
  - **out** (Tensor, 可选)：输出张量。如果指定，则将结果存储在该张量中，而不是创建一个新的张量。
  - **dtype** (torch.dtype, 可选)：生成的张量的数据类型。如果未指定，则根据 `start`、`end` 和 `step` 的类型推断。
  - **layout** (torch.layout, 可选)：张量的布局，默认为 `torch.strided`。
  - **device** (torch.device, 可选)：生成的张量所在的设备，默认为CPU。
  - **requires_grad** (bool, 可选)：是否为张量启用梯度计算，默认为 `False`。

- 返回值

  - 返回一个包含在 `[start, end)` 区间内均匀间隔值的张量。

- 示例

  - ```python
    import torch
    
    # 生成一个从 0 到 9 的张量
    x = torch.arange(10)
    print("从 0 到 9 的张量:", x)
    
    # 生成一个从 1 到 10 的张量
    y = torch.arange(1, 11)
    print("从 1 到 10 的张量:", y)
    
    # 生成一个从 0 到 9，步长为 2 的张量
    z = torch.arange(0, 10, 2)
    print("从 0 到 9，步长为 2 的张量:", z)
    
    # 生成一个从 5 到 0，步长为 -1 的张量
    w = torch.arange(5, 0, -1)
    print("从 5 到 0，步长为 -1 的张量:", w)
    
    # 指定数据类型为 torch.float32
    a = torch.arange(0, 10, dtype=torch.float32)
    print("指定数据类型为 torch.float32 的张量:", a)
    
    # 指定设备为 cuda
    b = torch.arange(0, 10, device='cuda')
    print("指定设备为 cuda 的张量:", b)
    ```





## X.contiguous().view(-1,d_model)

- 在 PyTorch 中，张量在内存中的存储可以是不连续的，这通常发生在以下几种情况：
  - **切片操作**：对张量进行切片操作可能会导致张量在内存中变得不连续。
  - **转置操作**：对张量进行转置操作（如 `permute` 或 `transpose`）也会使张量变得不连续。
  - **拼接操作**：对多个张量进行拼接操作（如 `cat`）可能会导致结果张量不连续。
- `.contiguous()` 方法会返回一个与原张量内容相同但在内存中连续存储的新张量。这在以下几种情况下非常有用：
  1. **使用某些需要连续张量的操作**：例如，`view` 方法要求输入张量在内存中是连续的。如果张量不连续，调用 `view` 会引发错误。
  2. **优化性能**：连续存储的张量在进行某些操作时可以更高效地利用内存和缓存，从而提高性能。
- 在 PyTorch 中，`view` 方法用于改变张量的形状，而不改变其数据。`view` 方法类似于 NumPy 中的 `reshape` 方法，但它要求输入张量在内存中是连续的。如果张量不连续，调用 `view` 方法会引发错误。
  - view中的参数可以填写1~n个数，其中-1代表自动计算剩下的维数。
  - 这里(-1,d_model)就是代表一个二维张量



## torch.nn

`from torch import nn`

`import torch.nn.functional as F`

### nn.TransformerEncoderLayer

- 作用简介

  - 编码器的作用是将输入序列编码为一种高维表示，以捕捉输入序列的语义信息。`nn.TransformerEncoderLayer` 用于定义编码器中的一个层，它由多个子层组成，包括自注意力机制（self-attention）、前馈神经网络和残差连接（residual connection）等

- 函数签名

  - ```python
    class torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None)
    ```

- 参数说明

  - **d_model** (`int`): 输入和输出的特征维度。
  - **nhead** (`int`): 自注意力机制中的头数。
  - **dim_feedforward** (`int`, 默认值为 2048): 前馈神经网络中隐藏层的维度。
  - **dropout** (`float`, 默认值为 0.1): dropout 的比例。
  - **activation** (`str` 或 `Callable[[Tensor], Tensor]`, 默认值为 `'relu'`): 前馈神经网络中的激活函数。可以是字符串（如 `'relu'` 或 `'gelu'`）或自定义的激活函数。
  - **layer_norm_eps** (`float`, 默认值为 `1e-5`): 层归一化中的小常数，用于数值稳定性。
  - **batch_first** (`bool`, 默认值为 `False`): 如果为 `True`，则输入和输出张量的形状为 `(batch, seq, feature)`，否则为 `(seq, batch, feature)`。
  - **norm_first** (`bool`, 默认值为 `False`): 如果为 `True`，则在每个子层之前进行层归一化，否则在每个子层之后进行层归一化。
  - **device** (`torch.device`, 可选): 指定张量所在的设备（CPU 或 GPU）。
  - **dtype** (`torch.dtype`, 可选): 指定张量的数据类型。

- 示例代码

  - ```python
    import torch
    import torch.nn as nn
    
    # 创建一个 Transformer 编码器层
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=False
    )
    
    # 创建一个输入张量，形状为 (batch_size, seq_len, d_model)
    batch_size = 1
    seq_len = 10
    d_model = 512
    input_tensor = torch.randn(batch_size, seq_len, d_model)
    
    # 创建一个 Transformer 编码器，只包含一个编码器层
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
    
    # 前向传播
    output = encoder(input_tensor)
    print(output.shape)  # 输出形状应为 (batch_size, seq_len, d_model)
    ```

  - ```python
    torch.Size([1, 10, 512])
    ```



### nn.RNN

- rnn 天然编码了位置信息

  - $h_{t}=f(h_{t-1},x_t)$
  - $f$ 是非线性激活函数，$h_{t-1}$ 是前一时间步的隐藏状态，$x_t$ 是当前时间步的输入。由于 $h_t$ 依赖于 $h_{t-1}$，而 $h_{t-1}$ 又依赖于 $h_{t-2}$，以此类推，隐藏状态包含了从初始时间步到当前时间步的所有历史信息。这种递归结构使得位置信息被隐式地编码在隐藏状态中。
  - RNN 通过其递归结构隐式地编码位置信息，而 Transformer 需要通过**显式添加位置编码**来获取位置信息。

- 签名

  - ```python
    torch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0, bidirectional=False)
    ```

- 参数说明

  - **input_size**（int）：输入特征的维度。例如，对于文本数据，如果每个词用 100 维的向量表示，则 `input_size` 为 100。
  - **hidden_size**（int）：隐藏层的特征维度。这是 RNN 单元内部状态的维度。
  - **num_layers**（int，可选）：RNN 的层数。默认为 1。多层 RNN 可以增加模型的复杂度和表达能力。
  - **nonlinearity**（str，可选）：非线性激活函数。可以是 `'tanh'` 或 `'relu'`。默认为 `'tanh'`。
  - **bias**（bool，可选）：是否在 RNN 单元中使用偏置项。默认为 `True`。
  - **batch_first**（bool，可选）：如果为 `True`，则输入和输出张量的形状为 `(batch, seq, feature)`，否则为 `(seq, batch, feature)`。默认为 `False`。
  - **dropout**（float，可选）：如果非零，则在 RNN 的每一层之间引入 Dropout 层，以防止过拟合。默认为 0。
  - **bidirectional**（bool，可选）：如果为 `True`，则使用双向 RNN。默认为 `False`。

- 返回值

  - **output**（Tensor）：RNN 的输出。形状为 `(seq_len, batch, num_directions * hidden_size)` 或 `(batch, seq_len, num_directions * hidden_size)`，取决于 `batch_first` 参数。
  - **h_n**（Tensor）：最后一个时间步的隐藏状态。形状为 `(num_layers * num_directions, batch, hidden_size)`。



## nn.Embedding

- 创建嵌入层的模块。嵌入层通常用于将离散的输入（如单词索引）转换为连续的向量表示

- 签名

  - ```python
    torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
    ```

- 参数说明

  - **num_embeddings**（int）：嵌入层的词汇表大小，即嵌入矩阵的行数。
  - **embedding_dim**（int）：嵌入向量的维度，即嵌入矩阵的列数。
  - **padding_idx**（int，可选）：如果指定，该索引对应的嵌入向量将被初始化为零，并且在训练过程中保持不变。这通常用于处理填充（padding）符号。默认为 `None`。
  - **max_norm**（float，可选）：如果指定，每个嵌入向量的范数将被重新缩放到不超过这个值。默认为 `None`。
  - **norm_type**（float，可选）：用于计算范数的类型。默认为 2.0，即 L2 范数。
  - **scale_grad_by_freq**（bool，可选）：如果为 `True`，则在反向传播时，嵌入向量的梯度将按其频率进行缩放。默认为 `False`。
  - **sparse**（bool，可选）：如果为 `True`，则梯度将被稀疏化，这可以节省内存。默认为 `False`。
  - **_weight**（Tensor，可选）：如果指定，将使用这个张量作为嵌入矩阵的初始权重。默认为 `None`。

- 返回值

  - 返回一个嵌入向量，形状为 `(input.size(0), input.size(1), embedding_dim)`，其中 `input.size(0)` 和 `input.size(1)` 分别是输入张量的批大小和序列长度。

- 示例

  - ```python
    import torch
    import torch.nn as nn
    
    # 定义嵌入层
    num_embeddings = 10  # 词汇表大小
    embedding_dim = 3  # 嵌入向量的维度
    embed = nn.Embedding(num_embeddings, embedding_dim)
    
    # 创建输入数据
    # 假设我们有一个批量大小为 2，序列长度为 5 的输入
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    
    # 前向传播
    embeded = embed(input)
    
    print("Embedding shape:", embeded.shape)  # 输出形状: (2, 5, 3)
    print("Embedding:", embeded)
    ```



### torch.bmm

- 正序、非转置相乘

- `torch.bmm` 是 PyTorch 中用于执行批量矩阵乘法（Batch Matrix-Matrix Multiplication）的函数。它的名字 "bmm" 表示 "batch matrix multiplication"。这个函数用于计算两个三维张量的矩阵乘法，每个张量都包含多个矩阵，这些矩阵在批量维度上进行独立的乘法操作。

- 签名

  - ```python
    torch.bmm(input, mat2, *, out=None) → Tensor
    ```

- 参数说明

  - **input** (`Tensor`): 第一批要相乘的矩阵，形状为 `(batch_size, n, m)`。
  - **mat2** (`Tensor`): 第二批要相乘的矩阵，形状为 `(batch_size, m, p)`。
  - **out** (`Tensor`, 可选): 输出张量，形状为 `(batch_size, n, p)`。

- 作用

  - `torch.bmm` 函数将批中的每对矩阵相乘，返回一个新的三维张量，形状为 `(batch_size, n, p)`。具体来说，如果 `input` 的形状为 `(batch_size, n, m)`，`mat2` 的形状为 `(batch_size, m, p)`，那么 `out` 的形状将为 `(batch_size, n, p)`。每个批次中的矩阵乘法独立进行，即：

  - $$
    out_i=input_i×mat2_i
    $$



### F.linear

- 在 PyTorch 中，`F.linear` 是一个用于执行线性变换的函数，它与 `nn.Linear` 层的功能类似，但使用方式略有不同。`F.linear` 通常用于在自定义层或函数中直接应用线性变换，而 `nn.Linear` 是一个预定义的层，可以直接添加到模型中。

- ```python
  torch.nn.functional.linear(input, weight, bias=None)
  ```

- 计算公式是$Y=XW^T+b$

- 参数说明

  - **input** (`Tensor`): 输入张量，形状为 `(N, *, in_features)`，其中 `*` 表示任意数量的额外维度。
  - **weight** (`Tensor`): 权重张量，形状为 `(out_features, in_features)`。
  - **bias** (`Tensor`, 可选): 偏置张量，形状为 `(out_features,)`。如果为 `None`，则不添加偏置。



### F.softmax

- 在 PyTorch 中，`F.softmax` 是一个用于计算 softmax 函数的函数。Softmax 函数通常用于多分类问题中，将模型的输出转换为概率分布。`F.softmax` 是 `torch.nn.functional` 模块中的一个函数，可以方便地在前向传播中使用。

- Softmax 函数将输入张量的每个元素转换为一个概率值，使得每个切片（沿着指定的维度）的元素和为 1。具体公式为： 

  - $$
    softmax(x_i)=\dfrac{exp⁡(x_i)}{∑_jexp⁡(x_j)}
    $$

  - 

- 其中，$x_i $是输入张量中的一个元素，$∑_jexp⁡(x_j)$是沿着指定维度的所有元素的指数和。

- 函数自签

  - ```python
    torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
    ```

- 参数说明

  - **input** (`Tensor`): 输入张量，通常是模型的输出。
  - **dim** (`int`): 沿着哪个维度计算 softmax。这个维度上的每个切片将被转换为概率分布。
  - **_stacklevel** (`int`, 默认值为 3): 用于错误消息的堆栈级别，通常不需要手动设置。
  - **dtype** (`torch.dtype`, 可选): 指定输出张量的数据类型。如果为 `None`，则输出张量的数据类型与输入张量相同。

- dim=-1 代表对最后一个维度进行操作  



### F.layer_norm

- `F.layer_norm` 是 PyTorch 中用于执行层归一化（Layer Normalization）的函数。层归一化是一种归一化技术，通常用于神经网络中的每个样本，而不是像批量归一化（Batch Normalization）那样对整个批次进行归一化。层归一化有助于稳定训练过程，特别是在处理具有复杂结构的模型（如Transformer）时。

- 签名

  - ```python
    torch.nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)
    ```

- 参数说明

  - **input** (`Tensor`): 输入张量，需要进行层归一化的数据。
  - **normalized_shape** (`List[int]` 或 `torch.Size`): 一个整数列表或 `torch.Size` 对象，表示需要归一化的维度的形状。这些维度通常是输入张量的最后几个维度。
    - 这些维度必须是输入张量的最后几个连续维度。
    - **最后一个维度**: 当 `normalized_shape` 只有一个元素时，表示归一化操作将作用于输入张量的最后一个维度。
    - 
  - **weight** (`Tensor`, 可选): 可选的权重张量，用于缩放归一化后的值。默认为 `None`。
  - **bias** (`Tensor`, 可选): 可选的偏置张量，用于偏移归一化后的值。默认为 `None`。
  - **eps** (`float`, 默认值为 `1e-5`): 一个小的常数，用于数值稳定性，防止分母为零。

- 

