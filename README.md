<div align="center">

# Ember

### A from-scratch autograd engine and neural-network library — written in pure Python, with **zero runtime dependencies**.

[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-0-blue)](requirements.txt)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](https://github.com/WsNodeX/ember)
[![Tests](https://img.shields.io/badge/tests-60%20passing-brightgreen)](tests)
[![Pure Python](https://img.shields.io/badge/implementation-pure%20Python-orange)](ember)

**Train a neural network with nothing but the Python standard library.**

`pip install` not required. `numpy` not required. Just `import ember`.

</div>

---

<p align="center">
  <a href="#-why">✨ 为什么</a> ·
  <a href="#-features">🚀 特性</a> ·
  <a href="#-install">📦 安装</a> ·
  <a href="#-quick-start">⚡ 快速开始</a> ·
  <a href="#-architecture">🏗️ 架构</a> ·
  <a href="#-api">🔌 API</a> ·
  <a href="#-roadmap">🧭 路线图</a> ·
  <a href="#-contributing">🤝 贡献</a> ·
  <a href="#-license">📄 许可证</a>
</p>

---

## ✨ 为什么

Every deep-learning framework — PyTorch, JAX, TensorFlow — rests on the same
two ideas: **a tensor that remembers how it was built**, and a **reverse-mode
differentiation pass** that walks that record backwards to compute gradients.
Ember is a small, readable implementation of exactly those two ideas, plus the
layer / loss / optimizer scaffolding you need to actually train models.

Why build one by hand?

- **Understanding.** Reading a 3,000-line autograd implementation teaches you
  more about backpropagation in an afternoon than a semester of treating
  `.backward()` as magic.
- **Zero dependencies.** It runs on a bare Python interpreter — a Raspberry Pi,
  an interview whiteboard VM, an airplane with no network. No `pip install`,
  no version conflicts.
- **A real résumé artifact.** A from-scratch engine is the kind of project that
  signals *you actually understand* the math behind the models, not just how to
  call a high-level API.

Ember is not a competitor to PyTorch. It is the thing PyTorch is made of,
shrunk until you can read all of it in one sitting.

## 🚀 特性

- **Reverse-mode autograd** over scalars, vectors and matrices, built from a
  dynamic computation graph — the same model as modern frameworks.
- **A pure-Python tensor backend** (`ember._array`) implementing matmul,
  broadcasting, softmax, reductions and common activations with **no NumPy**.
- **Neural-network blocks**: `Linear`, `ReLU` / `Sigmoid` / `Tanh` /
  `Softmax`, `Sequential`, and a `Module` base class with automatic parameter
  discovery.
- **Losses with analytic, numerically-stable backward passes**: `MSELoss`,
  `CrossEntropyLoss`, `BCEWithLogitsLoss`.
- **An `SGD` optimizer** with momentum, Nesterov acceleration and L2 weight
  decay.
- **Data utilities**: `TensorDataset` and a mini-batch `DataLoader` with
  shuffling.
- **Tested with 60 unit tests**, every operation validated against central
  finite-difference gradients.

## 📦 安装

No build step, no dependencies. Clone and use:

```bash
git clone https://github.com/WsNodeX/ember.git
cd ember
python -c "import ember; print(ember.__version__)"
```

Or, if you prefer a package install:

```bash
pip install -e .
```

Ember supports **Python 3.9+** and runs on Windows, macOS and Linux.

## ⚡ 快速开始

**Train a linear model** (`python examples/fit_linear.py`):

```python
import random
import ember as eb
from ember import Tensor, nn, SGD, TensorDataset, DataLoader

# Synthetic data: y = 3x + 2 + noise
rng = random.Random(42)
xs = [Tensor([rng.uniform(-2, 2)]) for _ in range(200)]
ys = [Tensor([3.0 * x.data[0] + 2.0 + rng.gauss(0, 0.05)]) for x in xs]

model = nn.Linear(1, 1, rng=random.Random(0))
opt = SGD(model.parameters(), lr=0.05)
loader = DataLoader(TensorDataset(xs, ys), batch_size=32, shuffle=True, rng=random.Random(1))

for _ in range(200):
    for xb, yb in loader:
        loss = nn.MSELoss()(model(xb), yb)
        loss.backward(); opt.step(); model.zero_grad()

print(model.weight.data, model.bias.data)
# -> weight ≈ [[3.0]], bias ≈ [2.0]
```

**Solve XOR** (`python examples/xor_mlp.py`) — a 2→8→2 MLP trained with
cross-entropy reaches **4/4 accuracy** in a fraction of a second:

```python
model = nn.Sequential(
    nn.Linear(2, 8, rng=random.Random(2)),
    nn.Tanh(),
    nn.Linear(8, 2, rng=random.Random(3)),
)
opt = SGD(model.parameters(), lr=0.1)
xb = Tensor([[0,0],[0,1],[1,0],[1,1]])
for _ in range(1500):
    loss = nn.CrossEntropyLoss()(model(xb), [0,1,1,0])
    loss.backward(); opt.step(); model.zero_grad()
```

Expected output of `examples/xor_mlp.py`:

```
epoch    0  loss=0.8014
epoch  300  loss=0.1080
...
epoch 1499  loss=0.0085

XOR predictions:
  [0, 0] -> class 0 (truth 0)
  [0, 1] -> class 1 (truth 1)
  [1, 0] -> class 1 (truth 1)
  [1, 1] -> class 0 (truth 0)

accuracy: 4/4
```

## 🏗️ 架构

Ember is layered. Each layer depends only on the one below it — you can read it
top to bottom.

```
┌─────────────────────────────────────────────────────────────┐
│  User API                                                    │
│  Sequential · Module.parameters() · Dataset / DataLoader     │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────────┐
│  ember.nn  — neural-network building blocks                  │
│  Linear · ReLU · Sigmoid · Tanh · Softmax · MSELoss ·        │
│  CrossEntropyLoss · BCEWithLogitsLoss                        │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────────┐
│  ember.ops  — differentiable operations                     │
│  add · sub · mul · div · matmul · relu · sigmoid · tanh ·    │
│  softmax · exp · log · pow · sum · mean · transpose          │
│  (each wires its reverse-mode backward closure)              │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────────┐
│  ember.Tensor  — autograd engine                            │
│  dynamic graph (_prev), topological backward(), grad accum.  │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────────┐
│  ember._array  — pure-Python math (0-D / 1-D / 2-D)         │
│  matmul · broadcast_ew · sum_axis · softmax · activations    │
│  NO numpy, NO C extensions — standard library only           │
└─────────────────────────────────────────────────────────────┘
```

The forward pass builds the graph; `Tensor.backward()` performs a reverse
topological walk, and each node's stored `_backward` closure pushes its local
gradient into its parents' `.grad`. That is the entire mechanism.

## 🔌 API

**Tensor**

| Method | Description |
| --- | --- |
| `Tensor(data, requires_grad=False)` | Create a scalar / vector / matrix. |
| `a + b`, `a * b`, `a @ b`, `a / b` | Differentiable arithmetic & matrix product. |
| `a.relu()`, `a.sigmoid()`, `a.tanh()`, `a.softmax()` | Activations. |
| `a.exp()`, `a.log()`, `a.sqrt()`, `a.pow(p)` | Math ops. |
| `a.sum()`, `a.mean()`, `a.transpose()` | Reductions & reshape. |
| `loss.backward()` | Run reverse-mode autodiff; fills `.grad`. |
| `t.item()` | Read a scalar tensor as a Python float. |

**Layers & training** (`ember.nn`, `ember.optim`, `ember.data`)

| Symbol | Description |
| --- | --- |
| `nn.Linear(in, out, bias=True)` | Affine layer `y = x·Wᵀ + b`. |
| `nn.ReLU / Sigmoid / Tanh / Softmax` | Activation modules. |
| `nn.Sequential(*layers)` | Compose modules in order. |
| `nn.MSELoss / CrossEntropyLoss / BCEWithLogitsLoss` | Common losses. |
| `SGD(params, lr, momentum, weight_decay, nesterov)` | Optimizer. |
| `TensorDataset(*cols)`, `DataLoader(ds, batch_size, shuffle)` | Batching. |

## 🧭 路线图

- [ ] Automatic broadcasting for N-D tensors (currently 0-D / 1-D / 2-D)
- [ ] Convolution and pooling layers for image models
- [ ] Recurrent cells (RNN / LSTM / GRU)
- [ ] GPU / vectorised backend behind the same `Tensor` interface
- [ ] Model persistence (save / load weights as JSON)
- [ ] A small text / vision dataset loader
- [ ] `torch`-style `no_grad()` context manager

## 🤝 贡献

Contributions are welcome — this is a teaching project, so clarity matters as
much as correctness. Please add unit tests (gradient checks are the gold
standard here) and keep new code dependency-free. Open an issue or a pull
request.

## 📄 许可证

Released under the [MIT License](LICENSE).

---

<br/>

# 手写一个深度学习框架：从零实现自动微分与神经网络

> 这是一篇可以直接发布到掘金 / 知乎 / 小红书 / B 站的技术科普文底稿。下面的内容既可当作 Ember 项目的 README，也可以稍作润色成为一篇独立的入门长文。

## 一、它解决了什么问题

现在几乎每个学 AI 的人，第一步都是 `pip install torch`，然后写
`loss.backward()`。代码能跑，模型能训，但很多人心里都有一个黑洞：**反向传播到底是怎么算梯度的？** 一旦换个损失、改个网络结构，或者面试被问到"为什么梯度是这样传的"，就容易卡壳。

市面上的框架把这件事封装得太好了，好到你看不到它的骨架。Ember 这个项目
（一个纯 Python、零依赖的自动微分 + 神经网络库）要做的，就是把这层壳剥掉，
用几百行可读的代码，把"前向建图、反向求导"这件事原原本本地写出来。它不追求
替代 PyTorch，而是成为**理解 PyTorch 的那块垫脚石**。

更现实一点：对正在准备 2027 年 AI / 后端实习的同学来说，"从零手写一个深度学习
框架"是简历上非常能打的硬通货——它直接证明你懂模型背后的数学，而不是只会调 API。

## 二、核心原理：反向模式自动微分

所有现代框架（PyTorch、JAX、TensorFlow）都建立在同一个核心机制上，分两步：

1. **前向建图**。每一次运算（如 `c = a + b`，`y = x @ W`）都不只是算个数值，
   而是顺手记下"我是怎么来的"——也就是它的直接父节点是谁。这样所有运算串起来，
   形成一棵（或一张）动态计算图。
2. **反向求导**。从最终的输出标量（通常是 loss）出发，沿着图**逆拓扑序**往回走，
   对每一个节点调用它预先存好的"反向规则"，把梯度一层层累加到父节点上。

Ember 里的 `Tensor` 就是这么设计的：每个张量带着 `_prev`（它依赖谁）和
`_backward`（它自己的反向闭包）。当你调用 `loss.backward()` 时，框架先做一遍
拓扑排序，再逆序执行每个节点的 `_backward`，梯度就一路流回了最底层的参数。

这套机制的精髓是**链式法则的本地化**：每个算子只关心"我的输出对输入的局部梯度
是多少"，至于整条链怎么串，交给图去管。这也是为什么 Ember 能支持任意深度的网络。

## 三、关键实现点（面试常问）

**1. 矩阵乘法的反向。** 前向是 `C = A @ B`，反向要算两个梯度：
`∂L/∂A = ∂L/∂C @ Bᵀ`，`∂L/∂B = Aᵀ @ ∂L/∂C`。
Ember 在 `ops.matmul` 里把 (m,k)@(k,n)、(m,k)@(k,)、(k,)@(k,n)、(k,)@(k,)
几种形状分别处理，每种都给出了正确的反向公式。

**2. 广播（broadcasting）的反向是"求和归约"。** 比如给一个 (batch, out) 的矩阵
加上一个 (out,) 的偏置向量，前向是把偏置复制到每一行；反向时，loss 对偏置的
梯度，就是 loss 对矩阵每一行该位置的梯度**加总**——因为偏置被复用了 batch 次。
这正是很多同学手写时最容易漏掉的一点。

**3. 数值稳定的 softmax + 交叉熵。** 把 `softmax` 和 `log` 合在一起算，并在 softmax
里先做 `max` 减均值防溢出；交叉熵的反向直接是 `(p - onehot) / N`，比"先算
softmax 再取 log"在数值上稳得多，也快得多。

**4. 参数的自动发现。** `Module.parameters()` 递归遍历对象里的所有 `Tensor`、
子模块、列表和字典，把需要梯度的权重全部收集起来交给优化器——和 PyTorch 的
`model.parameters()` 一个思路，但实现只有几十行。

**5. SGD 的三种增强。** 动量（momentum）、Nesterov 加速、L2 权重衰减，全部基于
一个梯度缓冲区实现，参数更新公式与主流框架一致。

## 四、与"直接调现成框架"对比

| 维度 | 直接用 PyTorch | 手写 Ember |
| --- | --- | --- |
| 上手速度 | 快，几行就能训模型 | 慢，要先写引擎 |
| 对原理的理解 | 容易停留在"会调用" | 必须吃透反向传播 |
| 依赖 / 部署 | 需要 numpy、torch 等 | 只要标准库，**零依赖** |
| 运行环境 | 需要能装包的环境 | 任何 Python 解释器都能跑 |
| 简历说服力 | 通用、同质化 | "从零实现"非常加分 |
| 生产可用性 | 强 | 弱（教学 / 学习定位） |

一句话：**能用框架的地方当然用框架；但想真正搞懂它，最好的办法是自己写一个。**

## 五、如何扩展这个项目

如果你想把它继续做大，下面这些方向都很有价值，而且每一个都能单独写成一篇技术文：

- 把张量从 0/1/2 维扩到 N 维，支持图像卷积；
- 实现 RNN / LSTM，做序列建模；
- 把底层数学换成向量化的实现（或接 GPU），速度能提升几个数量级，但对外接口不变；
- 加上权重的保存 / 加载（JSON 即可），让训好的模型能复现；
- 写一个 `no_grad()` 上下文，推理时关闭建图、省内存。

## 六、如何把本文扩展成一篇发布稿

- **标题建议**：《我用 600 行纯 Python 手写了 PyTorch 的内核》《面试被问反向传播？
  不如自己写一个自动微分引擎》《零依赖，从零实现一个神经网络框架》。
- **配图建议**：画一张"前向建图 → 反向求导"的示意图（Ember 的 README 里已经有一张
  分层架构图，可直接引用或重绘）；再放一张 XOR 训练 loss 下降曲线、最终 4/4 准确率的
  截图。
- **钩子开头**：从"你真的懂 `loss.backward()` 吗"这种反问切入，比平铺直叙更有点击率。
- **发布平台**：技术深度够，适合掘金 / 知乎；如果想做视频，B 站讲"手写框架"系列非常
  有受众，配合代码逐行讲解效果最好。

> 源码与全部示例均在仓库内，克隆即可运行：`git clone https://github.com/WsNodeX/ember.git`
