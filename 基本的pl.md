由于pytorch在机器学习领域的强势地位，使得很多机器学习的研究和机器学习的工程都使用pytorch来构建自己的项目

但pytorch在设计上有一些不方便之处，如**半精度训练、BatchNorm参数同步、单机多卡训练等**

于是有了pytorch_lightning，pytorch_lightning的设计初衷是：把更多的时间放在研究上，把更少的时间放在工程上

基本的pytorch_lightning使用主要分为四步

1.写一个model，继承自pl.LightningModule

2.写一个dataset，继承自pl.LightningDataModule

3.实例化一个trainer，trainer不需要自己写，给出需要的参数去实例化就可以了

4.然后训练，用trainer.fit方法就可以了

下面会用一个encoder-decoder模型在MNIST上训练的例子来讲一个最基本的pl模型是怎么搭建的

# model

```
class LitAutoModule(pl.LightningModule):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(x.shape[0], -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = F.mse_loss(x, x_hat)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr = 1e-3)
```

class model(pl.LightningModule)

必须要实现的方法是：\_\_init\_\_，training\_step和configure\_optimizers

本篇是最最基础的一篇，所以只讲这三个方法，其它方法会放在后面

## \_\_init\_\_

初始化方法，看到这里你肯定有疑惑——model的初始化需要做什么

一般来说，pl的模型需要先行构建pytorch的nn.model模型，之后把它封装进pl的模型中

### nn.model模型

```
class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
  
  def forward(self, x):
    return self.l1(x)

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
  
  def forward(self, x):
    return self.l1(x)
```

我也没有学习过nn.model模型是怎么搭建的，nn是torch下的一个库，我猜测，实现一个模型只需要实现\_\_init\_\_和forward就行了

也挺好搭建的，可以去找一下nn.Sequential()这个函数，把里面传入各种层，得到一个网络

#### \_\_init\_\_

super().\_\_init\_\_()

self.l1 = nn.Sequential(

    nn.Linear(输入的尺寸，输出的尺寸——其实我也不是很懂这个Linear，只知道是线性层)

    nn.ReLU()

    ……需要什么就自己搜吧，线性层，卷积层，激活层nn里面都有

)

例子：

```
def__init__(self):

    super().__init__()

    self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
```

#### forward

forward是需要一个输入的，其原型是forward(self, x)，x就是输入

代表模型的前向传播（其实我理解的就是输出，模型有输入就有输出嘛）

也很好写，return self.l1(x)就行了

写完了nn.module模型之后，接下来就是把它装进我们的大model里了

## training\_step

参数分别是self，batch和batch\_idx

### batch和batch\_idx

首先需要说一下传统的dataset和dataloader是怎么构建的

#### Dataset

dataset本质上还是一个x和一个y这两个tensor组成，它继承自torch.utils.data.Dataset

需要重写三个方法：

\_\_init\_\_，初始化

\_\_gititem\_\_，获取数据，给一个索引得到一对x和y

\_\_len\_\_，可以使用len()得到长度

#### Dataloader

专门用来迭代输出Dataset中的内容，我将其理解为采样——从Dataset中取一部分输出

一般形式为dataloader = Dataloader(……)

得到的dataloader是可迭代的，可以用for x, y in dataloader来取出其中的元素

目前主要用到的参数就是dataset，shuffle(是否打乱顺序)，和batch_size(一般为128)

说回batch和batch_idx，其实就是一个batch的数据(或者就理解为dataloader)，加上这一batch在dataset中的索引(batch_idx)

一个epoch中的数据量 = batch_size \* batch_idx

那么train_step中的step也就明了了——就是训练一步，训练一个batch

那么这里的batch是啥？其实就是一对x和y，他们都是一个tensor，并且第一维永远是batch_size

#### Transforms

from torchvision import transforms

如果要处理图像，可能需要在dataloader加载后使用transform来进行转换(把图像转成tensor)

因为不做图像处理所以先不展开了

### loss

loss是一个tensor，具体形式视loss函数和数据的维度而定

pytorch.nn.functional中有lossfunction和其它function

## configure_optimizers

返回一个优化器，一般用adam

```
optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

optim中的optimizers都可以像这样使用，这里的self.parameters()让我有些疑惑，它是怎么知道模型中有哪些参数的呢

我估计就是对成员变量进行了遍历

# Dataset

```
class MNISTDataModule(pl.LightningDataModule):
  def __init__(self, data_dir=r'./', batch_size=128):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

  def prepare_data(self):
    MNIST(self.data_dir, train=True, download=True)
    MNIST(self.data_dir, train=False, download=True)

  def setup(self, stage:str):
    if stage == 'fit':
      mnist_full = MNIST(self.data_dir, train=True, download=False, transform=self.transform)
      self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
    if stage == 'test':
      self.mnist_test = MNIST(self.data_dir, train=False, download=False, transform=self.transform)
    if stage == 'predict':
      self.mnist_predict = MNIST(self.data_dir, train=False, download=False, transform=self.transform)

  def train_dataloader(self):
    return DataLoader(self.mnist_train, self.batch_size)
  
  def val_dataloader(self):
    return DataLoader(self.mnist_val, self.batch_size)
  
  def test_dataloader(self):
    return DataLoader(self.mnist_test, self.batch_size)
  
  def predict_dataloader(self):
    return DataLoader(self.mnist_predict, self.batch_size)
```

有点长，不是吗

实现一个pl的dataset要继承pl.LightningDataModule

主要是一个\_\_init\_\_，一个setup，一个prepare_data和train_dataloader，val_dataloader，test_dataloader和predict_dataloader

## \_\_init\_\_

dataset的init其实就是设定一些基础参数供后面使用就好了，常见的参数有data_dir（数据保存的位置），batch_size之类的，这里除了这俩还实现了一个transform

transform一般是用在那种图片数据集的，是用来转换数据格式的，比如把一张图转成一个tensor，具体做什么我也没细查，反正它不是这里的重点

## prepare_data

到这里才是下载数据或者用其它方式得到数据，MNIST是torchvision.Datasets里面，self.data_dir是它存放的位置，train是False还是True代表了从不同的地方下载数据集，其实我也没搞懂为什么Train的数据集是一个地方，非Train的数据集在另外一个地方

prepare_data很重要，会被自动调用的，只管写不用我们手动调用，在需要加载数据集的时候会自动调

## setup

setup指定了不同的标签下返回不同的数据集，这里的标签有fit，test，predict，至于标签在哪，有哪些，这些我也不清楚

接下来就是各种dataloader了，在train的时候，会根据不同的阶段调用相应的函数
