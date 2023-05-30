# test_step

之前的模型只有training_step，可以说是最基本的

现在加上test_step，使之可以测试

```
def test_step(self, batch, batch_idx):
	x, y = batch
	x = x.view(x.size(0), -1)
	z = self.encoder(x)
	x_hat = self.decoder(z)
	test_loss = F.mse_loss(x, x_hat)
	self.log('test_loss', test_loss)
```

和之前的training_step简直一模一样，唯一不同的就是loss并没有返回，而是使用了self.log存了起来

看样子self.log是自带的

test需要在fit结束后，使用trainer.test(model, testdata)手动调用

# validation_step

如法炮制，一模一样的，最后的log中换成val_loss

trainer构建时通过传入check_val_every_n_epoch参数，可以规定几次训练后进行一次val检验，默认是1，即每次训练后都检验一次

val_check_interval是float型的参数，表示单次训练多次检验的情况，设为0.5则表示单次训练中要检验两次

# 模型恢复和保存

```
>>> from lightning.pytorch import Trainer
>>> from lightning.pytorch.callbacks import ModelCheckpoint

# saves checkpoints to 'my/path/' at every epoch
>>> checkpoint_callback = ModelCheckpoint(dirpath='my/path/')
>>> trainer = Trainer(callbacks=[checkpoint_callback])

# save epoch and val_loss in name
# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
>>> checkpoint_callback = ModelCheckpoint(
...     monitor='val_loss',
...     dirpath='my/path/',
...     filename='sample-mnist-{epoch:02d}-{val_loss:.2f}'
... )

# save epoch and val_loss in name, but specify the formatting yourself (e.g. to avoid problems with Tensorboard
# or Neptune, due to the presence of characters like '=' or '/')
# saves a file like: my/path/sample-mnist-epoch02-val_loss0.32.ckpt
>>> checkpoint_callback = ModelCheckpoint(
...     monitor='val/loss',
...     dirpath='my/path/',
...     filename='sample-mnist-epoch{epoch:02d}-val_loss{val/loss:.2f}',
...     auto_insert_metric_name=False
... )

# retrieve the best checkpoint after training
checkpoint_callback = ModelCheckpoint(dirpath='my/path/')
trainer = Trainer(callbacks=[checkpoint_callback])
model = ...
trainer.fit(model)
checkpoint_callback.best_model_path
```

lightning会自动报错最近训练epoch的模型到当前的工作空间(os.getcwd())，也可以在定义trainer的时候指定trainer = Trainer(default_root_dir='xxx')

关闭自动保存的参数是checkpoint_callback=False

自动保存时，可以自定义要监控的量，例如下面这段代码

```python

from pytorch_lightning.callbacks import ModelCheckpoint

class LitAutoEncoder(pl.LightningModule):
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)

        # 1. 计算需要监控的量
        loss = F.cross_entropy(y_hat, y)

        # 2. 使用log()函数标记该要监控的量,名字叫'val_loss'
        self.log('val_loss', loss)

# 3. 初始化`ModelCheckpoint`回调，并设置要监控的量
checkpoint_callback = ModelCheckpoint(monitor='val_loss')

# 4. 将该callback 放到其他的callback 的list中
trainer = Trainer(callbacks=[checkpoint_callback])
```

这里用到的ModelCheckpoint函数原型如下

```
CLASS pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint(filepath=None, monitor=None, verbose=False, save_last=None, save_top_k=None, save_weights_only=False, mode='auto', period=1, prefix='', dirpath=None, filename=None)

```

比较重要的参数有monitor，是需要监控的量，这个参数是string类型，它会通过你写入到log中的标记，比如log('val_loss', loss)时，能检测到val\_loss，默认这个参数为None，此时无脑保存最后一个epoch的模型

sava\_top\_k表示保存最好的k个模型

mode很重要，它决定了监控值是越大越好还是越小越好，比如我们监控的是一个loss，一般是越小越好，就取min，若是准确率等值则取max，auto则是自动(不靠谱)

获取最好的模型：

```
Model.load_from_checkpoint(checkpoint_callback.best_model_path)
```
