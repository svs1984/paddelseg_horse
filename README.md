# [AI训练营]PaddleSeg实现语义分割Baseline_paddelseg_horse

手把手教你基于PaddleSeg实现语义分割任务

------


# 一、作业任务

> 本次任务将基于PaddleSeg展开语义分割任务的学习与实践，baseline会提供PaddleSeg套件的基本使用，相关细节如有遗漏，可参考[10分钟上手PaddleSeg
](https://aistudio.baidu.com/aistudio/projectdetail/1672610?channelType=0&channel=0)

1. 选择提供的**五个数据集**中的一个数据集作为本次作业的训练数据，并**根据baseline跑通项目**

2. **可视化1-3张**预测图片与预测结果（本次提供的数据集没有验证与测试集，可将训练数据直接进行预测，展示训练数据的预测结果即可）


**加分项:**

3. 尝试**划分验证集**，再进行训练

4. 选择**其他网络**，调整训练参数进行更好的尝试


**PS**:PaddleSeg相关参考项目:

- [常规赛：PALM病理性近视病灶检测与分割基线方案](https://aistudio.baidu.com/aistudio/projectdetail/1941312)

- [PaddleSeg 2.0动态图：车道线图像分割任务简介](https://aistudio.baidu.com/aistudio/projectdetail/1752986?channelType=0&channel=0)

------

------

# 二、数据集说明

------

本项目使用的数据集是:[AI训练营]语义分割数据集合集，包含马分割，眼底血管分割，车道线分割，场景分割以及人像分割。

该数据集已加载到本环境中，位于:

**data/data103787/segDataset.zip**



```python
# unzip: 解压指令
# -o: 表示解压后进行输出
# -q: 表示静默模式，即不输出解压过程的日志
# -d: 解压到指定目录下，如果该文件夹不存在，会自动创建
!unzip -oq data/data103787/segDataset.zip -d segDataset
```

解压完成后，会在左侧文件目录多出一个**segDataset**的文件夹，该文件夹下有**5**个子文件夹：

- **horse -- 马分割数据**<二分类任务>

![](https://ai-studio-static-online.cdn.bcebos.com/2b12a7fab9ee409587a2aec332a70ba2bce0fcc4a10345a4aa38941db65e8d02)

- **fundusVessels -- 眼底血管分割数据**

> 灰度图，每个像素值对应相应的类别 -- 因此label不宜观察，但符合套件训练需要

![](https://ai-studio-static-online.cdn.bcebos.com/b515662defe548bdaa517b879722059ad53b5d87dd82441c8c4611124f6fdad0)

- **laneline -- 车道线分割数据**

![](https://ai-studio-static-online.cdn.bcebos.com/2aeccfe514e24cf98459df7c36421cddf78d9ddfc2cf41ffa4aafc10b13c8802)

- **facade -- 场景分割数据**

![](https://ai-studio-static-online.cdn.bcebos.com/57752d86fc5c4a10a3e4b91ae05a3e38d57d174419be4afeba22eb75b699112c)

- **cocome -- 人像分割数据**

> label非直接的图片，为json格式的标注文件，有需要的小伙伴可以看一看PaddleSeg的[PaddleSeg实战——人像分割](https://aistudio.baidu.com/aistudio/projectdetail/2177440?channelType=0&channel=0)


```python
# tree: 查看文件夹树状结构
# -L: 表示遍历深度
!tree segDataset -L 2
```

> 查看数据label的像素分布，可从中得知分割任务的类别数： 脚本位于: **show_segDataset_label_cls_id.py**

> 关于人像分割数据分析，这里不做提示，可参考[PaddleSeg实战——人像分割](https://aistudio.baidu.com/aistudio/projectdetail/2177440?channelType=0&channel=0)


```python
# 查看label中像素分类情况
!python show_segDataset_label_cls_id.py
```

# 三、数据预处理

> 这里就以horse数据作为示例

-----

- 首先，通过上边的像素值分析以及horse本身的标签表现，我们确定horse数据集为二分类任务

- 然而，实际label中，却包含多个像素值，因此需要将horse中的所有label进行一个预处理

- 预处理内容为: 0值不变，非0值变为1，然后再保存label

- **并且保存文件格式为png，单通道图片为Label图片，最好保存为png——否则可能出现异常像素**

**对应horse的预处理脚本，位于:**

parse_horse_label.py


```python
!python parse_horse_label.py
```

- 预处理完成后，配置训练的索引文件txt，方便后边套件读取数据

> txt创建脚本位于: **horse_create_train_list.py**

> 同时，生成的txt位于: **segDataset/horse/train_list.txt**


```python
# 创建训练的数据索引txt
# 格式如下
# line1: train_img1.jpg train_label1.png
# line2: train_img2.jpg train_label2.png
!python horse_create_train_list.py
```

# 四、使用套件开始训练

- 1.解压套件: 已挂载到本项目, 位于:**data/data102250/PaddleSeg-release-2.1.zip**


```python
# 解压套件
!unzip -oq work/data102250/PaddleSeg-release-2.1.zip
# 通过mv指令实现文件夹重命名
!mv PaddleSeg-release-2.1 PaddleSeg
```

- 2.选择模型，baseline选择**bisenet**, 位于: **PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml**


```python
!cp -r work/config/quick_start/bisenet_optic_disc_512x512_1k.yml PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml
```

- 3.配置模型文件

> 首先修改训练数据集加载的dataset类型:

![](https://ai-studio-static-online.cdn.bcebos.com/2f5363d71034490290f720ea8bb0d6873d7df2712d4b4e84ae41b0378aed8b89)

> 然后配置训练数据集如下:

![](https://ai-studio-static-online.cdn.bcebos.com/29547856db4b4bfc80aa3732e143f2788589f9316c694f369c9bd1da44b815dc)

> 类似的，配置验证数据集: -- **注意修改train_path为val_path**

![](https://ai-studio-static-online.cdn.bcebos.com/09713aaaed6b4611a525d25aae67e4f0538224f7ac0241eb941d97892bf6c4c1)

<font color="red" size=4>其它模型可能需要到: PaddleSeg/configs/$_base_$  中的数据集文件进行配置，但修改参数与bisenet中的数据集修改参数相同 </font>

![](https://ai-studio-static-online.cdn.bcebos.com/b154dcbf15e14f43aa13455c0ceeaaebe0489c9a09dd439f9d32e8b0a31355ec)


- 4.开始训练

使用PaddleSeg的train.py，传入模型文件即可开启训练


```python
!python PaddleSeg/train.py\
--config PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml\
--batch_size 4\
--iters 2000\
--learning_rate 0.01\
--save_interval 200\
--save_dir work/output\
--seed 2021\
--log_iters 20\
--do_eval\
--use_vdl

# --batch_size 4\  # 批大小
# --iters 2000\    # 迭代次数 -- 根据数据大小，批大小估计迭代次数
# --learning_rate 0.01\ # 学习率
# --save_interval 200\ # 保存周期 -- 迭代次数计算周期
# --save_dir PaddleSeg/output\ # 输出路径
# --seed 2021\ # 训练中使用到的随机数种子
# --log_iters 20\ # 日志频率 -- 迭代次数计算周期
# --do_eval\ # 边训练边验证
# --use_vdl # 使用vdl可视化记录
# 用于断训==即中断后继续上一次状态进行训练
# --resume_model model_dir
```


```python
# 单独进行评估 -- 上边do_eval就是这个工作
!python PaddleSeg/val.py\
--config PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml\
--model_path PaddleSeg/output/best_model/model.pdparams
# model_path： 模型参数路径
```

- 5.开始预测


```python
# 进行预测
!python PaddleSeg/predict.py\
--config PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml\
--model_path work/output/best_model/model.pdparams\
--image_path segDataset/horse/Images\
--save_dir PaddleSeg/output/horse
# image_path: 预测图片路径/文件夹 -- 这里直接对训练数据进行预测，得到预测结果
# save_dir： 保存预测结果的路径 -- 保存的预测结果为图片
```

    2021-08-15 15:30:11 [INFO]	Number of predict images = 328
    2021-08-15 15:30:11 [INFO]	Loading pretrained model from work/output/best_model/model.pdparams
    2021-08-15 15:30:11 [INFO]	There are 356/356 variables loaded into BiSeNetV2.
    2021-08-15 15:30:11 [INFO]	Start to predict...
    328/328 [==============================] - 17s 50ms/st


# 五、可视化预测结果

通过PaddleSeg预测输出的结果为图片，对应位于:PaddleSeg/output/horse

其中包含两种结果：

- 一种为掩膜图像，即叠加预测结果与原始结果的图像 -- 位于: **PaddleSeg/output/horse/added_prediction**
- 另一种为预测结果的伪彩色图像，即预测的结果图像 -- 位于: **PaddleSeg/output/horse/pseudo_color_prediction**


```python
# 查看预测结果文件夹分布
!tree PaddleSeg/output/horse -L 1
```

分别展示两个文件夹中的预测结果(下载每个预测结果文件夹中的一两张图片到本地，然后上传notebook)

上传说明:

![](https://ai-studio-static-online.cdn.bcebos.com/29cb48e1263a4ea49557a8564f289be1690e1b23dab7412388420f9c244f366c)

> 以下为展示结果

<font color='red' size=5> ---数据集 horse 的预测结果展示--- </font>

<font color='black' size=5> 掩膜图像： </font>

![](https://ai-studio-static-online.cdn.bcebos.com/58938dc8c40f4d3885b5e18eb8f5b49c78d55eb004dd4f4f9a894e47d55d923d)


<font color='black' size=5> 伪彩色图像： </font>

![](https://ai-studio-static-online.cdn.bcebos.com/bf89b47ea3ee44bcb19b39bb5203a2a52b653b7294304c7d944892011683b192)


<font color='red' size=5>特别声明，使用horse数据集进行提交时，预测结果展示不允许使用horse242.jpg和horse242.png的预测结果，否则将可能认定为未进行本baseline作业的训练、预测过程 </font>

# 六、提交作业流程

1. 生成项目版本

![](https://ai-studio-static-online.cdn.bcebos.com/1c19ac6cfb314353b5377421c74bc5d777dcb5724fad47c1a096df758198f625)

2. (有能力的同学可以多捣鼓一下)根据需要可将notebook转为markdown，自行提交到github上

![](https://ai-studio-static-online.cdn.bcebos.com/0363ab3eb0da4242844cc8b918d38588bb17af73c2e14ecd92831b89db8e1c46)

3. (一定要先生成项目版本哦)公开项目

![](https://ai-studio-static-online.cdn.bcebos.com/8a6a2352f11c4f3e967bdd061f881efc5106827c55c445eabb060b882bf6c827)

# 七、寄语

<font size=4>

最后祝愿大家都能完成本次作业，圆满结业，拿到属于自己独一无二的结业证书——这一次训练营，将是你们AI之路成长的开始！

希望未来的你们能够坚持在自己热爱的方向上继续自己的梦想！

同时也期待你们能够在社区中创造更多有创意有价值基于飞桨系列的AI项目，发扬开源精神！

<br>

最后，再次祝愿大家都能顺利结业！
 </font>
