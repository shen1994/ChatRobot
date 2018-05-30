# ChatRobot

## 0. 特别提醒  
> * 关于keras环境下seq2seq错误修改  
  ('_OptionalInputPlaceHolder' object has no attribute 'inbound_nodes')  
> * 0.0 使用keras2.1.0版本的第三方库(不推荐)  
> * 0.1 将recurrentshop\engine.py文件中837和842行中inbound_nodes更改为_inbound_nodes  

## 1. 效果展示  
### 1.0 `python train.py`执行效果图  
![image](https://github.com/shen1994/README/raw/master/images/ChatRobot_train.jpg)  
### 1.1 `python test.py`执行效果图  
![image](https://github.com/shen1994/README/raw/master/images/ChatRobot_predict.jpg)  
### 1.2 `python chat_robot.py`执行效果图  
![image](https://github.com/shen1994/README/raw/master/images/ChatRobot_chatchat.jpg)  

## 2. 软件安装  
> * 模型搭建第三方库Keras-2.1.6.tar.gz  
    私人地址: 链接: <https://pan.baidu.com/s/1ypoEgf6ITjcNalzTRtnvmw> 密码: uot8  
> * recurrentshop下载地址: <https://github.com/farizrahman4u/recurrentshop>  
> * seq2seq 下载地址: <https://github.com/farizrahman4u/seq2seq>  
> * 微博数据(关于餐饮业,数据未清洗)下载地址  
    私人地址: 链接: <https://pan.baidu.com/s/1g6l4_IDkLdLAjvrWf5sheQ> 密码: fxy3  

## 3. 参考链接  
* seq2seq讲解: <http://jacoxu.com/encoder_decoder>  
* seq2seq数据读取: <http://suriyadeepan.github.io/2016-06-28-easy-seq2seq>  
* seq2seq论文地址: <https://arxiv.org/abs/1409.3215>  
* seq2seq+attention论文地址: <https://arxiv.org/abs/1409.0473>  
* ChatRobot启发论文: <https://arxiv.org/abs/1503.02364>  
* seq2seq源码: <https://github.com/farizrahman4u/seq2seq>  
* seq2seq源码需求: <https://github.com/farizrahman4u/recurrentshop>  
* beamsearch源码参考: <https://github.com/yanwii/seq2seq>  
* bucket源码参考: <https://github.com/1228337123/tensorflow-seq2seq-chatbot-zh>  

## 4. 执行命令  
> * 生成序列文件,将文字编码为数字,不足补零  
`python data_process.py`  
> * 生成word2vec向量,包括编码向量和解码向量  
`python word2vec.py`  
> * 训练网络  
`python train.py`  
> * 测试  
`python test.py`  
> * 模型评分  
`python score.py`  
> * 智能问答  
`python chat_robot.py`  
> * 绘制word2vec向量分布图  
`python word2vec_plot.py`  

## 5. 更新  
> * Word2cut模型对陌生词汇的分词未解决,有时间搞定一下
