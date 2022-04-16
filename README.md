# Image Classifier Written with MXNet(Pytorch) + Flask

<p align="center">
<img src="https://raw.githubusercontent.com/XD-DENG/flask-app-for-mxnet-img-classifier/master/static/img/screenshot.png" alt="Drawing" style="width:40%;"/>
</p>

## Structure
```
flask-web-app
│  app.py(app后端)
│  classes.txt(分类labels)
│  README.md
│  requirements.txt
│  
├─static(前端css、js、已上传图片库，已打包好不用改)
│  ├─css
│  │      bootstrap.min.united.css 
│  ├─img_pool(图片库)
│  │  │     bobolink_0019_10552.jpg
│  │  │  
│  │  └─Bobolink(具体的某种分类)
│  │          bobolink_0008_9289.jpg
│  │          
│  └─js
│          bootstrap.min.js
│          jquery.min.js
│          
├─templates(前端html，可以发挥一下)
│      about.html
│      error.html
│      index.html
│      layout.html
```
## Deployment

### Step - 1: Clone This Project

先安装PyCharm与Anaconda
在Pycharm中打开该项目，用Anaconda给项目创建虚拟环境

### Step - 2: Environment

```
conda create -n flask-birds python=3.6
conda activate flask-birds
conda install --yes --file requirements.txt
```

### Step - 3: Download Pre-Trained Model

### Step - 4: Start Service

```
python app.py
```
