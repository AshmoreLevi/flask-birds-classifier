import shutil

import torch
from PIL import Image
from torchvision import transforms
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
from collections import namedtuple
import hashlib
import datetime

app = Flask(__name__)
# restrict the size of the file uploaded
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024


################################################
# Error Handling
################################################

@app.errorhandler(404)
def FUN_404(error):
    return render_template("error.html")


@app.errorhandler(405)
def FUN_405(error):
    return render_template("error.html")


@app.errorhandler(500)
def FUN_500(error):
    return render_template("error.html")


################################################
# Functions for running classifier
################################################

# define a simple data batch
Batch = namedtuple('Batch', ['data'])

# load the models and labels
model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True,
                       **{'topN': 6, 'device': 'cpu', 'num_classes': 200})
model.eval()

# 从label.txt文件将分类标签存到labels中，每个元素为一行标签，如001 dog
with open('classes.txt', 'r') as f:
    labels = [l.rstrip() for l in f]


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array
    """
    nts_net_transform = transforms.Compose([
        transforms.Resize((600, 600), Image.BILINEAR),
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = nts_net_transform(image)
    torch_images = image.unsqueeze(0)

    return torch_images


def get_image(file_location, local=True):
    """
    获取图像文件数据并转换为Tensor输入
    """
    file_name = file_location
    # 图片通道 BGR 转 RGB
    img = process_image(Image.open(file_name).convert('RGB'))

    if img is None:
        return None

    return img


def predict(file_location, local=False):
    """
    用mxnet训练好的.params和.json进行预测
    用预训练好的模型进行预测
    """
    img = get_image(file_location, local)

    # compute the prediction probabilities
    with torch.no_grad():
        top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(img)
        # prob: 全部类别的置信度向量 200
        # index: 排在前5的类别序号向量 5
        # index: tensor([ 12,   9, 110, 125, 121])
        _, index = torch.topk(concat_logits, 5)
        index = index[0]
        prob = torch.softmax(concat_logits[0], 0)

    # result 存放格式:
    # [[dog 0.8]
    #  [cat 0.1]]
    result = []
    for i in index:
        # label[1] : 001 dog
        # prob: 01(index) 0.45643
        result.append((labels[i].split(" ", 1)[1], prob[i]))

    return result


################################################
# Functions for Image Archive 存档
################################################

def FUN_resize_img(filename, resize_proportion=0.3):
    """
    FUN_resize_img() will resize the image passed to it as argument to be {resize_proportion} of the original size.
    """
    img = cv2.imread(filename)
    small_img = cv2.resize(img, (0, 0), fx=resize_proportion, fy=resize_proportion)
    cv2.imwrite(filename, small_img)


################################################
# Functions Building Endpoints
################################################

@app.route("/", methods=['POST', "GET"])
def FUN_root():
    # Run corresponding code when the user provides the image url 当用户提供图像url时，运行相应的代码
    # If user chooses to upload an image instead, endpoint "/upload_image" will be invoked
    if request.method == "POST":
        img_url = request.form.get("img_url")
        prediction_result = predict(img_url)
        print(prediction_result)
        return render_template("index.html", img_src=img_url, prediction_result=prediction_result)
    else:
        return render_template("index.html")


@app.route("/about/")
def FUN_about():
    return render_template("about.html")


# 判断图片是否合法
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp']


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload_image", methods=['POST'])
def FUN_upload_image():
    if request.method == 'POST':
        # check if the post request has the file part 检查post请求是否包含文件部分
        if 'file' not in request.files:
            return redirect(url_for("FUN_root"))
        file = request.files['file']

        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            return redirect(url_for("FUN_root"))

        # 上传了文件，就放在文件夹img_pool中
        if file and allowed_file(file.filename):
            filename = os.path.join("static/img_pool",
                                    # 原代码这里没有.encode("utf-8")，会报Unicode-objects must be encoded before hashing错
                                    hashlib.sha256(str(datetime.datetime.now()).encode("utf-8")).hexdigest() +
                                    secure_filename(file.filename).lower())
            file.save(filename)
            prediction_result = predict(filename, local=True)
            print(prediction_result)
            FUN_resize_img(filename)

            # 每预测到一项，创建该分类第一的文件夹，把图片放入相应的文件夹中
            dir_name = os.path.join("static/img_pool",
                                    prediction_result[0][0].replace(" ", "").split(',', 1)[0])
            if os.path.isdir(dir_name):
                print("dir is existed!")
            else:
                os.mkdir(dir_name)
            end_name = os.path.join(dir_name, secure_filename(file.filename).lower())
            shutil.copyfile(filename, end_name)

            return render_template("index.html", img_src=filename, prediction_result=prediction_result)
    return redirect(url_for("FUN_root"))


################################################
# Start the service
################################################
if __name__ == "__main__":
    app.run(debug=True)
