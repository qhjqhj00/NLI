# Base Images
FROM nvidia/cuda:10.0-base-ubuntu16.04
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:latest-py3

RUN pip install -i https://mirrors.aliyun.com/pypi/simple boto3==1.12.20 
RUN sudo apt-get update
RUN sudo apt-get install gcc -y
RUN pip install -i https://mirrors.aliyun.com/pypi/simple pydict-cedar

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /src/

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
