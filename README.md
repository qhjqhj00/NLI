# 天池新冠疫情相似句对判定大赛之佛跳墙

修改./src/run.sh中bert模型的路径
bert模型下载地址：http://36.112.85.6:4501/bert-base-chinese.tar.gz
```shell
cd src
./run.sh
```

| 数据预处理          | 模型                       | Dev Acc. |
| ------------------- | -------------------------- | -------- |
| 疾病 药品换成占位符 | Bert - Transformer encoder | 92.75    |
|                     |                            |          |
|                     |                            |          |

