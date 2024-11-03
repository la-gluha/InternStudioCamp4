# chapter0

## Linux

### ssh连接开发机并运行 hello_world.py

1. 创建开发机
2. 点击开发机的ssh连接，复制其中的命令，在本地运行，并输入密码
3. ssh映射，开发机自定义服务中的命令，或者直接使用vscode进行配置
4. 运行 hello_world.py，并访问本地

![linux ssh 连接](https://github.com/la-gluha/InternStudioCamp4/blob/main/chapter0/img/linux%20ssh%20%E8%BF%9E%E6%8E%A5.png)



## python

### leetcode

1. 分别统计两个字符串中每个字符出现的次数
2. 使用ransomNote中的字符统计减去magazine中的字符，获得ransomNote比magazine中多出的字符
3. 如果多出的字符为0，证明magazine可以表示randomNote

![leetcode](https://github.com/la-gluha/InternStudioCamp4/blob/main/chapter0/img/leetcode.png)



### debug

1. 直接运行，出现如下报错，根据报错信息以及response的内容，可以确认报错的原因是json解析出现了问题

   ![debug1](https://github.com/la-gluha/InternStudioCamp4/blob/main/chapter0/img/debug1.png)

2. 观察response的内容，其中多了如下字符，导致整体不符合json的格式：`json`、```

3. 在将字符串解析成json之前，去除多余的字符

   ![debug2](https://github.com/la-gluha/InternStudioCamp4/blob/main/chapter0/img/debug2.png)



## git

MR链接：https://github.com/InternLM/Tutorial/pull/2335

笔记仓库链接：https://github.com/la-gluha/InternStudioCamp4



## huggingface

![hf](https://github.com/la-gluha/InternStudioCamp4/blob/main/chapter0/img/hf.png)