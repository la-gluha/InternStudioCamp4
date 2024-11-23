# OpenCompass 评测书生大模型实践

## 通过API的方式进行评测

### 环境搭建

```bash
conda create -n opencompass python=3.10
conda activate opencompass

cd /root
git clone -b 0.3.3 https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
pip install -r requirements.txt
pip install huggingface_hub==0.25.2
```

```bash
# 设置api key
export INTERNLM_API_KEY=xxxxxxxxxxxxxxxxxxxxxxx
```

申请api key：https://internlm.intern-ai.org.cn/api/document

```bash
# 为api调用编写脚本
cd /root/opencompass/
touch opencompass/configs/models/openai/puyu_api.py
```

```python
import os
from opencompass.models import OpenAISDK


internlm_url = 'https://internlm-chat.intern-ai.org.cn/puyu/api/v1/' # 你前面获得的 api 服务地址
internlm_api_key = os.getenv('INTERNLM_API_KEY')

models = [
    dict(
        # abbr='internlm2.5-latest',
        type=OpenAISDK,
        path='internlm2.5-latest', # 请求服务时的 model name
        # 这里会直接读取之前设置的系统变量
        key=internlm_api_key, # API key
        openai_api_base=internlm_url, # 服务地址
        rpm_verbose=True, # 是否打印请求速率
        query_per_second=0.16, # 服务请求速率
        max_out_len=1024, # 最大输出长度
        max_seq_len=4096, # 最大输入长度
        temperature=0.01, # 生成温度
        batch_size=1, # 批处理大小
        retry=3, # 重试次数
    )
]
```

```bash
# 配置数据集
cd /root/opencompass/
touch opencompass/configs/datasets/demo/demo_cmmlu_chat_gen.py
```

```python
from mmengine import read_base

with read_base():
    from ..cmmlu.cmmlu_gen_c13365 import cmmlu_datasets


# 每个数据集只取前2个样本进行评测
for d in cmmlu_datasets:
    d['abbr'] = 'demo_' + d['abbr']
    d['reader_cfg']['test_range'] = '[0:1]' # 这里每个数据集只取1个样本, 方便快速评测.


```

```bash
# 开始测评
python run.py --models puyu_api.py --datasets demo_cmmlu_chat_gen.py --debug
```

测评结果

![image-20241124021604760](https://github.com/la-gluha/InternStudioCamp4/blob/main/chapter6/img/image-20241124021604760.png)

![image-20241124024336881](https://github.com/la-gluha/InternStudioCamp4/blob/main/chapter6/img/image-20241124024336881.png)



## 错误解决

```
Traceback (most recent call last):
  File "/root/opencompass/run.py", line 1, in <module>
    from opencompass.cli.main import main
  File "/root/opencompass/opencompass/cli/main.py", line 16, in <module>
    from opencompass.utils.run import (fill_eval_cfg, fill_infer_cfg,
  File "/root/opencompass/opencompass/utils/run.py", line 9, in <module>
    from opencompass.datasets.custom import make_custom_dataset_config
  File "/root/opencompass/opencompass/datasets/__init__.py", line 72, in <module>
    from .lveval import *  # noqa: F401, F403
  File "/root/opencompass/opencompass/datasets/lveval/__init__.py", line 1, in <module>
    from .evaluators import LVEvalF1Evaluator  # noqa: F401, F403
  File "/root/opencompass/opencompass/datasets/lveval/evaluators.py", line 12, in <module>
    from rouge import Rouge
ModuleNotFoundError: No module named 'rouge'
```

![image-20241124020939893](https://github.com/la-gluha/InternStudioCamp4/blob/main/chapter6/img/image-20241124020939893.png)

修改 /root/opencompass/opencompass/datasets/lveval/evaluators.py 文件 rouge -> rouge_chinese

![image-20241124021020371](https://github.com/la-gluha/InternStudioCamp4/blob/main/chapter6/img/image-20241124021020371.png)

```bash
# 安装依赖
conda install -c conda-forge rouge
```



## 课程链接

https://github.com/InternLM/Tutorial/tree/camp4/docs/L1/Evaluation