


# DeepSeek-R1 全流程微调指南

- [快速开始](#L13)
- [环境配置](#L21)
- [微调流程](#L49)
- [模型量化](#模型量化)



## 🚀 快速开始
本代码包以微调deepseekr1-1.5b为例
```bash
git clone https://github.com/vergileZhang/deepseek_finetuning.git
cd deepseek_finetuning
```


## 🔧 环境配置
### 原始依赖安装
```bash
pip install -r requirements.txt
```

### llama.cpp编译（原始命令）
llama.cpp 是一个专注于在边缘设备、个人PC上进行llm部署的高性能推理框架。
```bash
git clone https://github.com/ggerganov/llama.cpp 
cd llama.cpp
```
运行下行无报错，以确保您的CUDA Toolkits和CMake设置正确

```bash
nvcc -version
cmake --version
```
采用CUDA加速编译
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config
```
配置llama.cpp环境
```bash
pip install -r requirements.txt
```

## 🛠️ 微调流程
### 1. 数据预处理
```bash
# 保持原始处理流程不变
# 数据集格式要求：
# - 500条法律问答
# - JSON格式
# - 包含prompt/response字段
```

### 2. 执行微调
```bash
python train.py \
--base_model_path /path/to/your/base_model \
--dataset_path /path/to/your/dataset \
--lora_adapter_path /path/to/save/LoRA_adapter \
--model_path /path/to/save/finetuned_model
```

#### 参数说明
| 参数 | 原始说明 |
|------|----------|
| `--base_model_path` | 原始存放包含`.safetensors`文件的模型目录 |
| `--dataset_path` | 微调数据集存放路径 | 
| `--lora_adapter_path` | 微调产生的低秩矩阵参数，而非完整的模型权重 |
| `--model_path` | 微调后的完整参数保存路径 |


## 🗜️ 模型量化
### 转GGUF
原始模型权重为`.safetensors`文件，须转`.GGUF`后才可供OLLAMA部署
```bash
cd llama.cpp
python convert_hf_to_gguf.py /path/to/save/finetuned_model --outfile /path/to/save/GGUF
```

### 量化
```bash
./llama-quantize /path/to/save/GGUF /path/to/save/your_model-***.gguf ***
```
得到文件`your_model-***.gguf`，其中，`***`为需要您指定的量化程度，有下列多个标准可以选择。选择不同的量化，模型的推理效果不一样

 - `q2_K`
 - `q3_K`
 - `q3_K_S`
 - `q3_K_M`
 - `q3_K_L`
 - `q4_0`
 - `q4_1`
 - `q4_K`
 - `q4_K_S`
 - `q4_K_M`
 - `q5_0`
 - `q5_1`
 - `q5_K`
 - `q5_K_S`
 - `q5_K_M`
 - `q6_K`
 - `q8_0`
 - `f16`


## 📦 OLLAMA部署
### Modelfile
创建Modelfile文件，内容如下：
```dockerfile
FROM /path/to/save/quantized_GGUF
    
TEMPLATE """
{{ if .System }}<|system|>
{{ .System }}</s>{{ end }}
    
<|user|>
{{ .Prompt }}</s>
    
<|assistant|>
{{ .Response }}
"""
    
SYSTEM "You are a helpful assistant."
PARAMETER temperature 0.7
PARAMETER repeat_penalty 1.2
```

### 启动命令
```bash
ollama serve
ollama create your_model_name -f path/to/Modelfile
ollama run your_model_name
```




## 📚 参考文档
> Blockquote
> 原始参考文献和引用说明


