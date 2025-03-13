# 写在前面
本文档旨在梳理微调deepseekr1的全流程，包括：

 - 微调数据集的预处理
 - 微调及参数设置
 - 参数转GGUF及量化
 - OLLAMA部署


# 快速开始

    git clone https://github.com/vergileZhang/deepseek_finetuning.git
    cd deepseek_finetuning
    
## 0 环境准备
### 安装依赖库
    pip install -r requirements.txt
### llama.cpp的安装和配置
llama.cpp 是一个专注于在边缘设备、个人PC上进行llm部署的高性能推理框架。

    git clone https://github.com/ggerganov/llama.cpp 
    cd llama.cpp
运行下行无报错，以确保您的CUDA Toolkits和CMake设置正确

    nvcc -version
    cmake --version
采用CUDA加速编译

    cmake -B build -DGGML_CUDA=ON
    cmake --build build --config

配置llama.cpp环境

    pip install -r requirements.txt



## 1 微调数据集预处理

## 2 微调及参数设置

    python train.py \
    --base_model_path /path/to/your/base_model \
    --dataset_path /path/to/your/dataset \
    --lora_adapter_path /path/to/save/LoRA_adapter \
    --model_path /path/to/save/finetuned_model

其中，base_model_path是您从huggingface等平台下载的.safetensors模型文件夹；dataset_path路径存放您的微调数据集；lora_adapter_path用于存放微调产生的低秩矩阵参数，而非完整的模型权重；model_path用于存放微调后的完整参数。


## 3 转GGUF及量化
原始模型权重为.safetensors文件，须转.GGUF后才可供OLLAMA部署。

    cd llama.cpp
    python convert_hf_to_gguf.py /path/to/save/finetuned_model --outfile /path/to/save/GGUF
进一步，我们需要将模型量化

    ./llama-quantize /path/to/save/GGUF /path/to/save/quantized_GGUF q4_0



## 4 OLLAMA部署
