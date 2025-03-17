from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import torch
import os
import glob

def format_conversation(messages):
    """格式化 Qwen 对话数据"""
    formatted = ""
    for msg in messages:
        if msg["role"] == "system":
            formatted += f"[系统] {msg['content']}\n"
        elif msg["role"] == "user":
            formatted += f"[用户] {msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted += f"[助手] {msg['content']}\n"
    return formatted


def find_latest_folder(path):
    # 检查路径是否存在
    if not os.path.exists(path):
        print("指定路径不存在")
        return None
    # 获取指定路径下的所有文件夹
    folders = glob.glob(os.path.join(path, '*'))
    # 过滤出文件夹（排除文件）
    folders = [folder for folder in folders if os.path.isdir(folder)]
    # 如果没有找到文件夹，返回None
    if not folders:
        print("指定路径下没有文件夹")
        return None
    # 找到创建时间最晚的文件夹
    latest_folder = max(folders, key=os.path.getctime)

    return latest_folder

def main(args):
    # **加载模型**
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)


    # **设置 LoRA 参数**
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_token"],
    )
    # **添加 LoRA**
    model = get_peft_model(base_model, lora_config)
    # **打印可训练参数**
    model.print_trainable_parameters()


    # **加载数据**
    dataset = load_dataset("json", data_files=args.dataset_path)
    print(dataset)

    # **数据预处理**
    def preprocess_function(examples):
        """Tokenize 对话数据"""
        texts = [format_conversation(msgs) for msgs in examples["messages"]]
        print(f"Formatted texts: {texts[:5]}")
        return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # **数据 Collator**
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 关闭 MLM，启用自回归训练
    )


    # **训练参数**
    training_args = TrainingArguments(
        output_dir=args.lora_adapter_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=10,
        save_steps=50,
        save_total_limit=50,
        logging_steps=1,
        fp16=False,
        optim="adamw_torch",
        lr_scheduler_type="linear"
    )
    
    # **Trainer 训练**
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")

    # **保存 LoRA 适配器**
    model.save_pretrained(args.lora_adapter_path)
    tokenizer.save_pretrained(args.lora_adapter_path)


    if args.merge: 
        lora_model = PeftModel.from_pretrained(
            base_model,
            find_latest_folder(args.lora_adapter_path),
            torch_dtype=torch.float16,
            config=lora_config
        )
        model = lora_model.merge_and_unload()
        model.save_pretrained(args.mergemodel_path)
        tokenizer.save_pretrained(args.mergemodel_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="finetuning script")
    parser.add_argument(
        "--base_model_path",
        type=str,
        help="Path to the base model to be finetned.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset to be used for finetuning.",
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        help="Path to the folder where the LoRA adapter will be saved.",
    )
    parser.add_argument(
        "--merge",
        help="Whether to merge the model with LoRA.",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--mergemodel_path",
        help="Path to the folder where the merged model will be saved.",
        type=str,
    )
    
    args = parser.parse_args()

    main(args)



