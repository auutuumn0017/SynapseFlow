import os
import torch

try:
    import torch.distributed.fsdp
    if not hasattr(torch.distributed.fsdp, "FSDPModule"):
        # 凭空捏造一个假的 FSDPModule 塞进 PyTorch 里骗过 TRL 的检查
        torch.distributed.fsdp.FSDPModule = type("FSDPModule", (object,), {})
except ImportError:
    pass

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

def main():
    # ==========================================
    # 1. 配置项
    # ==========================================
    # 指向你本地的模型路径 (注意根据实际情况修改)
    MODEL_PATH = r"D:\autumn\CS_Experiment\DataAnalysis\SynapseFlow\Backend_RAG\models\DeepSeek-R1-Distill-Qwen-1.5B"
    DATA_PATH = r"data\sft_dataset.json"
    OUTPUT_DIR = "lora_model"

    max_seq_length = 2048 # 支持任意长度，Unsloth 会自动做 RoPE 缩放
    dtype = None          # None 会自动判断是 fp16 还是 bf16
    load_in_4bit = True   # 在 3050 上必须开启 4bit，否则显存不够

    # ==========================================
    # 2. 加载模型与 Tokenizer
    # ==========================================
    print("⏳ 正在通过 Unsloth 加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        trust_remote_code = True,
    )

    # ==========================================
    # 3. 挂载 LoRA 适配器 (注入训练参数)
    # ==========================================
    # 这一步是让模型主体冻结，只训练旁边挂载的一小部分参数
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # 秩大小，建议 8, 16, 32
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"], # 覆盖所有线性层
        lora_alpha = 16,
        lora_dropout = 0, # Dropout = 0 能加速训练
        bias = "none",    # 不训练 bias，加速
        use_gradient_checkpointing = "unsloth", # 极限优化显存的关键！
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # ==========================================
    # 4. 数据处理 (对齐你后端 RAG 的 Prompt)
    # ==========================================
    # 💡 修改点：将格式化函数放在 main() 内部，确保它能访问到上面定义好的 tokenizer
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        
        # 我们按照 main_api.py 中的推理格式来构建训练数据，确保训练和推理完全一致！
        for inst, inp, out in zip(instructions, inputs, outputs):
            system_prompt = "你是一个资深的机器学习讲师。请务必根据参考知识来回答学生的问题，用通俗口语化的讲课口吻。"
            user_msg = f"{inst}\n\n【参考知识】:\n{inp}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": out}
            ]
            
            # 使用模型自带的 chat_template 进行拼接，它会自动加上特定的 Token (如 <|im_start|>)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            
        return { "text" : texts }

    print("⏳ 正在加载并格式化数据集...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # ==========================================
    # 5. 训练器配置
    # ==========================================
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 2,
            packing = True, # 开启打包以解决 padding_free 的报错
            per_device_train_batch_size = 1, 
            gradient_accumulation_steps = 4, 
            warmup_steps = 5,
            num_train_epochs = 3, 
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit", 
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            save_strategy = "no", 
        ),
    )
    # ==========================================
    # 6. 开始训练与保存
    # ==========================================
    print("🚀 开始进行微调训练！")
    trainer_stats = trainer.train()

    print(f"✅ 训练完成！正在将 LoRA 权重保存到 {OUTPUT_DIR} ...")
    model.save_pretrained(OUTPUT_DIR) # 只会保存微调后极小的 LoRA 权重
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("🎉 全部搞定！你可以在 lora_model 文件夹里看到 adapter_model.safetensors 了。")

# 💡 最核心的修改：加上这段 Windows 多进程保护壳
if __name__ == "__main__":
    # Windows 推荐加上这一句，进一步保证子进程安全
    from multiprocessing import freeze_support
    freeze_support() 
    main()