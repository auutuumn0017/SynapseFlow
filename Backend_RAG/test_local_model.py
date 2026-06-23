from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# 1. 指定相对路径
# 获取当前 test_local_model.py 所在文件夹（Backend_RAG）
base_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接完整模型路径
model_path = os.path.join(base_dir, "models", "DeepSeek-R1-Distill-Qwen-1.5B")

print(f"⏳ 正在从 {model_path} 加载模型...")
print("（如果电脑没有独立显卡，加载和推理可能会稍慢，请耐心等待）")

try:
    # 2. 加载 Tokenizer 和 模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # device_map="auto" 会自动寻找你的显卡，如果没有显卡会自动用 CPU 跑
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, # 使用 float16 兼容绝大多数显卡
        device_map="auto"
    )
    print("✅ 模型加载成功！当前所在设备:", model.device)

    # 3. 构造测试对话
    # 我们故意问一个《统计学习方法》里的问题，看看它原生的回答是什么样
    messages = [
        {"role": "system", "content": "你是一个严谨的机器学习老师。"},
        {"role": "user", "content": "请用一句话解释一下什么是经验风险？"}
    ]

    # 4. 格式化并输入模型
    print("🧠 DeepSeek 正在思考中...")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 5. 生成回答
    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=512, 
        temperature=0.6     
    )

    # 剥离掉输入的 prompt，只提取回答部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n" + "="*50)
    print("🤖 完整回答输出：")
    print("="*50)
    print(response)

except Exception as e:
    print(f"\n❌ 报错了: {e}")