# 🧠 SynapseFlow

> **SynapseFlow**：面向交互式机器学习教学的**微调对齐与知识图谱双重增强**数字人智能体。
> 结合了本地量化小模型大语言模型推理、**基于 Unsloth 的 QLoRA 授课人格注入**、Neo4j 多跳混合图谱检索（Hybrid GraphRAG）以及极简的 WebRTC 实时流式数字人交互。

---

## ✨ 核心特性 (Features)

- 🕸️ **学术级知识图谱**：采用大模型信息抽取技术（LLM-IE），从李航《统计学习方法》核心章节中高精度提取了包含公式（LaTeX）与讲师笔记（Teaching Note）的垂直领域教学本体图谱。
- 🔍 **混合子图检索 (Hybrid GraphRAG)**：摒弃了传统的文本匹配，采用高维稠密向量（Dense Vector）进行余弦相似度检索，并创新性地向外执行 1-Hop 拓扑扩展，拉取前置知识与对比概念，大幅缓解模型幻觉。
- 🎓 **授课人格注入 (Persona Injection)**：针对大模型“机器味”过重的问题，专属构建了包含启发式提问与通俗比喻的 SFT 数据集，使用 `Unsloth` 在 RTX 3050 等极小显存环境下通过 `QLoRA` 进行了指令微调，赋予了数字人生动的“名师灵魂”。
- 🚀 **大模型端侧极限部署**：本地集成 15 亿参数级别的 `DeepSeek-R1-Distill-Qwen-1.5B`。基于 `bitsandbytes` 实现了 NF4 量化，将大模型显存峰值极致压缩至 2GB 以下。
- 🎨 **原生物理图谱渲染 & 毛玻璃 UI**：基于开源数字人底层架构进行深度重构。采用纯 Vanilla JS 开发了力导向图引擎（Force-Directed Graph），实时可视化 RAG 推理的“大脑状态”；全站采用 Apple-Style Glassmorphism 设计规范。

## 🏗️ 系统架构 (Architecture)

系统采用微服务前后端分离设计，包含四大核心模块：
1. **Graph Builder (知识提炼层)**: 解析 Markdown，生成稠密向量，经由 `neo4j` 原生驱动写入 AuraDB 云端数据库。
2. **Fine-Tuning (后训练微调层)**: 提取知识图谱语料构造 SFT 问答对，通过 Unsloth 进行高效 QLoRA 微调，实现风格对齐与授课人格注入。
3. **Backend_RAG (检索推理层)**: FastAPI 构建的后端，执行 Cypher 多跳查询，结合大语言模型 CoT 生成极具教学口吻的回答（支持动态挂载微调后的 LoRA 贴片）。
4. **LiveTalking Web (流式表现层)**: 基于 WebRTC 实现毫秒级音视频串流，与后端无缝打通进行语音生成 (TTS) 与唇形同步渲染。

## 📦 快速开始 (Getting Started)

### 1. 环境准备 (Prerequisites)
- 操作系统：Windows / Linux
- Python 版本：Python 3.10+
- 模型环境：NVIDIA GPU (显存 >= 6GB)
- 外部依赖：
  - [Neo4j AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/) 云端数据库账户
  - 阿里云 DashScope API Key (用于 Qwen-TTS)

### 2. 下载与安装

```bash
# 克隆仓库
git clone https://github.com/YourUsername/SynapseFlow.git
cd SynapseFlow

# 创建并激活 Conda 虚拟环境
conda create -n livetalking python=3.10
conda activate livetalking

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置密钥环境 (非常重要)
为了保护隐私，代码中不包含硬编码的 API 密钥。在运行前，请**务必在你的终端中配置环境变量**：

**Windows (PowerShell):**
```powershell
# 1. 设置阿里云语音合成 API Key
$env:DASHSCOPE_API_KEY="sk-你的真实密钥"

# 2. 解决终端特殊字符（Emoji）编码崩溃问题
$env:PYTHONIOENCODING="utf-8"
```

**Linux/Mac (Bash):**
```bash
export DASHSCOPE_API_KEY="sk-你的真实密钥"
```

*注意：同时你需要确认 `Graph_Builder/Neo4j-*.txt` 包含你正确的 Neo4j 云数据库连接配置（由于该文件包含密码，已在 `.gitignore` 中忽略，请自行创建）。*

### 4. 运行服务

**Step 1: 启动 RAG 与大模型后端接口**
```bash
cd Backend_RAG
python main_api.py
```
*API 启动后，可通过 `http://127.0.0.1:8000/docs` 查看 Swagger 接口文档。*

**Step 2: 启动 LiveTalking 数字人流媒体控制台**
新开一个终端窗口（别忘了同样激活环境和设置环境变量）：
```bash
cd LiveTalking
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1 --rag_api_url http://127.0.0.1:8000/api/chat
```

**Step 3: 访问可视化交互前端**
使用 Chrome 或 Edge 浏览器访问：
[http://127.0.0.1:8010/index.html](http://127.0.0.1:8010/index.html)
点击 **"连接"**，即可开启与大模型数字人讲师的实时语音对谈与图谱交互！

---

## 🛠️ 项目结构

```text
SynapseFlow/
├── Backend_RAG/           # 大模型推理与 GraphRAG 混合检索 API
├── Fine_tuning/           # 基于 Unsloth 的 QLoRA 极速微调脚本与 SFT 实验数据
├── Graph_Builder/         # 知识图谱向量化与 Neo4j 灌库脚本
├── LiveTalking/           # WebRTC 流媒体后端与前端 UI
│   ├── web/               # Vanilla JS + Glassmorphism 前端页面
│   └── tts/               # 文本转语音驱动模块
├── Data/                  # 原始 Markdown 笔记与 JSON 结构化抽取数据
└── ...
```

## 🤝 致谢 (Acknowledgments)
- 本项目底层数字人流媒体分发技术参考了开源库 [LiveTalking]。
- 知识图谱原始语料归功于对李航博士经典教材《统计学习方法》的开源笔记。
- 感谢大模型基座 [DeepSeek] 提供的逻辑思辨与长文本数据抽取支持。
