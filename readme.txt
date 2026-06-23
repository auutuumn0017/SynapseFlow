SynapseFlow A Knowledge-Graph Augmented Agent for Interactive Machine Learning Education
SynapseFlow：面向交互式机器学习教学的知识图谱增强智能体
数据集地址1：https://github.com/MLNLP-World/MachineLearning2025Spring--Notes
数据集地址2：https://github.com/SmirkCao/Lihang

Step1：
先把markdown文件用deepseek网页版变成json格式（手动粘贴过去），然后用代码直接把17章的json变为excel
prompt：
你现在是一位顶尖的知识图谱数据工程师。我将交给你一项长期任务：从李航《统计学习方法》的 17 章笔记中提取图谱数据。我将分 17 次把文本发给你。

### 📌 提取规则一：【节点】提取（6元组）
提取核心实体为 6 个属性：
1. "Name": 实体标准名称。
2. "Type": 选一 [Concept, Math, Evaluation, Task, Algorithm, Model, Strategy]。
3. "Definition": 严谨定义。
4. "Formula": LaTeX公式（需保留原始符号，若无填"无"）。
5. "Teaching_Note": 讲师笔记/吐槽（若无填"无"）。
6. "Source": 来源章节（如：CH01-策略）。

### 📌 提取规则二：【关系】提取（4元组）
提取实体间逻辑关系：
1. "Head": 起点实体。
2. "Relation": 选一 [COMPRISE, EQUIVALENT_TO, PREREQUISITE, COMPARE, APPLIED_TO, OPTIMIZED_BY]。
3. "Tail": 终点实体。
4. "Context": 关系成立的条件（若无填"无"）。

### 📤 输出格式要求（极其重要！）：
不要输出任何 Markdown 表格！请必须严格按照以下 JSON 格式输出，确保 JSON 语法合法（注意逗号和转义字符）。请将其放在一个 ```json 语法块中返回：

{
  "nodes": [
    {
      "Name": "交叉验证",
      "Type": "Evaluation",
      "Definition": "一种常用的模型选择方法...",
      "Formula": "无",
      "Teaching_Note": "注意验证集和测试集的区别...",
      "Source": "CH01-正则化"
    }
  ],
  "relationships": [
    {
      "Head": "极大似然估计",
      "Relation": "EQUIVALENT_TO",
      "Tail": "经验风险最小化",
      "Context": "当模型是条件概率分布且损失函数是对数损失时"
    }
  ]
}

如果你理解了，请只回复：“规则已掌握，准备接收 JSON 数据抽取。请发送第 1 章！”



非常棒。请继续按照严格的规则，提取以下第 n 章的文本内容。注意：如果遇到和前面章节相同的概念，请保持 Name (实体名称) 的一致性



结果是✅ 处理完成！共提取 207 个节点，278 条关系。确实比较小



Step2：
构建知识图谱
使用“云端免费版”（Neo4j AuraDB）
MATCH (n)-[r]->(m) RETURN n, r, m看到全部关系



Step3：
部署deepseek，完成RAG后端
4-bit 极限压缩模型，使得回答更快
然后创建 API 文件，有后端接口了，之后就做数字人就可以
运行main_api.py
http://127.0.0.1:8000/docs#/



Step4：
cd Frontend
npm install
npm run dev



Step5:use
使用方式：
先启动 RAG：
cd Backend_RAG
python main_api.py
再启动 LiveTalking：
cd LiveTalking
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1 --rag_api_url http://127.0.0.1:8000/api/chat
打开：
http://127.0.0.1:8010/index.html
先点击连接 WebRTC，再在 “RAG 问答” 里提问，RAG 回答后数字人会读出来。