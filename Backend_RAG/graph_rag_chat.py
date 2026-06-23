import os
import torch
# 🌟 新增了 BitsAndBytesConfig 和 SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig 
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# ==========================================
# 0. 解析 Neo4j 凭证文件函数
# ==========================================
def load_neo4j_credentials(file_path):
    """读取 Neo4j Aura 下载的 txt 配置文件"""
    credentials = {}
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到配置文件: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 忽略空行和以 # 开头的注释行
            if line and not line.startswith('#'):
                # 按第一个 '=' 分割键值对
                key, value = line.split('=', 1)
                credentials[key.strip()] = value.strip()
    return credentials

# ==========================================
# 1. 配置信息 (自动读取)
# ==========================================
# 凭证文件路径 (使用相对路径)
base_dir = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_FILE = os.path.join(base_dir, "..", "Graph_Builder", "Neo4j-fed30e46-Created-2026-06-21.txt")

print("⏳ 正在读取数据库配置文件...")
credentials = load_neo4j_credentials(CREDENTIALS_FILE)

# 从字典中提取所需信息
NEO4J_URI = credentials.get("NEO4J_URI")
NEO4J_USER = credentials.get("NEO4J_USERNAME")
NEO4J_PASSWORD = credentials.get("NEO4J_PASSWORD")

# 模型路径保持不变
base_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(base_dir, "models", "DeepSeek-R1-Distill-Qwen-1.5B")

# ==========================================
# 2. GraphRAG 核心引擎类
# ==========================================
class GraphRAGEngine:
    def __init__(self):
        print(f"⏳ 正在连接云端 Neo4j 图数据库 ({NEO4J_URI})...")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        print("⏳ 正在以 4-bit 量化加载 DeepSeek-R1 模型到显卡 (显存占用将大幅降低)...")
        
        # 🌟 核心修改点：配置 4-bit 压缩参数
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # 🌟 加载向量模型 (用于混合检索)
        print("⏳ 正在加载向量模型 (BAAI/bge-small-zh-v1.5)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=device)
        
        # 添加了 local_files_only=True，防止意外联网报错
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            quantization_config=quantization_config,  # 🌟 启用压缩
            device_map="auto",
            local_files_only=True
        )
        print("✅ GraphRAG 引擎初始化完毕！\n")

    def search_neo4j(self, query):
            """混合检索 (Hybrid Search) + 多跳子图检索：先通过向量找最相似节点，再拉取一层邻居"""
            # 1. 向量化用户问题
            query_vector = [float(x) for x in self.embedder.encode(query)]
            
            # 2. 向量检索 + 邻居遍历
            cypher = """
            CALL db.index.vector.queryNodes('knowledge_node_embedding', 2, $query_vector)
            YIELD node AS n, score
            OPTIONAL MATCH (n)-[r]-(m:KnowledgeNode)
            RETURN n.name AS name, n.definition AS definition, n.teaching_note AS teaching_note, score,
                   collect({rel_type: type(r), rel_context: r.context, neighbor_name: m.name, neighbor_def: m.definition}) AS neighbors
            """
            knowledge_context = ""
            with self.driver.session() as session:
                results = session.run(cypher, query_vector=query_vector)
                for record in results:
                    name = record["name"]
                    score = record["score"]
                    defn = record["definition"]
                    note = record["teaching_note"]
                    neighbors = record["neighbors"]
                    
                    # 过滤掉相似度太低的节点（可调）
                    if score < 0.6:
                        continue
                        
                    context_block = f"【核心知识点：{name} (相似度: {score:.2f})】\n定义：{defn}\n讲师补充：{note}\n"
                    
                    # 过滤掉不存在的空关联（当 OPTIONAL MATCH 没命中时）
                    valid_neighbors = [nb for nb in neighbors if nb.get('neighbor_name') is not None]
                    if valid_neighbors:
                        context_block += "关联知识点：\n"
                        for nb in valid_neighbors:
                            rel_type = nb.get('rel_type', '无')
                            rel_ctx = nb.get('rel_context', '无')
                            nb_name = nb.get('neighbor_name', '未知')
                            nb_def = nb.get('neighbor_def', '无')
                            
                            context_block += f"  - [{nb_name}] (关系: {rel_type}, 背景: {rel_ctx})。定义: {nb_def}\n"
                            
                    knowledge_context += context_block + "\n"
                    
            return knowledge_context

    def generate_answer(self, query, context):
        """调用本地 DeepSeek 生成回答"""
        # 优化 1：强化 System Prompt，严格限制输出格式
        system_prompt = (
            "你是一个资深的机器学习讲师。请务必根据以下【参考知识】来回答学生的问题。\n"
            "如果你在参考知识中看到了'讲师补充'的内容，请用通俗口语化的讲课口吻把它融入到你的回答中。\n"
            "【格式极度严格要求】：\n"
            "1. 绝对不要使用 Markdown 的标题符号（#）和列表加粗符号（*）。\n"
            "2. 全文必须像真人说话一样，用纯文本段落输出。\n"
            "3. 如果有数学公式，请正常保留 LaTeX 格式。\n\n"
            f"【参考知识】:\n{context if context else '无相关内部知识，请尽力依靠常识回答。'}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 优化 2：把最大输出 token 调大，防止话没说完被强行打断
        generated_ids = self.model.generate(
            **model_inputs, 
            max_new_tokens=1024,  # 从 512 提升到 1024
            temperature=0.6,
            repetition_penalty=1.1 # 稍微加一点惩罚，防止小模型像复读机一样重复
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = response.replace("**", "").replace("### ", "")
        return response

    def close(self):
        self.driver.close()

# ==========================================
# 3. 互动终端
# ==========================================
if __name__ == "__main__":
    engine = GraphRAGEngine()
    print("="*50)
    print("🎓 专属数字教师已上线！(输入 'quit' 退出)")
    print("你可以试着问它：《统计学习方法》里怎么解释经验风险？或者是验证集和测试集有什么区别？")
    print("="*50)
    
    try:
        while True:
            user_input = input("\n👨‍🎓 你的问题: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            print("🔍 正在检索图谱...")
            context = engine.search_neo4j(user_input)
            if context:
                print(f"💡 (图谱命中知识):\n{context.strip()}")
            else:
                print("💡 (未在图谱中找到相关实体，纯大模型回答)")
                
            print("🧠 教师正在思考...")
            answer = engine.generate_answer(user_input, context)
            
            # 分离思考过程和最终答案 (美化输出)
            if "</think>" in answer:
                think_part, final_answer = answer.split("</think>", 1)
                print(f"\n💭 思考过程:{think_part.replace('<think>', '').strip()}")
                print(f"\n🗣️ 回答:\n{final_answer.strip()}")
            else:
                print(f"\n🗣️ 回答:\n{answer.strip()}")
                
    finally:
        engine.close()
        print("\n👋 引擎已关闭。")