import os
import torch
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ==========================================
# 0. 解析 Neo4j 凭证文件
# ==========================================
def load_neo4j_credentials(file_path):
    """读取 Neo4j Aura 下载的 txt 配置文件"""
    credentials = {}
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到配置文件: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                credentials[key.strip()] = value.strip()
    return credentials

# ==========================================
# 1. 准备工作
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_FILE = os.path.join(base_dir, "Neo4j-fed30e46-Created-2026-06-21.txt")

print("⏳ 正在读取数据库配置文件...")
credentials = load_neo4j_credentials(CREDENTIALS_FILE)

NEO4J_URI = credentials.get("NEO4J_URI")
NEO4J_USER = credentials.get("NEO4J_USERNAME")
NEO4J_PASSWORD = credentials.get("NEO4J_PASSWORD")

# 嵌入模型选择 BAAI/bge-small-zh-v1.5 (轻量、极快，特别适合中文检索)
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

print(f"⏳ 正在加载向量模型: {MODEL_NAME} ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(MODEL_NAME, device=device)
dim_size = embedder.get_sentence_embedding_dimension()
print(f"✅ 模型加载成功，向量维度为: {dim_size}，使用计算设备: {device}")

# ==========================================
# 2. 执行更新
# ==========================================
class EmbeddingUpdater:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def create_vector_index(self):
        """创建 Neo4j 向量索引"""
        print("⏳ 正在检查并创建向量索引...")
        # 针对 Neo4j 5.x 的向量索引创建语法
        query = f"""
        CREATE VECTOR INDEX knowledge_node_embedding IF NOT EXISTS
        FOR (n:KnowledgeNode) ON (n.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dim_size},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        with self.driver.session() as session:
            try:
                session.run(query)
                print("✅ 向量索引创建成功 (或已存在)")
            except Exception as e:
                print(f"⚠️ 创建索引失败: {e}")

    def update_all_embeddings(self):
        """读取所有节点，计算并写入向量"""
        fetch_query = """
        MATCH (n:KnowledgeNode)
        RETURN id(n) AS node_id, n.name AS name, n.definition AS definition
        """
        
        with self.driver.session() as session:
            results = session.run(fetch_query)
            nodes = []
            for record in results:
                nodes.append({
                    "id": record["node_id"],
                    "text": f"实体名：{record['name']}。定义：{record['definition']}"
                })
        
        if not nodes:
            print("⚠️ 未找到任何 KnowledgeNode！")
            return
            
        print(f"⏳ 正在为 {len(nodes)} 个节点计算向量...")
        
        # 批量编码
        texts = [n["text"] for n in nodes]
        # BGE 模型建议中文前加 "为这个句子生成表示以用于检索相关文章：" 但对文档端我们通常不加，只在 query 端加
        embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        
        # 组装数据并批量写入
        update_data = []
        for i, node in enumerate(nodes):
            update_data.append({
                "node_id": node["id"],
                "embedding": [float(x) for x in embeddings[i]]
            })
            
        update_query = """
        UNWIND $batch AS data
        MATCH (n:KnowledgeNode) WHERE id(n) = data.node_id
        SET n.embedding = data.embedding
        """
        
        print("⏳ 正在将向量数据写回 Neo4j...")
        # 分批写入 (这里数据不多，直接一把写入)
        with self.driver.session() as session:
            session.run(update_query, batch=update_data)
            
        print("✅ 全部向量更新完毕！")

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    updater = EmbeddingUpdater(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        updater.create_vector_index()
        updater.update_all_embeddings()
    finally:
        updater.close()
