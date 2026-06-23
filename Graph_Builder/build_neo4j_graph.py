import pandas as pd
from neo4j import GraphDatabase
import os

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
            # 忽略空行和以 # 开头的注释行
            if line and not line.startswith('#'):
                # 按第一个 '=' 分割键值对
                key, value = line.split('=', 1)
                credentials[key.strip()] = value.strip()
    return credentials

# ==========================================
# 1. 配置信息 (自动读取)
# ==========================================
# 数据文件路径
DATA_ROOT = r"D:\autumn\CS_Experiment\数据分析平台实践\SynapseFlow\Data"
NODES_CSV = os.path.join(DATA_ROOT, "final_nodes.csv")
RELS_CSV = os.path.join(DATA_ROOT, "final_relationships.csv")

# 凭证文件路径 (使用原始字符串 r 避免转义问题)
CREDENTIALS_FILE = r"D:\autumn\CS_Experiment\数据分析平台实践\SynapseFlow\Graph_Builder\Neo4j-fed30e46-Created-2026-06-21.txt"

print("⏳ 正在读取数据库配置文件...")
credentials = load_neo4j_credentials(CREDENTIALS_FILE)

# 从字典中提取所需信息
NEO4J_URI = credentials.get("NEO4J_URI")
NEO4J_USER = credentials.get("NEO4J_USERNAME")
NEO4J_PASSWORD = credentials.get("NEO4J_PASSWORD")

# ==========================================
# 2. 定义 Neo4j 操作类
# ==========================================
class KnowledgeGraphBuilder:
    def __init__(self, uri, user, password):
        # 建立连接
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        """清空数据库"""
        query = "MATCH (n) DETACH DELETE n"
        with self.driver.session() as session:
            session.run(query)
            print("🧹 已清空图数据库中的旧数据。")

    def create_nodes(self, df_nodes):
        """批量创建节点"""
        query = """
        UNWIND $node_list AS node
        MERGE (n:KnowledgeNode {name: node.Name})
        SET n.type = node.Type,
            n.definition = node.Definition,
            n.formula = node.Formula,
            n.teaching_note = node.Teaching_Note,
            n.source = node.Source
        """
        node_list = df_nodes.fillna("无").to_dict('records')
        
        with self.driver.session() as session:
            session.run(query, node_list=node_list)
            print(f"✅ 成功导入 {len(node_list)} 个节点！")

    def create_relationships(self, df_rels):
        """逐条创建关系"""
        rel_list = df_rels.fillna("无").to_dict('records')
        success_count = 0
        
        with self.driver.session() as session:
            for rel in rel_list:
                head = rel['Head']
                tail = rel['Tail']
                relation_type = rel['Relation'].upper().replace(" ", "_")
                context = rel['Context']
                
                query = f"""
                MATCH (h:KnowledgeNode {{name: $head}})
                MATCH (t:KnowledgeNode {{name: $tail}})
                MERGE (h)-[r:`{relation_type}`]->(t)
                SET r.context = $context
                """
                try:
                    session.run(query, head=head, tail=tail, context=context)
                    success_count += 1
                except Exception as e:
                    print(f"⚠️ 创建关系失败 [{head} -> {tail}]: {e}")
                    
        print(f"✅ 成功导入 {success_count} 条关系！")

# ==========================================
# 3. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    print("⏳ 开始读取本地 CSV 数据...")
    df_n = pd.read_csv(NODES_CSV, encoding='utf-8-sig')
    df_r = pd.read_csv(RELS_CSV, encoding='utf-8-sig')
    
    print(f"⏳ 正在连接云端 Neo4j 数据库 ({NEO4J_URI})...")
    builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        builder.clear_database()
        builder.create_nodes(df_n)
        builder.create_relationships(df_r)
        
        print("🎉 全部构建完成！请前往 https://console.neo4j.io 打开你的 Aura 控制台查看知识图谱。")
    finally:
        builder.close()