import json
import os
import pandas as pd
import glob

# 固定Data根目录
data_root = r"D:\autumn\CS_Experiment\数据分析平台实践\SynapseFlow\Data"
# 拼接json文件夹完整路径
json_pattern = os.path.join(data_root, "data_json", "*.json")
json_files = glob.glob(json_pattern)

all_nodes = []
all_relationships = []

for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('```json'):
            content = content[7:-3] 
            
        data = json.loads(content)
        all_nodes.extend(data.get('nodes', []))
        all_relationships.extend(data.get('relationships', []))

df_nodes = pd.DataFrame(all_nodes)
df_rels = pd.DataFrame(all_relationships)

# 输出到Data文件夹下
df_nodes.to_csv(os.path.join(data_root, 'final_nodes.csv'), index=False, encoding='utf-8-sig')
df_rels.to_csv(os.path.join(data_root, 'final_relationships.csv'), index=False, encoding='utf-8-sig')

print(f"✅ 处理完成！共提取 {len(df_nodes)} 个节点，{len(df_rels)} 条关系。")
print(f"文件已保存至目录：{data_root}")