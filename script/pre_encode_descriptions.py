import os
import pickle
import json
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

desc_dir = '../../description/CSL-Daily/split_data/'  # 根据实际路径调整
output_path = './desc_bert_features.pkl'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 收集所有描述文本
all_desc = set()
for phase in ['train', 'dev', 'test']:
    phase_dir = os.path.join(desc_dir, phase)
    if not os.path.exists(phase_dir):
        print(f"[警告] {phase} 目录不存在: {phase_dir}")
        continue
    
    print(f"[开始] 扫描 {phase} 数据...")
    for fname in os.listdir(phase_dir):
        if not fname.endswith('.json'):
            continue
        try:
            with open(os.path.join(phase_dir, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 自动处理三种格式
                descriptions = []
                if isinstance(data, list):
                    # 格式1: 列表的字典 [{"filename": "xxx.jpg", "description": "desc"}, ...]
                    for item in data:
                        if isinstance(item, dict) and 'description' in item:
                            descriptions.append(item['description'])
                        elif isinstance(item, str):
                            # 格式1b: 直接字符串列表 ["desc1", "desc2", ...]
                            descriptions.append(item)
                elif isinstance(data, dict):
                    # 格式2：字典 {"descriptions": [...]}
                    descriptions = data.get('descriptions', [])
                else:
                    print(f"[警告] 未知格式 {fname}: {type(data)}")
                    continue
                
                # 收集描述
                for desc in descriptions:
                    if desc is not None and isinstance(desc, str):
                        desc_text = desc.strip()
                        if desc_text:  # 跳过空字符串
                            all_desc.add(desc_text)
        except Exception as e:
            print(f"[错误] 读取 {fname} 失败: {e}")

all_desc = list(all_desc)
print(f"共收集到唯一描述文本 {len(all_desc)} 条")

# 2. 加载BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese').to(device)
model.eval()

# 3. 编码并保存
desc2feat = {}
with torch.no_grad():
    for desc in tqdm(all_desc):
        inputs = tokenizer(desc, return_tensors='pt', truncation=True, max_length=32)
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        outputs = model(**inputs)
        # 取[CLS]特征
        feat = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        desc2feat[desc] = feat

# 保存到 pickle 文件
with open(output_path, 'wb') as f:
    pickle.dump(desc2feat, f)
print(f"已保存到 {output_path}")
print(f"特征字典包含 {len(desc2feat)} 条描述")
