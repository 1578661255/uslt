import os
import pickle
import json
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

desc_dir = '../../description/CSL-Daily/split_data/'  # 根据实际路径调整
output_path = './desc_bert_features.pkl'

# GPU 设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32  # GPU 批处理大小
print(f"[GPU信息] 使用设备: {device}")
if torch.cuda.is_available():
    print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("  警告：未检测到GPU，将使用CPU（速度较慢）")

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
print("\n[编码器] 加载BERT-base-chinese...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese').to(device)
model.eval()
print(f"[编码器] BERT已加载到 {device}")

# 3. 批量编码并保存
print("\n[编码] 开始编码描述文本...")
desc2feat = {}

# 分批处理
for batch_start in tqdm(range(0, len(all_desc), batch_size), desc="编码进度"):
    batch_end = min(batch_start + batch_size, len(all_desc))
    batch_descs = all_desc[batch_start:batch_end]
    
    with torch.no_grad():
        # 批量 tokenize
        inputs = tokenizer(batch_descs, return_tensors='pt', padding=True, truncation=True, max_length=32)
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        
        # 模型前向传播
        outputs = model(**inputs)
        
        # 提取[CLS]特征
        batch_feats = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
        
        # 存储特征
        for i, desc in enumerate(batch_descs):
            desc2feat[desc] = batch_feats[i].cpu()
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 保存到 pickle 文件
print("\n[保存] 将特征保存到文件...")
with open(output_path, 'wb') as f:
    pickle.dump(desc2feat, f)

# 统计信息
print(f"\n[完成] 编码完成！")
print(f"  保存路径: {os.path.abspath(output_path)}")
print(f"  特征字典条数: {len(desc2feat)}")
print(f"  文件大小: {os.path.getsize(output_path) / 1e6:.2f} MB")
print(f"  特征维度: 768")

if torch.cuda.is_available():
    print(f"\n[GPU] 最终显存占用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"      最大显存占用: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
