import os
import pickle
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
        continue
    for fname in os.listdir(phase_dir):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(phase_dir, fname), 'r', encoding='utf-8') as f:
            import json
            data = json.load(f)
            for desc in data.get('descriptions', []):
                if desc is not None:
                    all_desc.add(desc)

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

with open(output_path, 'wb'):
    pickle.dump(desc2feat, open(output_path, 'wb'))
print(f"已保存到 {output_path}")
