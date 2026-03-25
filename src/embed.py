import torch
from transformers import EsmModel, EsmTokenizer
from pathlib import Path
import numpy as np
import json

def parse_fasta(filepath):
    sequences = {}
    current_id = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:].split()[0]
            elif current_id:
                sequences[current_id] = line
    return sequences

# 데이터 로드, load the data
data_dir = Path('data')
seqs = {}
for fasta_file in data_dir.glob('*.txt'):
    seqs.update(parse_fasta(fasta_file))

ids = list(seqs.keys())
sequences = list(seqs.values())
print(f'총 서열 수: {len(sequences)}')

# 모델 로드, load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 디바이스: {device}')

tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
model = EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D')
model = model.to(device)
model.eval()

# 배치 임베딩, batch embedding
batch_size = 32
embeddings = []

for i in range(0, len(sequences), batch_size):
    batch_seqs = sequences[i:i+batch_size]
    inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # mean pooling
    batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    embeddings.append(batch_emb)
    
    if (i // batch_size) % 10 == 0:
        print(f'진행: {i+len(batch_seqs)}/{len(sequences)}')

embeddings = np.vstack(embeddings)

# 저장, save
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)
np.save(output_dir / 'embeddings.npy', embeddings)
with open(output_dir / 'ids.json', 'w') as f:
    json.dump(ids, f)

print(f'완료! 임베딩 shape: {embeddings.shape}')
print(f'저장 위치: outputs/embeddings.npy')
