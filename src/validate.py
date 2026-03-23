import numpy as np
import json
import torch
from transformers import EsmModel, EsmTokenizer
from pathlib import Path

# 생성된 서열 파싱
def parse_mpnn_fasta(filepath):
    seqs = {}
    current_id = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:]
            elif current_id and not current_id.startswith('2LXZ,'):
                seqs[current_id] = line
    return seqs

generated = parse_mpnn_fasta('outputs/mpnn_results/seqs/2LXZ.fa')
print(f'생성된 서열 수: {len(generated)}')

# ESM2 임베딩
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'디바이스: {device}')

tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
model = EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D')
model = model.to(device)
model.eval()

ids = list(generated.keys())
sequences = list(generated.values())

batch_size = 32
embeddings = []
for i in range(0, len(sequences), batch_size):
    batch = sequences[i:i+batch_size]
    inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())

gen_embeddings = np.vstack(embeddings)
print(f'생성 서열 임베딩 shape: {gen_embeddings.shape}')

# 기존 임베딩 + 클러스터 정보 로드
orig_embeddings = np.load('outputs/embeddings.npy')
orig_labels = np.load('outputs/labels.npy')
embedding_2d = np.load('outputs/embedding_2d.npy')

# UMAP 기존 reducer 재사용 불가 → 전체 합쳐서 새로 UMAP
import umap
all_embeddings = np.vstack([orig_embeddings, gen_embeddings])
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
all_2d = reducer.fit_transform(all_embeddings)

orig_2d = all_2d[:len(orig_embeddings)]
gen_2d = all_2d[len(orig_embeddings):]

# 시각화
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))

# 원본 클러스터
scatter = ax.scatter(orig_2d[:, 0], orig_2d[:, 1],
    c=orig_labels, cmap='tab20', s=5, alpha=0.5, label='Original AMP')

# 생성 서열
ax.scatter(gen_2d[:, 0], gen_2d[:, 1],
    c='red', s=20, alpha=0.8, marker='*', label='Generated', zorder=5)

ax.set_title('Validation — Generated Sequences in Embedding Space')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/validation_umap.png', dpi=150)
print('시각화 저장: outputs/validation_umap.png')

# Cluster 1 centroid와 거리 계산
with open('outputs/centroids.json') as f:
    centroids = json.load(f)

c1 = np.array(centroids['1'])
distances = np.linalg.norm(gen_embeddings - c1, axis=1)
print(f'\nCluster 1 centroid와의 거리:')
print(f'  평균: {distances.mean():.3f}')
print(f'  최소: {distances.min():.3f}')
print(f'  최대: {distances.max():.3f}')

# 원본 Cluster 1 서열과의 거리 비교
c1_mask = orig_labels == 1
c1_orig_distances = np.linalg.norm(orig_embeddings[c1_mask] - c1, axis=1)
print(f'\n원본 Cluster 1 서열과의 거리:')
print(f'  평균: {c1_orig_distances.mean():.3f}')

# 저장
np.save('outputs/gen_embeddings.npy', gen_embeddings)
np.save('outputs/gen_2d.npy', gen_2d)
print('\n완료!')
