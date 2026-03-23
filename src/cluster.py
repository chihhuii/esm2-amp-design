import numpy as np
import json
import umap
import hdbscan
import matplotlib.pyplot as plt
from pathlib import Path

embeddings = np.load('outputs/embeddings.npy')
with open('outputs/ids.json') as f:
    ids = json.load(f)

print(f'임베딩 shape: {embeddings.shape}')

reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
embedding_2d = reducer.fit_transform(embeddings)

clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10)
labels = clusterer.fit_predict(embedding_2d)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f'클러스터 수: {n_clusters}')
print(f'노이즈 포인트: {(labels == -1).sum()}')

print('\n=== 클러스터별 서열 샘플 ===')
for label in sorted(set(labels)):
    if label == -1:
        continue
    mask = np.where(labels == label)[0]
    print(f'\nCluster {label} ({len(mask)}개):')
    for idx in mask[:3]:
        print(f'  {ids[idx]}')

np.save('outputs/embedding_2d.npy', embedding_2d)
np.save('outputs/labels.npy', labels)

centroids = {}
for label in set(labels):
    if label == -1:
        continue
    mask = labels == label
    centroids[int(label)] = embeddings[mask].mean(axis=0).tolist()

with open('outputs/centroids.json', 'w') as f:
    json.dump(centroids, f)

print('\n완료!')

# 서열도 같이 출력
print('\n=== 클러스터별 서열 내용 ===')
seqs = {}
from pathlib import Path
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

for fasta_file in Path('data').glob('*.txt'):
    seqs.update(parse_fasta(fasta_file))

for label in sorted(set(labels)):
    if label == -1 or label == 0:  # 0은 너무 많아서 스킵
        continue
    mask = np.where(labels == label)[0]
    print(f'\nCluster {label} ({len(mask)}개):')
    for idx in mask[:3]:
        pid = ids[idx]
        print(f'  {pid}: {seqs.get(pid, "?")}')
