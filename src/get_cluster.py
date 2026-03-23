import numpy as np
import json
from pathlib import Path

# 로드
with open('outputs/ids.json') as f:
    ids = json.load(f)
labels = np.load('outputs/labels.npy')

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

seqs = {}
for fasta_file in Path('data').glob('*.txt'):
    seqs.update(parse_fasta(fasta_file))

# Cluster 1 서열 추출
mask = np.where(labels == 1)[0]
cluster1_seqs = {ids[i]: seqs[ids[i]] for i in mask if ids[i] in seqs}

print(f'Cluster 1 서열 수: {len(cluster1_seqs)}')
print('\n전체 서열:')
for pid, seq in cluster1_seqs.items():
    print(f'{pid} ({len(seq)}aa): {seq}')

# FASTA로 저장
with open('outputs/cluster1.fasta', 'w') as f:
    for pid, seq in cluster1_seqs.items():
        f.write(f'>{pid}\n{seq}\n')

print('\noutputs/cluster1.fasta 저장 완료')
