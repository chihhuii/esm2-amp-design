from pathlib import Path

def parse_fasta(filepath):
    sequences = {}
    current_id = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:].split()[0]  # DBAASPR_11 같은 ID만
            elif current_id:
                sequences[current_id] = line
    return sequences

# 두 파일 합치기
data_dir = Path('data')
seqs = {}
for fasta_file in data_dir.glob('*.txt'):
    seqs.update(parse_fasta(fasta_file))

print(f'총 서열 수: {len(seqs)}')
print(f'예시 3개:')
for id_, seq in list(seqs.items())[:3]:
    print(f'  {id_}: {seq[:30]}...')
