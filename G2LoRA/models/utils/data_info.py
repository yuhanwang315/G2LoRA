import torch

data = torch.load('GRAPHCLIP/processed_data/arxiv_2023.pt', map_location='cpu')

print(data.keys())
