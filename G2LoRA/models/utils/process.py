import torch
import os.path as osp
import json
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader,DataListLoader
import torch_geometric.transforms as T
import numpy as np
from ..utils.args import Arguments


def parse_source_data(name, data):
    config = Arguments().parse_args()
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    json_data = []

    with open(f'summary/summary-{name}.json', 'r', encoding='utf-8') as fcc_file: # subgraph-summary pair
    # with open(f'GRAPHCLIP/summary/summary-{name}.json', 'r', encoding='utf-8') as fcc_file: # subgraph-summary pair
        fcc_data = json.load(fcc_file)
        json_data = fcc_data

    collected_graph_data = []
    # collected_text_data = []
    for id, jd in enumerate(json_data):
        assert id == jd['id']
        edges = torch.tensor(jd['graph'])
        summary = jd['summary']
        # reindex
        node_idx = torch.unique(edges)
        node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
        sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
        target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
        edge_index = torch.IntTensor([sources_idx, target_idx]).long()
        graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], root_n_index=node_idx_map[jd['id']], summary=summary)
        graph=transform(graph) # add PE
        collected_graph_data.append(graph)
    return collected_graph_data

def parse_finetune_data(name, data):
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    json_data = []

    with open(f'/home/lyh/GRAPHCLIP/target_data/attack/photo.json', 'r', encoding='utf-8') as fcc_file: # subgraph-summary pair
        fcc_data = json.load(fcc_file)
        json_data = fcc_data

    collected_graph_data = []
    # collected_text_data = []
    for id, jd in enumerate(json_data):
        assert id == jd['id']
        edges = torch.tensor(jd['graph'])
        summary = jd['summary']
        # reindex
        node_idx = torch.unique(edges)
        node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
        sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
        target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
        edge_index = torch.IntTensor([sources_idx, target_idx]).long()
        graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], root_n_index=node_idx_map[jd['id']], summary=summary)
        graph=transform(graph) # add PE
        collected_graph_data.append(graph)
    return collected_graph_data

def parse_target_data(name, data):
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    collected_graph_data = []

    file_path_json = f'target_data/{name}.json'

    with open(file_path_json, 'r') as fcc_file:
        json_data = json.load(fcc_file)

    for id, jd in enumerate(json_data):
        assert id == jd['id']
        edges = torch.tensor(jd['graph'])
        if edges.shape[1] == 0:
            edges = torch.tensor([[id], [id]])
        node_idx = torch.unique(edges)
        node_idx_map = {j: i for i, j in enumerate(node_idx.numpy().tolist())}
        sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
        target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
        edge_index = torch.IntTensor([sources_idx, target_idx]).long()
        graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], root_n_index=node_idx_map[jd['id']])
        graph = transform(graph)  # add PE
        collected_graph_data.append(graph)
        # print(collected_graph_data)

    return collected_graph_data


def create_masks(data, num_train, num_val, num_test):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train + num_val]] = True
    test_mask[indices[num_train + num_val:num_train + num_val + num_test]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

def split_dataloader(data, graphs, batch_size, seed=0, name='cora'):
    data = create_masks(data, int(data.num_nodes * 0.6), int(data.num_nodes * 0.2), int(data.num_nodes * 0.2))
    train_idx = data.train_mask.nonzero().squeeze()
    val_idx = data.val_mask.nonzero().squeeze()
    test_idx = data.test_mask.nonzero().squeeze()
    train_dataset = [graphs[idx] for idx in train_idx]
    val_dataset = [graphs[idx] for idx in val_idx]
    test_dataset = [graphs[idx] for idx in test_idx]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # use DataListLoader for DP rather than DataLoader
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def split_dataloader_Graph(graphs, batch_size, seed=0, name='bace' , train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):

    num_graphs = len(graphs)
    # 打乱图的顺序
    indices = np.random.permutation(num_graphs)
    
    # 根据比例划分训练、验证、测试集
    train_end = int(num_graphs * train_ratio)
    val_end = int(num_graphs * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # 划分图数据集
    train_dataset = [graphs[idx] for idx in train_idx]
    val_dataset = [graphs[idx] for idx in val_idx]
    test_dataset = [graphs[idx] for idx in test_idx]
    
    # 创建 DataLoader
    train_loader = DataListLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataListLoader(val_dataset, batch_size=batch_size)
    test_loader = DataListLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader