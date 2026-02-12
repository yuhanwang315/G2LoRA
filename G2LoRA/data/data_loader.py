import copy
import json
import heapq
import torch

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
import os
import torch
from torch_geometric.data import Data
from torch.utils.data import Subset
import random
import heapq
from torch_geometric.utils import k_hop_subgraph
import copy
import torch_geometric.transforms as T
import numpy as np

def merge_data(data_list):
    # Initialize the merged Data object
    merged_x = []
    merged_edge_index = []
    merged_y = []
    merged_raw_texts = []
    merged_label_texts = []  # Initialize a list to store merged label_texts
    
    offset = 0  # To reassign ids when merging datasets
    y_offset = 0  #
    for data in data_list:
        # Merge node features
        merged_x.append(data.x)

        # Adjust edge_index by adding offset to node indices
        # Adding offset to edge_index to ensure uniqueness of node indices
        adjusted_edge_index = data.edge_index + offset  # Add offset to the edge indices
        merged_edge_index.append(adjusted_edge_index)

        # Merge labels
        merged_y.append(data.y+y_offset)
        # Merge raw_texts
        merged_raw_texts.extend(data.raw_texts)

        # Merge label_texts
        merged_label_texts.extend(data.label_texts)  # Combine label_texts

        # Update the offset for the next dataset (based on the number of nodes in the current graph)
        offset += data.x.size(0)  # Add the number of nodes from this data to the offset
        y_offset += len(data.label_texts)

    # Concatenate the merged lists into one final tensor or list
    merged_x = torch.cat(merged_x, dim=0)  # Merge node features along rows
    merged_edge_index = torch.cat(merged_edge_index, dim=1)  # Merge edge indices along columns
    merged_y = torch.cat(merged_y, dim=0)  # Merge labels
    # merged_raw_texts and merged_label_texts are already lists, no need to change
    # Create a new Data object for the merged graph
    merged_data = Data(
        x=merged_x,
        edge_index=merged_edge_index,
        y=merged_y,
        raw_texts=merged_raw_texts,
        label_texts=merged_label_texts,  # Add merged label_texts
    )

    return merged_data

def merge_id_by_class(data_list, id_by_classlist):
    merged_id_by_class = {}

    class_offset = 0
    node_offset = 0 

    for i, id_by_class in enumerate(id_by_classlist):
        for class_label, ids in id_by_class.items():
            new_class_label = class_label + class_offset

            if new_class_label not in merged_id_by_class:
                merged_id_by_class[new_class_label] = []

            merged_id_by_class[new_class_label].extend(
                [idx + node_offset for idx in ids]
            )
        class_offset += len(id_by_class)
        node_offset += data_list[i].x.size(0)

    return merged_id_by_class



class TextDataset(Dataset):
    def __init__(self, dataset, data_path, CL_type,type):
        self.datasets = dataset.split("+") 
        self.data_path = data_path
        self.CL_type = CL_type
        self.type = type 

        datalist = []
        id_by_classlist = []
        raw_texts_list = []  
        label_texts_list = []  
        if len(self.datasets) > 1:
            if CL_type == "task":
                for j, dataset_i in enumerate(self.datasets):
                    if self.type[j] == 'node':
                        data, id_by_class = self._load_data(dataset_i)
                    elif self.type[j] == 'edge':
                        data, id_by_class = self._load_data(dataset_i,self.type[j])
                    elif self.type[j] == 'graph':
                        data, id_by_class = self._load_graph_data(dataset_i,self.type[j])
                    raw_texts_list.append(data.raw_texts)
                    label_texts_list.append(data.label_texts)
                    datalist.append(data)
                    id_by_classlist.append(id_by_class)

                self.data = datalist
                self.id_by_class = id_by_classlist
                self.raw_texts = raw_texts_list
                self.label_texts = label_texts_list

            else:
                for dataset_i in self.datasets:
                    data, id_by_class = self._load_data(dataset_i)
                    datalist.append(data)
                    id_by_classlist.append(id_by_class)
                self.data = merge_data(datalist)
                self.raw_texts = self.data.raw_texts
                self.label_texts = self.data.label_texts
                self.id_by_class = merge_id_by_class(datalist,id_by_classlist)
                # print("JINTIAN:merge_id_by_class:",self.id_by_class)

        else:
            self.data, self.id_by_class = self._load_data(self.datasets[0])
            self.raw_texts = self.data.raw_texts
            self.label_texts = self.data.label_texts

    def __getitem__(self, idx):

        item = {}
        item['node_id'] = idx
        item["labels"] = self.data.y[idx].to(torch.long)
        item["raw_text"] = self.raw_texts[idx]
        item["label_text"] = self.label_texts[item["labels"]]

        return item

    def __len__(self):
        return len(self.raw_texts)
    
    def _get_label_text(self, dataset):
        label_text_list = None
        with open('LLM4GCL/common/label_text.json', 'r', encoding='utf-8') as f:
            label_text_list = json.load(f)[dataset]
        return label_text_list

    def _load_data(self, dataset ,type='node'):
        path = self.data_path + dataset + ".pt"
        data = torch.load(path)
        if dataset == 'products' or dataset == 'arxiv_23':
            if dataset == 'products':
                empty_label = [29, 33]
                delete_label = [22, 26, 27, 30, 34, 35, 38, 39, 40, 41, 43]
            elif dataset == 'arxiv_23':
                empty_label = [0, 19]
                delete_label = [12]

            mask = ~torch.isin(data.y, torch.tensor(delete_label))
            to_remove_idx = (~mask).nonzero(as_tuple=True)[0]
            remaining_idx = torch.arange(data.x.size(0))[~torch.isin(torch.arange(data.x.size(0)), to_remove_idx)]

            edge_mask = ~torch.isin(data.edge_index[0], to_remove_idx) & ~torch.isin(data.edge_index[1], to_remove_idx)
            data.edge_index = data.edge_index[:, edge_mask]

            node_map = {}
            edge_index = [[], []]
            for ori_idx, curr_idx in zip(remaining_idx.tolist(), [i for i in range(len(data.x[mask]))]):
                node_map[ori_idx] = curr_idx

            for i in range(data.edge_index.size(1)):
                if data.edge_index[0][i].item() in node_map.keys():
                    edge_index[0].append(node_map[data.edge_index[0][i].item()])

            for i in range(data.edge_index.size(1)):
                if data.edge_index[1][i].item() in node_map.keys():
                    edge_index[1].append(node_map[data.edge_index[1][i].item()])

            data.edge_index = torch.stack((torch.tensor(edge_index[0]), torch.tensor(edge_index[1])), dim=0)
            
            data.x = data.x[mask]
            data.y = data.y[mask]
            data.raw_texts = [data.raw_texts[i] for i in range(len(data.raw_texts)) if mask[i]]
            data.num_nodes = data.num_nodes - 1

            labels = data.y

            delete_label.extend(empty_label)
            delete_label.sort()
            labels = [label for label in data.y if label not in delete_label]
            labels = [label - sum(label > x for x in delete_label) for label in labels]
            labels = torch.tensor(labels)
            data.y = labels

        edge_index, _ = add_self_loops(data.edge_index)
        
        new_data = Data(
            x=data.x,
            edge_index=edge_index,
            y=data.y,
            raw_texts=data.raw_texts,
        )
        data = new_data
        print("new_data:",len(data.x))
        print("edge_index.max()",data.edge_index.max())
        labels = data.y
        class_list = labels.unique().numpy()
        id_by_class = {i: [] for i in class_list}
        for id, cla in enumerate(labels):
            id_by_class[cla.item()].append(id)

        num_nodes = [len(v) for _, v in id_by_class.items()]
        sorted_class_idx = heapq.nlargest(labels.max().item() + 1, enumerate(num_nodes), key=lambda x: x[1])

        # Re-order labels
        label_texts = self._get_label_text(dataset)
        sorted_label_texts = copy.deepcopy(label_texts)
        for i, (id, _) in enumerate(sorted_class_idx):
            class_idx = id_by_class[id]
            labels[class_idx] = i
            sorted_label_texts[i] = label_texts[id]
        data.label_texts = sorted_label_texts
        print("data.label_texts:",data,data.label_texts)
        class_list = labels.unique().numpy()
        id_by_class = {i: [] for i in class_list}
        for id, cla in enumerate(labels):
            id_by_class[cla.item()].append(id)

        print(f"--------------------------------------------")
        print(f"Load dataset {dataset}!")
        print(f"Node num: {data.x.shape[0]}")
        print(f"Edge num: {data.edge_index.shape[1]}")
        print(f"Class num: {data.y.max().item() + 1}")
        print(f"Feature dim: {data.x.shape[1]}")
        for cls in id_by_class.keys():
            print(f"Class {cls}: {len(id_by_class[cls])} samples")
        print(f"--------------------------------------------")



        return data, id_by_class



    def _load_graph_data(self, dataset, type='graph', per_class_cap=None, num_graphs_per_class=5, hop=1):
        path = self.data_path + dataset + ".pt"
        data = torch.load(path) 
        print("data:",data)
        transform = T.AddRandomWalkPE(
            walk_length=32,
            attr_name='pe'
        )
        all_graphs = [] 
        id_by_class = {} 
        label_texts = self._get_label_text(dataset)
        y = data.y
        num_classes = y.max().item() + 1 


        for c in range(num_classes):
            class_nodes = torch.where(y == c)[0]
            if class_nodes.numel() == 0:
                continue

            seeds_list = []
            split_size = 3 
            seeds_list = list(torch.split(class_nodes, split_size))
 
            c_key = int(c) 
            id_by_class.setdefault(c_key, [])
            for seeds in seeds_list:

                subset, sub_ei, mapping, _ = k_hop_subgraph(
                    node_idx=seeds,
                    num_hops=hop,
                    edge_index=data.edge_index,
                    num_nodes=data.num_nodes,
                    relabel_nodes=True
                )
                if sub_ei.size(1) == 0:
                    sub_ei = torch.tensor([[mapping], [mapping]], dtype=torch.long)
                selected_texts = [data.raw_texts[i] for i in seeds]
                selected_texts2 = [data.raw_texts[i] for i in subset]

                complete_text = " ".join(selected_texts)
                graph = Data(
                    x=data.x[subset],
                    edge_index=sub_ei,
                    y=data.y[seeds[0]],
                    root_n_index=mapping[0],
                    raw_texts=complete_text,
                    raw_text_node=selected_texts2,
                    label_texts=label_texts
                )
                if transform is not None:
                    graph = transform(graph)
                all_graphs.append(graph)

                gid = len(all_graphs)

                id_by_class[c_key].append(gid-1)
        new_data = Data(
                data=all_graphs, 
                raw_texts=[graph.raw_texts for graph in all_graphs],
                label_texts=label_texts, 
                y=torch.tensor([graph.y for graph in all_graphs])
            )
        
        return new_data, id_by_class

