ckpt_path = f"clip.pt"
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import pandas as pd
from ..models_GTAlign import GraphCLIP
import torch.optim as optim
import torch.nn as nn
from ..models_GTAlign.dp import create_logits, calculate_loss
from ..utils.args import Arguments
from ..utils.process import parse_target_data
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from torch_geometric.data import Data,Batch
import random
import os
import time
from ..models_GTAlign.attention_lora import Attention_LoRA
from ..models_GTAlign.gt import LoRAGPSConv
import math
from copy import deepcopy
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import k_hop_subgraph

from LLM4GCL.models import BaseModel
from LLM4GCL.backbones import LLaMANet
from LLM4GCL.common.utils import adjust_learning_rate, _save_checkpoint, _reload_best_model
from ..models_GTAlign.BertLayer import BertLayer

DEFAULT_THRESHOLD = 0.985
TOP_SELECT = 1
EPOCH_NUM = 4
TOP_K_RATIO = 0.1
LAMBDA_SCALE = 30

eval_template = {
    'cora': "this paper has a topic on {c}",
    'citeseer': "good paper of {c} ",
    'cora+citeseer': "this paper has a topic on {c} ",
    'pubmed': "{c}",
    'arxiv_2023': "it belongs to {c} research area",
    'wikics': "it belongs to {c} research area",
    'photo': "this product belongs to {c}",
    'computer': "is {c} category",
    'history': "this book belongs to {c}",
    'instagram': "This post belongs to {c}",
    'reddit': "{c}",
    'cora+citeseer+wikics':"{c}",
}

def compute_similarity_with_proto(current_proto, task_protos):
    similarity_dict = {}
    for k in range(len(task_protos)):
        task = task_protos[k]
        similarity = torch.dot(current_proto, task) / (torch.norm(current_proto) * torch.norm(task))
        similarity_dict[k] = similarity
    return similarity_dict

def build_ego_graphs_from_big_data(
    data,
    hop=1,
    pe_walk_length=32
):
    transform = T.AddRandomWalkPE(
        walk_length=pe_walk_length,
        attr_name='pe'
    )
    used_nodes = torch.unique(data.edge_index)
    used_nodes = used_nodes[used_nodes < data.x.size(0)]  
    num_nodes = data.x.size(0)
    collected_graph_data = []
    num_nodes = used_nodes.size(0)
    for node_id in range(num_nodes):
        subset, edge_index, mapping, _ = k_hop_subgraph(
            node_id,
            hop,
            data.edge_index,
            relabel_nodes=True
        )

        if edge_index.size(1) == 0:
            edge_index = torch.tensor([[mapping], [mapping]])

        graph = Data(
            x=data.x[subset],
            edge_index=edge_index,
            y=data.y[node_id],
            root_n_index=mapping
        )
        graph = transform(graph)

        collected_graph_data.append(graph)

    return collected_graph_data
class InnerProd(torch.nn.Module):
    def forward(self, x, y):
        assert x.size(1) == y.size(1), f"Feature dimension mismatch: x={x.size(1)}, y={y.size(1)}"
        assert x.size(0) == y.size(0), f"Number of edges mismatch: x={x.size(0)}, y={y.size(0)}"
        return (x * y).sum(dim=1)


class GraphCLIPPrompter:
    def __init__(self, model, tokenizer, device, config, model_name,datasets,random_seed=42,
                 tune_lr=1e-5, tune_epochs=5, tune_batch_size=4, weight_decay=1e-5,n_tasks=3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.seed = random_seed
        self.tune_lr = tune_lr
        self.tune_epochs = tune_epochs
        self.tune_batch_size = tune_batch_size
        self.weight_decay = weight_decay
        self.model.eval()
        self.feature_list = []
        self.feature_share = []
        self.project_type = []
        self.lamb = 0.95
        self.lame = 1.0
        self._cur_task=0
        self.feature_list_text = []
        self.feature_list_task = []
        self.feature_list_y = []
        self.project_type_text = []
        self.lamda = [[0 for _ in range(12)] for _ in range(12)]
        self.lamda_old = [[0 for _ in range(12)] for _ in range(12)]
        self.proto_list_dict = {}
        self.proto_index_dict=[]
        self.config=config
        self.model_name=model_name
        self.datasets=datasets
        self.n_tasks=n_tasks
        self.total_sessions =n_tasks
        self.feature_share_text=[]
        self.classifier=InnerProd()
        self.feature_share=[]

    def tune(self, target_graph,train_loader,val_loader,test_loader ,checkpoint_path,classes, c_descs, dataset_name,class_src=0,islora_g=True , islora_t=True ,task = -1,type_task='node'):
        
        self._cur_task=task

        print(f"Performing GraphCLIP prompt tuning (training mode)...")
        class_features = {}
        proto_list_dict={}
        print("class_src:",class_src)
        for label_idx in range(len(classes)):
            class_name = classes[label_idx]
            class_desc = c_descs[label_idx]
            prompt = eval_template[dataset_name].format(c=class_name) + class_desc
            current_texts = [prompt]  
            batch_t = self.tokenizer(current_texts, truncation=True, padding=True, return_tensors="pt", max_length=512).to(self.device)

            text_embs = self.model.encode_text(
                batch_t["input_ids"],
                batch_t.get('token_type_ids', None),
                batch_t["attention_mask"],
                task=task
            )
            id=label_idx+class_src
            if id not in proto_list_dict:
                proto_list_dict[id] = []  

            proto_list_dict[id]=text_embs[0]

        tuning_loader = train_loader
        labels = []
        edge_label_index=[]
        if type_task == 'node':
            for batch in tuning_loader:
                y = batch['labels']     
                print("ynode:",type(y))
                labels.extend(y)
        elif type_task == 'edge':
            for batch in tuning_loader:
                y = batch.edge_label       
                labels.extend(y)
                print("yedge:",type(y))
                edge_label_index.extend(batch.edge_label_index)
        elif type_task == 'graph':
            for batch in tuning_loader:
                y = [item['data'].y for item in batch]
                print("ygraph:",type(y))
                labels.extend(y)
        for name, param in self.model.graph_model.named_parameters():
            if "lora_A_k" in name or "lora_A_v" in name or "bias" in name:
                param.requires_grad_(False)
            for i in range(0,task):
                if "lora_B_k" + "." + str(i) in name:
                    param.requires_grad_(False)
                if "lora_B_v" + "." + str(i) in name:
                    param.requires_grad_(False)
            for i in range(task+1,self.n_tasks):
                if "lora_B_k" + "." + str(i) in name:
                    param.requires_grad_(False)
                if "lora_B_v" + "." + str(i) in name:
                    param.requires_grad_(False)
            if "lora_B_k" + "." + str(task) in name:
                param.requires_grad_(True)
            if "lora_B_v" + "." + str(task) in name:
                param.requires_grad_(True)


        with torch.no_grad():
            if task >0:
                for module in self.model.graph_model.modules():
                    if isinstance(module, LoRAGPSConv):
                            module.x.zero_()
                            module.x = None
                    if isinstance(module, Attention_LoRA):
                            module.x.zero_()
                            module.x = None

            if type_task == 'node':
                self.evaluate(target_graph,tuning_loader, classes, c_descs, dataset_name,task,class_src)
            elif type_task == 'edge':
                self.evaluate_link(target_graph,tuning_loader,task)
            elif type_task == 'graph':
                self.evaluate_graph(target_graph,tuning_loader, classes, c_descs, dataset_name,task)
            if task == 0:
                kk=0
                similarities_all = []
                for module in self.model.graph_model.modules():
                    similarities = []
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix
                        for i, x in enumerate(module.x):
                            label = labels[i].item() 
                            if label not in class_features:
                                class_features[label] = []
                            if len(class_features[label]) == 0:
                                class_features[label] = x.unsqueeze(0)  
                            else:
                                class_features[label] = torch.cat((class_features[label], x.unsqueeze(0)), dim=0)

                        module.x = None
                        proto_list_dict_layer={}
                        labels_set = np.unique(labels) 
                        if not kk in self.proto_list_dict.keys():
                            self.proto_list_dict[kk] = []
                        proto_list_dict_layer = self.proto_list_dict[kk]
                        for item in labels_set:
                            proto= proto_list_dict[item.item()]
                            if task == 0 and item == 0:
                                similarities.append(-1)
                            else:
                                similarity = compute_similarity_with_proto(proto,proto_list_dict_layer)
                                similarities.append(similarity)
                            proto_list_dict_layer.append(proto)


                        U, S, V = torch.linalg.svd(cur_matrix)
                        module.lora_A_k[task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                        module.lora_A_v[task].weight.data.copy_(U[:,:module.rank].T/math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0   
                        module.x=None  
                        self.proto_list_dict[kk] = proto_list_dict_layer
                        kk+=1
                        similarities_all.append(similarities)
            else:
                kk = 0
                similarities_all = []

                for module in self.model.graph_model.modules():
                    similarities = []
                    if isinstance(module, Attention_LoRA):
                        print("module.x.shapeA:",task,module.x.shape)
                        cur_matrix = module.cur_matrix
                        for i, x in enumerate(module.x):
                            if type_task == 'edge':
                                class_features = {}
                                for i in range(len(edge_label_index[0])):
                                    u = edge_label_index[0][i]
                                    v = edge_label_index[1][i]
                            
                                    x = torch.cat([module.x[u], module.x[v]], dim=-1)  

                                    label = labels[i].item() + class_src  

                                    if label not in class_features:
                                        class_features[label] = x.unsqueeze(0)
                                    else:
                                        class_features[label] = torch.cat(
                                            (class_features[label], x.unsqueeze(0)), dim=0
                                        )
                            else:
                                for i, x in enumerate(module.x):
                                    label = labels[i].item() 
                                    if label not in class_features:
                                        class_features[label] = []
                                    if len(class_features[label]) == 0:
                                        class_features[label] = x.unsqueeze(0) 
                                    else:
                                        class_features[label] = torch.cat((class_features[label], x.unsqueeze(0)), dim=0)
                        module.x = None
                        proto_list_dict_layer={}
                        labels_set = np.unique(labels) 
                        if not kk in self.proto_list_dict.keys():
                            self.proto_list_dict[kk] = []
                        proto_list_dict_layer = self.proto_list_dict[kk]
                        num_classes=len(proto_list_dict_layer)
                        for item in labels_set:
                            proto= proto_list_dict[item.item()]
                            if task == 0 and item == 0:
                                similarities.append(-1)
                            else:
                                similarity = compute_similarity_with_proto(proto,proto_list_dict_layer)
                                similarities.append(similarity)
                            proto_list_dict_layer.append(proto)

                        self.proto_list_dict[kk] = proto_list_dict_layer
                        mean_sims = {}
                        for c in range(num_classes):
                            vals = []
                            for d in similarities:
                                if c in d:
                                    vals.append(d[c])
                            mean_sims[c] = torch.mean(torch.stack(vals))
                        all_weights_all=[]
                        for feature_class_list in self.feature_list:
                            all_weights = []
                            class_union = None
                            class_intersection = None

                            for ii, feature in enumerate(feature_class_list):
                                feature = feature.cpu().numpy() if isinstance(feature, torch.Tensor) else feature
                                r = feature.shape[1]          
                                w = torch.exp(-mean_sims[ii])*10          
                                all_weights.extend([w] * r)

                                if class_union is None:
                                    class_union = feature
                                else:
                                    class_union = np.hstack((class_union, feature))

                                if class_intersection is None:
                                    class_intersection = feature
                                else:
                                    class_intersection = np.dot(
                                        np.dot(class_intersection, class_intersection.transpose(1,0)),
                                        feature
                                    )

                            all_weights_all.append(all_weights)
                            Uf = torch.Tensor(np.dot(class_union, class_union.transpose()))
                            Uf_share = torch.Tensor(np.dot(class_intersection, class_intersection.transpose()))
                            self.union_space.append(torch.tensor(class_union))
                            self.feature_mat.append(Uf)
                            self.feature_mat_share.append(Uf_share)
                       
                        feature_mat= torch.Tensor(np.dot(self.feature_list_y[kk], self.feature_list_y[kk].transpose(1,0)))
                        proj_g=torch.mm(feature_mat,cur_matrix)
                        g=cur_matrix
                        ratio = torch.norm(proj_g) / (torch.norm(g) + 1e-8)
                        device = cur_matrix.device  
                        self.feature_mat_share[kk] = self.feature_mat_share[kk].to(device)  
                        cur_matrix2 = cur_matrix.to(device)  
                        cur_matrix2 = torch.mm(self.feature_mat_share[kk],cur_matrix2)
                        UShare, SShare, VShare = torch.linalg.svd(cur_matrix2, full_matrices=False)
                        if ratio>0.9 :
                            for name, param in module.named_parameters():
                                for i in range(0,task):
                                    if "lora_B_k" + "." + str(i) in name:
                                        param.requires_grad_(True)
                                    if "lora_B_v" + "." + str(i) in name:
                                        param.requires_grad_(True)
                                    module.lora_A_k[i].weight.data.copy_(UShare[:,:module.rank].T/math.sqrt(3))
                                    module.lora_A_v[i].weight.data.copy_(UShare[:,:module.rank].T/math.sqrt(3))
                        weights = torch.tensor(all_weights_all[kk], device=self.union_space[kk].device)
                        W = torch.diag(weights)
                        proj = self.union_space[kk] @  W @ self.union_space[kk].transpose(1,0)
                        cur_matrix = cur_matrix - proj @ cur_matrix
                        cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)
                        module.lora_A_k[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                        module.lora_A_v[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
                        kk += 1
                        similarities_all.append(similarities)                        

        self.model.train()

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": filter(lambda p: p.requires_grad, self.model.graph_model.parameters()),
                    "lr": self.tune_lr,
                    "name": "graph",
                },
                {
                    "params": filter(lambda p: p.requires_grad, self.model.text_model.parameters()),
                    "lr": self.tune_lr,
                    "name": "text",
                },
            ],
            weight_decay=self.weight_decay,
        )
        if type_task == 'node':
            self.train_node(target_graph, tuning_loader, val_loader, optimizer, classes, c_descs, dataset_name, task, checkpoint_path,class_src)
        elif type_task == 'edge':
            self.train_link(target_graph,tuning_loader,val_loader, optimizer,task,checkpoint_path,classifier=self.classifier)
        elif type_task == 'graph':
            self.train_graph(target_graph, tuning_loader, val_loader,classes, c_descs, dataset_name,optimizer,task,checkpoint_path)

        with torch.no_grad():
            for module in self.model.graph_model.modules():
                if isinstance(module, LoRAGPSConv):
                    module.x.zero_()
                    module.x = None
                if isinstance(module, Attention_LoRA):
                    module.x.zero_()
                    module.x = None
            if type_task == 'node':
                self.evaluate(target_graph,tuning_loader, classes, c_descs, dataset_name,task,class_src)
            elif type_task == 'edge':
                self.evaluate_link(target_graph,tuning_loader,task,classifier=self.classifier)
            elif type_task == 'graph':
                self.evaluate_graph(target_graph,tuning_loader, classes, c_descs, dataset_name,task)
                
            threshold = 0.97 + task * 0.003
            mat_list = []
            class_features_all=[]
            for module in self.model.graph_model.modules():
                if isinstance(module, Attention_LoRA):
                    class_features= {}
                    if type_task == 'edge':
                        class_features = {}
                        for i in range(len(edge_label_index[0])):
                            u = edge_label_index[0][i]
                            v = edge_label_index[1][i]
                      
                            x1 = module.x[u]
                            x2 = module.x[v]
                            label = labels[i].item() + class_src  
                            if label not in class_features:
                                class_features[label] = x1.unsqueeze(0)
                                class_features[label] = x2.unsqueeze(0)
                            else:
                                class_features[label] = torch.cat(
                                    (class_features[label], x1.unsqueeze(0)), dim=0
                                )
                                class_features[label] = torch.cat(
                                    (class_features[label], x2.unsqueeze(0)), dim=0
                                )
                    else:
                        for i, x in enumerate(module.x):
                           
                            label = labels[i].item()+ class_src
                            if label not in class_features:
                                class_features[label] = []
                            if len(class_features[label]) == 0:
                                class_features[label] = x.unsqueeze(0) 
                            else:
                                class_features[label] = torch.cat((class_features[label], x.unsqueeze(0)), dim=0)
                    module.x = None
                    class_features_all.append(class_features)
                    mat_list.append(deepcopy(module.cur_matrix))
            num=0
            mat_list=[]
            for module in self.model.graph_model.modules():
                if isinstance(module, Attention_LoRA):
                    features = class_features_all[num]
                    num+=1
                    mat_class = []
                    for item in labels_set:
                        item = item.item() 
                        feature_classwise = features[item]
                        act = feature_classwise
                        activation = act.transpose(0, 1)
                        mat_class.append(activation)
                    mat_list.append(mat_class)
            if task == 0:
                self.feature_list , self.proto_index_dict = self.update_GPM(task, mat_list, threshold, self.feature_list, similarities_all)
            else: 
                self.feature_list,self.proto_index_dict = self.update_GPM (task, mat_list, threshold, self.feature_list, similarities_all, self.proto_index_dict)
            
            mat_list = []
            for module in self.model.graph_model.modules():
                if isinstance(module, Attention_LoRA):
                    mat_list.append(deepcopy(module.cur_matrix))
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0
            self.update_DualGPM(mat_list)
        self.model.eval()
        print(f"Prompt tuning (training) completed.")
        return

    def update_GPM (self, taskId, mat_list, threshold, feature_list=[], similarities = [], proto_index_dict = {}):
        print ('Threshold: ', threshold) 
        if not feature_list:  #task=0 
            for i in range(len(mat_list)):  
                similarity_sub = similarities[i]  
                feature_class_list = []
                activation = mat_list[i]  
                proto_class_index_dict = {}  
                for j in range(len(activation)):   
                    activation_task = activation[j] 
                    if j > 0: 
                        similarity = similarity_sub[j] 
                        U1, S1, Vh1 = np.linalg.svd(activation_task.cpu().numpy(), full_matrices=False)  
                        sval_total = (S1**2).sum()
                        for feature in feature_class_list:
                            feature_cpu = feature
                            activation_task_cpu = activation_task.cpu().numpy()
                            activation_task_hat = activation_task_cpu - np.dot(np.dot(feature_cpu, feature_cpu.T), activation_task_cpu)

                        U,S,Vh = np.linalg.svd(activation_task_hat, full_matrices=False)
                        sval_hat = (S**2).sum()
                        sval_ratio = (S**2)/sval_total               
                        accumulated_sval = (sval_total-sval_hat)/sval_total
                        r = 0
                        for ii in range (sval_ratio.shape[0]):
                            if accumulated_sval < threshold:
                                accumulated_sval += sval_ratio[ii]
                                r += 1
                            else:
                                break
                        if r == 0:
                            print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                            continue
                        feature_class_list.append(U[:,0:r])
                        proto_class_index_dict[j] = len(feature_class_list)-1

                    else: 
                        U, S, Vh = np.linalg.svd(activation_task.cpu().numpy(), full_matrices=False)
                        proto_class_index_dict[j] = j
                        sval_total = (S**2).sum()
                        sval_ratio = (S**2)/sval_total
                        r = np.sum(np.cumsum(sval_ratio)<threshold)
                        feature_class_list.append(U[:,0:r])
           
                proto_index_dict[i] = proto_class_index_dict
                feature_list.append(feature_class_list)
                U_task=np.hstack(feature_class_list)
                self.feature_list_task.append([U_task])

        else:
            for i in range(len(mat_list)):
                proto_class_index_dict = proto_index_dict[i] 
                similarity_sub = similarities[i]
                feature_class_list = feature_list[i]
                feature_class_list_cur=[]
                activation = mat_list[i]
                pre_class_count = taskId*len(activation)
                for task_id in range(len(activation)):
                    activation_task = activation[task_id]

                    similarity = similarity_sub[task_id]
            

                    U1, S1, Vh1 = np.linalg.svd(activation_task.cpu().numpy(), full_matrices=False)
                    sval_total = (S1**2).sum()

                    for feature in feature_class_list:

                        if isinstance(feature, torch.Tensor):
                            feature = feature.cpu().numpy()  
                        if isinstance(activation_task, torch.Tensor):
                            activation_task = activation_task.cpu().numpy()  

                        activation_task_hat = activation_task - np.dot(np.dot(feature, feature.transpose()), activation_task)
                    U,S,Vh = np.linalg.svd(activation_task_hat, full_matrices=False)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total
                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                        continue
                    feature_class_list.append(U[:,0:r])
                    feature_class_list_cur.append(U[:,0:r])
                    proto_class_index_dict[pre_class_count+task_id] = len(feature_class_list)-1
                self.feature_list_task[i].append(np.hstack(feature_class_list_cur))
        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(feature_list)):
            for j in range(len(feature_list[i])):
                print ('Layer {} and class {}: {}/{}'.format(i+1,j+1 ,feature_list[i][j].shape[1], feature_list[i][j].shape[0]))

        print('-'*40)
        return feature_list,proto_index_dict
    @torch.no_grad()
    def evaluate(self, target_graph,loader, classes, c_descs, dataset_name,task,class_src=0,pred_task_id=-1,class_range=[0,100]):
        self.model.eval()
        text_inputs = [eval_template[dataset_name].format(c=c) for c in classes]
        text_inputs = [ti + desc for ti, desc in zip(text_inputs, c_descs)]
        batch_t = self.tokenizer(text_inputs, truncation=True, padding=True, return_tensors="pt", max_length=512).to(self.device)
        with torch.no_grad():
            text_embs = self.model.encode_text(batch_t["input_ids"],
                                               batch_t.get('token_type_ids', None),
                                               batch_t["attention_mask"],task=task)
            text_embs /= text_embs.norm(dim=-1, keepdim=True)


        all_preds = []
        all_labels = []
        for batch_id in tqdm(loader, desc=f"Evaluating {dataset_name}", leave=False):
            idxs = batch_id['node_id'].tolist()    
            graphs = [target_graph[i] for i in idxs]
            batch = Batch.from_data_list(graphs).to(self.device)
            if not hasattr(batch, 'pe') or batch.pe is None:
                raise AttributeError("Batch object is missing 'pe' attribute. Please check if parse_target_data correctly added positional encoding.")
                
            with torch.no_grad():
                if pred_task_id>-1:
                    graph_embs, _ = self.model.encode_graph(batch,task=task)
                else:
                    graph_embs, _ = self.model.encode_graph(batch,task=task)
                graph_embs /= graph_embs.norm(dim=-1, keepdim=True)
                similarity = (100.0 * graph_embs @ text_embs.T)
                predictions = similarity.argmax(dim=1)
                labels=batch.y.cpu()
                if class_range is not None:
                    mask = (labels >= class_range[0]) & (labels <= class_range[1])
                    predictions = predictions[mask]
                    labels = labels[mask] - class_src 

            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro')

        return accuracy,f1

    def train_graph(self,target_graph, tuning_loader,val_loader, classes, c_descs, dataset_name,optimizer,task,checkpoint_path):
        progress_bar = tqdm(range(self.tune_epochs))
        progress_bar.set_description(f'Training | Iter {self.seed} | Session {task}')
        tolerate, best_acc_valid = 0, 0.
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.tune_epochs):
            total_loss = 0
            num_batches = 0
            batch_iterator = tqdm(tuning_loader, desc=f"Tune Epoch {epoch+1}/{self.tune_epochs}", leave=False)
            for batch in batch_iterator:
                idxs=[batch_i['node_id'] for batch_i in batch]
                graphs = [target_graph[i] for i in idxs]
                batch = Batch.from_data_list(graphs).to(self.device)
                optimizer.zero_grad()
                graph_embs,_ = self.model.encode_graph(batch,task)
                graph_embs=graph_embs.squeeze(dim=1) 
                labels=batch.y.cpu()
                batch_labels = labels
                current_texts = []
                for label_idx in batch_labels:
                    if 0 <= label_idx < len(classes):
                        class_name = classes[label_idx]
                        class_desc = c_descs[label_idx]
                        prompt = eval_template[dataset_name].format(c=class_name) + class_desc
                        current_texts.append(prompt)
                    else:
                        print(f"Warning: Invalid label index {label_idx} in batch.")
                        current_texts.append("invalid label")

                if not current_texts:
                    print("Warning: No valid text in the current batch, skipping.")
                    continue
                batch_t = self.tokenizer(current_texts, truncation=True, padding=True, return_tensors="pt", max_length=512).to(self.device)
                text_embs = self.model.encode_text(batch_t["input_ids"],
                                                   batch_t.get('token_type_ids', None),
                                                   batch_t["attention_mask"])

                graph_embs = graph_embs / graph_embs.norm(dim=-1, keepdim=True)
                text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

                logit_scale = self.model.logit_scale.exp()
                logits_per_graph, logits_per_text = create_logits(graph_embs, text_embs, logit_scale)
                loss = calculate_loss(logits_per_graph, logits_per_text, criterion)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                batch_iterator.set_postfix({"Loss": loss.item()})

            avg_loss = total_loss / num_batches if num_batches > 0 else 0

            graph_grad_norm = 0.0
            text_grad_norm = 0.0
            gg=0
            tt=0
            for name, p in self.model.named_parameters():
                if p .requires_grad:
                    if 'graph_model' in name and 'convs' in name:
                        graph_grad_norm += p.grad.norm(2).item()
                        gg+=1

                    elif 'text_model'in name and 'encoder' in name:
                        text_grad_norm += p.grad.norm(2).item()
                        tt+=1
            print("graph_grad_norm:",graph_grad_norm/gg,text_grad_norm/tt,gg,tt)
            eps = 1e-6
            rho_graph = graph_grad_norm/gg / (text_grad_norm/tt + eps)
            rho_text  = text_grad_norm/tt  / (graph_grad_norm/gg + eps)
            lamda_lr=0.5
            for group in optimizer.param_groups:
                print("group:",group["name"])
                if group["name"] == "text":
                    print("lr :",group["lr"] )
                    group["lr"] =self.tune_lr*rho_graph*lamda_lr 
            progress_bar.write("Session: {} | Epoch: {} | Loss: {:.4f}".format(task, epoch, avg_loss))

            if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                acc_valid, f1_valid = self.evaluate_graph(target_graph,val_loader, classes, c_descs, dataset_name,task)
                progress_bar.write("Session: {} | Epoch: {} | Acc Val: {:.4f} | F1 Val: {:.4f} | Tolerate: {}".format(task, epoch, acc_valid, f1_valid, tolerate))
                if acc_valid > best_acc_valid:
                    tolerate = 0
                    best_acc_valid = acc_valid
                    _save_checkpoint(self.model, optimizer, epoch, checkpoint_path, self.datasets, self.model_name, self.seed)
                else:
                    tolerate += 1
                    if tolerate > self.config['patience']: 
                        break

            progress_bar.set_postfix({
                'Loss': f"{avg_loss:.4f}",
                'Best Valid ACC': f"{best_acc_valid:.4f}",
                'Tolerate': tolerate
            })

            progress_bar.update(1)
        progress_bar.close()
        return avg_loss

    def train_node(self, target_graph, tuning_loader, val_loader, optimizer, classes, c_descs, dataset_name, task, checkpoint_path,class_src):
        criterion = nn.CrossEntropyLoss()
        progress_bar = tqdm(range(self.tune_epochs))
        progress_bar.set_description(f'Training | Iter {self.seed} | Session {task}')
        tolerate, best_acc_valid = 0, 0.
        modality_gaps_per_epoch=[]
        num_epoch=0
        graph_grad_norm = 0.0
        text_grad_norm = 0.0
        for epoch in range(self.tune_epochs):
            num_epoch+=1
            total_loss = 0
            num_batches = 0
            epoch_modality_gap=0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {param.shape}")
            for batch_id in tuning_loader:
                
                idxs = batch_id['node_id'].tolist()    
                graphs = [target_graph[i] for i in idxs]
                batch = Batch.from_data_list(graphs).to(self.device)
                if not hasattr(batch, 'pe') or batch.pe is None:
                    print("Warning: PE missing in tuning batch, may cause errors.")
                    continue

                optimizer.zero_grad()

                graph_embs, _ = self.model.encode_graph(batch ,task)

                batch_labels = [y - class_src for y in batch.y.cpu().tolist()]
                current_texts=[]
                for label_idx in batch_labels:
                    if 0 <= label_idx < len(classes):
                        class_name = classes[label_idx]
                        class_desc = c_descs[label_idx]
                        prompt = eval_template[dataset_name].format(c=class_name) + class_desc
                        current_texts.append(prompt)
                    else:
                        print(f"Warning: Invalid label index {label_idx} in batch.")
                        current_texts.append("invalid label")

                if not current_texts:
                    print("Warning: No valid text in the current batch, skipping.")
                    continue

                batch_t = self.tokenizer(current_texts, truncation=True, padding=True, return_tensors="pt", max_length=512).to(self.device)
                text_embs = self.model.encode_text(batch_t["input_ids"],
                                                   batch_t.get('token_type_ids', None),
                                                   batch_t["attention_mask"],task=task)
                graph_embs = graph_embs / graph_embs.norm(dim=-1, keepdim=True)
                text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
                logit_scale = self.model.logit_scale.exp()
                logits_per_graph, logits_per_text = create_logits(graph_embs, text_embs, logit_scale)
                loss = calculate_loss(logits_per_graph, logits_per_text, criterion)

                loss.backward()
                cosine_similarity = torch.matmul(graph_embs, text_embs.t()) 
                modality_gap = cosine_similarity.mean().item()  
                epoch_modality_gap += modality_gap 
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            progress_bar.write("Session: {} | Epoch: {} | Loss: {:.4f}".format(task, epoch, avg_loss))
            avg_modality_gap_for_epoch = epoch_modality_gap / num_batches if num_batches > 0 else 0
            modality_gaps_per_epoch.append(avg_modality_gap_for_epoch)
            gg=0
            tt=0
            for name, p in self.model.named_parameters():
                if p .requires_grad:
                    if 'graph_model' in name and 'convs' in name:
                        graph_grad_norm += p.grad.norm(2).item()
                        gg+=1

                    elif 'text_model'in name and 'encoder' in name:
                        text_grad_norm += p.grad.norm(2).item()
                        tt+=1
            print("graph_grad_norm:",graph_grad_norm/gg,text_grad_norm/tt,gg,tt)
            eps = 1e-6
            rho_graph = graph_grad_norm/gg / (text_grad_norm/tt + eps)
            rho_text  = text_grad_norm/tt  / (graph_grad_norm/gg + eps)
            lamda_lr=0.2
            for group in optimizer.param_groups:
                print("group:",group["name"])
                if group["name"] == "text":
                    print("lr :",group["lr"] )
                    group["lr"] =self.tune_lr*rho_graph*lamda_lr  
            if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                acc_valid, f1_valid = self.evaluate(target_graph,val_loader, classes, c_descs, dataset_name,task,class_src)
                progress_bar.write("Session: {} | Epoch: {} | Acc Val: {:.4f} | F1 Val: {:.4f} | Tolerate: {}".format(task, epoch, acc_valid, f1_valid, tolerate))
                if acc_valid > best_acc_valid:
                    tolerate = 0
                    best_acc_valid = acc_valid
                    _save_checkpoint(self.model, optimizer, epoch, checkpoint_path, self.datasets, self.model_name, self.seed)
                else:
                    tolerate += 1
                    if tolerate > self.config['patience']: 
                        break

            progress_bar.set_postfix({
                'Loss': f"{avg_loss:.4f}",
                'Best Valid ACC': f"{best_acc_valid:.4f}",
                'Tolerate': tolerate
            })

            progress_bar.update(1)
        progress_bar.close()
        num_epochs = list(range(1, num_epoch + 1))  
        modality_gap_df = pd.DataFrame({
            'Epoch': num_epochs,
            'Modality Gap': modality_gaps_per_epoch
        })
        for name, param in self.model.graph_model.named_parameters():
            if 'lora' in name:
                for i in range(0,task):
                    if "lora_B_k" + "." + str(i) in name:
                        param.requires_grad_(False)
                    if "lora_B_v" + "." + str(i) in name:
                        param.requires_grad_(False)
        plt.figure(figsize=(10, 6))
        plt.plot(num_epochs, modality_gaps_per_epoch, marker='o', color='b', label="Modality Gap")
        plt.xlabel('Epoch')
        plt.ylabel('cosine_similarity')
        plt.title(f'cosine_similarity over Epochs for Task {task}')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"./modality_gap_task_{task}.png")  
        return avg_loss
    def train_link(self,target_graph,data_loader,val_loader,optimizer,task,checkpoint_path,classifier=None):

        criterion = nn.CrossEntropyLoss()
        progress_bar = tqdm(range(self.tune_epochs))
        progress_bar.set_description(f'Training | Iter {self.seed} | Session {task}')
        tolerate, best_acc_valid = 0, 0.
        for epoch in range(self.tune_epochs):
            total_loss = 0
            num_batches = 0
            batch_iterator = tqdm(data_loader, desc=f"Tune Epoch {epoch+1}/{self.tune_epochs}", leave=False)
            for batch in data_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()  
                node_repr = self.model.graph_model.link(batch,task)
                batch_labels = batch.y.cpu().tolist()
                src_idx, dst_idx = batch.edge_label_index
                text=target_graph.raw_texts
                current_texts_dst = []
                global_src = batch.n_id[src_idx]
                global_dst = batch.n_id[dst_idx]
                for label_idx in global_dst:
                    current_texts_dst.append(text[label_idx])
                current_texts_src= []
                for label_idx in global_src:
                    current_texts_src.append(text[label_idx])
                # Tokenize the texts
                batch_t = self.tokenizer(current_texts_dst, truncation=True, padding=True, return_tensors="pt", max_length=512).to(self.device)
                text_embs_dst= self.model.encode_text(batch_t["input_ids"],
                                                batch_t.get('token_type_ids', None),
                                                batch_t["attention_mask"])
                batch_t_src = self.tokenizer(current_texts_src, truncation=True, padding=True, return_tensors="pt", max_length=512).to(self.device)
                text_embs_src= self.model.encode_text(batch_t_src["input_ids"],
                                                batch_t_src.get('token_type_ids', None),
                                                batch_t_src["attention_mask"])
                src_repr = node_repr[src_idx]
                graph_embs_src = src_repr / src_repr.norm(dim=-1, keepdim=True)
                dst_repr = node_repr[dst_idx]
                graph_embs_dst = dst_repr / dst_repr.norm(dim=-1, keepdim=True)
                text_embs_dst = text_embs_dst / text_embs_dst.norm(dim=-1, keepdim=True)
                text_embs_src = text_embs_src / text_embs_src.norm(dim=-1, keepdim=True)
                out = (graph_embs_src * text_embs_dst).sum(dim=1)
                out2 = (graph_embs_dst * text_embs_src).sum(dim=1)
                out_bidirectional = (out+out2)/2
                logit_scale = self.model.logit_scale.exp()
                len_g = int(len(out)/2)
                text_embs_d = torch.cat((text_embs_src[:len_g], text_embs_dst[len_g:]), dim=0)
                text_embs_s = torch.cat((text_embs_dst[:len_g], text_embs_src[len_g:]), dim=0)

                logits_per_graph, logits_per_text = create_logits(graph_embs_src, text_embs_d, logit_scale)
                loss1 = calculate_loss(logits_per_graph, logits_per_text, criterion)

                logits_per_graph, logits_per_text = create_logits(graph_embs_dst, text_embs_s, logit_scale)
                loss2= calculate_loss(logits_per_graph, logits_per_text, criterion)
                loss3=(loss1+loss2)/2
                labels = batch.edge_label  
                loss =F.binary_cross_entropy_with_logits(out, labels.float().to(self.device)) + loss3
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            graph_grad_norm = 0.0
            text_grad_norm = 0.0
            gg=0
            tt=0
            for name, p in self.model.named_parameters():
                if p .requires_grad:
                    if 'graph_model' in name and 'convs' in name:
                        graph_grad_norm += p.grad.norm(2).item()
                        gg+=1

                    elif 'text_model'in name and 'encoder' in name:
                        text_grad_norm += p.grad.norm(2).item()
                        tt+=1
            print("graph_grad_norm:",graph_grad_norm/gg,text_grad_norm/tt,gg,tt)
            eps = 1e-6
            rho_graph = graph_grad_norm/gg / (text_grad_norm/tt + eps)
            rho_text  = text_grad_norm/tt  / (graph_grad_norm/gg + eps)
            lamda_lr=0.5
            for group in optimizer.param_groups:
                print("group:",group["name"])
                if group["name"] == "text":
                    print("lr :",group["lr"] )
                    group["lr"] =self.tune_lr*rho_graph*lamda_lr  # mu < 1，降低 text 学习率
            progress_bar.write("Session: {} | Epoch: {} | Loss: {:.4f}".format(task, epoch, avg_loss))

            if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                auc,acc_valid, f1_valid, precision, recall = self.evaluate_link(target_graph,val_loader,task)
                progress_bar.write("Session: {} | Epoch: {} | Acc Val: {:.4f} | F1 Val: {:.4f} | Tolerate: {}".format(task, epoch, acc_valid, f1_valid, tolerate))
                if acc_valid > best_acc_valid:
                    tolerate = 0
                    best_acc_valid = acc_valid
                    _save_checkpoint(self.model, optimizer, epoch, checkpoint_path, self.datasets, self.model_name, self.seed)
                else:
                    tolerate += 1
                    if tolerate > self.config['patience']: 
                        break

            progress_bar.set_postfix({
                'Loss': f"{avg_loss:.4f}",
                'Best Valid ACC': f"{best_acc_valid:.4f}",
                'Tolerate': tolerate
            })

            progress_bar.update(1)
        progress_bar.close()
        return avg_loss
    @torch.no_grad()
    def evaluate_graph(self, target_graph,loader, classes, c_descs, dataset_name,task):
        self.model.eval()
        text_inputs = [eval_template[dataset_name].format(c=c) for c in classes]
        text_inputs = [ti + desc for ti, desc in zip(text_inputs, c_descs)]
        batch_t = self.tokenizer(text_inputs, truncation=True, padding=True, return_tensors="pt", max_length=512).to(self.device)
        with torch.no_grad():
            text_embs = self.model.encode_text(batch_t["input_ids"],
                                               batch_t.get('token_type_ids', None),
                                               batch_t["attention_mask"])
            text_embs /= text_embs.norm(dim=-1, keepdim=True)

        all_preds = []
        all_labels = []
        for batch_id in tqdm(loader, desc=f"Evaluating {dataset_name}", leave=False):
            idxs = [batch_i['node_id'] for batch_i in batch_id]
            graphs = [target_graph[i] for i in idxs]
            batch = Batch.from_data_list(graphs).to(self.device)
            if not hasattr(batch, 'pe') or batch.pe is None:
                raise AttributeError("Batch object is missing 'pe' attribute. Please check if parse_target_data correctly added positional encoding.")
                
            with torch.no_grad():
                graph_embs, _ = self.model.encode_graph(batch,task=task)
                graph_embs /= graph_embs.norm(dim=-1, keepdim=True)
                similarity = (100.0 * graph_embs @ text_embs.T)
                predictions = similarity.argmax(dim=1)
                labels=batch.y.cpu()
            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='macro')

        return accuracy,f1

    @torch.no_grad()
    def evaluate_link (self,text_dataset,data_loader,task, classification_threshold=0,classifier=None):
        self.model.eval()  
        ground_truths = []
        preds = []
    
        val_batch_iterator = tqdm(data_loader, desc=f"Validation Starting", leave=False)
        for batch in val_batch_iterator:
            batch = batch.to(self.device)
            node_repr = self.model.graph_model.link(batch,task)
            batch_labels = batch.edge_label.cpu().tolist()  
            global_src = batch.n_id[src_idx]
            global_dst = batch.n_id[dst_idx]
            text=text_dataset.raw_texts
            current_texts_dst = []
            current_texts_src = []
            for label_idx in global_src:
                current_texts_src.append(text[label_idx]) 
            for label_idx in global_dst:
                current_texts_dst.append(text[label_idx])

            batch_t = self.tokenizer(current_texts_src, truncation=True, padding=True, return_tensors="pt", max_length=512).to(self.device)
            text_embs_src = self.model.encode_text(batch_t["input_ids"],
                                        batch_t.get('token_type_ids', None),
                                        batch_t["attention_mask"])
            batch_t_dst = self.tokenizer(current_texts_dst, truncation=True, padding=True, return_tensors="pt", max_length=512).to(self.device)
            text_embs_dst = self.model.encode_text(batch_t_dst["input_ids"],
                                        batch_t_dst.get('token_type_ids', None),
                                        batch_t_dst["attention_mask"])
            graph_embs = node_repr[src_idx] / node_repr[src_idx].norm(dim=-1, keepdim=True)
            graph_embs2 = node_repr[dst_idx] / node_repr[dst_idx].norm(dim=-1, keepdim=True)
            text_embs_src  = text_embs_src / text_embs_src.norm(dim=-1, keepdim=True)
            text_embs_dst  = text_embs_dst / text_embs_dst.norm(dim=-1, keepdim=True)
            graph_embs /= graph_embs.norm(dim=-1, keepdim=True)
            graph_embs2 /= graph_embs2.norm(dim=-1, keepdim=True)
            out = (graph_embs * text_embs_dst).sum(dim=1)
            out2 = (graph_embs2 * text_embs_src).sum(dim=1)
            out_pred=(out + out2) / 2
            ground_truths.append(batch.edge_label)
            preds.append(out_pred)
        
        ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()
        preds = torch.cat(preds, dim=0).cpu().numpy()
        
        pred_labels = (preds > classification_threshold).astype(int)
        acc = accuracy_score(ground_truths, pred_labels)
        f1 = f1_score(ground_truths, pred_labels)
        precision = precision_score(ground_truths, pred_labels)
        recall = recall_score(ground_truths, pred_labels)
        auc = roc_auc_score(ground_truths, preds)
        
        return auc, acc, f1, precision, recall

    def update_DualGPM (self, mat_list):
        print("_cur_task:",self._cur_task,self.total_sessions)
        threshold = (self.lame - self.lamb)*self._cur_task/self.total_sessions + self.lamb
        print ('Threshold: ', threshold) 
        if len(self.feature_list_y) == 0:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold) #+1  
                if r < (activation.shape[0]/2):
                    self.feature_list_y.append(U[:,0:max(r,1)])
                    self.project_type.append('remove')
                else:
                    self.feature_list_y.append(U[:,0:max(r,1)])
                    self.project_type.append('retain')
                self.feature_share.append(U[:,0:max(r,1)])
        else:
            for i in range(len(mat_list)):
                if self.project_type[i] == 'remove':
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    act_hat = activation - np.dot(np.dot(self.feature_list_y[i],self.feature_list_y[i].transpose(1,0)),activation)
                    act_hat_share=np.dot(np.dot(self.feature_share[i],self.feature_share[i].transpose(1,0)),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    U_share,S_share,Vh_share = np.linalg.svd(act_hat_share, full_matrices=False)
                    

                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total
            
                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue

                    sval_hat = (S_share**2).sum()
                    sval_ratio = (S_share**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total

                    r_share = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r_share += 1
                        else:
                            break
                    if r_share == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue
                    Ui=np.hstack((self.feature_list_y[i],U[:,0:r]))  

                    if Ui.shape[1] > Ui.shape[0] :
                        self.feature_list_y[i]=Ui[:,0:Ui.shape[0]]
                    else:
                        self.feature_list_y[i]=Ui
                    self.feature_share[i] = U_share[:,0:r_share]
                  
                else:
                    assert self.project_type[i] == 'retain'
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    act_hat = np.dot(np.dot(self.feature_list_y[i],self.feature_list_y[i].transpose()),activation)
                    U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = sval_hat/sval_total

                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval >= (1-threshold):
                            accumulated_sval -= sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue
                    act_feature = self.feature_list_y[i] - np.dot(np.dot(U[:,0:r],U[:,0:r].transpose(0,1)),self.feature_list_y[i])
                    Ui, Si, Vi = np.linalg.svd(act_feature)
                    self.feature_list_y[i]=Ui[:,:self.feature_list_y[i].shape[1]-r]

        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list_y)):
            if self.project_type[i]=='remove' and (self.feature_list_y[i].shape[1] > (self.feature_list_y[i].shape[0]/2)):
                feature = self.feature_list_y[i]
                U, S, V = np.linalg.svd(feature)
                new_feature = U[:,feature.shape[1]:]
                self.feature_list_y[i] = new_feature
                self.project_type[i] = 'retain'
            elif self.project_type[i]=='retain':
                assert self.feature_list_y[i].shape[1] <= (self.feature_list_y[i].shape[0]/2)
            print ('Layer {} : {}/{} type {}'.format(i+1,self.feature_list_y[i].shape[1], self.feature_list_y[i].shape[0], self.project_type[i]))
        print('-'*40)

class GTAlign(BaseModel):
        
    def __init__(self, task_loader, result_logger, config, checkpoint_path, dataset, model_name, model_path, local_ce, seed, device,task_type,type):
        super(GTAlign, self).__init__(task_loader, result_logger, config, checkpoint_path, dataset, model_name, local_ce, seed, device,task_type,type)

        self.lm_type = config['lm']
        self.lr = float(config['lr'])
        self.weight_decay = float(config['weight_decay'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['lm_type'])
        self.device = device
        self.seed = seed
        self.task_type = task_type
        self.type = type
        self.islora_g = config['islora_g']
        self.islora_t = config['islora_t']
        self.epoch=config['epochs']
        self.config=config
        self.model_name=model_name
        self.n_tasks=self.session_num
        attn_kwargs = {'dropout': 0.0}
        self.model = GraphCLIP(384, 1024, 12, attn_kwargs, text_model='tiny', r=32, islora_g=self.config['islora_g'] , islora_t=self.config['islora_t'] ,n_tasks =self.session_num)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device), strict=False)
        self.model.to(self.device)
        self.task_prototypes = {}

    def get_optimizer(self, model):
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [{'params': params, 'lr': self.lr, 'weight_decay': self.weight_decay},],
            betas=(0.9, 0.95)
        )

        return optimizer

    def train(self,prompter, text_dataset_iso,train_loader,valid_loader,test_loader_isolate,checkpoint_path, classes, c_descs, dataset_name,class_src=0, islora_g=True , islora_t=True ,task =0,type = 'node'):
        prompter.tune(text_dataset_iso,train_loader,valid_loader,test_loader_isolate,checkpoint_path, classes, c_descs, dataset_name,class_src=class_src, islora_g=self.islora_g , islora_t=self.islora_t ,task =task,type_task=type)


    @torch.no_grad()
    def evaluate(self, prompter, text_dataset, test_loader,c_descs,classes,dataset_name,session,class_src=0,type ='node',task=False,task_joint='',pred_task_id=-1,class_range=[0,100]):
        logits_list, preds_list, labels_list = [], [], []
        acc_list = []
        f1_list = []
        if task:
            for j, type_i in enumerate(task_joint):
                print("classes:",classes)
                if type_i == 'node':
                    print("text_datasetaaa:",j,len(text_dataset[j]))
                    acc, f1=prompter.evaluate(text_dataset[j],test_loader[j], classes[j], c_descs[j], dataset_name,session,class_src=class_src,class_range=class_range)
                    acc_list.append(acc)
                    f1_list.append(f1)
                elif type_i == 'edge':
                    print("text_dataset[j]:",text_dataset[j])
                    _,acc, f1,_,_=prompter.evaluate_link(text_dataset[j],test_loader[j],session)
                    acc_list.append(acc)
                    f1_list.append(f1)
                elif type_i == 'graph':
                    acc, f1=prompter.evaluate_graph(text_dataset[j],test_loader[j],classes[j], c_descs[j],dataset_name,session)
                    acc_list.append(acc)
                    f1_list.append(f1)
            acc = sum(acc_list) / len(acc_list) if len(acc_list) > 0 else 0  # 确保列表不为空，避免除零错误
            f1 = sum(f1_list) / len(f1_list) if len(f1_list) > 0 else 0  # 同上
            return acc, f1, acc_list, f1_list
        else:
            if type == 'edge':
                _,acc, f1,_,_=prompter.evaluate_link(text_dataset,test_loader,session)
            elif type == 'node':
                print("nodeclasses:",classes)
                acc, f1 =prompter.evaluate(text_dataset,test_loader, classes, c_descs, dataset_name,session,class_src=class_src,pred_task_id=pred_task_id,class_range=class_range)
            elif type == 'graph':
                acc, f1=prompter.evaluate_graph(text_dataset,test_loader,classes, c_descs,dataset_name,session)
            return acc, f1

    @torch.no_grad()
    def update_task_prototype(self, target_graph, data_loader, task_id):
        all_feats = []

        for batch_id in data_loader:
            idxs = batch_id['node_id'].tolist()
            graphs = [target_graph[i] for i in idxs]

            for g in graphs:
                root_idx = int(g.root_n_index)
                feat = g.x[root_idx]        # [d]
                all_feats.append(feat)

        if len(all_feats) == 0:
            return

        all_feats = torch.stack(all_feats, dim=0)  
        proto = all_feats.mean(dim=0)       
        proto = proto.to(self.device)  
        self.task_prototypes[task_id] = proto


    @torch.no_grad()
    def predict_task_id(
        self,
        target_graph,
        data_loader,
        task
    ):
        """
        Predict task ID in a model-agnostic manner
        using initial node features only.
        """
        feats = []

        for batch_id in data_loader:
            idxs = batch_id['node_id'].tolist()
            graphs = [target_graph[i] for i in idxs]

            for g in graphs:
                root_idx = int(g.root_n_index)
                feat = g.x[root_idx]         
                feats.append(feat)

        if len(feats) == 0:
            return None
        feats = torch.stack(feats, dim=0)     
        query_proto = feats.mean(dim=0)        
        query_proto = F.normalize(query_proto, dim=0)
        query_proto = query_proto.to(self.device)
        best_task = None
        best_sim = -1e9

        for task_id, proto in self.task_prototypes.items():
            proto = proto.to(query_proto.device)
            sim = torch.dot(query_proto, proto)  
            print(f"sim(task={task_id}):", sim.item())

            if sim > best_sim:
                best_sim = sim
                best_task = task_id

        return best_task

        
    def fit(self, iter):
        optimizer = self.get_optimizer(self.model)
        prompter = GraphCLIPPrompter(self.model, self.tokenizer,  self.device ,self.config,self.model_name,self.dataset,random_seed=iter,tune_lr=self.lr,tune_epochs=self.epoch,tune_batch_size=self.config['batch_size'],weight_decay=self.weight_decay,n_tasks=self.n_tasks)
        text_dataset_iso_all=[]
        classes_task={}
        c_descs_task={}
        classes_all=[]
        c_descs_all=[]
        text_dataset_iso_node=None
        text_dataset_joint_all=[]
        for curr_session in range(self.session_num):
            if curr_session != 0:
                _reload_best_model(self.model, self.checkpoint_path, self.dataset, self.model_name, self.seed)
            if self.task_type == "FSNTIL":
                datasets = self.dataset.split('+')
                dataset_name = datasets[curr_session]
                data_all,class_src, class_dst, text_dataset_iso, text_dataset_joint, train_loader, valid_loader, test_loader_isolate, test_loader_joint = self.task_loader.get_task(curr_session,task_type=self.task_loader.type[curr_session],CL_type=self.task_loader.cl_type)
                if self.task_loader.type[curr_session] == 'node':
                    text_dataset_iso = build_ego_graphs_from_big_data(text_dataset_iso.data,hop=1)
                    text_dataset_iso_node=text_dataset_iso
                    text_dataset_joint_all.append(text_dataset_iso_node)
                elif self.task_loader.type[curr_session] == 'graph':
                    text_dataset_iso=text_dataset_iso.data
                    text_dataset_joint_all.append(text_dataset_iso)
                else:
                    text_dataset_joint_all.append(text_dataset_iso)
                type=self.task_loader.type[:curr_session+1]
                class_desc = pd.read_csv(f"./LLM4GCL/data/categories/{dataset_name}_categories.csv")
                c_descs = class_desc['description'].tolist()
                classes = class_desc["name"].tolist()
                classes_all.append(classes)
                c_descs_all.append(c_descs)
                self.train(prompter,text_dataset_iso,train_loader,valid_loader,test_loader_isolate,self.checkpoint_path, classes, c_descs, dataset_name, islora_g=self.islora_g , islora_t=self.islora_t ,task =curr_session,type = self.task_loader.type[curr_session])
                curr_acc_test_isolate, curr_f1_test_isolate = self.evaluate(prompter, text_dataset_iso, test_loader_isolate, c_descs,classes,dataset_name,curr_session,type = self.task_loader.type[curr_session])
                curr_acc_test_joint, curr_f1_test_joint,acc_list, f1_list= self.evaluate(prompter, text_dataset_joint_all, test_loader_joint, c_descs_all,classes_all,dataset_name,curr_session,type = self.task_loader.type[curr_session],task=True,task_joint=self.task_loader.type[:curr_session+1])

                print("Session: {} | Iso. Acc Test: {:.4f} | Iso. F1 Test: {:.4f}".format(curr_session, curr_acc_test_isolate, curr_f1_test_isolate))
                print("Session: {} | Jot. Acc Test: {:.4f} | Jot. F1 Test: {:.4f}".format(curr_session, curr_acc_test_joint, curr_f1_test_joint))
                print("acc_list:",acc_list)
                self.result_logger.add_new_results(acc_list, curr_acc_test_joint)

            else:
                dataset_name = self.dataset
                data_all,class_src, class_dst, text_dataset_iso, text_dataset_joint, train_loader, valid_loader, test_loader_isolate, test_loader_joint = self.task_loader.get_task(curr_session)
                text_dataset_iso = build_ego_graphs_from_big_data(data_all,hop=1)
                text_dataset_joint = build_ego_graphs_from_big_data(data_all,hop=1)
                class_desc = pd.read_csv(f"./LLM4GCL/data/categories/{dataset_name}_categories.csv")
                c_descs = class_desc['description'].tolist()
                classes = class_desc["name"].tolist()
                classes_cur=classes[class_src: class_dst]
                c_descs_cur=c_descs[class_src:class_dst]
                classes_joint=classes[:class_dst]
                c_descs_joint=c_descs[: class_dst]
                classes_task[curr_session] = classes[class_src: class_dst]
                c_descs_task[curr_session] = c_descs[class_src: class_dst]
                if curr_session ==0:
                    self.train(prompter,text_dataset_iso,train_loader,valid_loader,test_loader_isolate,self.checkpoint_path, classes_cur, c_descs_cur, dataset_name,islora_g=self.islora_g , islora_t=self.islora_t ,task =curr_session)
                    self.update_task_prototype(text_dataset_iso,train_loader,curr_session)
                else:
                    self.train(prompter,text_dataset_iso,train_loader,valid_loader,test_loader_isolate,self.checkpoint_path, classes_cur, c_descs_cur, dataset_name,class_src=class_src ,islora_g=self.islora_g , islora_t=self.islora_t ,task =curr_session)
                    self.update_task_prototype(text_dataset_iso,train_loader,curr_session)
                    pred_task_id = self.predict_task_id(text_dataset_iso,test_loader_isolate,curr_session)
                    classes_cur= classes_task[pred_task_id]
                    c_descs_cur=c_descs_task[pred_task_id]
                curr_acc_test_isolate, curr_f1_test_isolate = self.evaluate(prompter, text_dataset_iso, test_loader_isolate, classes_cur,c_descs_cur,dataset_name,curr_session,class_src=class_src)
                curr_acc_test_joint, curr_f1_test_joint = self.evaluate(prompter, text_dataset_joint, test_loader_joint, c_descs_joint,classes_joint,dataset_name,curr_session)
                print("Session: {} | Iso. Acc Test: {:.4f} | Iso. F1 Test: {:.4f}".format(curr_session, curr_acc_test_isolate, curr_f1_test_isolate))
                print("Session: {} | Jot. Acc Test: {:.4f} | Jot. F1 Test: {:.4f}".format(curr_session, curr_acc_test_joint, curr_f1_test_joint))
                acc_list = []
                for s in range(curr_session):
                    data_all,class_src, _, text_dataset_iso, _, _, _, test_loader_isolate, _ = self.task_loader.get_task(s)
                    text_dataset_iso = build_ego_graphs_from_big_data(data_all,hop=1)
                    pred_task_id = self.predict_task_id(text_dataset_iso,test_loader_isolate,s )
                    classes_cur= classes_task[pred_task_id]
                    c_descs_cur=c_descs_task[pred_task_id]
                    prev_acc_test_isolate, prev_f1_test_isolate = self.evaluate(prompter, text_dataset_iso, test_loader_isolate, c_descs_cur,classes_cur,dataset_name,curr_session,class_src=class_src,pred_task_id=curr_session)
                    acc_list.append(prev_acc_test_isolate)
                acc_list.append(curr_acc_test_isolate)
                print("acc_list:",acc_list)
                self.result_logger.add_new_results(acc_list, curr_acc_test_joint)

        return self.result_logger