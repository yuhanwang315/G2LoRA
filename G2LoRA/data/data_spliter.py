import copy
import torch
import random
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import Subset, DataLoader
from torch_geometric.loader import DataLoader,DataListLoader
from collections import defaultdict
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T
random.seed(42)

def update_edge_data_with_train_idx(text_dataset_isolate, train_idx):

    updated_edge_label_index_p = [train_idx[0],train_idx[1]]
    updated_edge_label_index_n= [train_idx[2],train_idx[3]]
    selected_nodes = list(set(train_idx[0] + train_idx[1] + train_idx[2] + train_idx[3]))

    train_idx_0 = torch.tensor(train_idx[0], dtype=torch.long)  
    train_idx_1 = torch.tensor(train_idx[1], dtype=torch.long)  
    train_idx_2 = torch.tensor(train_idx[2], dtype=torch.long) 
    train_idx_3 = torch.tensor(train_idx[3], dtype=torch.long) 

    updated_edge_label_index = torch.cat([train_idx_0, train_idx_2], dim=0), torch.cat([train_idx_1, train_idx_3], dim=0)
    updated_edge_label_index = torch.stack(updated_edge_label_index, dim=0)
    updated_edge_label = torch.tensor([1] * len(train_idx[0]) + [0] * len(train_idx[2]))
    updated_x = text_dataset_isolate.x[selected_nodes]
    updated_y = text_dataset_isolate.y[selected_nodes]  
    updated_raw_texts = [text_dataset_isolate.raw_texts[i] for i in selected_nodes]
    updated_data = Data(
        x=text_dataset_isolate.x,
        edge_index=text_dataset_isolate.edge_index,
        y=text_dataset_isolate.y,
        label_texts=text_dataset_isolate.label_texts,
        edge_label=updated_edge_label,  
        edge_label_index=updated_edge_label_index 
    )
    return updated_data

def select_graphs_from_dataset(text_dataset_isolate, train_idx):
    graphs_data = []
    
    
    selected_graphs = [text_dataset_isolate.data[i] for i in train_idx]  
    

    for idx, graph in zip(train_idx, selected_graphs):
        graph_data = {
            "data": graph,  
            "node_id": idx   
        }
        graphs_data.append(graph_data)  
    
    return graphs_data

class TaskLoader():
        
    def __init__(self, batch_size, text_dataset, cl_type, task_type, base_session, novel_session, ways, sessions, base_train_shots, train_shots, valid_shots, test_shots,type):
        self.batch_size = batch_size
        print("batch_size:",batch_size)
        self.text_dataset = text_dataset  
        self.data = text_dataset.data

        self.id_by_class = text_dataset.id_by_class
       
        self.cl_type = cl_type
      
        self.task_type = task_type
        self.type=type
        self.label_nums=[]
        if self.task_type in  ['FSNTIL']: 
            for j ,type_j in enumerate(self.type):
                if type_j == 'node':
                    self.label_nums.append(len(self.data[j].label_texts))
                elif type_j == 'edge':
                    self.label_nums.append(2)
                elif type_j == 'graph':
                    self.label_nums.append(len(self.data[j].label_texts))

        else:
            self.label_num = self.data.y.max().item() + 1


        self.base_session = base_session
        self.novel_session = novel_session
        self.ways = ways
        self.sessions = sessions
        self.base_train_shots = base_train_shots
        self.train_shots = train_shots
        self.valid_shots = valid_shots
        self.test_shots = test_shots
        # Task Split
        data_all,node_idx_per_class, train_idx_per_task, valid_idx_per_task, test_idx_per_task_isolate, test_idx_per_task_joint, dataset_per_task_isolate, dataset_per_task_joint = self._split_data()
        
        self.node_idx_per_class = node_idx_per_class
        self.train_idx_per_task = train_idx_per_task
        self.valid_idx_per_task = valid_idx_per_task
        self.test_idx_per_task_isolate = test_idx_per_task_isolate
        self.test_idx_per_task_joint = test_idx_per_task_joint
        self.dataset_per_task_isolate = dataset_per_task_isolate
        self.dataset_per_task_joint = dataset_per_task_joint
        self.test_loader_joint_all = []
        self.text_dataset_joint_all=[]
        self.type_joint =[]
        self.data_all=data_all

    def get_joint_task(self, ):
        all_train_idx, all_valid_idx, all_test_idx = [], [], []
        for task_id in range(self.sessions):
            all_train_idx.extend(self.train_idx_per_task[task_id])
            all_valid_idx.extend(self.valid_idx_per_task[task_id])
            all_test_idx.extend(self.test_idx_per_task_isolate[task_id])

        text_dataset = self.dataset_per_task_joint[self.sessions - 1]

        train_dataset = Subset(text_dataset, all_train_idx)
        val_dataset = Subset(text_dataset, all_valid_idx)
        test_dataset = Subset(text_dataset, all_test_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        class_num = self.data.y[all_train_idx].max().item() + 1

        return class_num, text_dataset, train_loader, valid_loader, test_loader

    def get_task(self, task_id, subset = -1,task_type='node',CL_type='class'):
        import psutil
        import os
        def print_mem(tag):
            p = psutil.Process(os.getpid())
            print(f"[{tag}] RSS = {p.memory_info().rss / 1024**3:.2f} GB")

        print("task_type:",task_type)
        if task_id >= self.sessions:
            raise f"Task id {task_id} is larger than total number of tasks {self.sessions} !"
        train_idx = self.train_idx_per_task[task_id]
        valid_idx = self.valid_idx_per_task[task_id]
        test_idx_isolate = self.test_idx_per_task_isolate[task_id]
        test_idx_joint = self.test_idx_per_task_joint[task_id]
       
        if subset != -1:
            train_idx = self._stratified_sample(train_idx, self.data.y[train_idx], subset)
            valid_idx = self._stratified_sample(valid_idx, self.data.y[valid_idx], subset)

        text_dataset_isolate = self.dataset_per_task_isolate[task_id]
        text_dataset_joint = self.dataset_per_task_joint[task_id]
        if task_type == 'edge':
            transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
            neighbors=[0,0]
            train_dataset = update_edge_data_with_train_idx(text_dataset_isolate, train_idx)
            val_dataset = update_edge_data_with_train_idx(text_dataset_isolate, valid_idx)
            test_dataset_isolate = update_edge_data_with_train_idx(text_dataset_isolate, test_idx_isolate)
            train_file = '/data/root/LLM4GCL-main/LLM4GCL/data/edge_train_dataset.pt'
            val_file = '/data/root/LLM4GCL-main/LLM4GCL/data/edge_val_dataset.pt'
            test_file = '/data/root/LLM4GCL-main/LLM4GCL/data/edge_test_dataset_isolate.pt'

            if os.path.exists(train_file):
                train_dataset = torch.load(train_file)
                print("Loaded train dataset from file.")
            else:
                train_dataset = transform(train_dataset)
                torch.save(train_dataset, train_file)
                print("Transformed and saved train dataset.")

            if os.path.exists(val_file):
                val_dataset = torch.load(val_file)
                print("Loaded validation dataset from file.")
            else:
                val_dataset = transform(val_dataset)
                torch.save(val_dataset, val_file)
                print("Transformed and saved validation dataset.")

            if os.path.exists(test_file):
                test_dataset_isolate = torch.load(test_file)
                print("Loaded test dataset from file.")
            else:
                test_dataset_isolate = transform(test_dataset_isolate)
                torch.save(test_dataset_isolate, test_file)
                print("Transformed and saved test dataset.")

            print(f"Positional encoding calculation complet")
            # print(train_dataset)
            train_loader = LinkNeighborLoader(data=train_dataset, num_neighbors=neighbors,edge_label_index=train_dataset.edge_label_index,edge_label=train_dataset.edge_label,batch_size=self.batch_size,shuffle=False,)
            valid_loader = LinkNeighborLoader(data=val_dataset, num_neighbors=neighbors,edge_label_index=val_dataset.edge_label_index,edge_label=val_dataset.edge_label,batch_size=self.batch_size,shuffle=False,)
            test_loader_isolate = LinkNeighborLoader(data=test_dataset_isolate, num_neighbors=neighbors,edge_label_index=test_dataset_isolate.edge_label_index,edge_label=test_dataset_isolate.edge_label,batch_size=self.batch_size,shuffle=False,)
            self.test_loader_joint_all.append(test_loader_isolate)
            if  len(self.type_joint)==0:
                self.text_dataset_joint_all.append([text_dataset_joint])
            else:
                self.text_dataset_joint_all.append(text_dataset_joint)

            if  len(self.type_joint)==0:
                self.type_joint.append([task_type])
            else:

                type_joint = self.type_joint[-1]
                type_joint.append(task_type)
                self.type_joint.append(type_joint)

        elif task_type == 'graph':
            train_dataset = select_graphs_from_dataset(text_dataset_isolate, train_idx)
            val_dataset = select_graphs_from_dataset(text_dataset_isolate, valid_idx)
            test_dataset_isolate = select_graphs_from_dataset(text_dataset_isolate, test_idx_isolate)

            train_loader = DataListLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            valid_loader = DataListLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader_isolate = DataListLoader(test_dataset_isolate, batch_size=self.batch_size, shuffle=False)
            self.test_loader_joint_all.append(test_loader_isolate)
            if  len(self.type_joint)==0:
                self.text_dataset_joint_all.append([text_dataset_joint])
            else:
                self.text_dataset_joint_all.append(text_dataset_joint)
            if  len(self.type_joint)==0:
                self.type_joint.append([task_type])
            else:
                type_joint = self.type_joint[-1]
                type_joint.append(task_type)
                self.type_joint.append(type_joint)
                

        else:
            train_dataset = Subset(text_dataset_isolate, train_idx)
            print("train_idx:",len(train_idx))
            val_dataset = Subset(text_dataset_isolate, valid_idx)
            test_dataset_isolate = Subset(text_dataset_isolate, test_idx_isolate)
            test_dataset_joint = Subset(text_dataset_joint, test_idx_joint)
            sample = train_dataset[0]
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            valid_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader_isolate = DataLoader(test_dataset_isolate, batch_size=self.batch_size, shuffle=False)
            test_loader_joint = DataLoader(test_dataset_joint, batch_size=self.batch_size, shuffle=False)
            if CL_type == 'task':
                self.test_loader_joint_all.append(test_loader_isolate)
                if  len(self.type_joint)==0:
                    self.text_dataset_joint_all.append([text_dataset_joint])
                else:
                    self.text_dataset_joint_all.append(text_dataset_joint)
                if  len(self.type_joint)==0:
                    self.type_joint.append([task_type])
                else:
                    type_joint = self.type_joint[-1]
                    type_joint.append(task_type)
                    self.type_joint.append(type_joint)

        if self.task_type == 'FSNCIL':
            if task_id == 0: # Base Session
                class_src, class_dst = 0, self.base_session
            else:
                class_src, class_dst = self.base_session + (task_id - 1) * self.ways, min(self.base_session + task_id * self.ways, self.label_num)
        if self.task_type == 'NCIL':
            class_src, class_dst = task_id * self.ways, min((task_id + 1) * self.ways, self.label_num)
        if self.task_type == 'FSNDIL' :
            class_src = sum(self.ways[:task_id])
            class_dst = class_src + self.ways[task_id]

        if self.task_type == 'FSNTIL':
            class_src, class_dst = 0 , self.ways[task_id]
        if CL_type == 'task':
            test_loader_joint = self.test_loader_joint_all
            text_dataset_joint=self.text_dataset_joint_all

        return self.data_all,class_src, class_dst, text_dataset_isolate, text_dataset_joint, train_loader, valid_loader, test_loader_isolate, test_loader_joint

    def _split_data(self): 
        node_idx_per_class = []
        train_idx_per_task = []
        valid_idx_per_task = []
        test_idx_per_task_isolate = []
        test_idx_per_task_joint = []
        dataset_per_task_isolate = []  
        dataset_per_task_joint = []  
        prev_dataset=[]

        if self.task_type == 'FSNTIL':

            all_class = []

            for j ,dataset_type in enumerate(self.type):
                dataset = self.data[j]  

                if dataset_type == 'node':
                    node_labels = dataset.y  
                    class_list = node_labels.unique(sorted=True).tolist()
                    all_class.append(class_list)

                elif dataset_type == 'edge':
                    edge_labels = dataset.y  
                    class_list = [0,1]
                    all_class.append(class_list)

                elif dataset_type == 'graph':
                    graph_labels = dataset.y  
                    class_list = graph_labels.unique(sorted=True).tolist()
                    all_class.append(class_list)
                
            train_shots, valid_shots, test_shots = self.train_shots, self.valid_shots, self.test_shots
            train_num, valid_num, test_num = train_shots, valid_shots, test_shots

            for kk, t in enumerate(self.type):
                if self.type[kk] == 'node':
                    node_idx_curr_class = []
                    train_idx_curr_task = []
                    valid_idx_curr_task = []
                    test_idx_curr_task = []
                    curr_task_class_idx = all_class[kk]
                    for cla in curr_task_class_idx:
                        node_idx = self.id_by_class[kk][cla]
                        node_idx_curr_class.extend(node_idx)
                        node_num = len(node_idx)
                        train_shots, valid_shots, test_shots = self.train_shots, self.valid_shots, self.test_shots
                        if node_num < (train_shots + valid_shots + test_shots):
                            train_num, valid_num, test_num = int(node_num * 0.5), int(node_num * 0.1), int(node_num * 0.4)
                            if (train_num + valid_num + test_num) > node_num:
                                train_num -= train_num + valid_num + test_num - node_num
                        else:
                            train_num, valid_num, test_num = train_shots, valid_shots, test_shots
                        if train_num>train_shots:
                            train_num=train_shots
                        random.shuffle(node_idx)
                        train_idx_curr_task.extend(node_idx[: train_num])
                        valid_idx_curr_task.extend(node_idx[train_num : train_num + valid_num])
                        test_idx_curr_task.extend(node_idx[train_num + valid_num: train_num + valid_num + test_num])
                    node_idx_per_class.append(node_idx_curr_class)
                    train_idx_per_task.append(train_idx_curr_task)
                    valid_idx_per_task.append(valid_idx_curr_task)
                    test_idx_per_task_isolate.append(test_idx_curr_task)
                    test_idx_per_task_joint.append(test_idx_curr_task)

                    text_dataset = copy.deepcopy(self.text_dataset)
                    text_dataset.data = self.data[kk]
                    text_dataset.id_by_class = self.id_by_class[kk]
                    text_dataset.raw_texts = self.text_dataset.raw_texts[kk]
                    text_dataset.label_texts = self.text_dataset.label_texts[kk]
                    curr_dataset = text_dataset 
                    if len(prev_dataset) == 0:
                        prev_dataset = text_dataset
                    else:
                        prev_dataset = [prev_dataset,curr_dataset]
                    dataset_per_task_isolate.append(curr_dataset)
                    dataset_per_task_joint.append(prev_dataset)

                elif t == 'edge':
                    edge_idx_curr_class = []
                    train_idx_curr_task = []
                    valid_idx_curr_task = []
                    test_idx_curr_task = []

                    transform = RandomLinkSplit(is_undirected=True, num_val=0.1, num_test=0.1, add_negative_train_samples=True)
                    train_data, val_data, test_data = transform(self.data[kk])
                    train_edge_index = train_data.edge_label_index  
                    train_edge_labels = train_data.edge_label 
                    test_edge_index = test_data.edge_label_index  
                    test_edge_labels = test_data.edge_label 
                    val_edge_index = val_data.edge_label_index  
                    val_edge_labels =val_data.edge_label 

                    train_shots, valid_shots, test_shots = self.train_shots, self.valid_shots, self.test_shots

                    train_num, valid_num, test_num = train_shots, valid_shots, test_shots

                    positive_edges_train = train_edge_index[:, train_edge_labels == 1]
                    negative_edges_train = train_edge_index[:, train_edge_labels == 0]
                    positive_edges_test = test_edge_index[:, test_edge_labels == 1]
                    negative_edges_test = test_edge_index[:, test_edge_labels == 0]
                    positive_edges_val = val_edge_index[:, val_edge_labels == 1]
                    negative_edges_val = val_edge_index[:, val_edge_labels == 0]


                    train_shots, valid_shots, test_shots = self.train_shots, self.valid_shots, self.test_shots
                    train_idx_curr_task = []
                    valid_idx_curr_task = []
                    test_idx_curr_task = []
                    train_idx_curr_task.extend(positive_edges_train[:, :train_shots].tolist())
                    valid_idx_curr_task.extend(positive_edges_val[:, :valid_shots].tolist())
                    test_idx_curr_task.extend(positive_edges_test[:, :test_shots].tolist())


                    train_idx_curr_task.extend(negative_edges_train[:, :train_shots].tolist())
                    valid_idx_curr_task.extend(negative_edges_val[:, :valid_shots].tolist())
                    test_idx_curr_task.extend(negative_edges_test[:, :test_shots].tolist())

                    node_idx_per_class.append([])
                    train_idx_per_task.append(train_idx_curr_task)
                    valid_idx_per_task.append(valid_idx_curr_task)
                    
                    test_idx_per_task_isolate.append(test_idx_curr_task)
                    test_idx_per_task_joint.append(test_idx_curr_task)

                    curr_dataset = train_data  
                    if len(prev_dataset) == 0:
                        prev_dataset = curr_dataset
                    else:
                        prev_dataset = [prev_dataset,curr_dataset]
                    dataset_per_task_isolate.append(curr_dataset)
                    dataset_per_task_joint.append(prev_dataset)

                elif t == 'graph':
                    graph_idx_curr_class = []
                    train_idx_curr_task = []
                    valid_idx_curr_task = []
                    test_idx_curr_task = []
                    curr_task_class_idx = all_class[kk]
                    for cla in curr_task_class_idx:
                        graph_idx = self.id_by_class[kk][cla]
                        graph_idx_curr_class.extend(graph_idx)
                        graph_num = len(graph_idx)
                        train_shots, valid_shots, test_shots = self.train_shots, self.valid_shots, self.test_shots
                        
                        if graph_num < (train_shots + valid_shots + test_shots):
                            train_num, valid_num, test_num = int(graph_num * 0.5), int(graph_num * 0.1), int(graph_num * 0.4)
                            if (train_num + valid_num + test_num) > graph_num:
                                train_num -= train_num + valid_num + test_num - graph_num
                        else:
                            train_num, valid_num, test_num = train_shots, valid_shots, test_shots
                        if train_num>train_shots:
                            train_num=train_shots
                        random.shuffle(graph_idx)
                        train_idx_curr_task.extend(graph_idx[:train_num])
                        valid_idx_curr_task.extend(graph_idx[train_num:train_num + valid_num])
                        test_idx_curr_task.extend(graph_idx[train_num + valid_num:train_num + valid_num + test_num])
                    node_idx_per_class.append(graph_idx_curr_class)
                    train_idx_per_task.append(train_idx_curr_task)
                    valid_idx_per_task.append(valid_idx_curr_task)
                    test_idx_per_task_isolate.append(test_idx_curr_task)
                    test_idx_per_task_joint.append(test_idx_curr_task)
                    curr_dataset = self.data[kk]   
                    if len(prev_dataset) == 0:
                        prev_dataset = curr_dataset
                    else:
                        if isinstance(prev_dataset, list):
                            new_prev_dataset = prev_dataset + [curr_dataset]
                        else:
                            prev_dataset = [prev_dataset, curr_dataset]
                    dataset_per_task_isolate.append(curr_dataset)
                    dataset_per_task_joint.append(new_prev_dataset)

            return self.data, node_idx_per_class, train_idx_per_task, valid_idx_per_task, test_idx_per_task_isolate, test_idx_per_task_joint, dataset_per_task_isolate, dataset_per_task_joint

        else:
            all_class = self.data.y.unique(sorted=True).tolist()
            offset=0
            offset_r=0
            for i in range(self.sessions):
                node_idx_curr_class = []
                train_idx_curr_task = []
                valid_idx_curr_task = []
                test_idx_curr_task = []


                if self.task_type == 'FSNCIL':
                    if i == 0: # Base Session
                        curr_task_class_idx = all_class[ : self.base_session]
                    else:
                        curr_task_class_idx = all_class[self.base_session + (i - 1) * self.ways : min(self.base_session + i * self.ways, self.label_num)]
                if self.task_type == 'NCIL':
                    curr_task_class_idx = all_class[i * self.ways : min((i + 1) * self.ways, self.label_num)]
                if self.task_type == 'FSNDIL' :
                    offset_r += self.ways[i]
                    offset_r_class=offset_r
                    offset_l_class=offset
                    curr_task_class_idx = all_class[offset :offset_r]
                    offset += self.ways[i]              
                for cla in curr_task_class_idx:
                    node_idx = self.id_by_class[cla]
                    node_idx_curr_class.extend(node_idx)
                    node_num = len(node_idx)

                    if self.task_type == 'FSNCIL':
                        if i == 0: # Base Session
                            train_shots, valid_shots, test_shots = self.base_train_shots, self.valid_shots, self.test_shots
                        else:
                            train_shots, valid_shots, test_shots = self.train_shots, self.valid_shots, self.test_shots
                    if self.task_type == 'NCIL':
                        train_shots, valid_shots, test_shots = self.train_shots, self.valid_shots, self.test_shots
                    if self.task_type == 'FSNDIL' :
                        train_shots, valid_shots, test_shots = self.train_shots, self.valid_shots, self.test_shots
                    if node_num < (train_shots + valid_shots + test_shots):
                        train_num, valid_num, test_num = int(node_num * 0.5), int(node_num * 0.1), int(node_num * 0.4)
                        if (train_num + valid_num + test_num) > node_num:
                            train_num -= train_num + valid_num + test_num - node_num
                    else:
                        train_num, valid_num, test_num = train_shots, valid_shots, test_shots
                    # print("train_num:",train_num)
                    if train_num>train_shots:
                        train_num=train_shots
                    print("train_num:",train_num)
                    random.shuffle(node_idx)

                    train_idx_curr_task.extend(node_idx[: train_num])
                    
                    valid_idx_curr_task.extend(node_idx[train_num : train_num + valid_num])
                    test_idx_curr_task.extend(node_idx[train_num + valid_num: train_num + valid_num + test_num])
        
                node_idx_per_class.append(node_idx_curr_class)
                train_idx_per_task.append(train_idx_curr_task)
                valid_idx_per_task.append(valid_idx_curr_task)
                test_idx_per_task_isolate.append(test_idx_curr_task)

                if i != 0:
                    test_idx_per_task_joint.append(test_idx_per_task_joint[-1] + test_idx_curr_task)
                else:
                    test_idx_per_task_joint.append(test_idx_curr_task)  

                if self.task_type == 'FSNCIL':
                    if i == 0: # Base Session
                        curr_dataset = self._adjust_graph(all_class[ : self.base_session])
                        prev_dataset = self._adjust_graph(all_class[ : self.base_session])
                    else:
                        curr_dataset = self._adjust_graph(all_class[self.base_session + (i - 1) * self.ways : min(self.base_session + i * self.ways, self.label_num)])
                        prev_dataset = self._adjust_graph(all_class[: min(self.base_session + i * self.ways, self.label_num)])  #多个任务数据合并的图
                if self.task_type == 'NCIL':
                    curr_dataset = self._adjust_graph(all_class[i * self.ways : min((i + 1) * self.ways, self.label_num)])
                    prev_dataset = self._adjust_graph(all_class[: min((i + 1) * self.ways, self.label_num)])

                if self.task_type == 'FSNDIL' :
                    curr_dataset = self._adjust_graph(all_class[offset_l_class :offset_r_class])
                    prev_dataset = self._adjust_graph(all_class[: offset_r])


                dataset_per_task_isolate.append(curr_dataset)
                dataset_per_task_joint.append(prev_dataset) 

            return self.data,node_idx_per_class, train_idx_per_task, valid_idx_per_task, test_idx_per_task_isolate, test_idx_per_task_joint, dataset_per_task_isolate, dataset_per_task_joint
    def _adjust_graph(self, class_id):
        node_mask = torch.zeros(len(self.data.y), dtype=torch.bool)
        observe_idx = []
        for cls in class_id:
            observe_idx.extend(self.id_by_class[cls])

        node_mask[observe_idx] = True
        edge_index = self.data.edge_index
        mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        new_edge_index = edge_index[:, mask]

        text_dataset = copy.copy(self.text_dataset)  
        text_dataset.data = copy.copy(self.text_dataset.data)
        text_dataset.data.edge_index = new_edge_index
        return text_dataset

    def _adjust_graph_add(self, dataset, class_id,i):
        node_mask = torch.zeros(len(dataset.y), dtype=torch.bool)
        observe_idx = []
        for cls in class_id:
            observe_idx.extend(self.id_by_class[i][cls])

        node_mask[observe_idx] = True
        edge_index = dataset.edge_index

        mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        new_edge_index = edge_index[:, mask]

        text_dataset = copy.deepcopy(self.text_dataset)
        text_dataset.data[i].edge_index = new_edge_index

        return text_dataset

    def _stratified_sample(self, indices, labels, n_samples):
        label_to_indices = defaultdict(list)
        for idx, label in zip(indices, labels):
            label_to_indices[label.item()].append(idx)

        unique_labels = list(label_to_indices.keys())
        label_counts = [len(label_to_indices[l]) for l in unique_labels]
        proportions = np.array(label_counts) / sum(label_counts)
        samples_per_label = (proportions * n_samples).astype(int)

        samples_per_label = np.maximum(samples_per_label, 1)
        total = sum(samples_per_label)

        while total > n_samples:
            max_idx = np.argmax(samples_per_label)
            if samples_per_label[max_idx] > 1:
                samples_per_label[max_idx] -= 1
                total -= 1

        sampled_indices = []
        for i, label in enumerate(unique_labels):
            sample_size = samples_per_label[i]
            sampled = random.sample(label_to_indices[label], min(sample_size, len(label_to_indices[label])))
            sampled_indices.extend(sampled)

        return sampled_indices
