import os
import json
import torch
import numpy as np

from textwrap import shorten


# Genreal Prompts for Inference
def get_genreal_prompts(dataset, label_text):
    prompts_dict = {
        'cora': f"Question: Which of the following sub-categories of AI does this paper belong to? Here are the {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}. Reply only one category that you think this paper might belong to. Only reply the category phrase without any other explanation words.\n\nAnswer: ",
        'citeseer': f"Question: Which of the following theme does this paper belong to? Here are the {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}. Reply only one category that you think this paper might belong to. Only reply the category full name I give you without any other words.\n\nAnswer: ",
        'wikics': f"Question: Which of the following branch of Computer science does this Wikipedia-based dataset belong to? Here are the {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}. Reply only one category that you think this paper might belong to. Only reply the category full name without any other words.\n\nAnswer: ",
        'photo': f"Question: Which of the following categories does this photo item belong to? Here are the {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}. Reply only one category that you think this item might belong to. Only reply the category name I give of the category without any other words.\n\nAnswer: ",
        'products': f"Question: Which of the following categories of Amazon products does this item belong to? Here are the {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}. Reply only one category that you think this item might belong to. Only reply the category name without any other explanation words.\n\nAnswer: ",
        'arxiv_23': f"Question: Which of the following arXiv CS sub-categories does this dataset belong to? Here are the {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}. Reply only one category that you think this paper might belong to. Only reply the category name I given without any other words, please don't use your own words.\n\nAnswer:",
        'arxiv': f"Question: Which of the following arXiv CS sub-categories does this dataset belong to? Here are the {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}. Reply only one category that you think this paper might belong to. Only reply the category phrase without any other explanation words.\n\nAnswer: "
    }

    return prompts_dict[dataset]


# Prompts for LLaGA
def get_LLaGA_prompts(dataset, label_text):
    SYSTEM_PROMPT = "You are a helpful language and graph assistant. You can understand the graph content provided by the user and assist with the node classification task by outputting the label that is most likely to apply to the node."

    LLaGA_descriptions_dict = {
        "cora": f"Given a node-centered graph: <graph>, each node represents a paper, we need to classify the center node into {len(label_text)} classes: {', '.join([label.lower() for label in label_text])}, please tell me which class the center node belongs to?",
        "citeseer": f"Given a node-centered graph: <graph>, each node represents a paper, we need to classify the center node into {len(label_text)} classes: {', '.join([label.lower() for label in label_text])}, please tell me which class the center node belongs to?",
        "wikics": f"Given a node-centered graph: <graph>, each node represents an entity, we need to classify the center node into {len(label_text)} classes: {', '.join([label.lower() for label in label_text])}, please tell me which class the center node belongs to?",
        "photo": f"Given a node-centered graph: <graph>, each node represents an item, we need to classify the center node into {len(label_text)} classes: {', '.join([label.lower() for label in label_text])}, please tell me which class the center node belongs to?",
        "products": f"Given a node-centered graph: <graph>, each node represents an item, we need to classify the center node into {len(label_text)} classes: {', '.join([label.lower() for label in label_text])}, please tell me which class the center node belongs to?",
        "arxiv_23": f"Given a node-centered graph: <graph>, we need to classify the center node into {len(label_text)} classes: {len(label_text)} classes: {', '.join([label.lower() for label in label_text])}, please tell me which class the center node belongs to?",
        "arxiv": f"Given a node-centered graph: <graph>, we need to classify the center node into {len(label_text)} classes: {len(label_text)} classes: {', '.join([label.lower() for label in label_text])}, please tell me which class the center node belongs to?",
    }

    return SYSTEM_PROMPT, LLaGA_descriptions_dict[dataset]


# Prompts for GraphGPT
MATCHING_TEMPLATES = {
    "academic_network": "Given a sequence of graph tokens <graph> that constitute a subgraph of a citation graph, where the first token represents the central node of the subgraph, and the remaining nodes represent the first or second order neighbors of the central node. Each graph token contains the title and abstract information of the paper at this node. Here is a list of paper titles: {{paper_titles}}. Please reorder the list of papers according to the order of graph tokens (i.e., complete the matching of graph tokens and papers).",
    "social_network": "Given a sequence of graph tokens <graph> that constitute a subgraph of a social network, where the first token represents the central node (user) of the subgraph, and the remaining nodes represent the first or second order neighbors of the central node. Each graph token contains the profile description of the user represented by this node. Here is a list of user profile descriptions: {{user_profiles}}. Please reorder the list of users according to the order of the graph tokens (i.e., complete the matching of graph tokens and users).",
    "ecommerce_network": "Given a sequence of graph tokens <graph> that constitute a subgraph of an e-commerce network, where the first token represents the central node (item) of the subgraph, and the remaining nodes represent the first or second order neighbors of the central node. Each graph token contains the comment of the item represented by this node. Here is a list of item comments: {{item_comments}}. Please reorder the list of items according to the order of the graph tokens (i.e., complete the matching of graph tokens and items). "
} 

GraphGPT_DESC = {
    "cora": 'Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, with the following information: \n{{raw_text}}\n Question: Which of the following specific research does this paper belong to: {{label_names}}. Directly give the full name of the most likely category of this paper.',
    "citeseer": 'Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, with the following information: \n{{raw_text}}\n Question: Which of the following specific research does this paper belong to: {{label_names}}. Directly give the full name of the most likely category of this paper.',
    "wikics": 'Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, with the following information: \n{{raw_text}}\n Question: Which of the following specific research does this paper belong to: {{label_names}}. Directly give the full name of the most likely category of this paper.', 
    "photo": "Given an e-commerce network: \n<graph>\nwhere the 0-th node is the target item, with the following information: \n{{raw_text}}\n Question: Which of the following categories does this item belong to: {{label_names}}. Directly give the full name of the most likely category of this photo item. ",
    "products": "Given an e-commerce network: \n<graph>\nwhere the 0-th node is the target item, with the following information: \n{{raw_text}}\n Question: Which of the following categories does this product belong to: {{label_names}}. Directly give the full name of the most likely category of this product. ",
    "arxiv_23": 'Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, with the following information: \n{{raw_text}}\n Question: Which of the following arXiv CS sub-category does this paper belong to: {{label_names}}. Directly give the most likely arXiv CS sub-categories of this paper.', 
    "arxiv": 'Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, with the following information: \n{{raw_text}}\n Question: Which of the following arXiv CS sub-category does this paper belong to: {{label_names}}. Directly give the most likely arXiv CS sub-categories of this paper.', 

}
def get_edge_prediction_prompts(
    edge_pairs,                 # Tensor [2, N] for edge pairs
    edge_labels,                # Tensor [N], 0/1 for edge labels
    graph_data,
    text,
    label_index, class_num,
    dataset="default",                # "ego" | "neighbors"
    hop=[20, 20],
    mode="ego", 
    include_label=False, 
    max_node_text_len=256,
):
    """
    Generate instruction prompts for link prediction:
    Given two nodes, determine whether there exists an edge between them.
    """

    # -------- helper --------
    def get_subgraph(node_idx, edge_index, hop):
        current = torch.tensor([node_idx], device=edge_index.device)
        all_hops = []
        for _ in hop:
            mask = torch.isin(edge_index[0], current) | torch.isin(edge_index[1], current)
            new_nodes = torch.unique(
                torch.cat([edge_index[0][mask], edge_index[1][mask]]),
            )
            diff = new_nodes[~torch.isin(new_nodes, current)]
            all_hops.append(diff.cpu().tolist())
            current = torch.unique(torch.cat([current, new_nodes]))
        return all_hops

    def node_token(dataset):
        if dataset in ["products", "computer", "history"]:
            return "Product id: "
        if dataset == "photo":
            return "Photo id: "
        if dataset == "wikics":
            return "Page id: "
        return "Paper id: "

    # -------- normalize input --------
    if isinstance(edge_pairs, torch.Tensor):
        edge_pairs = edge_pairs.cpu().tolist()
    if isinstance(edge_labels, torch.Tensor):
        edge_labels = edge_labels.cpu().tolist()

    assert edge_pairs[0] and edge_pairs[1] and len(edge_pairs[0]) == len(edge_labels), \
        "edge_pairs and edge_labels must have the same length."

    # -------- prompt templates --------
    prefix = (
        f"You are a good graph reasoner. "
        f"Given two nodes from the {dataset} dataset, answer the question.\n"
    )

    question = (
        "Given two nodes in the graph, determine whether there exists an edge between them.\n"
        "Answer with either 'no' or 'yes'.\n"
    )

    label_text = ["no", "yes"]
    node_id_token = node_token(dataset)

    instruction_list = []

    # -------- build prompts --------
    for i, (u, v) in enumerate(zip(edge_pairs[0], edge_pairs[1])):
        u, v = int(u), int(v)
        y = int(edge_labels[i])

        context = prefix

        # Node A
        context += "\n## Node A:\n"
        context += f"{node_id_token}{u}\n"
        if mode != "pure":
            context += f"Text: {shorten(text[u], max_node_text_len)}\n"

        # Node B
        context += "\n## Node B:\n"
        context += f"{node_id_token}{v}\n"
        if mode != "pure":
            context += f"Text: {shorten(text[v], max_node_text_len)}\n"

        # Optional neighbors
        if mode == "neighbors":
            u_ngh = get_subgraph(u, graph_data.edge_index, hop)
            v_ngh = get_subgraph(v, graph_data.edge_index, hop)

            context += "\n## Neighbors of Node A (partial):\n"
            for h, nodes in enumerate(u_ngh):
                nodes = nodes[: hop[h]]
                for n in nodes:
                    context += f"{node_id_token}{n}\n"

            context += "\n## Neighbors of Node B (partial):\n"
            for h, nodes in enumerate(v_ngh):
                nodes = nodes[: hop[h]]
                for n in nodes:
                    context += f"{node_id_token}{n}\n"

        instruction_list.append({
            "Context": context,
            "Question": question,
            "Answer": label_text[y]
        })

    return instruction_list
def get_subgraph(node_idx, edge_index, hop):
    current_nodes = torch.tensor([node_idx])
    all_hop_neighbors = []

    for _ in range(len(hop)):
        mask = torch.isin(edge_index[0], current_nodes) | torch.isin(edge_index[1], current_nodes)
        new_nodes = torch.unique(torch.cat((edge_index[0][mask], edge_index[1][mask])))
        diff_nodes_set = set(new_nodes.numpy()) - set(current_nodes.numpy())
        diff_nodes = torch.tensor(list(diff_nodes_set))  
        all_hop_neighbors.append(diff_nodes.tolist())
        current_nodes = torch.unique(torch.cat((current_nodes, new_nodes)))

    return all_hop_neighbors

def get_structure_prompts(node_index, label_idx, label_text, graph_data, text, all_hop_neighbors, hop, include_label, dataset, mode, max_node_text_len=256):

    prompt_str = ""
    if dataset == 'products':
        node_id_token = f'Product id: '
    elif dataset == 'photo':
        node_id_token = f'Photo id: '
    elif dataset == 'wikics':
        node_id_token = f'Page id: '
    else:
        node_id_token = f'Paper id: '

    for h in range(0, len(hop)):
        neighbors_at_hop = all_hop_neighbors[h]
        neighbors_at_hop = np.unique(np.array(neighbors_at_hop))
        np.random.shuffle(neighbors_at_hop)

        if h == 0:
            neighbors_at_hop = neighbors_at_hop[:hop[0]]
        else:
            neighbors_at_hop = neighbors_at_hop[:hop[1]]

        if len(neighbors_at_hop) > 0:
            if dataset == 'product':
                prompt_str += f"\nKnown neighbor products at hop {h + 1} (partial, may be incomplete):\n"
            elif dataset == 'photo':
                prompt_str += f"\nKnown neighbor photos at hop {h + 1} (partial, may be incomplete):\n"
            elif dataset == 'wikics':
                prompt_str += f"\nKnown neighbor pages at hop {h + 1} (partial, may be incomplete):\n"
            else:
                prompt_str += f"\nKnown neighbor papers at hop {h + 1} (partial, may be incomplete):\n"

            for neighbor_idx in neighbors_at_hop:
                neighbor_text = shorten(text[neighbor_idx], width=max_node_text_len, placeholder="...")
                if mode != 'pure':
                    prompt_str += f"\n{node_id_token}{neighbor_idx}\nText: {neighbor_text}\n"
                else:
                    prompt_str += f"\n{node_id_token}{neighbor_idx}"

                if include_label and neighbor_idx in label_idx: # Notice label leakage !!!
                    label = label_text[graph_data.y[neighbor_idx].item()]
                    prompt_str += f"Label: {label}\n"

    return prompt_str
def get_instruction_prompts_graph(
    node_ids,  # List or Tensor of node ids
    text_dataset,  # List of Data objects (each representing a graph)
    raw_texts,  # List of raw texts for each node
    label_index,  # List of labels or class indices for each node
    class_num,  # Number of classes
    dataset="default",  # Dataset name
    mode="neighbors",  # "ego" | "neighbors" | "pure"
    include_label=False,  # Whether to include labels
    max_node_text_len=256,  # Maximum length for node text
):
    """
    Generate instruction prompts for link prediction or node classification.
    For each graph in `text_dataset`, generate instructions for each node pair.
    """
    # -------- helper function to handle node text --------
    def node_token(dataset):
        if dataset in ["products", "computer", "history"]:
            return "Product id: "
        if dataset == "photo":
            return "Photo id: "
        if dataset == "wikics":
            return "Page id: "
        return "Paper id: "
    label_text_list = text_dataset[0].label_texts
    # print("label_text:",label_text_list)
    label_text = label_text_list
    prefix = (
        f"You are a good graph reasoner. "
        f"Given a graph from the {dataset} dataset, answer the question.\n"
    )
    # print("node_ids:",node_ids)
    question_prompts_dict = {
        'photo': f"Please predict which of the following categories does this photo item belong to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}",}
    question = question_prompts_dict[dataset] + '\nDo not provide your reasoning.\n Answer:\n\n'
    node_id_token = node_token(dataset)

    instruction_list = []
    selected_data = [text_dataset[i] for i in node_ids]
    prompt_str = ""
    for i, graph_data in enumerate(selected_data):
        graph_id = i  # Index of the graph
        y = label_index[i]
        # print("graph_data:",graph_data)
        context = prefix
        node_id=node_ids[i]
        prefix_prompt = prefix
        context = prefix_prompt + '\n## Target node:\n' + str(i)
        raw_text = shorten(graph_data.raw_texts, width=max_node_text_len, placeholder="...")
        context = f"{context}Text: {raw_text}\n"
        edge_index=graph_data.edge_index
        # Adding the relationships between nodes (neighbors)
        # We need to get the neighbors of each node in this graph based on edge_index
        context += "\n## Node Relationships:\n"
        for node_id in range(len(graph_data.x)):  # 遍历每个节点
            node_text = graph_data.raw_text_node[node_id]  # 获取节点特征（文本）
            
            # 获取该节点的所有邻居
            neighbor_ids = edge_index[1][edge_index[0] == node_id].tolist()  # 找到目标节点为当前节点的所有边
            neighbors_text = [graph_data.x[neighbor_id] for neighbor_id in neighbor_ids]  # 获取邻居节点的文本

            # 构建该节点的文本描述
            prompt_str += f"Known neighbor photos at hop 1: "  
            prompt_str += f"\nPhoto id: {node_id}\nText: {node_text}\n"  # 当前节点的文本描述
            # prompt_str += f"Known neighbor photos at hop 1: "  # 这里可以调整 hop 层数
            # for idx, neighbor_id in enumerate(neighbor_ids):
            #     neighbor_text = neighbors_text[idx]  # 获取邻居节点的文本描述
            #     prompt_str += f"\nPhoto id: {neighbor_id}\nText: {neighbor_text}"
            # prompt_str += f"\n{node_id_token}{neighbor_idx}\nText: {neighbor_text}\n"
            # all_hop_neighbors = get_subgraph(node_id, graph_data.edge_index, hop=[20, 20])
            # prompt_str = get_structure_prompts(node_id, label_index, label_text, graph_data, text, all_hop_neighbors, hop, include_label, dataset, mode, max_node_text_len)
        context += prompt_str
        answer = label_text[graph_data.y]
        # print("answer:",answer)
        # Adding question
        instruction_list.append({
            "Context": context,
            "Question": question,
            "Answer": answer
        })
        # print("instruction_list:",instruction_list)

    # print("instruction_list:",instruction_list[0])
    return instruction_list

# Instructions for SimGCL
def get_instruction_prompts(node_index, graph_data, text, label_index, class_num, dataset, hop=[20, 20], mode="ego", include_label=False, max_node_text_len=256,task='node'):

    def get_subgraph(node_idx, edge_index, hop):
        current_nodes = torch.tensor([node_idx])
        all_hop_neighbors = []

        for _ in range(len(hop)):
            mask = torch.isin(edge_index[0], current_nodes) | torch.isin(edge_index[1], current_nodes)
            new_nodes = torch.unique(torch.cat((edge_index[0][mask], edge_index[1][mask])))
            diff_nodes_set = set(new_nodes.numpy()) - set(current_nodes.numpy())
            diff_nodes = torch.tensor(list(diff_nodes_set))  
            all_hop_neighbors.append(diff_nodes.tolist())
            current_nodes = torch.unique(torch.cat((current_nodes, new_nodes)))

        return all_hop_neighbors
    
    def get_structure_prompts(node_index, label_idx, label_text, graph_data, text, all_hop_neighbors, hop, include_label, dataset, mode, max_node_text_len=256):

        prompt_str = ""
        if dataset == 'products':
            node_id_token = f'Product id: '
        elif dataset == 'photo':
            node_id_token = f'Photo id: '
        elif dataset == 'wikics':
            node_id_token = f'Page id: '
        else:
            node_id_token = f'Paper id: '

        for h in range(0, len(hop)):
            neighbors_at_hop = all_hop_neighbors[h]
            neighbors_at_hop = np.unique(np.array(neighbors_at_hop))
            np.random.shuffle(neighbors_at_hop)

            if h == 0:
                neighbors_at_hop = neighbors_at_hop[:hop[0]]
            else:
                neighbors_at_hop = neighbors_at_hop[:hop[1]]

            if len(neighbors_at_hop) > 0:
                if dataset == 'product':
                    prompt_str += f"\nKnown neighbor products at hop {h + 1} (partial, may be incomplete):\n"
                elif dataset == 'photo':
                    prompt_str += f"\nKnown neighbor photos at hop {h + 1} (partial, may be incomplete):\n"
                elif dataset == 'wikics':
                    prompt_str += f"\nKnown neighbor pages at hop {h + 1} (partial, may be incomplete):\n"
                else:
                    prompt_str += f"\nKnown neighbor papers at hop {h + 1} (partial, may be incomplete):\n"

                for neighbor_idx in neighbors_at_hop:
                    neighbor_text = shorten(text[neighbor_idx], width=max_node_text_len, placeholder="...")
                    if mode != 'pure':
                        prompt_str += f"\n{node_id_token}{neighbor_idx}\nText: {neighbor_text}\n"
                    else:
                        prompt_str += f"\n{node_id_token}{neighbor_idx}"

                    if include_label and neighbor_idx in label_idx: # Notice label leakage !!!
                        label = label_text[graph_data.y[neighbor_idx].item()]
                        prompt_str += f"Label: {label}\n"

        return prompt_str

    label_text_list = graph_data.label_texts
    # print("label_text:",label_text_list)
    label_text = label_text_list[ :class_num]
    if task=='edge':
        label_text = ['no', 'yes']
    prefix_prompts_dict = {
        'neighbors': f'You are a good graph reasoner. Given a graph description from {dataset} dataset, understand the structure and answer the question.\n',
        'ego': f'You are a good graph reasoner. Given target node information from {dataset} dataset, answer the question.\n',
        'pure': f'You are a good graph reasoner. Given a graph description from {dataset} dataset, understand the structure and answer the question.\n',
    }

    question_prompts_dict = {
        'cora': f"Please predict which of the following sub-categories of AI does this paper belong to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}",
        'citeseer': f"Please predict which of the following theme does this paper belong to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}",
        'wikics': f"Please predict which branch of Computer Science this Wikipedia-based dataset belongs to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}",
        'photo': f"Please predict which of the following categories does this photo item belong to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}",
        'products': f"Please predict which of the following categories does this target item from Amazon belong to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}",
        'arxiv_23': f"Please predict the most appropriate original arxiv identifier for the paper. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}.",
        'arxiv': f"Please predict the most appropriate original arxiv identifier for the paper. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}.",
        'computer': f"Please predict which of the following categories this Amazon computer-related product belongs to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}",  
        'history': f"Please predict which of the following categories this historical-related product belongs to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}",
        'cora+citeseer': f"Please predict which of the following research themes does this paper belong to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}",
        'default': f"Given two nodes in the graph, determine whether there exists an edge between them. Answer with either 'no' or 'yes'.",
        'bbbp': "Given a molecular graph, determine whether the molecule can penetrate the blood-brain barrier. Answer with 'yes' or 'no'.",
        'instagram': f"Please predict which type of Instagram user this account belongs to. Choose from the following {len(label_text)} categories: {', '.join([label.lower() for label in label_text])}.",
 }


    instruction_list = []
    for id in node_index:

        if dataset == 'products':
            node_id = f'Product id: {id}\n'
        elif dataset == 'photo':
            node_id = f'Photo id: {id}\n'
        elif dataset == 'wikics':
            node_id = f'Page id: {id}\n'
        else:
            node_id = f'Paper id: {id}\n'
                
                
        prefix_prompt = prefix_prompts_dict[mode]
        context = prefix_prompt + '\n## Target node:\n' + node_id
        if task=='edge':
            dataset=='default'
            question = question_prompts_dict[dataset] + '\nDo not provide your reasoning.\n Answer:\n\n'
            answer = label_text[int(graph_data.edge_label[id].item())]
        else:
            question = question_prompts_dict[dataset] + '\nDo not provide your reasoning.\n Answer:\n\n'
            answer = label_text[graph_data.y[id].item()]

        if mode == 'neighbors':
            raw_text = shorten(text[id], width=max_node_text_len, placeholder="...")
            context = f"{context}Text: {raw_text}\n"

            all_hop_neighbors = get_subgraph(id, graph_data.edge_index, hop)
            prompt_str = get_structure_prompts(id, label_index, label_text, graph_data, text, all_hop_neighbors, hop, include_label, dataset, mode, max_node_text_len)
            context += prompt_str

        elif mode == 'ego':
            raw_text = shorten(text[id], width=max_node_text_len, placeholder="...")
            context = f"{context}Text: {raw_text}\n"

        elif mode == 'pure':
            all_hop_neighbors = get_subgraph(id, graph_data.edge_index, hop)
            prompt_str = get_structure_prompts(id, label_index, label_text, text, graph_data, all_hop_neighbors, hop, include_label, dataset, mode, max_node_text_len)
            context += prompt_str

        else:
            raise ValueError('Invalid mode!')

        instruction_list.append({
            "Context": context,
            "Question": question,
            "Answer": answer
        })

    # print("instruction_list node:",instruction_list[0])
    return instruction_list