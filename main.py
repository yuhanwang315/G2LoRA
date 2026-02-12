import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import yaml
import argparse

from LLM4GCL.experiment import Experiment
from LLM4GCL.common.utils import load_config, merge_params, update_config, select_hyperparameters


model_dict = {
    'GNN': ['BareGNN', 'EWC', 'LwF', 'cosine', 'TEEN', 'TPP'],
    'LM': ['RoBERTa', 'BERT', 'LLaMA', 'SimpleCIL'], 
    'GLM': ['LM_emb', 'GraphPrompter', 'ENGINE', 'LLaGA', 'GraphGPT', 'SimGCL','GTAlign','GTAlign_SDlora','G2P2'], 
}

exp_settings = {
    'FSNCIL': {
        'base_session': {'photo': 4, 'computer': 4, 'history': 4},
        'novel_session': {'photo': 8, 'computer': 6, 'history': 8},
        'ways': {'photo': 4, 'computer': 3, 'history': 4},
        'sessions': {'photo': 3, 'computer': 3, 'history': 3},
        'base_train_shots': {'photo': 10, 'computer': 10, 'history': 10},
        'train_shots': {'photo': 10, 'computer': 10, 'history': 10},
        'valid_shots': {'photo': 50, 'computer': 50, 'history': 50},
        'test_shots': {'photo': 400, 'computer': 800, 'history': 400},
    },
    'FSNTIL': {
        'sessions': {'cora+wikics+photo': 3},
        'ways':  {'cora+wikics+photo': [7,2,12]},
        'train_shots': {'cora+wikics+photo': 10},
        'valid_shots': {'cora+wikics+photo': 50},
        'test_shots': {'cora+wikics+photo': 100},
    },
    'FSNDIL' : {
        'sessions': {'cora+citeseer+wikics': 3},
        'ways': {'cora+citeseer+wikics': [7,6,10]},
        'train_shots': {'cora+citeseer+wikics': 10},
        'valid_shots': {'cora+citeseer+wikics': 50},
        'test_shots': {'cora+citeseer+wikics': 100},
    }
}
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', 
                        type=str, 
                        default='cora', 
                        choices=['cora',  'cora+wikics+photo','cora+citeseer+wikics+photo+history+computer','citeseer','cora+wikics+bbbp','wikics+cora+bbbp', 'wikics', 'wikics+instagram+bbbp','photo', 'cora+citeseer+wikics+photo+history+computer', 'products', 'arxiv_23', 'arxiv','computer','cora+citeseer','history','instagram+wikics+bbbp','cora+citeseer+wikics','bbbp+instagram+wikics'], 
                        help='the name of TAG dataset')
    parser.add_argument('--data_path', type=str, default='./LLM4GCL/data/', help='the path of TAG dataset')
    parser.add_argument('--type', 
                        type=str, 
                        default='node',  # 以逗号分隔的字符串
                        help='the type of task')

    # Model
    parser.add_argument('--model_type', 
                        type=str, 
                        default='LM', 
                        choices=['GNN', 'LM', 'GLM'], 
                        help='Specify the type of model to use. '
                            ' "GNN": Use only Graph Neural Network (GNN) for training and inference. '
                            ' "LM": Use only Language Model (LM) for training and inference. '
                            ' "GLM": Combine Graph Neural Network or Graph and Language Model (LM) into a unified model.')
    parser.add_argument('--model', type=str, default='GPT', help='the name of model, must match with the model_type')
    parser.add_argument('--model_path', type=str, default='./LLM4GCL/model/', help='the path to load pre-trained models')
    parser.add_argument('--ckpt_path', type=str, default='./LLM4GCL/ckpt/', help='the path to store best model weights')

    # Settings
    parser.add_argument('--cl_type', type=str, default='class', choices=['class','domain','task'], help='The type of CL. E.g., class is for class incremental learning')
    parser.add_argument('--task_type', type=str, default='NCIL', choices=['NCIL', 'FSNCIL','FSNDIL','FSNTIL'], help='The type of continual tasks.')

    # Training
    parser.add_argument('--local_ce', default=False, action='store_true')
    parser.add_argument('--ntrail', type=int, default=1, help='repetition count of experiments')
    parser.add_argument('--gpu_num', type=int, default=0, help='the selected GPU number')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    # Tuning
    parser.add_argument('--hyperparam_search', default=False, action='store_true')
    parser.add_argument('--search_type', type=str, default='grid', choices=['grid', 'random'])
    parser.add_argument('--num_samples', type=int, default=10)

    args = parser.parse_args()
  
    assert args.model in model_dict[args.model_type], f"Model type '{args.model_type}' does not support model '{args.model}'."
    # assert args.split_ratio[0] + args.split_ratio[1] + args.split_ratio[2] <= 1, f"The sum of split ratio is larger than 1."
    args.config_path = './configs/{}/{}.yaml'.format(args.model_type, args.model)
    config = load_config(args.config_path)

    if args.task_type == 'FSNCIL':
        args.base_session = exp_settings[args.task_type]['base_session'][args.dataset]
        args.novel_session = exp_settings[args.task_type]['novel_session'][args.dataset]
        args.base_train_shots = exp_settings[args.task_type]['base_train_shots'][args.dataset]
    elif args.task_type == 'FSNTIL':
        args.base_session = None
        args.novel_session = None
        args.base_train_shots = None
    elif args.task_type == 'FSNDIL':
        args.base_session = None
        args.novel_session = None
        args.base_train_shots = None
    args.ways = exp_settings[args.task_type]['ways'][args.dataset]
    args.sessions = exp_settings[args.task_type]['sessions'][args.dataset]
    args.train_shots = exp_settings[args.task_type]['train_shots'][args.dataset]
    print("args.train_shots :",args.train_shots )
    args.valid_shots = exp_settings[args.task_type]['valid_shots'][args.dataset]
    args.test_shots = exp_settings[args.task_type]['test_shots'][args.dataset]

    if args.hyperparam_search:

        if 'default' not in config or 'search_space' not in config:
            raise ValueError("Hyperparameter search requires the default and search_space sections in the config file")

        default_params = config['default']
        search_space = config['search_space']
        selected_params = select_hyperparameters(search_space, args.search_type, args.num_samples)
        
        best_performance = -float('inf')
        best_params = None
        best_metric = None
        
        for params in selected_params:

            current_params = merge_params(default_params, params)
            exp = Experiment(args, current_params)
            avg_acc_iso_mean, avg_fgt_iso_mean, avg_acc_jot_mean, last_acc_jot_mean = exp.run()

            if avg_acc_iso_mean + avg_fgt_iso_mean> best_performance:
                best_performance = avg_acc_iso_mean + avg_fgt_iso_mean
                best_params = params
                best_params = params
                best_metric = {
                    'Iso. Avg ACC': "{:.4f}".format(avg_acc_iso_mean),
                    'Iso. Avg FGT': "{:.4f}".format(avg_fgt_iso_mean),
                    'Jot. Avg ACC': "{:.4f}".format(avg_acc_jot_mean),
                    'Jot. Last ACC': "{:.4f}".format(last_acc_jot_mean)
                }
        
        if best_params is not None and best_metric is not None:
            update_config(args.config_path, args.dataset, best_params, best_metric)
                
    else:
        if 'default' not in config:
            raise ValueError("The config file must contain the default section")

        final_params = config['default'].copy()
        if 'best_' + args.dataset in config and config['best_' + args.dataset]:
            final_params = merge_params(final_params, config['best_' + args.dataset])

        exp = Experiment(args, final_params)
        exp.run()
