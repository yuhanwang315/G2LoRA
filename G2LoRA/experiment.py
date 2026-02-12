import torch
import numpy as np

from LLM4GCL.models import *
import LLM4GCL.models as models
from LLM4GCL.common.metric import CLMetric
from LLM4GCL.data import TextDataset, TaskLoader
from LLM4GCL.common.utils import seed_everything


class Experiment(object):
    def __init__(self, args, config):
        self.dataset = args.dataset
        self.data_path = args.data_path

        self.model_type = args.model_type
        self.model_name = args.model
        self.model_path = args.model_path
        self.ckpt_path = args.ckpt_path
        
        self.cl_type = args.cl_type
        self.task_type = args.task_type

        self.base_session = args.base_session
        self.novel_session = args.novel_session
        self.ways = args.ways
        self.sessions = args.sessions
        self.base_train_shots = args.base_train_shots
        self.train_shots = args.train_shots
        self.valid_shots = args.valid_shots
        self.test_shots = args.test_shots
        self.type = args.type.split(',')

        self.local_ce = args.local_ce
        self.ntrail = args.ntrail
        self.device = torch.device("cuda:" + str(args.gpu_num) if torch.cuda.is_available() else "cpu")

        # Load Dataset
        self.text_dataset = TextDataset(dataset=self.dataset, data_path=self.data_path, CL_type=self.cl_type, type=self.type)

        # Load Model Configs
        self.config = config

        # Genreate CL Tasks
        self.task_loader = TaskLoader(batch_size=self.config['batch_size'], 
                                      text_dataset=self.text_dataset, 
                                      cl_type=self.cl_type, 
                                      task_type=self.task_type, 
                                      base_session=self.base_session, 
                                      novel_session=self.novel_session, 
                                      ways=self.ways, 
                                      sessions=self.sessions, 
                                      base_train_shots=self.base_train_shots,
                                      train_shots=self.train_shots, 
                                      valid_shots=self.valid_shots, 
                                      test_shots=self.test_shots,
                                      type=self.type )

        
    def run(self, ):
        assert self.ntrail <= len(self.config['seed']), f"repetition num is larger than the length of seed list!"
        avg_acc_iso_list, avg_fgt_iso_list, avg_acc_jot_list, last_acc_jot_list = [], [], [], []
        for iter in range(0,self.ntrail):
            seed = self.config['seed'][iter]
            seed_everything(seed)
            # Model Initialization
            result_logger = CLMetric()
            import psutil, os

            def print_mem(tag):
                p = psutil.Process(os.getpid())
                print(f"[{tag}] RSS = {p.memory_info().rss / 1024**3:.2f} GB")
            print_mem("AAAbefore ego")      
            if self.model_name in ['BareGNN', 'EWC', 'LwF', 'cosine', 'TEEN', 'TPP']:
                model = getattr(models, self.model_name)(
                    task_loader=self.task_loader, 
                    result_logger=result_logger, 
                    config=self.config, 
                    checkpoint_path=self.ckpt_path, 
                    dataset=self.dataset, 
                    model_name=self.model_name, 
                    local_ce=self.local_ce,
                    seed=seed, 
                    device=self.device,
                    task_type=self.task_type,
                    type=self.type,
                    )
            elif self.model_name in ['BERT', 'RoBERTa','G2P2', 'LLaMA', 'SimpleCIL', 'LM_emb', 'GraphPrompter', 'ENGINE', 'LLaGA', 'GraphGPT', 'SimGCL','GTAlign' ,'GTAlign_SDlora']:
                model = getattr(models, self.model_name)(
                    task_loader=self.task_loader, 
                    result_logger=result_logger, 
                    config=self.config, 
                    checkpoint_path=self.ckpt_path, 
                    dataset=self.dataset, 
                    model_name=self.model_name, 
                    model_path=self.model_path,
                    local_ce=self.local_ce,
                    seed=seed, 
                    device=self.device,
                    task_type=self.task_type,
                    type=self.type
                )
            else:
                raise ValueError(f'Unsupported model {self.model_name}!')
            print_mem("mid ego") 
            self.model = model
            print_mem("mid2 ego") 
            result_logger = self.model.fit(iter)
            print_mem("mid2 ego") 
            avg_acc_iso, avg_fgt_iso, avg_acc_jot, last_acc_jot = result_logger.get_results()
            del model 
            import gc 
            gc.collect()
            print_mem("AAAafter ego")      
            print(f"Iso. | Avg ACC: {avg_acc_iso:.4f} | Avg FGT: {avg_fgt_iso:.4f}")
            print(f"Jot. | Avg ACC: {avg_acc_jot:.4f} | Last ACC: {last_acc_jot:.4f}")

            avg_acc_iso_list.append(avg_acc_iso)
            avg_fgt_iso_list.append(avg_fgt_iso)
            avg_acc_jot_list.append(avg_acc_jot)
            last_acc_jot_list.append(last_acc_jot)

        avg_acc_iso_list = np.array(avg_acc_iso_list)
        avg_fgt_iso_list = np.array(avg_fgt_iso_list)
        avg_acc_jot_list = np.array(avg_acc_jot_list)
        last_acc_jot_list = np.array(last_acc_jot_list)

        avg_acc_iso_mean = np.mean(avg_acc_iso_list)
        avg_fgt_iso_mean = np.mean(avg_fgt_iso_list)
        avg_acc_jot_mean = np.mean(avg_acc_jot_list)
        last_acc_jot_mean = np.mean(last_acc_jot_list)

        avg_acc_iso_std = np.std(avg_acc_iso_list)
        avg_fgt_iso_std = np.std(avg_fgt_iso_list)
        avg_acc_jot_std = np.std(avg_acc_jot_list)
        last_acc_jot_std = np.std(last_acc_jot_list)

        print(f"--------------------------------------------")
        print(f"Finish !")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset}")
        print(f"CL Type: {self.cl_type}")
        print(f"Task Type: {self.task_type}")
        print(f"Iso. | Avg ACC: {avg_acc_iso_mean:.4f} ± {avg_acc_iso_std:.4f} | Avg FGT: {avg_fgt_iso_mean:.4f} ± {avg_fgt_iso_std:.4f}")
        print(f"Jot. | Avg ACC: {avg_acc_jot_mean:.4f} ± {avg_acc_jot_std:.4f} | Last ACC: {last_acc_jot_mean:.4f} ± {last_acc_jot_std:.4f}")
        print(f"--------------------------------------------")
        print(f"Iso. | Avg ACC: {avg_acc_iso_mean * 100:.2f} ± {avg_acc_iso_std * 100:.2f} | Avg FGT: {avg_fgt_iso_mean * 100:.2f} ± {avg_fgt_iso_std * 100:.2f}")
        print(f"Jot. | Avg ACC: {avg_acc_jot_mean * 100:.2f} ± {avg_acc_jot_std * 100:.2f} | Last ACC: {last_acc_jot_mean * 100:.2f} ± {last_acc_jot_std * 100:.2f}")
        print(f"--------------------------------------------")
        return avg_acc_iso_mean, avg_fgt_iso_mean, avg_acc_jot_mean, last_acc_jot_mean
