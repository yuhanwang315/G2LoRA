import argparse

class Arguments:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        # Dataset
        self.parser.add_argument('--dataset', type=str, help="dataset name", default='cora')
        self.parser.add_argument('--source_data', type=str, help="dataset name", default='pubmed')
        self.parser.add_argument('--target_data', type=str, help="dataset name", default='history')
        self.parser.add_argument('--dataname', type=str, help="dataset name", default='history')
        # 修改 argparse 中的参数类型
        self.parser.add_argument('--islora_g', type=int, help="Whether to use LoRA for graph", default=0)
        self.parser.add_argument('--islora_t', type=int, help="Whether to use LoRA for text", default=0)



        # Model configuration256
        self.parser.add_argument('--ckpt', type=str, help="the name of checkpoint", default='pretrained_graphclip')
        self.parser.add_argument('--layer_num', type=int, help="the number of encoder's layers", default=2)
        self.parser.add_argument('--hidden_size', type=int, help="the hidden size", default=64)
        self.parser.add_argument('--dropout', type=float, help="dropout rate", default=0.5)
        self.parser.add_argument('--activation', type=str, help="activation function", default='relu',
                                 choices=['relu', 'elu', 'hardtanh', 'leakyrelu', 'prelu', 'rrelu'])
        # self.parser.add_argument('--use_bn', action='store_true', help="use BN or not")
        self.parser.add_argument('--last_activation', action='store_true', help="the last layer will use activation function or not")
        self.parser.add_argument('--model', type=str, help="model name", default='GNN',
                                 choices=['GNN'])
        self.parser.add_argument('--norm', type=str, help="the type of normalization, id denotes Identity(w/o norm), bn is batchnorm, ln is layernorm", default='id',
                                 choices=['id', 'bn', 'ln'])
        # self.parser.add_argument('--encoder', type=str, help="the type of encoder", default='GCN_Encoder',
        #                          choices=['GCN_Encoder', 'GAT_Encoder', 'SAGE_Encoder', 'GIN_Encoder', 'MLP_Encoder', 'GCNII_Encoder'])
        # Training settings
        self.parser.add_argument('--optimizer', type=str, help="the kind of optimizer", default='adam',
                                 choices=['adam', 'sgd', 'adamw', 'nadam', 'radam'])
        self.parser.add_argument('--lr', type=float, help="learning rate", default=1e-5) 
        self.parser.add_argument('--weight_decay', type=float, help="weight decay", default=1e-5)
        self.parser.add_argument('--epochs', type=int, help="training epochs", default=30)
        self.parser.add_argument('--batch_size', type=int, help="the batch size", default=256)
        self.parser.add_argument('--noise_ratio', type=float, help="the ratio of noise", default=0)
        self.parser.add_argument('--seed', type=int, help="random seed for reproducibility", default=8)
        self.parser.add_argument('--warmup_ratio', type=float, help="ratio of warmup steps", default=0.1)
        self.parser.add_argument('--lr_step_size', type=int, help="step size for StepLR scheduler", default=10)
        self.parser.add_argument('--lr_gamma', type=float, help="gamma for StepLR scheduler", default=0.5)
        self.parser.add_argument('--patience', type=int, help="patience for early stopping", default=5)

        # softclip
        self.parser.add_argument('--temperature', type=float, help="the temperature of softmax", default=0.07)
        self.parser.add_argument('--beta', type=float, help="the weight of soft label", default=0.2)
        self.parser.add_argument('--lambda_val', type=float, help="the weight of negative samples", default=0)
        self.parser.add_argument('--mu', type=float, help="the weight of clip loss", default=1.0)
        self.parser.add_argument('--original_weight', type=float, help="the weight of original loss", default=0)
        self.parser.add_argument('--softclip_weight', type=float, help="the weight of softclip loss", default=1.0)

        # DPO parameters
        self.parser.add_argument('--alpha', type=float, help="dynamic adjustment coefficient for DPO", default=0.5)
        self.parser.add_argument('--momentum', type=float, help="momentum for moving averages in DPO", default=0.9)
        self.parser.add_argument('--filter_ratio', type=float, help="initial ratio of samples to keep in DPO", default=0.9)
        self.parser.add_argument('--align_weight', type=float, help="weight for alignment loss in DPO", default=1.0)
        self.parser.add_argument('--theta_init', type=float, help="initial value for theta in DPO", default=0.5)
        self.parser.add_argument('--fixed_theta', type=float, help="fixed value for theta in DPO (overrides dynamic adjustment)", default=None)
        self.parser.add_argument('--disable_dynamic', action='store_true', help="disable dynamic adjustment in DPO")

        # GNN parameters
        self.parser.add_argument('--gnn_hid', type=int, help="hidden dimension for GNN", default=512)
        self.parser.add_argument('--gnn_output', type=int, help="output dimension for GNN", default=384)

        # Processing node attributes
        self.parser.add_argument('--llm', action='store_true', help="use the output of llm as node features")
        self.parser.add_argument('--peft', type=str, help="the type of peft", default='lora',
                                 choices=['lora', 'prefix', 'prompt', 'adapter', 'ia3'])
        self.parser.add_argument('--lm_type', type=str, help="the type of lm", default='tiny',
                                 choices=['tiny', 'sbert', 'deberta', 'bert', 'e5', 'llama2', 'llama3', 'llama2-14', 'qwen2', 'qwen2.5-0.5b', 'tiny', 'sbert2'])

        # used for sampling
        self.parser.add_argument('--subsampling', action='store_true', help="subsampling, training with subgraphs")
        self.parser.add_argument('--restart', type=float, help="the restart ratio of random walking", default=0.5)
        self.parser.add_argument('--walk_steps', type=int, help="the steps of random walking", default=64)
        self.parser.add_argument('--k', type=int, help="the hop of neighboors", default=1)
        self.parser.add_argument('--sampler', type=str, help="the choice of sampler, random walk or k-hop sampling", default='rw',
                                 choices=['rw', 'khop', 'shadow'])
        self.parser.add_argument('--walk_length', type=int, help="random walk length for positional encoding", default=32)

        # prompt type
        self.parser.add_argument('--prompt', type=str, help="the type of prompt tuning", default='gppt',
                                 choices=['gppt', 'graphprompt', 'prog', 'gpf'])

    def parse_args(self):
        return self.parser.parse_args()
