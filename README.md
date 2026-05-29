# G2LoRA

## Environment Setup

1.  **Create a Python environment:**
    It is recommended to use a virtual environment.
    ```bash
    conda create -n G2LoRA python=3.10 
    conda activate G2LoRA
    ```

2.  **Install dependencies:**
    Install the required packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    **Note:** Please ensure you have a compatible version of PyTorch and PyTorch Geometric installed that matches your CUDA version if using GPUs. You might need to install these separately by following official instructions from their respective websites if issues arise with the `requirements.txt` or if you have specific CUDA needs.

Our code was tested with Python 3.10. Experiments were primarily conducted on NVIDIA H100 (80GB VRAM)，A GPU with sufficient memory is recommended.

## Running Experiments

Experiments are organized into three main settings: CIL, TIL, and DIL. Each setting can be launched using the corresponding run.sh scripts provided in the repository.

### 1. CIL Setting

## ▶️ Quick Start

```bash
GLM=('GTAlign')
datasets=('photo' 'computer' 'history')

for method in "${GLM[@]}"; do
    for dataset in "${datasets[@]}"; do

        python main.py \
            --dataset "$dataset" \
            --model_type GLM \
            --model "$method" \
            --cl_type class \
            --task_type FSNCIL \
            --hyperparam_search \
            --search_type grid \
            --num_samples 10

        output_dir="./outputs/${method}/FSNCIL"
        mkdir -p "$output_dir"

        python main.py \
            --dataset "$dataset" \
            --model_type GLM \
            --model "$method" \
            --cl_type class \
            --task_type FSNCIL \
             --ntrail 5 > "$output_file"

    done
done
```


### 2. DIL Setting
## ▶️ Quick Start

```bash
GLM=('GTAlign')
datasets=('cora+citeseer+wikics')

for method in "${GLM[@]}"; do
    for dataset in "${datasets[@]}"; do

        python main.py \
            --dataset "$dataset" \
            --model_type GLM \
            --model "$method" \
            --cl_type domain \
            --task_type FSNDIL \
            --hyperparam_search \
            --search_type grid \
            --num_samples 10

        output_dir="./outputs/${method}/FSNDIL"
        mkdir -p "$output_dir"

        python main.py \
            --dataset "$dataset" \
            --model_type GLM \
            --model "$method" \
            --cl_type domain \
            --task_type FSNDIL \
             --ntrail 5 > "$output_file"

    done
done
```



### 3. TIL Setting
## ▶️ Quick Start

```bash
GLM=('GTAlign')
datasets=('cora+wikics+photo')
type='node,edge,graph'

for method in "${GLM[@]}"; do
    for dataset in "${datasets[@]}"; do
        # First search hyper-parameters
        python main.py \
            --dataset "$dataset" \
            --model_type "GLM" \
            --model "$method" \
            --cl_type 'task' \
            --task_type 'FSNTIL' \
            --ntrail 1 \
            --hyperparam_search \
            --search_type 'grid' \
            --num_samples 10 \
            --type "$type" 

        output_dir="./output/results/${type}/${method}/FSNTIL"
        mkdir -p "$output_dir"

        # Then Repeat with best hyper-parameters
        output_file="${output_dir}/${dataset}.txt"
        python main.py \
            --dataset "$dataset" \
            --model_type "GLM" \
            --model "$method" \
            --cl_type 'task' \
            --task_type 'FSNTIL' \
            --ntrail 5 \
            --type "$type" 
    done
done
```

### We release the core implementation of G2LoRA. 
### The complete codebase and datasets will be released upon acceptance.

