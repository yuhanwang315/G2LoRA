# G2LoRA

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

---

## ⚙️ Important Arguments

| Argument | Description |
|---------|------------|
| --dataset | dataset name |
| --model | backbone model |
| --task_type | FSNCIL / FSNDIL / FSNTIL |
| --cl_type | class / domain / task |
| --hyperparam_search | enable search |
| --ntrail | repeat runs |


```

---

