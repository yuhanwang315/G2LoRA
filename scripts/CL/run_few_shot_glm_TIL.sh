declare -A methods

# GLM=('LM_emb' 'ENGINE' 'GraphPrompter' 'GraphGPT' 'LLaGA' 'SimGCL')
 GLM=('GTAlign')
# datasets=('instagram+wikics+bbbp')
# type='node,edge,graph'
datasets=('cora+wikics+photo')
type='node,edge,graph'

for method in "${GLM[@]}"; do
    for dataset in "${datasets[@]}"; do
        # First search hyper-parameters
        # python main.py \
        #     --dataset "$dataset" \
        #     --model_type "GLM" \
        #     --model "$method" \
        #     --cl_type 'task' \
        #     --task_type 'FSNTIL' \
        #     --ntrail 1 \
        #     --hyperparam_search \
        #     --search_type 'grid' \
        #     --num_samples 10 \
        #     --type "$type" 

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
