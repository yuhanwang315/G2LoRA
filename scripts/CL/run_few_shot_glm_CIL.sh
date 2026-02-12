#类别增量CIL

declare -A methods

# GLM=('LM_emb' 'ENGINE' 'GraphPrompter' 'GraphGPT' 'LLaGA' 'SimGCL' 'GTalign')
# datasets=('photo' 'computer' 'history')
GLM=('GTAlign')
datasets=('photo' 'computer' 'history' )

for method in "${GLM[@]}"; do
    for dataset in "${datasets[@]}"; do
        # First search hyper-parameters
        python main.py \
            --dataset "$dataset" \
            --model_type "GLM" \
            --model "$method" \
            --cl_type 'class' \
            --task_type 'FSNCIL' \
            --ntrail 1 \
            --hyperparam_search \
            --search_type 'grid' \
            --num_samples 10

        output_dir="./output/results/${type}/${method}/FSNTIL"
        mkdir -p "$output_dir"

        # Then Repeat with best hyper-parameters
        output_file="${output_dir}/${dataset}.txt"
        python main.py \
            --dataset "$dataset" \
            --model_type "LM" \
            --model "$method" \
            --cl_type 'class' \
            --task_type 'FSNCIL' \
            --ntrail 5
    done
done
