# AutoTinyBERT: Automatic Hyper-parameter Optimization for Efficient Pre-trained Language Models (ACL 2021)

## Overview

Pre-trained language models (PLMs) have achieved great success in natural language processing. 
Most of PLMs follow the default setting of architecture hyper-parameters
 (e.g., the hidden dimension is a quarter of the intermediate dimension in feed-forward sub-networks) 
 in BERT. In this paper, we adopt the one-shot Neural Architecture Search (NAS) to 
 automatically search architecture hyper-parameters for efficient pre-trained language models (at least 6x faster than BERT-base). 
 Our framework is illustrated as follows: 

<img src="AutoTinyBERT_overview.PNG" width="1000" height="610"/>

For more details about the techniques of AutoTinyBERT, please refer to our paper:



## Model Zoo

We release the Model Zoo of AutoTinyBERT here, Speedup is compared with BERT-base (L12D768)

| Version     | Speedup (CPU)     |  SQuADv1 (dev)    | GLUE (dev) | Link |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|  S1         | 7.2x | 83.3 | 78.3 | [S1](https://pan.baidu.com/s/16ugFWK5D9HhYPSpvptg1yg)[b4db] |
|  S2         | 15.7x| 78.1 | 76.4 | [S2](https://pan.baidu.com/s/151iOhPPAQjFM4eSu_9PcpA)[pq9i] |
|  S3         | 20.2x| 75.8 | 75.3 | [S3](https://pan.baidu.com/s/1PEDbD08-AZvuoAusyNxVIA)[a52b] |
|  S4         | 27.2x| 71.9 | 73.0 | [S4](https://pan.baidu.com/s/1ykqNFHLK93TJBosJX876sQ)[msen] |
|  KD-S1      | 4.6x | 87.6 | 81.2 | [KD-S1](https://pan.baidu.com/s/1uj8EuED2HeH6heMKAxHv_A)[lv15] |
|  KD-S2      | 9.0x | 84.6 | 77.5 | [KD-S2](https://pan.baidu.com/s/18ytClliS4IEe7t60ZD7Dew)[agob] |
|  KD-S3      | 10.7x| 83.3 | 76.2 | [KD-S3](https://pan.baidu.com/s/1pGpqZ_XDMqR69HY-YS-8GQ)[9pi2] |
|  KD-S4      | 17.0x| 78.7 | 73.5 | [KD-S4](https://pan.baidu.com/s/1ceJ6CvaNrXXlrIt4lF6QSg)[l9lc] |



## Use in Transformers
Our released code can directly load the pre-trained models, and also the models can be used 
in [Huggingface Transformers](https://github.com/huggingface/transformers) by small modifications as follows:

```
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        ### Before modifications:
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        ### After modifications: 
        try:
            qkv_size = config.qkv_size
        except:
            qkv_size = config.hidden_size

        self.attention_head_size = int(qkv_size / config.num_attention_heads)

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        ### Before modifications:
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        ### After modifications:
        try:
            qkv_size = config.qkv_size
        except:
            qkv_size = config.hidden_size
        self.dense = nn.Linear(qkv_size, config.hidden_size)
```

 

## Quick Start Guide

### Step 1: Setup Model Directory
AutoTinyBERT requires a proper model directory with vocabulary and config files. Run the setup script:

```bash
# Option 1: Run with bash
bash setup_superbert_model.sh

# Option 2: Make executable and run directly
chmod +x setup_superbert_model.sh
./setup_superbert_model.sh

# This creates experiments/models/superbert_model/ with:
# - config.json (with SuperBERT-specific qkv_size parameter)
# - vocab.txt (BERT vocabulary)
```

### Step 2: Generate Training Data
Generate preprocessed training data from a corpus. You can use various data sources:

#### Option A: Download WikiText Dataset (Quick Start)
```bash
# Download WikiText-103 (a smaller Wikipedia dataset for testing)
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
cd wikitext-103-raw

# Combine train and valid for more data
cat wiki.train.raw wiki.valid.raw > ../corpus.txt
cd ..

# Generate training data
python generate_data.py \
    --train_corpus corpus.txt \
    --output_dir experiments/data/pretrain_data \
    --bert_model bert-base-uncased \
    --do_lower_case \
    --max_seq_len 128
```

#### Option B: Use Wikipedia Dump (Production Scale)
```bash
# Download Wikipedia dump (English)
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Extract and preprocess (requires wikiextractor)
pip install wikiextractor
python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2 \
    --processes 8 --output wiki_extracted

# Combine extracted files into corpus
find wiki_extracted -name '*.txt' -exec cat {} \; > wiki_corpus.txt

# Generate training data
python generate_data.py \
    --train_corpus wiki_corpus.txt \
    --output_dir experiments/data/pretrain_data \
    --bert_model bert-base-uncased \
    --do_lower_case \
    --max_seq_len 128 \
    --reduce_memory  # Use for large datasets
```

#### Option C: Use BookCorpus or Custom Dataset
```bash
# If you have BookCorpus or other text data
# Format: One sentence per line, blank lines between documents

# Generate training data
python generate_data.py \
    --train_corpus path/to/your/corpus.txt \
    --output_dir experiments/data/pretrain_data \
    --bert_model bert-base-uncased \
    --do_lower_case \
    --max_seq_len 128
```

**Note**: The corpus should be formatted with one sentence per line, and documents separated by blank lines.

### Step 3: Train SuperPLM

#### Option A: MLM Pre-training (Self-supervised)
```bash
# Single GPU training
python pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/models/superbert_model \
    --cache_dir experiments/output \
    --epochs 3 \
    --train_batch_size 8 \
    --learning_rate 1e-4 \
    --max_seq_length 128 \
    --masked_lm_prob 0.15 \
    --do_lower_case \
    --mlm_loss \
    --scratch

# Multi-GPU training with torchrun (recommended for PyTorch 1.9+)
torchrun --nproc_per_node=8 pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/models/superbert_model \
    --cache_dir experiments/output \
    --epochs 3 \
    --train_batch_size 8 \
    --learning_rate 1e-4 \
    --max_seq_length 128 \
    --masked_lm_prob 0.15 \
    --do_lower_case \
    --fp16 \
    --mlm_loss \
    --scratch
```

#### Option B: Knowledge Distillation (Teacher-Student)
```bash
# Single GPU training
python pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/models/superbert_model \
    --teacher_model bert-base-uncased \
    --cache_dir experiments/output \
    --epochs 3 \
    --train_batch_size 8 \
    --learning_rate 1e-4 \
    --max_seq_length 128 \
    --do_lower_case \
    --scratch

# Multi-GPU training with torchrun (recommended for PyTorch 1.9+)
torchrun --nproc_per_node=8 pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/models/superbert_model \
    --teacher_model bert-base-uncased \
    --cache_dir experiments/output \
    --epochs 3 \
    --train_batch_size 8 \
    --learning_rate 1e-4 \
    --max_seq_length 128 \
    --do_lower_case \
    --fp16 \
    --scratch
```

### Common Issues and Solutions

1. **Model Path Error**: If you get "vocab.txt not found" or "config.json not found":
   - Run `bash setup_superbert_model.sh` to create proper model directory
   - Or use HuggingFace model names directly: `--student_model bert-base-uncased`

2. **Permission Denied on /cache**: The script now uses the `--cache_dir` argument properly. Use a writable directory:
   ```bash
   --cache_dir ./experiments/cache  # or ~/cache or /tmp/cache
   ```

3. **NumPy Deprecation Errors**: Fixed in the latest version
   - `np.int` → `np.int32`
   - `np.bool` → `bool`

4. **Warmup Configuration**: Smart adaptive warmup
   - Automatically switches to proportion-based warmup if steps exceed total training
   - Shows warnings and actual warmup steps in logs
   - **Best practices:**
     - Large datasets: Use `--warmup_steps` (e.g., 10000)
     - Small datasets: Use `--warmup_proportion` (e.g., 0.1 for 10%)
     - Default: 10% warmup is recommended for most cases

5. **Directory Creation Race Condition**: Fixed with proper synchronization
   - Only rank 0 creates directories
   - Other processes wait via `torch.distributed.barrier()`

6. **CUDA/Distributed Training Issues**: 
   - For single GPU, omit `--local_rank` or set it to 0
   - The script now handles negative GPU index properly

7. **Memory Issues**: Reduce `--train_batch_size` or use `--gradient_accumulation_steps`

### Architecture Search Space
AutoTinyBERT searches over these hyperparameters:
- **Layers**: 1-8 transformer layers
- **Hidden Size**: 128-768 dimensions
- **QKV Size**: 180-768 dimensions (attention mechanism)
- **FFN Size**: 128-3072 dimensions (feed-forward network)
- **Attention Heads**: 1-12 heads


### Random|Fast|Evolved Search
We first build the Latency predictor Lat(\*) by the `inference_time_evaluation.py` and `latency_predictor.py`. The first 
script is used to generate the dataset and second one aims to build the Lat(\*) classifier trained on the generated dataset. 
Through these scripts, we get the model file `time.pt` of Lat(\*). Then, we do the search as follows:

```
[1] Obtain candidates 
python searcher.py --ckpt_path latency/mlm_model/time.pt \
    --latency_constraint 7 --method Candidate --model MLM \
    --candidate_file cands/mlm_7x

the candidates will be saved in ${candidate_file}$ and you can set the specific ${latency_constraint}$.

[2] Random Search
python searcher.py --ckpt_path latency/mlm_model/time.pt \
    --candidate_file cands/mlm_7x --latency_constraint 7 \
     --method Random --model MLM --output_file cands/1st_generation.cands

[3] Fast Search
python searcher.py --ckpt_path latency/mlm_model/time.pt \
    --candidate_file cands/mlm_7x --latency_constraint 7 \
    --method Fast --model MLM --output_file cands/1st_generation.fast.cands

[4] Evaluation of candidates
python superbert_run_en_classifier.py --data_dir "dataset/glue/MNLI dataset/SQuAD" \
 --model model/SuperBERT_MLM/ --task_name "mnli squad" --output_dir output/ \
 --do_lower_case --arches_file cands/1st_generation.fast.cands 
 
 ${model}$ means the directory of pre-trained SuperBERT model.

[5] Evolved Search
 python searcher.py --ckpt_path latency/mlm_model/time.pt  --candidate_file cands/mlm_7x \
 --latency_constraint 7 --method Evolved --model MLM --output_file cands/1st_generation.evo.cands \
 --arch_perfs_file output/subbert.results
 
 ${arch_perfs_file}$ means the results of sub-models generated by [4].
```

For the evolutionary search, we should perform [2] to generate first generation of architectures, then evaluate it by [4],
and do evolutionary algorithm [5] with the evaluation results to generate next generation. We iteratively 
perform the processes of [4] and [5] util the maximum iteration is achieved.

### Further Train
After the search, we obtain the optimal architecture. Then we extract the corresponding the sub-model
by `submodel_extractor.py` and do the further training by `pre_training.py`.

```
## Sub-model extraction
python submodel_extractor.py --model model/SuperBERT_MLM/ \
--arch "{'sample_layer_num': 5, 'sample_num_attention_heads': [8, 8, 8, 8, 8], 'sample_qkv_sizes': [512, 512, 512, 512, 512], 'sample_hidden_size': 564, 'sample_intermediate_sizes': [1054, 1054, 1054, 1054, 1054]}" \
--output extracted_model/

## Further train
### For the mlm-loss setting:
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --nnodes=$2 \
    --node_rank=$3 \
    --master_addr=$4 \
    --master_port=$5 \
    pre_training.py \
    --pregenerated_data ${train_data_dir} \
    --cache_dir ${cache_dir} \
    --epochs ${epochs} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --train_batch_size ${train_batch_size} \
    --learning_rate ${learning_rate} \
    --max_seq_length ${max_seq_length} \
    --student_model ${student_model} \
    --masked_lm_prob 0.15 \
    --do_lower_case --fp16 --mlm_loss --further_train

${student_model}$ means the extracted sub-model.
```
The kd-loss setting uses a similar command except for 'kd_loss' parameter.

## Requirements
* Latency is evaluated on Intel(R) Xeon(R) CPU E7-4850 v2 @ 2.30GHz
* Apex for fp16 training
* NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)


## Acknowledgements
Our code is developed based on [HAT](https://github.com/pytorch/fairseq) and
 [Transformers](https://github.com/huggingface/transformers).
