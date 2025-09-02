# AutoTinyBERT Experiment Setup Guide

This guide provides complete instructions for running AutoTinyBERT experiments with prepared data and configurations.

## 📁 Directory Structure

```
AutoTinyBERT/experiments/
├── data/
│   ├── corpus/
│   │   └── wiki_corpus.txt        # Training corpus (prepared)
│   └── pretrain_data/             # Generated training data (will be created)
├── models/
│   ├── bert-base/
│   │   ├── vocab.txt              # BERT vocabulary (downloaded)
│   │   └── config.json            # BERT config (downloaded)
│   └── super_config/
│       └── config.json            # SuperNet configuration (prepared)
└── output/                        # Training outputs (will be created)
```

## 🚀 Quick Start

### Step 0: Install Dependencies

```bash
# IMPORTANT: Install in this order to avoid dependency issues

# 1. First install PyTorch (choose based on your system)
# For CPU only:
pip install torch torchvision torchaudio

# For CUDA 11.8 (check your CUDA version with nvidia-smi):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Then install other required packages
pip install transformers numpy tqdm tensorboard tensorboardX

# 3. Optional: Install apex for mixed precision training (GPU only)
# ONLY install apex AFTER torch is successfully installed
# Note: This requires CUDA toolkit and torch with CUDA support
git clone https://github.com/NVIDIA/apex
cd apex
# If this fails, skip it - apex is optional
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

**Note**: If apex installation fails, you can still run AutoTinyBERT without it by adding `--no_cuda` flag to training commands.

### Step 1: Generate Training Data

```bash
# From AutoTinyBERT directory
python generate_data.py \
    --train_corpus experiments/data/corpus/wiki_corpus.txt \
    --bert_model experiments/models/bert-base \
    --output_dir experiments/data/pretrain_data \
    --do_lower_case \
    --max_seq_len 128
```

### Step 2: Train SuperNet

You have two options for training:

#### Option A: MLM Loss (Simpler, No Teacher Required)
```bash
python pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/models/super_config \
    --cache_dir experiments/output \
    --epochs 3 \
    --train_batch_size 16 \
    --learning_rate 5e-5 \
    --max_seq_length 128 \
    --masked_lm_prob 0.15 \
    --do_lower_case \
    --mlm_loss \
    --scratch \
    --local_rank 0
```

#### Option B: Knowledge Distillation (Better Quality)
```bash
# First download a teacher model (e.g., BERT or ELECTRA)
python pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/models/super_config \
    --teacher_model bert-base-uncased \
    --cache_dir experiments/output \
    --epochs 3 \
    --train_batch_size 16 \
    --learning_rate 5e-5 \
    --max_seq_length 128 \
    --masked_lm_prob 0.0 \
    --do_lower_case \
    --scratch \
    --local_rank 0
```

### Step 3: Build Latency Predictor

```bash
# Generate latency dataset
python AutoTinyBERT/inference_time_evaluation.py \
    --model_path AutoTinyBERT_experiment/output

# Train latency predictor
python AutoTinyBERT/latency_predictor.py \
    --output AutoTinyBERT_experiment/output/time.pt
```

### Step 4: Architecture Search

```bash
# 1. Generate candidate architectures
python AutoTinyBERT/searcher.py \
    --ckpt_path AutoTinyBERT_experiment/output/time.pt \
    --latency_constraint 7 \
    --method Candidate \
    --model MLM \
    --candidate_file AutoTinyBERT_experiment/output/candidates.txt

# 2. Search for optimal architecture (choose one method)

# Random Search (baseline)
python AutoTinyBERT/searcher.py \
    --ckpt_path AutoTinyBERT_experiment/output/time.pt \
    --candidate_file AutoTinyBERT_experiment/output/candidates.txt \
    --latency_constraint 7 \
    --method Random \
    --model MLM \
    --output_file AutoTinyBERT_experiment/output/random_arch.txt

# Fast Search (gradient-based)
python AutoTinyBERT/searcher.py \
    --ckpt_path AutoTinyBERT_experiment/output/time.pt \
    --candidate_file AutoTinyBERT_experiment/output/candidates.txt \
    --latency_constraint 7 \
    --method Fast \
    --model MLM \
    --output_file AutoTinyBERT_experiment/output/fast_arch.txt

# Evolutionary Search (best results)
python AutoTinyBERT/searcher.py \
    --ckpt_path AutoTinyBERT_experiment/output/time.pt \
    --candidate_file AutoTinyBERT_experiment/output/candidates.txt \
    --latency_constraint 7 \
    --method Evolved \
    --model MLM \
    --output_file AutoTinyBERT_experiment/output/evolved_arch.txt
```

### Step 5: Extract and Fine-tune Best Architecture

```bash
# Extract the best sub-model
python AutoTinyBERT/submodel_extractor.py \
    --model AutoTinyBERT_experiment/output/superbert \
    --arch "{'sample_layer_num': 4, 'sample_num_attention_heads': [6, 6, 6, 6], 'sample_qkv_sizes': [384, 384, 384, 384], 'sample_hidden_size': 384, 'sample_intermediate_sizes': [1536, 1536, 1536, 1536]}" \
    --output AutoTinyBERT_experiment/output/extracted_model

# Further train the extracted model
python AutoTinyBERT/pre_training.py \
    --pregenerated_data AutoTinyBERT_experiment/data/pretrain_data \
    --student_model AutoTinyBERT_experiment/output/extracted_model \
    --cache_dir AutoTinyBERT_experiment/output/final_model \
    --epochs 3 \
    --train_batch_size 32 \
    --learning_rate 3e-5 \
    --mlm_loss \
    --further_train \
    --local_rank 0
```

## 📊 Understanding the Process

### What Makes AutoTinyBERT Different?

1. **SuperNet Training**: Instead of training a fixed architecture like TinyBERT, AutoTinyBERT trains a SuperNet that contains many sub-networks
2. **Architecture Search**: Uses NAS to find the optimal architecture for your latency constraints
3. **One-Shot Training**: All sub-networks share weights, making the search efficient

### Key Concepts

- **SuperNet**: A large network containing all possible architectures in the search space
- **Sub-network Sampling**: During training, different architectures are sampled from the SuperNet
- **Latency Predictor**: Estimates inference time without actual deployment
- **Search Methods**:
  - Random: Baseline method, samples architectures randomly
  - Fast: Gradient-based search for quick results
  - Evolutionary: Iterative improvement for best results

### Search Space Configuration

The search space is defined in `models/super_config/config.json`:

```json
{
  "layer_num_space": [1, 8],           # 1 to 8 layers
  "hidden_size_space": [128, 768],     # 128 to 768 hidden dimensions
  "qkv_size_space": [180, 768],        # Attention dimension range
  "intermediate_size_space": [128, 3072], # FFN dimension range
  "head_num_space": [1, 12]            # 1 to 12 attention heads
}
```

## 🔧 Customization

### Using Your Own Data

Replace `wiki_corpus.txt` with your own corpus:
- One sentence per line
- Blank lines separate documents
- UTF-8 encoding

### Adjusting for Different Hardware

Modify batch size and gradient accumulation based on your GPU memory:
```bash
--train_batch_size 8 \
--gradient_accumulation_steps 4  # Effective batch size = 8 * 4 = 32
```

### Different Latency Constraints

Change `--latency_constraint` to target different speedups:
- `7`: 7x faster than BERT (balanced)
- `15`: 15x faster (more aggressive)
- `3`: 3x faster (higher quality)

## 📈 Expected Results

| Speedup | Model Size | Expected Accuracy |
|---------|------------|-------------------|
| 7x      | ~14MB      | ~83% (SQuAD)     |
| 15x     | ~7MB       | ~78% (SQuAD)     |
| 27x     | ~4MB       | ~72% (SQuAD)     |

## 🐛 Troubleshooting

### Missing Dependencies
If you encounter `ModuleNotFoundError`:
```bash
# Quick fix - install all dependencies:
pip install torch transformers numpy tqdm tensorboard tensorboardX

# Check your environment:
python -c "import torch, transformers, tensorboard; print('✅ All packages installed')"
```

### Out of Memory
- Reduce `--train_batch_size`
- Increase `--gradient_accumulation_steps`
- Use `--fp16` for mixed precision training

### Slow Training
- Ensure you're using GPU: check with `nvidia-smi`
- Use multiple GPUs with distributed training
- Reduce `--max_seq_length` if possible

### Poor Results
- Train for more epochs
- Use knowledge distillation instead of MLM loss
- Try different search methods (evolutionary usually best)

## 📚 Key Files Reference

- `generate_data.py`: Preprocesses raw text into training format
- `pre_training.py`: Trains the SuperNet
- `searcher.py`: Performs architecture search
- `submodel_extractor.py`: Extracts specific architectures
- `utils.py`: Contains sampling functions for architectures
- `debug_setup.md`: Troubleshooting guide for common issues
- `QUICK_START.md`: Condensed setup instructions

## 📦 Required Python Packages

| Package | Purpose | Required |
|---------|---------|----------|
| `torch` | Deep learning framework | ✅ Yes |
| `transformers` | Hugging Face models | ✅ Yes |
| `numpy` | Numerical operations | ✅ Yes |
| `tqdm` | Progress bars | ✅ Yes |
| `tensorboard` | Training visualization | ✅ Yes |
| `tensorboardX` | TensorBoard alternative | ✅ Yes |
| `apex` | Mixed precision training | ⚠️ Optional (GPU) |

## 🎯 Next Steps

1. **Evaluate on downstream tasks**: Test your model on GLUE benchmarks
2. **Deploy the model**: Use ONNX or TorchScript for production
3. **Experiment with search space**: Modify the architecture ranges
4. **Try different teachers**: Use ELECTRA or RoBERTa as teachers

## 📖 Additional Resources

- [Original Paper](https://aclanthology.org/2021.acl-long.400/)
- [TinyBERT Paper](https://arxiv.org/abs/1909.10351) (for comparison)
- [Neural Architecture Search Survey](https://arxiv.org/abs/1808.05377)

## 💡 Tips for Success

1. **Start small**: Use the provided sample data first
2. **Monitor training**: Watch loss curves to ensure convergence
3. **Save checkpoints**: Training can be resumed if interrupted
4. **Compare methods**: Try different search algorithms
5. **Iterate**: The evolutionary search improves over generations

---

**Note**: This experiment setup uses a small corpus for demonstration. For production models, use larger datasets like Wikipedia + BookCorpus (>10GB of text).
