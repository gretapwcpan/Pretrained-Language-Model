# AutoTinyBERT Issues and Solutions

## Critical Issues Found

### 1. **Data Generation Bug (generate_data.py)**
**Problem**: Hardcoded `/cache/shelf.db` path causes permission errors
```python
# Line 54-55 in original generate_data.py
self.document_shelf = shelve.open('/cache/shelf.db', flag='n', protocol=-1)
```

**Solution**: Use the fixed version `generate_data_fixed.py` which:
- Uses temporary directory for shelf database
- Reduces default file count from 28 to 4 (avoids empty files)
- Distributes data more evenly across files

### 2. **Empty Training Data Files**
**Problem**: With small datasets, most of the 28 generated files end up empty (0 samples)

**Solution**: The fixed script:
- Collects all examples first
- Dynamically adjusts number of output files based on data size
- Ensures minimum examples per file

### 3. **Distributed Training Hanging**
**Problem**: Training hangs with multi-GPU setup when dataset is too small

**Root Causes**:
- Only 18 total samples across all files
- Uneven distribution across 4 GPUs (18 samples ÷ 32 batch size)
- Warmup steps (10,000) exceed total steps (3)
- Distributed synchronization deadlock

**Solutions**:

#### A. For Testing/Small Datasets - Use Single GPU:
```bash
# Single GPU training (recommended for small datasets)
python pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/models/superbert_model \
    --cache_dir experiments/output \
    --epochs 3 \
    --train_batch_size 8 \
    --learning_rate 1e-4 \
    --max_seq_length 128 \
    --masked_lm_prob 0.15 \
    --warmup_proportion 0.1 \
    --do_lower_case \
    --mlm_loss \
    --scratch
```

#### B. For Production - Generate More Data First:
```bash
# Step 1: Create a larger corpus (example with repeated text for testing)
cat corpus.txt corpus.txt corpus.txt corpus.txt > large_corpus.txt

# Step 2: Generate data with fixed script
python generate_data_fixed.py \
    --train_corpus large_corpus.txt \
    --output_dir experiments/data/pretrain_data \
    --bert_model bert-base-uncased \
    --do_lower_case \
    --max_seq_len 128 \
    --num_files 4 \
    --min_examples_per_file 1000

# Step 3: Then use multi-GPU training
torchrun --nproc_per_node=4 pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/models/superbert_model \
    --cache_dir experiments/output \
    --epochs 3 \
    --train_batch_size 32 \
    --learning_rate 1e-4 \
    --max_seq_length 128 \
    --masked_lm_prob 0.15 \
    --warmup_steps 1000 \
    --do_lower_case \
    --fp16 \
    --mlm_loss \
    --scratch
```

### 4. **Deprecated Commands in README**
**Problem**: README uses deprecated `torch.distributed.launch`

**Solution**: Use `torchrun` instead:
```bash
# Old (deprecated):
python -m torch.distributed.launch --nproc_per_node=8 pre_training.py

# New (correct):
torchrun --nproc_per_node=8 pre_training.py
```

## Quick Fix Guide

### For Immediate Testing:

1. **Generate data with the fixed script:**
```bash
python generate_data_fixed.py \
    --train_corpus your_corpus.txt \
    --output_dir experiments/data/pretrain_data \
    --bert_model bert-base-uncased \
    --do_lower_case \
    --max_seq_len 128 \
    --num_files 1  # Single file for small datasets
```

2. **Check generated data:**
```bash
# Count examples in each file
for f in experiments/data/pretrain_data/*.json; do
    echo "$f: $(wc -l < $f) examples"
done
```

3. **If data is small (<1000 examples), use single GPU:**
```bash
python pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model bert-base-uncased \
    --cache_dir ./cache \
    --epochs 3 \
    --train_batch_size 8 \
    --learning_rate 1e-4 \
    --max_seq_length 128 \
    --masked_lm_prob 0.15 \
    --warmup_proportion 0.1 \
    --do_lower_case \
    --mlm_loss \
    --scratch
```

## Minimum Data Requirements

For stable multi-GPU training with 4 GPUs:
- **Minimum**: 1,000 examples (allows for proper batching)
- **Recommended**: 10,000+ examples
- **Batch size**: 8-32 per GPU
- **Rule of thumb**: Total examples should be >> (batch_size × num_gpus × num_steps)

## Verification Checklist

Before training:
1. ✅ Check data files are not empty
2. ✅ Ensure total examples > 100 × num_gpus
3. ✅ Verify warmup_steps < total_training_steps
4. ✅ Use single GPU for datasets < 1000 examples
5. ✅ Use `torchrun` instead of `torch.distributed.launch`

## Sample Corpus for Testing

Create a test corpus with sufficient data:
```bash
cat > test_corpus.txt << 'EOF'
Machine learning is a subset of artificial intelligence.
It focuses on building systems that learn from data.
Neural networks are computational models inspired by biological neural networks.

Deep learning is a class of machine learning algorithms.
It uses multiple layers to progressively extract features.
Transformers have revolutionized natural language processing.

BERT is a transformer-based model for NLP tasks.
It uses bidirectional training to understand context.
Pre-training on large corpora improves downstream performance.

AutoML aims to automate machine learning workflows.
Neural architecture search finds optimal model architectures.
Knowledge distillation transfers knowledge from large to small models.
EOF

# Repeat to create larger corpus
for i in {1..100}; do cat test_corpus.txt >> large_test_corpus.txt; echo "" >> large_test_corpus.txt; done
```

This will create a corpus with ~1200 examples when processed, sufficient for testing.
