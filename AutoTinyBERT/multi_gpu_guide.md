# Multi-GPU Training Guide for AutoTinyBERT

## Can Option 1 (6K sentences) Support Multiple GPUs?

**Yes, but with considerations:**

### Data Requirements for Multi-GPU

| GPUs | Batch Size/GPU | Total Batch | Min Examples Needed | Option 1 (6K sentences) |
|------|---------------|-------------|-------------------|------------------------|
| 2 | 8 | 16 | ~500 | ✅ Works well (~1,500 examples) |
| 4 | 8 | 32 | ~1,000 | ⚠️ Borderline (~1,500 examples) |
| 8 | 8 | 64 | ~2,000 | ❌ Too small |

**Option 1 generates ~1,500 examples from 6K sentences**, which is:
- ✅ **Good for 2 GPUs**
- ⚠️ **Marginal for 4 GPUs** (may work but not optimal)
- ❌ **Too small for 8 GPUs**

## Recommended Commands by Data Size

### For Option 1 (Quick Test - ~1,500 examples):

#### 2 GPU Training (Recommended)
```bash
torchrun --nproc_per_node=2 pre_training.py \
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
    --fp16 \
    --mlm_loss \
    --scratch
```

#### 4 GPU Training (Possible but not optimal)
```bash
# Reduce batch size to 4 per GPU to ensure enough batches
torchrun --nproc_per_node=4 pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/models/superbert_model \
    --cache_dir experiments/output \
    --epochs 5 \
    --train_batch_size 4 \
    --learning_rate 1e-4 \
    --max_seq_length 128 \
    --masked_lm_prob 0.15 \
    --warmup_proportion 0.1 \
    --do_lower_case \
    --fp16 \
    --mlm_loss \
    --scratch
```

### For Option 2 (WikiText - ~100,000+ examples):

#### 4 GPU Training (Optimal)
```bash
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

#### 8 GPU Training
```bash
torchrun --nproc_per_node=8 pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/models/superbert_model \
    --cache_dir experiments/output \
    --epochs 3 \
    --train_batch_size 32 \
    --learning_rate 1e-4 \
    --max_seq_length 128 \
    --masked_lm_prob 0.15 \
    --warmup_steps 2000 \
    --do_lower_case \
    --fp16 \
    --mlm_loss \
    --scratch
```

### For Option 4 (All Combined - ~104,000+ examples):

Best for full multi-GPU training with any configuration.

## How to Generate More Data for Multi-GPU

### Quick Solution: Repeat Option 1 Multiple Times
```bash
# Generate base corpus
./generate_more_data.sh  # Choose option 1

# Multiply the corpus for more data
for i in {1..10}; do 
    cat large_test_corpus.txt >> massive_corpus.txt
    echo "" >> massive_corpus.txt
done

# Process with fixed script
python generate_data_fixed.py \
    --train_corpus massive_corpus.txt \
    --output_dir experiments/data/pretrain_data \
    --bert_model bert-base-uncased \
    --do_lower_case \
    --max_seq_len 128 \
    --num_files 8  # More files for larger data
```

This creates ~15,000 examples, suitable for 4-8 GPUs.

## Rule of Thumb for Multi-GPU

**Minimum examples needed = num_gpus × batch_size_per_gpu × 50**

Examples:
- 2 GPUs, batch_size=8: Need at least 2×8×50 = 800 examples
- 4 GPUs, batch_size=8: Need at least 4×8×50 = 1,600 examples  
- 8 GPUs, batch_size=8: Need at least 8×8×50 = 3,200 examples

## Checking Your Data Size

Always verify before training:
```bash
# Count total examples
total=0
for f in experiments/data/pretrain_data/*.json; do
    count=$(wc -l < "$f")
    echo "$f: $count examples"
    total=$((total + count))
done
echo "Total: $total examples"

# Determine GPU count
if [ $total -lt 1000 ]; then
    echo "Use 1 GPU (single GPU mode)"
elif [ $total -lt 2000 ]; then
    echo "Use 2 GPUs maximum"
elif [ $total -lt 4000 ]; then
    echo "Use 4 GPUs maximum"
else
    echo "Can use 8+ GPUs"
fi
```

## Summary

**For Option 1 (6K sentences → ~1,500 examples):**
- ✅ **1 GPU**: Perfect
- ✅ **2 GPUs**: Works well
- ⚠️ **4 GPUs**: Possible with reduced batch size
- ❌ **8 GPUs**: Not recommended

**Want more GPUs? Generate more data:**
- Option 2 (WikiText): Supports 4-8 GPUs easily
- Option 4 (All combined): Best for any multi-GPU setup
- Or multiply Option 1 corpus as shown above
