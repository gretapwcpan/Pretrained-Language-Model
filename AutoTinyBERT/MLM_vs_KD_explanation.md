# MLM Pre-training vs Knowledge Distillation in AutoTinyBERT

## Overview Comparison

| Aspect | MLM Pre-training (Self-supervised) | Knowledge Distillation (Teacher-Student) |
|--------|-------------------------------------|------------------------------------------|
| **Teacher Model** | ❌ Not needed | ✅ Required (e.g., BERT-base) |
| **Training Objective** | Predict masked tokens | Match teacher's outputs |
| **Training Speed** | Faster (no teacher inference) | Slower (teacher inference needed) |
| **Model Quality** | Good baseline | Usually better (learns from teacher) |
| **Memory Usage** | Lower (one model) | Higher (two models in memory) |
| **Use Case** | Training from scratch | Compressing existing model |

## 1. MLM Pre-training (Self-supervised)

### What it is:
- **Masked Language Modeling**: Randomly masks 15% of tokens and trains model to predict them
- **Self-supervised**: Learns from raw text without labels
- **No teacher**: Model learns patterns directly from data

### How it works:
```
Input:  "The [MASK] is very [MASK] today"
Target: "The weather is very nice today"
```

### Command Example:
```bash
python pre_training.py \
    --mlm_loss \              # ← Uses MLM objective
    --masked_lm_prob 0.15 \   # ← Masks 15% of tokens
    --scratch                 # ← Trains from random initialization
    # No --teacher_model needed
```

### Advantages:
- ✅ Faster training (no teacher model overhead)
- ✅ Less memory usage
- ✅ Can learn domain-specific patterns from your data
- ✅ Independent of existing models

### Disadvantages:
- ❌ May need more training data/time to reach good performance
- ❌ No guidance from a stronger model
- ❌ Quality depends entirely on your training data

## 2. Knowledge Distillation (Teacher-Student)

### What it is:
- **Teacher-Student Learning**: Small model (student) learns from large model (teacher)
- **Soft targets**: Student learns from teacher's probability distributions
- **Compression technique**: Transfers knowledge from large to small model

### How it works:
```
Input:     "The weather is very nice today"
Teacher:   [0.8, 0.1, 0.05, 0.05, ...]  # Probability distribution
Student:   Learns to match teacher's distribution
```

The student learns two things:
1. **Attention patterns**: How the teacher pays attention to different tokens
2. **Hidden representations**: The teacher's internal representations

### Command Example:
```bash
python pre_training.py \
    --teacher_model bert-base-uncased \  # ← Requires teacher
    --student_model experiments/models/superbert_model \
    # No --mlm_loss flag
    # No --masked_lm_prob needed
```

### Advantages:
- ✅ Better final performance (learns from strong teacher)
- ✅ Faster convergence (guided learning)
- ✅ Inherits teacher's knowledge
- ✅ Good for model compression

### Disadvantages:
- ❌ Slower training (teacher inference for every batch)
- ❌ Higher memory usage (both models in memory)
- ❌ Limited by teacher's knowledge
- ❌ Requires a good pre-trained teacher model

## Which Should You Use?

### Use MLM Pre-training when:
- You want faster training iteration
- You have domain-specific data
- You don't have a good teacher model
- Memory is limited
- You're experimenting with architectures

### Use Knowledge Distillation when:
- You want best possible performance
- You have a good teacher model (like BERT-base)
- You have sufficient memory (2x model size)
- You're creating a production model
- You want to compress an existing model

## Practical Example in AutoTinyBERT

### For MLM (faster, simpler):
```bash
# Generate data
python generate_data_fixed.py \
    --train_corpus corpus.txt \
    --output_dir experiments/data/pretrain_data \
    --bert_model bert-base-uncased \
    --do_lower_case \
    --max_seq_len 128

# Train with MLM
python pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model bert-base-uncased \
    --cache_dir experiments/output \
    --epochs 3 \
    --train_batch_size 16 \
    --learning_rate 1e-4 \
    --max_seq_length 128 \
    --masked_lm_prob 0.15 \    # MLM-specific
    --warmup_proportion 0.1 \
    --do_lower_case \
    --mlm_loss \                # MLM flag
    --scratch                   # Train from scratch
```

### For Knowledge Distillation (better quality):
```bash
# Same data generation

# Train with KD
python pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/models/superbert_model \
    --teacher_model bert-base-uncased \  # Teacher required
    --cache_dir experiments/output \
    --epochs 3 \
    --train_batch_size 8 \      # Smaller batch (more memory needed)
    --learning_rate 1e-4 \
    --max_seq_length 128 \
    --do_lower_case \
    --scratch
    # No --mlm_loss flag
    # No --masked_lm_prob
```

## Performance Comparison (Typical Results)

| Metric | MLM Only | Knowledge Distillation |
|--------|----------|----------------------|
| Training Time | 1x | 1.5-2x |
| Memory Usage | 1x | 2x |
| Final Accuracy | 85-90% | 90-95% |
| Convergence Speed | Slower | Faster |

## Hybrid Approach

Some researchers combine both:
1. First: Pre-train with MLM to get a decent model
2. Then: Fine-tune with KD to improve quality

This can give you the benefits of both approaches but requires more training time.

## Summary

- **MLM**: Learn patterns directly from data (self-supervised)
- **KD**: Learn from a teacher model's knowledge (supervised by teacher)
- **MLM is faster** but KD usually gives **better quality**
- Choose based on your requirements: speed vs quality, memory constraints, and whether you have a good teacher model
