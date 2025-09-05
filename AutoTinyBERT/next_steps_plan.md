# AutoTinyBERT Next Steps - After Single GPU Training

## Current Status
✅ Fixed pre_training.py issues (NumPy deprecations, distributed training, cache directory)
✅ Successfully conducted single GPU runs
✅ Basic experiment structure set up

## Recommended Next Steps

### Phase 1: Scale Up Training (Immediate)

#### 1.1 Multi-GPU Training
Since you've successfully run single GPU training, the next logical step is to scale up:

```bash
# Test with 2 GPUs first
torchrun --nproc_per_node=2 pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/models/superbert_model \
    --cache_dir experiments/output \
    --epochs 3 \
    --train_batch_size 16 \
    --learning_rate 1e-4 \
    --max_seq_length 128 \
    --masked_lm_prob 0.15 \
    --do_lower_case \
    --fp16 \
    --mlm_loss \
    --scratch

# Then scale to all available GPUs (e.g., 4 or 8)
torchrun --nproc_per_node=8 pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/models/superbert_model \
    --cache_dir experiments/output \
    --epochs 10 \
    --train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --max_seq_length 128 \
    --masked_lm_prob 0.15 \
    --do_lower_case \
    --fp16 \
    --mlm_loss \
    --scratch
```

#### 1.2 Knowledge Distillation Training
After MLM pre-training, try knowledge distillation for better performance:

```bash
# Download teacher model if not already available
python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"

# Run knowledge distillation
torchrun --nproc_per_node=4 pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/models/superbert_model \
    --teacher_model bert-base-uncased \
    --cache_dir experiments/output \
    --epochs 5 \
    --train_batch_size 16 \
    --learning_rate 3e-5 \
    --max_seq_length 128 \
    --do_lower_case \
    --fp16 \
    --scratch
```

### Phase 2: Architecture Search (After Pre-training)

#### 2.1 Build Latency Predictor
```bash
# Generate latency dataset
python inference_time_evaluation.py \
    --model_path experiments/output/checkpoint-best \
    --output_dir latency/dataset

# Train latency predictor
python latency_predictor.py \
    --data_dir latency/dataset \
    --output_model latency/mlm_model/time.pt
```

#### 2.2 Search for Optimal Architecture
```bash
# Get candidate architectures
python searcher.py \
    --ckpt_path latency/mlm_model/time.pt \
    --latency_constraint 7 \
    --method Candidate \
    --model MLM \
    --candidate_file cands/mlm_7x

# Random search baseline
python searcher.py \
    --ckpt_path latency/mlm_model/time.pt \
    --candidate_file cands/mlm_7x \
    --latency_constraint 7 \
    --method Random \
    --model MLM \
    --output_file cands/1st_generation.random.cands

# Fast search
python searcher.py \
    --ckpt_path latency/mlm_model/time.pt \
    --candidate_file cands/mlm_7x \
    --latency_constraint 7 \
    --method Fast \
    --model MLM \
    --output_file cands/1st_generation.fast.cands
```

### Phase 3: Evaluation on Downstream Tasks

#### 3.1 Download GLUE Dataset
```bash
# Install datasets library
pip install datasets

# Download GLUE tasks
python -c "
from datasets import load_dataset
for task in ['mnli', 'sst2', 'cola', 'qqp', 'mrpc', 'qnli', 'rte', 'wnli']:
    dataset = load_dataset('glue', task)
    print(f'Downloaded {task}')
"
```

#### 3.2 Evaluate Candidate Architectures
```bash
# Evaluate on GLUE and SQuAD
python superbert_run_en_classifier.py \
    --data_dir dataset/glue \
    --model experiments/output/checkpoint-best \
    --task_name "mnli sst2 cola qqp" \
    --output_dir experiments/evaluation \
    --do_lower_case \
    --arches_file cands/1st_generation.fast.cands
```

#### 3.3 Evolutionary Search (Iterative)
```bash
# Run evolutionary search based on evaluation results
python searcher.py \
    --ckpt_path latency/mlm_model/time.pt \
    --candidate_file cands/mlm_7x \
    --latency_constraint 7 \
    --method Evolved \
    --model MLM \
    --output_file cands/2nd_generation.evo.cands \
    --arch_perfs_file experiments/evaluation/subbert.results
```

### Phase 4: Extract and Fine-tune Best Model

#### 4.1 Extract Optimal Sub-model
```bash
# Example architecture (replace with your best found architecture)
python submodel_extractor.py \
    --model experiments/output/checkpoint-best \
    --arch "{'sample_layer_num': 5, 'sample_num_attention_heads': [8, 8, 8, 8, 8], 'sample_qkv_sizes': [512, 512, 512, 512, 512], 'sample_hidden_size': 564, 'sample_intermediate_sizes': [1054, 1054, 1054, 1054, 1054]}" \
    --output experiments/extracted_model/
```

#### 4.2 Further Training of Extracted Model
```bash
# Continue training the extracted model
torchrun --nproc_per_node=4 pre_training.py \
    --pregenerated_data experiments/data/pretrain_data \
    --student_model experiments/extracted_model \
    --cache_dir experiments/output_final \
    --epochs 5 \
    --train_batch_size 32 \
    --learning_rate 2e-5 \
    --max_seq_length 128 \
    --masked_lm_prob 0.15 \
    --do_lower_case \
    --fp16 \
    --mlm_loss \
    --further_train
```

### Phase 5: Production Deployment

#### 5.1 Model Optimization
```bash
# Convert to ONNX for inference optimization
python -c "
from transformers import AutoModel
import torch
model = AutoModel.from_pretrained('experiments/extracted_model')
dummy_input = torch.randint(0, 1000, (1, 128))
torch.onnx.export(model, dummy_input, 'model.onnx', 
                  input_names=['input_ids'],
                  output_names=['output'],
                  dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}})
"
```

#### 5.2 Benchmark Final Model
```bash
# Measure inference speed
python inference_time_evaluation.py \
    --model_path experiments/extracted_model \
    --batch_sizes "1,8,16,32" \
    --seq_lengths "32,64,128,256" \
    --output_file benchmarks.json
```

## Monitoring and Debugging Tips

### Check Training Progress
```bash
# Monitor loss curves
tensorboard --logdir experiments/output/runs

# Check GPU utilization
nvidia-smi -l 1

# Monitor distributed training
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

### Common Issues to Watch For

1. **OOM Errors**: Reduce batch_size or use gradient_accumulation_steps
2. **Slow Multi-GPU**: Check NCCL_DEBUG=INFO for communication issues
3. **Convergence Issues**: Adjust learning rate schedule or warmup steps
4. **Data Loading Bottleneck**: Increase num_workers in DataLoader

## Expected Timeline

- **Phase 1**: 1-2 days (depending on data size and GPU resources)
- **Phase 2**: 2-3 days (architecture search is iterative)
- **Phase 3**: 1-2 days (evaluation on multiple tasks)
- **Phase 4**: 1 day (extraction and fine-tuning)
- **Phase 5**: 1 day (optimization and benchmarking)

## Success Metrics

- [ ] Multi-GPU training achieves near-linear speedup
- [ ] Model achieves >75% of BERT-base performance
- [ ] Inference speed is >5x faster than BERT-base
- [ ] Model size is <50MB (compressed)
- [ ] GLUE average score >75

## Resources and References

-
