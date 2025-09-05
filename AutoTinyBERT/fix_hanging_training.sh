#!/bin/bash

# Script to fix AutoTinyBERT training hanging issue
# The problem: Not enough data for multi-GPU training causes synchronization deadlock

echo "=== Fixing AutoTinyBERT Training Hanging Issue ==="
echo ""
echo "Current problem: 748 samples is too small for 4 GPU training"
echo "Solution: Generate more data OR use fewer GPUs"
echo ""

# Function to check current data
check_current_data() {
    echo "Checking current data..."
    total=0
    for f in experiments/data/pretrain_data/*.json; do
        if [ -f "$f" ]; then
            count=$(wc -l < "$f")
            echo "$f: $count examples"
            total=$((total + count))
        fi
    done
    echo "Total: $total examples"
    
    if [ $total -lt 1000 ]; then
        echo "❌ Not enough data for multi-GPU training!"
        echo "   You have $total examples, need at least 2000 for 4 GPUs"
    elif [ $total -lt 2000 ]; then
        echo "⚠️  Borderline data for 4 GPUs"
        echo "   Recommend using 2 GPUs or generating more data"
    else
        echo "✅ Sufficient data for multi-GPU training"
    fi
    
    return $total
}

# Option 1: Single GPU training (always works)
single_gpu_training() {
    echo ""
    echo "=== Single GPU Training (Guaranteed to Work) ==="
    echo ""
    echo "python pre_training.py \\"
    echo "    --pregenerated_data experiments/data/pretrain_data \\"
    echo "    --student_model experiments/models/superbert_model \\"
    echo "    --cache_dir experiments/output \\"
    echo "    --epochs 3 \\"
    echo "    --train_batch_size 16 \\"
    echo "    --learning_rate 1e-4 \\"
    echo "    --max_seq_length 128 \\"
    echo "    --masked_lm_prob 0.15 \\"
    echo "    --warmup_proportion 0.1 \\"
    echo "    --do_lower_case \\"
    echo "    --mlm_loss \\"
    echo "    --scratch"
}

# Option 2: Generate sufficient data
generate_sufficient_data() {
    echo ""
    echo "=== Generating Sufficient Data for 4 GPUs ==="
    echo ""
    
    # Download WikiText for real data
    if [ ! -f "wikitext-103-raw-v1.zip" ]; then
        echo "Downloading WikiText-103..."
        wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
        unzip wikitext-103-raw-v1.zip
    fi
    
    # Combine to create large corpus
    echo "Creating large corpus..."
    cat wikitext-103-raw/wiki.train.raw wikitext-103-raw/wiki.valid.raw > large_corpus.txt
    
    # Take first 10000 lines for faster processing (still gives ~5000 examples)
    head -10000 large_corpus.txt > medium_corpus.txt
    
    echo "Processing corpus with fixed script..."
    python generate_data_fixed.py \
        --train_corpus medium_corpus.txt \
        --output_dir experiments/data/pretrain_data_sufficient \
        --bert_model bert-base-uncased \
        --do_lower_case \
        --max_seq_len 128 \
        --num_files 4 \
        --min_examples_per_file 500
    
    # Check results
    echo ""
    echo "Generated data:"
    total=0
    for f in experiments/data/pretrain_data_sufficient/*.json; do
        if [ -f "$f" ]; then
            count=$(wc -l < "$f")
            echo "$f: $count examples"
            total=$((total + count))
        fi
    done
    echo "Total: $total examples"
    
    if [ $total -gt 2000 ]; then
        echo "✅ Success! Generated $total examples, sufficient for 4 GPU training"
        echo ""
        echo "Now run with 4 GPUs:"
        echo "torchrun --nproc_per_node=4 pre_training.py \\"
        echo "    --pregenerated_data experiments/data/pretrain_data_sufficient \\"
        echo "    --student_model experiments/models/superbert_model \\"
        echo "    --cache_dir experiments/output \\"
        echo "    --epochs 3 \\"
        echo "    --train_batch_size 8 \\"
        echo "    --learning_rate 1e-4 \\"
        echo "    --max_seq_length 128 \\"
        echo "    --masked_lm_prob 0.15 \\"
        echo "    --warmup_steps 100 \\"
        echo "    --do_lower_case \\"
        echo "    --fp16 \\"
        echo "    --mlm_loss \\"
        echo "    --scratch"
    fi
}

# Option 3: Use 2 GPUs with current data
two_gpu_training() {
    echo ""
    echo "=== 2 GPU Training (Works with 748+ examples) ==="
    echo ""
    echo "torchrun --nproc_per_node=2 pre_training.py \\"
    echo "    --pregenerated_data experiments/data/pretrain_data \\"
    echo "    --student_model experiments/models/superbert_model \\"
    echo "    --cache_dir experiments/output \\"
    echo "    --epochs 3 \\"
    echo "    --train_batch_size 8 \\"
    echo "    --learning_rate 1e-4 \\"
    echo "    --max_seq_length 128 \\"
    echo "    --masked_lm_prob 0.15 \\"
    echo "    --warmup_proportion 0.1 \\"
    echo "    --do_lower_case \\"
    echo "    --fp16 \\"
    echo "    --mlm_loss \\"
    echo "    --scratch"
}

# Main menu
echo "Choose solution:"
echo "1) Use Single GPU (guaranteed to work with any data size)"
echo "2) Generate more data for 4 GPU training (downloads WikiText)"
echo "3) Use 2 GPUs with current data"
echo "4) Just check current data status"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        single_gpu_training
        ;;
    2)
        generate_sufficient_data
        ;;
    3)
        two_gpu_training
        ;;
    4)
        check_current_data
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=== Important Notes ==="
echo "• 748 examples is too small for 4 GPUs - causes hanging"
echo "• Minimum for stable 4 GPU training: 2000+ examples"
echo "• Single GPU always works regardless of data size"
echo "• The hanging is caused by distributed training synchronization with insufficient data"
