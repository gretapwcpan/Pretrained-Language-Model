#!/bin/bash

# Script to reuse TinyBERT data for AutoTinyBERT training
# This converts TinyBERT format to AutoTinyBERT format

echo "=== Reusing TinyBERT Data for AutoTinyBERT ==="
echo ""

# Function to convert TinyBERT epoch files to AutoTinyBERT format
convert_tinybert_data() {
    echo "Converting TinyBERT data to AutoTinyBERT format..."
    
    python3 << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path

def convert_tinybert_to_autotinybert(input_file, output_dir, file_index):
    """
    Convert TinyBERT epoch_X.json format to AutoTinyBERT train_doc_tokens_ngrams_X.json format
    
    TinyBERT format:
    {
        "tokens": [...],
        "segment_ids": [...],
        "is_random_next": bool,
        "masked_lm_positions": [...],
        "masked_lm_labels": [...]
    }
    
    AutoTinyBERT format (simpler):
    {
        "tokens": [...]  # Just the tokens without [CLS], [SEP], [MASK]
    }
    """
    
    output_file = output_dir / f"train_doc_tokens_ngrams_{file_index}.json"
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        line_count = 0
        for line in f_in:
            data = json.loads(line.strip())
            
            # Extract tokens and remove special tokens for AutoTinyBERT
            tokens = data["tokens"]
            
            # Remove [CLS], [SEP] tokens but keep the actual text
            cleaned_tokens = []
            for token in tokens:
                if token not in ["[CLS]", "[SEP]"]:
                    # If it was a [MASK] token, we need the original
                    # For now, we'll keep the tokens as is (including [MASK])
                    # AutoTinyBERT will re-mask them during training
                    if token == "[MASK]":
                        # Skip masked tokens for clean corpus
                        continue
                    cleaned_tokens.append(token)
            
            # Only write if we have tokens
            if cleaned_tokens:
                output_data = {"tokens": cleaned_tokens}
                f_out.write(json.dumps(output_data) + '\n')
                line_count += 1
        
        print(f"Converted {line_count} examples to {output_file}")
    
    return line_count

# Check for TinyBERT data
tinybert_data_dirs = [
    Path("../TinyBERT/data"),
    Path("../TinyBERT/pretrain_data"),
    Path("../TinyBERT/output"),
    Path("TinyBERT/data"),
    Path("TinyBERT/pretrain_data"),
    Path("TinyBERT/output")
]

found_data = False
for data_dir in tinybert_data_dirs:
    if data_dir.exists():
        epoch_files = list(data_dir.glob("epoch_*.json"))
        if epoch_files:
            found_data = True
            print(f"Found {len(epoch_files)} TinyBERT epoch files in {data_dir}")
            
            # Create output directory
            output_dir = Path("experiments/data/pretrain_data_from_tinybert")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert each epoch file
            total_examples = 0
            for i, epoch_file in enumerate(sorted(epoch_files)):
                print(f"\nProcessing {epoch_file.name}...")
                examples = convert_tinybert_to_autotinybert(epoch_file, output_dir, i)
                total_examples += examples
            
            print(f"\n✓ Successfully converted {total_examples} total examples")
            print(f"✓ Output saved to {output_dir}")
            break

if not found_data:
    print("No TinyBERT epoch files found. You need to generate them first.")
    print("\nTo generate TinyBERT data:")
    print("1. Create a corpus file (one sentence per line, blank lines between documents)")
    print("2. Run TinyBERT's pregenerate_training_data.py:")
    print("   python TinyBERT/pregenerate_training_data.py \\")
    print("       --train_corpus your_corpus.txt \\")
    print("       --bert_model bert-base-uncased \\")
    print("       --output_dir TinyBERT/pretrain_data \\")
    print("       --epochs_to_generate 3 \\")
    print("       --max_seq_len 128 \\")
    print("       --masked_lm_prob 0.15 \\")
    print("       --max_predictions_per_seq 20 \\")
    print("       --do_lower_case")

PYTHON_SCRIPT
}

# Function to generate TinyBERT data first
generate_tinybert_data() {
    echo "Generating TinyBERT pre-training data..."
    
    # Check if corpus exists
    if [ ! -f "$1" ]; then
        echo "Error: Corpus file $1 not found!"
        return 1
    fi
    
    # Create output directory
    mkdir -p TinyBERT/pretrain_data
    
    # Generate data using TinyBERT's script
    python TinyBERT/pregenerate_training_data.py \
        --train_corpus "$1" \
        --bert_model bert-base-uncased \
        --output_dir TinyBERT/pretrain_data \
        --epochs_to_generate 3 \
        --max_seq_len 128 \
        --masked_lm_prob 0.15 \
        --max_predictions_per_seq 20 \
        --do_lower_case \
        --do_whole_word_mask
    
    echo "✓ TinyBERT data generated in TinyBERT/pretrain_data"
}

# Function to use existing corpus or create one
prepare_corpus() {
    # Check for existing corpus files
    if [ -f "large_test_corpus.txt" ]; then
        echo "Found existing large_test_corpus.txt"
        CORPUS_FILE="large_test_corpus.txt"
    elif [ -f "wikitext_corpus.txt" ]; then
        echo "Found existing wikitext_corpus.txt"
        CORPUS_FILE="wikitext_corpus.txt"
    elif [ -f "corpus.txt" ]; then
        echo "Found existing corpus.txt"
        CORPUS_FILE="corpus.txt"
    else
        echo "No corpus found. Creating a test corpus..."
        # Use the same test corpus from generate_more_data.sh
        bash generate_more_data.sh
        CORPUS_FILE="large_test_corpus.txt"
    fi
    
    echo "Using corpus: $CORPUS_FILE"
}

# Main menu
echo "Choose an option:"
echo "1) Convert existing TinyBERT epoch files to AutoTinyBERT format"
echo "2) Generate new TinyBERT data from corpus, then convert"
echo "3) Check for existing TinyBERT data"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        convert_tinybert_data
        ;;
    2)
        prepare_corpus
        generate_tinybert_data "$CORPUS_FILE"
        echo ""
        convert_tinybert_data
        ;;
    3)
        echo "Checking for TinyBERT data..."
        for dir in TinyBERT/pretrain_data TinyBERT/data TinyBERT/output ../TinyBERT/pretrain_data; do
            if [ -d "$dir" ]; then
                echo "Found directory: $dir"
                ls -la "$dir" | grep "epoch_" || echo "  No epoch files found"
            fi
        done
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=== Next Steps ==="
echo ""
echo "If conversion was successful, train AutoTinyBERT with:"
echo ""
echo "# For single GPU (recommended for small datasets):"
echo "python pre_training.py \\"
echo "    --pregenerated_data experiments/data/pretrain_data_from_tinybert \\"
echo "    --student_model experiments/models/superbert_model \\"
echo "    --cache_dir experiments/output \\"
echo "    --epochs 3 \\"
echo "    --train_batch_size 8 \\"
echo "    --learning_rate 1e-4 \\"
echo "    --max_seq_length 128 \\"
echo "    --masked_lm_prob 0.15 \\"
echo "    --warmup_proportion 0.1 \\"
echo "    --do_lower_case \\"
echo "    --mlm_loss \\"
echo "    --scratch"
echo ""
echo "# For multi-GPU (if you have enough data):"
echo "torchrun --nproc_per_node=4 pre_training.py \\"
echo "    --pregenerated_data experiments/data/pretrain_data_from_tinybert \\"
echo "    --student_model experiments/models/superbert_model \\"
echo "    --cache_dir experiments/output \\"
echo "    --epochs 3 \\"
echo "    --train_batch_size 32 \\"
echo "    --learning_rate 1e-4 \\"
echo "    --max_seq_length 128 \\"
echo "    --masked_lm_prob 0.15 \\"
echo "    --warmup_steps 1000 \\"
echo "    --do_lower_case \\"
echo "    --fp16 \\"
echo "    --mlm_loss \\"
echo "    --scratch"
