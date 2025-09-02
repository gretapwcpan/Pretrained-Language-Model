#!/bin/bash

# Setup script for AutoTinyBERT SuperBERT model configuration

echo "Setting up SuperBERT model configuration for AutoTinyBERT..."

# Create model directory
mkdir -p experiments/models/superbert_model
cd experiments/models/superbert_model

# Create SuperBERT config.json
cat > config.json << 'EOF'
{
  "architectures": ["BertForMaskedLM"],
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 30522,
  "qkv_size": 768
}
EOF

echo "Config file created: config.json"

# Download vocab.txt from BERT base uncased
echo "Downloading vocabulary file..."
wget -q https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt

if [ $? -eq 0 ]; then
    echo "Vocabulary file downloaded: vocab.txt"
else
    echo "Failed to download vocab.txt. Please download manually from:"
    echo "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
fi

cd ../../../

echo ""
echo "Setup complete! SuperBERT model directory created at:"
echo "  experiments/models/superbert_model/"
echo ""
echo "Files created:"
echo "  - config.json (SuperBERT configuration with qkv_size)"
echo "  - vocab.txt (BERT vocabulary)"
echo ""
echo "You can now use this model directory for training:"
echo ""
echo "python pre_training.py \\"
echo "    --pregenerated_data experiments/data/pretrain_data \\"
echo "    --student_model experiments/models/superbert_model \\"
echo "    --teacher_model bert-base-uncased \\"
echo "    --cache_dir experiments/output \\"
echo "    --local_rank -1 \\"
echo "    --epochs 1 \\"
echo "    --train_batch_size 8 \\"
echo "    --mlm_loss \\"
echo "    --scratch"
