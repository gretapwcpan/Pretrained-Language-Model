#!/bin/bash

# Script to generate more training data for AutoTinyBERT
# This provides multiple options for creating a larger corpus

echo "=== AutoTinyBERT Data Generation Script ==="
echo ""

# Option 1: Quick test corpus (small but sufficient for testing)
create_test_corpus() {
    echo "Creating test corpus..."
    cat > test_corpus_base.txt << 'EOF'
Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.
Neural networks are computational models inspired by biological neural networks in animal brains.
Deep learning is a class of machine learning algorithms that uses multiple layers to progressively extract higher-level features from raw input.

Natural language processing enables computers to understand, interpret, and generate human language.
Transformers have revolutionized NLP by using self-attention mechanisms to process sequential data.
BERT uses bidirectional training to understand context from both left and right sides of a token.

Pre-training on large corpora helps models learn general language representations.
Fine-tuning adapts pre-trained models to specific downstream tasks with smaller datasets.
Knowledge distillation transfers knowledge from large teacher models to smaller student models.

The attention mechanism allows models to focus on relevant parts of the input sequence.
Multi-head attention enables models to attend to information from different representation subspaces.
Positional encodings help transformers understand the order of tokens in a sequence.

Tokenization breaks text into smaller units that models can process.
Subword tokenization handles out-of-vocabulary words by breaking them into known subunits.
WordPiece tokenization is used by BERT to create a vocabulary of subword units.

The masked language modeling objective trains models to predict masked tokens.
Next sentence prediction helps models understand relationships between sentences.
Sequence-to-sequence models transform input sequences into output sequences.

Transfer learning leverages knowledge from pre-trained models for new tasks.
Few-shot learning enables models to learn from limited examples.
Zero-shot learning allows models to perform tasks without task-specific training.

Model compression reduces the size of neural networks while maintaining performance.
Quantization reduces the precision of model weights to decrease memory usage.
Pruning removes unnecessary connections in neural networks.

Gradient descent optimizes model parameters by following the negative gradient.
Adam optimizer combines momentum and adaptive learning rates for efficient training.
Learning rate scheduling adjusts the learning rate during training for better convergence.

Batch normalization normalizes inputs to each layer to stabilize training.
Dropout randomly deactivates neurons during training to prevent overfitting.
Regularization techniques help models generalize better to unseen data.

The embedding layer maps discrete tokens to continuous vector representations.
Hidden states capture intermediate representations as data flows through the network.
The output layer produces final predictions based on the processed representations.

Cross-entropy loss measures the difference between predicted and true probability distributions.
Perplexity evaluates language models by measuring how well they predict text.
BLEU score assesses the quality of machine-translated text.

Data augmentation creates synthetic training examples to improve model robustness.
Back-translation generates paraphrases by translating text to another language and back.
Token replacement augmentation randomly replaces tokens with similar words.

Distributed training parallelizes model training across multiple GPUs or machines.
Mixed precision training uses both 16-bit and 32-bit floating-point numbers to speed up training.
Gradient accumulation simulates larger batch sizes by accumulating gradients over multiple steps.

The transformer architecture consists of encoder and decoder stacks.
Self-attention allows each position to attend to all positions in the previous layer.
Feed-forward networks apply point-wise transformations to each position.

Layer normalization normalizes inputs across features for each training example.
Residual connections help gradients flow through deep networks.
Position-wise feed-forward networks process each position independently.

Beam search explores multiple hypotheses during sequence generation.
Greedy decoding selects the most probable token at each step.
Top-k sampling restricts sampling to the k most likely tokens.

The vocabulary size affects model capacity and computational cost.
Sequence length limitations require strategies for processing long documents.
Attention complexity scales quadratically with sequence length.

Model interpretability helps understand what neural networks learn.
Attention visualization reveals which parts of the input models focus on.
Probing tasks evaluate what linguistic knowledge models capture.

Continual learning enables models to learn new tasks without forgetting old ones.
Multi-task learning trains models to perform multiple related tasks simultaneously.
Meta-learning helps models learn how to learn new tasks quickly.

The pre-training corpus quality significantly impacts model performance.
Domain adaptation fine-tunes models for specific domains or genres.
Cross-lingual models can understand and generate text in multiple languages.

Evaluation metrics measure different aspects of model performance.
Human evaluation provides qualitative assessment of model outputs.
Automatic metrics enable rapid iteration during model development.

The computational cost of training large models requires efficient implementations.
Model parallelism distributes different parts of the model across devices.
Data parallelism replicates the model and distributes data across devices.

Hyperparameter tuning finds optimal settings for model training.
Neural architecture search automatically discovers effective model architectures.
AutoML automates the machine learning pipeline from data to deployment.

The scaling laws describe how model performance improves with size and data.
Larger models generally perform better but require more resources.
Efficient architectures aim to achieve good performance with fewer parameters.

Prompt engineering designs inputs to elicit desired behaviors from models.
In-context learning enables models to learn from examples in the prompt.
Chain-of-thought prompting improves reasoning by generating intermediate steps.

Model deployment requires optimization for inference speed and memory usage.
Edge deployment runs models on resource-constrained devices.
Cloud deployment leverages scalable infrastructure for serving models.

The ethics of AI involves ensuring models are fair, transparent, and beneficial.
Bias in training data can lead to biased model predictions.
Responsible AI practices help develop trustworthy and reliable systems.

Federated learning trains models on distributed data without centralizing it.
Privacy-preserving techniques protect sensitive information during training.
Differential privacy adds noise to prevent individual data points from being identified.

The future of NLP involves more capable and efficient models.
Multimodal models process and generate different types of data like text and images.
General-purpose models can adapt to a wide variety of tasks and domains.
EOF

    # Repeat the base corpus to create a larger dataset
    echo "Expanding corpus (this may take a moment)..."
    > large_test_corpus.txt
    for i in {1..100}; do
        cat test_corpus_base.txt >> large_test_corpus.txt
        echo "" >> large_test_corpus.txt  # Add blank line between documents
    done
    
    echo "✓ Created large_test_corpus.txt with approximately 6,000 sentences"
}

# Option 2: Download WikiText-103 (smaller Wikipedia dataset)
download_wikitext() {
    echo "Downloading WikiText-103 dataset..."
    if [ ! -f "wikitext-103-raw-v1.zip" ]; then
        wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
        unzip wikitext-103-raw-v1.zip
    fi
    
    echo "Combining WikiText files..."
    cat wikitext-103-raw/wiki.train.raw wikitext-103-raw/wiki.valid.raw > wikitext_corpus.txt
    echo "✓ Created wikitext_corpus.txt"
}

# Option 3: Download BookCorpus sample (if available)
download_bookcorpus_sample() {
    echo "Downloading BookCorpus sample..."
    # Note: Full BookCorpus requires special access. This downloads a sample.
    wget https://raw.githubusercontent.com/soskek/bookcorpus/master/download_list.txt -O bookcorpus_urls.txt
    echo "Note: Full BookCorpus requires special access. Consider using WikiText or creating synthetic data."
}

# Option 4: Generate synthetic corpus with specific patterns
generate_synthetic_corpus() {
    echo "Generating synthetic training corpus..."
    python3 << 'PYTHON_SCRIPT'
import random

# Templates for generating diverse sentences
templates = [
    "The {adjective} {noun} {verb} {adverb} in the {location}.",
    "{subject} {verb} {object} because {reason}.",
    "When {condition}, {subject} {action}.",
    "The {concept} of {field} is {description}.",
    "{technology} enables {capability} by {method}.",
    "Research shows that {finding} when {condition}.",
    "The {system} processes {input} to produce {output}.",
    "{method} improves {metric} by {technique}.",
    "In {domain}, {approach} is used for {purpose}.",
    "The {algorithm} optimizes {objective} using {strategy}."
]

adjectives = ["efficient", "robust", "scalable", "innovative", "advanced", "optimal", "flexible", "powerful", "accurate", "reliable"]
nouns = ["model", "system", "algorithm", "network", "architecture", "framework", "method", "approach", "technique", "solution"]
verbs = ["processes", "analyzes", "transforms", "optimizes", "learns", "adapts", "generates", "predicts", "classifies", "extracts"]
adverbs = ["quickly", "efficiently", "accurately", "reliably", "effectively", "automatically", "dynamically", "seamlessly", "robustly", "consistently"]
locations = ["cloud", "edge device", "data center", "production environment", "distributed system", "neural network", "training pipeline", "inference engine"]
subjects = ["The model", "The system", "The algorithm", "The network", "The framework", "The pipeline", "The architecture", "The platform"]
objects = ["data", "features", "patterns", "representations", "embeddings", "predictions", "classifications", "outputs", "results", "insights"]
reasons = ["it improves performance", "it reduces latency", "it increases accuracy", "it enhances efficiency", "it enables scalability"]
conditions = ["data is available", "training completes", "optimization converges", "validation succeeds", "deployment occurs"]
actions = ["performs inference", "updates weights", "processes batches", "generates outputs", "makes predictions"]
concepts = ["principle", "theory", "methodology", "paradigm", "approach", "technique", "strategy", "mechanism"]
fields = ["machine learning", "deep learning", "natural language processing", "computer vision", "reinforcement learning", "neural networks"]
descriptions = ["fundamental to modern AI", "essential for performance", "critical for success", "important for scalability", "key to efficiency"]
technologies = ["Deep learning", "Transfer learning", "Federated learning", "Reinforcement learning", "Meta-learning", "Continual learning"]
capabilities = ["automated decision making", "pattern recognition", "predictive analytics", "intelligent automation", "adaptive behavior"]
methods = ["neural network training", "gradient optimization", "attention mechanisms", "embedding techniques", "regularization strategies"]
findings = ["models improve with scale", "attention enhances performance", "pre-training helps downstream tasks", "distillation preserves knowledge"]
systems = ["transformer", "encoder", "decoder", "classifier", "generator", "discriminator", "optimizer", "scheduler"]
inputs = ["text sequences", "token embeddings", "feature vectors", "raw data", "batched samples", "training examples"]
outputs = ["predictions", "classifications", "embeddings", "representations", "probabilities", "logits", "attention weights"]
metrics = ["accuracy", "efficiency", "throughput", "latency", "performance", "quality", "robustness", "generalization"]
techniques = ["regularization", "normalization", "optimization", "augmentation", "distillation", "quantization", "pruning"]
domains = ["NLP", "computer vision", "speech recognition", "recommendation systems", "time series analysis", "graph learning"]
approaches = ["supervised learning", "unsupervised learning", "self-supervised learning", "semi-supervised learning", "few-shot learning"]
purposes = ["classification", "generation", "translation", "summarization", "question answering", "information extraction"]
algorithms = ["gradient descent", "backpropagation", "attention mechanism", "convolution operation", "recurrent network"]
objectives = ["loss function", "accuracy metric", "perplexity score", "BLEU score", "F1 score", "precision", "recall"]
strategies = ["mini-batch processing", "adaptive learning rates", "early stopping", "curriculum learning", "transfer learning"]

def generate_sentence():
    template = random.choice(templates)
    sentence = template.format(
        adjective=random.choice(adjectives),
        noun=random.choice(nouns),
        verb=random.choice(verbs),
        adverb=random.choice(adverbs),
        location=random.choice(locations),
        subject=random.choice(subjects),
        object=random.choice(objects),
        reason=random.choice(reasons),
        condition=random.choice(conditions),
        action=random.choice(actions),
        concept=random.choice(concepts),
        field=random.choice(fields),
        description=random.choice(descriptions),
        technology=random.choice(technologies),
        capability=random.choice(capabilities),
        method=random.choice(methods),
        finding=random.choice(findings),
        system=random.choice(systems),
        input=random.choice(inputs),
        output=random.choice(outputs),
        metric=random.choice(metrics),
        technique=random.choice(techniques),
        domain=random.choice(domains),
        approach=random.choice(approaches),
        purpose=random.choice(purposes),
        algorithm=random.choice(algorithms),
        objective=random.choice(objectives),
        strategy=random.choice(strategies)
    )
    return sentence

# Generate corpus
with open("synthetic_corpus.txt", "w") as f:
    for doc in range(1000):  # 1000 documents
        sentences_per_doc = random.randint(5, 20)
        for _ in range(sentences_per_doc):
            f.write(generate_sentence() + "\n")
        f.write("\n")  # Blank line between documents

print("✓ Generated synthetic_corpus.txt with ~10,000 sentences")
PYTHON_SCRIPT
}

# Main menu
echo "Choose data generation option:"
echo "1) Quick Test Corpus (6K sentences, good for testing)"
echo "2) Download WikiText-103 (100M+ tokens, good for real training)"
echo "3) Generate Synthetic Corpus (10K sentences, diverse patterns)"
echo "4) All of the above"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        create_test_corpus
        CORPUS_FILE="large_test_corpus.txt"
        ;;
    2)
        download_wikitext
        CORPUS_FILE="wikitext_corpus.txt"
        ;;
    3)
        generate_synthetic_corpus
        CORPUS_FILE="synthetic_corpus.txt"
        ;;
    4)
        create_test_corpus
        download_wikitext
        generate_synthetic_corpus
        echo "Combining all corpora..."
        cat large_test_corpus.txt wikitext_corpus.txt synthetic_corpus.txt > combined_corpus.txt
        CORPUS_FILE="combined_corpus.txt"
        echo "✓ Created combined_corpus.txt"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=== Next Steps ==="
echo "1. Generate training data using the fixed script:"
echo ""
echo "python generate_data_fixed.py \\"
echo "    --train_corpus $CORPUS_FILE \\"
echo "    --output_dir experiments/data/pretrain_data \\"
echo "    --bert_model bert-base-uncased \\"
echo "    --do_lower_case \\"
echo "    --max_seq_len 128 \\"
echo "    --num_files 4"
echo ""
echo "2. Check the generated data:"
echo "for f in experiments/data/pretrain_data/*.json; do echo \"\$f: \$(wc -l < \$f) examples\"; done"
echo ""
echo "3. Start training (single GPU for small data, multi-GPU for large):"
echo "See FIXES_AND_SOLUTIONS.md for training commands"
