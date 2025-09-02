#!/usr/bin/env python3
"""
Verify AutoTinyBERT experiment setup
"""

import os
import json
import sys

def check_file(path, description):
    """Check if a file exists and report status"""
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"✅ {description}: {path} ({size:,} bytes)")
        return True
    else:
        print(f"❌ {description}: {path} (NOT FOUND)")
        return False

def check_json(path, description):
    """Check if a JSON file is valid"""
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"✅ {description}: {path} (valid JSON)")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ {description}: {path} (invalid JSON: {e})")
            return False
    else:
        print(f"❌ {description}: {path} (NOT FOUND)")
        return False

def main():
    print("=" * 60)
    print("AutoTinyBERT Experiment Setup Verification")
    print("=" * 60)
    
    all_good = True
    
    # Check directory structure
    print("\n📁 Directory Structure:")
    dirs = [
        "data/corpus",
        "data/pretrain_data",
        "models/bert-base",
        "models/super_config",
        "output"
    ]
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ (NOT FOUND)")
            all_good = False
    
    # Check data files
    print("\n📄 Data Files:")
    all_good &= check_file("data/corpus/wiki_corpus.txt", "Training corpus")
    
    # Check model files
    print("\n🤖 Model Files:")
    all_good &= check_file("models/bert-base/vocab.txt", "BERT vocabulary")
    all_good &= check_json("models/bert-base/config.json", "BERT config")
    all_good &= check_json("models/super_config/config.json", "SuperNet config")
    
    # Check documentation
    print("\n📚 Documentation:")
    all_good &= check_file("README.md", "README")
    
    # Check corpus content
    print("\n📊 Corpus Statistics:")
    corpus_path = "data/corpus/wiki_corpus.txt"
    if os.path.exists(corpus_path):
        with open(corpus_path, 'r') as f:
            lines = f.readlines()
        
        # Count sentences and documents
        sentences = [l for l in lines if l.strip()]
        documents = 1  # Start with 1
        for line in lines:
            if line.strip() == "":
                documents += 1
        
        print(f"  Total lines: {len(lines):,}")
        print(f"  Sentences: {len(sentences):,}")
        print(f"  Documents: {documents:,}")
        print(f"  Avg words per sentence: {sum(len(s.split()) for s in sentences) / len(sentences):.1f}")
    
    # Check SuperNet configuration
    print("\n⚙️ SuperNet Configuration:")
    config_path = "models/super_config/config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if "layer_num_space" in config:
            print(f"  Layer range: {config['layer_num_space']}")
        if "hidden_size_space" in config:
            print(f"  Hidden size range: {config['hidden_size_space']}")
        if "head_num_space" in config:
            print(f"  Attention heads range: {config['head_num_space']}")
    
    # Final status
    print("\n" + "=" * 60)
    if all_good:
        print("✅ All checks passed! Ready to run AutoTinyBERT experiments.")
        print("\nNext steps:")
        print("1. Generate training data (from AutoTinyBERT directory):")
        print("   python generate_data.py \\")
        print("       --train_corpus experiments/data/corpus/wiki_corpus.txt \\")
        print("       --bert_model experiments/models/bert-base \\")
        print("       --output_dir experiments/data/pretrain_data \\")
        print("       --do_lower_case")
        print("\n2. Follow the README.md for training instructions")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    main()
