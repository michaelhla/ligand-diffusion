from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re
import json
import os
from pathlib import Path
from tqdm import tqdm

class SMILESBPETokenizer:
    def __init__(self, vocab_size: int = 1000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = {
            'PAD': 11,
            'START': 12,
            'END': 13,
        }
        # Initialize with base vocabulary (single characters)
        self.base_vocab = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 
            'P': 5, 'Cl': 6, 'Br': 7, 'I': 8, 'H': 9,
            'UNK': 10,
            '(': 14, ')': 15,
            '[': 16, ']': 17,
            '=': 18, '#': 19, ':': 20,
            '+': 21, '-': 22, '.': 23,
            '/': 24, '\\': 25, '@': 26, '*': 27,
            '1': 28, '2': 29, '3': 30, '4': 31, '5': 32,
            '6': 33, '7': 34, '8': 35, '9': 36
        }
        self.merges = {}
        self.vocab = self.base_vocab.copy()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    
    def train(self, datasets):
        """Train BPE tokenizer directly on datasets."""
        # Count initial character frequencies
        word_freqs = defaultdict(int)
        print("Tokenizing SMILES strings...")
        
        # Check for cached tokenization data
        cache_dir = Path("checkpoints/tokenizer_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        word_freqs_file = cache_dir / "word_freqs.json"
        tokens_file = cache_dir / "tokens.json"

        if word_freqs_file.exists() and tokens_file.exists():
            print("Loading cached tokenization data...")
            with open(word_freqs_file, 'r') as f:
                word_freqs = defaultdict(int, json.load(f))
            with open(tokens_file, 'r') as f:
                all_tokens = json.load(f)
        else:
            # Process datasets directly
            print("Processing datasets and caching tokenization data...")
            all_tokens = {}
            for dataset in datasets:
                for data in tqdm(dataset, desc=f"Processing dataset of size {len(dataset)}"):
                    if data is not None and data['ligand'].smiles is not None:
                        tokens = self._tokenize_smiles(data['ligand'].smiles)
                        token_str = ' '.join(tokens)
                        word_freqs[token_str] += 1
                        all_tokens[data['ligand'].smiles] = tokens
            
            # Cache the tokenization data
            with open(word_freqs_file, 'w') as f:
                json.dump(dict(word_freqs), f)
            with open(tokens_file, 'w') as f:
                json.dump(all_tokens, f)

        # Initialize vocabulary with character-level tokens
        vocab = self.base_vocab.copy()
        next_token_id = max(vocab.values()) + 1

        remaining_merges = self.vocab_size - len(vocab)
        pbar = tqdm(total=remaining_merges, desc="Training BPE")
        
        while len(vocab) < self.vocab_size:
            # Count pair frequencies
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])
            if best_pair[1] < self.min_frequency:
                break

            # Create new token
            new_token = ''.join(best_pair[0])
            vocab[new_token] = next_token_id
            next_token_id += 1
            self.merges[best_pair[0]] = new_token

            # Update word frequencies
            new_word_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                new_word = word.replace(' '.join(best_pair[0]), new_token)
                new_word_freqs[new_word] += freq
            word_freqs = new_word_freqs
            
            pbar.update(1)

        pbar.close()
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}

    def _tokenize_smiles(self, smiles: str) -> List[str]:
        """Tokenize SMILES string into initial character-level tokens."""
        pattern = '('+'|'.join(re.escape(key) for key in sorted(self.base_vocab.keys(), key=len, reverse=True))+')'
        tokens = re.findall(pattern, smiles)
        return tokens

    def encode(self, smiles: str) -> List[int]:
        """Encode SMILES string to token IDs."""
        tokens = self._tokenize_smiles(smiles)
        bpe_tokens = []
        i = 0
        while i < len(tokens):
            # Try to merge longest sequence possible
            longest_merge = 1
            for j in range(min(len(tokens), i + 10), i, -1):
                current_sequence = ''.join(tokens[i:j])
                if current_sequence in self.vocab:
                    longest_merge = j - i
                    break
            
            if longest_merge > 1:
                bpe_tokens.append(self.vocab[''.join(tokens[i:i+longest_merge])])
                i += longest_merge
            else:
                token = tokens[i]
                bpe_tokens.append(self.vocab.get(token, self.vocab['UNK']))
                i += 1

        # Add special tokens
        return [self.special_tokens['START']] + bpe_tokens + [self.special_tokens['END']]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to SMILES string."""
        tokens = []
        for tid in token_ids:
            if tid in [self.special_tokens['START'], self.special_tokens['END'], self.special_tokens['PAD']]:
                continue
            tokens.append(self.reverse_vocab[tid])
        return ''.join(tokens)

    def save(self, path: str):
        """Save tokenizer vocabulary and merges to file."""
        # Convert tuple keys to strings for JSON serialization
        serializable_merges = {f"{pair[0]}|{pair[1]}": value 
                                for pair, value in self.merges.items()}
        
        save_dict = {
            'vocab': self.vocab,
            'merges': serializable_merges,
            'special_tokens': self.special_tokens
        }
        with open(path, 'w') as f:
            json.dump(save_dict, f)

    @classmethod
    def load(cls, path: str):
        """Load tokenizer from file."""
        with open(path, 'r') as f:
            save_dict = json.load(f)
        
        tokenizer = cls(vocab_size=len(save_dict['vocab']))
        tokenizer.vocab = save_dict['vocab']
        # Convert string keys back to tuples
        tokenizer.merges = {tuple(k.split('|')): v 
                          for k, v in save_dict['merges'].items()}
        tokenizer.special_tokens = save_dict['special_tokens']
        tokenizer.reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        return tokenizer