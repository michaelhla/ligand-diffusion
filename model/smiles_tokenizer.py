from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re
import json
import os
from pathlib import Path

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

    def train(self, smiles_list: List[str]):
        """Train BPE tokenizer on a list of SMILES strings."""
        # Count initial character frequencies
        word_freqs = defaultdict(int)
        for smiles in smiles_list:
            tokens = self._tokenize_smiles(smiles)
            word_freqs[' '.join(tokens)] += 1

        # Initialize vocabulary with character-level tokens
        vocab = self.base_vocab.copy()
        next_token_id = max(vocab.values()) + 1

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
        save_dict = {
            'vocab': self.vocab,
            'merges': self.merges,
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
        tokenizer.merges = save_dict['merges']
        tokenizer.special_tokens = save_dict['special_tokens']
        tokenizer.reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        return tokenizer