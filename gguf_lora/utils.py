import torch
from gguf import GGUFReader
import re
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers import normalizers

class GGUFTokenizer:
    def __init__(self, gguf_path):
        self.reader = GGUFReader(gguf_path)
        self.vocab = self._extract_vocab()
        self.bos_token_id = self._get_special_token_id('tokenizer.ggml.bos_token_id')
        self.eos_token_id = self._get_special_token_id('tokenizer.ggml.eos_token_id')
        self.unk_token_id = self._get_special_token_id('tokenizer.ggml.unk_token_id')
        self.bpe_merges = self._extract_merges()
        self.tokenizer = self._build_bpe_tokenizer()

    def _get_special_token_id(self, key):
        field = self.reader.fields.get(key)
        if field and field.data:
            return int(field.data[0])
        return None

    def _extract_vocab(self):
        field = self.reader.fields['tokenizer.ggml.tokens']
        vocab = []
        for idx in field.data:
            part = field.parts[idx]
            token_bytes = part.tobytes() if hasattr(part, 'tobytes') else bytes(part)
            vocab.append(token_bytes.decode('utf-8', errors='replace'))
        print(f"[DEBUG] Extracted vocab size: {len(vocab)}")
        return vocab

    def _extract_merges(self):
        merges_field = self.reader.fields.get('tokenizer.ggml.merges')
        if not merges_field:
            raise RuntimeError('No BPE merges found in GGUF (tokenizer.ggml.merges)')
        merges_bytes = merges_field.parts
        merges = []
        skipped = 0
        for m in merges_bytes:
            try:
                if hasattr(m, 'tobytes'):
                    line = m.tobytes().decode('utf-8').strip()
                elif isinstance(m, (bytes, bytearray)):
                    line = m.decode('utf-8').strip()
                else:
                    line = bytes(m).decode('utf-8').strip()
            except UnicodeDecodeError:
                skipped += 1
                continue
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) == 2:
                    merges.append(tuple(parts))
        if skipped > 0:
            print(f"[INFO] Skipped {skipped} non-UTF8 merge lines in GGUF merges.")
        return merges

    def _build_bpe_tokenizer(self):
        vocab_dict = {token: i for i, token in enumerate(self.vocab)}
        merges = self.bpe_merges
        model = models.BPE(vocab=vocab_dict, merges=merges, unk_token=self.vocab[self.unk_token_id or 0])
        tokenizer = Tokenizer(model)
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
        ])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer

    def encode(self, text):
        # Use the HuggingFace BPE tokenizer
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def encode_tensor(self, text, device=None):
        ids = self.encode(text)
        return torch.tensor(ids, dtype=torch.long, device=device)
