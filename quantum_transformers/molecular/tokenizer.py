from __future__ import annotations
from typing import Dict, List, Optional, Union
import re

class MolecularTokenizer:

    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    MASK_TOKEN = "[MASK]"

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab or self._build_default_vocab()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def _build_default_vocab(self) -> Dict[str, int]:

        tokens = [
            self.PAD_TOKEN, self.UNK_TOKEN, self.CLS_TOKEN,
            self.SEP_TOKEN, self.MASK_TOKEN,
        ]

        atoms = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
                 "B", "Si", "Se", "Te", "As", "Al", "Zn", "Cu", "Fe"]

        symbols = ["(", ")", "[", "]", "=", "#", "@", "+", "-",
                   ".", "/", "\\", "%", "1", "2", "3", "4", "5",
                   "6", "7", "8", "9", "0"]

        all_tokens = tokens + atoms + symbols
        return {t: i for i, t in enumerate(all_tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.UNK_TOKEN]

    def tokenize(self, text: str) -> List[str]:

        raise NotImplementedError

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
    ) -> List[int]:

        tokens = self.tokenize(text)

        tokens = [self.CLS_TOKEN] + tokens + [self.SEP_TOKEN]

        ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]

        if max_length:
            if len(ids) > max_length:
                ids = ids[:max_length]
            elif padding and len(ids) < max_length:
                ids += [self.pad_token_id] * (max_length - len(ids))

        return ids

    def decode(self, ids: List[int]) -> str:

        tokens = [self.inv_vocab.get(i, self.UNK_TOKEN) for i in ids]

        tokens = [t for t in tokens if t not in [
            self.PAD_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN
        ]]
        return "".join(tokens)

    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: int = 128,
        padding: bool = True,
        return_tensors: str = None,
    ):

        import torch

        if isinstance(text, str):
            text = [text]

        encoded = [self.encode(t, max_length, padding) for t in text]

        if return_tensors == "pt":
            return torch.tensor(encoded)

        return encoded

class SMILESTokenizer(MolecularTokenizer):

    ATOM_PATTERN = r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    def tokenize(self, smiles: str) -> List[str]:

        tokens = re.findall(self.ATOM_PATTERN, smiles)
        return tokens

    def is_valid_smiles(self, smiles: str) -> bool:

        try:
            tokens = self.tokenize(smiles)
            return len(tokens) > 0
        except Exception:
            return False

class AtomTokenizer(MolecularTokenizer):

    ATOM_PATTERN = r"([A-Z][a-z]?)"

    def __init__(self):
        super().__init__()
        self.vocab = self._build_atom_vocab()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def _build_atom_vocab(self) -> Dict[str, int]:

        atoms = [
            self.PAD_TOKEN, self.UNK_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN,
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
            "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
            "Sb", "Te", "I", "Xe"
        ]
        return {a: i for i, a in enumerate(atoms)}

    def tokenize(self, formula: str) -> List[str]:

        return re.findall(self.ATOM_PATTERN, formula)

class SelfiesTokenizer(MolecularTokenizer):

    SELFIES_PATTERN = r"(\[[^\]]+\])"

    def __init__(self):
        super().__init__()

        selfies_tokens = [
            "[C]", "[=C]", "[#C]", "[N]", "[=N]", "[#N]",
            "[O]", "[=O]", "[S]", "[=S]", "[P]", "[F]",
            "[Cl]", "[Br]", "[I]", "[Ring1]", "[Ring2]",
            "[Branch1]", "[Branch2]", "[=Branch1]", "[=Branch2]",
        ]

        self.vocab = {
            self.PAD_TOKEN: 0, self.UNK_TOKEN: 1,
            self.CLS_TOKEN: 2, self.SEP_TOKEN: 3,
        }
        for i, token in enumerate(selfies_tokens):
            self.vocab[token] = i + 4

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, selfies: str) -> List[str]:

        return re.findall(self.SELFIES_PATTERN, selfies)
