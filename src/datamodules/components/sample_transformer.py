from enum import Enum
from io import TextIOWrapper
import pandas as pd
import torch
import torch.nn.functional as F

from typing import Any, List, Optional, TextIO, Dict, Callable, Union


class Padding(Enum):
    NONE = "NONE"
    ZEROS = "ZEROS"
    EMPTYS = "EMPTYS"
    END = "END"


class SampleTransformer:
    def __init__(
        self,
        annotations: pd.DataFrame,
        vocab: pd.DataFrame,
        size: int,
        transform: Callable[
            [Union[List[str], TextIOWrapper], Dict, Optional[Callable]],
            torch.Tensor,
        ],
        padding: Optional[Padding],
        input_mapping_type: str,
        glove_vectors_file: Optional[TextIO] = None,
        encoder: Optional[Callable] = None,
    ):

        self.annotations = annotations
        self.vocab = vocab
        self.size = size
        self.transform = transform
        self.padding = padding
        self.input_mapping_type = input_mapping_type
        self.glove_vectors_file = glove_vectors_file
        self.encoder = encoder

        self.term_onehot_mapping: Dict[str, torch.Tensor] = dict()
        self.term_glove_mapping: Dict[str, torch.Tensor] = dict()
        self.term_index_mapping: Dict[str, torch.Tensor] = dict()
        self._init_token_vector_mapping()

    def __call__(self, sample: Union[List[str], TextIOWrapper]) -> Any:
        return self.transform_sample(sample)

    def _init_token_vector_mapping(self):
        if "onehot" == self.input_mapping_type:
            self._init_onehot_mapping()
        if "glove" == self.input_mapping_type:
            self._init_glove_mapping()
        if "index" == self.input_mapping_type:
            self._init_index_mapping()

    def _init_onehot_mapping(self):
        for i, row in self.vocab.iterrows():
            one_hot = torch.zeros(len(self.vocab), dtype=torch.float)
            one_hot[:, i] = 1
            self.term_onehot_mapping[row["term"]] = one_hot

    def _init_glove_mapping(self):
        assert self.glove_vectors_file is not None
        for line in self.glove_vectors_file.readlines():
            line = line.replace("\n", "")
            token = line.split(" ")[0]
            vector = line.split(" ")[1:]
            vector = list(map(float, vector))
            self.term_glove_mapping[token] = torch.tensor(vector, dtype=torch.float)

    def _init_index_mapping(self):
        for i, row in self.vocab.iterrows():
            self.term_index_mapping[row["term"]] = torch.tensor([i])

    def get_mapping(self, type: str) -> Dict:
        return getattr(self, f"term_{type}_mapping")

    def _apply_padding(self, sample: torch.Tensor, mapping_type: str) -> torch.Tensor:
        max_n_vec = self.size
        vec_size = sample.size(1)
        diff = max_n_vec - sample.size(dim=0)
        if self.padding == Padding.ZEROS:
            p1d = (0, 0, 0, diff)
            sample = F.pad(sample, p1d)

        mapping = self.get_mapping(mapping_type)

        if self.padding == Padding.EMPTYS:
            if mapping_type == "glove":
                e_vec = torch.zeros(vec_size, dtype=torch.float)
            elif diff > 0:
                e_vec = mapping["EMPTY"]
                pad = [e_vec for i in range(diff)]
                pad = torch.stack(pad).view(-1)
                sample_flat = sample.view(-1)
                sample = torch.cat((sample_flat, pad)).view(max_n_vec, vec_size)

        if self.padding == Padding.END:
            if mapping_type == "glove":
                end_vec = torch.ones(vec_size, dtype=torch.float)
            else:
                end_vec = mapping["END"]
            sample_flat = sample.view(-1)
            sample = torch.cat((sample_flat, end_vec)).view(
                sample.size(0) + 1, vec_size
            )

        return sample

    def transform_sample(self, sample: Union[List[str], TextIOWrapper]) -> torch.Tensor:
        mapping = self.get_mapping(self.input_mapping_type)
        return self._apply_padding(
            self.transform(sample, mapping, self.encoder), self.input_mapping_type
        )

    def retransform_sample(self, sample: Union[torch.Tensor, List[int]]) -> List[str]:
        if isinstance(sample, torch.Tensor):
            return [self.vocab.iloc[idx.item()]["term"] for idx in sample]
        return [self.vocab.iloc[idx]["term"] for idx in sample]
