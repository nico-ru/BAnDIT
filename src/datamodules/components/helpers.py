import torch
from io import TextIOWrapper

from typing import Callable, Dict, List, Optional, TextIO, Union


def readFile(file: TextIO, *args):
    return file.read()


def requestToEncTensor(
    sample: Union[List[str], TextIOWrapper],
    token_vector_mapping: Dict[str, torch.Tensor],
    encoder: Optional[Callable],
) -> torch.Tensor:
    tns = requestToTensor(sample, token_vector_mapping)
    if encoder is not None:
        return encoder(tns).data
    return tns


def requestToTensor(
    sample: Union[List[str], TextIOWrapper],
    token_vector_mapping: Dict[str, torch.Tensor],
) -> torch.Tensor:
    if isinstance(sample, TextIOWrapper):
        return readFileToTensor(sample, token_vector_mapping)
    return termsToTensor(sample, token_vector_mapping)


def termsToTensor(
    terms: List[str], token_vector_mapping: Dict[str, torch.Tensor]
) -> torch.Tensor:
    vectors = list(
        map(
            lambda x: token_vector_mapping[x]
            if x in token_vector_mapping
            else token_vector_mapping["UNK"],
            terms,
        )
    )
    return torch.stack(vectors)


def readFileToTensor(
    file: TextIO, token_vector_mapping: Dict[str, torch.Tensor]
) -> torch.Tensor:
    terms = file.read().split(" ")
    return termsToTensor(terms, token_vector_mapping)
