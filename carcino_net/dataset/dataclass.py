from typing import TypedDict, Union, List
import torch
import numpy as np


class ModelInput(TypedDict):
    img: Union[np.ndarray, torch.Tensor]
    uri: str
    mask: Union[np.ndarray, torch.Tensor]
    label: Union[str, int]


class ModelOutput(TypedDict):

    loss: torch.Tensor
    logits: torch.Tensor
    mask: torch.Tensor
    uri: Union[str, List[str]]

