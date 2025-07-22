
from typing import *
import functools
from node_visitor.node_visitor import NodeVisitor
import nodes
import elementwise
import torch

class ConvertToTorchNodeVisitor(NodeVisitor[torch.Tensor]):
    def __init__(self):
        self._memo: Dict[nodes.TensorNode, torch.Tensor] = {}
    # TODO
    # (might need to add visitors for ElementwiseFunction too...)
