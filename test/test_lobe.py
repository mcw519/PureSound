import sys

import numpy as np
import pytest
import torch
from puresound.nnet.lobe.rnn import FSMN, ConditionFSMN
from puresound.nnet.lobe.trivial import SplitMerge

sys.path.insert(0, './')


@pytest.mark.nnet
@pytest.mark.parametrize('l_ctx, r_ctx', [(3, 3), (3, 0)])
def test_fsmn_block(l_ctx, r_ctx):
    input_x = torch.rand(3, 256, 100)
    memory = torch.rand(3, 192, 100)
    fsmn_block = FSMN(256, 256, 192, l_ctx, r_ctx)

    with torch.no_grad():
        output_x, memory = fsmn_block(input_x, memory)
    
    assert input_x.shape[-1] == output_x.shape[-1] == memory.shape[-1]

    input_x[..., 50:] = np.inf
    fsmn_block = fsmn_block.eval()
    with torch.no_grad():
        output_x, memory = fsmn_block(input_x, memory)
    
    if r_ctx == 0: assert np.where(np.isnan(output_x) == True)[-1][0] == 50


@pytest.mark.nnet
def test_conditional_fsmn_block():
    input_x = torch.rand(3, 256, 100)
    memory = torch.rand(3, 128, 100)
    dvec = torch.rand(3, 192)
    fsmn_block1 = ConditionFSMN(256, 256, 128, 192, 3, 3)
    fsmn_block2 = ConditionFSMN(256, 256, 128, 192, 3, 3, use_film=True)

    with torch.no_grad():
        output_x1, memory1 = fsmn_block1(input_x, dvec, memory)
        output_x2, memory2 = fsmn_block2(input_x, dvec, memory)
    
    assert input_x.shape[-1] == output_x1.shape[-1] == memory1.shape[-1]
    assert input_x.shape[-1] == output_x2.shape[-1] == memory2.shape[-1]


@pytest.mark.nnet
def test_split_and_merge():
    input_x = torch.rand(3, 256, 1000)
    split_x, rest = SplitMerge.split(input_x, 40)
    merge_x = SplitMerge.merge(split_x, rest)
    assert torch.allclose(input_x, merge_x)
