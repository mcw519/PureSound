import sys

import torch
from puresound.nnet.skim import MemLSTM, SegLSTM
from puresound.streaming.skim_inference import StreamingSkiM

sys.path.insert(0, './')


def test_mem_lstm():
    memlstm = MemLSTM(10, causal=True, dropout=0)
    memlstm.eval()
    h = torch.rand(1, 10, 1, 10).float().detach()
    c = torch.rand(1, 10, 1, 10).float().detach()
    with torch.no_grad():
        h1, c1, h1_hid, c1_hid = memlstm(h, c, return_all=True, streaming=True)
        h2_hid = None
        c2_hid = None
        h2_out = []
        c2_out = []
        for i in range(10):
            h2 = h[:, i, :, :].view(1, 1, 1, 10)
            c2 = c[:, i, :, :].view(1, 1, 1, 10)
            h2, c2, h2_hid, c2_hid = memlstm.streaming_forward(h2, c2, h2_hid, c2_hid, return_all=True)
            h2_out.append(h2)
            c2_out.append(c2)
    
    h2 = torch.cat(h2_out, dim=1)
    c2 = torch.cat(c2_out, dim=1)

    assert torch.mean((h1 - h2).abs()) < 1e-6, f"mean abs error: {torch.mean((h1 - h2).abs())}, max_error: {(h1 - h2).abs().max()}"
    assert torch.mean((c1 - c2).abs()) < 1e-6, f"mean abs error: {torch.mean((c1 - c2).abs())}, max_error: {(c1 - c2).abs().max()}"


def test_seg_lstm():
    seglstm = SegLSTM(10, 20, causal=True, dropout=True)
    seglstm.eval()
    x = torch.rand(1, 20, 10).float().detach() # [1, K, C]
    h = torch.rand(1, 1, 20).float().detach()
    c = torch.rand(1, 1, 20).float().detach()
    with torch.no_grad():
        y1, h1, c1 = seglstm(x, h, c)
        y2, h2, c2 = seglstm.streaming_forward(x, h, c)
    
    assert torch.mean((y1 - y2).abs()) < 1e-6, f"mean abs error: {torch.mean((y1 - y2).abs())}, max_error: {(y1 - y2).abs().max()}"
    assert torch.mean((h1 - h2).abs()) < 1e-6, f"mean abs error: {torch.mean((h1 - h2).abs())}, max_error: {(h1 - h2).abs().max()}"
    assert torch.mean((c1 - c2).abs()) < 1e-6, f"mean abs error: {torch.mean((c1 - c2).abs())}, max_error: {(c1 - c2).abs().max()}"


def test_streaming_skim_no_overlap():
    model = StreamingSkiM(5, 20, 5, seg_size=10, seg_overlap=False, causal=True, n_blocks=4, embed_dim=10, embed_norm=True, embed_fusion='FiLM', block_with_embed=[1, 1, 1, 1])
    model.eval()
    x = torch.rand(1, 5, 1000).float().detach()
    d = torch.rand(1, 10).float().detach()
    with torch.no_grad():
        y1 = model(x, d)

    y2 = []
    seg_h = None
    seg_c = None
    mem_h_hidden = None
    mem_c_hidden = None
    with torch.no_grad():
        for i in range(x.shape[-1]//10):
            x_inp = x[..., i*10:(i+1)*10].float().detach()
            out, seg_h, mem_h_hidden, seg_c, mem_c_hidden = model.step_chunk(x_inp.permute(0, 2, 1), seg_h, mem_h_hidden, seg_c, mem_c_hidden, d)
            y2.append(out)

    y2 = torch.cat(y2, dim=-1)
    print(y1[:, 0, :])
    print(y2[:, 0, :])
    assert torch.mean((y1 - y2).abs()) < 1e-7, f"mean abs error: {torch.mean((y1 - y2).abs())}, max_error: {(y1 - y2).abs().max()}"

    model.init_status()
    y3 = []
    for fid in range(x.shape[-1]):
        x_inp = x[..., fid].view(1, -1, 1).float().detach()
        y3.append(model.step_frame(x_inp, d))
        
    y3 = torch.cat(y3, dim=-1)
    print(y1[:, 0, :])
    print(y3[:, 0, :])
    assert torch.mean((y1 - y3).abs()) < 1e-7, f"mean abs error: {torch.mean((y1 - y3).abs())}, max_error: {(y1 - y3).abs().max()}"
    assert torch.mean((y2 - y3).abs()) < 1e-7, f"mean abs error: {torch.mean((y2 - y3).abs())}, max_error: {(y2 - y3).abs().max()}"
