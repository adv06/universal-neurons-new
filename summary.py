import os
import numpy as np
import tqdm
import torch
import einops
import datasets
import argparse
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from utils import *

# Silence HuggingFace tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def bin_activations(acts, edges, counts):
    # acts: [L, D, T] → flatten T
    L, D, T = acts.shape
    flat = acts.reshape(L * D, T).contiguous()      # make contiguous for searchsorted
    edges_c = edges.contiguous()
    bins = torch.searchsorted(edges_c, flat)        # [L*D, T]
    # bincount per row
    bc = torch.stack([
        torch.bincount(bins[i], minlength=edges_c.size(0) + 1)
        for i in range(L * D)
    ], dim=0)
    counts[:] += bc.view(L, D, -1).to(counts.dtype)


def update_top_k(acts, idxs, vals, offset):
    # acts: [L, D, T], idxs/vals: [L, D, K]
    L, D, T = acts.shape
    flat = acts.reshape(L * D, T)
    positions = torch.arange(T, device=acts.device) + offset
    flat_idx = idxs.view(-1, idxs.size(-1))
    flat_val = vals.view(-1, vals.size(-1))
    for i in range(L * D):
        a = torch.cat([flat_val[i], flat[i]])
        b = torch.cat([flat_idx[i], positions])
        topk = torch.topk(a, flat_val.size(1))
        flat_val[i] = a[topk.indices]
        flat_idx[i] = b[topk.indices]


def save_act(x, hook):
    hook.ctx['a'] = x.detach().cpu().float()


def summarize(args, model, ds, device):
    dmlp, nlay = model.cfg.d_mlp, model.cfg.n_layers
    L = min(nlay, args.max_layers)

    edges  = torch.linspace(-10, 15, args.n_bins)
    counts = torch.zeros(L, dmlp, args.n_bins + 1, dtype=torch.int32)
    idxs   = torch.zeros(L, dmlp, args.top_k, dtype=torch.int64)
    vals   = torch.zeros(L, dmlp, args.top_k, dtype=torch.float32)

    pre  = [f'blocks.{l}.mlp.hook_pre'  for l in range(L)]
    post = [f'blocks.{l}.mlp.hook_post' for l in range(L)]
    hooks = [(h, save_act) for h in pre + post]

    dl = DataLoader(ds['tokens'], batch_size=args.batch_size, shuffle=False)
    offset = 0
    for batch in tqdm.tqdm(dl, disable=not args.verbose):
        batch = batch.to(device)
        model.run_with_hooks(batch, fwd_hooks=hooks)

        p = torch.stack([model.hook_dict[h].ctx.pop('a') for h in pre], dim=2)
        q = torch.stack([model.hook_dict[h].ctx.pop('a') for h in post], dim=2)
        model.reset_hooks()

        p = einops.rearrange(p, 'b c l d -> l d (b c)')
        q = einops.rearrange(q, 'b c l d -> l d (b c)')

        bin_activations(p, edges, counts)
        update_top_k(q, idxs, vals, offset)

        b, c = batch.shape
        offset += b * c

        del p, q
        torch.cuda.empty_cache()

    out = os.path.join(args.output_dir, args.model.replace('/', '_'))
    os.makedirs(out, exist_ok=True)
    torch.save(counts, f'{out}/bin_counts.pt')
    torch.save(edges,  f'{out}/bin_edges.pt')
    torch.save(idxs,   f'{out}/topk_idxs.pt')
    torch.save(vals,   f'{out}/topk_vals.pt')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model',        default='stanford-crfm/alias-gpt2-small-x21')
    p.add_argument('--dataset_path', default='pile')
    p.add_argument('--output_dir',   default='sum_data')
    p.add_argument('--batch_size',   type=int, default=16)
    p.add_argument('--n_bins',       type=int, default=64)
    p.add_argument('--top_k',        type=int, default=20)
    p.add_argument('--max_layers',   type=int, default=4)
    p.add_argument('--verbose',      action='store_true')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HookedTransformer.from_pretrained(
        args.model, device=device, checkpoint_value=397000
    ).to(device).eval()
    torch.set_grad_enabled(False)

    ds = datasets.load_from_disk("/content/universal-neurons-new/token_datasets/gpt2/monology/pile")
    ds = ds.map(lambda x: {'tokens': np.clip(x['tokens'], 0, model.cfg.d_vocab-1)},
                batched=True, batch_size=500)
    ds.set_format(type='torch', columns=['tokens'])

    summarize(args, model, ds, device)
