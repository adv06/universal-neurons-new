import os
import copy
import argparse
import einops
import torch
import pandas as pd
import tqdm
from transformer_lens import HookedTransformer
from utils import timestamp, adjust_precision, vector_histogram, vector_moments


def load_composition_scores():
    raise NotImplementedError


def compute_neuron_composition(model, layer, zero_diag=False):
    """
    Compute cosine similarities between all layers' MLP in/out weights and the given layer's weights.
    Returns four tensors of shape [layers, layers, neurons].
    """
    device = next(model.parameters()).device
    # rearrange fc1 weights: [layers, d_model, d_mlp] -> [layers, neurons, d_model]
    W_in = einops.rearrange(model.W_in, 'l d n -> l n d').to(device)
    # rearrange fc2 weights: [layers, d_mlp, d_model] -> [layers, neurons, d_model]
    W_out = einops.rearrange(model.W_out, 'l n d -> l n d').to(device)

    # normalize
    W_in = W_in / torch.norm(W_in, dim=-1, keepdim=True)
    W_out = W_out / torch.norm(W_out, dim=-1, keepdim=True)

    # compute all four compositions with consistent axes
    in_in_cos  = torch.einsum('lnd,md->mln', W_in,  W_in[layer])
    in_out_cos = torch.einsum('lnd,md->mln', W_out, W_in[layer])
    out_in_cos = torch.einsum('lnd,md->mln', W_in,  W_out[layer])
    out_out_cos= torch.einsum('lnd,md->mln', W_out, W_out[layer])

    if zero_diag:
        diag = torch.arange(in_in_cos.size(1), device=device)
        in_in_cos[:,diag,diag] = 0
        in_out_cos[:,diag,diag] = 0
        out_in_cos[:,diag,diag] = 0
        out_out_cos[:,diag,diag] = 0

    return in_in_cos, in_out_cos, out_in_cos, out_out_cos


def compute_attention_composition(model, layer):
    device = next(model.parameters()).device
    W_in = einops.rearrange(model.W_in[layer], 'd n -> n d').to(device)
    W_in /= torch.norm(W_in, dim=-1, keepdim=True)
    W_out = model.W_out[layer].to(device)
    W_out /= torch.norm(W_out, dim=-1, keepdim=True)

    k_comps, q_comps, v_comps, o_comps = [], [], [], []
    for attn_layer in range(model.cfg.n_layers):
        W_QK = model.QK[attn_layer].T.AB.to(device)
        W_QK /= torch.norm(W_QK, dim=(1, 2), keepdim=True)
        k_comps.append(einops.einsum(W_QK, W_out, 'h q d, n d -> n h q').norm(dim=-1))
        q_comps.append(einops.einsum(W_QK, W_out, 'h d k, n d -> n h k').norm(dim=-1))

        W_OV = model.OV[attn_layer].T.AB.to(device)
        W_OV /= torch.norm(W_OV, dim=(1, 2), keepdim=True)
        v_comps.append(einops.einsum(W_OV, W_out, 'h o d, n d -> n h o').norm(dim=-1))
        o_comps.append(einops.einsum(W_OV, W_in, 'h d v, n d -> n h v').norm(dim=-1))

    return (
        torch.stack(k_comps, dim=1),
        torch.stack(q_comps, dim=1),
        torch.stack(v_comps, dim=1),
        torch.stack(o_comps, dim=1)
    )


def compute_vocab_composition(model, layer):
    device = next(model.parameters()).device
    W_in = einops.rearrange(model.W_in[layer], 'd n -> n d').to(device)
    W_in /= torch.norm(W_in, dim=-1, keepdim=True)
    W_out = model.W_out[layer].to(device)
    W_out /= torch.norm(W_out, dim=-1, keepdim=True)

    W_E = model.W_E.to(device)
    W_E /= torch.norm(W_E, dim=-1, keepdim=True)
    W_U = model.W_U.to(device)
    W_U /= torch.norm(W_U, dim=0, keepdim=True)

    in_E_cos = einops.einsum(W_E, W_in, 'v d, n d -> n v')
    in_U_cos = einops.einsum(W_U, W_in, 'd v, n d -> n v')
    out_E_cos = einops.einsum(W_E, W_out, 'v d, n d -> n v')
    out_U_cos = einops.einsum(W_U, W_out, 'd v, n d -> n v')

    return in_E_cos, in_U_cos, out_E_cos, out_U_cos


def compute_neuron_statistics(model):
    device = next(model.parameters()).device
    W_in = einops.rearrange(model.W_in, 'l d n -> l n d').to(device)
    W_out = model.W_out.to(device)

    layers, d_mlp, _ = W_in.shape
    W_in_norms = torch.norm(W_in, dim=-1)
    W_out_norms = torch.norm(W_out, dim=-1)
    dot_product = (W_in * W_out).sum(dim=-1)
    cos_sim = dot_product / (W_in_norms * W_out_norms)

    W_in_norms = W_in_norms.detach().cpu()
    W_out_norms = W_out_norms.detach().cpu()
    b_in = model.b_in.to(device).detach().cpu()
    cos_sim = cos_sim.detach().cpu()

    index = pd.MultiIndex.from_product(
        [range(layers), range(d_mlp)],
        names=["layer", "neuron_ix"]
    )
    stat_df = pd.DataFrame({
        "input_weight_norm": W_in_norms.flatten().numpy(),
        "input_bias": b_in.flatten().numpy(),
        "output_weight_norm": W_out_norms.flatten().numpy(),
        "in_out_sim": cos_sim.flatten().numpy()
    }, index=index)
    return stat_df


def run_weight_summary(args, model):
    save_path = os.path.join(args.save_path, args.model, 'weights', str(args.checkpoint))
    os.makedirs(save_path, exist_ok=True)

    stat_df = compute_neuron_statistics(model)
    stat_df.to_csv(os.path.join(save_path, 'neuron_stats.csv'))

    device = next(model.parameters()).device
    k_list, q_list, v_list, o_list = [], [], [], []
    for layer in tqdm.tqdm(range(model.cfg.n_layers), desc="Attn layers"):
        k, q, v, o = compute_attention_composition(model, layer)
        k_list.append(k); q_list.append(q)
        v_list.append(v); o_list.append(o)
    torch.save(torch.stack(k_list).to(torch.float16).cpu(), os.path.join(save_path, 'k_comps.pt'))
    torch.save(torch.stack(q_list).to(torch.float16).cpu(), os.path.join(save_path, 'q_comps.pt'))
    torch.save(torch.stack(v_list).to(torch.float16).cpu(), os.path.join(save_path, 'v_comps.pt'))
    torch.save(torch.stack(o_list).to(torch.float16).cpu(), os.path.join(save_path, 'o_comps.pt'))

    bin_edges = torch.linspace(-1, 1, 100, device=device)
    vocab_comp = {t: {k: [] for k in [
        'top_vocab_value','top_vocab_ix','bottom_vocab_value','bottom_vocab_ix',
        'comp_hist','comp_mean','comp_var','comp_skew','comp_kurt'
    ]} for t in ['E_in','U_in','E_out','U_out']}
    for layer in tqdm.tqdm(range(model.cfg.n_layers), desc="Vocab layers"):
        comps = compute_vocab_composition(model, layer)
        for tname, comp in zip(vocab_comp.keys(), comps):
            comp = comp.to(device)
            hist = vector_histogram(comp.cpu(), bin_edges.cpu())
            top, top_ix = torch.topk(comp, 100, dim=1)
            bottom, bottom_ix = torch.topk(comp, 100, dim=1, largest=False)
            mean, var, skew, kurt = vector_moments(comp)
            d = vocab_comp[tname]
            d['comp_hist'].append(hist.cpu())
            d['top_vocab_value'].append(top.cpu())
            d['top_vocab_ix'].append(top_ix.cpu())
            d['bottom_vocab_value'].append(bottom.cpu())
            d['bottom_vocab_ix'].append(bottom_ix.cpu())
            d['comp_mean'].append(mean.cpu())
            d['comp_var'].append(var.cpu())
            d['comp_skew'].append(skew.cpu())
            d['comp_kurt'].append(kurt.cpu())
    for tname, dd in vocab_comp.items():
        torch.save({k: torch.stack(v) for k, v in dd.items()},
                   os.path.join(save_path, f'{tname.lower()}_comps.pt'))

    neuron_comp = {t: {k: [] for k in [
        'top_neuron_value','top_neuron_ix','bottom_neuron_value','bottom_neuron_ix',
        'comp_hist','comp_mean','comp_var','comp_skew','comp_kurt'
    ]} for t in ['in_in','in_out','out_in','out_out']}
    for layer in tqdm.tqdm(range(model.cfg.n_layers), desc="Neuron layers"):
        comps = compute_neuron_composition(model, layer, zero_diag=True)
        for tname, comp in zip(neuron_comp.keys(), comps):
            comp = comp.to(device)
            comp_flat = einops.rearrange(comp, 'm l n -> m (l n)')
            hist = vector_histogram(comp_flat.cpu(), bin_edges.cpu())
            top, top_ix = torch.topk(comp_flat, 20, dim=1)
            bottom, bottom_ix = torch.topk(comp_flat, 20, dim=1, largest=False)
            mean, var, skew, kurt = vector_moments(comp_flat)
            dd = neuron_comp[tname]
            dd['comp_hist'].append(hist.cpu())
            dd['top_neuron_value'].append(top.cpu())
            dd['top_neuron_ix'].append(top_ix.cpu())
            dd['bottom_neuron_value'].append(bottom.cpu())
            dd['bottom_neuron_ix'].append(bottom_ix.cpu())
            dd['comp_mean'].append(mean.cpu())
            dd['comp_var'].append(var.cpu())
            dd['comp_skew'].append(skew.cpu())
            dd['comp_kurt'].append(kurt.cpu())
    for tname, dd in neuron_comp.items():
        torch.save({k: torch.stack(v) for k, v in dd.items()},
                   os.path.join(save_path, f'{tname}_comps.pt'))

    print('finished weight summary')


def run_full_weight_analysis(model, save_precision=8, save_path='results/weights'):
    save_path = os.path.join(save_path, model.cfg.model_name)
    os.makedirs(save_path, exist_ok=True)

    print(f'{timestamp()} starting full analysis')
    stat_df = compute_neuron_statistics(model)
    stat_df.to_csv(os.path.join(save_path, 'neuron_stats.csv'))
    print(f'{timestamp()} saved neuron stats')

    for layer in range(model.cfg.n_layers):
        in_in, in_out, out_in, out_out = compute_neuron_composition(model, layer)
        for name, tensor in zip(['in_in','in_out','out_in','out_out'],
                                [in_in, in_out, out_in, out_out]):
            tensor = tensor.to(device)
            path = os.path.join(save_path, f'{name}_{layer}.pt')
            torch.save(adjust_precision(tensor, save_precision, per_channel=False, cos_sim=True).cpu(), path)
        print(f'{timestamp()} saved neuron cosines for layer {layer}')

    print(f'{timestamp()} full analysis complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', default='stanford-crfm/alias-gpt2-small-x21')
    parser.add_argument('--checkpoint', type=int, default=None)  
    parser.add_argument('--save_precision', type=int, default=8, choices=[8,16,32])
    parser.add_argument('--save_path', default='weight_data')
    parser.add_argument('--compute_full_stats', action='store_true')
    args = parser.parse_args()

    print(f'{timestamp()} loading model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        args.model,
        device=device,
        checkpoint_value=args.checkpoint
    )
    model.to(device)

    if args.compute_full_stats:
        run_full_weight_analysis(
            model,
            save_precision=args.save_precision,
            save_path=args.save_path
        )
    else:
        run_weight_summary(args, model)
