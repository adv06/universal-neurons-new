import pandas as pd
import torch
from transformer_lens import HookedTransformer

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
# 2) Then add your project root if you still need it:
PROJECT_ROOT = "/content/universal-neurons-new"
sys.path.insert(1, PROJECT_ROOT)


from summary_viewer import load_dataset_summary, load_weights_summary
from activations import get_activation_sparsity_df, make_pile_subset_distribution_activation_summary_df
from weights import neuron_vocab_cosine_moments
from correlations import make_correlation_result_df

def make_neuron_stat_df(model_name, dataset_name):
    dataset_summaries = load_dataset_summary(model_name, dataset_name)
    weight_summaries = load_weights_summary(model_name)
    dataset_summaries = {dataset_name: dataset_summaries}

    neuron_df = weight_summaries['neuron_stats']
    neuron_df = neuron_df.rename(
        columns={'neuron_ix': 'neuron'}).set_index(['layer', 'neuron'])
    neuron_df['weight_norm_penalty'] = neuron_df.input_weight_norm.values**2 + \
        neuron_df.output_weight_norm.values**2
    neuron_df.head()

    try:
        vocab_comps = torch.load(
            f'summary_data/{model_name}/weights/vocab_comps.pt')

        n_layers = neuron_df.reset_index().layer.max() + 1
        d_mlp = neuron_df.reset_index().neuron.max() + 1
        neuron_vocab_moment_df = pd.DataFrame({
            'vocab_mean': vocab_comps['U_out']['comp_mean'].flatten().numpy(),
            'vocab_var': vocab_comps['U_out']['comp_var'].flatten().numpy(),
            'vocab_skew': vocab_comps['U_out']['comp_skew'].flatten().numpy(),
            'vocab_kurt': vocab_comps['U_out']['comp_kurt'].flatten().numpy(),
        }, index=pd.MultiIndex.from_product([range(n_layers), range(d_mlp)]))
        neuron_vocab_moment_df.index.names = ['layer', 'neuron']
    except FileNotFoundError:
        print('Cached vocab moments not found, computing now')
        model = HookedTransformer.from_pretrained(model_name, device='cpu')
        model.requires_grad_(False)

        neuron_vocab_moment_df = neuron_vocab_cosine_moments(model)

        del model

    bin_edges = torch.linspace(-10, 15, 256)
    sparsity_df = get_activation_sparsity_df(dataset_summaries, bin_edges)

    act_moments_df = make_pile_subset_distribution_activation_summary_df(
        dataset_summaries, bin_edges, include_all=True)

    full_distr_moment_df = act_moments_df.swaplevel(0, 1, axis=1)['all']

    stat_df = pd.concat([
        neuron_df,
        full_distr_moment_df,
        neuron_vocab_moment_df,
        pd.DataFrame(sparsity_df['all']).rename(columns={'all': 'sparsity'})
    ], axis=1)

    stat_df = stat_df.rename(columns={
        'input_weight_norm': 'w_in_norm',
        'output_weight_norm': 'w_out_norm',
        'weight_norm_penalty': 'l2_penalty'
    })

    return stat_df


def make_corr_compare_df(all_corr_df):
    max_all_corr = all_corr_df.reset_index().groupby(
        ['layer', 'neuron']).max_corr.max()
    mean_all_corr = all_corr_df.reset_index().groupby(
        ['layer', 'neuron']).max_corr.mean()
    min_all_corr = all_corr_df.reset_index().groupby(
        ['layer', 'neuron']).max_corr.min()

    max_all_baseline = all_corr_df.reset_index().groupby(
        ['layer', 'neuron']).baseline.max()
    mean_all_baseline = all_corr_df.reset_index().groupby(
        ['layer', 'neuron']).baseline.mean()
    min_all_baseline = all_corr_df.reset_index().groupby(
        ['layer', 'neuron']).baseline.min()

    compare_df = pd.DataFrame({
        'max_corr': max_all_corr,
        'mean_corr': mean_all_corr,
        'min_corr': min_all_corr,
        'max_baseline': max_all_baseline,
        'min_baseline': min_all_baseline,
        'mean_baseline': mean_all_baseline
    }).reset_index().set_index(['layer', 'neuron'])

    return compare_df

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_1_name', default='pythia-70m',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--model_2_name', default='pythia-70m-v0')
    parser.add_argument(
        '--token_dataset', type=str)

    parser.add_argument(
        '--baseline', type=str, default='none',
        choices=['none', 'gaussian', 'permutation', 'rotation'])
    parser.add_argument(
        '--similarity_type', type=str, default='pearson',
        choices=['pearson', 'jaccard', 'cosine'])
    parser.add_argument(
        '--jaccard_threshold', type=float, default=0)

    parser.add_argument(
        '--batch_size', default=32, type=int)
    parser.add_argument(
        '--model_1_device', type=str, default='cpu')
    parser.add_argument(
        '--model_2_device', type=str, default='cpu')
    parser.add_argument(
        '--correlation_device', type=str, default='cpu')

    parser.add_argument(
        '--save_full_correlation_matrix', action='store_true',
        help='Whether to save the full correlation matrix (always save the summary)')
    parser.add_argument(
        '--save_precision', type=int, default=16, choices=[8, 16, 32],
        help='Number of bits to use for saving full correlation matrix')

    args = parser.parse_args()


    no_c = make_neuron_stat_df("gpt2", "pile")
    c = make_correlation_result_df(args.model_1_name, args.model_2_name, args.token_dataset, args.similarity_type, args.baseline, result_dir='correlation_results')
    combined_df = pd.concat([no_c, c], axis=1)
    combined_df.to_csv('combined_neuron_stats.csv', index=False)
    

  
