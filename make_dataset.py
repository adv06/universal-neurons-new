import os
import argparse
import datasets
import torch
from transformer_lens import HookedTransformer

def chunkify(tok_list, chunk_size):
    """Split a flat list of tokens into list of length‐chunk_size lists."""
    return [tok_list[i:i+chunk_size] for i in range(0, len(tok_list), chunk_size)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', default='gpt2',
        help='Name of the transformer‐lens model to load')
    parser.add_argument(
        '--hf_dataset', default='NeelNanda/pile-10k',
        help='HuggingFace dataset ID for the Pile')
    parser.add_argument(
        '--hf_split', default='train',
        help='Which split of the HF dataset to stream')
    parser.add_argument(
        '--ctx_len', default=512, type=int,
        help='Max context length / sequence length')
    parser.add_argument(
        '--n_tokens', default=20000000, type=int,
        help='Total number of tokens to accumulate') 
    parser.add_argument(
        '--output_dir', default='token_datasets',
        help='Where to save the tokenized dataset')
    args = parser.parse_args()

    # Load model and get actual context length from config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HookedTransformer.from_pretrained(args.model, device=device)
    tokenizer = model.tokenizer
    
    # Use model's actual context length if user requested longer
    ctx_len = min(args.ctx_len, model.cfg.n_ctx)
    print(f"Using context length: {ctx_len} (model max: {model.cfg.n_ctx})")

    # Stream dataset and process in chunks
    ds_stream = datasets.load_dataset(
        args.hf_dataset,
        split=args.hf_split,
        streaming=True
    )

    all_tokens = []
    total = 0
    for ex in ds_stream:
        # Tokenize with truncation to model's max length
        toks = tokenizer.encode(
            ex['text'], 
            truncation=True,  # Enable truncation
            max_length=ctx_len,  # Truncate to context length
            return_tensors='np'  # Get numpy array for memory efficiency
        )[0].tolist()
        
        # Add tokens if we have space
        remaining = args.n_tokens - total
        if remaining <= 0:
            break
            
        add_toks = toks[:remaining]
        all_tokens.extend(add_toks)
        total += len(add_toks)


    # Chunk into sequences of length ≤ model's context window
    seqs = chunkify(all_tokens, ctx_len)


    # Build and save dataset
    hf_ds = datasets.Dataset.from_dict({'tokens': seqs})
    model_family = args.model.replace('/', '_')
    save_path = os.path.join(args.output_dir, model_family)
    os.makedirs(save_path, exist_ok=True)
    hf_ds.save_to_disk(os.path.join(save_path, f"pile_{args.n_tokens:,}_tokens"))

