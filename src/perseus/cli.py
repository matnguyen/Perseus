#!/usr/bin/env python3
"""
perseus_cli.py
==============
Unified command-line interface for Perseus operations.

Provides subcommands for:
  - filter: Filter Kraken outputs using a trained model
  - extract: Extract features from Kraken output (inference mode)
  - extract-train: Extract features from Kraken output with training labels
  - train: Train a Perseus model

Usage:
    perseus_cli.py filter --help
    perseus_cli.py extract --help
    perseus_cli.py extract-train --help
    perseus_cli.py train --help
"""

import sys
import argparse


def main():
    """Main entry point for the Perseus CLI."""
    parser = argparse.ArgumentParser(
        prog="perseus",
        description="Perseus - Taxonomic classification refinement tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter Kraken output using a trained model
  perseus filter --input-shards shards/ --input-kraken raw.txt --model-path model.pt --output-path filtered.txt
  
  # Extract features for inference
  perseus extract kraken_output.txt features/ --format shards
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True
    
    # ==================== FILTER SUBCOMMAND ====================
    filter_parser = subparsers.add_parser(
        'filter',
        help='Filter Kraken outputs using a trained Perseus model',
        description='Filter Kraken outputs using a trained Perseus model.'
    )
    filter_parser.add_argument("--input-shards", type=str, required=True,
                               help="Path to the input manifest file containing sequences to filter.")
    filter_parser.add_argument("--input-kraken", type=str, required=True,
                               help="Path to the Kraken output file to be filtered.")
    filter_parser.add_argument("--batch-size", type=int, default=128,
                               help="Batch size for processing sequences.")
    filter_parser.add_argument("--output-path", type=str, required=True,
                               help="Path to save the filtered Kraken output.")
    filter_parser.add_argument("--cache-shards", type=int, default=1,
                               help="Shards kept in RAM per worker")
    filter_parser.add_argument("--downcast", choices=["none","fp16"], default="fp16",
                               help="Downcast shard tensors in cache")
    filter_parser.add_argument("--cpu-float32", action="store_true",
                               help="Cast samples to float32 on CPU before batching")
    filter_parser.add_argument("--num-workers", type=int, default=4,
                               help="Number of DataLoader workers")
    filter_parser.add_argument("--calibration-dir", type=str, default=None,
                               help="Directory containing calibrators")
    filter_parser.add_argument("--split-dir", type=str, default=None,
                               help='Directory containing train/val splits (if applicable)')
    filter_parser.add_argument("--seed", type=int, default=667,
                               help="Random seed for reproducibility")
    filter_parser.add_argument("--select-one-per-seq", action="store_true",
                               help="Select one row per sequence ID from Kraken output using model probabilities")
    filter_parser.add_argument("--model-path", type=str,
                               help="Path to the trained perseus model file.")
    
    # ==================== EXTRACT SUBCOMMAND ====================
    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract features from Kraken output (inference mode)',
        description='Stream Kraken output and extract per-(seq,taxon) features for inference.'
    )
    extract_parser.add_argument('file_path', type=str,
                                help='Path to the Kraken output file')
    extract_parser.add_argument('output_path', type=str,
                                help='Path to output directory')
    extract_parser.add_argument('--rows-per-chunk', type=int, default=20000,
                                help='Rows per DataFrame chunk for pools')
    extract_parser.add_argument('--max-bins-per-seq', type=int, default=None,
                                help='Max bins per (seq_id, taxon) (default: None)')
    extract_parser.add_argument('--shard-size', type=int, default=4096,
                                help='Samples per shard (.pt)')
    extract_parser.add_argument('--target-length', type=int, default=0,
                                help='Resample time to this length for shards (0 = pad to shard max)')
    extract_parser.add_argument('--to-dtype', choices=['float32','float16','bfloat16'],
                                default='float32', help='Stored dtype for shard tensor')
    extract_parser.add_argument('--min-tax-kmers', type=int, default=0,
                                help='Minimum k-mers assigned to a taxon for it to be considered')
    extract_parser.add_argument('--threads', type=int, default=0,
                                help='Number of worker processes (0=auto)')
    extract_parser.add_argument('--log-level', default='INFO',
                                choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                                help='Set the logging level (default: INFO)')
    
    # ==================== EXTRACT-TRAIN SUBCOMMAND ====================
    # extract_train_parser = subparsers.add_parser(
    #     'extract-train',
    #     help='Extract features from Kraken output with training labels',
    #     description='Stream Kraken output and extract features with Option-B training labels.'
    # )
    # extract_train_parser.add_argument('file_path', type=str,
    #                                   help='Path to the Kraken output file')
    # extract_train_parser.add_argument('output_path', type=str,
    #                                   help='Path to save the processed output (dir or .parquet)')
    # extract_train_parser.add_argument('--rows-per-chunk', type=int, default=20000,
    #                                   help='Rows per DataFrame chunk for pools')
    # extract_train_parser.add_argument('--max-bins-per-seq', type=int, default=None,
    #                                   help='Max bins per (seq_id, taxon) (default: None)')
    # extract_train_parser.add_argument('--format', choices=['parquet','shards'], default='parquet',
    #                                   help='Output format: parquet (original) or shards (.pt, channel-first)')
    # extract_train_parser.add_argument('--shard-size', type=int, default=4096,
    #                                   help='Samples per shard (.pt)')
    # extract_train_parser.add_argument('--target-length', type=int, default=1024,
    #                                   help='Resample time to this length for shards (0 = pad to shard max)')
    # extract_train_parser.add_argument('--to-dtype', choices=['float32','float16','bfloat16'],
    #                                   default='float32', help='Stored dtype for shard tensor')
    # extract_train_parser.add_argument('--mess-truth-file', type=str, default=None,
    #                                   help='Path to MESS truth file for Option-B labeling')
    # extract_train_parser.add_argument('--mess-input-file', type=str, default=None,
    #                                   help='Path to MESS input file for Option-B labeling')
    # extract_train_parser.add_argument('--topk-taxa', type=int, default=None,
    #                                   help='Number of top taxa to consider per sequence for Option-B labeling')
    # extract_train_parser.add_argument('--min-tax-kmers', type=int, default=10,
    #                                   help='Minimum k-mers assigned to a taxon for it to be considered')
    # extract_train_parser.add_argument('--neg-extra', type=int, default=None,
    #                                   help='Number of extra negative taxa to sample per sequence')
    # extract_train_parser.add_argument('--threads', type=int, default=0,
    #                                   help='Number of worker processes (0=auto)')
    # extract_train_parser.add_argument('--log-level', default='INFO',
    #                                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    #                                   help='Set the logging level (default: INFO)')
    
    # ==================== TRAIN SUBCOMMAND ====================
    # train_parser = subparsers.add_parser(
    #     'train',
    #     help='Train a Perseus model',
    #     description='Train from (train,val) shard manifests with Option-B labels.'
    # )
    # train_parser.add_argument("--train", required=False,
    #                           help="Train shard directory OR train_manifest.json")
    # train_parser.add_argument("--val", required=False,
    #                           help="Val shard directory OR val_manifest.json")
    # train_parser.add_argument("--input", required=False,
    #                           help="Shard directory OR single manifest (used for both if --val missing)")
    # train_parser.add_argument("--epochs", type=int, default=20,
    #                           help="Number of training epochs")
    # train_parser.add_argument("--batch", type=int, default=64,
    #                           help="Batch size")
    # train_parser.add_argument("--lr", type=float, default=1e-3,
    #                           help="Learning rate")
    # train_parser.add_argument("--num-workers", type=int, default=2,
    #                           help="Number of DataLoader workers")
    # train_parser.add_argument("--model", choices=["cnn","restcn"], default="cnn",
    #                           help="Model architecture")
    # train_parser.add_argument("--save", default="model_cf.pt",
    #                           help="Path to save the trained model")
    # train_parser.add_argument("--target", choices=["any","rank","per-rank"], default="per-ranks",
    #                           help="Training target mode")
    
    # train_group = train_parser.add_mutually_exclusive_group()
    # train_group.add_argument("--rank", choices=["superkingdom","phylum","class","order",
    #                                             "family","genus","species"],
    #                         help="Use only samples with this predicted canonical rank")
    # train_group.add_argument("--ranks", action="store_true",
    #                         help="Train one model per predicted canonical rank (loop)")
    
    # train_parser.add_argument("--rank_cache", default=None,
    #                           help="Optional cache path for rank index")
    # train_parser.add_argument("--log-level", default="INFO",
    #                           help="DEBUG | INFO | WARNING | ERROR | CRITICAL")
    # train_parser.add_argument("--crop-max", type=int, default=1024,
    #                           help="Max crop length for TRAIN loader (no crop for VAL)")
    # train_parser.add_argument("--cache-shards", type=int, default=1,
    #                           help="Shards kept in RAM per worker")
    # train_parser.add_argument("--downcast", choices=["none","fp16"], default="fp16",
    #                           help="Downcast shard tensors in cache")
    # train_parser.add_argument("--cpu-float32", action="store_true",
    #                           help="Cast samples to float32 on CPU before batching")
    # train_parser.add_argument("--split-dir", type=str, default=None,
    #                           help='Directory containing train/val splits')
    # train_parser.add_argument("--val-samples-per-shard", type=int, default=None,
    #                           help="(Val only) Number of samples to draw per shard per epoch")
    # train_parser.add_argument("--seed", type=int, default=667,
    #                           help="Random seed for reproducibility")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Route to appropriate module based on command
    if args.command == 'filter':
        from perseus import filter as filter_module
        # Convert parsed args back to sys.argv format for the module
        sys.argv = ['filter']
        for key, value in vars(args).items():
            if key == 'command':
                continue
            if isinstance(value, bool):
                if value:
                    sys.argv.append(f'--{key.replace("_", "-")}')
            elif value is not None:
                sys.argv.append(f'--{key.replace("_", "-")}')
                sys.argv.append(str(value))
        # Re-import to trigger __main__ block
        import runpy
        runpy.run_module('perseus.filter', run_name='__main__')
        
    elif args.command == 'extract':
        from perseus import extract as extract_module
        sys.argv = ['extract', args.file_path, args.output_path]
        for key, value in vars(args).items():
            if key in ['command', 'file_path', 'output_path']:
                continue
            if isinstance(value, bool):
                if value:
                    sys.argv.append(f'--{key.replace("_", "-")}')
            elif value is not None:
                sys.argv.append(f'--{key.replace("_", "-")}')
                sys.argv.append(str(value))
        import runpy
        runpy.run_module('perseus.extract', run_name='__main__')
        
    # elif args.command == 'extract-train':
    #     from perseus import extract_train_data
    #     sys.argv = ['extract_train_data', args.file_path, args.output_path]
    #     for key, value in vars(args).items():
    #         if key in ['command', 'file_path', 'output_path']:
    #             continue
    #         if isinstance(value, bool):
    #             if value:
    #                 sys.argv.append(f'--{key.replace("_", "-")}')
    #         elif value is not None:
    #             sys.argv.append(f'--{key.replace("_", "-")}')
    #             sys.argv.append(str(value))
    #     import runpy
    #     runpy.run_module('perseus.extract_train_data', run_name='__main__')
        
    # elif args.command == 'train':
    #     from perseus import train as train_module
    #     sys.argv = ['train']
    #     for key, value in vars(args).items():
    #         if key == 'command':
    #             continue
    #         if isinstance(value, bool):
    #             if value:
    #                 sys.argv.append(f'--{key.replace("_", "-")}')
    #         elif value is not None:
    #             sys.argv.append(f'--{key.replace("_", "-")}')
    #             sys.argv.append(str(value))
    #     import runpy
    #     runpy.run_module('perseus.train', run_name='__main__')


if __name__ == '__main__':
    main()
