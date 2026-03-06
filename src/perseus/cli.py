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
    filter_parser.add_argument("input_shards", type=str, 
                               help="Path to directory containing shard files; will search for 'manifest.json' manifest file.")
    filter_parser.add_argument("input_kraken", type=str, 
                               help="Path to the Kraken output file to be filtered.")
    filter_parser.add_argument("output_path", type=str, 
                               help="Path to save the filtered Kraken output.")
    filter_parser.add_argument("--batch-size", type=int, default=128,
                               help="Batch size for processing sequences.")
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
    filter_parser.add_argument("--output-all", action="store_true", 
                        help="Output all model probabilities for each rank instead of just the predicted taxid.")
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Route to appropriate module based on command
    if args.command == 'filter':
        # Convert parsed args back to sys.argv format for the module
        sys.argv = ['filter', args.input_shards, args.input_kraken, args.output_path]
        for key, value in vars(args).items():
            if key in ['command', 'input_shards', 'input_kraken', 'output_path']:
                continue
            if isinstance(value, bool):
                if value:
                    sys.argv.append(f'--{key.replace("_", "-")}')
            elif value is not None:
                sys.argv.append(f'--{key.replace("_", "-")}')
                sys.argv.append(str(value))
        # Re-import to trigger __main__ block
        import runpy
        runpy.run_module('perseus.commands.filter', run_name='__main__')
        
    elif args.command == 'extract':
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
        runpy.run_module('perseus.commands.extract', run_name='__main__')

if __name__ == '__main__':
    main()
