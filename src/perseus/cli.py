#!/usr/bin/env python3
"""
perseus_cli.py
==============
Unified command-line interface for Perseus operations.

Provides subcommands for:
  - filter: Filter Kraken outputs using a trained Perseus model
  - extract: Extract features from Kraken output for inference

Usage:
    perseus_cli.py filter --help
    perseus_cli.py extract --help
"""

import sys
import argparse
import logging
import runpy
from importlib.metadata import version

BANNER = r"""
____                              
|  _ \ ___ _ __ ___  ___ _   _ ___ 
| |_) / _ \ '__/ __|/ _ \ | | / __|
|  __/  __/ |  \__ \  __/ |_| \__ \
|_|   \___|_|  |___/\___|\__,_|___/
"""

def main():
    # Parent parser for common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('-v',
                               action='store_true',
                               help="Enable verbose logging"
    )
    
    parser = argparse.ArgumentParser(
        prog='perseus',
        description=BANNER + "\nPerseus - lineage-aware refinement of Kraken2 classifications",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
    # Setup ETE3 taxonomy database for Perseus
    perseus setup <path_to_directory>

    # Extract features for inference
    perseus extract <kraken_file> <output_shards_directory>
    
    # Filter Kraken2 output using pre-trained model
    perseus filter <shards_directory> <kraken_file> <output_path>
    """
    )
    
    parser.add_argument('--version',
                        action='version',
                        version=f"%(prog)s {version('perseus-metagenomics')}",
                        help='Show program version and exit'
    )
    
    subparsers = parser.add_subparsers(dest='command', help="Available commands")
    
    # ==================== setup subcommand ====================
    setup_parser = subparsers.add_parser(
        'setup',
        parents=[common_parser],
        help='Setup ETE3 taxonomic database for Perseus',
        description=BANNER + "\nSetup ETE3 taxonomic database for Perseus",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    setup_parser.add_argument('db_dir', type=str,
                              help="Directory to store ETE3 taxonomy database")
    setup_parser.add_argument('--update', action='store_true',
                              help="Force update of the ETE3 taxonomy database even if it already exists")
    
    # ==================== filter subcommand ====================
    filter_parser = subparsers.add_parser(
        'filter',
        parents=[common_parser],
        help='Filter Kraken outputs using a trained Perseus model',
        description=BANNER + "\nFilter Kraken outputs using a trained Perseus model",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    filter_parser.add_argument('input_shards', type=str, 
                               help="Path to directory containing shard files")
    filter_parser.add_argument('input_kraken', type=str, 
                               help="Path to Kraken output file to filter")
    filter_parser.add_argument('output_path', type=str, 
                               help="Path for the output filtered Kraken file")
    filter_parser.add_argument('db_dir', type=str, 
                               help="Directory containing ETE3 taxonomy database")
    filter_parser.add_argument('--batch-size', type=int, default=128,
                               help=argparse.SUPPRESS)
    filter_parser.add_argument('--cache-shards', type=int, default=1,
                               help=argparse.SUPPRESS)
    filter_parser.add_argument('--downcast', choices=['none','fp16'], default='fp16',
                               help=argparse.SUPPRESS)
    filter_parser.add_argument('--cpu-float32', action='store_true',
                               help=argparse.SUPPRESS)
    filter_parser.add_argument('--num-workers', type=int, default=4,
                               help=argparse.SUPPRESS)
    filter_parser.add_argument('--calibration-dir', type=str, default=None,
                               help=argparse.SUPPRESS)
    filter_parser.add_argument('--split-dir', type=str, default=None,
                               help=argparse.SUPPRESS)
    filter_parser.add_argument('--seed', type=int, default=667,
                               help="Random seed for reproducibility (default: 667)")
    filter_parser.add_argument('--output-all', action='store_true', 
                               help="Output all sequences instead of just the most likely taxonomic assignment for each sequence")
    filter_parser.add_argument('--model-path', type=str,
                               help=argparse.SUPPRESS)
    
    # ==================== extract subcommand ====================
    extract_parser = subparsers.add_parser(
        'extract',
        parents=[common_parser],
        help='Extract features from Kraken output',
        description=BANNER + "\nExtract features from Kraken output for inference",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    extract_parser.add_argument('file_path', type=str,
                                help="Path to the Kraken output file")
    extract_parser.add_argument('output_path', type=str,
                                help="Path to output directory")
    extract_parser.add_argument('db_dir', type=str, 
                                help="Directory containing ETE3 taxonomy database")
    extract_parser.add_argument('--rows-per-chunk', type=int, default=20000,
                                help=argparse.SUPPRESS)
    extract_parser.add_argument('--max-bins-per-seq', type=int, default=None,
                                help=argparse.SUPPRESS)
    extract_parser.add_argument('--shard-size', type=int, default=4096,
                                help=argparse.SUPPRESS)
    extract_parser.add_argument('--target-length', type=int, default=0,
                                help=argparse.SUPPRESS)
    extract_parser.add_argument('--to-dtype', choices=['float32','float16','bfloat16'],
                                default='float32', help=argparse.SUPPRESS)
    extract_parser.add_argument('--min-tax-kmers', type=int, default=0,
                                help="Minimum number of taxonomic k-mers required to include a sequence in the output (default: 0, i.e. include all sequences)")
    extract_parser.add_argument('--threads', type=int, default=0,
                                help="Number of threads to use for feature extraction (default: number of CPU cores)")
    
    subparser_map = {
        'filter': filter_parser,
        'extract': extract_parser,
        'setup': setup_parser
    }
    
    # ==================== Manual help if no command or args provided ====================
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if len(sys.argv) == 2 and sys.argv[1] in subparser_map:
        subparser_map[sys.argv[1]].print_help()
        sys.exit(0)
        
    # ====================== Parse arguments ====================
    args = parser.parse_args()
    
    # ====================== Set up logging ====================
    if args.v:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            datefmt="%H:%M:%S",
        )
    
    # ====================== Dispatch to appropriate command module ====================
    if args.command == 'setup':
        sys.argv = ['setup', args.db_dir]
        if args.update:
            sys.argv.append('--update')
        runpy.run_module('perseus.commands.setup', run_name='__main__')
        
    elif args.command == 'filter':
        # Convert parsed args back to sys.argv format for the module
        sys.argv = ['filter', args.input_shards, args.input_kraken, args.output_path, args.db_dir]
        for key, value in vars(args).items():
            if key in ['command', 'input_shards', 'input_kraken', 'output_path', 'db_dir']:
                continue
            if isinstance(value, bool):
                if value:
                    sys.argv.append(f'--{key.replace("_", "-")}')
            elif value is not None:
                sys.argv.append(f'--{key.replace("_", "-")}')
                sys.argv.append(str(value))
        runpy.run_module('perseus.commands.filter', run_name='__main__')
        
    elif args.command == 'extract':
        sys.argv = ['extract', args.file_path, args.output_path, args.db_dir]
        for key, value in vars(args).items():
            if key in ['command', 'file_path', 'output_path', 'db_dir']:
                continue
            if isinstance(value, bool):
                if value:
                    sys.argv.append(f'--{key.replace("_", "-")}')
            elif value is not None:
                sys.argv.append(f'--{key.replace("_", "-")}')
                sys.argv.append(str(value))
        runpy.run_module('perseus.commands.extract', run_name='__main__')

if __name__ == '__main__':
    main()
