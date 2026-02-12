#!/usr/bin/env python3
"""
Convenience runner for all training entry points.

Usage:
    python run.py <command>
    
Commands:
    bpe          - Train and test BPE tokenizer
    skipgram     - Train skip-gram embeddings
    lm           - Train simple language model
    lm-pos       - Train position-aware language model
    rnn          - Train RNN language model
    all          - Run all training scripts
"""
import sys
import subprocess
from pathlib import Path


ENTRY_POINTS = {
    "bpe": "entry_points/train_bpe.py",
    "skipgram": "entry_points/train_skipgram.py",
    "lm": "entry_points/train_lm.py",
    "lm-pos": "entry_points/train_lm_positional.py",
    "rnn": "entry_points/train_rnn.py",
}


def print_usage():
    print("Usage: python run.py <command>")
    print()
    print("Commands:")
    print("  bpe          - Train and test BPE tokenizer")
    print("  skipgram     - Train skip-gram embeddings")
    print("  lm           - Train simple language model")
    print("  lm-pos       - Train position-aware language model")
    print("  rnn          - Train RNN language model")
    print("  all          - Run all training scripts")
    print()


def run_script(script_path):
    """Run a Python script and return success status."""
    full_path = Path(__file__).parent / script_path
    if not full_path.exists():
        print(f"Error: Script not found: {full_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running: {script_path}")
    print('='*60 + '\n')
    
    result = subprocess.run([sys.executable, str(full_path)])
    return result.returncode == 0


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "all":
        success = True
        for name, script in ENTRY_POINTS.items():
            if not run_script(script):
                success = False
                print(f"\nFailed: {name}")
            else:
                print(f"\nCompleted: {name}")
        
        if success:
            print("\n" + "="*60)
            print("All scripts completed successfully!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("Some scripts failed!")
            print("="*60)
            sys.exit(1)
    
    elif command in ENTRY_POINTS:
        if not run_script(ENTRY_POINTS[command]):
            sys.exit(1)
    
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
