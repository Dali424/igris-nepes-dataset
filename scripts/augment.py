#!/usr/bin/env python3
"""Run video or native rendering augmentation with the dataset Python."""
import argparse
import os
from pathlib import Path
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset-python', default=os.environ.get('NEPES_DATASET_PYTHON', sys.executable))
    args, _ = parser.parse_known_args()
    executable = Path(args.dataset_python).expanduser().absolute()
    if not executable.is_file():
        raise SystemExit('Set --dataset-python or NEPES_DATASET_PYTHON to the dataset environment Python')
    script = Path(__file__).resolve().parent / 'augmentation' / 'augment.py'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    os.execve(str(executable), [str(executable), '-B', str(script), *sys.argv[1:]], env)
