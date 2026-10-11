"""Execute with this interpreter in a project-local kernel, preserving every real output."""
import argparse
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpec


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--notebook', default='scientific_claim_verification.ipynb')
    p.add_argument('--timeout', type=int, default=14400)
    args = p.parse_args()
    path = Path(args.notebook)
    notebook = nbformat.read(path, as_version=4)
    manager = AsyncKernelManager(kernel_name='python3')
    manager._kernel_spec = KernelSpec(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                                     display_name='Python (SciFact)', language='python')
    def progress(cell, cell_index, **kwargs):
        print(f'Executing cell {cell_index + 1}/{len(notebook.cells)}', flush=True)
    client = NotebookClient(notebook, timeout=args.timeout, kernel_name='python3', km=manager,
                            resources={'metadata': {'path': str(Path.cwd())}}, allow_errors=False,
                            on_cell_start=progress, startup_timeout=30)
    print('Starting Jupyter kernel with', sys.executable, flush=True)
    try:
        client.execute(cleanup_kc=True)
    finally:
        nbformat.write(notebook, path)
    nbformat.validate(notebook)
    errors = [o for c in notebook.cells if c.cell_type == 'code' for o in c.outputs if o.output_type == 'error']
    assert not errors
    print('Notebook executed and validated:', path, flush=True)
