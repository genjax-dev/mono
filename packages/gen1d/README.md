# Gen1D

## Development Setup

Before running the following steps, run `./scripts/install-environment.sh` in the root directory to install `pixi`.

### CPU Development Environment

For CPU-based development:

1. Create and activate the CPU environment:

```bash
pixi install -e cpu
```
2. When using VS Code/Cursor:

- Open Command Palette (Ctrl/Cmd + Shift + P)
- Select `Python: Select Interpreter`
- Choose `packages/gen1d/.pixi/envs/cpu/bin/python`

3. For Jupyter notebooks:

- Open the notebook (e.g. `notebooks/gen1d.ipynb`)
- Select the `packages/gen1d/.pixi/envs/cpu/bin/python` kernel
- You can now run the notebook

### GPU Development Environment

For GPU-based development (Linux only):

3. Create and activate the GPU environment:
   ```bash
   pixi install -e gpu
   ```

4. When using VS Code/Cursor:

- Open Command Palette (Ctrl/Cmd + Shift + P)
- Select `Python: Select Interpreter`
- Choose `packages/gen1d/.pixi/envs/gpu/bin/python`

5. For Jupyter notebooks:

- Open the notebook (e.g. `notebooks/gen1d.ipynb`)
- Select the `packages/gen1d/.pixi/envs/gpu/bin/python` kernel
- You can now run the notebook
