# Overview
TODO: @mathieu

# CPU
To run `notebooks/gen2d.ipynb` on CPU via Cursor:
1. `pixi install -e cpu`
2. Command palette: `Python: Select Interpreter` and choose `.pixi/envs/cpu/bin/python`
3. Open the notebook and select the `.pixi/envs/cpu/bin/python` kernel
4. Run the notebook

# GPU
To run `notebooks/gen2d.ipynb` on GPU via Cursor:
1. Spin up a GPU VM and connect: https://www.notion.so/Working-with-VMs-18ac15e3585b8009a7b5f49918e42231?pvs=4
2. `pixi install -e gpu`
3. Command palette: `Python: Select Interpreter` and choose `.pixi/envs/gpu/bin/python`
4. Open the notebook and select the `.pixi/envs/gpu/bin/python` kernel
5. Run the notebook
