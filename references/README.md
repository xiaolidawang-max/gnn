# Reference implementations

This directory collects upstream research implementations used for studying vascular/physics GNNs and neural operators.

- DIMON: https://github.com/MinglangYin/DIMON — MIT
- GraphCast / WeatherNext: https://github.com/google-deepmind/weathernext — Apache-2.0
- NOEM: https://github.com/lu-group/noem — Apache-2.0
- SIRE: https://github.com/MIAGroupUT/SIRE — linked as a git submodule
- HAGMN-UQ: https://github.com/MIILab-MTU/HAGMN-UQ — linked as a git submodule
- VesselDiffusion: https://github.com/gzq17/VesselDiffusion — linked as a git submodule

The root repository also contains the imported reference implementation of:
- Physics-informed graph neural networks for flow field estimation in carotid arteries:
  https://github.com/sukjulian/physics-informed-gnn — MIT

Clone with submodules:
```bash
git clone --recursive https://github.com/xiaolidawang-max/gnn.git
```

## Additional public implementations

- Stanford gROM — https://github.com/StanfordCBCL/gROM
  - Learning reduced-order models for cardiovascular simulations with graph neural networks.
- PIGNN-1D-Blood-Flow — https://github.com/ahmetsenemse/PIGNN-1D-Blood-Flow-Simulation
  - Physics-Informed Graph Neural Networks to Solve 1-D Equations of Blood Flow.
- MAgNET — https://github.com/saurabhdeshpande93/MAgNET
  - Graph U-Net architecture for mesh-based simulations.
- Adaptive-Multiscale-GNN — https://github.com/rperera12/Adaptive-mesh-based-Multiscale-Graph-Neural-Network
  - Adaptive/multiscale mesh GNN work by Perera and Agrawal.
- HydroGraphNet — https://github.com/sandeepangh782/HydrographNet-PINN
  - Interpretable physics-informed GNN for fluid/flood dynamics.
- operator-cow — https://github.com/Wojtyjot/operator-cow
  - Reconstructing cerebral hemodynamics from sparse data using Neural Operator Transformers.
- coronary-mesh-convolution — https://github.com/sukjulian/coronary-mesh-convolution
  - SE(3)-equivariant coronary artery mesh learning.
- LaB-GATr — https://github.com/sukjulian/lab-gatr
  - Geometric Algebra Transformer for large biomedical meshes.
- graph-physics — https://github.com/DonsetPG/graph-physics
  - MeshGraphNet / Transformer / multigraph framework with aneurysm CFD examples.

These are linked as git submodules to preserve each upstream repository intact.
Run:
```bash
git submodule update --init --recursive
```
after cloning this repository.
