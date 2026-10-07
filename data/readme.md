# Datasets

| Dataset | File | Source |
|---|---|---|
| Cora ML | `cora.npz` | [Bojchevski & Günnemann, 2018](https://github.com/abojchevski/graph2gauss) |
| Citeseer | `citeseer.npz` | [Bojchevski & Günnemann, 2018](https://github.com/abojchevski/graph2gauss) |

Both graphs are loaded by `train_eval_data/get_dataset.py`, which symmetrizes the
adjacency matrix, removes self-loops, and keeps the largest connected component.
Splits are five fixed, stratified 10% / 10% / 80% train / val / test partitions
(`get_splits`), following [Mujkanovic et al., 2022](https://arxiv.org/abs/2301.00012).
