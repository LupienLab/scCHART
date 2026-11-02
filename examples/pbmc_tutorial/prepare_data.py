"""Prepare the PBMC 5k scATAC input for the scCHART tutorial.

Source: the annotated 10x Genomics PBMC 5k scATAC object distributed with the
SnapATAC2 tutorials (``snapatac2.datasets.pbmc5k(type="annotated_h5ad")``):
500-bp genome tiles, 250,000 selected tiles (``var["selected"]``), cell-type
labels in ``obs["cell_type"]`` and a UMAP in ``obsm["X_umap"]``. The file is
downloaded from its public SnapATAC2 mirror and checked against the SHA-256 in
the SnapATAC2 dataset registry.

The tutorial keeps the nine populations with at least 140 cells and removes
MAIT (246 cells), NK (236) and pDC (34). The output holds 3,921 cells x
250,000 selected tiles (integer counts), the labels and the UMAP.

Usage::

    uv run python examples/pbmc_tutorial/prepare_data.py            # download (~840 MB)
    uv run python examples/pbmc_tutorial/prepare_data.py --source /path/to/atac_pbmc_5k_annotated.h5ad
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

URL = "https://osf.io/download/e9vc3/"
SHA256 = "592f1551c27d0cfe4d81e7febad624d6b7d3ebf977b0c3ea64e06b3f3d76f078"
EXCLUDED = ("MAIT", "NK", "pDC")
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "data" / "pbmc5k_scchart_input.h5ad"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {URL} -> {dest}")
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(URL) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh)
        tmp.rename(dest)
    return dest


def _strings(ds) -> np.ndarray:
    return ds.asstr()[:] if ds.dtype.kind in "OS" else ds[:]


def _obs_column(h, key) -> np.ndarray:
    col = h["obs"][key]
    if isinstance(col, h5py.Group):  # categorical
        return _strings(col["categories"]).astype(str)[col["codes"][:]]
    return _strings(col).astype(str)


def build(source: Path, out: Path) -> AnnData:
    """Read only the parts the tutorial needs (the source also stores fragments)."""
    with h5py.File(source, "r") as h:
        shape = tuple(int(x) for x in h["X"].attrs["shape"])
        X = sparse.csr_matrix((h["X/data"][:], h["X/indices"][:], h["X/indptr"][:]), shape=shape)
        cells = _strings(h["obs"][h["obs"].attrs["_index"]]).astype(str)
        tiles = _strings(h["var"][h["var"].attrs["_index"]]).astype(str)
        cell_type = _obs_column(h, "cell_type")
        umap = np.asarray(h["obsm"]["X_umap"][:], dtype=float)[:, :2]
        selected = h["var"]["selected"][:].astype(bool)
    keep = ~np.isin(cell_type, EXCLUDED)
    cols = np.flatnonzero(selected)
    X = X[keep][:, cols].tocsr()
    X.sort_indices()
    adata = AnnData(X=X, obs=pd.DataFrame({"cell_type": pd.Categorical(cell_type[keep])},
                                          index=pd.Index(cells[keep], name="barcode")),
                    var=pd.DataFrame(index=pd.Index(tiles[cols], name="tile")))
    adata.obsm["X_umap"] = umap[keep]
    adata.uns["source"] = {"file": source.name, "sha256": SHA256, "url": URL,
                           "excluded_cell_types": list(EXCLUDED), "features": "var['selected'] tiles"}
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out, compression="gzip")
    return adata


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, help="local copy of atac_pbmc_5k_annotated.h5ad (skips the download)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output h5ad (default {DEFAULT_OUT})")
    a = p.parse_args()
    source = a.source or download(HERE / "data" / "atac_pbmc_5k_annotated.h5ad")
    digest = sha256(source)
    if digest != SHA256:
        raise SystemExit(f"SHA-256 mismatch for {source}: {digest} (expected {SHA256})")
    adata = build(source, a.out)
    print(f"wrote {a.out}: {adata.n_obs:,} cells x {adata.n_vars:,} tiles, nnz = {adata.X.nnz:,}")
    print(adata.obs["cell_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
