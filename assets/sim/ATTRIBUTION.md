# Real-image assets for the camera simulation

Everything in this folder is cut out of, or rectified from, **photos of the
datasets our detector was trained on**. Nothing is drawn or generated here.

| Folder | What | Source dataset |
|---|---|---|
| `sprites/person/`, `sprites/forklift/` | workers and forklifts, cut out with SAM 2.1 from the labelled boxes | Ultralytics **Construction-PPE** |
| `sprites/unknown/` | traffic cones, tool cases, a pallet stack: objects with **no detector class**, cut out from hand-picked boxes (`sprites/unknown_objects.json`) | Ultralytics **Construction-PPE** |
| `sprites/package/` | parcels, cut out with the dataset's own segmentation polygons | Ultralytics **Package-seg** |
| `textures/` | rack faces, floor, walls with dock doors, ceiling: flat regions of warehouse photos warped to a front view | Ultralytics **Construction-PPE** |

## Licences and credit

* **Ultralytics Construction-PPE Dataset**, Ultralytics team, November 2026,
  licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  <https://docs.ultralytics.com/datasets/detect/construction-ppe>
  According to its README the dataset "combines human-collected and
  generated images".
  *Changes made here:* objects were cut out with a segmentation model and
  resized; planar regions were perspective-warped, cropped and
  brightness-flattened.
* **Package-seg** (Ultralytics / Roboflow Universe "factory_package").
  <https://docs.ultralytics.com/datasets/segment/package-seg>
  *Changes made here:* parcels were cut out along their polygon labels and
  resized.

These assets are **not** covered by the repository's MIT licence; the dataset
licences above apply to them.

## Splits (why the evaluation is honest)

`sprites/index.json` records, for every sprite, the split of the photo it
came from (`train` / `val` / `test` of the 3-class dataset built by
`scripts/build_warehouse_dataset.py`).

* Composite frames used to fine-tune the detector contain **train** sprites
  only.
* The demo and the in-loop evaluation use **val + test** sprites only: real
  people, forklifts and parcels the detector never saw during training.

## Rebuild

```bash
python -m scripts.build_warehouse_dataset --src data_downloads --download \
    --out datasets/processed/warehouse3
python scripts/build_sprite_bank.py --src data_downloads \
    --dataset datasets/processed/warehouse3 --out assets/sim/sprites
python scripts/build_textures.py --src data_downloads --out assets/sim/textures
```

`sprites/curation.json` lists the cut-outs that were rejected by eye (broken
masks). `textures/sources.json` lists the photo and region of every texture.
