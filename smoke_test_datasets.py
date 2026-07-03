"""Data-only smoke test for HiVG's native TransVGDataset loader.

Confirms, for each dataset, that the split_root .pth annotation index loads,
the image files it points at actually exist and open, and a batch collates
correctly through the real train-time transform pipeline (datasets.build_dataset
+ utils.misc.collate_fn) -- i.e. everything hivg_train.py needs before it ever
touches a model or GPU. Catches path/layout mistakes (wrong im_dir, missing
images, wrong split_by) in seconds instead of after a multi-hour SLURM job
fails on iteration 1.

No GPU/model required. Run from within HiVG/ (relative imports):
    cd HiVG && python smoke_test_datasets.py
    cd HiVG && python smoke_test_datasets.py --datasets unc unc+ gref_umd referit flickr
"""
import argparse
import os
import traceback

from torch.utils.data import DataLoader

import utils.misc as utils
from datasets import build_dataset

ALL_DATASETS = ["unc", "unc+", "gref_umd", "referit", "flickr"]


def make_args(dataset: str, data_root: str, split_root: str, max_query_len: int) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=dataset,
        data_root=data_root,
        split_root=split_root,
        imsize=224,
        aug_scale=False,
        aug_crop=False,
        aug_blur=False,
        aug_translate=False,
        max_query_len=max_query_len,
        prompt="",
        use_seg_mask=False,
    )


def check_dataset(dataset: str, split: str, data_root: str, split_root: str,
                   max_query_len: int, num_samples: int, batch_size: int) -> bool:
    print(f"\n=== {dataset} ({split}) ===")
    args = make_args(dataset, data_root, split_root, max_query_len)
    try:
        ds = build_dataset(split, args)
    except Exception:
        print(f"[FAIL] could not build dataset:\n{traceback.format_exc()}")
        return False

    print(f"  {len(ds)} examples in split_root/{dataset}/{dataset}_{split}.pth")
    if len(ds) == 0:
        print("[FAIL] empty split")
        return False

    try:
        for idx in range(min(num_samples, len(ds))):
            img, img_mask, word_id, word_mask, bbox, img_file, phrase, bbox_ori, obj_mask = ds[idx]
            print(f"  sample {idx}: img={tuple(img.shape)} bbox={bbox} file={img_file!r} phrase={phrase!r}")
    except Exception:
        print(f"[FAIL] __getitem__ raised (bad image path / annotation format?):\n{traceback.format_exc()}")
        return False

    try:
        loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=False,
                             collate_fn=utils.collate_fn, num_workers=0)
        batch = next(iter(loader))
        img_batch = batch[0].tensors
        print(f"  batch ok: img batch shape={tuple(img_batch.shape)}")
    except Exception:
        print(f"[FAIL] DataLoader/collate_fn raised:\n{traceback.format_exc()}")
        return False

    print(f"[PASS] {dataset}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    parser.add_argument("--split", default="val", help="val for unc/unc+/gref_umd/flickr; test also valid for referit")
    parser.add_argument("--data_root", default="$PLG_GROUPS_STORAGE/plggwie/plgmazurekagh/grounding_data")
    parser.add_argument("--split_root", default="$PLG_GROUPS_STORAGE/plggwie/plgmazurekagh/grounding_data/data")
    parser.add_argument("--max_query_len", default=77, type=int)
    parser.add_argument("--num_samples", default=3, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    args = parser.parse_args()

    data_root = os.path.expandvars(args.data_root)
    split_root = os.path.expandvars(args.split_root)
    print(f"data_root={data_root}\nsplit_root={split_root}")

    results = {}
    for dataset in args.datasets:
        results[dataset] = check_dataset(
            dataset, args.split, data_root, split_root,
            args.max_query_len, args.num_samples, args.batch_size,
        )

    print("\n=== summary ===")
    for dataset, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {dataset}")
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    import os
    main()
