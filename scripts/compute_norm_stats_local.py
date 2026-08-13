#!/usr/bin/env python3
"""本地版 norm_stats 计算: 把 config 的远端 repo_id 替换为本地路径后计算。

用法:
  python scripts/compute_norm_stats_local.py <config_name> <local_repo_dir>

示例:
  python scripts/compute_norm_stats_local.py \
      pi05_force_total_task_joint_only /mnt/hdd/sfy/datasets/total_task_flexiv

说明:
  - config 里的 repo_id 是远端路径 (/data/group1/junjie008/datasets/...),
    本脚本用 dataclasses.replace 覆盖为本地路径, 复用 compute_norm_stats
    的完整 transform 管道 (delta actions / ft_state / force_target)。
  - norm_stats.json 写到 <local_repo_dir>/norm_stats.json, 训练时
    _load_norm_stats 会从数据集目录加载 (asset_id=repo_id 绝对路径)。
"""
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def main(config_name: str, local_repo_dir: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    # 覆盖 repo_id → 本地路径
    new_data = dataclasses.replace(config.data, repo_id=local_repo_dir)
    config = dataclasses.replace(config, data=new_data)
    print(f"[norm-stats-local] config={config_name}, repo_id → {local_repo_dir}")

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    if data_config.rlds_data_dir is not None:
        raise NotImplementedError("RLDS path not supported here")
    print(f"[norm-stats-local] Creating torch dataset for repo_id={data_config.repo_id}")
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // config.batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // config.batch_size
        shuffle = False
    print(f"[norm-stats-local] len(dataset)={len(dataset)}, num_batches={num_batches}")
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    if getattr(data_config, "use_ft_history", False):
        keys.append("ft_state")
        stats["ft_state"] = normalize.RunningStats()

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            if key in batch:
                stats[key].update(np.asarray(batch[key]))
        if "force_target" in batch and "force_target" not in stats:
            stats["force_target"] = normalize.RunningStats()
        if "force_target" in stats:
            stats["force_target"].update(np.asarray(batch["force_target"]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # normalize.save appends "norm_stats.json" internally → pass the repo dir.
    output_path = pathlib.Path(local_repo_dir)
    print(f"[norm-stats-local] Writing stats to: {output_path / 'norm_stats.json'}")
    normalize.save(output_path, norm_stats)
    for key, st in norm_stats.items():
        print(f"  {key}: dim={len(st.mean)} q01={np.round(np.asarray(st.q01)[:3], 3)}")


if __name__ == "__main__":
    tyro.cli(main)
