import dataclasses

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


class _ToyForceDataset:
    def __init__(self):
        self._items = [
            {
                "episode_index": 0,
                "frame_index": 0,
                "observation.wrist_force": np.array([1, 10, 100], dtype=np.float32),
                "observation.wrist_torque": np.array([2, 20, 200], dtype=np.float32),
            },
            {
                "episode_index": 0,
                "frame_index": 1,
                "observation.wrist_force": np.array([3, 30, 300], dtype=np.float32),
                "observation.wrist_torque": np.array([4, 40, 400], dtype=np.float32),
            },
            {
                "episode_index": 0,
                "frame_index": 2,
                "observation.wrist_force": np.array([5, 50, 500], dtype=np.float32),
                "observation.wrist_torque": np.array([6, 60, 600], dtype=np.float32),
            },
            {
                "episode_index": 1,
                "frame_index": 0,
                "observation.wrist_force": np.array([7, 70, 700], dtype=np.float32),
                "observation.wrist_torque": np.array([8, 80, 800], dtype=np.float32),
            },
        ]

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)


def test_force_history_augmented_dataset_zero_pads_episode_start():
    dataset = _data_loader.ForceHistoryAugmentedDataset(_ToyForceDataset(), history_frames=3)

    item = dataset[0]

    np.testing.assert_allclose(
        item["observation.wrist_force_history"],
        np.array([[0, 0, 0], [0, 0, 0], [1, 10, 100]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        item["observation.wrist_torque_history"],
        np.array([[0, 0, 0], [0, 0, 0], [2, 20, 200]], dtype=np.float32),
    )


def test_force_history_augmented_dataset_reads_previous_frames_without_cross_episode_leak():
    dataset = _data_loader.ForceHistoryAugmentedDataset(_ToyForceDataset(), history_frames=3)

    item = dataset[2]
    np.testing.assert_allclose(
        item["observation.wrist_force_history"],
        np.array([[1, 10, 100], [3, 30, 300], [5, 50, 500]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        item["observation.wrist_torque_history"],
        np.array([[2, 20, 200], [4, 40, 400], [6, 60, 600]], dtype=np.float32),
    )

    new_episode_item = dataset[3]
    np.testing.assert_allclose(
        new_episode_item["observation.wrist_force_history"],
        np.array([[0, 0, 0], [0, 0, 0], [7, 70, 700]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        new_episode_item["observation.wrist_torque_history"],
        np.array([[0, 0, 0], [0, 0, 0], [8, 80, 800]], dtype=np.float32),
    )
