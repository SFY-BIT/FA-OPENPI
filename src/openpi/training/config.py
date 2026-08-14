"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.pi0_force as pi0_force
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.piper_policy as piper_policy
import openpi.policies.force_piper_policy as force_piper_policy
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()
    # If > 0, augment each sample with zero-padded past force/torque history.
    force_history_frames: int = 0

    # Internal: the model config after EEF norm-stats injection (set by
    # LeRobotPiperDataConfig.create when use_eef_loss=True). train.py reads
    # this to build the model with the injected quantile stats.
    _eef_model_config: Any | None = None
    # FT history mode: read precomputed wrench_history column instead of runtime collection.
    use_ft_history: bool = False
    ft_history_steps: int = 60


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotPiperDataConfig(DataConfigFactory):
    """Config for Piper datasets stored in LeRobot format."""

    observation_image_key: str = "observation.images.one"
    observation_wrist_image_key: str = "observation.images.two"
    observation_right_wrist_image_key: str | None = None  # dual-arm: right wrist camera
    observation_state_key: str = "observation.state"
    action_key: str = "action"
    prompt_key: str = "prompt"
    use_delta_joint_actions: bool = True
    use_delta_gripper_actions: bool = False  # If True, grip (dim 6) also converted to delta
    use_force_data: bool = False  # If True, concat wrist_force + wrist_torque into state
    predict_force: bool = False  # If True, also predict next-frame force (13-dim output)
    # If True, force/torque is already stored inside observation.state (e.g. 13-dim
    # state = [joints+gripper, Fx,Fy,Fz,Tx,Ty,Tz]) and ForceInStatePiperInputs is used
    # instead of ForcePiperInputs. The dual-head model then predicts force via a
    # separate force_out_proj head and a separate force_target key.
    force_in_state: bool = False
    # Force history sequence encoding (FAWAM-style).
    # When True, ForceInStatePiperInputs reads precomputed wrench_history and
    # emits a separate ft_state key instead of embedding force into state.
    use_ft_history: bool = False
    ft_history_steps: int = 60         # T: number of history frames
    state_mask_indices: tuple[int, ...] = ()
    action_mask_indices: tuple[int, ...] = ()
    state_mask_value: float = 0.0
    action_mask_value: float = 0.0
    default_prompt: str | None = None
    # Number of action dimensions in the dataset.  Used to build the delta
    # action mask and to configure PiperOutputs.  Default 8 = 7 joints + 1
    # gripper (single-arm Piper/Panda).  Set to 14 for dual-arm ARX X5
    # (7 + 7 joints, no gripper).
    action_dim: int = 8

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_mapping = {
            "observation/image": self.observation_image_key,
            "observation/wrist_image": self.observation_wrist_image_key,
            "observation/state": self.observation_state_key,
            "actions": self.action_key,
            "prompt": self.prompt_key,
        }
        # Dual-arm: add right wrist camera if configured
        if self.observation_right_wrist_image_key is not None:
            repack_mapping["observation/right_wrist_image"] = self.observation_right_wrist_image_key
        # If using force data, also repack force/torque keys (dot → slash)
        if self.use_force_data and not self.force_in_state:
            # Force stored in separate wrist_force/wrist_torque fields.
            repack_mapping["observation/wrist_force"] = "observation.wrist_force"
            repack_mapping["observation/wrist_torque"] = "observation.wrist_torque"
            # Legacy force history keys (written by ForceHistoryAugmentedDataset).
            repack_mapping["observation/wrist_force_history"] = "observation.wrist_force_history"
            repack_mapping["observation/wrist_torque_history"] = "observation.wrist_torque_history"
        # If force_in_state, force lives inside observation.state already; no extra
        # repack for the state itself. The force history (past K state frames) is
        # written by ForceHistoryAugmentedDataset as observation.state_history and
        # must be repack'd to observation/state_history for ForceInStatePiperInputs.
        if self.use_force_data and self.force_in_state and not self.use_ft_history:
            repack_mapping["observation/state_history"] = "observation.state_history"
        # If using ft history, repack the precomputed wrench_history column.
        if self.use_force_data and self.use_ft_history:
            repack_mapping["observation/wrench_history"] = "observation.wrench_history"

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_mapping)]
        )

        if self.use_force_data and self.force_in_state:
            # Dual-head path: force embedded in observation.state.
            control_action_dim = getattr(model_config, 'control_action_dim', None) or getattr(model_config, 'force_start_idx', 7)
            data_transforms = _transforms.Group(
                inputs=[force_piper_policy.ForceInStatePiperInputs(
                    model_type=model_config.model_type,
                    predict_force=self.predict_force,
                    force_history_frames=getattr(model_config, 'force_history_frames', 1),
                    force_start_idx=getattr(model_config, 'force_start_idx', 7),
                    force_dim=getattr(model_config, 'force_dim', 6),
                    use_ft_history=getattr(model_config, 'use_ft_history', False),
                    ft_history_steps=getattr(model_config, 'ft_history_steps', 60),
                )],
                outputs=[force_piper_policy.ForceInStatePiperOutputs(
                    predict_force=self.predict_force,
                    control_action_dim=control_action_dim,
                    force_start_idx=getattr(model_config, 'force_start_idx', 7),
                    force_dim=getattr(model_config, 'force_dim', 6),
                    force_history_frames=getattr(model_config, 'force_history_frames', 1),
                )],
            )
        elif self.use_force_data:
            data_transforms = _transforms.Group(
                inputs=[force_piper_policy.ForcePiperInputs(
                    model_type=model_config.model_type,
                    predict_force=self.predict_force,
                    force_history_frames=getattr(model_config, 'force_history_frames', 1),
                )],
                outputs=[force_piper_policy.ForcePiperOutputs(
                    predict_force=self.predict_force,
                )],
            )
        else:
            data_transforms = _transforms.Group(
                inputs=[piper_policy.PiperInputs(model_type=model_config.model_type)],
                outputs=[piper_policy.PiperOutputs(action_dim=self.action_dim)],
            )
        if self.use_delta_joint_actions:
            if self.predict_force and not self.force_in_state:
                # Legacy single-head: 14 dims = joints(7) delta, grip(1) delta, force(6) delta
                if self.use_delta_gripper_actions:
                    delta_action_mask = _transforms.make_bool_mask(8, 6)
                else:
                    delta_action_mask = _transforms.make_bool_mask(7, -1, 6)
            elif self.predict_force and self.force_in_state:
                # Dual-head: action target is control-only (joints+gripper, no force).
                if self.use_delta_gripper_actions:
                    delta_action_mask = _transforms.make_bool_mask(self.action_dim)
                else:
                    delta_action_mask = _transforms.make_bool_mask(self.action_dim - 1, -1)
            elif self.use_delta_gripper_actions:
                # all action_dim dims delta (e.g. 8 = 7 joint + 1 grip for single-arm)
                delta_action_mask = _transforms.make_bool_mask(self.action_dim)
            else:
                # Single-arm default: 7 joints delta + 1 gripper absolute.
                # Dual-arm (action_dim=14): 6 joints delta + 1 grip absolute per arm,
                # matching the Aloha reference config make_bool_mask(6, -1, 6, -1).
                if self.action_dim == 8:
                    delta_action_mask = _transforms.make_bool_mask(7, -1)
                else:
                    delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        if self.state_mask_indices or self.action_mask_indices:
            data_transforms = data_transforms.push(
                inputs=[
                    _transforms.MaskStateActionDims(
                        state_indices=self.state_mask_indices,
                        action_indices=self.action_mask_indices,
                        state_value=self.state_mask_value,
                        action_value=self.action_mask_value,
                    )
                ]
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        # EEF pose loss: inject quantile norm stats (q01/q99) into the model config
        # so compute_loss can unnormalize joints back to physical space before FK.
        # The dataset norm_stats are loaded by create_base_config; we pick the
        # `actions` (joint delta) and `state` (absolute joints) quantiles.
        norm_stats = self._load_norm_stats(
            epath.Path(self.assets.assets_dir or assets_dirs), self.assets.asset_id or self.repo_id
        )
        if (
            getattr(model_config, "use_eef_loss", False)
            and norm_stats is not None
            and "actions" in norm_stats
            and "state" in norm_stats
        ):
            a_stats, s_stats = norm_stats["actions"], norm_stats["state"]
            # actions: first 6 dims are joint deltas (7th is gripper, unused by FK).
            # state: first 6 dims are absolute joints (7th is gripper).
            act_q01 = np.asarray(a_stats.q01)[:6].astype(np.float32)
            act_q99 = np.asarray(a_stats.q99)[:6].astype(np.float32)
            st_q01 = np.asarray(s_stats.q01)[:6].astype(np.float32)
            st_q99 = np.asarray(s_stats.q99)[:6].astype(np.float32)
            model_config = dataclasses.replace(
                model_config,
                eef_action_q01=act_q01,
                eef_action_q99=act_q99,
                eef_state_q01=st_q01,
                eef_state_q99=st_q99,
            )
            logging.info("EEF loss: injected action/state quantile norm stats (q01/q99)")

        action_sequence_keys = [self.action_key]
        if self.predict_force:
            if self.force_in_state:
                action_sequence_keys.append(self.observation_state_key)
            else:
                action_sequence_keys.extend(["observation.wrist_force", "observation.wrist_torque"])

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=tuple(action_sequence_keys),
            force_history_frames=(
                0 if getattr(model_config, "use_ft_history", False)
                else (getattr(model_config, "force_history_frames", 0) if self.use_force_data else 0)
            ),
            use_ft_history=getattr(model_config, "use_ft_history", False),
            ft_history_steps=getattr(model_config, "ft_history_steps", 60),
            _eef_model_config=model_config if getattr(model_config, "use_eef_loss", False) else None,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    new_module_lr_multiplier: float = 1.0  # LR multiplier for limoe/force params (>1 = higher LR)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_piper_finetune",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30, discrete_state_input=False),
        data=LeRobotPiperDataConfig(
            repo_id="/data/group1/junjie008/siyuan-V2",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=True,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=100_000,
    ),
    #
    # Two-stage: pretrain pure vision → LoRA + LIMoE + force
    #
    TrainConfig(
        name="pi05_usb_pretrain",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30, discrete_state_input=False),
        data=LeRobotPiperDataConfig(
            repo_id="/data/group1/junjie008/datasets/usb_insert_openpi_v2_F",
            assets=AssetsConfig(asset_id="usb_pretrain_7dim"),
            observation_image_key="observation.images.agent",
            observation_wrist_image_key="observation.images.wrist",
            default_prompt="Insert the USB into the port",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_force_lora_stage2",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,  # FIX: was missing -> force loss weight stayed 0
            force_start_idx=7,
            force_history_frames=5, force_loss_weight=0.1,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=10.0,
        data=LeRobotPiperDataConfig(
            repo_id="/data/group1/junjie008/datasets/usb_insert_openpi_v2_F",
            assets=AssetsConfig(asset_id="usb_force_13dim"),
            observation_image_key="observation.images.agent",
            observation_wrist_image_key="observation.images.wrist",
            default_prompt="Insert the USB into the port",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=True,
            use_force_data=True,
            predict_force=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.Pi0ForceWeightLoader("REPLACE_WITH_STAGE1_CKPT_PATH"),
        num_train_steps=10_000,
    ),
    # Local LoRA + LIMoE + Force (for local rollout testing)
    TrainConfig(
        name="pi05_force_lora_local",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            force_start_idx=8, force_history_frames=3,
            force_loss_weight=0.3,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=10.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/robosuite/datasets/panda-force",
            observation_image_key="observation.images.agent",
            observation_wrist_image_key="observation.images.wrist",
            default_prompt="Insert the USB into the port",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=True,
            use_force_data=True,
            predict_force=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.Pi0ForceWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=10_000,
    ),
    # Local LoRA + LIMoE + Force, force_loss_weight=0.1 (lower force loss weight variant)
    TrainConfig(
        name="pi05_force_lora_local_w01",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            force_start_idx=8, force_history_frames=3,
            force_loss_weight=0.1,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=10.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/robosuite/datasets/panda-force",
            observation_image_key="observation.images.agent",
            observation_wrist_image_key="observation.images.wrist",
            default_prompt="Insert the USB into the port",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=True,
            use_force_data=True,
            predict_force=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.Pi0ForceWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=10_000,
    ),
    # Panda full fine-tune: LIMoE + force input, no force output head, all params trainable
    TrainConfig(
        name="pi05_panda_full",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            use_force=True, predict_force=False,
            force_start_idx=8,  # Panda: 7 joints + 1 gripper
            force_history_frames=3,
            num_experts=4, num_top_k=1,
        ),
        data=LeRobotPiperDataConfig(
            repo_id="/data/group1/junjie008/datasets/panda-force-full",
            observation_image_key="observation.images.agent",
            observation_wrist_image_key="observation.images.wrist",
            default_prompt="Insert the USB into the port",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=True,
            use_force_data=True,
            predict_force=False,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.Pi0ForceWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    # Panda baseline: pure Pi0.5, no force input, no LIMoE, no force output
    # For ablation comparison against pi05_panda_full
    TrainConfig(
        name="pi05_panda_noforce",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=30, discrete_state_input=False,
        ),
        data=LeRobotPiperDataConfig(
            repo_id="/data/group1/junjie008/datasets/panda-noforce",
            observation_image_key="observation.images.agent",
            observation_wrist_image_key="observation.images.wrist",
            default_prompt="Insert the USB into the port",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=True,
            use_force_data=False,
            predict_force=False,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    # Panda no-force LOCAL: same as pi05_panda_noforce but with local dataset path
    # so that norm_stats can be loaded from local filesystem for inference
    TrainConfig(
        name="pi05_panda_noforce_local",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=30, discrete_state_input=False,
        ),
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/robosuite/datasets/panda-noforce",
            observation_image_key="observation.images.agent",
            observation_wrist_image_key="observation.images.wrist",
            default_prompt="Insert the USB into the port",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=True,
            use_force_data=False,
            predict_force=False,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    # RoboDojo ARX X5 dual-arm no-force LOCAL: 14-dim state/action (6+1 joints+grip
    # per arm), 3 cameras (cam_high + cam_left_wrist + cam_right_wrist), 3 assembly
    # tasks. Pure pose policy for the first-stage filtered self-imitation pipeline.
    # Matches pi05_base_aloha_full_sim_arx-x5_seed_0 reference config: pi05 mode with
    # discrete_state_input=True (auto), action_horizon=50, delta mask keeps gripper
    # (dim 6, 13) absolute.
    TrainConfig(
        name="pi05_robodojo_x5_noforce_local",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=50,
        ),
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/lerobot_datasets/robodojo_arx_x5_fine_assembly_v21_v21/unified_robot/robodojo_arx_x5_fine_assembly_v21",
            observation_image_key="observation.images.cam_high",
            observation_wrist_image_key="observation.images.cam_left_wrist",
            observation_right_wrist_image_key="observation.images.cam_right_wrist",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,  # gripper stays absolute via delta mask
            use_force_data=False,
            predict_force=False,
            action_dim=14,  # dual-arm: 7 + 7 (6 joint + 1 grip per arm)
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    # RoboDojo ARX X5 dual-arm no-force REMOTE: same as local but with remote
    # dataset path for SLURM A100 full fine-tuning.
    # Dataset: /data/group1/junjie008/datasets/robodojo_arx_x5_fine_assembly_v21
    TrainConfig(
        name="pi05_robodojo_x5_noforce",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=50,
        ),
        data=LeRobotPiperDataConfig(
            repo_id="/data/group1/junjie008/datasets/robodojo_arx_x5_fine_assembly_v21",
            observation_image_key="observation.images.cam_high",
            observation_wrist_image_key="observation.images.cam_left_wrist",
            observation_right_wrist_image_key="observation.images.cam_right_wrist",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,  # gripper stays absolute via delta mask
            use_force_data=False,
            predict_force=False,
            action_dim=14,  # dual-arm: 7 + 7 (6 joint + 1 grip per arm)
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_usb_insert",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30, discrete_state_input=False),
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/robosuite/datasets/usb_insert_openpi_v2",
            observation_image_key="observation.images.agent",
            observation_wrist_image_key="observation.images.wrist",
            default_prompt="Insert the USB into the port",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    #
    # ForceVLA / LIMoE (Sparse MoE) configs.
    #
    TrainConfig(
        name="pi05_force_usb_insert",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            use_force=True,  # Force/torque from wrist sensor → LIMoE
            predict_force=True,  # FIX: was missing -> force loss weight stayed 0
            force_start_idx=7,  # state = [joints(7), f0(3), t0(3), ..., f{K-1}(3), t{K-1}(3)]
            force_history_frames=5,  # Past 5 frames of force as input
            force_loss_weight=0.3,  # Force loss weighted 3x less than joints (was 0.1)
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,  # 5x LR for LIMoE + force (from scratch)
        data=LeRobotPiperDataConfig(
            repo_id="/data/group1/junjie008/datasets/usb_insert_openpi_v2_F",
            observation_image_key="observation.images.agent",
            observation_wrist_image_key="observation.images.wrist",
            default_prompt="Insert the USB into the port",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=True,
            use_force_data=True,  # Concat wrist_force + wrist_torque → state
            predict_force=True,   # Also predict next-frame force (13-dim output)
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.Pi0ForceWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_force_usb_insert_lora",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,  # FIX: was missing -> force loss weight stayed 0
            force_start_idx=7,
            force_history_frames=5,
            force_loss_weight=0.1,
            num_experts=4, num_top_k=1,
        ),
        data=LeRobotPiperDataConfig(
            repo_id="/data/group1/junjie008/datasets/usb_insert_openpi_v2_F",
            observation_image_key="observation.images.agent",
            observation_wrist_image_key="observation.images.wrist",
            default_prompt="Insert the USB into the port",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=True,
            use_force_data=True,
            predict_force=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=1e-5,
            decay_steps=1_000_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.Pi0ForceWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    # Dual-head Pi0Force for datasets where force/torque is stored INSIDE
    # observation.state (e.g. stamp_seal_flexiv on Piper/Flexiv):
    #   observation.state = [q1..q6, gripper, Fx,Fy,Fz,Tx,Ty,Tz]  (13-dim)
    #   action             = [target_q1..target_q6, target_gripper] (7-dim)
    # Uses ForceInStatePiperInputs + a separate force_out_proj head + a separate
    # force_target key (dual-head multi-task architecture).
    # Three-stage gradient routing:
    #   * VLM / vision        <- action_loss only
    #   * action expert+LIMoE <- 0.9*action_loss + 0.1*force_loss
    #   * force_out_proj      <- force_loss only
    #   * action_out_proj     <- action_loss only
    TrainConfig(
        name="pi05_force_stamp_seal",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            use_force=True, predict_force=True,
            control_action_dim=7,        # Piper control action: 6 joints + 1 gripper
            force_start_idx=7,           # force/torque starts at state dim 7
            force_dim=6,                 # Fx,Fy,Fz,Tx,Ty,Tz
            force_history_frames=2,      # 2 frames of force history as input
            # Three-stage gradient routing (scheme B+):
            #   VLM/vision <- 1.0*action, LIMoE+expert <- 1.0*action+0.1*force,
            #   action_out_proj <- 1.0*action, force_out_proj <- 1.0*force.
            grad_route_mode="three_stage",
            action_loss_weight=0.9,      # (unused in scheme B+, kept for compat)
            force_loss_weight=0.1,       # force weight for LIMoE+expert group
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="stamp seal",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,  # gripper absolute, 6 joints delta
            use_force_data=True,
            predict_force=True,
            force_in_state=True,        # force lives inside observation.state
            action_dim=7,               # control action is 7-dim (no force in action target)
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.Pi0ForceWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    # REMOTE version of pi05_force_stamp_seal for SLURM A100 training.
    # Identical to local except dataset path points to the remote server.
    # Dataset: /data/group1/junjie008/datasets/stamp_seal_flexiv
    TrainConfig(
        name="pi05_force_stamp_seal_remote",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            use_force=True, predict_force=True,
            control_action_dim=7,        # Piper control action: 6 joints + 1 gripper
            force_start_idx=7,           # force/torque starts at state dim 7
            force_dim=6,                 # Fx,Fy,Fz,Tx,Ty,Tz
            force_history_frames=2,      # 2 frames of force history as input
            # Three-stage gradient routing (scheme B+):
            #   VLM/vision <- 1.0*action, LIMoE+expert <- 1.0*action+0.1*force,
            #   action_out_proj <- 1.0*action, force_out_proj <- 1.0*force.
            grad_route_mode="three_stage",
            action_loss_weight=0.9,      # (unused in scheme B+, kept for compat)
            force_loss_weight=0.1,       # force weight for LIMoE+expert group
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/data/group1/junjie008/datasets/stamp_seal_v2_flexiv",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="stamp seal",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,  # gripper absolute, 6 joints delta
            use_force_data=True,
            predict_force=True,
            force_in_state=True,        # force lives inside observation.state
            action_dim=7,               # control action is 7-dim (no force in action target)
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.Pi0ForceWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    # ── FT History 模式：时序编码器 + 全局力向量 ──
    # 数据集需提前用 precompute_force_history.py 预处理为 _ft60 版本
    TrainConfig(
        name="pi05_force_stamp_seal_ft60",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            # ── FT History 新字段 ──
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.1,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="stamp seal",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.Pi0ForceWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    # ── FT History 全参 K=16（分段编码）：本地 ──
    # 对齐主干成熟配方：全解冻（无 freeze_filter）+ new_module_lr_multiplier=5.0
    #   (RouterWeights 1× 基 LR；limoe 专家/force/state_proj 5×)
    # 冷启动 pi05_base；ft_encoder(input_dim=ceil(60/16)*6=24)/ft_proj 随机初始化。
    # 目的：判决性实验——验证「force token 形态是路由限制」，
    #       与主干 K=2 legacy (已有 4w) 直接对比。
    TrainConfig(
        name="pi05_force_stamp_seal_ft60_k16",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=16,          # ← K=16 分段编码（每段 4 帧）
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.1,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="stamp seal",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.Pi0ForceWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
    ),
    # ── FT History 全参 K=16（分段编码）：远端 ──
    # 与本地全参 K=16 唯一区别：repo_id 指向远端数据集路径。
    TrainConfig(
        name="pi05_force_stamp_seal_ft60_k16_remote",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=16,          # ← K=16 分段编码（每段 4 帧）
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.1,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/data/group1/junjie008/datasets/stamp_seal_v2_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="stamp seal",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.Pi0ForceWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
    ),
    # ── FT History 全参 K=16：本地路径版（与 _remote 唯一区别：repo_id 本地）──
    # 用途: 本地启动 server / 离线回放, 避免远端 SLURM 路径不可达。
    # 与 _remote 完全同构（模型/数据/超参一致），仅 repo_id 改为本地数据集路径。
    TrainConfig(
        name="pi05_force_stamp_seal_ft60_k16_local",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=16,          # ← K=16 分段编码（每段 4 帧）
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.1,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="stamp seal",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.Pi0ForceWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
    ),
    # ── FT History LoRA 版本：快速验证数据流 ──
    TrainConfig(
        name="pi05_force_stamp_seal_ft60_lora",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.1,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=10.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="stamp seal",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.Pi0ForceWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=5_000,
    ),
    # ── FT History + ForceVLA-style LoRA：从 openpi-force checkpoint 热启动 ──
    # VLM: LoRA（主干冻结，仅 LoRA 可训练）
    # 视觉(SigLIP): 全参训练（不受 .*llm.* 冻结影响）
    # LIMoE + force_out_proj + state_proj: 从 openpi-force ckpt 加载后继续训练
    # ft_encoder + ft_proj: 随机初始化，10x LR
    TrainConfig(
        name="pi05_force_stamp_seal_ft60_forcevla_lora",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.1,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=10.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="stamp seal",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # 从 openpi-force 已训练的 checkpoint 热启动（含预训练 LIMoE + force_out_proj）
        weight_loader=weight_loaders.Pi0ForceWeightLoader(
            "/mnt/hdd/sfy/openpi-force_checkpoints_bak/12000/params"
        ),
        log_interval=10,
        save_interval=2000,
        keep_period=10000,
        num_train_steps=30_000,
    ),
    # ── FT History + ForceVLA-style LoRA, K=2 (多 force token) ──
    # 与 forcevla_lora (K=1) 唯一区别：ft_num_tokens=2 → ft_proj 输出 2 个 force token。
    # 目的：验证「force token 数量 K」对 MoE 路由的影响 ——
    #   主干(K=2 per-frame) 能形成 force 分散/依附于主流专家；
    #   ft60 K=1 时 force 100% 依附单一专家(E1)，router 置信度仅 0.25。
    # 此配置对齐主干的 K=2，仅训练 LoRA/新模块，从 openpi-force 12000 热启动。
    TrainConfig(
        name="pi05_force_stamp_seal_ft60_forcevla_lora_k2",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=2,           # ← K=2: 2 个 force token（对齐主干）
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.1,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=10.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="stamp seal",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # 从 openpi-force 已训练的 checkpoint 热启动（含预训练 LIMoE + force_out_proj）
        weight_loader=weight_loaders.Pi0ForceWeightLoader(
            "/mnt/hdd/sfy/openpi-force_checkpoints_bak/12000/params"
        ),
        log_interval=10,
        save_interval=2000,
        keep_period=10000,
        num_train_steps=30_000,
    ),
    # ── FT History + ForceVLA-style LoRA, K=16 (分段编码) ──
    # 与 forcevla_lora (K=1) 唯一区别：ft_num_tokens=16 → 60 帧历史切成 16 段
    # （每段 4 帧），每段用共享 FTEncoder 编码成独立 token，共享 ft_proj 投影。
    # 目的：验证「force token 数量/形态是路由限制」——K=1 时 force 100% 依附
    # 单一专家且 gate prob 仅 0.25；K=16 后 force 有 16 个语义不同的 token，
    # 期望出现跨专家分散或独立专家趋势（对齐主干 K=2 的分散行为）。
    TrainConfig(
        name="pi05_force_stamp_seal_ft60_forcevla_lora_k16",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=16,          # ← K=16: 分段编码（每段 4 帧）
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.1,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=10.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/datasets/stamp_seal_v2_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="stamp seal",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # 从 openpi-force 已训练的 checkpoint 热启动（含预训练 LIMoE + force_out_proj）；
        # ft_encoder (input_dim=24) / ft_proj 与旧 shape 不匹配 → weight_loader 随机初始化。
        weight_loader=weight_loaders.Pi0ForceWeightLoader(
            "/mnt/hdd/sfy/openpi-force_checkpoints_bak/12000/params"
        ),
        log_interval=10,
        save_interval=2000,
        keep_period=10000,
        num_train_steps=30_000,
    ),
    # ── Erase Board, FT History + ForceVLA-style LoRA, K=16, 弱化 force 权重 ──
    # 背景：stamp_seal 系列后期 total_loss 几乎全部由 force loss 主导
    # （action_loss≈0.005 vs force_loss≈0.07），怀疑训练波动/退化源于此。
    # 本 config 将 force 权重整体调小一个量级：
    #   * force_head_loss_weight = 0.01  (force_out_proj head 路径，原硬编码 1.0)
    #   * force_loss_weight      = 0.01  (LIMoE+expert 路径，原 0.1)
    #   * force_frame_spike_weight = 2.0 (帧加权，原 20.0，接触帧强调削弱一个量级)
    # 冷启动：从 pi05_base（本地缓存）初始化，LIMoE/force/ft_encoder 随机初始化。
    # 数据集需先用 precompute_force_history.py 生成 erase_board_flexiv_ft60。
    TrainConfig(
        name="pi05_force_erase_board_ft60_forcevla_lora_k16",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=16,          # K=16: 分段编码（每段 4 帧）
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.01,          # LIMoE+expert 力路径：0.1 → 0.01
            force_head_loss_weight=0.01,     # force_out_proj head 路径：1.0 → 0.01
            force_frame_spike_weight=2.0,    # 帧加权：20.0 → 2.0（削弱一个量级）
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/datasets/erase_board_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="erase the board",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # 从 pi05_base 冷启动（本地缓存命中 gs://openpi-assets/checkpoints/pi05_base/params）；
        # limoe/force/ft_encoder/ft_proj 不在 base 中 → weight_loader 随机初始化。
        weight_loader=weight_loaders.Pi0ForceWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        log_interval=10,
        save_interval=2000,
        keep_period=10000,
        num_train_steps=30_000,
    ),
    # ── Erase Board, FT History + EEF pose loss（0.7 关节 + 0.3 EEF） ──
    # 目的：真机擦除段末端 yaw 漂移（action 自发，q4 持续转动 40°）——EEF loss
    # 用可微 FK 把关节 delta 的"末端效果"放大，约束末端位姿（尤其姿态）。
    #   * 数据集不动（无新增字段）
    #   * gt = FK(q_cur + target_delta)，pred = FK(q_cur + pred_delta)
    #   * 旋转矩阵差 loss（无回绕、无欧拉角奇异）
    #   * total_action = 0.7 * 关节 loss + 0.3 * EEF loss
    # 反归一化（quantile q01/q99）由 LeRobotPiperDataConfig 自动注入 model config。
    # 需先对新数据集跑 compute_norm_stats（用户手动执行）。
    TrainConfig(
        name="pi05_force_erase_board_eef",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=16,          # K=16: 分段编码（每段 4 帧）
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.01,          # LIMoE+expert 力路径
            force_head_loss_weight=0.01,     # force_out_proj head 路径
            force_frame_spike_weight=2.0,    # 帧加权
            # EEF pose loss: 0.6 关节 + 0.4 EEF（工具末端 0.211m 含传感器）
            # EEF 内部分量: 0.3*位置 + 2.0*姿态（折中: 比旧 1.0 强, 比 3.0 温和）
            # 权重依据: 0.5/3.0 实验 joint 退化(action_loss 3倍劣化)但 rot 没更快收敛;
            # 折中 0.6/2.0 保住 joint, 同时 rot 监督比旧权重强 2 倍。
            use_eef_loss=True,
            action_joint_weight=0.6,
            tool_extension=0.211,
            eef_pos_weight=0.3,
            eef_angle_weight=2.0,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/datasets/erase_board_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="erase the board",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # 从 3w (29999) checkpoint 热启动（含 lora/limoe/force/ft_encoder 全量权重）。
        # 新优化器状态（不从旧 train_state 恢复）→ 干净启动 + EEF loss 新目标。
        weight_loader=weight_loaders.Pi0ForceWeightLoader(
            "/mnt/hdd/sfy/FA-openpi/checkpoints/pi05_force_erase_board_ft60_forcevla_lora_k16/erase_board_ft60_k16_w001/29999/params"
        ),
        log_interval=10,
        save_interval=2000,
        keep_period=10000,
        num_train_steps=50_000,
    ),
    # ── EEF-only 消融版（本地）：只用 EEF loss，joint loss 只算不参与总 loss ──
    #   用途: 验证纯 EEF 监督能否单独驱动 q4/q5/q6 学到正确位姿 (消融实验)
    #   与 pi05_force_erase_board_eef 唯一区别: eef_only_mode=True
    #   (compute_loss 里 action_loss_weighted = eef_loss, joint loss 仅日志播报)
    TrainConfig(
        name="pi05_force_erase_board_eef_only",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=16,
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.01,
            force_head_loss_weight=0.01,
            force_frame_spike_weight=2.0,
            # EEF-only: joint loss 只算不参与总 loss (eef_only_mode=True)
            use_eef_loss=True,
            eef_only_mode=True,
            action_joint_weight=0.6,   # 保留但 eef_only_mode 下不用
            tool_extension=0.211,
            eef_pos_weight=0.3,
            eef_angle_weight=2.0,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/datasets/erase_board_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="erase the board",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.Pi0ForceWeightLoader(
            "/mnt/hdd/sfy/FA-openpi/checkpoints/pi05_force_erase_board_ft60_forcevla_lora_k16/erase_board_ft60_k16_w001/29999/params"
        ),
        log_interval=10,
        save_interval=500,
        keep_period=10000,
        num_train_steps=10_000,
    ),
    # ── EEF 全参远端版（与本地 pi05_force_erase_board_eef 唯一区别：
    #    repo_id 与 weight_loader 指向远端 /data/group1/junjie008 路径）──
    TrainConfig(
        name="pi05_force_erase_board_eef_remote",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=16,
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.01,
            force_head_loss_weight=0.01,
            force_frame_spike_weight=2.0,
            use_eef_loss=True,
            action_joint_weight=0.6,
            tool_extension=0.211,
            eef_angle_weight=2.0,
            eef_pos_weight=0.3,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/data/group1/junjie008/datasets/erase_board_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="erase the board",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        # 从本地 3w (29999) checkpoint 热启动（远端路径，需先 put 上传）。
        weight_loader=weight_loaders.Pi0ForceWeightLoader(
            "/data/group1/junjie008/FA-openpi/checkpoints/pi05_force_erase_board_ft60_forcevla_lora_k16/erase_board_ft60_k16_w001/29999/params"
        ),
        log_interval=10,
        save_interval=2000,
        keep_period=10000,
        num_train_steps=50_000,
    ),
    #
    # Existing Piper configs.
    #
    TrainConfig(
        name="pi05_piper_finetune_joint3mask",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/lerobot.act/data_piper_multi_record_cam_act_10_2",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=True,
            state_mask_indices=(3,),
            action_mask_indices=(3,),
            state_mask_value=0.0,
            action_mask_value=0.0,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-5,
            decay_steps=1_000_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_piper_low_mem_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/lerobot.act/data_piper_multi_record_cam_act_10_2",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=True,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=1e-5,
            decay_steps=1_000_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_piper_low_mem_finetune_warmup1k",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/lerobot.act/data_piper_multi_record_cam_act_10_2",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=True,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-5,
            decay_steps=1_000_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_piper_low_mem_finetune_warmup1k_joint3mask",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/lerobot.act/data_piper_multi_record_cam_act_10_2",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=True,
            state_mask_indices=(3,),
            action_mask_indices=(3,),
            state_mask_value=0.0,
            action_mask_value=0.0,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-5,
            decay_steps=1_000_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_piper_low_mem_finetune_warmup1k_joint3mask_freeze_vision",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/lerobot.act/data_piper_multi_record_cam_act_10_2",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=True,
            state_mask_indices=(3,),
            action_mask_indices=(3,),
            state_mask_value=0.0,
            action_mask_value=0.0,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-5,
            decay_steps=1_000_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=nnx.Any(
            pi0_config.Pi0Config(
                pi05=True,
                action_horizon=10,
                discrete_state_input=False,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            nnx_utils.PathRegex(".*PaliGemma/img.*"),
        ),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
    # ── total_task 数据集 (EEF 坐标 state): 3 个监督模式消融 (远端 LoRA) ──
    # 数据集: convert_dataset_to_eef.py 转换后 (state 前 6 维 = EEF xyz+rpy, gripper 不变)
    # 通用字段: repo_id /data/group1/junjie008/datasets/total_task_flexiv_eef (远端)
    #           weight_loader 指向 3w (29999) checkpoint
    # 与 *_local 差异: 仅 repo_id (远端路径) + warmup 步数 (2w2) + bs=32 + 3w 步
    # 3 变体:
    #   1) eef_only  : 只用 EEF loss (joint loss 只播报)
    #   2) joint_only: 只用 joint loss (use_eef_loss=False)
    #   3) joint_eef  : joint + EEF, 老权重 (0.7/1.0), EEF warmup 2w2 步后完成
    TrainConfig(
        name="pi05_force_total_task_eef_only_remote",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=16,
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.01,
            force_head_loss_weight=0.01,
            force_frame_spike_weight=2.0,
            # EEF-only: 数据集已转换为 EEF 坐标 (state/action 前 6 维 = EEF xyz+rpy),
            # loss 直接在 EEF 空间算, 无需 FK 分支 (use_eef_loss=False)。
            use_eef_loss=False,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/data/group1/junjie008/datasets/total_task_flexiv_eef",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="perform the task",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.Pi0ForceWeightLoader(
            "/data/group1/junjie008/FA-openpi/checkpoints/pi05_force_erase_board_ft60_forcevla_lora_k16/erase_board_ft60_k16_w001/29999/params"
        ),
        log_interval=10,
        save_interval=500,
        keep_period=2000,
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_force_total_task_joint_only_remote",
        # NOTE: 数据集是 total_task_flexiv_ft60 (带 wrench_history 列);
        #       repo_id 用 ft60 名, 因为 use_ft_history=True 需要该列.
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=16,
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.01,
            force_head_loss_weight=0.01,
            force_frame_spike_weight=2.0,
            use_eef_loss=False,          # 纯 joint loss (原数据集)
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/data/group1/junjie008/datasets/total_task_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="perform the task",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.Pi0ForceWeightLoader(
            "/data/group1/junjie008/FA-openpi/checkpoints/pi05_force_erase_board_ft60_forcevla_lora_k16/erase_board_ft60_k16_w001/29999/params"
        ),
        log_interval=10,
        save_interval=500,
        keep_period=2000,
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_force_total_task_eef_joint_remote",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=16,
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.01,
            force_head_loss_weight=0.01,
            force_frame_spike_weight=2.0,
            # joint + EEF: 老权重 (0.7 joint + 0.3 EEF, pos 0.3 + rot 1.0)
            # EEF warmup 2w2 步: 前 22000 步 EEF 线性升温, joint 主导;
            # 22000-30000 步 EEF 完全生效 (联合训练)。
            use_eef_loss=True,
            eef_only_mode=False,
            eef_warmup_steps=22000,     # EEF 在 2w2 步后才完全生效
            action_joint_weight=0.7,    # 老权重
            tool_extension=0.211,
            eef_pos_weight=0.3,         # 老权重
            eef_angle_weight=1.0,       # 老权重
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/data/group1/junjie008/datasets/total_task_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="perform the task",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.Pi0ForceWeightLoader(
            "/data/group1/junjie008/FA-openpi/checkpoints/pi05_force_erase_board_ft60_forcevla_lora_k16/erase_board_ft60_k16_w001/29999/params"
        ),
        log_interval=10,
        save_interval=500,
        keep_period=2000,
        num_train_steps=30_000,
    ),
    # ────────────────────────────────────────────────────────────────────────
    # total_task 本地测试版 (bs=8, 本地数据集/权重路径) — 与远端同名 config 一一对应
    #   eef_only_local : 数据集 total_task_flexiv_eef (EEF 坐标, loss 直接算)
    #   joint_only_local: 数据集 total_task_flexiv_ft60 (纯 joint loss)
    #   eef_joint_local : 数据集 total_task_flexiv_ft60 (joint + FK EEF, warmup 2w)
    # 权重: 本地 29999 checkpoint 热启动
    # ────────────────────────────────────────────────────────────────────────
    TrainConfig(
        name="pi05_force_total_task_eef_only_local",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=16,
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.01,
            force_head_loss_weight=0.01,
            force_frame_spike_weight=2.0,
            # EEF 坐标数据集: loss 直接在 EEF 空间算, 无需 FK 分支
            use_eef_loss=False,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/datasets/total_task_flexiv_eef",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="perform the task",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.Pi0ForceWeightLoader(
            "/mnt/hdd/sfy/FA-openpi/checkpoints/pi05_force_erase_board_ft60_forcevla_lora_k16/erase_board_ft60_k16_w001/29999/params"
        ),
        log_interval=10,
        save_interval=2000,
        keep_period=10000,
        num_train_steps=50_000,
    ),
    TrainConfig(
        name="pi05_force_total_task_joint_only_local",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=16,
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.01,
            force_head_loss_weight=0.01,
            force_frame_spike_weight=2.0,
            use_eef_loss=False,          # 纯 joint loss
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/datasets/total_task_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="perform the task",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.Pi0ForceWeightLoader(
            "/mnt/hdd/sfy/FA-openpi/checkpoints/pi05_force_erase_board_ft60_forcevla_lora_k16/erase_board_ft60_k16_w001/29999/params"
        ),
        log_interval=10,
        save_interval=2000,
        keep_period=10000,
        num_train_steps=50_000,
    ),
    TrainConfig(
        name="pi05_force_total_task_eef_joint_local",
        model=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            use_force=True, predict_force=True,
            control_action_dim=7,
            force_start_idx=7,
            force_dim=6,
            force_history_frames=2,
            use_ft_history=True,
            ft_history_steps=60,
            ft_input_dim=360,
            ft_output_dim=256,
            ft_encoder_type="mlp",
            ft_num_tokens=16,
            grad_route_mode="three_stage",
            action_loss_weight=0.9,
            force_loss_weight=0.01,
            force_head_loss_weight=0.01,
            force_frame_spike_weight=2.0,
            # joint + EEF: 老权重 (0.7 joint + 0.3 EEF, pos 0.3 + rot 1.0)
            # EEF warmup 2w 步: 前 20000 步 EEF 线性升温, joint 主导
            use_eef_loss=True,
            eef_only_mode=False,
            eef_warmup_steps=20000,
            action_joint_weight=0.7,
            tool_extension=0.211,
            eef_pos_weight=0.3,
            eef_angle_weight=1.0,
            num_experts=4, num_top_k=1,
        ),
        new_module_lr_multiplier=5.0,
        data=LeRobotPiperDataConfig(
            repo_id="/mnt/hdd/sfy/datasets/total_task_flexiv_ft60",
            observation_image_key="observation.image",
            observation_wrist_image_key="observation.wrist_image",
            default_prompt="perform the task",
            use_delta_joint_actions=True,
            use_delta_gripper_actions=False,
            use_force_data=True,
            predict_force=True,
            force_in_state=True,
            use_ft_history=True,
            ft_history_steps=60,
            action_dim=7,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_force.Pi0ForceConfig(
            pi05=True, action_horizon=30, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.Pi0ForceWeightLoader(
            "/mnt/hdd/sfy/FA-openpi/checkpoints/pi05_force_erase_board_ft60_forcevla_lora_k16/erase_board_ft60_k16_w001/29999/params"
        ),
        log_interval=10,
        save_interval=2000,
        keep_period=10000,
        num_train_steps=50_000,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
