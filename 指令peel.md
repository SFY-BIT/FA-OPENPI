cd /mnt/hdd/sfy/FA-openpi && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_task_peel_ft60_noforce \
    --port=8000 \
    policy:checkpoint \
    --policy.config=pi05_plain_total_task_peel_joint_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/peel_05_joint/39999



cd /mnt/hdd/sfy/FA-openpi && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_task_peel_ft60 \
    --port=8000 \
    policy:checkpoint \
    --policy.config=pi05_force_total_task_peel_joint_only_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/peel_FA_joint/39999



cd /mnt/hdd/sfy/FA-openpi && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_task_peel_eef_abs_noforce \
    --port=8000 \
    --action-space=EEF --action-rep=abs \
    policy:checkpoint \
    --policy.config=pi05_plain_total_task_peel_eef_v2_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/peel_05_eef/39999



cd /mnt/hdd/sfy/FA-openpi && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_task_peel_eef_abs \
    --port=8000 \
    --action-space=EEF --action-rep=abs \
    policy:checkpoint \
    --policy.config=pi05_force_total_task_peel_eef_v2_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/peel_FA_eef/39999



cd /mnt/hdd/sfy/FA-openpi && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlinf

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
python scripts/serve_policy.py \
    --norm-stats-dir=/mnt/hdd/sfy/datasets/total_task_peel_ft60 \
    --port=8000 \
    policy:checkpoint \
    --policy.config=pi05_force_total_task_peel_eef_joint_remote \
    --policy.dir=/mnt/hdd/sfy/FA-openpi/checkpoints/peel_FA_daul/39999
