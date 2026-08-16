"""JAX 可微 Piper 正向运动学（含夹爪+传感器末端延伸）。

从 RoboTwin/assets/embodiments/piper/urdf/piper_description.urdf 提取的关节参数
（与 scripts/piper_fk.py 一致），用 jax.numpy 实现以便在 compute_loss 中
支持反向传播（EEF 位姿 loss）。

末端约定:
  - link6 = 第 6 关节后的 wrist 末端
  - 夹爪 (joint7/8, 0.13503m) + 力传感器 (0.076m) 沿 link6 z 轴延伸
  - 真实接触点 = link6 + z * tool_extension (默认 0.211m)

用法:
  from openpi.models import piper_fk_jax
  T = piper_fk_jax.fk_batch(joints)          # [..., 6] -> [..., 4, 4]
  xyz, rpy = piper_fk_jax.pose_from_joints(joints)  # [..., 6] -> ([...,3],[...,3])
"""

import jax
import jax.numpy as jnp

# URDF 关节参数 (SolidWorks 导出, 与 piper_fk.py 的 JOINTS 一致)
# 每项: (parent->child 固定变换 xyz, rpy, 关节轴, 类型)
#   fixed: 只应用固定变换
#   revolute: 固定变换 + 绕轴旋转 q
# 顺序与 state 关节 [q1..q6] 对应
JOINT_PARAMS = [
    dict(xyz=jnp.array([0.0, 0.0, 0.123]), rpy=jnp.array([0.0, 0.0, -1.5708]), axis=jnp.array([0.0, 0.0, 1.0]), revolute=True),
    dict(xyz=jnp.array([0.0, 0.0, 0.0]), rpy=jnp.array([1.5708, 0.0, -1.5708]), axis=jnp.array([0.0, 0.0, 1.0]), revolute=True),
    dict(xyz=jnp.array([0.28358, 0.028726, 0.0]), rpy=jnp.array([0.0, 0.0, 0.10095]), axis=jnp.array([0.0, 0.0, 1.0]), revolute=True),
    dict(xyz=jnp.array([-0.24221, 0.068514, 0.0]), rpy=jnp.array([-1.5708, 0.0, 1.3826]), axis=jnp.array([0.0, 0.0, 1.0]), revolute=True),
    dict(xyz=jnp.array([0.0, 0.0, 0.0]), rpy=jnp.array([1.5708, 0.0, 0.0]), axis=jnp.array([0.0, 0.0, 1.0]), revolute=True),
    dict(xyz=jnp.array([0.0, 0.091, 0.0014165]), rpy=jnp.array([-1.5708, -3.1415926, 0.0]), axis=jnp.array([0.0, 0.0, 1.0]), revolute=True),
]

# 默认工具延伸: 夹爪 0.13503 + 传感器 0.076
DEFAULT_TOOL_EXTENSION = 0.211


def _rot_x(a: jax.Array) -> jax.Array:
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: jax.Array) -> jax.Array:
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: jax.Array) -> jax.Array:
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_rpy(rpy: jax.Array) -> jax.Array:
    """URDF rpy: 固定轴 X->Y->Z 顺序 R = Rz*Ry*Rx"""
    return _rot_z(rpy[2]) @ _rot_y(rpy[1]) @ _rot_x(rpy[0])


def _rot_axis(axis: jax.Array, theta: jax.Array) -> jax.Array:
    """绕单位轴旋转 (Rodrigues)"""
    axis = axis / jnp.linalg.norm(axis)
    K = jnp.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return jnp.eye(3) + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * (K @ K)


def _mat4(R: jax.Array, t: jax.Array) -> jax.Array:
    T = jnp.zeros((4, 4), dtype=R.dtype)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(t)
    T = T.at[3, 3].set(1.0)
    return T


def fk(q: jax.Array, tool_extension: float = DEFAULT_TOOL_EXTENSION) -> jax.Array:
    """单个位姿 FK: q[6] -> T[4,4] (link6 末端 + tool_extension 延伸)。"""
    T = jnp.eye(4, dtype=q.dtype)
    for i, jp in enumerate(JOINT_PARAMS):
        # 固定变换
        Tf = _mat4(_rot_rpy(jp["rpy"]), jp["xyz"])
        T = T @ Tf
        # 关节旋转
        if jp["revolute"]:
            Rq = _rot_axis(jp["axis"], q[i])
            Tq = _mat4(Rq, jnp.zeros(3, dtype=q.dtype))
            T = T @ Tq
    # 工具延伸: link6 z 轴方向 + tool_extension
    z_axis = T[:3, :3] @ jnp.array([0.0, 0.0, 1.0], dtype=q.dtype)
    T_tool = T.at[:3, 3].set(T[:3, 3] + z_axis * tool_extension)
    return T_tool


def fk_batch(q: jax.Array, tool_extension: float = DEFAULT_TOOL_EXTENSION) -> jax.Array:
    """批量 FK: q[..., 6] -> T[..., 4, 4]。"""
    return jax.vmap(lambda qq: fk(qq, tool_extension))(q)


def pose_from_joints(q: jax.Array, tool_extension: float = DEFAULT_TOOL_EXTENSION):
    """批量: q[..., 6] -> (xyz[..., 3], rpy[..., 3])。"""
    flat = q.reshape(-1, 6)
    Ts = fk_batch(flat, tool_extension)
    xyz = Ts[..., :3, 3]
    R = Ts[..., :3, :3]
    # ZYX 欧拉角 (与 piper_fk.pose_to_xyz_rpy 一致)
    sy = jnp.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
    roll = jnp.arctan2(R[..., 2, 1], R[..., 2, 2])
    pitch = jnp.arctan2(-R[..., 2, 0], sy)
    yaw = jnp.arctan2(R[..., 1, 0], R[..., 0, 0])
    rpy = jnp.stack([roll, pitch, yaw], axis=-1)
    return xyz.reshape(q.shape[:-1] + (3,)), rpy.reshape(q.shape[:-1] + (3,))


def wrap_angle(d: jax.Array) -> jax.Array:
    """角度回绕到 [-pi, pi]。"""
    return (d + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def jacobian(q: jax.Array, tool_extension: float = DEFAULT_TOOL_EXTENSION) -> jax.Array:
    """EEF 雅可比矩阵 J(q): [6, 6]。

    J = ∂(xyz, rpy)/∂q, 在关节位置 q 处线性化。
    J @ delta 给出 EEF 增量 (d_xyz, d_rpy)，无角度回绕（线性小量）。
    """
    def _pose6(qq: jax.Array) -> jax.Array:
        xyz, rpy = pose_from_joints(qq[None, :], tool_extension)
        return jnp.concatenate([xyz[0], rpy[0]], axis=-1)  # [6]

    return jax.jacfwd(_pose6)(q)  # [6, 6]


def jacobian_batch(q: jax.Array, tool_extension: float = DEFAULT_TOOL_EXTENSION) -> jax.Array:
    """批量雅可比: q[..., 6] -> J[..., 6, 6]。"""
    return jax.vmap(lambda qq: jacobian(qq, tool_extension))(q)


def eef_delta_from_joint_delta(
    q_current: jax.Array,
    joint_delta: jax.Array,
    tool_extension: float = DEFAULT_TOOL_EXTENSION,
) -> jax.Array:
    """用雅可比把关节 delta 投影为 EEF delta（无绕角）。

    q_current: [..., 6] 当前绝对关节角（物理空间）
    joint_delta: [..., 6] 关节增量（物理空间）
    返回: [..., 6] EEF 增量 = J(q_current) @ joint_delta
    """
    J = jacobian_batch(q_current, tool_extension)          # [..., 6, 6]
    return jnp.einsum("...ij,...j->...i", J, joint_delta)  # [..., 6]


def eef_ik(
    target_pose: jax.Array,
    q_init: jax.Array,
    tool_extension: float = DEFAULT_TOOL_EXTENSION,
    max_iter: int = 60,
    tol: float = 1e-4,
    lr: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """数值 IK: EEF 位姿 [xyz(3), rpy(3)] → 关节角 [6]。

    用含 tool_extension 延伸的雅可比迭代求解，与训练/转换用的
    fk()/pose_from_joints() 完全同一坐标系（link6 + 夹爪 + 传感器）。

    Args:
        target_pose: [6] 目标 EEF 位姿 [x, y, z, roll, pitch, yaw]
        q_init: [6] 初始关节角（建议用当前关节，保证解连续性）
        tool_extension: 工具延伸长度（默认 0.211 = 夹爪 0.13503 + 传感器 0.076）
        max_iter: 最大迭代次数
        tol: 收敛阈值（位置 m，角度 rad）
        lr: 更新步长（阻尼系数，0.5~1.0）

    Returns:
        (q_sol, converged, err): 解出的关节角 / 是否收敛 / 最终误差 [6]
    """
    lam = 1e-3

    def _step(q: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        xyz, rpy = pose_from_joints(q[None, :], tool_extension)  # [1,3]
        cur = jnp.concatenate([xyz[0], rpy[0]], axis=-1)          # [6]
        err = target_pose - cur
        # 角度分量回绕（避免 ±π 跳变）
        err = err.at[3:].set(wrap_angle(err[3:]))
        J = jacobian(q, tool_extension)                           # [6,6]
        # 阻尼最小二乘: delta_q = (J^T J + λI)^-1 J^T err
        dq = jnp.linalg.solve(J.T @ J + lam * jnp.eye(6), J.T @ err)
        return q + lr * dq, err, dq

    def _body(q, _):
        q_new, _, _ = _step(q)
        return q_new, q_new

    # 固定迭代次数（避免 Python for 循环在 jit/vmap 下的 tracer 问题）。
    q, _ = jax.lax.scan(_body, q_init, None, length=max_iter)
    _, err, _ = _step(q)
    converged = jnp.linalg.norm(err) < tol
    return q, converged, err


def eef_ik_batch(
    target_poses: jax.Array,
    q_inits: jax.Array,
    tool_extension: float = DEFAULT_TOOL_EXTENSION,
    max_iter: int = 60,
    tol: float = 1e-4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """批量数值 IK: target_poses [..., 6], q_inits [..., 6] → q_sol [..., 6]。"""
    return jax.vmap(
        lambda tp, qi: eef_ik(tp, qi, tool_extension, max_iter, tol)
    )(target_poses, q_inits)
