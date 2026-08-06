#!/usr/bin/env python
"""Piper 6-DoF 正向运动学（FK）解算 —— 用于真机日志 vs 数据集 EEF 轨迹对比

从 piper_description.urdf 提取的关节参数（SolidWorks 导出）：
  joint1-6: revolute，6 轴机械臂；joint7-8: prismatic（gripper）
用法: python piper_fk.py <jsonl> <输出csv>  [--state-key state] [--joint-dim 6]
"""
import json, re, sys, numpy as np

# ---- URDF 关节参数（从 piper_description.urdf 解析）----
URDF_PATH = "/mnt/hdd/sfy/RoboTwin/assets/embodiments/piper/urdf/piper_description.urdf"

def _load_urdf_joints(path):
    urdf = re.sub(r"<!--.*?-->", "", open(path).read(), flags=re.S)
    joints = re.findall(r'<joint\s+name="([^"]+)"\s+type="([^"]+)"\s*>(.*?)</joint>', urdf, re.S)
    out = []
    for name, typ, body in joints:
        xyz = re.search(r'xyz="([^"]+)"', body)
        rpy = re.search(r'rpy="([^"]+)"', body)
        ax = re.search(r'<axis\s+xyz="([^"]+)"\s*/>', body)
        out.append(dict(
            name=name, type=typ,
            xyz=np.array([float(v) for v in (xyz.group(1) if xyz else "0 0 0").split()]),
            rpy=np.array([float(v) for v in (rpy.group(1) if rpy else "0 0 0").split()]),
            axis=np.array([float(v) for v in (ax.group(1) if ax else "0 0 1").split()]),
        ))
    return out

JOINTS = _load_urdf_joints(URDF_PATH)

def rot_z(a): return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
def rot_y(a): return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
def rot_x(a): return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])

def rot_rpy(rpy):
    """URDF rpy: 绕固定轴 X->Y->Z 顺序旋转 R = Rz*Ry*Rx"""
    return rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])

def rot_axis(axis, theta):
    """绕任意单位轴旋转（Rodrigues）"""
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)

def fk(q, joint_indices=(0,1,2,3,4,5)):
    """输入 6 个关节角，返回末端 (link6) 位姿 T (4x4)。joint_indices 映射 state 里的关节序号。"""
    T = np.eye(4)
    for jidx in joint_indices:
        j = JOINTS[jidx]
        # 固定变换: T = T @ (R(rpy) | xyz; 0 0 0 1)
        Rf = rot_rpy(j["rpy"])
        Tf = np.eye(4); Tf[:3,:3] = Rf; Tf[:3,3] = j["xyz"]
        T = T @ Tf
        # 关节旋转（仅 revolute）
        if j["type"] == "revolute":
            qv = q[jidx] if np.ndim(q) else q
            Rq = rot_axis(j["axis"], qv)
            Tq = np.eye(4); Tq[:3,:3] = Rq
            T = T @ Tq
    return T

def pose_to_xyz_rpy(T):
    """提取 xyz 和 ZYX 欧拉角（roll,pitch,yaw）"""
    xyz = T[:3,3]
    R = T[:3,:3]
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-9:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0
    return xyz, np.array([roll, pitch, yaw])

def eef_pose_from_state(state, joint_indices=(0,1,2,3,4,5)):
    """从 7/13 维 state 提取前 6 关节角并 FK，返回 (xyz, rpy)"""
    q = np.array(state, dtype=np.float64)[:6]
    T = fk(q, joint_indices)
    return pose_to_xyz_rpy(T)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", help="真机日志 jsonl 或数据集 csv")
    p.add_argument("out_csv", help="输出 csv: step,x,y,z,roll,pitch,yaw")
    p.add_argument("--state-key", default="state")
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    rows = []
    for i, line in enumerate(open(args.jsonl)):
        if args.max_frames and i >= args.max_frames: break
        d = json.loads(line)
        st = d.get(args.state_key)
        if st is None or not isinstance(st, list) or len(st) < 6: continue
        xyz, rpy = eef_pose_from_state(st)
        rows.append((d.get("step", i), *xyz, *rpy))
    with open(args.out_csv, "w") as f:
        f.write("step,x,y,z,roll,pitch,yaw\n")
        for r in rows:
            f.write(",".join(f"{v:.6f}" for v in r) + "\n")
    print(f"FK done: {len(rows)} frames -> {args.out_csv}")
