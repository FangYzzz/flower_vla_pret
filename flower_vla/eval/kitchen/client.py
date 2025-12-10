# franka_client.py
import cv2
import numpy as np
import requests
import json_numpy
from json_numpy import loads
json_numpy.patch()

SERVER = "http://<server_ip>:8003"

# 1) 先 reset 一下任务
task_text = "open the cabinet and pick up the mug"
requests.post(f"{SERVER}/reset", json={"text": task_text})

# 2) 在你的控制循环里：
while True:
    # 从两路相机读图像（示意）
    # primary: 第三人称相机
    # secondary: 手腕相机
    ret1, img1 = primary_cam.read()   # HxWx3, uint8 (BGR)
    ret2, img2 = secondary_cam.read()

    if not (ret1 and ret2):
        continue

    # OpenCV 是 BGR，这里最好转成 RGB，和训练时保持一致
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 发送请求（json_numpy 会自动把 numpy 数组编码成 JSON）
    resp = requests.post(
        f"{SERVER}/query",
        json={
            "primary_image": img1,
            "secondary_image": img2,
            # 可选：ensemble / multistep 覆盖
        }
    )

    action = loads(resp.json())  # 变回 numpy 数组
    # action 应该形状类似 (1, 8) 或 (8,)
    # 前 7 维: Franka 7 个关节目标位置 (rad)
    # 最后一维: gripper (开合)

    joint_targets = action[0, :7]
    gripper_target = action[0, 7]

    # 把 joint_targets 发给你的 Franka 控制接口：
    # - 如果你是 ROS2 + ros2_control: 发布到 /joint_trajectory_controller/follow_joint_trajectory
    # - 或者用 libfranka / Franka ROS 的 joint position controller
    send_joint_command(joint_targets)
    send_gripper_command(gripper_target)

    # 根据 DATASET_FREQUENCY_MAP 设置控制频率，比如 20 Hz / 10 Hz
    rate.sleep()
