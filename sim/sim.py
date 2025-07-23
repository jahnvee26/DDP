import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
from PIL import Image

from pyrep import PyRep
from pyrep.robots.arms.arm import Arm
from pyrep.objects.vision_sensor import VisionSensor

from ik_solver.ik_script import inverse_kinematics
from vla_planner import run_vla_action  # ← direct import, no need for server

SCENE_FILE = '/home/janvi/CoppeliaSim_Edu_V4_9_0_rev6_Ubuntu24_04/scenes/roarm.ttt'  # ← Update with actual scene path
ARM_NAME = 'RoArm'
CAMERA_NAME = 'Vision_sensor'

pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()

arm = Arm(ARM_NAME)
camera = VisionSensor(CAMERA_NAME)

print(f"[INFO] Loaded robot: {ARM_NAME}")
print(f"[INFO] Loaded camera: {CAMERA_NAME}")

try:
    while True:
        # 1. Capture
        img_np = camera.capture_rgb()
        img_np = (img_np * 255).astype(np.uint8)
        image = Image.fromarray(img_np)

        # 2. VLA
        prompt = "In: What action should the robot take to pick up the red cube?\nOut:"
        action = run_vla_action(image, prompt)
        print(f"[INFO] VLA Action: {action}")

        # 3. IK
        target_x = action['x']
        target_y = action['y']
        target_z = action['z']
        target_pitch = action['pitch']

        joint_angles = inverse_kinematics(target_x, target_y, target_z, target_pitch)
        arm.set_joint_target_positions(joint_angles)

        pr.step()
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[INFO] Keyboard interrupt — shutting down.")

finally:
    pr.stop()
    pr.shutdown()
