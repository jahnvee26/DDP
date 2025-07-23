import numpy as np

# === Link lengths (updated from URDF) ===
L1 = 0.062   # base height
L2 = 0.16    # upper arm
L3 = 0.21    # forearm
L4 = 0.05    # wrist to gripper

def inverse_kinematics(x, y, z, pitch=0.0, roll=0.0):
    """
    Calculates joint angles for 5-DoF arm.
    Inputs:
        x, y, z: desired end-effector position
        pitch: desired wrist pitch angle (rad)
        roll: desired wrist roll angle (rad)
    Returns:
        List of 5 joint angles [theta1, theta2, theta3, theta4, theta5]
    """

    # 1) Base rotation (yaw)
    theta1 = np.arctan2(y, x)

    # 2) Project to arm plane
    r = np.sqrt(x**2 + y**2)
    z_offset = z - L1

    # Compensate wrist link length along the approach direction (pitch)
    r_eff = r - L4 * np.cos(pitch)
    z_eff = z_offset - L4 * np.sin(pitch)

    D = np.sqrt(r_eff**2 + z_eff**2)

    # 3) Check reachability
    if D > (L2 + L3):
        raise ValueError("Target out of reach!")

    # 4) Elbow angle (Law of Cosines)
    cos_theta3 = (D**2 - L2**2 - L3**2) / (2 * L2 * L3)
    theta3 = np.arccos(np.clip(cos_theta3, -1.0, 1.0))

    # 5) Shoulder angle
    alpha = np.arctan2(z_eff, r_eff)
    beta = np.arccos(np.clip((L2**2 + D**2 - L3**2) / (2 * L2 * D), -1.0, 1.0))
    theta2 = alpha + beta

    # 6) Wrist pitch
    theta4 = pitch - theta2 - theta3

    # 7) Wrist roll (directly from desired roll)
    theta5 = roll

    return [theta1, theta2, theta3, theta4, theta5]
