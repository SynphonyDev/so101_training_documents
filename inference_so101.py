#!/usr/bin/env python3
"""
SO-101 SmolVLA Inference Script
Runs trained policy on real robot using the official lerobot inference pipeline
"""

import time
import torch
import numpy as np

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device


def main():
    # Configuration
    ROBOT_PORT = "/dev/tty.usbmodem5A7A0549711"
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    TASK = "Pick the orange object and place it in the box"
    MODEL_PATH = "synphony/smolvla-so101-demo"

    print("=" * 70)
    print("SO-101 SmolVLA Inference")
    print("=" * 70)

    # Load policy
    print(f"\n[1/4] Loading SmolVLA policy from {MODEL_PATH}...")
    policy = SmolVLAPolicy.from_pretrained(MODEL_PATH)
    policy.eval()
    device = get_safe_torch_device(str(next(policy.parameters()).device))
    print(f"✓ Model loaded on device: {device}")
    print(f"✓ Model parameters: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M")

    # Initialize robot with camera
    print(f"\n[2/4] Connecting to SO-101 robot on {ROBOT_PORT}...")
    # Configure only the real camera (camera1)
    # The other cameras (camera2, camera3, empty_camera_0, empty_camera_1) will be added manually
    camera_config = {
        "camera1": OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=30
        )
    }
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id="so101_follower",
        cameras=camera_config,
        use_degrees=True  # CRITICAL: Model was trained with degrees, not normalized range!
    )
    robot = SO101Follower(robot_config)
    robot.connect()
    print("✓ Robot connected")

    # Create dataset features from robot features
    print(f"\n[3/4] Setting up preprocessing pipeline...")
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # CRITICAL FIX: Add all expected cameras to observation features
    # Model expects 5 cameras but robot only provides camera1
    # We need to register all cameras in the observation features so they're included in dataset_features
    extended_obs_features = robot.observation_features.copy()

    # Add the missing cameras with appropriate shapes (HWC format)
    if "camera1" in robot.observation_features:
        extended_obs_features["camera2"] = (256, 256, 3)  # camera2 and camera3 are 256x256
        extended_obs_features["camera3"] = (256, 256, 3)
        extended_obs_features["empty_camera_0"] = (480, 640, 3)  # empty cameras are 480x640
        extended_obs_features["empty_camera_1"] = (480, 640, 3)

    # CRITICAL FIX: Use hw_to_dataset_features instead of create_initial_features
    # This properly converts hardware features (including cameras) to dataset format
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(extended_obs_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # Create preprocessor and postprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=MODEL_PATH,
        dataset_stats=None,  # Using model's built-in stats
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
        },
    )
    print("✓ Preprocessing pipeline ready")

    # Reset policy before inference
    print(f"\n[4/4] Preparing for inference...")
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    print("✓ Policy ready")

    # Run inference loop
    print(f"\n[Running] Starting inference loop...")
    print(f"Task: '{TASK}'")
    print("Press Ctrl+C to stop\n")

    try:
        step = 0
        while True:
            start_time = time.time()

            # Get observation from robot (includes camera images and joint positions)
            obs = robot.get_observation()

            # CRITICAL FIX: Add dummy cameras BEFORE processing
            # Model expects 5 cameras but robot only provides camera1
            # Add them to the raw observation so build_dataset_frame can include them
            if "camera1" in obs:
                # Create dummy images matching expected shapes (HWC format, uint8)
                dummy_256 = np.zeros((256, 256, 3), dtype=np.uint8)  # For camera2, camera3
                dummy_480 = np.zeros((480, 640, 3), dtype=np.uint8)  # For empty cameras

                obs["camera2"] = dummy_256
                obs["camera3"] = dummy_256
                obs["empty_camera_0"] = dummy_480
                obs["empty_camera_1"] = dummy_480

            # Apply robot observation processor (default is identity)
            obs_processed = robot_observation_processor(obs)

            # Build observation frame in dataset format (now includes all 5 cameras)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            # Predict action using the official pipeline
            action_tensor = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=TASK,
                robot_type=robot.name,
            )

            # Convert action tensor to robot action dict format
            action_dict = make_robot_action(action_tensor, dataset_features)

            # Apply robot action processor (default is identity)
            action_to_send = robot_action_processor((action_dict, obs))

            # Send action to robot
            robot.send_action(action_to_send)

            # Timing and logging
            inference_time = time.time() - start_time
            step += 1

            if step % 10 == 0:
                # Extract first few action values for display
                action_values = list(action_dict.values())[:3]
                print(f"Step {step:4d} | Time: {inference_time*1000:5.1f}ms | "
                      f"Action: {action_values}")

            # Control loop rate (30 Hz)
            time.sleep(max(0, 1/30 - inference_time))

    except KeyboardInterrupt:
        print("\n\nStopping inference...")

    finally:
        # Cleanup
        print("\nDisconnecting devices...")
        robot.disconnect()
        print("✓ Done")


if __name__ == "__main__":
    main()
