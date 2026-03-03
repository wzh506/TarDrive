from pathlib import Path
from typing import Any, Dict, List

import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import plot_bev_with_agent
from nuplan.planning.script.builders.logging_builder import build_logger

logger = logging.getLogger(__name__)

# Reuse the same config hierarchy as run_pdm_score
CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def build_scene_loader(cfg: DictConfig, agent: AbstractAgent) -> SceneLoader:
    """Build a SceneLoader consistent with evaluation configs."""
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)

    # Use the agent's sensor config so features match the model's expectations
    sensor_config: SensorConfig = agent.get_sensor_config()

    scene_loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )
    print(f'sensor_config is {sensor_config}')
    # print(f'scene_filter is {scene_filter}')
    return scene_loader


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Visualize BEV maps with human and DiffusionDrive trajectories.

    Usage example (single-GPU, sequential worker):

        python navsim/planning/script/run_bev_visualization.py \
            train_test_split=navtest \
            worker=sequential \
            agent=diffusiondrive_agent \
            agent.checkpoint_path=/path/to/diffusiondrive_navsim.ckpt \
            experiment_name=diffusiondrive_bev_vis
    """

    build_logger(cfg)

    # Build agent from Hydra config (e.g., diffusiondrive_agent)
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    logger.info("Building SceneLoader for split=%s", cfg.train_test_split.data_split)
    scene_loader = build_scene_loader(cfg, agent)

    if len(scene_loader) == 0:
        logger.warning("No scenes found for split '%s'", cfg.train_test_split.data_split)
        return

    # Decide how many scenes to visualize; can be overridden via CLI if desired
    num_scenes: int = getattr(cfg, "num_scenes", 10)
    tokens: List[str] = scene_loader.tokens[:num_scenes]

    output_root = Path("./scripts/fig")
    output_root.mkdir(parents=True, exist_ok=True) #保存路径
    logger.info("Saving BEV visualizations for %d scenes to %s", len(tokens), output_root)

    for idx, token in enumerate(tokens):
        logger.info("[%d/%d] Visualizing token=%s", idx + 1, len(tokens), token)
        scene = scene_loader.get_scene_from_token(token)

        from navsim.visualization.plots import plot_bev_frame

        # 绘制bev
        frame_idx = scene.scene_metadata.num_history_frames - 1 # current frame
        fig, ax = plot_bev_frame(scene, frame_idx)
        out_path = output_root / f"navsim_bev_{token}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")


        # 绘制agent
        from navsim.visualization.plots import plot_bev_with_agent
        from navsim.agents.constant_velocity_agent import ConstantVelocityAgent

        # agent = ConstantVelocityAgent() #这里agent用的这个
        fig, ax = plot_bev_with_agent(scene, agent)
        out_path = output_root / f"navsim_agent_{token}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")


        # 绘制cameras
        from navsim.visualization.plots import plot_cameras_frame

        fig, ax = plot_cameras_frame(scene, frame_idx)
        out_path = output_root / f"navsim_cameras_{token}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")


        # 绘制cameras with annotations
        from navsim.visualization.plots import plot_cameras_frame_with_annotations

        fig, ax = plot_cameras_frame_with_annotations(scene, frame_idx)
        out_path = output_root / f"navsim_cameras_label_{token}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

        # 绘制cameras with lidar
        from navsim.visualization.plots import plot_cameras_frame_with_lidar

        fig, ax = plot_cameras_frame_with_lidar(scene, frame_idx)
        out_path = output_root / f"navsim_cameras_lidar_{token}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")


        # 绘制带有轨迹的camera图像，可以放在图中直接使用的那种
        from navsim.visualization.plots import plot_camera_with_trajectories
        from navsim.agents.constant_velocity_agent import ConstantVelocityAgent

        # 先建一个 agent（你也可以换成 diffusiondrive_agent）
        # agent = ConstantVelocityAgent()

        # Camera 图：比如前视 cam_f0
        fig, ax = plot_camera_with_trajectories(scene, agent, camera_name="cam_f0")
        out_path = output_root / f"navsim_agent_cam_f0_{token}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        # plt.close(fig)






        # 绘制自定义的曲线
        from navsim.visualization.plots import configure_bev_ax
        from navsim.visualization.bev import add_annotations_to_bev_ax, add_lidar_to_bev_ax


        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        ax.set_title("Custom plot")

        add_annotations_to_bev_ax(ax, scene.frames[frame_idx].annotations)
        add_lidar_to_bev_ax(ax, scene.frames[frame_idx].lidar)

        # configures frame to BEV view
        configure_bev_ax(ax)

        out_path = output_root / f"navsim_custum_{token}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

        # 绘制gif，lidar,camera因为未知原因缺少标注，等等问题，暂时无法使用~
        from navsim.visualization.plots import frame_plot_to_gif

        frame_indices = [idx for idx in range(len(scene.frames[:100]))]  # 前100 frames in scene
        file_name = f"{output_root}/navsim_gif{token}.gif"
        images = frame_plot_to_gif(file_name, plot_cameras_frame, scene, frame_indices)
        print(f"Saved GIF to {file_name}")



    logger.info("Finished BEV visualization.")


if __name__ == "__main__":
    main()