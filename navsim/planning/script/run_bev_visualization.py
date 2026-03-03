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

    output_root = Path(cfg.output_dir) / "bev_visualization"
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Saving BEV visualizations for %d scenes to %s", len(tokens), output_root)

    for idx, token in enumerate(tokens):
        logger.info("[%d/%d] Visualizing token=%s", idx + 1, len(tokens), token)
        scene = scene_loader.get_scene_from_token(token)

        # Plot BEV with human (GT) and agent (DiffusionDrive) trajectory
        fig, ax = plot_bev_with_agent(scene, agent)

        out_path = output_root / f"navsim_bev_{token}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    logger.info("Finished BEV visualization.")


if __name__ == "__main__":
    main()
#这个可以直接绘制论文中要得bev图像