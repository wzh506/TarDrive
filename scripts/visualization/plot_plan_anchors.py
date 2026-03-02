import os
import numpy as np
import matplotlib.pyplot as plt

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig


def main(output_path: str | None = None) -> None:
    """Load plan_anchor (kmeans_navsim_traj_20.npy) and plot as 2D trajectories.

    - Each mode (20) will be drawn as one polyline in (x, y).
    - Time step index is implicit along the curve.
    """
    cfg = TransfuserConfig()
    plan_anchor_path = cfg.plan_anchor_path

    if not os.path.exists(plan_anchor_path):
        raise FileNotFoundError(f"plan_anchor_path not found: {plan_anchor_path}")

    anchors = np.load(plan_anchor_path)  # expected shape: (20, 8, 2)

    if anchors.ndim != 3 or anchors.shape[-1] != 2:
        raise ValueError(f"Unexpected plan_anchor shape: {anchors.shape}, expected (num_modes, T, 2)")

    num_modes, T, _ = anchors.shape
    print(f"Loaded plan anchors from {plan_anchor_path}, shape = {anchors.shape}")

    plt.figure(figsize=(6, 6))

    # Plot each mode as a trajectory in XY
    for i in range(num_modes):
        traj = anchors[i]  # (T, 2)
        x = traj[:, 0]
        y = traj[:, 1]
        plt.plot(x, y, marker="o", label=f"mode {i}")

    # Draw ego origin
    plt.scatter([0.0], [0.0], c="k", marker="x", s=80, label="ego (0,0)")

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Plan Anchors (kmeans_navsim_traj_20)")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    # 如果轨迹太多，可以把 legend 去掉或只显示一部分
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # 默认保存到 assets/plan_anchors.png
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fig")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "plan_anchors.png")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
