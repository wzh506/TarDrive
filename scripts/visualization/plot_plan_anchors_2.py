import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
# from navsim.agents.tardrive.transfuser_config import TransfuserConfig

def main(output_path: str | None = None) -> None:
    """Load plan_anchor (kmeans_navsim_traj_20.npy) and plot as 2D trajectories
    with anchored Gaussian distributions (Clean version).
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

    plt.switch_backend('Agg')
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # 获取 tab20 颜色表
    colors = cm.get_cmap('tab20', num_modes)
    np.random.seed(42)

    # Plot each mode as a trajectory in XY with Gaussian scatter
    for i in range(num_modes):
        traj = anchors[i]  # (T, 2)
        x = traj[:, 0]
        y = traj[:, 1]
        c = colors(i)
        
        num_scatter = 800  # 每条轨迹的散点数量
        
        t_steps = np.arange(T)
        t_scatter = np.random.uniform(0, T - 1, num_scatter)
        
        x_center = np.interp(t_scatter, t_steps, x)
        y_center = np.interp(t_scatter, t_steps, y)
        
        std_dev = 0.2 + 0.5 * (t_scatter / (T - 1)) 
        
        x_noise = x_center + np.random.normal(0, std_dev * 0.8, num_scatter)
        y_noise = y_center + np.random.normal(0, std_dev * 1.2, num_scatter)
        
        # [修改点2]：调大散点(s=12)，加重颜色(alpha=0.4)
        ax.scatter(x_noise, y_noise, color=c, s=12, alpha=0.4, edgecolors='none', zorder=1)
        
        # 绘制主轨迹线
        ax.plot(x, y, color=c, linewidth=2.5, zorder=2)

    # [修改点1]：删除了 FancyBboxPatch (自车框) 的绘制代码

    # [修改点3]：保证比例一致，并隐藏所有坐标轴、网格和边框
    ax.set_aspect('equal')
    ax.axis('off')

    # 默认保存路径
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fig")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "plan_anchors_gaussian_clean.png")

    plt.tight_layout()
    # bbox_inches='tight' 和 pad_inches=0 确保导出的图片极其干净，没有多余白边
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    print(f"Saved figure to: {output_path}")

if __name__ == "__main__":
    main()