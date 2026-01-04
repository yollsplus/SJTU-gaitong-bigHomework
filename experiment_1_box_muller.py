"""
实验一：Box-Muller变换 - 从均匀分布生成正态分布
知识点：均匀分布在统计计算中的特殊地位

原理：
Box-Muller变换利用两个独立的U(0,1)均匀分布随机变量，
通过以下变换生成两个独立的标准正态分布随机变量：
    Z1 = sqrt(-2*ln(U1)) * cos(2*π*U2)
    Z2 = sqrt(-2*ln(U1)) * sin(2*π*U2)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils import print_section, print_result, save_figure, ResultLogger

class BoxMullerTransform:
    """Box-Muller变换实现"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def generate_uniform(self, n):
        """生成均匀分布随机数"""
        return np.random.uniform(0, 1, n)
    
    def box_muller_transform(self, n):
        """
        使用Box-Muller变换生成正态分布随机数
        
        参数:
            n: 需要生成的随机数个数
        返回:
            标准正态分布随机数数组
        """
        # 生成两组均匀分布随机数
        u1 = self.generate_uniform(n)
        u2 = self.generate_uniform(n)
        
        # Box-Muller变换
        z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
        
        return z1, z2
    
    def generate_normal(self, n, mu=0, sigma=1):
        """
        生成指定均值和标准差的正态分布随机数
        
        参数:
            n: 样本数量
            mu: 均值
            sigma: 标准差
        """
        z1, z2 = self.box_muller_transform(n)
        # 使用两组数据
        z = np.concatenate([z1, z2])[:n]
        return mu + sigma * z

def run_experiment():
    """运行实验"""
    print_section("实验一：Box-Muller变换生成正态分布")
    
    logger = ResultLogger("实验一_Box-Muller变换")
    
    bm = BoxMullerTransform()
    n_samples = 10000
    
    # 1. 生成标准正态分布
    logger.add_section("步骤1：生成标准正态分布 N(0,1)")
    z1, z2 = bm.box_muller_transform(n_samples)
    
    # 统计检验
    ks_stat_z1, p_value_z1 = stats.kstest(z1, 'norm')
    ks_stat_z2, p_value_z2 = stats.kstest(z2, 'norm')
    
    logger.add_result("Z1 K-S检验统计量", ks_stat_z1)
    logger.add_result("Z1 p值", p_value_z1)
    logger.add_result("Z2 K-S检验统计量", ks_stat_z2)
    logger.add_result("Z2 p值", p_value_z2)
    
    # 2. 生成不同参数的正态分布
    logger.add_section("步骤2：生成 N(5, 2²) 正态分布")
    normal_samples = bm.generate_normal(n_samples, mu=5, sigma=2)
    
    sample_mean = np.mean(normal_samples)
    sample_std = np.std(normal_samples, ddof=1)
    logger.add_result("样本均值", sample_mean)
    logger.add_result("样本标准差", sample_std)
    logger.add_result("理论均值", 5)
    logger.add_result("理论标准差", 2)
    logger.add_result("均值误差", abs(sample_mean - 5))
    logger.add_result("标准差误差", abs(sample_std - 2))
    
    # 3. 可视化
    create_visualizations(z1, z2, normal_samples)
    
    logger.add_section("实验完成")
    logger.add_line("生成图表:")
    logger.add_line("  - exp1_box_muller_transform.png")
    logger.add_line("  - exp1_custom_normal.png")
    
    logger.save()
    print("\n✓ 实验一完成！")
    return z1, z2, normal_samples

def create_visualizations(z1, z2, normal_samples):
    """创建可视化图表"""
    
    # 图1：Box-Muller变换过程展示
    fig1 = plt.figure(figsize=(15, 10))
    
    # 子图1：均匀分布到正态分布的转换（2D散点图）
    ax1 = plt.subplot(2, 3, 1)
    n_demo = 1000
    u1 = np.random.uniform(0, 1, n_demo)
    u2 = np.random.uniform(0, 1, n_demo)
    scatter = ax1.scatter(u1, u2, alpha=0.5, c=np.arange(n_demo), cmap='viridis', s=20)
    ax1.set_xlabel('U1 ~ Uniform(0,1)', fontsize=11)
    ax1.set_ylabel('U2 ~ Uniform(0,1)', fontsize=11)
    ax1.set_title('(a) 均匀分布输入空间', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='样本序号')
    
    # 子图2：极坐标变换
    ax2 = plt.subplot(2, 3, 2, projection='polar')
    r = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    scatter = ax2.scatter(theta, r, alpha=0.5, c=np.arange(n_demo), cmap='viridis', s=20)
    ax2.set_title('(b) 极坐标变换 (r, θ)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='样本序号')
    
    # 子图3：正态分布输出空间
    ax3 = plt.subplot(2, 3, 3)
    z1_demo = r * np.cos(theta)
    z2_demo = r * np.sin(theta)
    scatter = ax3.scatter(z1_demo, z2_demo, alpha=0.5, c=np.arange(n_demo), cmap='viridis', s=20)
    ax3.set_xlabel('Z1 ~ N(0,1)', fontsize=11)
    ax3.set_ylabel('Z2 ~ N(0,1)', fontsize=11)
    ax3.set_title('(c) 正态分布输出空间', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax3.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='样本序号')
    
    # 子图4：Z1的直方图和理论分布
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(z1, bins=60, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Z1样本')
    x = np.linspace(-4, 4, 100)
    ax4.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='N(0,1)理论密度')
    ax4.set_xlabel('Z1值', fontsize=11)
    ax4.set_ylabel('密度', fontsize=11)
    ax4.set_title('(d) Z1分布与理论对比', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 子图5：Z2的直方图和理论分布
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(z2, bins=60, density=True, alpha=0.7, color='lightcoral', edgecolor='black', label='Z2样本')
    ax5.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='N(0,1)理论密度')
    ax5.set_xlabel('Z2值', fontsize=11)
    ax5.set_ylabel('密度', fontsize=11)
    ax5.set_title('(e) Z2分布与理论对比', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 子图6：Q-Q图
    ax6 = plt.subplot(2, 3, 6)
    stats.probplot(z1, dist="norm", plot=ax6)
    ax6.set_title('(f) Z1的Q-Q图', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('实验一：Box-Muller变换详细过程', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_figure(fig1, 'exp1_box_muller_transform.png')
    plt.close()
    
    # 图2：自定义参数正态分布
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 直方图
    axes[0, 0].hist(normal_samples, bins=50, density=True, alpha=0.7, 
                     color='lightgreen', edgecolor='black', label='生成样本')
    x = np.linspace(-2, 12, 200)
    axes[0, 0].plot(x, stats.norm.pdf(x, 5, 2), 'r-', linewidth=2, label='N(5,4)理论密度')
    axes[0, 0].set_xlabel('值', fontsize=11)
    axes[0, 0].set_ylabel('密度', fontsize=11)
    axes[0, 0].set_title('(a) N(5, 2²)分布拟合', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 累积分布函数
    sorted_samples = np.sort(normal_samples)
    empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    theoretical_cdf = stats.norm.cdf(sorted_samples, 5, 2)
    
    axes[0, 1].plot(sorted_samples, empirical_cdf, 'b-', linewidth=1.5, 
                     label='经验CDF', alpha=0.7)
    axes[0, 1].plot(sorted_samples, theoretical_cdf, 'r--', linewidth=2, 
                     label='理论CDF')
    axes[0, 1].set_xlabel('值', fontsize=11)
    axes[0, 1].set_ylabel('累积概率', fontsize=11)
    axes[0, 1].set_title('(b) 累积分布函数对比', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 核密度估计
    axes[1, 0].hist(normal_samples, bins=50, density=True, alpha=0.5, 
                     color='lightgreen', edgecolor='black', label='直方图')
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(normal_samples)
    x_kde = np.linspace(-2, 12, 200)
    axes[1, 0].plot(x_kde, kde(x_kde), 'b-', linewidth=2, label='核密度估计')
    axes[1, 0].plot(x_kde, stats.norm.pdf(x_kde, 5, 2), 'r--', linewidth=2, 
                     label='理论密度')
    axes[1, 0].set_xlabel('值', fontsize=11)
    axes[1, 0].set_ylabel('密度', fontsize=11)
    axes[1, 0].set_title('(c) 核密度估计对比', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q图
    stats.probplot(normal_samples, dist=stats.norm, sparams=(5, 2), plot=axes[1, 1])
    axes[1, 1].set_title('(d) Q-Q图检验', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('实验一：自定义参数正态分布 N(5, 2²)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig2, 'exp1_custom_normal.png')
    plt.close()

if __name__ == "__main__":
    run_experiment()
