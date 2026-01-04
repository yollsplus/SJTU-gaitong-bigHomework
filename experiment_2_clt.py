"""
实验二：中心极限定理的数值验证
知识点：中心极限定理 (Central Limit Theorem, CLT)

原理：
设 X1, X2, ..., Xn 是独立同分布的随机变量，期望为μ，方差为σ²，
则当n足够大时，样本均值的标准化变量近似服从标准正态分布：
    (X̄ - μ) / (σ/√n) → N(0, 1)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from utils import print_section, print_result, save_figure, ResultLogger

class CentralLimitTheorem:
    """中心极限定理验证"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        
    def generate_samples(self, distribution, n_samples, sample_size, **params):
        """
        生成样本并计算样本均值
        
        参数:
            distribution: 分布类型 ('uniform', 'exponential', 'binomial', 'poisson')
            n_samples: 重复抽样次数
            sample_size: 每次抽样的样本量
            params: 分布参数
        """
        sample_means = []
        
        for _ in range(n_samples):
            if distribution == 'uniform':
                a, b = params.get('a', 0), params.get('b', 1)
                samples = np.random.uniform(a, b, sample_size)
            elif distribution == 'exponential':
                lam = params.get('lambda', 1)
                samples = np.random.exponential(1/lam, sample_size)
            elif distribution == 'binomial':
                n, p = params.get('n', 10), params.get('p', 0.5)
                samples = np.random.binomial(n, p, sample_size)
            elif distribution == 'poisson':
                lam = params.get('lambda', 5)
                samples = np.random.poisson(lam, sample_size)
            elif distribution == 'chi2':
                df = params.get('df', 2)
                samples = np.random.chisquare(df, sample_size)
            else:
                raise ValueError(f"不支持的分布: {distribution}")
            
            sample_means.append(np.mean(samples))
        
        return np.array(sample_means)
    
    def get_theoretical_params(self, distribution, **params):
        """获取理论均值和标准差"""
        if distribution == 'uniform':
            a, b = params.get('a', 0), params.get('b', 1)
            mu = (a + b) / 2
            sigma = np.sqrt((b - a)**2 / 12)
        elif distribution == 'exponential':
            lam = params.get('lambda', 1)
            mu = 1 / lam
            sigma = 1 / lam
        elif distribution == 'binomial':
            n, p = params.get('n', 10), params.get('p', 0.5)
            mu = n * p
            sigma = np.sqrt(n * p * (1 - p))
        elif distribution == 'poisson':
            lam = params.get('lambda', 5)
            mu = lam
            sigma = np.sqrt(lam)
        elif distribution == 'chi2':
            df = params.get('df', 2)
            mu = df
            sigma = np.sqrt(2 * df)
        
        return mu, sigma

def run_experiment():
    """运行实验"""
    print_section("实验二：中心极限定理的数值验证")
    
    logger = ResultLogger("实验二_中心极限定理")
    
    clt = CentralLimitTheorem()
    
    # 实验配置
    distributions = [
        ('uniform', {'a': 0, 'b': 10}, '均匀分布U(0,10)'),
        ('exponential', {'lambda': 0.5}, '指数分布Exp(0.5)'),
        ('chi2', {'df': 2}, '卡方分布χ²(2)'),
        ('poisson', {'lambda': 5}, '泊松分布Poisson(5)')
    ]
    
    sample_sizes = [5, 10, 30, 50]
    n_samples = 10000
    
    results = {}
    
    # 对每个分布进行实验
    for dist_name, params, label in distributions:
        logger.add_section(label)
        mu, sigma = clt.get_theoretical_params(dist_name, **params)
        logger.add_result("理论均值 μ", mu)
        logger.add_result("理论标准差 σ", sigma)
        
        results[dist_name] = {}
        
        for n in sample_sizes:
            sample_means = clt.generate_samples(dist_name, n_samples, n, **params)
            
            # 标准化
            standardized = (sample_means - mu) / (sigma / np.sqrt(n))
            
            # K-S检验
            ks_stat, p_value = stats.kstest(standardized, 'norm')
            
            results[dist_name][n] = {
                'sample_means': sample_means,
                'standardized': standardized,
                'ks_stat': ks_stat,
                'p_value': p_value,
                'mean': np.mean(sample_means),
                'std': np.std(sample_means, ddof=1)
            }
            
            logger.add_line(f"  n={n:3d}: 样本均值={np.mean(sample_means):.4f}, "
                  f"样本标准差={np.std(sample_means, ddof=1):.4f}, "
                  f"K-S统计量={ks_stat:.4f}, p值={p_value:.4f}")
    
    # 可视化
    create_visualizations(clt, distributions, sample_sizes, results, n_samples)
    
    # 收敛速度分析
    analyze_convergence_rate(clt, distributions)
    
    logger.add_section("实验完成")
    logger.add_line("生成图表:")
    logger.add_line("  - exp2_clt_comparison.png")
    logger.add_line("  - exp2_qq_plots.png")
    logger.add_line("  - exp2_convergence_rate.png")
    logger.save()
    
    print("\n✓ 实验二完成！")
    return results

def create_visualizations(clt, distributions, sample_sizes, results, n_samples):
    """创建可视化图表"""
    
    # 图1：不同分布、不同样本量的CLT效果
    fig1, axes = plt.subplots(len(distributions), len(sample_sizes), 
                              figsize=(18, 12))
    
    for i, (dist_name, params, label) in enumerate(distributions):
        mu, sigma = clt.get_theoretical_params(dist_name, **params)
        
        for j, n in enumerate(sample_sizes):
            ax = axes[i, j]
            standardized = results[dist_name][n]['standardized']
            
            # 绘制直方图
            ax.hist(standardized, bins=50, density=True, alpha=0.7,
                   color='skyblue', edgecolor='black')
            
            # 理论正态分布
            x = np.linspace(-4, 4, 100)
            ax.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')
            
            # 设置标题和标签
            if i == 0:
                ax.set_title(f'n = {n}', fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(label, fontsize=10)
            
            # K-S检验结果
            ks_stat = results[dist_name][n]['ks_stat']
            p_value = results[dist_name][n]['p_value']
            ax.text(0.05, 0.95, f'K-S={ks_stat:.3f}\np={p_value:.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-4, 4)
            
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('实验二：中心极限定理 - 标准化样本均值分布', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig1, 'exp2_clt_comparison.png')
    plt.close()
    
    # 图2：Q-Q图矩阵
    fig2, axes = plt.subplots(len(distributions), len(sample_sizes), 
                              figsize=(18, 12))
    
    for i, (dist_name, params, label) in enumerate(distributions):
        for j, n in enumerate(sample_sizes):
            ax = axes[i, j]
            standardized = results[dist_name][n]['standardized']
            
            stats.probplot(standardized, dist="norm", plot=ax)
            ax.get_lines()[0].set_markerfacecolor('skyblue')
            ax.get_lines()[0].set_markeredgecolor('black')
            ax.get_lines()[0].set_markersize(3)
            ax.get_lines()[1].set_color('red')
            ax.get_lines()[1].set_linewidth(2)
            
            if i == 0:
                ax.set_title(f'n = {n}', fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(label, fontsize=10)
            
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('实验二：Q-Q图 - 正态性检验', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig2, 'exp2_qq_plots.png')
    plt.close()

def analyze_convergence_rate(clt, distributions):
    """分析收敛速度"""
    print("\n步骤：分析收敛速度")
    
    sample_sizes = np.arange(5, 201, 5)
    n_samples = 5000
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (dist_name, params, label) in enumerate(distributions):
        ax = axes[idx]
        mu, sigma = clt.get_theoretical_params(dist_name, **params)
        
        ks_stats = []
        theoretical_std = []
        empirical_std = []
        
        print(f"  分析 {label}...")
        for n in tqdm(sample_sizes, desc=f"  {label}", leave=False):
            sample_means = clt.generate_samples(dist_name, n_samples, n, **params)
            standardized = (sample_means - mu) / (sigma / np.sqrt(n))
            
            ks_stat, _ = stats.kstest(standardized, 'norm')
            ks_stats.append(ks_stat)
            
            theoretical_std.append(sigma / np.sqrt(n))
            empirical_std.append(np.std(sample_means, ddof=1))
        
        # 绘制K-S统计量随样本量的变化
        ax_twin = ax.twinx()
        
        line1 = ax.plot(sample_sizes, ks_stats, 'b-', linewidth=2, 
                       label='K-S统计量', marker='o', markersize=3)
        line2 = ax_twin.plot(sample_sizes, theoretical_std, 'r--', linewidth=2,
                            label='理论标准差')
        line3 = ax_twin.plot(sample_sizes, empirical_std, 'g:', linewidth=2,
                            label='经验标准差')
        
        ax.set_xlabel('样本量 n', fontsize=11)
        ax.set_ylabel('K-S统计量', fontsize=11, color='b')
        ax_twin.set_ylabel('标准差', fontsize=11, color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax_twin.tick_params(axis='y', labelcolor='r')
        
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=9)
        
        # 添加显著性水平线
        ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, 
                  label='α=0.05')
    
    plt.suptitle('实验二：中心极限定理收敛速度分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp2_convergence_rate.png')
    plt.close()

if __name__ == "__main__":
    run_experiment()
