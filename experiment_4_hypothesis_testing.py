"""
实验四：假设检验的功效分析
知识点：假设检验（t检验、卡方检验、K-S检验）及检验功效

原理：
1. 第一类错误α：拒绝真的原假设的概率（显著性水平）
2. 第二类错误β：接受假的原假设的概率
3. 检验功效Power = 1 - β：拒绝假的原假设的概率
4. 分析不同检验方法的功效随效应量、样本量的变化
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from utils import print_section, print_result, save_figure

class HypothesisTestingPower:
    """假设检验功效分析"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def one_sample_t_test_power(self, true_mean, null_mean, std, 
                                sample_size, alpha=0.05, n_simulations=1000):
        """
        单样本t检验的功效
        H0: μ = null_mean  vs  H1: μ ≠ null_mean
        """
        rejections = 0
        p_values = []
        
        for _ in range(n_simulations):
            # 从真实分布抽样
            sample = np.random.normal(true_mean, std, sample_size)
            
            # 执行t检验
            t_stat, p_value = stats.ttest_1samp(sample, null_mean)
            p_values.append(p_value)
            
            if p_value < alpha:
                rejections += 1
        
        power = rejections / n_simulations
        return power, p_values
    
    def two_sample_t_test_power(self, mean1, mean2, std1, std2,
                                sample_size, alpha=0.05, n_simulations=1000):
        """
        双样本t检验的功效
        H0: μ1 = μ2  vs  H1: μ1 ≠ μ2
        """
        rejections = 0
        
        for _ in range(n_simulations):
            sample1 = np.random.normal(mean1, std1, sample_size)
            sample2 = np.random.normal(mean2, std2, sample_size)
            
            t_stat, p_value = stats.ttest_ind(sample1, sample2)
            
            if p_value < alpha:
                rejections += 1
        
        power = rejections / n_simulations
        return power
    
    def ks_test_power(self, dist1_name, dist1_params, dist2_name, dist2_params,
                     sample_size, alpha=0.05, n_simulations=1000):
        """
        K-S检验的功效（检验分布的差异）
        """
        rejections = 0
        
        for _ in range(n_simulations):
            # 从第一个分布抽样
            if dist1_name == 'normal':
                sample = np.random.normal(dist1_params['mu'], dist1_params['sigma'], sample_size)
            elif dist1_name == 'exponential':
                sample = np.random.exponential(dist1_params['scale'], sample_size)
            
            # 与第二个分布比较
            if dist2_name == 'normal':
                ks_stat, p_value = stats.kstest(sample, lambda x: stats.norm.cdf(x, 
                                                dist2_params['mu'], dist2_params['sigma']))
            elif dist2_name == 'exponential':
                ks_stat, p_value = stats.kstest(sample, 'expon', 
                                                args=(0, dist2_params['scale']))
            
            if p_value < alpha:
                rejections += 1
        
        power = rejections / n_simulations
        return power
    
    def chi2_test_power(self, expected_probs, actual_probs, sample_size,
                       alpha=0.05, n_simulations=1000):
        """
        卡方检验的功效（拟合优度检验）
        """
        rejections = 0
        k = len(expected_probs)  # 类别数
        
        for _ in range(n_simulations):
            # 从实际分布抽样
            sample = np.random.choice(k, size=sample_size, p=actual_probs)
            observed = np.bincount(sample, minlength=k)
            expected = np.array(expected_probs) * sample_size
            
            # 卡方检验
            chi2_stat, p_value = stats.chisquare(observed, expected)
            
            if p_value < alpha:
                rejections += 1
        
        power = rejections / n_simulations
        return power

def run_experiment():
    """运行实验"""
    print_section("实验四：假设检验的功效分析")
    
    tester = HypothesisTestingPower()
    
    # 实验1: 单样本t检验功效分析
    print("\n【实验4.1】单样本t检验功效分析")
    experiment_one_sample_t_test(tester)
    
    # 实验2: 双样本t检验功效分析
    print("\n【实验4.2】双样本t检验功效分析")
    experiment_two_sample_t_test(tester)
    
    # 实验3: K-S检验功效分析
    print("\n【实验4.3】K-S检验功效分析")
    experiment_ks_test(tester)
    
    # 实验4: 卡方检验功效分析
    print("\n【实验4.4】卡方拟合优度检验功效分析")
    experiment_chi2_test(tester)
    
    # 实验5: Type I和Type II错误可视化
    print("\n【实验4.5】第一类和第二类错误可视化")
    visualize_type_errors(tester)
    
    print("\n✓ 实验四完成！")

def experiment_one_sample_t_test(tester):
    """单样本t检验功效分析"""
    
    null_mean = 0
    std = 1
    sample_sizes = [10, 20, 30, 50, 100]
    effect_sizes = np.linspace(0, 1.5, 30)  # Cohen's d
    alpha = 0.05
    
    results = {}
    
    for n in sample_sizes:
        powers = []
        print(f"  计算样本量n={n}的功效曲线...", end='')
        
        for effect_size in effect_sizes:
            true_mean = null_mean + effect_size * std
            power, _ = tester.one_sample_t_test_power(true_mean, null_mean, std, n, alpha)
            powers.append(power)
        
        results[n] = powers
        print(f" ✓")
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sample_sizes)))
    for idx, n in enumerate(sample_sizes):
        ax.plot(effect_sizes, results[n], linewidth=2.5, label=f'n={n}',
               marker='o', markersize=4, color=colors[idx])
    
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, 
              label='功效=0.8 (推荐水平)', alpha=0.7)
    ax.axhline(y=alpha, color='orange', linestyle=':', linewidth=1.5,
              label=f'α={alpha}', alpha=0.7)
    
    ax.set_xlabel('效应量 (Cohen\'s d)', fontsize=12, fontweight='bold')
    ax.set_ylabel('检验功效 (1-β)', fontsize=12, fontweight='bold')
    ax.set_title('单样本t检验：功效随效应量和样本量的变化', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    save_figure(fig, 'exp4_one_sample_t_power.png')
    plt.close()
    
    # 打印结果表
    print("\n  功效分析结果 (效应量=0.5时):")
    print(f"    {'样本量':<10} {'功效':<10} {'所需样本量(功效≥0.8)':<25}")
    for n in sample_sizes:
        idx = np.argmin(np.abs(effect_sizes - 0.5))
        power = results[n][idx]
        # 找到达到0.8功效所需的最小样本量
        if power >= 0.8:
            required_n = n
        else:
            required_n = "需要更大样本"
        print(f"    {n:<10} {power:<10.4f} {str(required_n):<25}")

def experiment_two_sample_t_test(tester):
    """双样本t检验功效分析"""
    
    mean1 = 0
    std = 1
    sample_sizes = [10, 20, 30, 50, 100]
    mean_differences = np.linspace(0, 2, 30)
    alpha = 0.05
    
    results = {}
    
    for n in sample_sizes:
        powers = []
        print(f"  计算样本量n={n}的功效曲线...", end='')
        
        for mean_diff in mean_differences:
            mean2 = mean1 + mean_diff
            power = tester.two_sample_t_test_power(mean1, mean2, std, std, n, alpha)
            powers.append(power)
        
        results[n] = powers
        print(f" ✓")
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子图1: 功效曲线
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(sample_sizes)))
    for idx, n in enumerate(sample_sizes):
        ax1.plot(mean_differences, results[n], linewidth=2.5, label=f'n={n}',
                marker='s', markersize=4, color=colors[idx])
    
    ax1.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5,
               label='功效=0.8', alpha=0.7)
    ax1.set_xlabel('均值差异 (|μ₁ - μ₂|)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('检验功效', fontsize=12, fontweight='bold')
    ax1.set_title('(a) 功效曲线', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 热力图
    power_matrix = np.array([results[n] for n in sample_sizes])
    im = ax2.imshow(power_matrix, aspect='auto', cmap='RdYlGn', 
                    vmin=0, vmax=1, origin='lower')
    
    ax2.set_xlabel('均值差异指数', fontsize=12, fontweight='bold')
    ax2.set_ylabel('样本量', fontsize=12, fontweight='bold')
    ax2.set_title('(b) 功效热力图', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(len(sample_sizes)))
    ax2.set_yticklabels([f'n={n}' for n in sample_sizes])
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('检验功效', fontsize=11)
    
    plt.suptitle('双样本t检验：功效分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp4_two_sample_t_power.png')
    plt.close()

def experiment_ks_test(tester):
    """K-S检验功效分析"""
    
    sample_sizes = [30, 50, 100, 200, 500]
    
    # 场景1: 正态分布均值偏移
    print("  场景1: 检测正态分布均值偏移")
    mean_shifts = np.linspace(0, 2, 25)
    results_mean_shift = {}
    
    for n in sample_sizes:
        powers = []
        print(f"    n={n}...", end='')
        
        for shift in mean_shifts:
            power = tester.ks_test_power('normal', {'mu': shift, 'sigma': 1},
                                        'normal', {'mu': 0, 'sigma': 1},
                                        n, n_simulations=500)
            powers.append(power)
        
        results_mean_shift[n] = powers
        print(f" ✓")
    
    # 场景2: 检测分布类型差异（正态 vs 指数）
    print("  场景2: 检测分布类型差异")
    scale_values = np.linspace(0.5, 3, 25)
    results_dist_diff = {}
    
    for n in sample_sizes:
        powers = []
        print(f"    n={n}...", end='')
        
        for scale in scale_values:
            power = tester.ks_test_power('exponential', {'scale': scale},
                                        'normal', {'mu': scale, 'sigma': scale/2},
                                        n, n_simulations=500)
            powers.append(power)
        
        results_dist_diff[n] = powers
        print(f" ✓")
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 场景1
    colors = plt.cm.coolwarm(np.linspace(0, 0.9, len(sample_sizes)))
    for idx, n in enumerate(sample_sizes):
        ax1.plot(mean_shifts, results_mean_shift[n], linewidth=2.5, 
                label=f'n={n}', marker='o', markersize=4, color=colors[idx])
    
    ax1.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('均值偏移量', fontsize=12, fontweight='bold')
    ax1.set_ylabel('K-S检验功效', fontsize=12, fontweight='bold')
    ax1.set_title('(a) 检测正态分布均值偏移', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 场景2
    for idx, n in enumerate(sample_sizes):
        ax2.plot(scale_values, results_dist_diff[n], linewidth=2.5,
                label=f'n={n}', marker='s', markersize=4, color=colors[idx])
    
    ax2.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('指数分布scale参数', fontsize=12, fontweight='bold')
    ax2.set_ylabel('K-S检验功效', fontsize=12, fontweight='bold')
    ax2.set_title('(b) 检测分布类型差异', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('K-S检验：功效分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp4_ks_test_power.png')
    plt.close()

def experiment_chi2_test(tester):
    """卡方检验功效分析"""
    
    # 检验掷骰子是否公平
    expected_probs = [1/6] * 6  # 公平骰子
    
    sample_sizes = [30, 60, 100, 200, 500]
    
    # 不同程度的偏离
    bias_levels = np.linspace(0, 0.15, 30)
    results = {}
    
    for n in sample_sizes:
        powers = []
        print(f"  样本量n={n}...", end='')
        
        for bias in bias_levels:
            # 不公平骰子：第一个面概率增加，其他面平均减少
            actual_probs = np.array([1/6 + bias] + [1/6 - bias/5]*5)
            
            power = tester.chi2_test_power(expected_probs, actual_probs, n,
                                          n_simulations=500)
            powers.append(power)
        
        results[n] = powers
        print(f" ✓")
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子图1: 功效曲线
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(sample_sizes)))
    for idx, n in enumerate(sample_sizes):
        ax1.plot(bias_levels, results[n], linewidth=2.5, label=f'n={n}',
                marker='o', markersize=4, color=colors[idx])
    
    ax1.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='功效=0.8')
    ax1.set_xlabel('偏离程度', fontsize=12, fontweight='bold')
    ax1.set_ylabel('卡方检验功效', fontsize=12, fontweight='bold')
    ax1.set_title('(a) 功效随偏离程度的变化', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 示例不公平骰子的概率分布
    bias_examples = [0, 0.05, 0.10, 0.15]
    x = np.arange(1, 7)
    width = 0.2
    
    for i, bias in enumerate(bias_examples):
        actual_probs = np.array([1/6 + bias] + [1/6 - bias/5]*5)
        offset = (i - 1.5) * width
        ax2.bar(x + offset, actual_probs, width, label=f'偏离={bias:.2f}',
               alpha=0.8)
    
    ax2.axhline(y=1/6, color='red', linestyle='--', linewidth=1.5,
               label='公平骰子', alpha=0.7)
    ax2.set_xlabel('骰子面', fontsize=12, fontweight='bold')
    ax2.set_ylabel('概率', fontsize=12, fontweight='bold')
    ax2.set_title('(b) 不同偏离程度的概率分布', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('卡方拟合优度检验：功效分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp4_chi2_test_power.png')
    plt.close()

def visualize_type_errors(tester):
    """可视化第一类和第二类错误"""
    
    # 设置
    null_mean = 0
    std = 1
    alpha = 0.05
    n = 30
    
    # 在原假设下的分布
    x = np.linspace(-4, 4, 1000)
    null_dist = stats.norm.pdf(x, null_mean, std/np.sqrt(n))
    
    # 临界值
    critical_value = stats.norm.ppf(1 - alpha/2, null_mean, std/np.sqrt(n))
    
    # 不同真实均值下的功效和第二类错误
    true_means = [0.3, 0.6, 0.9]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: 第一类错误
    ax = axes[0, 0]
    ax.plot(x, null_dist, 'b-', linewidth=2, label='H₀: μ=0 (真实)')
    ax.fill_between(x[x >= critical_value], 0, null_dist[x >= critical_value],
                    alpha=0.3, color='red', label=f'拒绝域 (α={alpha})')
    ax.fill_between(x[x <= -critical_value], 0, null_dist[x <= -critical_value],
                    alpha=0.3, color='red')
    ax.axvline(critical_value, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(-critical_value, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('样本均值', fontsize=11)
    ax.set_ylabel('密度', fontsize=11)
    ax.set_title('(a) 第一类错误 (Type I Error)\n拒绝真的H₀', 
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 子图2-4: 第二类错误和功效
    for idx, true_mean in enumerate(true_means):
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        ax = axes[row, col]
        
        # 备择假设下的分布
        alt_dist = stats.norm.pdf(x, true_mean, std/np.sqrt(n))
        
        # 计算功效和β
        power, _ = tester.one_sample_t_test_power(true_mean, null_mean, std, n, alpha)
        beta = 1 - power
        
        # 绘制分布
        ax.plot(x, null_dist, 'b-', linewidth=2, label='H₀: μ=0', alpha=0.5)
        ax.plot(x, alt_dist, 'g-', linewidth=2, label=f'H₁: μ={true_mean}')
        
        # 填充区域
        mask_accept = (x >= -critical_value) & (x <= critical_value)
        ax.fill_between(x[mask_accept], 0, alt_dist[mask_accept],
                       alpha=0.3, color='orange', 
                       label=f'β (Type II) = {beta:.3f}')
        
        mask_reject_right = x >= critical_value
        mask_reject_left = x <= -critical_value
        ax.fill_between(x[mask_reject_right], 0, alt_dist[mask_reject_right],
                       alpha=0.3, color='green')
        ax.fill_between(x[mask_reject_left], 0, alt_dist[mask_reject_left],
                       alpha=0.3, color='green',
                       label=f'功效 (1-β) = {power:.3f}')
        
        ax.axvline(critical_value, color='red', linestyle='--', linewidth=1.5)
        ax.axvline(-critical_value, color='red', linestyle='--', linewidth=1.5)
        
        ax.set_xlabel('样本均值', fontsize=11)
        ax.set_ylabel('密度', fontsize=11)
        ax.set_title(f'({chr(98+idx)}) 真实μ={true_mean}\n功效={power:.3f}, β={beta:.3f}',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'实验四：假设检验的两类错误与功效 (n={n}, α={alpha})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp4_type_errors.png')
    plt.close()

if __name__ == "__main__":
    run_experiment()
