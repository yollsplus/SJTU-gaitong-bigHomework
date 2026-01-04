"""
实验六：Bootstrap方法及其应用
知识点：Bootstrap重抽样方法 (非参数统计推断)

原理：
1. Bootstrap是一种基于重抽样的统计推断方法
2. 从原始样本中有放回地抽样，生成Bootstrap样本
3. 利用Bootstrap样本估计统计量的分布、置信区间等
4. 不依赖于总体分布的参数假设
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from utils import print_section, print_result, save_figure

class BootstrapMethod:
    """Bootstrap方法实现"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def bootstrap_sample(self, data, n_bootstrap=1000):
        """
        生成Bootstrap样本
        
        参数:
            data: 原始数据
            n_bootstrap: Bootstrap重复次数
        """
        n = len(data)
        bootstrap_samples = np.random.choice(data, size=(n_bootstrap, n), replace=True)
        return bootstrap_samples
    
    def bootstrap_confidence_interval(self, data, statistic_func, 
                                     confidence=0.95, n_bootstrap=1000):
        """
        Bootstrap置信区间（百分位法）
        
        参数:
            data: 原始数据
            statistic_func: 统计量函数
            confidence: 置信水平
            n_bootstrap: Bootstrap重复次数
        """
        bootstrap_samples = self.bootstrap_sample(data, n_bootstrap)
        
        # 计算每个Bootstrap样本的统计量
        bootstrap_statistics = np.array([statistic_func(sample) 
                                        for sample in bootstrap_samples])
        
        # 百分位法
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_statistics, alpha/2 * 100)
        upper = np.percentile(bootstrap_statistics, (1 - alpha/2) * 100)
        
        return bootstrap_statistics, lower, upper
    
    def bootstrap_bias_correction(self, data, statistic_func, n_bootstrap=1000):
        """
        Bootstrap偏差校正
        """
        # 原始统计量
        original_stat = statistic_func(data)
        
        # Bootstrap估计
        bootstrap_samples = self.bootstrap_sample(data, n_bootstrap)
        bootstrap_statistics = np.array([statistic_func(sample) 
                                        for sample in bootstrap_samples])
        
        # 偏差估计
        bias = np.mean(bootstrap_statistics) - original_stat
        
        # 校正后的估计
        corrected_stat = original_stat - bias
        
        return original_stat, bias, corrected_stat, bootstrap_statistics
    
    def bootstrap_standard_error(self, data, statistic_func, n_bootstrap=1000):
        """
        Bootstrap标准误估计
        """
        bootstrap_samples = self.bootstrap_sample(data, n_bootstrap)
        bootstrap_statistics = np.array([statistic_func(sample) 
                                        for sample in bootstrap_samples])
        
        standard_error = np.std(bootstrap_statistics, ddof=1)
        
        return standard_error, bootstrap_statistics

def run_experiment():
    """运行实验"""
    print_section("实验六：Bootstrap方法及其应用")
    
    bs = BootstrapMethod()
    
    # 实验1: Bootstrap置信区间
    print("\n【实验6.1】Bootstrap置信区间估计")
    experiment_confidence_interval(bs)
    
    # 实验2: Bootstrap偏差校正
    print("\n【实验6.2】Bootstrap偏差校正")
    experiment_bias_correction(bs)
    
    # 实验3: Bootstrap标准误估计
    print("\n【实验6.3】Bootstrap标准误估计")
    experiment_standard_error(bs)
    
    # 实验4: Bootstrap假设检验
    print("\n【实验6.4】Bootstrap假设检验")
    experiment_hypothesis_test(bs)
    
    # 实验5: 不同统计量的Bootstrap分析
    print("\n【实验6.5】不同统计量的Bootstrap分析")
    experiment_various_statistics(bs)
    
    print("\n✓ 实验六完成！")

def experiment_confidence_interval(bs):
    """Bootstrap置信区间实验"""
    
    # 生成数据（从偏态分布）
    np.random.seed(42)
    data = np.random.gamma(2, 2, size=50)
    
    n_bootstrap = 10000
    confidence = 0.95
    
    # 对不同统计量计算置信区间
    statistics = [
        ('均值', np.mean),
        ('中位数', np.median),
        ('标准差', np.std),
        ('偏度', lambda x: stats.skew(x))
    ]
    
    results = {}
    
    for name, func in statistics:
        print(f"\n  {name}的Bootstrap置信区间:")
        
        original_stat = func(data)
        bootstrap_stats, lower, upper = bs.bootstrap_confidence_interval(
            data, func, confidence, n_bootstrap
        )
        
        results[name] = {
            'original': original_stat,
            'bootstrap_stats': bootstrap_stats,
            'ci_lower': lower,
            'ci_upper': upper
        }
        
        print(f"    原始估计: {original_stat:.6f}")
        print(f"    {confidence*100}%置信区间: [{lower:.6f}, {upper:.6f}]")
        print(f"    区间宽度: {upper - lower:.6f}")
    
    # 可视化
    visualize_bootstrap_ci(results, confidence, data)

def visualize_bootstrap_ci(results, confidence, data):
    """可视化Bootstrap置信区间"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        bootstrap_stats = result['bootstrap_stats']
        original = result['original']
        ci_lower = result['ci_lower']
        ci_upper = result['ci_upper']
        
        # 绘制Bootstrap分布
        ax.hist(bootstrap_stats, bins=50, density=True, alpha=0.7,
               color='skyblue', edgecolor='black', label='Bootstrap分布')
        
        # 原始估计
        ax.axvline(original, color='blue', linestyle='-', linewidth=2,
                  label=f'原始估计={original:.3f}')
        
        # 置信区间
        ax.axvline(ci_lower, color='red', linestyle='--', linewidth=2,
                  label=f'{confidence*100}% CI')
        ax.axvline(ci_upper, color='red', linestyle='--', linewidth=2)
        
        # 填充置信区间
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='red')
        
        # 添加文本信息
        ax.text(0.05, 0.95, f'CI: [{ci_lower:.3f}, {ci_upper:.3f}]',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               fontsize=9)
        
        ax.set_xlabel(f'{name}', fontsize=11, fontweight='bold')
        ax.set_ylabel('密度', fontsize=11, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {name}的Bootstrap分布', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('实验六：Bootstrap置信区间估计', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp6_bootstrap_ci.png')
    plt.close()
    
    # 额外可视化：原始数据分布
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(data, bins=20, density=True, alpha=0.7, color='lightgreen',
           edgecolor='black', label='原始数据')
    
    # 拟合的Gamma分布
    params = stats.gamma.fit(data)
    x = np.linspace(data.min(), data.max(), 100)
    ax.plot(x, stats.gamma.pdf(x, *params), 'r-', linewidth=2,
           label=f'拟合Gamma分布')
    
    # 核密度估计
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    ax.plot(x, kde(x), 'b--', linewidth=2, label='核密度估计')
    
    ax.set_xlabel('值', fontsize=12, fontweight='bold')
    ax.set_ylabel('密度', fontsize=12, fontweight='bold')
    ax.set_title(f'原始数据分布 (n={len(data)})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'exp6_original_data.png')
    plt.close()

def experiment_bias_correction(bs):
    """Bootstrap偏差校正实验"""
    
    # 生成数据
    np.random.seed(42)
    true_mean = 10
    true_std = 2
    data = np.random.normal(true_mean, true_std, 30)
    
    n_bootstrap = 5000
    
    # 测试不同统计量的偏差
    statistics = [
        ('标准差', lambda x: np.std(x, ddof=0), true_std),
        ('方差', lambda x: np.var(x, ddof=0), true_std**2),
        ('变异系数', lambda x: np.std(x, ddof=0) / np.mean(x), true_std / true_mean),
        ('峰度', lambda x: stats.kurtosis(x), 0)  # 正态分布峰度为0
    ]
    
    print(f"\n  样本量: {len(data)}")
    print(f"  {'统计量':<12} {'原始估计':<12} {'偏差':<12} {'校正后':<12} {'真实值':<12}")
    
    results = {}
    
    for name, func, true_value in statistics:
        original, bias, corrected, bootstrap_stats = bs.bootstrap_bias_correction(
            data, func, n_bootstrap
        )
        
        results[name] = {
            'original': original,
            'bias': bias,
            'corrected': corrected,
            'bootstrap_stats': bootstrap_stats,
            'true_value': true_value
        }
        
        print(f"  {name:<12} {original:<12.6f} {bias:<12.6f} "
              f"{corrected:<12.6f} {true_value:<12.6f}")
    
    # 可视化
    visualize_bias_correction(results)

def visualize_bias_correction(results):
    """可视化偏差校正"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        bootstrap_stats = result['bootstrap_stats']
        original = result['original']
        corrected = result['corrected']
        true_value = result['true_value']
        bias = result['bias']
        
        # Bootstrap分布
        ax.hist(bootstrap_stats, bins=50, density=True, alpha=0.7,
               color='lightblue', edgecolor='black', label='Bootstrap分布')
        
        # 各种估计
        ax.axvline(true_value, color='green', linestyle='-', linewidth=2.5,
                  label=f'真实值={true_value:.3f}', alpha=0.8)
        ax.axvline(original, color='blue', linestyle='--', linewidth=2,
                  label=f'原始估计={original:.3f}')
        ax.axvline(corrected, color='red', linestyle=':', linewidth=2,
                  label=f'校正后={corrected:.3f}')
        
        # 显示偏差
        if abs(bias) > 1e-6:
            y_pos = ax.get_ylim()[1] * 0.5
            ax.annotate('', xy=(original, y_pos), xytext=(corrected, y_pos),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
            ax.text((original + corrected)/2, y_pos*1.1, f'bias={bias:.4f}',
                   ha='center', fontsize=9, color='red')
        
        ax.set_xlabel(name, fontsize=11, fontweight='bold')
        ax.set_ylabel('密度', fontsize=11, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {name}的偏差校正', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('实验六：Bootstrap偏差校正', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp6_bias_correction.png')
    plt.close()

def experiment_standard_error(bs):
    """Bootstrap标准误估计实验"""
    
    # 不同样本量下的标准误
    sample_sizes = [10, 20, 30, 50, 100]
    true_mean = 5
    true_std = 2
    n_bootstrap = 5000
    
    results = {}
    
    print(f"\n  {'样本量':<10} {'Bootstrap SE':<15} {'理论SE':<15} {'相对误差(%)':<15}")
    
    for n in sample_sizes:
        np.random.seed(42)
        data = np.random.normal(true_mean, true_std, n)
        
        # Bootstrap标准误
        bs_se, bootstrap_stats = bs.bootstrap_standard_error(
            data, np.mean, n_bootstrap
        )
        
        # 理论标准误
        theoretical_se = true_std / np.sqrt(n)
        
        # 相对误差
        rel_error = abs(bs_se - theoretical_se) / theoretical_se * 100
        
        results[n] = {
            'bs_se': bs_se,
            'theoretical_se': theoretical_se,
            'bootstrap_stats': bootstrap_stats
        }
        
        print(f"  {n:<10} {bs_se:<15.6f} {theoretical_se:<15.6f} {rel_error:<15.4f}")
    
    # 可视化
    visualize_standard_error(results, sample_sizes)

def visualize_standard_error(results, sample_sizes):
    """可视化标准误"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图1: 标准误对比
    bs_se_list = [results[n]['bs_se'] for n in sample_sizes]
    theo_se_list = [results[n]['theoretical_se'] for n in sample_sizes]
    
    x = np.arange(len(sample_sizes))
    width = 0.35
    
    ax1.bar(x - width/2, bs_se_list, width, label='Bootstrap SE',
           alpha=0.8, color='skyblue', edgecolor='black')
    ax1.bar(x + width/2, theo_se_list, width, label='理论SE',
           alpha=0.8, color='lightcoral', edgecolor='black')
    
    ax1.set_xlabel('样本量', fontsize=12, fontweight='bold')
    ax1.set_ylabel('标准误', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Bootstrap vs 理论标准误', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'n={n}' for n in sample_sizes])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 子图2: 标准误随样本量的变化（对数尺度）
    ax2.plot(sample_sizes, bs_se_list, 'o-', linewidth=2, markersize=8,
            label='Bootstrap SE', color='blue')
    ax2.plot(sample_sizes, theo_se_list, 's--', linewidth=2, markersize=8,
            label='理论SE = σ/√n', color='red')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('样本量 (对数尺度)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('标准误 (对数尺度)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) 标准误收敛分析', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('实验六：Bootstrap标准误估计', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp6_standard_error.png')
    plt.close()

def experiment_hypothesis_test(bs):
    """Bootstrap假设检验实验"""
    
    # 生成两组数据
    np.random.seed(42)
    group1 = np.random.normal(10, 2, 30)
    group2 = np.random.normal(11, 2, 30)
    
    # 原始均值差
    observed_diff = np.mean(group1) - np.mean(group2)
    
    # Bootstrap假设检验（排列检验）
    n_bootstrap = 10000
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    bootstrap_diffs = []
    
    print(f"  原始均值差: {observed_diff:.6f}")
    print(f"  进行{n_bootstrap}次Bootstrap排列检验...", end='')
    
    for _ in tqdm(range(n_bootstrap), desc="  ", leave=False):
        # 随机排列
        np.random.shuffle(combined)
        boot_group1 = combined[:n1]
        boot_group2 = combined[n1:]
        
        boot_diff = np.mean(boot_group1) - np.mean(boot_group2)
        bootstrap_diffs.append(boot_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # 计算p值
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
    
    print(f" ✓")
    print(f"  Bootstrap p值: {p_value:.6f}")
    
    # 传统t检验对比
    t_stat, t_pvalue = stats.ttest_ind(group1, group2)
    print(f"  传统t检验p值: {t_pvalue:.6f}")
    
    # 可视化
    visualize_hypothesis_test(group1, group2, bootstrap_diffs, 
                             observed_diff, p_value, t_pvalue)

def visualize_hypothesis_test(group1, group2, bootstrap_diffs, 
                              observed_diff, p_value, t_pvalue):
    """可视化Bootstrap假设检验"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: 两组数据的分布
    ax1 = axes[0, 0]
    ax1.hist(group1, bins=15, alpha=0.6, label='组1', density=True,
            color='skyblue', edgecolor='black')
    ax1.hist(group2, bins=15, alpha=0.6, label='组2', density=True,
            color='lightcoral', edgecolor='black')
    ax1.axvline(np.mean(group1), color='blue', linestyle='--', linewidth=2,
               label=f'组1均值={np.mean(group1):.2f}')
    ax1.axvline(np.mean(group2), color='red', linestyle='--', linewidth=2,
               label=f'组2均值={np.mean(group2):.2f}')
    ax1.set_xlabel('值', fontsize=11, fontweight='bold')
    ax1.set_ylabel('密度', fontsize=11, fontweight='bold')
    ax1.set_title('(a) 两组数据分布', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: Bootstrap均值差分布
    ax2 = axes[0, 1]
    ax2.hist(bootstrap_diffs, bins=50, density=True, alpha=0.7,
            color='lightgreen', edgecolor='black', label='Bootstrap分布')
    ax2.axvline(0, color='green', linestyle='-', linewidth=2,
               label='零假设(diff=0)')
    ax2.axvline(observed_diff, color='red', linestyle='--', linewidth=2,
               label=f'观测差={observed_diff:.3f}')
    ax2.axvline(-observed_diff, color='red', linestyle='--', linewidth=2)
    
    # 填充拒绝域
    critical_value = np.abs(observed_diff)
    mask_reject = np.abs(bootstrap_diffs) >= critical_value
    if np.any(mask_reject):
        ax2.axvspan(-5, -critical_value, alpha=0.2, color='red')
        ax2.axvspan(critical_value, 5, alpha=0.2, color='red')
    
    ax2.text(0.05, 0.95, f'Bootstrap p={p_value:.4f}\nt-test p={t_pvalue:.4f}',
            transform=ax2.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontsize=9)
    
    ax2.set_xlabel('均值差', fontsize=11, fontweight='bold')
    ax2.set_ylabel('密度', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Bootstrap排列检验分布', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 箱线图对比
    ax3 = axes[1, 0]
    bp = ax3.boxplot([group1, group2], labels=['组1', '组2'],
                     patch_artist=True, widths=0.6)
    colors = ['skyblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('值', fontsize=11, fontweight='bold')
    ax3.set_title('(c) 箱线图对比', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 子图4: QQ图
    ax4 = axes[1, 1]
    stats.probplot(group1 - np.mean(group1), dist="norm", plot=ax4)
    ax4.get_lines()[0].set_markerfacecolor('skyblue')
    ax4.get_lines()[0].set_label('组1')
    
    # 添加组2的QQ图
    pp = stats.probplot(group2 - np.mean(group2), dist="norm")
    ax4.plot(pp[0][0], pp[0][1], 'ro', markersize=4, alpha=0.6, label='组2')
    
    ax4.set_title('(d) 正态性检验(中心化)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('实验六：Bootstrap假设检验', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp6_hypothesis_test.png')
    plt.close()

def experiment_various_statistics(bs):
    """不同统计量的Bootstrap分析"""
    
    # 生成混合分布数据
    np.random.seed(42)
    data1 = np.random.normal(5, 1, 40)
    data2 = np.random.normal(10, 1.5, 40)
    data = np.concatenate([data1, data2])
    
    n_bootstrap = 5000
    
    # 分析各种统计量
    statistics = [
        ('均值', np.mean),
        ('中位数', np.median),
        ('10%分位数', lambda x: np.percentile(x, 10)),
        ('90%分位数', lambda x: np.percentile(x, 90)),
        ('四分位距', lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
        ('偏度', lambda x: stats.skew(x))
    ]
    
    results = {}
    
    print(f"\n  统计量分析 (n={len(data)}):")
    print(f"  {'统计量':<15} {'估计值':<12} {'Bootstrap SE':<15} {'95% CI':<30}")
    
    for name, func in statistics:
        original = func(data)
        bs_se, bootstrap_stats = bs.bootstrap_standard_error(data, func, n_bootstrap)
        _, ci_lower, ci_upper = bs.bootstrap_confidence_interval(data, func, 0.95, n_bootstrap)
        
        results[name] = {
            'original': original,
            'se': bs_se,
            'ci': (ci_lower, ci_upper),
            'bootstrap_stats': bootstrap_stats
        }
        
        print(f"  {name:<15} {original:<12.4f} {bs_se:<15.6f} "
              f"[{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # 可视化
    visualize_various_statistics(data, results)

def visualize_various_statistics(data, results):
    """可视化不同统计量"""
    
    fig = plt.figure(figsize=(15, 10))
    
    # 原始数据分布
    ax_data = plt.subplot(3, 3, 1)
    ax_data.hist(data, bins=30, density=True, alpha=0.7, color='lightblue',
                edgecolor='black', label='数据分布')
    
    # 添加核密度估计
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    ax_data.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    ax_data.set_xlabel('值', fontsize=10, fontweight='bold')
    ax_data.set_ylabel('密度', fontsize=10, fontweight='bold')
    ax_data.set_title('(a) 原始数据 (混合分布)', fontsize=10, fontweight='bold')
    ax_data.legend(fontsize=8)
    ax_data.grid(True, alpha=0.3)
    
    # 各统计量的Bootstrap分布
    plot_idx = 2
    for name, result in results.items():
        if plot_idx > 9:
            break
        
        ax = plt.subplot(3, 3, plot_idx)
        
        bootstrap_stats = result['bootstrap_stats']
        original = result['original']
        ci_lower, ci_upper = result['ci']
        
        ax.hist(bootstrap_stats, bins=40, density=True, alpha=0.7,
               color='lightgreen', edgecolor='black')
        ax.axvline(original, color='blue', linestyle='-', linewidth=2,
                  label=f'估计值={original:.3f}')
        ax.axvline(ci_lower, color='red', linestyle='--', linewidth=1.5)
        ax.axvline(ci_upper, color='red', linestyle='--', linewidth=1.5,
                  label='95% CI')
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='red')
        
        ax.set_xlabel(name, fontsize=9, fontweight='bold')
        ax.set_ylabel('密度', fontsize=9, fontweight='bold')
        ax.set_title(f'({chr(97+plot_idx-1)}) {name}', 
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    plt.suptitle('实验六：不同统计量的Bootstrap分析', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp6_various_statistics.png')
    plt.close()

if __name__ == "__main__":
    run_experiment()
