"""
实验五：蒙特卡洛方法
知识点：蒙特卡洛模拟在统计计算和积分估计中的应用

原理：
1. 利用随机抽样估计确定性问题
2. π的估计：通过随机投点法
3. 定积分估计：∫f(x)dx ≈ (b-a)/n * Σf(Xi)
4. 置信区间的蒙特卡洛估计
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from utils import print_section, print_result, save_figure

class MonteCarloSimulation:
    """蒙特卡洛模拟"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def estimate_pi(self, n_samples):
        """
        用投点法估计π
        在单位正方形内随机投点，计算落在1/4圆内的比例
        """
        x = np.random.uniform(0, 1, n_samples)
        y = np.random.uniform(0, 1, n_samples)
        
        # 判断点是否在1/4圆内
        inside_circle = (x**2 + y**2) <= 1
        pi_estimate = 4 * np.sum(inside_circle) / n_samples
        
        return pi_estimate, x, y, inside_circle
    
    def estimate_integral(self, func, a, b, n_samples):
        """
        蒙特卡洛积分估计
        估计 ∫[a,b] f(x)dx
        """
        # 均匀抽样
        x = np.random.uniform(a, b, n_samples)
        y = func(x)
        
        # 积分估计
        integral_estimate = (b - a) * np.mean(y)
        
        # 标准误差
        std_error = (b - a) * np.std(y, ddof=1) / np.sqrt(n_samples)
        
        return integral_estimate, std_error, x, y
    
    def estimate_integral_importance_sampling(self, func, a, b, n_samples):
        """
        重要性采样
        使用重要性采样改进积分估计
        """
        # 使用正态分布作为重要性分布
        mu = (a + b) / 2
        sigma = (b - a) / 6
        
        # 从重要性分布采样
        x = np.random.normal(mu, sigma, n_samples)
        
        # 过滤在[a,b]范围内的点
        mask = (x >= a) & (x <= b)
        x_valid = x[mask]
        
        if len(x_valid) == 0:
            return None, None
        
        # 计算权重
        y = func(x_valid)
        p_x = stats.uniform.pdf(x_valid, a, b-a)  # 目标分布
        q_x = stats.norm.pdf(x_valid, mu, sigma)  # 重要性分布
        
        weights = p_x / q_x
        integral_estimate = np.mean(y * weights)
        
        return integral_estimate, None
    
    def confidence_interval_simulation(self, distribution, params, 
                                      sample_size, n_simulations, alpha=0.05):
        """
        蒙特卡洛方法估计置信区间的覆盖率
        """
        coverage_count = 0
        ci_lower_list = []
        ci_upper_list = []
        
        if distribution == 'normal':
            true_mean = params['mu']
            true_std = params['sigma']
            
            for _ in range(n_simulations):
                # 生成样本
                sample = np.random.normal(true_mean, true_std, sample_size)
                
                # 计算置信区间
                sample_mean = np.mean(sample)
                sample_std = np.std(sample, ddof=1)
                margin = stats.t.ppf(1-alpha/2, sample_size-1) * sample_std / np.sqrt(sample_size)
                
                ci_lower = sample_mean - margin
                ci_upper = sample_mean + margin
                
                ci_lower_list.append(ci_lower)
                ci_upper_list.append(ci_upper)
                
                # 检查是否覆盖真实参数
                if ci_lower <= true_mean <= ci_upper:
                    coverage_count += 1
        
        coverage_rate = coverage_count / n_simulations
        return coverage_rate, ci_lower_list, ci_upper_list

def run_experiment():
    """运行实验"""
    print_section("实验五：蒙特卡洛方法")
    
    mc = MonteCarloSimulation()
    
    # 实验1: 估计π
    print("\n【实验5.1】用蒙特卡洛方法估计π")
    experiment_estimate_pi(mc)
    
    # 实验2: 定积分估计
    print("\n【实验5.2】蒙特卡洛积分估计")
    experiment_integral_estimation(mc)
    
    # 实验3: 置信区间覆盖率
    print("\n【实验5.3】置信区间覆盖率的蒙特卡洛验证")
    experiment_confidence_interval(mc)
    
    # 实验4: 期权定价（Black-Scholes模型）
    print("\n【实验5.4】金融应用：期权定价")
    experiment_option_pricing(mc)
    
    print("\n✓ 实验五完成！")

def experiment_estimate_pi(mc):
    """估计π的实验"""
    
    sample_sizes = [100, 500, 1000, 5000, 10000, 50000]
    estimates = []
    errors = []
    
    print(f"  {'样本量':<10} {'π估计值':<15} {'误差':<15} {'相对误差(%)':<15}")
    
    for n in sample_sizes:
        pi_est, _, _, _ = mc.estimate_pi(n)
        error = abs(pi_est - np.pi)
        rel_error = error / np.pi * 100
        
        estimates.append(pi_est)
        errors.append(error)
        
        print(f"  {n:<10} {pi_est:<15.8f} {error:<15.8f} {rel_error:<15.6f}")
    
    # 收敛性分析
    convergence_analysis_pi(mc)
    
    # 可视化投点过程
    visualize_pi_estimation(mc)

def convergence_analysis_pi(mc):
    """π估计的收敛性分析"""
    
    max_samples = 100000
    check_points = np.logspace(2, 5, 50).astype(int)
    
    estimates = []
    
    print("\n  进行收敛性分析...", end='')
    for n in tqdm(check_points, desc="  ", leave=False):
        pi_est, _, _, _ = mc.estimate_pi(n)
        estimates.append(pi_est)
    print(" ✓")
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: 估计值随样本量的变化
    ax1.plot(check_points, estimates, 'b-', linewidth=2, alpha=0.7, 
            label='π估计值')
    ax1.axhline(y=np.pi, color='red', linestyle='--', linewidth=2, 
               label=f'真实值π={np.pi:.6f}')
    ax1.fill_between(check_points, np.pi-0.05, np.pi+0.05, 
                     alpha=0.2, color='red', label='±0.05误差带')
    ax1.set_xscale('log')
    ax1.set_xlabel('样本量 (对数尺度)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('π估计值', fontsize=12, fontweight='bold')
    ax1.set_title('(a) 估计值收敛过程', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 绝对误差
    errors = np.abs(np.array(estimates) - np.pi)
    ax2.plot(check_points, errors, 'r-', linewidth=2, marker='o', markersize=4)
    
    # 理论收敛率 O(1/√n)
    theoretical_rate = 1 / np.sqrt(check_points) * errors[0] * np.sqrt(check_points[0])
    ax2.plot(check_points, theoretical_rate, 'g--', linewidth=2, 
            label='理论收敛率 O(1/√n)')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('样本量 (对数尺度)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('绝对误差 (对数尺度)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) 误差收敛率', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('实验五：π估计的收敛性分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp5_pi_convergence.png')
    plt.close()

def visualize_pi_estimation(mc):
    """可视化投点过程"""
    
    sample_sizes = [100, 500, 2000, 10000]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, n in enumerate(sample_sizes):
        ax = axes[idx]
        
        pi_est, x, y, inside = mc.estimate_pi(n)
        
        # 绘制点
        ax.scatter(x[inside], y[inside], c='red', s=1, alpha=0.5, label='圆内')
        ax.scatter(x[~inside], y[~inside], c='blue', s=1, alpha=0.5, label='圆外')
        
        # 绘制1/4圆
        theta = np.linspace(0, np.pi/2, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        ax.plot(circle_x, circle_y, 'g-', linewidth=2, label='1/4圆')
        
        # 设置
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'n={n}, π≈{pi_est:.6f}, 误差={abs(pi_est-np.pi):.6f}',
                    fontsize=11, fontweight='bold')
        
        if idx == 0:
            ax.legend(fontsize=9, loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('实验五：蒙特卡洛方法估计π - 投点可视化', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp5_pi_visualization.png')
    plt.close()

def experiment_integral_estimation(mc):
    """积分估计实验"""
    
    # 测试函数
    functions = [
        (lambda x: x**2, 0, 1, 1/3, 'x²'),
        (lambda x: np.sin(x), 0, np.pi, 2, 'sin(x)'),
        (lambda x: np.exp(-x**2), 0, 1, 0.746824, 'e^(-x²)'),
        (lambda x: 1/(1+x**2), 0, 1, np.pi/4, '1/(1+x²)')
    ]
    
    sample_sizes = [100, 500, 1000, 5000, 10000]
    
    results = {}
    
    for func, a, b, true_value, name in functions:
        print(f"\n  函数: {name}, 区间: [{a}, {b}], 真实值: {true_value:.6f}")
        print(f"    {'样本量':<10} {'估计值':<15} {'标准误差':<15} {'误差':<15}")
        
        results[name] = {'estimates': [], 'errors': [], 'std_errors': []}
        
        for n in sample_sizes:
            est, std_err, _, _ = mc.estimate_integral(func, a, b, n)
            error = abs(est - true_value)
            
            results[name]['estimates'].append(est)
            results[name]['errors'].append(error)
            results[name]['std_errors'].append(std_err)
            
            print(f"    {n:<10} {est:<15.8f} {std_err:<15.8f} {error:<15.8f}")
    
    # 可视化
    visualize_integral_estimation(mc, functions, results, sample_sizes)

def visualize_integral_estimation(mc, functions, results, sample_sizes):
    """可视化积分估计"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (func, a, b, true_value, name) in enumerate(functions):
        ax = axes[idx]
        
        # 绘制函数曲线
        x_plot = np.linspace(a, b, 1000)
        y_plot = func(x_plot)
        ax.plot(x_plot, y_plot, 'b-', linewidth=2, label=f'f(x) = {name}')
        ax.fill_between(x_plot, 0, y_plot, alpha=0.3, color='lightblue',
                       label=f'积分={true_value:.4f}')
        
        # 显示蒙特卡洛采样点
        n_demo = 100
        _, _, x_sample, y_sample = mc.estimate_integral(func, a, b, n_demo)
        ax.scatter(x_sample, y_sample, c='red', s=10, alpha=0.5, 
                  label=f'MC样本(n={n_demo})', zorder=5)
        
        ax.set_xlabel('x', fontsize=11, fontweight='bold')
        ax.set_ylabel('f(x)', fontsize=11, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('实验五：蒙特卡洛积分估计 - 函数可视化', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp5_integral_functions.png')
    plt.close()
    
    # 误差收敛图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set1(range(len(functions)))
    
    for idx, (func, a, b, true_value, name) in enumerate(functions):
        ax = axes[idx]
        
        errors = results[name]['errors']
        std_errors = results[name]['std_errors']
        
        ax.plot(sample_sizes, errors, 'o-', linewidth=2, color=colors[idx],
               label='实际误差', markersize=6)
        ax.plot(sample_sizes, std_errors, 's--', linewidth=2, color='red',
               label='标准误差', markersize=6, alpha=0.7)
        
        # 理论收敛率
        theoretical = errors[0] * np.sqrt(sample_sizes[0] / np.array(sample_sizes))
        ax.plot(sample_sizes, theoretical, ':', linewidth=2, color='green',
               label='理论 O(1/√n)')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('样本量', fontsize=11, fontweight='bold')
        ax.set_ylabel('误差', fontsize=11, fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('实验五：积分估计误差收敛分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp5_integral_convergence.png')
    plt.close()

def experiment_confidence_interval(mc):
    """置信区间覆盖率实验"""
    
    true_mu = 5
    true_sigma = 2
    alpha = 0.05
    expected_coverage = 1 - alpha
    
    sample_sizes = [10, 20, 30, 50, 100]
    n_simulations = 1000
    
    print(f"\n  理论覆盖率: {expected_coverage:.4f} ({(1-alpha)*100}%)")
    print(f"  {'样本量':<10} {'实际覆盖率':<15} {'偏差':<15}")
    
    results = {}
    
    for n in sample_sizes:
        print(f"  n={n}...", end='')
        coverage, ci_lower, ci_upper = mc.confidence_interval_simulation(
            'normal', {'mu': true_mu, 'sigma': true_sigma}, n, n_simulations, alpha
        )
        
        results[n] = {
            'coverage': coverage,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bias': coverage - expected_coverage
        }
        
        print(f" 覆盖率={coverage:.4f}, 偏差={coverage - expected_coverage:+.4f}")
    
    # 可视化
    visualize_confidence_interval(results, sample_sizes, true_mu, expected_coverage)

def visualize_confidence_interval(results, sample_sizes, true_mu, expected_coverage):
    """可视化置信区间"""
    
    fig = plt.figure(figsize=(14, 10))
    
    # 子图1-3: 显示部分置信区间
    for plot_idx, n in enumerate([10, 30, 100]):
        ax = plt.subplot(2, 2, plot_idx + 1)
        
        ci_lower = results[n]['ci_lower'][:50]  # 只显示前50个
        ci_upper = results[n]['ci_upper'][:50]
        
        # 判断是否覆盖
        covers = [(l <= true_mu <= u) for l, u in zip(ci_lower, ci_upper)]
        
        for i, (l, u, cover) in enumerate(zip(ci_lower, ci_upper, covers)):
            color = 'green' if cover else 'red'
            ax.plot([i, i], [l, u], color=color, linewidth=1.5, alpha=0.7)
            center = (l + u) / 2
            ax.plot(i, center, 'o', color=color, markersize=3)
        
        ax.axhline(y=true_mu, color='blue', linestyle='--', linewidth=2,
                  label=f'真实μ={true_mu}')
        
        coverage = results[n]['coverage']
        ax.set_xlabel('模拟次数', fontsize=11)
        ax.set_ylabel('置信区间', fontsize=11)
        ax.set_title(f'({chr(97+plot_idx)}) n={n}, 覆盖率={coverage:.3f}',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 子图4: 覆盖率随样本量变化
    ax = plt.subplot(2, 2, 4)
    
    coverages = [results[n]['coverage'] for n in sample_sizes]
    
    ax.plot(sample_sizes, coverages, 'bo-', linewidth=2, markersize=8,
           label='实际覆盖率')
    ax.axhline(y=expected_coverage, color='red', linestyle='--', linewidth=2,
              label=f'理论覆盖率={expected_coverage:.3f}')
    ax.fill_between(sample_sizes, expected_coverage - 0.02, 
                    expected_coverage + 0.02, alpha=0.2, color='red')
    
    ax.set_xlabel('样本量', fontsize=12, fontweight='bold')
    ax.set_ylabel('覆盖率', fontsize=12, fontweight='bold')
    ax.set_title('(d) 覆盖率随样本量的变化', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.0)
    
    plt.suptitle('实验五：95%置信区间的蒙特卡洛验证', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp5_confidence_interval.png')
    plt.close()

def experiment_option_pricing(mc):
    """期权定价实验（Black-Scholes模型）"""
    
    # 参数设置
    S0 = 100  # 初始股价
    K = 105   # 行权价
    T = 1     # 到期时间（年）
    r = 0.05  # 无风险利率
    sigma = 0.2  # 波动率
    
    n_simulations = 10000
    n_steps = 252  # 每年252个交易日
    dt = T / n_steps
    
    print(f"  参数设置:")
    print(f"    初始股价 S0 = {S0}")
    print(f"    行权价格 K = {K}")
    print(f"    到期时间 T = {T}年")
    print(f"    无风险利率 r = {r}")
    print(f"    波动率 σ = {sigma}")
    
    # 蒙特卡洛模拟股价路径
    print(f"\n  进行{n_simulations}次蒙特卡洛模拟...")
    
    stock_paths = np.zeros((n_simulations, n_steps + 1))
    stock_paths[:, 0] = S0
    
    for t in tqdm(range(1, n_steps + 1), desc="  ", leave=False):
        z = np.random.standard_normal(n_simulations)
        stock_paths[:, t] = stock_paths[:, t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )
    
    # 计算期权价值
    ST = stock_paths[:, -1]  # 到期股价
    
    # 看涨期权payoff
    call_payoff = np.maximum(ST - K, 0)
    call_price = np.exp(-r * T) * np.mean(call_payoff)
    call_std = np.exp(-r * T) * np.std(call_payoff, ddof=1) / np.sqrt(n_simulations)
    
    # 看跌期权payoff
    put_payoff = np.maximum(K - ST, 0)
    put_price = np.exp(-r * T) * np.mean(put_payoff)
    put_std = np.exp(-r * T) * np.std(put_payoff, ddof=1) / np.sqrt(n_simulations)
    
    # Black-Scholes解析解
    from scipy.stats import norm
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_bs = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    put_bs = K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    print(f"\n  看涨期权定价:")
    print(f"    蒙特卡洛估计: {call_price:.4f} ± {call_std:.4f}")
    print(f"    Black-Scholes解析解: {call_bs:.4f}")
    print(f"    误差: {abs(call_price - call_bs):.4f}")
    
    print(f"\n  看跌期权定价:")
    print(f"    蒙特卡洛估计: {put_price:.4f} ± {put_std:.4f}")
    print(f"    Black-Scholes解析解: {put_bs:.4f}")
    print(f"    误差: {abs(put_price - put_bs):.4f}")
    
    # 可视化
    visualize_option_pricing(stock_paths, ST, K, call_payoff, put_payoff)

def visualize_option_pricing(stock_paths, ST, K, call_payoff, put_payoff):
    """可视化期权定价"""
    
    fig = plt.figure(figsize=(14, 10))
    
    # 子图1: 股价路径
    ax1 = plt.subplot(2, 2, 1)
    n_plot = 100
    time_steps = np.linspace(0, 1, stock_paths.shape[1])
    
    for i in range(n_plot):
        ax1.plot(time_steps, stock_paths[i], alpha=0.3, linewidth=0.5, color='blue')
    
    ax1.axhline(y=K, color='red', linestyle='--', linewidth=2, label=f'行权价K={K}')
    ax1.set_xlabel('时间(年)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('股价', fontsize=11, fontweight='bold')
    ax1.set_title(f'(a) 股价路径模拟 (显示{n_plot}条)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 到期股价分布
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(ST, bins=50, density=True, alpha=0.7, color='lightblue',
            edgecolor='black', label='到期股价分布')
    ax2.axvline(x=K, color='red', linestyle='--', linewidth=2, label=f'行权价K={K}')
    ax2.axvline(x=np.mean(ST), color='green', linestyle=':', linewidth=2,
               label=f'均值={np.mean(ST):.2f}')
    ax2.set_xlabel('股价', fontsize=11, fontweight='bold')
    ax2.set_ylabel('密度', fontsize=11, fontweight='bold')
    ax2.set_title('(b) 到期股价分布', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 看涨期权收益分布
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(call_payoff, bins=50, density=True, alpha=0.7, color='lightgreen',
            edgecolor='black')
    ax3.set_xlabel('看涨期权收益', fontsize=11, fontweight='bold')
    ax3.set_ylabel('密度', fontsize=11, fontweight='bold')
    ax3.set_title(f'(c) 看涨期权收益分布\n均值={np.mean(call_payoff):.4f}',
                 fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 看跌期权收益分布
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(put_payoff, bins=50, density=True, alpha=0.7, color='lightcoral',
            edgecolor='black')
    ax4.set_xlabel('看跌期权收益', fontsize=11, fontweight='bold')
    ax4.set_ylabel('密度', fontsize=11, fontweight='bold')
    ax4.set_title(f'(d) 看跌期权收益分布\n均值={np.mean(put_payoff):.4f}',
                 fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('实验五：蒙特卡洛方法在期权定价中的应用', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp5_option_pricing.png')
    plt.close()

if __name__ == "__main__":
    run_experiment()
