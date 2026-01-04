"""
实验三：参数估计方法对比研究
知识点：参数估计（极大似然估计MLE vs 矩估计MME）

原理：
1. 矩估计(Method of Moments): 用样本矩估计总体矩
2. 极大似然估计(MLE): 最大化似然函数 L(θ|x) = ∏f(xi|θ)
3. 对比两种方法的估计效果、偏差和方差
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from tqdm import tqdm
from utils import print_section, print_result, save_figure, ResultLogger

class ParameterEstimation:
    """参数估计实现"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    # ====== 正态分布 ======
    def normal_mle(self, data):
        """正态分布的MLE"""
        mu_hat = np.mean(data)
        sigma_hat = np.sqrt(np.mean((data - mu_hat)**2))
        return mu_hat, sigma_hat
    
    def normal_mme(self, data):
        """正态分布的矩估计"""
        mu_hat = np.mean(data)
        sigma_hat = np.sqrt(np.var(data, ddof=0))
        return mu_hat, sigma_hat
    
    # ====== 指数分布 ======
    def exponential_mle(self, data):
        """指数分布的MLE: λ = 1/x̄"""
        lambda_hat = 1 / np.mean(data)
        return lambda_hat
    
    def exponential_mme(self, data):
        """指数分布的矩估计: λ = 1/x̄"""
        lambda_hat = 1 / np.mean(data)
        return lambda_hat
    
    # ====== 伽玛分布 ======
    def gamma_mle(self, data):
        """伽玛分布的MLE (数值优化)"""
        def neg_log_likelihood(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return np.inf
            return -np.sum(stats.gamma.logpdf(data, alpha, scale=1/beta))
        
        # 使用矩估计作为初始值
        alpha_init, beta_init = self.gamma_mme(data)
        result = minimize(neg_log_likelihood, [alpha_init, beta_init], 
                         method='Nelder-Mead')
        return result.x[0], result.x[1]
    
    def gamma_mme(self, data):
        """伽玛分布的矩估计"""
        mean = np.mean(data)
        var = np.var(data, ddof=1)
        alpha_hat = mean**2 / var
        beta_hat = mean / var
        return alpha_hat, beta_hat
    
    # ====== 贝塔分布 ======
    def beta_mle(self, data):
        """贝塔分布的MLE (数值优化)"""
        def neg_log_likelihood(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return np.inf
            return -np.sum(stats.beta.logpdf(data, alpha, beta))
        
        alpha_init, beta_init = self.beta_mme(data)
        result = minimize(neg_log_likelihood, [alpha_init, beta_init],
                         method='Nelder-Mead')
        return result.x[0], result.x[1]
    
    def beta_mme(self, data):
        """贝塔分布的矩估计"""
        mean = np.mean(data)
        var = np.var(data, ddof=1)
        
        alpha_hat = mean * (mean * (1 - mean) / var - 1)
        beta_hat = (1 - mean) * (mean * (1 - mean) / var - 1)
        return alpha_hat, beta_hat

def run_experiment():
    """运行实验"""
    print_section("实验三：参数估计方法对比研究")
    
    logger = ResultLogger("实验三_参数估计")
    
    estimator = ParameterEstimation()
    
    # 实验1: 正态分布参数估计
    logger.add_section("【实验3.1】正态分布 N(μ=5, σ²=4)")
    experiment_normal(estimator, logger)
    
    # 实验2: 指数分布参数估计
    logger.add_section("【实验3.2】指数分布 Exp(λ=0.5)")
    experiment_exponential(estimator, logger)
    
    # 实验3: 伽玛分布参数估计
    logger.add_section("【实验3.3】伽玛分布 Gamma(α=2, β=0.5)")
    experiment_gamma(estimator, logger)
    
    # 实验4: 贝塔分布参数估计
    logger.add_section("【实验3.4】贝塔分布 Beta(α=2, β=5)")
    experiment_beta(estimator, logger)
    
    # 综合对比
    comprehensive_comparison(estimator)
    
    logger.add_section("实验完成")
    logger.add_line("生成图表:")
    logger.add_line("  - exp3_normal_estimation.png")
    logger.add_line("  - exp3_exponential_estimation.png")
    logger.add_line("  - exp3_mse_comparison.png")
    logger.save()
    
    print("\n✓ 实验三完成！")

def experiment_normal(estimator, logger=None):
    """正态分布参数估计实验"""
    true_mu, true_sigma = 5, 2
    sample_sizes = [10, 30, 50, 100, 500]
    n_simulations = 1000
    
    results_mle = {n: {'mu': [], 'sigma': []} for n in sample_sizes}
    results_mme = {n: {'mu': [], 'sigma': []} for n in sample_sizes}
    
    for n in sample_sizes:
        for _ in tqdm(range(n_simulations), desc=f"  n={n}", leave=False):
            data = np.random.normal(true_mu, true_sigma, n)
            
            # MLE
            mu_mle, sigma_mle = estimator.normal_mle(data)
            results_mle[n]['mu'].append(mu_mle)
            results_mle[n]['sigma'].append(sigma_mle)
            
            # MME
            mu_mme, sigma_mme = estimator.normal_mme(data)
            results_mme[n]['mu'].append(mu_mme)
            results_mme[n]['sigma'].append(sigma_mme)
    
    # 分析结果
    if logger:
        logger.add_line("\n均值参数μ的估计:")
    else:
        print("\n  均值参数μ的估计:")
    
    header_line = f"    {'n':<6} {'MLE均值':<12} {'MLE偏差':<12} {'MLE MSE':<12} " \
                  f"{'MME均值':<12} {'MME偏差':<12} {'MME MSE':<12}"
    if logger:
        logger.add_line(header_line)
    else:
        print(header_line)
    for n in sample_sizes:
        mle_mu = np.array(results_mle[n]['mu'])
        mme_mu = np.array(results_mme[n]['mu'])
        
        mle_bias = np.mean(mle_mu) - true_mu
        mme_bias = np.mean(mme_mu) - true_mu
        mle_mse = np.mean((mle_mu - true_mu)**2)
        mme_mse = np.mean((mme_mu - true_mu)**2)
        
        print(f"    {n:<6} {np.mean(mle_mu):<12.6f} {mle_bias:<12.6f} {mle_mse:<12.6f} "
              f"{np.mean(mme_mu):<12.6f} {mme_bias:<12.6f} {mme_mse:<12.6f}")
    
    print("\n  标准差参数σ的估计:")
    print(f"    {'n':<6} {'MLE均值':<12} {'MLE偏差':<12} {'MLE MSE':<12} "
          f"{'MME均值':<12} {'MME偏差':<12} {'MME MSE':<12}")
    for n in sample_sizes:
        mle_sigma = np.array(results_mle[n]['sigma'])
        mme_sigma = np.array(results_mme[n]['sigma'])
        
        mle_bias = np.mean(mle_sigma) - true_sigma
        mme_bias = np.mean(mme_sigma) - true_sigma
        mle_mse = np.mean((mle_sigma - true_sigma)**2)
        mme_mse = np.mean((mme_sigma - true_sigma)**2)
        
        print(f"    {n:<6} {np.mean(mle_sigma):<12.6f} {mle_bias:<12.6f} {mle_mse:<12.6f} "
              f"{np.mean(mme_sigma):<12.6f} {mme_bias:<12.6f} {mme_mse:<12.6f}")
    
    # 可视化
    visualize_normal_estimation(results_mle, results_mme, sample_sizes, 
                                true_mu, true_sigma)

def visualize_normal_estimation(results_mle, results_mme, sample_sizes, 
                                true_mu, true_sigma):
    """可视化正态分布估计结果"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # μ的估计分布
    for idx, n in enumerate([10, 50, 500]):
        ax = axes[0, idx]
        mle_mu = results_mle[n]['mu']
        mme_mu = results_mme[n]['mu']
        
        ax.hist(mle_mu, bins=40, alpha=0.6, label='MLE', density=True, 
               color='skyblue', edgecolor='black')
        ax.hist(mme_mu, bins=40, alpha=0.6, label='MME', density=True,
               color='lightcoral', edgecolor='black')
        ax.axvline(true_mu, color='red', linestyle='--', linewidth=2, 
                  label=f'真值={true_mu}')
        ax.axvline(np.mean(mle_mu), color='blue', linestyle=':', linewidth=2,
                  label=f'MLE均值={np.mean(mle_mu):.2f}')
        ax.set_xlabel('μ估计值', fontsize=11)
        ax.set_ylabel('密度', fontsize=11)
        ax.set_title(f'n={n}时μ的估计', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # σ的估计分布
    for idx, n in enumerate([10, 50, 500]):
        ax = axes[1, idx]
        mle_sigma = results_mle[n]['sigma']
        mme_sigma = results_mme[n]['sigma']
        
        ax.hist(mle_sigma, bins=40, alpha=0.6, label='MLE', density=True,
               color='skyblue', edgecolor='black')
        ax.hist(mme_sigma, bins=40, alpha=0.6, label='MME', density=True,
               color='lightcoral', edgecolor='black')
        ax.axvline(true_sigma, color='red', linestyle='--', linewidth=2,
                  label=f'真值={true_sigma}')
        ax.axvline(np.mean(mle_sigma), color='blue', linestyle=':', linewidth=2,
                  label=f'MLE均值={np.mean(mle_sigma):.2f}')
        ax.set_xlabel('σ估计值', fontsize=11)
        ax.set_ylabel('密度', fontsize=11)
        ax.set_title(f'n={n}时σ的估计', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('实验三：正态分布参数估计 - MLE vs MME', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp3_normal_estimation.png')
    plt.close()

def experiment_exponential(estimator):
    """指数分布参数估计实验"""
    true_lambda = 0.5
    sample_sizes = [10, 30, 50, 100, 500]
    n_simulations = 1000
    
    results_mle = {n: [] for n in sample_sizes}
    results_mme = {n: [] for n in sample_sizes}
    
    for n in sample_sizes:
        for _ in tqdm(range(n_simulations), desc=f"  n={n}", leave=False):
            data = np.random.exponential(1/true_lambda, n)
            
            lambda_mle = estimator.exponential_mle(data)
            lambda_mme = estimator.exponential_mme(data)
            
            results_mle[n].append(lambda_mle)
            results_mme[n].append(lambda_mme)
    
    # 分析结果
    print(f"\n  参数λ的估计 (真值={true_lambda}):")
    print(f"    {'n':<6} {'MLE均值':<12} {'MLE偏差':<12} {'MLE MSE':<12}")
    for n in sample_sizes:
        mle_lambda = np.array(results_mle[n])
        mle_bias = np.mean(mle_lambda) - true_lambda
        mle_mse = np.mean((mle_lambda - true_lambda)**2)
        
        print(f"    {n:<6} {np.mean(mle_lambda):<12.6f} {mle_bias:<12.6f} {mle_mse:<12.6f}")
    
    # 可视化
    visualize_exponential_estimation(results_mle, sample_sizes, true_lambda)

def visualize_exponential_estimation(results_mle, sample_sizes, true_lambda):
    """可视化指数分布估计结果"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    for idx, n in enumerate(sample_sizes):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        lambda_mle = results_mle[n]
        
        ax.hist(lambda_mle, bins=40, alpha=0.7, density=True,
               color='lightgreen', edgecolor='black', label='MLE估计')
        ax.axvline(true_lambda, color='red', linestyle='--', linewidth=2,
                  label=f'真值={true_lambda}')
        ax.axvline(np.mean(lambda_mle), color='blue', linestyle=':', linewidth=2,
                  label=f'均值={np.mean(lambda_mle):.3f}')
        
        # 添加统计信息
        bias = np.mean(lambda_mle) - true_lambda
        mse = np.mean((np.array(lambda_mle) - true_lambda)**2)
        ax.text(0.98, 0.95, f'Bias={bias:.4f}\nMSE={mse:.4f}',
               transform=ax.transAxes, verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               fontsize=9)
        
        ax.set_xlabel('λ估计值', fontsize=11)
        ax.set_ylabel('密度', fontsize=11)
        ax.set_title(f'n={n}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 隐藏最后一个子图
    axes[1, 2].axis('off')
    
    plt.suptitle('实验三：指数分布参数估计 Exp(λ=0.5)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp3_exponential_estimation.png')
    plt.close()

def experiment_gamma(estimator):
    """伽玛分布参数估计实验"""
    true_alpha, true_beta = 2, 0.5
    sample_sizes = [50, 100, 200]
    n_simulations = 500
    
    results_mle = {n: {'alpha': [], 'beta': []} for n in sample_sizes}
    results_mme = {n: {'alpha': [], 'beta': []} for n in sample_sizes}
    
    for n in sample_sizes:
        for _ in tqdm(range(n_simulations), desc=f"  n={n}", leave=False):
            data = np.random.gamma(true_alpha, 1/true_beta, n)
            
            # MLE
            try:
                alpha_mle, beta_mle = estimator.gamma_mle(data)
                results_mle[n]['alpha'].append(alpha_mle)
                results_mle[n]['beta'].append(beta_mle)
            except:
                pass
            
            # MME
            alpha_mme, beta_mme = estimator.gamma_mme(data)
            results_mme[n]['alpha'].append(alpha_mme)
            results_mme[n]['beta'].append(beta_mme)
    
    print(f"\n  伽玛分布参数估计 (真值: α={true_alpha}, β={true_beta}):")
    print(f"    {'n':<6} {'α_MLE':<10} {'α_MME':<10} {'β_MLE':<10} {'β_MME':<10}")
    for n in sample_sizes:
        alpha_mle_mean = np.mean(results_mle[n]['alpha'])
        alpha_mme_mean = np.mean(results_mme[n]['alpha'])
        beta_mle_mean = np.mean(results_mle[n]['beta'])
        beta_mme_mean = np.mean(results_mme[n]['beta'])
        
        print(f"    {n:<6} {alpha_mle_mean:<10.4f} {alpha_mme_mean:<10.4f} "
              f"{beta_mle_mean:<10.4f} {beta_mme_mean:<10.4f}")

def experiment_beta(estimator):
    """贝塔分布参数估计实验"""
    true_alpha, true_beta = 2, 5
    sample_sizes = [50, 100, 200]
    n_simulations = 500
    
    results_mle = {n: {'alpha': [], 'beta': []} for n in sample_sizes}
    results_mme = {n: {'alpha': [], 'beta': []} for n in sample_sizes}
    
    for n in sample_sizes:
        for _ in tqdm(range(n_simulations), desc=f"  n={n}", leave=False):
            data = np.random.beta(true_alpha, true_beta, n)
            
            # MLE
            try:
                alpha_mle, beta_mle = estimator.beta_mle(data)
                results_mle[n]['alpha'].append(alpha_mle)
                results_mle[n]['beta'].append(beta_mle)
            except:
                pass
            
            # MME
            alpha_mme, beta_mme = estimator.beta_mme(data)
            results_mme[n]['alpha'].append(alpha_mme)
            results_mme[n]['beta'].append(beta_mme)
    
    print(f"\n  贝塔分布参数估计 (真值: α={true_alpha}, β={true_beta}):")
    print(f"    {'n':<6} {'α_MLE':<10} {'α_MME':<10} {'β_MLE':<10} {'β_MME':<10}")
    for n in sample_sizes:
        alpha_mle_mean = np.mean(results_mle[n]['alpha'])
        alpha_mme_mean = np.mean(results_mme[n]['alpha'])
        beta_mle_mean = np.mean(results_mle[n]['beta'])
        beta_mme_mean = np.mean(results_mme[n]['beta'])
        
        print(f"    {n:<6} {alpha_mle_mean:<10.4f} {alpha_mme_mean:<10.4f} "
              f"{beta_mle_mean:<10.4f} {beta_mme_mean:<10.4f}")

def comprehensive_comparison(estimator):
    """综合对比MSE随样本量变化"""
    print("\n步骤：生成综合对比图...")
    
    # 正态分布
    true_mu, true_sigma = 5, 2
    sample_sizes = np.arange(10, 501, 10)
    n_sim = 200
    
    mse_mle_mu = []
    mse_mme_mu = []
    mse_mle_sigma = []
    mse_mme_sigma = []
    
    for n in tqdm(sample_sizes, desc="  正态分布MSE分析"):
        mle_mu_list, mme_mu_list = [], []
        mle_sigma_list, mme_sigma_list = [], []
        
        for _ in range(n_sim):
            data = np.random.normal(true_mu, true_sigma, n)
            mu_mle, sigma_mle = estimator.normal_mle(data)
            mu_mme, sigma_mme = estimator.normal_mme(data)
            
            mle_mu_list.append(mu_mle)
            mme_mu_list.append(mu_mme)
            mle_sigma_list.append(sigma_mle)
            mme_sigma_list.append(sigma_mme)
        
        mse_mle_mu.append(np.mean((np.array(mle_mu_list) - true_mu)**2))
        mse_mme_mu.append(np.mean((np.array(mme_mu_list) - true_mu)**2))
        mse_mle_sigma.append(np.mean((np.array(mle_sigma_list) - true_sigma)**2))
        mse_mme_sigma.append(np.mean((np.array(mme_sigma_list) - true_sigma)**2))
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(sample_sizes, mse_mle_mu, 'b-', linewidth=2, label='MLE', marker='o', markersize=3)
    axes[0].plot(sample_sizes, mse_mme_mu, 'r--', linewidth=2, label='MME', marker='s', markersize=3)
    axes[0].set_xlabel('样本量 n', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('μ参数的MSE比较', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    axes[1].plot(sample_sizes, mse_mle_sigma, 'b-', linewidth=2, label='MLE', marker='o', markersize=3)
    axes[1].plot(sample_sizes, mse_mme_sigma, 'r--', linewidth=2, label='MME', marker='s', markersize=3)
    axes[1].set_xlabel('样本量 n', fontsize=12)
    axes[1].set_ylabel('MSE', fontsize=12)
    axes[1].set_title('σ参数的MSE比较', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.suptitle('实验三：MLE vs MME - MSE随样本量变化', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'exp3_mse_comparison.png')
    plt.close()

if __name__ == "__main__":
    run_experiment()
