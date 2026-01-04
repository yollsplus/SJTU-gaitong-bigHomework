"""
快速结果保存脚本
为所有实验添加文本输出重定向功能
"""

import sys
import os
from datetime import datetime

# 运行所有实验并保存输出到文本文件
experiments = [
    ("实验一：Box-Muller变换", "experiment_1_box_muller"),
    ("实验二：中心极限定理验证", "experiment_2_clt"),
    ("实验三：参数估计方法对比", "experiment_3_parameter_estimation"),
    ("实验四：假设检验功效分析", "experiment_4_hypothesis_testing"),
    ("实验五：蒙特卡洛方法", "experiment_5_monte_carlo"),
    ("实验六：Bootstrap方法", "experiment_6_bootstrap")
]

# 创建结果目录
os.makedirs('results', exist_ok=True)

# 汇总文件
summary_file = open('results/实验结果汇总.txt', 'w', encoding='utf-8')
summary_file.write("="*70 + "\n")
summary_file.write("概率论与数理统计 - 数值模拟实验结果汇总\n")
summary_file.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
summary_file.write("="*70 + "\n\n")

for idx, (name, module_name) in enumerate(experiments, 1):
    print(f"\n正在保存 [{idx}/6] {name} 的结果...")
    
    output_file = f'results/{name.replace("：", "_")}_输出.txt'
    
    # 重定向stdout到文件
    original_stdout = sys.stdout
    
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        
        try:
            module = __import__(module_name)
            print(f"\n{'='*70}")
            print(f"{name}")
            print(f"{'='*70}\n")
            module.run_experiment()
        except Exception as e:
            print(f"\n错误: {str(e)}")
    
    # 恢复stdout
    sys.stdout = original_stdout
    
    # 读取并添加到汇总文件
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        summary_file.write(f"\n{'='*70}\n")
        summary_file.write(f"{name}\n")
        summary_file.write(f"{'='*70}\n")
        summary_file.write(content)
        summary_file.write("\n\n")
    
    print(f"✓ 已保存到: {output_file}")

summary_file.close()
print(f"\n✓ 所有结果已汇总到: results/实验结果汇总.txt")
print("✓ 每个实验的详细输出也已单独保存")
