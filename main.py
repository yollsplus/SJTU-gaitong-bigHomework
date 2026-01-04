"""
概率论与数理统计 - 数值模拟实验项目
主运行脚本

运行所有6个实验并生成完整的结果
"""

import time
import sys
from utils import print_section, ensure_results_dir

def run_all_experiments():
    """运行所有实验"""
    
    print("="*70)
    print(" " * 15 + "概率论与数理统计 - 数值模拟实验")
    print("="*70)
    print()
    
    ensure_results_dir()
    
    experiments = [
        ("实验一：Box-Muller变换", "experiment_1_box_muller"),
        ("实验二：中心极限定理验证", "experiment_2_clt"),
        ("实验三：参数估计方法对比", "experiment_3_parameter_estimation"),
        ("实验四：假设检验功效分析", "experiment_4_hypothesis_testing"),
        ("实验五：蒙特卡洛方法", "experiment_5_monte_carlo"),
        ("实验六：Bootstrap方法", "experiment_6_bootstrap")
    ]
    
    total_start_time = time.time()
    
    for idx, (name, module_name) in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(f"正在运行 [{idx}/6] {name}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        try:
            # 动态导入模块
            module = __import__(module_name)
            module.run_experiment()
            
            elapsed_time = time.time() - start_time
            print(f"\n✓ {name} 完成！ (耗时: {elapsed_time:.2f}秒)")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"\n✗ {name} 失败！ (耗时: {elapsed_time:.2f}秒)")
            print(f"错误信息: {str(e)}")
            import traceback
            traceback.print_exc()
    
    total_elapsed_time = time.time() - total_start_time
    
    print(f"\n{'='*70}")
    print(f" " * 20 + "所有实验完成！")
    print(f"{'='*70}")
    print(f"\n总耗时: {total_elapsed_time:.2f}秒 ({total_elapsed_time/60:.2f}分钟)")
    print(f"\n所有结果已保存到 'results' 目录")
    print(f"共生成约 {len(experiments) * 3} 张高质量图表")

def run_single_experiment(exp_number):
    """运行单个实验"""
    
    experiments = {
        1: ("实验一：Box-Muller变换", "experiment_1_box_muller"),
        2: ("实验二：中心极限定理验证", "experiment_2_clt"),
        3: ("实验三：参数估计方法对比", "experiment_3_parameter_estimation"),
        4: ("实验四：假设检验功效分析", "experiment_4_hypothesis_testing"),
        5: ("实验五：蒙特卡洛方法", "experiment_5_monte_carlo"),
        6: ("实验六：Bootstrap方法", "experiment_6_bootstrap")
    }
    
    if exp_number not in experiments:
        print(f"错误：实验编号 {exp_number} 不存在！")
        print("请选择 1-6 之间的实验编号。")
        return
    
    name, module_name = experiments[exp_number]
    
    print(f"\n{'='*70}")
    print(f"正在运行 {name}")
    print(f"{'='*70}\n")
    
    ensure_results_dir()
    start_time = time.time()
    
    try:
        module = __import__(module_name)
        module.run_experiment()
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ {name} 完成！ (耗时: {elapsed_time:.2f}秒)")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {name} 失败！ (耗时: {elapsed_time:.2f}秒)")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 运行指定实验
        try:
            exp_num = int(sys.argv[1])
            run_single_experiment(exp_num)
        except ValueError:
            print("错误：请提供有效的实验编号（1-6）")
            print("用法: python main.py [实验编号]")
            print("或者运行所有实验: python main.py")
    else:
        # 运行所有实验
        run_all_experiments()
