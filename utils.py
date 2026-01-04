"""
工具函数模块
提供通用的绘图配置和辅助函数
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置中文字体和绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
sns.set_style("whitegrid")
sns.set_palette("husl")

def ensure_results_dir():
    """确保结果目录存在"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir

def save_figure(fig, filename, dpi=300):
    """保存高质量图片"""
    results_dir = ensure_results_dir()
    filepath = results_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"✓ 图片已保存: {filepath}")

def print_section(title):
    """打印美观的分节标题"""
    width = 60
    print("\n" + "="*width)
    print(f"{title:^{width}}")
    print("="*width + "\n")

def print_result(label, value, unit=""):
    """打印格式化的结果"""
    if isinstance(value, float):
        print(f"  {label}: {value:.6f} {unit}")
    else:
        print(f"  {label}: {value} {unit}")

class ResultLogger:
    """实验结果记录器"""
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.results_dir = ensure_results_dir()
        self.log_file = self.results_dir / f"{experiment_name}_results.txt"
        self.lines = []
        
        # 写入标题
        self.add_line("="*70)
        self.add_line(f"{experiment_name} - 实验结果")
        self.add_line("="*70)
        self.add_line("")
    
    def add_line(self, text=""):
        """添加一行文本"""
        self.lines.append(text)
        print(text)
    
    def add_section(self, title):
        """添加分节标题"""
        self.add_line("")
        self.add_line("-"*70)
        self.add_line(title)
        self.add_line("-"*70)
        self.add_line("")
    
    def add_result(self, label, value, unit=""):
        """添加结果"""
        if isinstance(value, float):
            line = f"  {label}: {value:.6f} {unit}"
        else:
            line = f"  {label}: {value} {unit}"
        self.add_line(line)
    
    def add_table_header(self, headers):
        """添加表格标题"""
        header_line = "  " + " | ".join(f"{h:^12}" for h in headers)
        self.add_line(header_line)
        self.add_line("  " + "-"*len(header_line))
    
    def add_table_row(self, values):
        """添加表格行"""
        row_line = "  " + " | ".join(f"{str(v):^12}" for v in values)
        self.add_line(row_line)
    
    def save(self):
        """保存结果到文件"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))
        print(f"\n✓ 结果已保存到: {self.log_file}")
