import argparse
import torch
import torch.nn as nn
import torchvision.models as models

import matplotlib.pyplot as plt

def plot_fc_distribution(fc_layer, save_path=None):
    """
    以图片形式展示全连接层权重的数据分布。

    参数:
        fc_layer (nn.Linear): 全连接层对象。
        save_path (str, optional): 图片保存路径。如果为 None，则直接显示图片。
    """
    # 获取全连接层的权重
    weights = fc_layer.weight.data.cpu().numpy().flatten()

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=100, color='blue', alpha=0.7, edgecolor='black')
    plt.title("full connection", fontsize=16)
    plt.xlabel("weights", fontsize=14)
    plt.ylabel("frequency", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存或显示图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    else:
        plt.show()

# 定义函数，计算并打印全连接层的统计信息
def print_fc_stats(fc_layer):
    # 获取全连接层的权重
    weights = fc_layer.weight.data

    # 计算最大值、最小值、均值和方差
    max_value = torch.max(weights).item()
    min_value = torch.min(weights).item()
    mean_value = torch.mean(weights).item()
    var_value = torch.var(weights).item()

    # 打印结果
    print(f"全连接层统计信息:")
    print(f"  - 最大值: {max_value}")
    print(f"  - 最小值: {min_value}")
    print(f"  - 均值: {mean_value}")
    print(f"  - 方差: {var_value}")

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="打印 ResNet18 全连接层的统计信息")
    parser.add_argument('--load', type=str, required=True, help="模型路径")
    args = parser.parse_args()

    # 加载模型
    model = models.resnet18(weights=None)  # 初始化 ResNet18
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 200)  # 确保全连接层与预训练模型一致
    model.load_state_dict(torch.load(args.load))  # 加载模型权重
    model.eval()  # 设置为评估模式

    # 打印全连接层的统计信息
    print_fc_stats(model.fc)
    plot_fc_distribution(model.fc, 'full_fc.png')

if __name__ == "__main__":
    main()