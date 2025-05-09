# plotting_utils.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.font_manager as fm
import os

def _load_font_properties(font_filename="NotoSansSC-VariableFont_wght.ttf", font_subdir="fonts"):
    """
    辅助函数：加载字体属性对象。
    """
    font_prop = None
    try:
        current_dir = os.getcwd() # 在 Notebook 环境中通常是 Notebook 所在目录
        font_path = os.path.join(current_dir, font_subdir, font_filename)
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            print(f"绘图：成功加载字体属性: {font_prop.get_name()} (来自 {font_path})")
        else:
            print(f"绘图错误：找不到字体文件 '{font_path}'")
    except Exception as e:
        print(f"绘图错误：加载字体属性时出错: {e}")
    return font_prop

def plot_overall_cm(cm_np, display_labels, model_name="",
                    font_filename="NotoSansSC-VariableFont_wght.ttf", font_subdir="fonts"):
    """
    绘制整体混淆矩阵。

    Args:
        cm_np (np.ndarray): 混淆矩阵 (NumPy 数组)。
        display_labels (list): 类别标签名称列表。
        model_name (str, optional): 模型名称，用于标题。默认为 ""。
        font_filename (str, optional): 字体文件名。默认为 "NotoSansSC-VariableFont_wght.ttf"。
        font_subdir (str, optional): 字体文件所在子目录。默认为 "fonts"。
    """
    print("\n绘制整体混淆矩阵...")
    font_prop = _load_font_properties(font_filename, font_subdir)
    plt.rcParams['axes.unicode_minus'] = False # 确保负号显示

    if font_prop and cm_np is not None:
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_np, display_labels=display_labels)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, values_format='d')

        # --- 手动设置字体 ---
        title = f'整体混淆矩阵 (Overall Confusion Matrix)'
        if model_name:
            title += f' - {model_name}'
        plt.title(title, fontproperties=font_prop)
        try:
            ax = disp.ax_
            for label in ax.get_xticklabels():
                label.set_fontproperties(font_prop)
            for label in ax.get_yticklabels():
                label.set_fontproperties(font_prop)
        except Exception as tick_e:
            print(f"警告：设置刻度标签字体时出错: {tick_e}")

        plt.tight_layout()
        plt.show()
    elif not font_prop:
        print("错误：未能加载字体属性，无法正确绘制带中文的混淆矩阵。")
    else:
        print("错误：混淆矩阵数据为空。")

def plot_acc_vs_snr(snr_acc_dict, model_name="",
                    font_filename="NotoSansSC-VariableFont_wght.ttf", font_subdir="fonts"):
    """
    绘制 Accuracy vs. SNR 曲线。

    Args:
        snr_acc_dict (dict): 包含 {snr: accuracy} 的字典。
        model_name (str, optional): 模型名称，用于标题和图例。默认为 ""。
        font_filename (str, optional): 字体文件名。默认为 "NotoSansSC-VariableFont_wght.ttf"。
        font_subdir (str, optional): 字体文件所在子目录。默认为 "fonts"。
    """
    print("\n绘制 Accuracy vs. SNR 曲线...")
    font_prop = _load_font_properties(font_filename, font_subdir)
    plt.rcParams['axes.unicode_minus'] = False # 确保负号显示

    if not snr_acc_dict:
        print("未能计算 SNR 准确率数据，无法绘制曲线。")
        return

    if not font_prop:
        print("错误：未能加载字体属性，无法正确绘制带中文的图表。")
        # 仍然尝试绘制，但中文可能显示异常
        # return # 如果希望字体失败则不绘制，取消注释

    # 按 SNR 排序
    sorted_snrs = sorted(snr_acc_dict.keys())
    accuracies = [snr_acc_dict[snr] for snr in sorted_snrs] # 转换为百分比

    plt.figure(figsize=(10, 6))
    plot_label = f'{model_name} 模型准确率' if model_name else '模型准确率'
    plt.plot(sorted_snrs, accuracies, marker='o', linestyle='-', label=plot_label)

    title = f'准确率 vs. 信噪比 (Accuracy vs. SNR)'
    if model_name:
        title += f' - {model_name}'

    # --- 手动设置字体 ---
    plt.xlabel('信噪比 (SNR)', fontproperties=font_prop if font_prop else None)
    plt.ylabel('准确率 (%)', fontproperties=font_prop if font_prop else None)
    plt.title(title, fontproperties=font_prop if font_prop else None)

    plt.grid(True)
    plt.xticks(sorted_snrs)
    plt.ylim(0, 105)
    plt.legend(prop=font_prop if font_prop else None)
    plt.show()


def plot_snr_cms(snr_cms_dict_np, display_labels, model_name="",
                 font_filename="NotoSansSC-VariableFont_wght.ttf", font_subdir="fonts",
                 save_individual_figs=False, individual_fig_dir="snr_cm_plots"):
    """
    绘制每个 SNR 下的混淆矩阵。
    会先绘制包含所有 SNR 的网格大图，然后为每个 SNR 绘制并显示/保存单独的小图。

    Args:
        snr_cms_dict_np (dict): 包含 {snr: cm_np_array} 的字典。
        display_labels (list): 类别标签名称列表。
        model_name (str, optional): 模型名称，用于标题。默认为 ""。
        font_filename (str, optional): 字体文件名。默认为 "NotoSansSC-VariableFont_wght.ttf"。
        font_subdir (str, optional): 字体文件所在子目录。默认为 "fonts"。
        save_individual_figs (bool, optional): 是否保存每个 SNR 的单独图像。默认为 False。
        individual_fig_dir (str, optional): 保存单独图像的目录名。默认为 "snr_cm_plots"。
    """
    print("\n绘制不同 SNR 下的混淆矩阵...")
    font_prop = _load_font_properties(font_filename, font_subdir)
    plt.rcParams['axes.unicode_minus'] = False # 确保负号显示

    if not snr_cms_dict_np:
        print("没有找到有效的 SNR 混淆矩阵数据来绘制。")
        return

    if not font_prop:
        print("警告：未能加载字体属性，中文可能无法正确显示。")
        # 仍然尝试绘制

    snr_list_plot = sorted(snr_cms_dict_np.keys())
    n_snrs = len(snr_list_plot)
    n_cols = 5
    n_rows = int(np.ceil(n_snrs / n_cols))

    if n_snrs == 0:
        print("没有有效的 SNR 数据。")
        return

    # --- 1. 绘制网格大图 ---
    print("绘制包含所有 SNR 的网格混淆矩阵图...")
    fig_grid, axes_grid = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    # 处理 axes_grid 不是二维数组的情况
    if n_snrs == 1:
        axes_grid = np.array([[axes_grid]])
    elif n_rows == 1:
        axes_grid = axes_grid.reshape(1, -1)
    elif n_cols == 1:
        axes_grid = axes_grid.reshape(-1, 1)

    ax_idx = 0
    for snr in snr_list_plot:
        if snr in snr_cms_dict_np:
            row_idx = ax_idx // n_cols
            col_idx = ax_idx % n_cols
            ax = axes_grid[row_idx, col_idx]

            cm_snr_np = snr_cms_dict_np[snr]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_snr_np, display_labels=display_labels)
            disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=90, values_format='d', colorbar=False) # 网格图通常省略色条

            # 设置子图标题 (非中文)
            ax.set_title(f'SNR = {snr} dB')

            # --- 手动设置刻度标签字体 ---
            if font_prop:
                try:
                    for label in ax.get_xticklabels():
                        label.set_fontproperties(font_prop)
                    for label in ax.get_yticklabels():
                        label.set_fontproperties(font_prop)
                except Exception as tick_e:
                    print(f"警告：为网格图 SNR={snr} 设置刻度标签字体时出错: {tick_e}")

            # 简化坐标轴标签
            if col_idx != 0:
                ax.set_ylabel('')
            else:
                ax.set_ylabel("真实标签", fontproperties=font_prop if font_prop else None) # 第1列显示 Y 轴标签
            if row_idx != n_rows - 1:
                 ax.set_xlabel('')
            else:
                 ax.set_xlabel("预测标签", fontproperties=font_prop if font_prop else None) # 最后1行显示 X 轴标签

            ax_idx += 1

    # 隐藏多余的子图网格
    while ax_idx < n_rows * n_cols:
        row_idx = ax_idx // n_cols
        col_idx = ax_idx % n_cols
        if n_rows * n_cols > 1:
             fig_grid.delaxes(axes_grid[row_idx, col_idx])
        ax_idx += 1

    # 设置总标题
    suptitle = '每个 SNR 下的混淆矩阵'
    if model_name:
        suptitle += f' - {model_name}'
    plt.suptitle(suptitle, fontsize=16, fontproperties=font_prop if font_prop else None)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("网格混淆矩阵图绘制完成。")

    # --- 2. 绘制并显示/保存每个 SNR 的单独小图 ---
    print("\n绘制每个 SNR 的单独混淆矩阵图...")
    if save_individual_figs:
        os.makedirs(individual_fig_dir, exist_ok=True)
        print(f"单独图像将保存到 '{individual_fig_dir}' 目录。")

    for snr in snr_list_plot:
        if snr in snr_cms_dict_np:
            cm_snr_np = snr_cms_dict_np[snr]

            fig_single, ax_single = plt.subplots(figsize=(8, 6)) # 创建新图
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_snr_np, display_labels=display_labels)
            disp.plot(ax=ax_single, cmap=plt.cm.Blues, xticks_rotation=45, values_format='d', colorbar=True) # 单独图可以显示色条

            # --- 手动设置字体 ---
            title_single = f'混淆矩阵 (Confusion Matrix)'
            if model_name:
                title_single += f' - {model_name}'
            title_single += f' (SNR = {snr} dB)'
            ax_single.set_title(title_single, fontproperties=font_prop if font_prop else None)

            if font_prop:
                try:
                    for label in ax_single.get_xticklabels():
                        label.set_fontproperties(font_prop)
                    for label in ax_single.get_yticklabels():
                        label.set_fontproperties(font_prop)
                    ax_single.set_xlabel("预测标签", fontproperties=font_prop)
                    ax_single.set_ylabel("真实标签", fontproperties=font_prop)
                except Exception as tick_e:
                    print(f"警告：为单独图 SNR={snr} 设置字体时出错: {tick_e}")
            else:
                 ax_single.set_xlabel("预测标签")
                 ax_single.set_ylabel("真实标签")


            plt.tight_layout()

            if save_individual_figs:
                # 构建安全的文件名
                safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name)
                snr_str = f"neg{-snr}" if snr < 0 else f"{snr}"
                filename = f"cm_{safe_model_name}_snr_{snr_str}dB.png"
                filepath = os.path.join(individual_fig_dir, filename)
                try:
                    plt.savefig(filepath)
                    print(f"已保存: {filepath}")
                except Exception as save_e:
                    print(f"错误：保存图像 {filepath} 失败: {save_e}")
                plt.close(fig_single) # 保存后关闭图形，避免过多窗口
            else:
                plt.show() # 不保存则直接显示

    print("单独混淆矩阵图绘制完成。")