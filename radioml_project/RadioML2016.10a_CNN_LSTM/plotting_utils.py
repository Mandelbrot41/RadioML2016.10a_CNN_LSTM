# plotting_utils.py
# 包含用于可视化调制识别模型评估结果的函数。
# 主要依赖 matplotlib 和 scikit-learn。
# 支持加载自定义字体以正确显示中文等非 ASCII 字符。

import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns # seaborn 未在此文件中使用，已移除
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.font_manager as fm
import os
import traceback # 导入 traceback 模块用于打印详细错误信息

# --- 全局设置 ---
# 确保 matplotlib 能够正确显示负号
plt.rcParams['axes.unicode_minus'] = False

def _load_font_properties(font_filename="NotoSansSC-VariableFont_wght.ttf", font_subdir="fonts"):
    """
    辅助函数：加载字体属性对象 (matplotlib.font_manager.FontProperties)。
    用于在绘图时指定字体，特别是为了支持中文显示。

    Args:
        font_filename (str, optional): 字体文件的名称。
                                       默认为 "NotoSansSC-VariableFont_wght.ttf"。
        font_subdir (str, optional): 字体文件相对于当前工作目录的子目录名。
                                     默认为 "fonts"。

    Returns:
        matplotlib.font_manager.FontProperties or None:
            加载成功则返回字体属性对象，否则返回 None。
    """
    font_prop = None
    try:
        # 注意: os.getcwd() 返回的是 *运行脚本时* 的当前工作目录。
        # 在 Notebook 环境中，这通常是 Notebook 文件所在的目录。
        # 如果脚本在其他环境中运行，可能需要调整路径逻辑，
        # 例如使用相对于此 .py 文件的路径 (os.path.join(os.path.dirname(__file__), font_subdir, font_filename))
        # 或依赖 matplotlib 的字体查找机制。
        current_dir = os.getcwd()
        font_path = os.path.join(current_dir, font_subdir, font_filename)
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            # 打印成功加载的信息（可选，用于调试）
            print(f"绘图：成功加载字体属性: {font_prop.get_name()} (来自 {font_path})")
        else:
            print(f"绘图错误：找不到字体文件 '{font_path}'。请确保字体文件存在于指定路径。")
            # 尝试查找系统默认支持中文的字体作为备选（可选）
            # for font in fm.fontManager.ttflist:
            #     if 'SimHei' in font.name or 'Microsoft YaHei' in font.name: # 示例
            #         font_prop = fm.FontProperties(fname=font.fname)
            #         print(f"绘图：使用备选字体: {font_prop.get_name()}")
            #         break
            # if not font_prop:
            #     print("绘图警告：也未能找到合适的备选系统字体。")

    except Exception as e:
        print(f"绘图错误：加载字体属性时发生异常: {e}")
        traceback.print_exc()

    if not font_prop:
         print("绘图警告：未能加载指定字体，绘图中的中文可能无法正确显示。")

    return font_prop

def plot_overall_cm(cm_np, display_labels, model_name="",
                    font_filename="NotoSansSC-VariableFont_wght.ttf", font_subdir="fonts"):
    """
    绘制整体混淆矩阵。

    Args:
        cm_np (np.ndarray): 混淆矩阵 (NumPy 数组, 形状应为 N x N)。
        display_labels (list): 长度为 N 的类别标签名称列表 (字符串)。
        model_name (str, optional): 模型名称，用于图表标题。默认为 ""。
        font_filename (str, optional): 字体文件名。默认为 "NotoSansSC-VariableFont_wght.ttf"。
        font_subdir (str, optional): 字体文件所在子目录。默认为 "fonts"。
    """
    print("\n绘制整体混淆矩阵...")

    # 输入验证
    if cm_np is None or not isinstance(cm_np, np.ndarray) or cm_np.ndim != 2 or cm_np.shape[0] != cm_np.shape[1]:
        print("错误：提供的混淆矩阵数据无效 (必须是二维方阵 NumPy 数组)。")
        return
    if not display_labels or len(display_labels) != cm_np.shape[0]:
        print("错误：提供的 display_labels 无效或其长度与混淆矩阵维度不匹配。")
        return

    font_prop = _load_font_properties(font_filename, font_subdir)
    # plt.rcParams['axes.unicode_minus'] = False # 已移至全局

    plt.figure(figsize=(10, 8)) # 创建新的图形
    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_np, display_labels=display_labels)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, values_format='d') # 'd' 格式化为整数

        # --- 设置标题和标签字体 ---
        title = '整体混淆矩阵 (Overall Confusion Matrix)'
        if model_name:
            title += f' - {model_name}'

        # 如果成功加载字体，则应用
        if font_prop:
             plt.title(title, fontproperties=font_prop)
             # 尝试为刻度标签设置字体（注意：这有时可能因 matplotlib 版本或后端而异）
             try:
                 ax = disp.ax_ # 获取 ConfusionMatrixDisplay 的 Axes 对象
                 for label in ax.get_xticklabels():
                     label.set_fontproperties(font_prop)
                 for label in ax.get_yticklabels():
                     label.set_fontproperties(font_prop)
                 ax.set_xlabel("预测标签", fontproperties=font_prop)
                 ax.set_ylabel("真实标签", fontproperties=font_prop)
             except Exception as tick_e:
                 print(f"警告：设置刻度标签字体时出错: {tick_e}")
                 # 即使刻度标签字体失败，也继续设置标题
                 plt.xlabel("预测标签")
                 plt.ylabel("真实标签")
        else:
             plt.title(title) # 未加载字体则使用默认字体
             plt.xlabel("预测标签")
             plt.ylabel("真实标签")


        plt.tight_layout() # 调整布局防止标签重叠
        plt.show() # 显示图形

    except Exception as plot_e:
        print(f"错误：绘制整体混淆矩阵时发生异常: {plot_e}")
        traceback.print_exc()

def plot_acc_vs_snr(snr_acc_dict, model_name="",
                    font_filename="NotoSansSC-VariableFont_wght.ttf", font_subdir="fonts"):
    """
    绘制准确率 (Accuracy) vs. 信噪比 (SNR) 曲线。

    Args:
        snr_acc_dict (dict): 字典，键是 SNR 值 (int or float)，值是对应的准确率 (float, 0 到 1 之间)。
        model_name (str, optional): 模型名称，用于图表标题和图例。默认为 ""。
        font_filename (str, optional): 字体文件名。默认为 "NotoSansSC-VariableFont_wght.ttf"。
        font_subdir (str, optional): 字体文件所在子目录。默认为 "fonts"。
    """
    print("\n绘制 Accuracy vs. SNR 曲线...")

    # 输入验证
    if not snr_acc_dict or not isinstance(snr_acc_dict, dict):
        print("错误：提供的 SNR 准确率数据无效或为空，无法绘制曲线。")
        return

    font_prop = _load_font_properties(font_filename, font_subdir)
    # plt.rcParams['axes.unicode_minus'] = False # 已移至全局

    try:
        # 按 SNR 对字典进行排序，以确保绘图顺序正确
        sorted_snrs = sorted(snr_acc_dict.keys())
        # 将准确率转换为百分比显示
        accuracies = [snr_acc_dict[snr] * 100 for snr in sorted_snrs]

        plt.figure(figsize=(10, 6)) # 创建新的图形
        plot_label = f'{model_name} 模型准确率' if model_name else '模型准确率'
        plt.plot(sorted_snrs, accuracies, marker='o', linestyle='-', label=plot_label)

        # --- 设置标题和标签字体 ---
        title = '准确率 vs. 信噪比 (Accuracy vs. SNR)'
        if model_name:
            title += f' - {model_name}'

        plt.xlabel('信噪比 (SNR dB)', fontproperties=font_prop if font_prop else None)
        plt.ylabel('准确率 (%)', fontproperties=font_prop if font_prop else None)
        plt.title(title, fontproperties=font_prop if font_prop else None)

        plt.grid(True) # 显示网格
        plt.xticks(sorted_snrs) # 设置 X 轴刻度为所有 SNR 值
        plt.ylim(0, 105) # 设置 Y 轴范围略大于 100%
        plt.legend(prop=font_prop if font_prop else None) # 显示图例，应用字体
        plt.tight_layout()
        plt.show() # 显示图形

    except Exception as plot_e:
        print(f"错误：绘制 Accuracy vs. SNR 曲线时发生异常: {plot_e}")
        traceback.print_exc()


def plot_snr_cms(snr_cms_dict_np, display_labels, model_name="",
                 font_filename="NotoSansSC-VariableFont_wght.ttf", font_subdir="fonts",
                 save_individual_figs=False, individual_fig_dir="snr_cm_plots"):
    """
    绘制每个 SNR 下的混淆矩阵。
    首先绘制包含所有 SNR 混淆矩阵的网格图，然后可以选择性地
    为每个 SNR 绘制并保存/显示单独的混淆矩阵图。

    Args:
        snr_cms_dict_np (dict): 字典，键是 SNR 值，值是对应的混淆矩阵 (NumPy 数组)。
        display_labels (list): 类别标签名称列表 (字符串)。
        model_name (str, optional): 模型名称，用于标题和文件名。默认为 ""。
        font_filename (str, optional): 字体文件名。默认为 "NotoSansSC-VariableFont_wght.ttf"。
        font_subdir (str, optional): 字体文件所在子目录。默认为 "fonts"。
        save_individual_figs (bool, optional): 是否将每个 SNR 的混淆矩阵保存为单独的图像文件。
                                              默认为 False。
        individual_fig_dir (str, optional): 如果 save_individual_figs 为 True，
                                              指定保存单独图像的目录名。默认为 "snr_cm_plots"。
    """
    print("\n绘制不同 SNR 下的混淆矩阵...")

    # 输入验证
    if not snr_cms_dict_np or not isinstance(snr_cms_dict_np, dict):
        print("错误：提供的 SNR 混淆矩阵数据无效或为空，无法绘制。")
        return
    if not display_labels:
        print("错误：未提供 display_labels，无法绘制带标签的混淆矩阵。")
        return

    font_prop = _load_font_properties(font_filename, font_subdir)
    # plt.rcParams['axes.unicode_minus'] = False # 已移至全局

    snr_list_plot = sorted(snr_cms_dict_np.keys()) # 获取并排序 SNR 值
    n_snrs = len(snr_list_plot)

    if n_snrs == 0:
        print("错误：SNR 混淆矩阵字典为空。")
        return

    # --- 计算网格布局 ---
    n_cols = 5 # 每行最多显示 5 个子图
    n_rows = int(np.ceil(n_snrs / n_cols))

    # --- 1. 绘制网格大图 ---
    print("绘制包含所有 SNR 的网格混淆矩阵图...")
    try:
        fig_grid, axes_grid = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        # 处理 axes_grid 在只有一行/一列/一个图时的形状问题，确保可以按二维索引访问
        if n_snrs == 1:
            axes_grid = np.array([[axes_grid]])
        elif n_rows == 1:
            axes_grid = axes_grid.reshape(1, -1)
        elif n_cols == 1:
            axes_grid = axes_grid.reshape(-1, 1)

        ax_idx = 0
        for snr in snr_list_plot:
            if snr in snr_cms_dict_np and snr_cms_dict_np[snr] is not None:
                row_idx = ax_idx // n_cols
                col_idx = ax_idx % n_cols
                ax = axes_grid[row_idx, col_idx]

                cm_snr_np = snr_cms_dict_np[snr]
                # 检查混淆矩阵维度是否与标签匹配
                if cm_snr_np.shape[0] != len(display_labels):
                     print(f"警告：SNR = {snr} dB 的混淆矩阵维度 ({cm_snr_np.shape[0]}) 与 display_labels 长度 ({len(display_labels)}) 不匹配，跳过此子图。")
                     ax.set_title(f'SNR = {snr} dB (数据错误)')
                     ax.axis('off') # 隐藏轴
                     ax_idx += 1
                     continue

                disp = ConfusionMatrixDisplay(confusion_matrix=cm_snr_np, display_labels=display_labels)
                # 在网格图中，通常省略颜色条以节省空间
                disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=90, values_format='d', colorbar=False)

                ax.set_title(f'SNR = {snr} dB') # 设置子图标题

                # --- 手动设置刻度标签字体（如果加载成功） ---
                if font_prop:
                    try:
                        for label in ax.get_xticklabels():
                            label.set_fontproperties(font_prop)
                            label.set_fontsize(8) # 减小字体以防重叠
                        for label in ax.get_yticklabels():
                            label.set_fontproperties(font_prop)
                            label.set_fontsize(8)
                    except Exception as tick_e:
                        print(f"警告：为网格图 SNR={snr} 设置刻度标签字体时出错: {tick_e}")

                # --- 简化坐标轴标签，避免拥挤 ---
                if col_idx != 0: # 只在第一列显示 Y 轴标签
                    ax.set_ylabel('')
                    ax.set_yticklabels([]) # 隐藏刻度标签
                else:
                    ax.set_ylabel("真实标签", fontproperties=font_prop if font_prop else None)
                if row_idx != n_rows - 1: # 只在最后一行显示 X 轴标签
                     ax.set_xlabel('')
                     ax.set_xticklabels([]) # 隐藏刻度标签
                else:
                     ax.set_xlabel("预测标签", fontproperties=font_prop if font_prop else None)

                ax_idx += 1
            else:
                 # 如果某个 SNR 的数据缺失，也跳过并在网格中标注
                 row_idx = ax_idx // n_cols
                 col_idx = ax_idx % n_cols
                 ax = axes_grid[row_idx, col_idx]
                 ax.set_title(f'SNR = {snr} dB (无数据)')
                 ax.axis('off') # 隐藏轴
                 ax_idx += 1


        # --- 隐藏网格中多余的子图 ---
        while ax_idx < n_rows * n_cols:
            row_idx = ax_idx // n_cols
            col_idx = ax_idx % n_cols
            # 确保 fig_grid.delaxes 接收正确的 Axes 对象
            if n_rows * n_cols > 1:
                 fig_grid.delaxes(axes_grid[row_idx, col_idx])
            ax_idx += 1

        # --- 设置网格图的总标题 ---
        suptitle = '每个 SNR 下的混淆矩阵 (Confusion Matrix per SNR)'
        if model_name:
            suptitle += f' - {model_name}'
        plt.suptitle(suptitle, fontsize=16, fontproperties=font_prop if font_prop else None)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局，留出总标题空间
        plt.show() # 显示网格图
        print("网格混淆矩阵图绘制完成。")

    except Exception as grid_plot_e:
         print(f"错误：绘制网格混淆矩阵图时发生异常: {grid_plot_e}")
         traceback.print_exc()


    # --- 2. 绘制并显示/保存每个 SNR 的单独小图 ---
    print("\n绘制/保存每个 SNR 的单独混淆矩阵图...")
    if save_individual_figs:
        try:
            os.makedirs(individual_fig_dir, exist_ok=True) # 创建保存目录
            print(f"单独图像将保存到 '{individual_fig_dir}' 目录。")
        except OSError as e:
            print(f"错误：无法创建目录 '{individual_fig_dir}': {e}")
            save_individual_figs = False # 如果无法创建目录，则不保存

    for snr in snr_list_plot:
        if snr in snr_cms_dict_np and snr_cms_dict_np[snr] is not None:
            cm_snr_np = snr_cms_dict_np[snr]

            # 检查维度匹配（再次检查，以防万一）
            if cm_snr_np.shape[0] != len(display_labels):
                 print(f"警告：SNR = {snr} dB 的混淆矩阵维度与 display_labels 长度不匹配，跳过此单独图。")
                 continue

            fig_single, ax_single = plt.subplots(figsize=(8, 6)) # 为每个 SNR 创建新图形
            try:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_snr_np, display_labels=display_labels)
                # 单独图可以显示颜色条
                disp.plot(ax=ax_single, cmap=plt.cm.Blues, xticks_rotation=45, values_format='d', colorbar=True)

                # --- 设置标题和标签字体 ---
                title_single = '混淆矩阵 (Confusion Matrix)'
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
                        ax_single.set_xlabel("预测标签")
                        ax_single.set_ylabel("真实标签")
                else:
                     ax_single.set_xlabel("预测标签")
                     ax_single.set_ylabel("真实标签")

                plt.tight_layout()

                if save_individual_figs:
                    # 构建安全的文件名 (替换或移除特殊字符)
                    safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name) if model_name else "model"
                    # 处理负 SNR 的文件名
                    snr_str = f"neg{-snr}" if snr < 0 else f"{snr}"
                    filename = f"cm_{safe_model_name}_snr_{snr_str}dB.png"
                    filepath = os.path.join(individual_fig_dir, filename)
                    try:
                        plt.savefig(filepath)
                        print(f"已保存: {filepath}")
                    except Exception as save_e:
                        print(f"错误：保存图像 {filepath} 失败: {save_e}")
                    finally:
                         plt.close(fig_single) # 保存后关闭图形，防止显示过多窗口
                else:
                    plt.show() # 不保存则直接显示

            except Exception as single_plot_e:
                 print(f"错误：绘制 SNR = {snr} dB 的单独混淆矩阵时发生异常: {single_plot_e}")
                 traceback.print_exc()
                 plt.close(fig_single) # 出错时也关闭图形
            finally:
                # 确保在非保存模式下显示后关闭图形，或者在循环外统一处理
                if not save_individual_figs:
                     plt.close(fig_single) # 如果只是显示，显示后关闭


    print("单独/网格混淆矩阵图绘制完成。")