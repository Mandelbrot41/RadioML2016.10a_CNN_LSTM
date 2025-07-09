# plotting_utils.py
# 包含用于可视化调制识别模型评估结果的函数。
# 主要依赖 matplotlib 和 scikit-learn。
# 支持加载自定义字体以正确显示中文等非 ASCII 字符。

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.font_manager as fm
import os
import traceback # 导入 traceback 模块用于打印详细错误信息

# --- 全局设置 ---
# 确保 matplotlib 能够正确显示负号
plt.rcParams['axes.unicode_minus'] = False

def _load_font_properties(font_filename="NotoSansSC-Regular.ttf", font_subdir="fonts"):
    """
    辅助函数：加载字体属性对象 (matplotlib.font_manager.FontProperties)。
    用于在绘图时指定字体，特别是为了支持中文显示。

    Args:
        font_filename (str, optional): 字体文件的名称。
                                       默认为 "NotoSansSC-Regular.ttf"。
        font_subdir (str, optional): 字体文件相对于当前工作目录的子目录名。
                                     默认为 "fonts"。

    Returns:
        matplotlib.font_manager.FontProperties or None:
            加载成功则返回字体属性对象，否则返回 None。
    """
    font_prop = None
    try:
        current_dir = os.getcwd()
        font_path = os.path.join(current_dir, font_subdir, font_filename)
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            print(f"绘图：成功加载字体属性: {font_prop.get_name()} (来自 {font_path})")
        else:
            print(f"绘图错误：找不到字体文件 '{font_path}'。请确保字体文件存在于指定路径。")
    except Exception as e:
        print(f"绘图错误：加载字体属性时发生异常: {e}")
        traceback.print_exc()

    if not font_prop:
         print("绘图警告：未能加载指定字体，绘图中的中文可能无法正确显示。")
    return font_prop

def _get_safe_model_name_suffix(model_name, default_if_empty="model"):
    """辅助函数：生成用于文件名的安全模型名称字符串后缀。"""
    if model_name:
        safe_name = "".join(c if c.isalnum() else "_" for c in model_name)
        return f"_{safe_name}"
    elif default_if_empty:
        return f"_{default_if_empty}"
    return ""


def plot_overall_cm(cm_np, display_labels, model_name="",
                    font_filename="NotoSansSC-Regular.ttf", font_subdir="fonts",
                    save_fig=True, fig_dir="plots", fig_filename_base="overall_cm"):
    """
    绘制整体混淆矩阵。

    Args:
        cm_np (np.ndarray): 混淆矩阵 (NumPy 数组, 形状应为 N x N)。
        display_labels (list): 长度为 N 的类别标签名称列表 (字符串)。
        model_name (str, optional): 模型名称，用于图表标题和文件名。默认为 ""。
        font_filename (str, optional): 字体文件名。默认为 "NotoSansSC-Regular.ttf"。
        font_subdir (str, optional): 字体文件所在子目录。默认为 "fonts"。
        save_fig (bool, optional): 是否保存图像。默认为 True。
        fig_dir (str, optional): 保存图像的目录。默认为 "plots"。
        fig_filename_base (str, optional): 保存图像的基础文件名 (不含后缀)。
                                         默认为 "overall_cm"。
    """
    print("\n绘制整体混淆矩阵...")

    if cm_np is None or not isinstance(cm_np, np.ndarray) or cm_np.ndim != 2 or cm_np.shape[0] != cm_np.shape[1]:
        print("错误：提供的混淆矩阵数据无效 (必须是二维方阵 NumPy 数组)。")
        return
    if not display_labels or len(display_labels) != cm_np.shape[0]:
        print("错误：提供的 display_labels 无效或其长度与混淆矩阵维度不匹配。")
        return

    font_prop = _load_font_properties(font_filename, font_subdir)
    
    fig = plt.figure(figsize=(10, 8)) # 创建新的图形
    ax = fig.gca() # 获取当前 Axes

    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_np, display_labels=display_labels)
        disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45, values_format='d')

        title = '整体混淆矩阵 (Overall Confusion Matrix)'
        if model_name:
            title += f' - {model_name}'

        if font_prop:
             ax.set_title(title, fontproperties=font_prop)
             try:
                 for label in ax.get_xticklabels():
                     label.set_fontproperties(font_prop)
                 for label in ax.get_yticklabels():
                     label.set_fontproperties(font_prop)
                 ax.set_xlabel("预测标签", fontproperties=font_prop)
                 ax.set_ylabel("真实标签", fontproperties=font_prop)
             except Exception as tick_e:
                 print(f"警告：设置刻度标签字体时出错: {tick_e}")
                 ax.set_xlabel("预测标签")
                 ax.set_ylabel("真实标签")
        else:
             ax.set_title(title)
             ax.set_xlabel("预测标签")
             ax.set_ylabel("真实标签")

        plt.tight_layout()

        if save_fig:
            try:
                os.makedirs(fig_dir, exist_ok=True)
                model_suffix = _get_safe_model_name_suffix(model_name, default_if_empty="model" if not model_name and fig_filename_base == "overall_cm" else "")
                filename = f"{fig_filename_base}{model_suffix}.png"
                filepath = os.path.join(fig_dir, filename)
                plt.savefig(filepath)
                print(f"整体混淆矩阵已保存到: {filepath}")
                plt.close(fig)
            except Exception as save_e:
                print(f"错误：保存整体混淆矩阵图像失败: {save_e}")
                traceback.print_exc()
                if not plt.isinteractive(): # 如果不是交互模式，可能需要显示
                    plt.show()
                else:
                    plt.close(fig) # 否则关闭以避免问题
        else:
            plt.show()

    except Exception as plot_e:
        print(f"错误：绘制整体混淆矩阵时发生异常: {plot_e}")
        traceback.print_exc()
        plt.close(fig) # 确保出错时关闭图形


def plot_acc_vs_snr(snr_acc_dict, model_name="",
                    font_filename="NotoSansSC-Regular.ttf", font_subdir="fonts",
                    save_fig=True, fig_dir="plots", fig_filename_base="acc_vs_snr"):
    """
    绘制准确率 (Accuracy) vs. 信噪比 (SNR) 曲线。

    Args:
        snr_acc_dict (dict): 字典，键是 SNR 值 (int or float)，值是对应的准确率 (float, 0 到 1 之间)。
        model_name (str, optional): 模型名称，用于图表标题、图例和文件名。默认为 ""。
        font_filename (str, optional): 字体文件名。默认为 "NotoSansSC-Regular.ttf"。
        font_subdir (str, optional): 字体文件所在子目录。默认为 "fonts"。
        save_fig (bool, optional): 是否保存图像。默认为 True。
        fig_dir (str, optional): 保存图像的目录。默认为 "plots"。
        fig_filename_base (str, optional): 保存图像的基础文件名 (不含后缀)。
                                         默认为 "acc_vs_snr"。
    """
    print("\n绘制 Accuracy vs. SNR 曲线...")

    if not snr_acc_dict or not isinstance(snr_acc_dict, dict):
        print("错误：提供的 SNR 准确率数据无效或为空，无法绘制曲线。")
        return

    font_prop = _load_font_properties(font_filename, font_subdir)

    fig, ax = plt.subplots(figsize=(10, 6)) # 创建新的图形和 Axes

    try:
        sorted_snrs = sorted(snr_acc_dict.keys())
        accuracies = [snr_acc_dict[snr] * 100 for snr in sorted_snrs]

        plot_label = f'{model_name} 模型准确率' if model_name else '模型准确率'
        ax.plot(sorted_snrs, accuracies, marker='o', linestyle='-', label=plot_label)

        title = '准确率 vs. 信噪比 (Accuracy vs. SNR)'
        if model_name:
            title += f' - {model_name}'

        ax.set_xlabel('信噪比 (SNR dB)', fontproperties=font_prop if font_prop else None)
        ax.set_ylabel('准确率 (%)', fontproperties=font_prop if font_prop else None)
        ax.set_title(title, fontproperties=font_prop if font_prop else None)

        ax.grid(True)
        ax.set_xticks(sorted_snrs)
        ax.set_ylim(0, 105)
        ax.legend(prop=font_prop if font_prop else None)
        
        plt.tight_layout()

        if save_fig:
            try:
                os.makedirs(fig_dir, exist_ok=True)
                model_suffix = _get_safe_model_name_suffix(model_name, default_if_empty="model" if not model_name and fig_filename_base == "acc_vs_snr" else "")
                filename = f"{fig_filename_base}{model_suffix}.png"
                filepath = os.path.join(fig_dir, filename)
                plt.savefig(filepath)
                print(f"Accuracy vs. SNR 曲线已保存到: {filepath}")
                plt.close(fig)
            except Exception as save_e:
                print(f"错误：保存 Accuracy vs. SNR 曲线图像失败: {save_e}")
                traceback.print_exc()
                if not plt.isinteractive():
                    plt.show()
                else:
                    plt.close(fig)
        else:
            plt.show()

    except Exception as plot_e:
        print(f"错误：绘制 Accuracy vs. SNR 曲线时发生异常: {plot_e}")
        traceback.print_exc()
        plt.close(fig)


def plot_snr_cms(snr_cms_dict_np, display_labels, model_name="",
                 font_filename="NotoSansSC-Regular.ttf", font_subdir="fonts",
                 save_individual_figs=True, individual_fig_dir="snr_cm_plots",
                 save_grid_fig=True, grid_fig_dir="plots", grid_fig_filename_base="snr_cms_grid"):
    """
    绘制每个 SNR 下的混淆矩阵。
    首先绘制包含所有 SNR 混淆矩阵的网格图 (可保存)，然后可以选择性地
    为每个 SNR 绘制并保存/显示单独的混淆矩阵图 (默认保存)。

    Args:
        snr_cms_dict_np (dict): 字典，键是 SNR 值，值是对应的混淆矩阵 (NumPy 数组)。
        display_labels (list): 类别标签名称列表 (字符串)。
        model_name (str, optional): 模型名称，用于标题和文件名。默认为 ""。
        font_filename (str, optional): 字体文件名。默认为 "NotoSansSC-Regular.ttf"。
        font_subdir (str, optional): 字体文件所在子目录。默认为 "fonts"。
        save_individual_figs (bool, optional): 是否将每个 SNR 的混淆矩阵保存为单独的图像文件。
                                              默认为 True。
        individual_fig_dir (str, optional): 如果 save_individual_figs 为 True，
                                              指定保存单独图像的目录名。默认为 "snr_cm_plots"。
        save_grid_fig (bool, optional): 是否保存网格混淆矩阵图。默认为 True。
        grid_fig_dir (str, optional): 保存网格图的目录。默认为 "plots"。
        grid_fig_filename_base (str, optional): 保存网格图的基础文件名。默认为 "snr_cms_grid"。
    """
    print("\n绘制不同 SNR 下的混淆矩阵...")

    if not snr_cms_dict_np or not isinstance(snr_cms_dict_np, dict):
        print("错误：提供的 SNR 混淆矩阵数据无效或为空，无法绘制。")
        return
    if not display_labels:
        print("错误：未提供 display_labels，无法绘制带标签的混淆矩阵。")
        return

    font_prop = _load_font_properties(font_filename, font_subdir)
    snr_list_plot = sorted(snr_cms_dict_np.keys())
    n_snrs = len(snr_list_plot)

    if n_snrs == 0:
        print("错误：SNR 混淆矩阵字典为空。")
        return

    n_cols = 5
    n_rows = int(np.ceil(n_snrs / n_cols))

    print("处理包含所有 SNR 的网格混淆矩阵图...")
    fig_grid, axes_grid = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4.5)) # 稍微调整大小
    if n_snrs == 1:
        axes_grid = np.array([[axes_grid]])
    elif n_rows == 1:
        axes_grid = axes_grid.reshape(1, -1)
    elif n_cols == 1:
        axes_grid = axes_grid.reshape(-1, 1)

    try:
        ax_idx = 0
        for snr_val in snr_list_plot:
            if snr_val in snr_cms_dict_np and snr_cms_dict_np[snr_val] is not None:
                row_idx, col_idx = divmod(ax_idx, n_cols)
                # 检查索引是否越界 (当 n_snrs 不是 n_cols 的整数倍时，最后一行的 axes_grid 可能不足)
                if row_idx >= n_rows or col_idx >= n_cols :
                    print(f"警告: 索引 ({row_idx}, {col_idx}) 超出 axes_grid 范围。跳过 SNR={snr_val}")
                    ax_idx += 1
                    continue

                ax = axes_grid[row_idx, col_idx]
                cm_snr_np = snr_cms_dict_np[snr_val]

                if cm_snr_np.shape[0] != len(display_labels):
                     print(f"警告：SNR = {snr_val} dB 的混淆矩阵维度与标签长度不匹配，跳过此子图。")
                     ax.set_title(f'SNR = {snr_val} dB (数据错误)', fontproperties=font_prop)
                     ax.axis('off')
                     ax_idx += 1
                     continue

                disp = ConfusionMatrixDisplay(confusion_matrix=cm_snr_np, display_labels=display_labels)
                disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=90, values_format='d', colorbar=False)
                ax.set_title(f'SNR = {snr_val} dB', fontproperties=font_prop if font_prop else None)

                if font_prop:
                    try:
                        for label in ax.get_xticklabels():
                            label.set_fontproperties(font_prop); label.set_fontsize(8)
                        for label in ax.get_yticklabels():
                            label.set_fontproperties(font_prop); label.set_fontsize(8)
                    except Exception as tick_e:
                        print(f"警告：为网格图 SNR={snr_val} 设置刻度标签字体时出错: {tick_e}")

                if col_idx != 0:
                    ax.set_ylabel(''); ax.set_yticklabels([])
                else:
                    ax.set_ylabel("真实标签", fontproperties=font_prop if font_prop else None)
                if row_idx != n_rows - 1:
                     ax.set_xlabel(''); ax.set_xticklabels([])
                else:
                     ax.set_xlabel("预测标签", fontproperties=font_prop if font_prop else None)
                ax_idx += 1
            else:
                 row_idx, col_idx = divmod(ax_idx, n_cols)
                 if row_idx < n_rows and col_idx < n_cols:
                     ax = axes_grid[row_idx, col_idx]
                     ax.set_title(f'SNR = {snr_val} dB (无数据)', fontproperties=font_prop if font_prop else None)
                     ax.axis('off')
                 ax_idx += 1
        
        while ax_idx < n_rows * n_cols:
            row_idx, col_idx = divmod(ax_idx, n_cols)
            if row_idx < n_rows and col_idx < n_cols:
                 axes_grid[row_idx, col_idx].axis('off')
            ax_idx += 1

        suptitle = '每个 SNR 下的混淆矩阵 (Confusion Matrix per SNR)'
        if model_name:
            suptitle += f' - {model_name}'
        fig_grid.suptitle(suptitle, fontsize=16, fontproperties=font_prop if font_prop else None)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_grid_fig:
            try:
                os.makedirs(grid_fig_dir, exist_ok=True)
                model_suffix = _get_safe_model_name_suffix(model_name, default_if_empty="model" if not model_name and grid_fig_filename_base == "snr_cms_grid" else "")
                filename = f"{grid_fig_filename_base}{model_suffix}.png"
                filepath = os.path.join(grid_fig_dir, filename)
                plt.savefig(filepath)
                print(f"网格混淆矩阵图已保存到: {filepath}")
                plt.close(fig_grid)
            except Exception as save_e:
                print(f"错误：保存网格混淆矩阵图失败: {save_e}")
                traceback.print_exc()
                if not plt.isinteractive(): plt.show()
                else: plt.close(fig_grid)
        else:
            plt.show()
        print("网格混淆矩阵图处理完成。")

    except Exception as grid_plot_e:
         print(f"错误：绘制网格混淆矩阵图时发生异常: {grid_plot_e}")
         traceback.print_exc()
         plt.close(fig_grid)

    # --- 2. 绘制并显示/保存每个 SNR 的单独小图 ---
    if save_individual_figs: # 检查是否需要处理单独图像
        print("\n处理每个 SNR 的单独混淆矩阵图...")
        try:
            os.makedirs(individual_fig_dir, exist_ok=True)
            print(f"单独 SNR 混淆矩阵图像将保存到 '{individual_fig_dir}' 目录。")
        except OSError as e:
            print(f"错误：无法创建目录 '{individual_fig_dir}': {e}。将不保存单独图像。")
            save_individual_figs = False # 如果无法创建目录，则不保存

    # 仅在 save_individual_figs 为 True 或 需要显示（不保存网格图且不保存单独图时）时执行循环
    # 这个循环主要用于保存
    if save_individual_figs: # 主要控制是否进入此逻辑块
        for snr in snr_list_plot:
            if snr in snr_cms_dict_np and snr_cms_dict_np[snr] is not None:
                cm_snr_np = snr_cms_dict_np[snr]
                if cm_snr_np.shape[0] != len(display_labels):
                     print(f"警告：SNR = {snr} dB 的混淆矩阵维度与 display_labels 长度不匹配，跳过此单独图。")
                     continue

                fig_single, ax_single = plt.subplots(figsize=(8, 6))
                try:
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm_snr_np, display_labels=display_labels)
                    disp.plot(ax=ax_single, cmap=plt.cm.Blues, xticks_rotation=45, values_format='d', colorbar=True)

                    title_single = '混淆矩阵 (Confusion Matrix)'
                    if model_name: title_single += f' - {model_name}'
                    title_single += f' (SNR = {snr} dB)'
                    ax_single.set_title(title_single, fontproperties=font_prop if font_prop else None)

                    if font_prop:
                        try:
                            for label in ax_single.get_xticklabels(): label.set_fontproperties(font_prop)
                            for label in ax_single.get_yticklabels(): label.set_fontproperties(font_prop)
                            ax_single.set_xlabel("预测标签", fontproperties=font_prop)
                            ax_single.set_ylabel("真实标签", fontproperties=font_prop)
                        except Exception as tick_e:
                            print(f"警告：为单独图 SNR={snr} 设置字体时出错: {tick_e}")
                            ax_single.set_xlabel("预测标签"); ax_single.set_ylabel("真实标签")
                    else:
                         ax_single.set_xlabel("预测标签"); ax_single.set_ylabel("真实标签")
                    
                    plt.tight_layout()

                    # save_individual_figs 已经在此块顶部检查过了
                    safe_model_name_str = "".join(c if c.isalnum() else "_" for c in model_name) if model_name else "model"
                    snr_str = f"neg{-snr}" if snr < 0 else f"{snr}"
                    filename = f"cm_{safe_model_name_str}_snr_{snr_str}dB.png"
                    filepath = os.path.join(individual_fig_dir, filename)
                    try:
                        plt.savefig(filepath)
                        print(f"已保存: {filepath}")
                    except Exception as save_e:
                        print(f"错误：保存图像 {filepath} 失败: {save_e}")
                    finally:
                         plt.close(fig_single) # 总是关闭单个图像，无论保存成功与否

                except Exception as single_plot_e:
                     print(f"错误：绘制 SNR = {snr} dB 的单独混淆矩阵时发生异常: {single_plot_e}")
                     traceback.print_exc()
                     plt.close(fig_single) # 出错时也关闭图形
    elif not save_grid_fig: # 如果网格图不保存，且单独图也不（之前）保存，则需要逐个显示
        print("\n逐个显示每个 SNR 的单独混淆矩阵图 (因未选择保存)...")
        for snr in snr_list_plot:
            if snr in snr_cms_dict_np and snr_cms_dict_np[snr] is not None:
                # 此处省略与上面几乎相同的绘图代码，仅将保存部分改为 plt.show() 和 plt.close())
                # save_individual_figs 默认为 True，此分支不会执行。
                print(f"显示 SNR = {snr} dB 的混淆矩阵 (未保存)。")
                temp_fig, temp_ax = plt.subplots(figsize=(8,6))
                disp = ConfusionMatrixDisplay(confusion_matrix=snr_cms_dict_np[snr], display_labels=display_labels)
                disp.plot(ax=temp_ax, cmap=plt.cm.Blues, xticks_rotation=45, values_format='d', colorbar=True)
                title_single = f'混淆矩阵 - {model_name} (SNR = {snr} dB)' if model_name else f'混淆矩阵 (SNR = {snr} dB)'
                temp_ax.set_title(title_single, fontproperties=font_prop if font_prop else None)

                plt.tight_layout()
                plt.show()
                plt.close(temp_fig)


    print("所有 SNR 相关混淆矩阵图处理完成。")