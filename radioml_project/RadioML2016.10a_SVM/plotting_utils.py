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
    辅助函数：加载字体属性对象。
    """
    font_prop = None
    try:
        current_dir = os.getcwd()
        font_path = os.path.join(current_dir, font_subdir, font_filename)
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            print(f"绘图：成功加载字体属性: {font_prop.get_name()} (来自 {font_path})")
        else:
            print(f"绘图错误：找不到字体文件 '{font_path}'")
    except Exception as e:
        print(f"绘图错误：加载字体属性时出错: {e}")
        traceback.print_exc() # 打印更详细的错误信息
    return font_prop

def _get_safe_model_name_suffix(model_name, default_if_empty="model"):
    """辅助函数：生成用于文件名的安全模型名称字符串后缀。"""
    if model_name:
        safe_name = "".join(c if c.isalnum() else "_" for c in model_name)
        return f"_{safe_name}"
    elif default_if_empty: # 仅当 model_name 为空且需要默认值时使用
        return f"_{default_if_empty}"
    return ""


def plot_overall_cm(cm_np, display_labels, model_name="",
                    font_filename="NotoSansSC-Regular.ttf", font_subdir="fonts",
                    save_fig=True, fig_dir="plots", fig_filename_base="overall_cm"):
    """
    绘制整体混淆矩阵。

    Args:
        cm_np (np.ndarray): 混淆矩阵 (NumPy 数组)。
        display_labels (list): 类别标签名称列表。
        model_name (str, optional): 模型名称，用于标题和文件名。默认为 ""。
        font_filename (str, optional): 字体文件名。默认为 "NotoSansSC-Regular.ttf"。
        font_subdir (str, optional): 字体文件所在子目录。默认为 "fonts"。
        save_fig (bool, optional): 是否保存图像。默认为 True。
        fig_dir (str, optional): 保存图像的目录。默认为 "plots"。
        fig_filename_base (str, optional): 保存图像的基础文件名 (不含后缀)。
                                         默认为 "overall_cm"。
    """
    print("\n绘制整体混淆矩阵...")
    font_prop = _load_font_properties(font_filename, font_subdir)

    if cm_np is None:
        print("错误：混淆矩阵数据为空。")
        return
    if not display_labels:
        print("错误：未提供 display_labels。")
        return
    
    # 字体加载失败不会完全阻止绘图，但中文会受影响
    if not font_prop:
        print("警告：未能加载字体属性，绘图中的中文可能无法正确显示。")

    fig, ax = plt.subplots(figsize=(10, 8)) # 创建 Figure 和 Axes

    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_np, display_labels=display_labels)
        disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45, values_format='d')

        title = '整体混淆矩阵 (Overall Confusion Matrix)'
        if model_name:
            title += f' - {model_name}'
        
        ax.set_title(title, fontproperties=font_prop if font_prop else None)
        if font_prop:
            try:
                for label in ax.get_xticklabels():
                    label.set_fontproperties(font_prop)
                for label in ax.get_yticklabels():
                    label.set_fontproperties(font_prop)
                ax.set_xlabel("预测标签", fontproperties=font_prop)
                ax.set_ylabel("真实标签", fontproperties=font_prop)
            except Exception as tick_e:
                print(f"警告：设置刻度标签字体时出错: {tick_e}")
                ax.set_xlabel("预测标签") # Fallback
                ax.set_ylabel("真实标签") # Fallback
        else:
            ax.set_xlabel("预测标签")
            ax.set_ylabel("真实标签")

        plt.tight_layout()

        if save_fig:
            try:
                os.makedirs(fig_dir, exist_ok=True)
                model_suffix = _get_safe_model_name_suffix(model_name, default_if_empty="" if model_name else "") # 如果model_name为空，则不添加"_model"
                filename = f"{fig_filename_base}{model_suffix}.png"
                filepath = os.path.join(fig_dir, filename)
                plt.savefig(filepath)
                print(f"整体混淆矩阵已保存到: {filepath}")
                plt.close(fig)
            except Exception as save_e:
                print(f"错误：保存整体混淆矩阵图像失败: {save_e}")
                traceback.print_exc()
                if not plt.isinteractive(): plt.show() # 尝试显示，如果不在交互模式
                else: plt.close(fig)
        else:
            plt.show()

    except Exception as plot_e:
        print(f"错误：绘制整体混淆矩阵时发生异常: {plot_e}")
        traceback.print_exc()
        plt.close(fig)


def plot_acc_vs_snr(snr_acc_dict, model_name="",
                    font_filename="NotoSansSC-Regular.ttf", font_subdir="fonts",
                    save_fig=True, fig_dir="plots", fig_filename_base="acc_vs_snr"):
    """
    绘制 Accuracy vs. SNR 曲线。

    Args:
        snr_acc_dict (dict): 包含 {snr: accuracy} 的字典。accuracy 应为 0-100 的值。
        model_name (str, optional): 模型名称，用于标题、图例和文件名。默认为 ""。
        font_filename (str, optional): 字体文件名。默认为 "NotoSansSC-Regular.ttf"。
        font_subdir (str, optional): 字体文件所在子目录。默认为 "fonts"。
        save_fig (bool, optional): 是否保存图像。默认为 True。
        fig_dir (str, optional): 保存图像的目录。默认为 "plots"。
        fig_filename_base (str, optional): 保存图像的基础文件名 (不含后缀)。
                                         默认为 "acc_vs_snr"。
    """
    print("\n绘制 Accuracy vs. SNR 曲线...")
    font_prop = _load_font_properties(font_filename, font_subdir)

    if not snr_acc_dict:
        print("未能计算 SNR 准确率数据，无法绘制曲线。")
        return

    if not font_prop:
        print("警告：未能加载字体属性，绘图中的中文可能无法正确显示。")

    sorted_snrs = sorted(snr_acc_dict.keys())
    accuracies = [snr_acc_dict[snr] for snr in sorted_snrs] 

    fig, ax = plt.subplots(figsize=(10, 6)) # 创建 Figure 和 Axes

    try:
        plot_label = f'{model_name} 模型准确率' if model_name else '模型准确率'
        ax.plot(sorted_snrs, accuracies, marker='o', linestyle='-', label=plot_label)

        title = '准确率 vs. 信噪比 (Accuracy vs. SNR)'
        if model_name:
            title += f' - {model_name}'

        ax.set_xlabel('信噪比 (SNR dB)', fontproperties=font_prop if font_prop else None) # X轴标签通常包含单位
        ax.set_ylabel('准确率 (%)', fontproperties=font_prop if font_prop else None)
        ax.set_title(title, fontproperties=font_prop if font_prop else None)

        ax.grid(True)
        ax.set_xticks(sorted_snrs)
        ax.set_ylim(0, 105) # Y 轴范围
        ax.legend(prop=font_prop if font_prop else None)
        
        plt.tight_layout()

        if save_fig:
            try:
                os.makedirs(fig_dir, exist_ok=True)
                model_suffix = _get_safe_model_name_suffix(model_name, default_if_empty="" if model_name else "")
                filename = f"{fig_filename_base}{model_suffix}.png"
                filepath = os.path.join(fig_dir, filename)
                plt.savefig(filepath)
                print(f"Accuracy vs. SNR 曲线已保存到: {filepath}")
                plt.close(fig)
            except Exception as save_e:
                print(f"错误：保存 Accuracy vs. SNR 曲线图像失败: {save_e}")
                traceback.print_exc()
                if not plt.isinteractive(): plt.show()
                else: plt.close(fig)
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
    会先绘制包含所有 SNR 的网格大图 (可保存)，然后为每个 SNR 绘制并 (默认)保存单独的小图。

    Args:
        snr_cms_dict_np (dict): 包含 {snr: cm_np_array} 的字典。
        display_labels (list): 类别标签名称列表。
        model_name (str, optional): 模型名称，用于标题和文件名。默认为 ""。
        font_filename (str, optional): 字体文件名。默认为 "NotoSansSC-Regular.ttf"。
        font_subdir (str, optional): 字体文件所在子目录。默认为 "fonts"。
        save_individual_figs (bool, optional): 是否保存每个 SNR 的单独图像。默认为 True。
        individual_fig_dir (str, optional): 保存单独图像的目录名。默认为 "snr_cm_plots"。
        save_grid_fig (bool, optional): 是否保存网格混淆矩阵图。默认为 True。
        grid_fig_dir (str, optional): 保存网格图的目录。默认为 "plots"。
        grid_fig_filename_base (str, optional): 保存网格图的基础文件名。默认为 "snr_cms_grid"。
    """
    print("\n绘制不同 SNR 下的混淆矩阵...")
    font_prop = _load_font_properties(font_filename, font_subdir)

    if not snr_cms_dict_np:
        print("没有找到有效的 SNR 混淆矩阵数据来绘制。")
        return
    if not display_labels:
        print("错误：未提供 display_labels。")
        return
        
    if not font_prop:
        print("警告：未能加载字体属性，绘图中的中文可能无法正确显示。")

    snr_list_plot = sorted(snr_cms_dict_np.keys())
    n_snrs = len(snr_list_plot)
    if n_snrs == 0:
        print("没有有效的 SNR 数据。")
        return

    n_cols = 5
    n_rows = int(np.ceil(n_snrs / n_cols))

    # --- 1. 绘制网格大图 ---
    print("处理包含所有 SNR 的网格混淆矩阵图...")
    fig_grid, axes_grid = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4.5))
    if n_snrs == 1: axes_grid = np.array([[axes_grid]])
    elif n_rows == 1: axes_grid = axes_grid.reshape(1, -1)
    elif n_cols == 1: axes_grid = axes_grid.reshape(-1, 1)
    
    try:
        ax_idx = 0
        for snr_val in snr_list_plot:
            if snr_val in snr_cms_dict_np and snr_cms_dict_np[snr_val] is not None:
                row_idx, col_idx = divmod(ax_idx, n_cols)
                if row_idx >= n_rows or col_idx >= n_cols : continue # 安全检查
                
                ax = axes_grid[row_idx, col_idx]
                cm_snr_np = snr_cms_dict_np[snr_val]

                if cm_snr_np.shape[0] != len(display_labels):
                     print(f"警告：SNR = {snr_val} dB 的混淆矩阵维度与标签长度不匹配，跳过此子图。")
                     ax.set_title(f'SNR = {snr_val} dB (数据错误)', fontproperties=font_prop)
                     ax.axis('off'); ax_idx += 1; continue

                disp = ConfusionMatrixDisplay(confusion_matrix=cm_snr_np, display_labels=display_labels)
                disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=90, values_format='d', colorbar=False)
                ax.set_title(f'SNR = {snr_val} dB', fontproperties=font_prop if font_prop else None)

                if font_prop:
                    try:
                        for label in ax.get_xticklabels(): label.set_fontproperties(font_prop); label.set_fontsize(8)
                        for label in ax.get_yticklabels(): label.set_fontproperties(font_prop); label.set_fontsize(8)
                    except Exception as tick_e: print(f"警告：为网格图 SNR={snr_val} 设置刻度标签字体时出错: {tick_e}")
                
                ax.set_ylabel("真实标签" if col_idx == 0 else "", fontproperties=font_prop if font_prop else None)
                if col_idx !=0 : ax.set_yticklabels([])
                ax.set_xlabel("预测标签" if row_idx == n_rows - 1 else "", fontproperties=font_prop if font_prop else None)
                if row_idx != n_rows -1 : ax.set_xticklabels([])
                ax_idx += 1
            else: # 处理数据缺失的情况
                row_idx, col_idx = divmod(ax_idx, n_cols)
                if row_idx < n_rows and col_idx < n_cols:
                    ax = axes_grid[row_idx, col_idx]
                    ax.set_title(f'SNR = {snr_val} dB (无数据)', fontproperties=font_prop if font_prop else None)
                    ax.axis('off')
                ax_idx += 1
        
        while ax_idx < n_rows * n_cols: # 隐藏多余子图
            row_idx, col_idx = divmod(ax_idx, n_cols)
            if row_idx < n_rows and col_idx < n_cols: axes_grid[row_idx, col_idx].axis('off')
            ax_idx += 1

        suptitle = '每个 SNR 下的混淆矩阵 (网格图)'
        if model_name: suptitle += f' - {model_name}'
        fig_grid.suptitle(suptitle, fontsize=16, fontproperties=font_prop if font_prop else None)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_grid_fig:
            try:
                os.makedirs(grid_fig_dir, exist_ok=True)
                model_suffix = _get_safe_model_name_suffix(model_name, default_if_empty="" if model_name else "")
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
         plt.close(fig_grid) # 确保关闭

    # --- 2. 绘制并显示/保存每个 SNR 的单独小图 ---
    if save_individual_figs:
        print("\n处理每个 SNR 的单独混淆矩阵图 (保存模式)...")
        try:
            os.makedirs(individual_fig_dir, exist_ok=True)
            print(f"单独 SNR 混淆矩阵图像将保存到 '{individual_fig_dir}' 目录。")
        except OSError as e:
            print(f"错误：无法创建目录 '{individual_fig_dir}': {e}。将不保存单独图像。")
            save_individual_figs = False # 出错则不保存

    # 仅当 save_individual_figs 为 True (现在是默认) 时进入此循环执行保存
    # 如果不保存，则不进入此循环，因为网格图已经显示或保存了
    if save_individual_figs:
        for snr in snr_list_plot:
            if snr in snr_cms_dict_np and snr_cms_dict_np[snr] is not None:
                cm_snr_np = snr_cms_dict_np[snr]
                if cm_snr_np.shape[0] != len(display_labels):
                     print(f"警告：SNR = {snr} dB 的混淆矩阵维度与标签长度不匹配，跳过此单独图。")
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
                            ax_single.set_xlabel("预测标签"); ax_single.set_ylabel("真实标签") # Fallback
                    else:
                         ax_single.set_xlabel("预测标签"); ax_single.set_ylabel("真实标签")
                    
                    plt.tight_layout()

                    # 文件名使用原始脚本中的"model"作为空模型名的占位符，保持一致性
                    safe_model_name_str = "".join(c if c.isalnum() else "_" for c in model_name) if model_name else "model"
                    snr_str = f"neg{-snr}" if snr < 0 else f"{snr}" # 原文件名逻辑
                    filename = f"cm_{safe_model_name_str}_snr_{snr_str}dB.png" # 原文件名逻辑
                    filepath = os.path.join(individual_fig_dir, filename)
                    try:
                        plt.savefig(filepath)
                        print(f"已保存: {filepath}")
                    except Exception as save_e:
                        print(f"错误：保存图像 {filepath} 失败: {save_e}")
                    # 无论是否保存成功，都关闭，因为这是在循环中为每个 SNR 创建的图
                    plt.close(fig_single) 

                except Exception as single_plot_e:
                     print(f"错误：绘制 SNR = {snr} dB 的单独混淆矩阵时发生异常: {single_plot_e}")
                     traceback.print_exc()
                     plt.close(fig_single) # 出错时也关闭图形
            # 如果 snr 数据无效或为空，则不执行任何操作
    
    print("所有 SNR 相关混淆矩阵图处理完成。")