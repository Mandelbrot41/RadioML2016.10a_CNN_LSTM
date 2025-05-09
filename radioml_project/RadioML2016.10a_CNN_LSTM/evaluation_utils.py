# evaluation_utils.py

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import traceback

def calculate_metrics(y_true_np, y_pred_np, target_names, labels=None):
    """
    计算并打印整体评估指标（准确率、分类报告、混淆矩阵）。
    ... (文档字符串保持不变) ...
    """
    print("\n计算整体性能指标...")

    # --- 输入验证 ---
    # 检查必需的 NumPy 数组是否有效
    if y_true_np is None or not isinstance(y_true_np, np.ndarray) or y_true_np.size == 0:
        print("错误：y_true_np 无效或为空。")
        return 0.0, "错误：y_true_np 无效或为空。", None
    if y_pred_np is None or not isinstance(y_pred_np, np.ndarray) or y_pred_np.size == 0:
        print("错误：y_pred_np 无效或为空。")
        return 0.0, "错误：y_pred_np 无效或为空。", None
    if y_true_np.shape != y_pred_np.shape:
        print("错误：y_true_np 和 y_pred_np 的形状不匹配。")
        return 0.0, "错误：y_true_np 和 y_pred_np 的形状不匹配。", None
    # 检查 target_names
    if target_names is None or not isinstance(target_names, list) or len(target_names) == 0:
        print("错误：target_names 无效或为空。")
        # 仍然可以计算准确率和矩阵（不带标签）
        overall_accuracy = accuracy_score(y_true_np, y_pred_np)
        conf_matrix_overall_np = confusion_matrix(y_true_np, y_pred_np)
        print(f"整体准确率 (Overall Accuracy): {overall_accuracy:.4f}")
        print("\n整体混淆矩阵 (NumPy 数组):\n")
        print(conf_matrix_overall_np)
        return overall_accuracy, "错误：target_names 为空", conf_matrix_overall_np
    # --- 验证结束 ---

    # --- 确定用于报告和混淆矩阵的标签列表 ---
    if labels is None:
        num_classes = len(target_names)
        explicit_labels = np.arange(num_classes)
    else:
        explicit_labels = np.asarray(labels) # 转为 NumPy 数组

    # --- 计算整体准确率 ---
    try:
        overall_accuracy = accuracy_score(y_true_np, y_pred_np)
        print(f"整体准确率 (Overall Accuracy): {overall_accuracy:.4f}")
    except Exception as acc_e:
        print(f"计算准确率时出错: {acc_e}")
        overall_accuracy = 0.0

    # --- 计算分类报告 ---
    report = None
    try:
        if len(target_names) != len(explicit_labels):
             raise ValueError(f"target_names (len {len(target_names)}) 和 explicit_labels (len {len(explicit_labels)}) 的长度必须匹配。")
        report = classification_report(y_true_np, y_pred_np,
                                      labels=explicit_labels,
                                      target_names=target_names,
                                      digits=4,
                                      zero_division=0)
        print("\n分类报告 (Classification Report):\n")
        print(report)
    except ValueError as ve:
        print(f"错误：计算分类报告失败 (ValueError): {ve}")
        print("  请检查:")
        print("  1. 'labels' 和 'target_names' 的长度是否一致。")
        print("  2. 'target_names' 中的名称是否与 'labels' 中的整数标签正确对应。")
        print("  3. y_true_np 和 y_pred_np 中是否包含 'labels' 中未定义的标签值。")
        print(f"  y_true unique labels in data: {np.unique(y_true_np)}")
        print(f"  y_pred unique labels in data: {np.unique(y_pred_np)}")
        print(f"  Provided explicit_labels for report: {explicit_labels}")
        print(f"  Provided target_names for report: {target_names}")
        report = "计算分类报告失败 (ValueError)。"
    except Exception as report_e:
        print(f"错误：计算分类报告失败 (其他异常): {report_e}")
        traceback.print_exc()
        report = f"计算分类报告失败 ({type(report_e).__name__})。"


    # --- 计算整体混淆矩阵 ---
    print("\n计算整体混淆矩阵 (使用 scikit-learn)...")
    conf_matrix_overall_np = None
    try:
        conf_matrix_overall_np = confusion_matrix(y_true_np, y_pred_np, labels=explicit_labels)
        print("\n整体混淆矩阵 (NumPy 数组):\n")
        print(conf_matrix_overall_np)
    except Exception as cm_e:
        print(f"错误：计算混淆矩阵失败: {cm_e}")
        traceback.print_exc()


    return overall_accuracy, report, conf_matrix_overall_np

def calculate_snr_metrics(y_true_np, y_pred_np, snr_np, labels):
    """
    计算并打印每个 SNR 值下的准确率和混淆矩阵。
    ... (文档字符串保持不变) ...
    """
    print("\n按 SNR 计算性能指标...")

    # --- 输入验证 ---
    if y_true_np is None or not isinstance(y_true_np, np.ndarray) or y_true_np.size == 0:
        print("错误：y_true_np 无效或为空。")
        return {}, {}
    if y_pred_np is None or not isinstance(y_pred_np, np.ndarray) or y_pred_np.size == 0:
        print("错误：y_pred_np 无效或为空。")
        return {}, {}
    if snr_np is None or not isinstance(snr_np, np.ndarray) or snr_np.size == 0:
        print("错误：snr_np 无效或为空。")
        return {}, {}
    if not (len(y_true_np) == len(y_pred_np) == len(snr_np)):
        print(f"错误：输入的数组长度不匹配: y_true({len(y_true_np)}), y_pred({len(y_pred_np)}), snr({len(snr_np)})")
        return {}, {}
    # 明确检查 labels 是否为 None 或空列表/数组
    if labels is None or (hasattr(labels, '__len__') and len(labels) == 0):
         print("错误：未提供有效的 'labels' 列表，无法计算混淆矩阵。")
         return {}, {}
    # --- 验证结束 ---

    # 获取测试集中所有唯一的 SNR 值并排序
    snr_values_test = sorted(list(np.unique(snr_np)))
    print(f"测试集中的 SNR 值: {snr_values_test}")

    snr_accuracy_dict = {}       # 存储 {SNR: Accuracy}
    snr_conf_matrix_np_dict = {} # 存储 {SNR: Confusion Matrix}

    # 确保 labels 是 NumPy 数组
    explicit_labels_snr = np.asarray(labels)

    # --- 遍历每个 SNR 值 ---
    for snr in snr_values_test:
        indices = np.where(snr_np == snr)[0]

        if len(indices) > 0:
            labels_snr = y_true_np[indices]
            preds_snr = y_pred_np[indices]

            try:
                acc = accuracy_score(labels_snr, preds_snr)
                snr_accuracy_dict[snr] = float(acc)
                cm_snr_np = confusion_matrix(labels_snr, preds_snr, labels=explicit_labels_snr)
                snr_conf_matrix_np_dict[snr] = cm_snr_np
                print(f"SNR = {snr:>3d} dB: Accuracy = {acc:.4f}, Samples = {len(indices)}")
            except Exception as e:
                print(f"计算 SNR = {snr} dB 的指标时出错: {e}")
                snr_accuracy_dict[snr] = 0.0
                snr_conf_matrix_np_dict[snr] = None # 标记为 None
        else:
            print(f"SNR = {snr:>3d} dB: 未找到样本。")
            snr_accuracy_dict[snr] = 0.0
            snr_conf_matrix_np_dict[snr] = None

    return snr_accuracy_dict, snr_conf_matrix_np_dict