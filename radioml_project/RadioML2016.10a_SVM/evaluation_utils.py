# evaluation_utils.py
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def calculate_metrics(y_true_np, y_pred_np, target_names, labels=None):
    """
    计算整体评估指标。
    接收 NumPy 数组作为输入。

    Args:
        y_true_np (np.ndarray): 真实标签 (1D NumPy 数组, 整数)。
        y_pred_np (np.ndarray): 预测标签 (1D NumPy 数组, 整数)。
        target_names (list): 类别名称列表 (字符串)。
        labels (list, optional): 用于混淆矩阵的标签列表/范围。默认为 None。

    Returns:
        tuple: 包含 overall_accuracy (float), report (str), conf_matrix_overall_np (np.ndarray) 的元组。
    """
    print("\n计算整体性能指标...")
    if labels is None:
        labels = np.arange(len(target_names))

    # 整体准确率
    overall_accuracy = accuracy_score(y_true_np, y_pred_np)
    print(f"整体准确率 (Overall Accuracy): {overall_accuracy:.4f}")

    # 分类报告
    print("\n计算分类报告...")
    try:
        report = classification_report(y_true_np, y_pred_np, target_names=target_names, digits=4)
        print("\n分类报告 (Classification Report):\n")
        print(report)
    except Exception as report_e:
        print(f"错误：计算分类报告失败: {report_e}")
        report = "计算分类报告失败。"

    # 整体混淆矩阵 (使用 scikit-learn，输入 NumPy)
    print("\n计算整体混淆矩阵 (使用 scikit-learn)...")
    conf_matrix_overall_np = confusion_matrix(y_true_np, y_pred_np, labels=labels)
    print("\n整体混淆矩阵 (NumPy 数组):\n")
    print(conf_matrix_overall_np)

    return overall_accuracy, report, conf_matrix_overall_np

def calculate_snr_metrics(y_true_np, y_pred_np, snr_np, labels):
    """
    按 SNR 计算准确率和混淆矩阵。
    接收 NumPy 数组作为输入。

    Args:
        y_true_np (np.ndarray): 真实标签 (1D NumPy 数组)。
        y_pred_np (np.ndarray): 预测标签 (1D NumPy 数组)。
        snr_np (np.ndarray): 每个样本对应的 SNR 值 (1D NumPy 数组)。
        labels (list or np.ndarray): 用于混淆矩阵的标签列表/范围。

    Returns:
        tuple: 包含 snr_accuracy_dict (dict), snr_conf_matrix_np_dict (dict) 的元组。
    """
    print("\n按 SNR 计算性能指标...")
    snr_values_test = sorted(list(np.unique(snr_np)))
    print(f"测试集中的 SNR 值: {snr_values_test}")

    snr_accuracy_dict = {}       # 存储每个 SNR 的准确率
    snr_conf_matrix_np_dict = {} # 存储每个 SNR 的混淆矩阵 (NumPy)

    for snr in snr_values_test:
        # 找到当前 SNR 对应的样本索引 (使用 NumPy)
        indices = np.where(snr_np == snr)[0]

        if len(indices) > 0:
            # 提取当前 SNR 的真实标签和预测标签 (NumPy)
            labels_snr = y_true_np[indices]
            preds_snr = y_pred_np[indices]

            # 计算当前 SNR 的准确率
            acc = accuracy_score(labels_snr, preds_snr)
            snr_accuracy_dict[snr] = float(acc) # 存储为 Python float

            # 计算当前 SNR 的混淆矩阵
            cm_snr_np = confusion_matrix(labels_snr, preds_snr, labels=labels)
            snr_conf_matrix_np_dict[snr] = cm_snr_np

            print(f"SNR = {snr:>3d} dB: Accuracy = {acc:.4f}, Samples = {len(indices)}")
        else:
            print(f"SNR = {snr:>3d} dB: No samples found in test set.")

    return snr_accuracy_dict, snr_conf_matrix_np_dict