# 基于机器学习的通信信号识别算法研究

这是一个用于本科毕业设计的项目，主旨是研究和实现基于传统机器学习与深度学习的通信信号调制方式识别算法。

## 简介

本项目利用多种机器学习模型对通信信号进行调制方式识别。项目主要包含以下三个核心部分：

1.**基于特征工程的传统机器学习方法**：
提取信号的统计特征、频谱特征以及高阶累积量 (HOC) 特征。使用支持向量机 (SVM) 和随机森林 (RF) 模型进行分类。

2.**基于深度学习的端到端方法**：
直接使用原始的I/Q信号数据，无需手动提取特征。构建并训练一个卷积神经网络-长短期记忆网络 (CNN-LSTM) 混合模型进行识别。

3.**性能评估与对比**：
在不同信噪比 (SNR) 条件下，对各个模型的识别准确率、混淆矩阵等性能指标进行评估和可视化。

## 数据集

本项目使用公开的 **RadioML 2016.10a** 数据集。该数据集包含了11种不同的调制类型，每种调制类型都在-20dB到18dB的信噪比范围内采样。

**调制类型包括**: 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM。

## 项目结构

radioml_project/

├── RadioML2016.10a_CNN_LSTM/

│   ├── radioml_preprocessing_cnn_lstm.ipynb  # 深度学习模型数据预处理

│   ├── train_cnn_lstm.ipynb                  # CNN-LSTM 模型训练与评估

│   ├── evaluation_utils.py                   # 评估工具函数

│   └── plotting_utils.py                     # 绘图工具函数

│

├── RadioML2016.10a_RF/

│   ├── radioml_preprocessing_svm_rf.ipynb    # 传统机器学习模型数据预处理

│   ├── train_evaluate_rf.ipynb               # 随机森林模型训练与评估

│   ├── evaluation_utils.py

│   └── plotting_utils.py

│

├── RadioML2016.10a_SVM/

│   ├── radioml_preprocessing_svm_rf.ipynb    # (与RF共用)

│   ├── train_evaluate_svm.ipynb              # SVM 模型训练与评估

│   ├── evaluation_utils.py

│   └── plotting_utils.py

│

└── model_snr_results/

├── snr_acc_cnn_lstm.pkl

├── snr_acc_rf.pkl

└── snr_acc_svm.pkl

## 使用说明

### 1. 环境配置

CNN-LSTM部分基于`pytorch-2.7.0`，SVM和RF部分基于`RAPIDS-25.04`。

### 2. 数据准备

1.下载 **RadioML 2016.10a** 数据集 (`RML2016.10a_dict.pkl`)。
2.将数据集文件放置在项目根目录下的 `data/` 文件夹中 (如果目录不存在，请创建)。

### 3. 运行流程

#### a. 数据预处理

* 对于 **SVM** 和 **RF** 模型，运行 `RadioML2016.10a_SVM/radioml_preprocessing_svm_rf.ipynb`。这将进行特征提取并生成 `processed_ml_features/` 目录。
* 对于 **CNN-LSTM** 模型，运行 `RadioML2016.10a_CNN_LSTM/radioml_preprocessing_cnn_lstm.ipynb`。这将对原始I/Q数据进行归一化和格式转换，并生成 `processed_cnn_lstm_data_powernorm/` 目录。

#### b. 模型训练与评估

预处理步骤完成后，分别运行以下notebook来进行模型训练和评估：

* **SVM**: `RadioML2016.10a_SVM/train_evaluate_svm.ipynb`
* **RF**: `RadioML2016.10a_RF/train_evaluate_rf.ipynb`
* **CNN-LSTM**: `RadioML2016.10a_CNN_LSTM/train_cnn_lstm.ipynb`

每个notebook都会训练相应的模型，并在测试集上进行评估。评估结果（如图表和性能指标）会被保存在 `plots/` 和 `model_snr_results/` 等目录中。

## 结果

模型的性能评估结果，包括在不同信噪比下的准确率曲线、整体混淆矩阵以及各信噪比下的混淆矩阵，将由训练评估notebook自动生成并保存。

* **准确率 vs. SNR 曲线**: 直观展示了模型在不同噪声水平下的识别能力。
* **混淆矩阵**: 详细展示了模型对各类调制信号的分类情况，有助于分析哪些信号类型更容易被混淆。

## 许可证

本项目采用 [MIT](https://choosealicense.com/licenses/mit/) 许可证。
