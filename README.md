# 项目文件使用说明

文本分类实验
运行环境：conda ；依赖安装：`pip install -r requirements.txt`。

| 文件 | 作用 | 如何使用 |
|---|---|---|
| `draft_main.py` | 主流程脚本：数据加载 → 文本清洗 → 划分 → TF-IDF → 四个模型的训练/验证/预测 | `python draft_main.py`；在文件末尾 `__main__` 中取消注释来选择要运行哪个模型 |
| `tuning.py` | 超参数扫描脚本（NB 的 alpha、SVC 的核与 C、LR 的 C、MLP 的隐藏层×alpha 网格） | `python tuning.py`，自动生成 `tuning_results.csv` |
| `plot_results.py` | 读取结果表，绘制"四模型 × 四组参数"共 16 柱的对比图 | `python plot_results.py`，生成 `model_comparison.png` |
| `tuning_results.csv` | 调参实验的完整结果表（模型 / 参数 / 验证集准确率） |  |
| `model_comparison.png` | 调参后各模型验证集准确率的对比柱状图 |  |
| `*_predictions.csv` | 各模型对测试集的预测结果（一行一个类别标签），运行哪个模型就生成哪个 | 从 `__main__` 中启用某模型运行后生成；最终提交时把最优模型的预测文件作为提交文件 |
| `predictions.csv` | 早期版本生成的预测文件（保留作参考） |  |
| `train_data.csv` | 有标签训练数据（7,368 条，10 类） | 脚本自动读取 |
| `test_data_unlabeled.csv` | 无标签测试数据（2,457 条，仅用于预测提交） | 脚本自动读取 |
| `TUNING.md` | 调参指南 |  |
| `requirements.txt` | Python 依赖清单（pandas / scikit-learn / matplotlib） | `pip install -r requirements.txt` |
| `.gitignore` | Git 忽略规则 |  |