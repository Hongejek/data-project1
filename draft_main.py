# my_experiment.py

# 1. 从我们提供的帮助脚本中导入加载函数
import pandas as pd

def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled

# 2. 调用函数来获取数据
#    这个函数会自动读取 .csv 文件并返回你需要的所有内容
X_all, y_all, X_test = load_data()

# 3. 划分验证集: 从有标签数据中分层抽取 20% 作为验证集
#    最终得到三个数据集:
#      - 训练集 X_train / y_train (80%): 用于拟合 TF-IDF 词表和训练模型
#      - 验证集 X_val   / y_val   (20%): 用于调参和模型选择
#      - 测试集 X_test (无标签): 来自 test_data_unlabeled.csv, 仅用于生成最终预测提交
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all,
    test_size=0.2,        # 验证集占 20%
    stratify=y_all,       # 分层抽样, 保持 10 个类别的比例一致
    random_state=42,      # 固定随机种子, 保证结果可复现
)

# 4. (验证步骤) 检查一下数据是否加载、划分成功
print("--- 数据加载与划分成功 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"训练集标签数量: {len(y_train)}")
print(f"验证集样本数量: {len(X_val)}")
print(f"验证集标签数量: {len(y_val)}")
print(f"无标签测试集样本数量: {len(X_test)}")
print("-" * 20)

# 检查训练集与验证集的类别分布是否一致 (分层抽样应保证比例相同)
print("训练集标签分布:")
print(pd.Series(y_train).value_counts().sort_index().to_string())
print("验证集标签分布:")
print(pd.Series(y_val).value_counts().sort_index().to_string())
print("-" * 20)

# 打印第一个训练样本和它的标签，感受一下数据
print("第一个训练样本内容:")
print(X_train[0])
print(f"\n第一个训练样本的标签: {y_train[0]}")
print("-" * 20)

# 打印第一个验证集样本和它的标签
print("第一个验证样本内容:")
print(X_val[0])
print(f"\n第一个验证样本的标签: {y_val[0]}")
print("-" * 20)

# 打印第一个需要你预测的测试样本
print("第一个无标签测试样本内容:")
print(X_test[0])
print("\n" + "="*50)

# --- 在这里开始你的实验！ ---
# 现在，你可以使用三个数据集来进行特征提取、模型训练、验证和预测了:
#   X_train / y_train -> 训练集 (用于拟合)
#   X_val   / y_val   -> 验证集 (用于调参和模型选择)
#   X_test            -> 无标签测试集 (用于最终预测并提交)

# 举例：
# 1. 创建TF-IDF向量化器（注意: 只能在训练集上 fit, 防止数据泄漏）
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)   # 训练集: fit + transform
X_val_tfidf = vectorizer.transform(X_val)           # 验证集: 只 transform
X_test_tfidf = vectorizer.transform(X_test)         # 测试集: 只 transform

# 2. 训练一个模型...
from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)

# 3. 在验证集上评估模型效果，用于调参和模型选择...
from sklearn.metrics import accuracy_score
val_predictions = svm_model.predict(X_val_tfidf)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"验证集准确率: {val_accuracy:.4f}")

# 4. 对无标签测试集进行预测...
predictions = svm_model.predict(X_test_tfidf)

# 5. 保存你的预测结果...
pd.DataFrame(predictions).to_csv('predictions.csv', index=False, header=False)