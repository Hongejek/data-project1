# my_experiment.py

# 1. 数据加载: 读取两个原始 CSV 文件
#    - train_data.csv          : 有标签数据 (列: text, target)
#    - test_data_unlabeled.csv : 无标签数据 (列: text), 用于最终预测提交
#    返回: X_all(全部文本) / y_all(全部标签) / X_test(无标签测试文本)
import pandas as pd


def load_data() -> tuple[list[str], list[int], list[str]]:
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')

    # 结构检查: 确认需要的列都存在
    assert {'text', 'target'} <= set(train_df.columns), "train_data.csv 缺少 text/target 列"
    assert 'text' in test_df.columns, "test_data_unlabeled.csv 缺少 text 列"

    # 缺失值检查: 若 text 为 NaN, astype(str) 会把它变成字符串 "nan" 混入数据, 需显式剔除
    n_missing = (
        int(train_df['text'].isna().sum())
        + int(train_df['target'].isna().sum())
        + int(test_df['text'].isna().sum())
    )
    if n_missing > 0:
        print(f"[加载检查] 发现 {n_missing} 条缺失记录, 已剔除")
    train_df = train_df.dropna(subset=['text', 'target'])
    test_df = test_df.dropna(subset=['text'])

    X_all = train_df['text'].astype(str).tolist()
    y_all = train_df['target'].astype(int).tolist()
    X_test = test_df['text'].astype(str).tolist()
    return X_all, y_all, X_test


# 2. 调用加载函数, 并检查加载结果
X_all, y_all, X_test = load_data()

print("--- 数据加载完成 ---")
print(f"有标签数据: {len(X_all)} 条 (列: text, target)")
print(f"无标签测试数据: {len(X_test)} 条 (列: text)")
print(f"标签类别数: {len(set(y_all))}, 取值范围: [{min(y_all)}, {max(y_all)}]")
print(f"重复文本数量: 有标签 {pd.Series(X_all).duplicated().sum()} 条 | 无标签 {pd.Series(X_test).duplicated().sum()} 条")

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
# print("第一个训练样本内容:")
# print(X_train[0])
# print(f"\n第一个训练样本的标签: {y_train[0]}")
# print("-" * 20)

# 打印第一个验证集样本和它的标签
# print("第一个验证样本内容:")
# print(X_val[0])
# print(f"\n第一个验证样本的标签: {y_val[0]}")
# print("-" * 20)

# 打印第一个需要你预测的测试样本
# print("第一个无标签测试样本内容:")
# print(X_test[0])
# print("\n" + "="*50)

# 5. 文本预处理: 清洗文本
#    清洗流水线: 去邮件头(保留 Subject) -> 去引用 -> 去引导句 -> 截签名
#                -> 删邮箱/Message-ID -> 去 uuencode 块 -> 空白规范化
import re


def clean_text(text: str) -> str:
    """对单条帖子文本做清洗, 返回清洗后的文本。"""
    # (1) 邮件头: 去掉 "Subject:" 前缀(保留主题词), 其余字段行整行删除
    text = re.sub(r'(?im)^subject:\s*', '', text)
    text = re.sub(
        r'(?im)^(from|to|organization|lines|nntp[\w-]*|distribution|reply-to|sender|in-reply-to|'
        r'message-id|references|xref|newsreader|x-[\w-]+|keywords|summary|expires|path|'
        r'followup-to|approved|supersedes|originator|date|content[\w-]*|mime-version):.*$',
        '', text,
    )

    # (2) 引用块: 删除 ">" / "=>" / "|" 开头的引用行, 以及 "In article ... writes:" 引导句
    text = re.sub(r'(?m)^[ \t]*(?:=>|>|\|).*$', '', text)
    text = re.sub(r'(?is)\bin article\b.{0,300}?\bwrit\w*:', ' ', text)

    # (3) 签名区: 在只有 "--"(可带空格) 的行处截断, 丢弃后面的签名内容
    text = re.split(r'(?m)^--[ \t]*$', text)[0]

    # (4) 邮箱地址与 Message-ID: 替换为空格 (无主题信息, 且可能泄漏发帖人身份)
    text = re.sub(r'[\w.+-]+@[\w.-]+', ' ', text)
    text = re.sub(r'<[^<>\s]+@[^<>\s]+>', ' ', text)

    # (5) uuencode 二进制块: 删除 "begin xxx" 标记行, 以及超长无空格字符行(编码数据)
    text = re.sub(r'(?m)^begin\s+[0-7]{3}.*$', '', text)
    text = re.sub(r'(?m)^\S{40,}$', '', text)

    # (6) 空白与控制字符规范化: 控制字符 -> 空格; 连续空白(含换行) -> 单个空格
    text = re.sub(r'[\x00-\x08\x0b-\x1f]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# 对训练集/验证集/测试集施加完全相同的清洗
raw_first = X_train[0]                                   # 存一条清洗前的样本用于对比
n_before = sum(len(s) for s in X_train) / len(X_train)   # 清洗前平均长度
X_train = [clean_text(s) for s in X_train]
X_val = [clean_text(s) for s in X_val]
X_test = [clean_text(s) for s in X_test]
n_after = sum(len(s) for s in X_train) / len(X_train)    # 清洗后平均长度

print("--- 文本清洗完成 (A 类清洗) ---")
print(f"训练集平均长度: {n_before:.0f} -> {n_after:.0f} 字符")
print("清洗前后对比 (训练集第 1 条, 前 200 字符):")
print("【清洗前】", raw_first[:200].replace('\n', ' ⏎ '), "...")
print("【清洗后】", X_train[0][:200], "...")
print("-" * 20)

# --- 在这里开始你的实验！ ---
# 现在，你可以使用三个数据集来进行特征提取、模型训练、验证和预测了:
#   X_train / y_train -> 训练集 (用于拟合)
#   X_val   / y_val   -> 验证集 (用于调参和模型选择)
#   X_test            -> 无标签测试集 (用于最终预测并提交)

# 1. 特征工程: TF-IDF 向量化 (文本 -> 数值矩阵)
#    关键原则: 词表与 IDF 只能从训练集学习(fit), 验证/测试集只做 transform, 防止数据泄漏
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)     # 全局词表: 按词频排序保留前 5000 个词 (与类别无关)
X_train_tfidf = vectorizer.fit_transform(X_train)   # 训练集: fit(学词表+IDF) + transform
X_val_tfidf = vectorizer.transform(X_val)           # 验证集: 只 transform (沿用训练集词表)
X_test_tfidf = vectorizer.transform(X_test)         # 测试集: 只 transform (沿用训练集词表)

# 向量化结果检查
print("--- TF-IDF 向量化完成 ---")
print(f"训练集矩阵: {X_train_tfidf.shape[0]} 条 x {X_train_tfidf.shape[1]} 维")
print(f"验证集矩阵: {X_val_tfidf.shape[0]} 条 x {X_val_tfidf.shape[1]} 维")
print(f"测试集矩阵: {X_test_tfidf.shape[0]} 条 x {X_test_tfidf.shape[1]} 维")
print(f"词表大小: {len(vectorizer.vocabulary_)}")
nnz = X_train_tfidf.getnnz()
print(f"训练矩阵非零值占比: {nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]):.3%}")
print("-" * 20)

# 2. 朴素贝叶斯 (MultinomialNB)
def MultinomialNB(X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf):
    """朴素贝叶斯 (MultinomialNB) 完整流程: 训练 -> 验证集评估 -> 测试集预测 -> 保存预测结果"""
    # 函数名与 sklearn 类名相同, 用别名导入避免冲突
    from sklearn.naive_bayes import MultinomialNB as NBClassifier
    from sklearn.metrics import accuracy_score

    # (1) 训练模型 (alpha: 平滑参数, NB 最主要的可调超参数)
    model = NBClassifier(alpha=1)
    print("--- 开始训练朴素贝叶斯模型 ---")
    model.fit(X_train_tfidf, y_train)
    print("模型训练完成!")

    # (2) 验证集评估 (模型选择/对比的依据)
    val_predictions = model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print("--- 朴素贝叶斯 (MultinomialNB) 验证集对比 ---")
    print(f"验证集准确率: {val_accuracy:.4f}")

    # (3) 对无标签测试集预测, 并保存提交文件
    predictions = model.predict(X_test_tfidf)
    pd.DataFrame(predictions).to_csv('predictions.csv', index=False, header=False)

    return val_accuracy


# 3. 支持向量机 (SVC)
def SVC(X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf):
    """支持向量机 (SVC) 完整流程: 训练 -> 验证集评估 -> 测试集预测 -> 保存预测结果"""
    # 函数名与 sklearn 类名相同, 用别名导入避免冲突
    from sklearn.svm import SVC as SVCClassifier
    from sklearn.metrics import accuracy_score

    # (1) 训练模型
    # 文本数据(高维稀疏)优先用线性核, 效果好且速度快
    # C 是正则化参数: 越小越容忍错分(边界更简单), 越大越严格要求拟合; 调参可试 [0.1, 1, 10, 100]
    model = SVCClassifier(kernel='linear', C=100, random_state=42)
    print("--- 开始训练支持向量机 (SVM) 模型 ---")
    model.fit(X_train_tfidf, y_train)
    print("模型训练完成!") 

    # (2) 验证集评估 (模型选择/对比的依据)
    val_predictions = model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print("--- 支持向量机 (SVC) 验证集对比 ---")
    print(f"验证集准确率: {val_accuracy:.4f}")

    # (3) 对无标签测试集预测, 并保存提交文件
    predictions = model.predict(X_test_tfidf)
    pd.DataFrame(predictions).to_csv('predictions.csv', index=False, header=False)

    return val_accuracy


# 4. 逻辑回归 (LogisticRegression)
def LogisticRegression(X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf):
    """逻辑回归 (LogisticRegression) 完整流程: 训练 -> 验证集评估 -> 测试集预测 -> 保存预测结果"""
    # 函数名与 sklearn 类名相同, 用别名导入避免冲突
    from sklearn.linear_model import LogisticRegression as LRClassifier
    from sklearn.metrics import accuracy_score

    # (1) 训练模型
    # C 是正则化强度的倒数: C 越小正则化越强(防过拟合), C 越大越弱; 调参可试 [0.1, 1, 10, 100]
    # max_iter: 最大迭代次数; 若出现 ConvergenceWarning 说明未收敛, 需增大 (如 2000)
    model = LRClassifier(C=10, max_iter=2000, random_state=42)
    print("--- 开始训练逻辑回归模型 ---")
    model.fit(X_train_tfidf, y_train)
    print("模型训练完成!")

    # (2) 验证集评估 (模型选择/对比的依据)
    val_predictions = model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print("--- 逻辑回归 (LogisticRegression) 验证集对比 ---")
    print(f"验证集准确率: {val_accuracy:.4f}")

    # (3) 对无标签测试集预测, 并保存提交文件
    predictions = model.predict(X_test_tfidf)
    pd.DataFrame(predictions).to_csv('predictions.csv', index=False, header=False)

    return val_accuracy


# 5. 多层感知机 (MLPClassifier)
def MLPClassifier(X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf):
    """多层感知机 (MLPClassifier) 完整流程: 训练 -> 验证集评估 -> 测试集预测 -> 保存预测结果"""
    # 函数名与 sklearn 类名相同, 用别名导入避免冲突
    from sklearn.neural_network import MLPClassifier as MLPModel
    from sklearn.metrics import accuracy_score

    # (1) 训练模型
    # hidden_layer_sizes=(100,): 一个含 100 个神经元的隐藏层; 可试 (50,) (200,) (100, 50)
    # alpha: L2 正则化项, 越大正则化越强; 可试 [0.0001, 0.001, 0.01, 0.1]
    # max_iter: 最大迭代次数; 若出现 ConvergenceWarning 说明未收敛, 需增大
    model = MLPModel(hidden_layer_sizes=(100,), alpha=0.0001, max_iter=300, random_state=42)
    print("--- 开始训练多层感知机 (MLP) 模型 ---")
    model.fit(X_train_tfidf, y_train)
    print("模型训练完成!")

    # (2) 验证集评估 (模型选择/对比的依据)
    val_predictions = model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print("--- 多层感知机 (MLP) 验证集对比 ---")
    print(f"验证集准确率: {val_accuracy:.4f}")

    # (3) 对无标签测试集预测, 并保存提交文件
    predictions = model.predict(X_test_tfidf)
    pd.DataFrame(predictions).to_csv('predictions.csv', index=False, header=False)

    return val_accuracy


if __name__ == "__main__":
    # 依次运行各算法的完整流程
    # MultinomialNB(X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf)
    #SVC(X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf)
    #LogisticRegression(X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf)
    MLPClassifier(X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf)