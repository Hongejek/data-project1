import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from draft_main import clean_text, MultinomialNB, SVC, LogisticRegression, MLPClassifier


# ---------- 1. 流水线: 加载 -> 清洗 (复用 draft_main 的 clean_text) -> 划分 -> TF-IDF ----------
train_df = pd.read_csv('train_data.csv')
X_all = [clean_text(s) for s in train_df['text'].astype(str)]
y_all = train_df['target'].astype(int).tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
print(f'流水线就绪: 训练 {X_train_tfidf.shape}, 验证 {X_val_tfidf.shape}')
print()


# ---------- 2. 参数扫描 ----------
results = []


def log(model_name, param_desc, acc):
    results.append({'模型': model_name, '参数': param_desc, '验证集准确率': round(acc, 4)})
    print(f'{model_name:<6s} | {param_desc:<24s} | {acc:.4f}')


print('== 朴素贝叶斯: alpha 扫描 ==')
for a in [0.01, 0.1, 0.5, 1, 2]:
    log('朴素贝叶斯', f'alpha={a}',
        MultinomialNB(X_train_tfidf, y_train, X_val_tfidf, y_val, alpha=a, save=False))

print('== SVC: 线性核 C 扫描 + rbf 对比 ==')
for c in [0.1, 1, 10, 100]:
    log('支持向量机', f'linear C={c}',
        SVC(X_train_tfidf, y_train, X_val_tfidf, y_val, kernel='linear', C=c, save=False))
log('支持向量机', 'rbf C=1 gamma=scale',
    SVC(X_train_tfidf, y_train, X_val_tfidf, y_val, kernel='rbf', C=1, gamma='scale', save=False))

print('== 逻辑回归: C 扫描 ==')
for c in [0.1, 1, 10, 100]:
    log('逻辑回归', f'C={c}',
        LogisticRegression(X_train_tfidf, y_train, X_val_tfidf, y_val, C=c, save=False))

print('== 多层感知机: 隐藏层 × 正则化 alpha 扫描 ==')
for hls in [(50,), (100,), (200,)]:
    for a in [0.0001, 0.001, 0.01]:
        log('多层感知机', f'hidden={hls} alpha={a}',
            MLPClassifier(X_train_tfidf, y_train, X_val_tfidf, y_val,
                          hidden_layer_sizes=hls, alpha=a, save=False))


# ---------- 3. 汇总结果 ----------
df = pd.DataFrame(results)
print()
print('== 全部结果 ==')
print(df.to_string(index=False))

best = df.loc[df.groupby('模型')['验证集准确率'].idxmax()].reset_index(drop=True)
print()
print('== 各模型最佳配置 ==')
print(best.to_string(index=False))
df.to_csv('tuning_results.csv', index=False, encoding='utf-8-sig')


# ---------- 4. 对比柱状图 ----------
# 绘图独立在 plot_results.py: 读取 tuning_results.csv 即可重新出图, 无需重新扫描
print()
print('已保存: tuning_results.csv (结果表); 运行 plot_results.py 生成对比柱状图')
