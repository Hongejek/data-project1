# plot_results.py — 读取 tuning_results.csv, 绘制对比柱状图
# 图内容: 4 个模型 × 每个模型 4 组候选参数 = 16 根柱子
# 运行: python plot_results.py   (生成 model_comparison.png)
import pandas as pd

import matplotlib
matplotlib.use('Agg')                       # 无显示环境, 只保存图片
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文显示: 借用 Windows 的微软雅黑 (失败则退用系统 Droid Sans Fallback)
for _f in ['/mnt/c/Windows/Fonts/msyh.ttc',
           '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf']:
    try:
        font_manager.fontManager.addfont(_f)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=_f).get_name()
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('tuning_results.csv')

# 每个模型选取 4 组候选参数 (与调参实验一致)
selection = [
    ('朴素贝叶斯', ['alpha=0.01', 'alpha=0.1', 'alpha=1', 'alpha=2']),
    ('支持向量机', ['linear C=0.1', 'linear C=1', 'linear C=10', 'rbf C=1 gamma=scale']),
    ('逻辑回归', ['C=0.1', 'C=1', 'C=10', 'C=100']),
    # 最优 1 组 + 另外 3 组对照 (隐藏层 × 正则化 alpha 扫描结果)
    ('多层感知机', ['hidden=(50,) alpha=0.0001', 'hidden=(100,) alpha=0.0001',
                'hidden=(200,) alpha=0.0001', 'hidden=(100,) alpha=0.01']),
]
colors = {'朴素贝叶斯': '#4C72B0', '支持向量机': '#DD8452',
          '逻辑回归': '#55A868', '多层感知机': '#C44E52'}

labels, values, bar_colors, group_of_bar = [], [], [], []
for gi, (model, params) in enumerate(selection):
    for p in params:
        hit = df[(df['模型'] == model) & (df['参数'] == p)]
        if hit.empty:
            raise SystemExit(f'结果表中找不到配置: {model} / {p}')
        labels.append(p.replace('hidden=', ''))   # MLP 标签去掉冗余前缀, 避免图内文字过长
        values.append(float(hit['验证集准确率'].iloc[0]))
        bar_colors.append(colors[model])
        group_of_bar.append(gi)

# ---------- 绘图 ----------
x = list(range(len(labels)))
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.bar(x, values, color=bar_colors, width=0.72, edgecolor='white')

# 柱顶标注准确率
for xi, v in zip(x, values):
    ax.text(xi, v + 0.0015, f'{v:.4f}', ha='center', va='bottom', fontsize=7.5)

# 组间分隔线与组名
for k in (3.5, 7.5, 11.5):
    ax.axvline(k, color='gray', linestyle=':', alpha=0.5)
lo, hi = min(values), max(values)
ax.set_ylim(lo - 0.02, hi + 0.035)
for gi, (model, _p) in enumerate(selection):
    center = gi * 4 + 1.5
    ax.text(center, hi + 0.027, model, ha='center', fontsize=11,
            fontweight='bold', color=colors[model])

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('验证集准确率')
ax.set_title('不同模型参数验证集准确率对比')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)

print('已保存: model_comparison.png (16 柱)')
print()
print('各组最佳参数:')
for model, params in selection:
    best_param, best_acc = None, -1.0
    for p in params:
        acc = float(df[(df['模型'] == model) & (df['参数'] == p)]['验证集准确率'].iloc[0])
        if acc > best_acc:
            best_param, best_acc = p, acc
    print(f'  {model}: {best_param} -> {best_acc:.4f}')
