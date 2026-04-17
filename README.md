# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：李金彪
- **学号**：112304260132
- **班级**：数据1231

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview
- **提交日期**：2026/4/16

- **GitHub 仓库地址**：https://github.com/AleenLee/lijinbiao-112304260132test.git
- **GitHub README 地址**：https://github.com/AleenLee/lijinbiao-112304260132test.git

> 注意：GitHub 仓库首页或 README 页面中，必须能看到"姓名 + 学号"，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：0.90276
- **Private Score**（如有）：0.902766
- **排名**（如能看到可填写）：无

---

## 5. Kaggle 截图
截图文件：`112304260132_李金彪_kaggle_score.png`
---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**
- 使用 BeautifulSoup 库去除 HTML 标签（如 `<br />`、`<br/><br/>` 等）
- 使用正则表达式 `[^a-zA-Z]` 去除非字母字符和标点符号
- 将所有文本转换为小写，保证 "Movie" 和 "movie" 被视为同一个词
- 处理缩写形式，如 "don't" → "do not"、"won't" → "will not"，保留否定意义
- 去除英语停用词，但**保留否定词**（not, no, never, nor 等），因为否定词对情感分析至关重要
- 例如 "not good" 和 "good" 如果去掉 "not"，情感含义完全相反

---

### （2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**
- **自己训练 Word2Vec 模型**，使用训练集和测试集的所有评论作为语料库
- 词向量维度设置为 **300 维**，能够捕捉更丰富的语义信息
- 上下文窗口大小设为 10，最小词频为 2
- 使用 **Skip-gram 模型**（sg=1），相比 CBOW 更适合小数据集
- 使用 **TF-IDF 加权平均**方法得到句子向量：
  - 对每个词计算 TF-IDF 权重
  - 将句子中所有词向量按权重加权求和
  - 相比简单平均，能突出重要词的贡献

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**
- 尝试了 **随机森林 (Random Forest)** 和 **逻辑回归 (Logistic Regression)** 两个分类模型
- 使用 **5 折交叉验证**评估模型性能，以 **AUC（ROC 曲线下面积）** 为评价指标
- 实验结果：
  - 随机森林 5折交叉验证 AUC: 0.89xx
  - 逻辑回归 5折交叉验证 AUC: 0.91xx
- **最终采用逻辑回归模型**，因为其交叉验证 AUC 更高
- 对特征进行标准化处理，提高模型收敛速度和效果

---

## 7. 实验流程
请简要说明你的实验流程。

示例：
1. 读取训练集和测试集
2. 对文本进行预处理
3. 训练或加载 Word2Vec 模型
4. 将每条文本表示为句向量
5. 用训练集训练分类器
6. 在测试集上预测结果
7. 生成 submission 文件并提交 Kaggle

**我的实验流程：**
1. 使用 pandas 读取 labeledTrainData.tsv（训练集，25000条带标签评论）和 testData.tsv（测试集，25000条无标签评论）
2. 对文本进行预处理：去除 HTML 标签、处理缩写、去除非字母字符、转小写、去停用词（保留否定词）
3. 使用 gensim 库训练 Word2Vec 模型，词向量维度 300，使用 Skip-gram 模型
4. 将每条评论转换为句向量：使用 TF-IDF 加权平均词向量方法
5. 对特征进行标准化处理
6. 使用 5 折交叉验证评估随机森林和逻辑回归模型，选择 AUC 更高的逻辑回归模型
7. 在全部训练数据上训练最终模型
8. 在测试集上预测情感标签（0 或 1）
9. 生成 submission.csv 文件，格式为 `id,sentiment`，提交到 Kaggle

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

示例：
- `data/`：存放数据文件
- `src/`：存放源代码
- `notebooks/`：存放实验 notebook
- `images/`：存放 README 中使用的图片
- `submission/`：存放提交文件

**我的项目结构：**
```text
demo/
├─ data/                    # 存放数据文件
│   ├─ labeledTrainData.tsv # 训练数据（25000条带标签评论）
│   └─ testData.tsv         # 测试数据（25000条无标签评论）
├─ src/                     # 存放源代码
│   └─ sentiment_analysis.py # 情感分析主程序
├─ submission/              # 存放提交文件
│   └─ submission.csv       # Kaggle 提交文件
├─ images/                  # 存放 README 中使用的图片
│   └─ kaggle_score.png     # Kaggle 成绩截图
└─ readme.md                # 实验说明文档
```
