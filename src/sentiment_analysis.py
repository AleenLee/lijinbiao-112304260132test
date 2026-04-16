"""
机器学习实验：基于 Word2Vec 的情感预测
学生：李金彪
学号：112304260132
班级：数据1231
"""

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. 数据读取
# ============================================
print("=" * 50)
print("1. 读取数据")
print("=" * 50)

train = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("data/testData.tsv", header=0, delimiter="\t", quoting=3)

print(f"训练集形状: {train.shape}")
print(f"测试集形状: {test.shape}")
print(f"训练集列名: {train.columns.values}")

# ============================================
# 2. 文本预处理
# ============================================
print("\n" + "=" * 50)
print("2. 文本预处理")
print("=" * 50)

def review_to_words(raw_review):
    """
    将原始评论转换为清洗后的词语列表
    处理步骤：
    1. 去除HTML标签
    2. 去除非字母字符
    3. 转换为小写
    4. 分词
    5. 去除停用词
    """
    # 1. 去除HTML标签
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()

    # 2. 去除非字母字符
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # 3. 转换为小写并分词
    words = letters_only.lower().split()

    # 4. 去除停用词
    try:
        from nltk.corpus import stopwords
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]
    except:
        # 如果没有下载停用词，使用简单的停用词列表
        simple_stops = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                       'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                       'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                       'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for',
                       'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                       'before', 'after', 'above', 'below', 'between', 'under', 'again',
                       'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                       'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                       'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                       's', 't', 'just', 'don', 'now', 'i', 'me', 'my', 'myself', 'we', 'our',
                       'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
                       'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it',
                       'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                       'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                       'am', 'if', 'because', 'until', 'while', 'about', 'against', 'up',
                       'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
                       'once'}
        meaningful_words = [w for w in words if w not in simple_stops]

    return meaningful_words

def review_to_sentence_list(raw_review):
    """
    将评论转换为句子列表（用于Word2Vec训练）
    """
    # 去除HTML标签
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()

    # 去除非字母字符
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # 转换为小写并分词
    words = letters_only.lower().split()

    return words

# 预处理训练集
print("预处理训练集...")
clean_train_reviews = []
for i in range(len(train["review"])):
    if (i + 1) % 5000 == 0:
        print(f"  处理进度: {i + 1}/{len(train['review'])}")
    clean_train_reviews.append(review_to_words(train["review"][i]))

print("预处理测试集...")
clean_test_reviews = []
for i in range(len(test["review"])):
    if (i + 1) % 5000 == 0:
        print(f"  处理进度: {i + 1}/{len(test['review'])}")
    clean_test_reviews.append(review_to_words(test["review"][i]))

print(f"预处理完成！训练集样本数: {len(clean_train_reviews)}, 测试集样本数: {len(clean_test_reviews)}")

# ============================================
# 3. Word2Vec 特征表示
# ============================================
print("\n" + "=" * 50)
print("3. Word2Vec 特征表示")
print("=" * 50)

# 训练Word2Vec模型
print("训练Word2Vec模型...")
all_reviews = clean_train_reviews + clean_test_reviews

# Word2Vec 参数说明：
# vector_size: 词向量维度 (100维)
# window: 上下文窗口大小 (5)
# min_count: 忽略出现次数少于该值的词 (5)
# workers: 并行线程数
word2vec_model = Word2Vec(sentences=all_reviews,
                          vector_size=100,
                          window=5,
                          min_count=5,
                          workers=4)

print(f"词汇表大小: {len(word2vec_model.wv)}")

# 将句子转换为向量（使用平均词向量）
def get_avg_feature_vector(reviews, model, num_features):
    """
    计算每条评论的平均词向量
    """
    feature_vector = np.zeros((len(reviews), num_features), dtype="float32")
    index = 0

    for review in reviews:
        feature_vec = np.zeros(num_features, dtype="float32")
        n_words = 0

        for word in review:
            if word in model.wv:
                n_words += 1
                feature_vec = np.add(feature_vec, model.wv[word])

        if n_words > 0:
            feature_vec = np.divide(feature_vec, n_words)

        feature_vector[index] = feature_vec
        index += 1

    return feature_vector

print("生成训练集特征向量...")
train_data_features = get_avg_feature_vector(clean_train_reviews, word2vec_model, 100)

print("生成测试集特征向量...")
test_data_features = get_avg_feature_vector(clean_test_reviews, word2vec_model, 100)

print(f"训练集特征形状: {train_data_features.shape}")
print(f"测试集特征形状: {test_data_features.shape}")

# ============================================
# 4. 分类模型训练
# ============================================
print("\n" + "=" * 50)
print("4. 分类模型训练")
print("=" * 50)

# 尝试多个分类器
print("尝试不同的分类器...")

# 4.1 随机森林
print("\n训练随机森林...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_scores = cross_val_score(rf_model, train_data_features, train["sentiment"], cv=5, scoring='roc_auc')
print(f"随机森林 5折交叉验证 AUC: {rf_scores.mean():.4f} (+/- {rf_scores.std()*2:.4f})")

# 4.2 逻辑回归
print("\n训练逻辑回归...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_scores = cross_val_score(lr_model, train_data_features, train["sentiment"], cv=5, scoring='roc_auc')
print(f"逻辑回归 5折交叉验证 AUC: {lr_scores.mean():.4f} (+/- {lr_scores.std()*2:.4f})")

# 选择最佳模型
if lr_scores.mean() > rf_scores.mean():
    print("\n选择逻辑回归作为最终模型")
    final_model = lr_model
    best_model_name = "Logistic Regression"
else:
    print("\n选择随机森林作为最终模型")
    final_model = rf_model
    best_model_name = "Random Forest"

# 在全部训练数据上训练最终模型
print(f"\n使用 {best_model_name} 在全部训练数据上训练...")
final_model.fit(train_data_features, train["sentiment"])

# ============================================
# 5. 预测与生成提交文件
# ============================================
print("\n" + "=" * 50)
print("5. 预测与生成提交文件")
print("=" * 50)

# 预测测试集
print("预测测试集...")
predictions = final_model.predict(test_data_features)

# 生成提交文件
output = pd.DataFrame(data={"id": test["id"], "sentiment": predictions})
output.to_csv("submission/submission.csv", index=False, quoting=3)

print(f"提交文件已保存到: submission/submission.csv")
print(f"预测结果统计: 正面评论 {sum(predictions)}, 负面评论 {len(predictions) - sum(predictions)}")

# ============================================
# 6. 实验总结
# ============================================
print("\n" + "=" * 50)
print("6. 实验总结")
print("=" * 50)
print("""
实验方法说明：

1. 文本预处理：
   - 使用BeautifulSoup去除HTML标签
   - 使用正则表达式去除非字母字符
   - 转换为小写
   - 分词并去除英语停用词

2. Word2Vec特征表示：
   - 自己训练Word2Vec模型（使用训练集和测试集的所有评论）
   - 词向量维度：100维
   - 句子向量：使用平均词向量方法

3. 分类模型：
   - 尝试了随机森林和逻辑回归
   - 使用5折交叉验证评估模型性能
   - 最终选择交叉验证AUC最高的模型

4. 实验流程：
   1. 读取训练集和测试集
   2. 对文本进行预处理
   3. 训练Word2Vec模型
   4. 将每条文本表示为句向量（平均词向量）
   5. 用训练集训练分类器
   6. 在测试集上预测结果
   7. 生成submission文件
""")

print("实验完成！")
