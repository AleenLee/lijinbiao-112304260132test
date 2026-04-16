"""
机器学习实验：基于 Word2Vec 的情感预测（优化版）
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
from sklearn.preprocessing import StandardScaler
from collections import Counter
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

# ============================================
# 2. 文本预处理（优化版）
# ============================================
print("\n" + "=" * 50)
print("2. 文本预处理（优化版）")
print("=" * 50)

# 停用词列表（排除否定词 - 对情感分析至关重要！）
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does', 'doing', 'down',
    'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having',
    'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i',
    'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most',
    'my', 'myself', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such',
    'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
    'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until',
    'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while',
    'who', 'whom', 'why', 'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves'
}

# 否定词 - 必须保留！
NEGATION_WORDS = {'not', 'no', 'never', 'nor', 'neither', 'nobody', 'nothing', 'nowhere', 'none', 'cannot'}

stop_words = STOP_WORDS - NEGATION_WORDS

def review_to_words_optimized(raw_review):
    """优化版文本预处理：保留否定词"""
    # 1. 去除HTML标签
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()

    # 2. 处理缩写（保留否定意义）
    review_text = re.sub(r"n't", " not", review_text)
    review_text = re.sub(r"'re", " are", review_text)
    review_text = re.sub(r"'s", " is", review_text)
    review_text = re.sub(r"'d", " would", review_text)
    review_text = re.sub(r"'ll", " will", review_text)
    review_text = re.sub(r"'ve", " have", review_text)
    review_text = re.sub(r"'m", " am", review_text)

    # 3. 去除非字母字符
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # 4. 转换为小写并分词
    words = letters_only.lower().split()

    # 5. 去除停用词（保留否定词）
    meaningful_words = [w for w in words if w not in stop_words and len(w) > 1]

    return meaningful_words

# 预处理
print("预处理训练集...")
clean_train_reviews = [review_to_words_optimized(train["review"][i]) for i in range(len(train))]
print("预处理测试集...")
clean_test_reviews = [review_to_words_optimized(test["review"][i]) for i in range(len(test))]
print("预处理完成！")

# ============================================
# 3. Word2Vec 特征表示
# ============================================
print("\n" + "=" * 50)
print("3. Word2Vec 特征表示")
print("=" * 50)

print("训练Word2Vec模型...")
all_reviews = clean_train_reviews + clean_test_reviews

word2vec_model = Word2Vec(
    sentences=all_reviews,
    vector_size=300,
    window=10,
    min_count=2,
    workers=4,
    sg=1,
    negative=10,
    epochs=12
)

print(f"词汇表大小: {len(word2vec_model.wv)}")

# TF-IDF加权平均词向量
def get_feature_vector_tfidf(reviews, model, num_features):
    """TF-IDF加权平均词向量"""
    doc_count = len(reviews)
    word_doc_count = Counter()
    for review in reviews:
        for word in set(review):
            word_doc_count[word] += 1

    feature_vector = np.zeros((len(reviews), num_features), dtype="float32")

    for idx, review in enumerate(reviews):
        feature_vec = np.zeros(num_features, dtype="float32")
        n_words = 0
        word_counts = Counter(review)

        for word in review:
            if word in model.wv:
                tf = word_counts[word] / len(review)
                idf = np.log(doc_count / (1 + word_doc_count[word]))
                n_words += 1
                feature_vec += model.wv[word] * tf * idf

        if n_words > 0:
            feature_vec /= n_words
        feature_vector[idx] = feature_vec

    return feature_vector

print("生成特征向量...")
train_data_features = get_feature_vector_tfidf(clean_train_reviews, word2vec_model, 300)
test_data_features = get_feature_vector_tfidf(clean_test_reviews, word2vec_model, 300)

# 标准化
scaler = StandardScaler()
train_data_features = scaler.fit_transform(train_data_features)
test_data_features = scaler.transform(test_data_features)

print(f"训练集特征形状: {train_data_features.shape}")

# ============================================
# 4. 分类模型训练
# ============================================
print("\n" + "=" * 50)
print("4. 分类模型训练")
print("=" * 50)

# 逻辑回归
print("训练逻辑回归...")
lr_model = LogisticRegression(C=0.5, max_iter=2000, random_state=42, class_weight='balanced')
lr_scores = cross_val_score(lr_model, train_data_features, train["sentiment"], cv=5, scoring='roc_auc')
print(f"逻辑回归 5折交叉验证 AUC: {lr_scores.mean():.4f}")

# 随机森林
print("训练随机森林...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1, class_weight='balanced')
rf_scores = cross_val_score(rf_model, train_data_features, train["sentiment"], cv=5, scoring='roc_auc')
print(f"随机森林 5折交叉验证 AUC: {rf_scores.mean():.4f}")

# 选择最佳模型
if lr_scores.mean() > rf_scores.mean():
    print(f"\n选择逻辑回归，AUC: {lr_scores.mean():.4f}")
    final_model = lr_model
else:
    print(f"\n选择随机森林，AUC: {rf_scores.mean():.4f}")
    final_model = rf_model

final_model.fit(train_data_features, train["sentiment"])

# ============================================
# 5. 预测与生成提交文件
# ============================================
print("\n" + "=" * 50)
print("5. 预测与生成提交文件")
print("=" * 50)

predictions = final_model.predict(test_data_features)
output = pd.DataFrame(data={"id": test["id"], "sentiment": predictions})
output.to_csv("submission/submission.csv", index=False, quoting=3)

print(f"提交文件已保存到: submission/submission.csv")
print(f"预测结果: 正面 {sum(predictions)}, 负面 {len(predictions) - sum(predictions)}")
print("\n实验完成！")
