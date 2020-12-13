from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from sklearn import metrics

# 导入MNIST数据集
(X_T, Y_T), (X_TE, Y_TE) = tf.keras.datasets.mnist.load_data()

# 实例化决策树分类器
clf = DecisionTreeClassifier()

# 准备数据
X_T = X_T.reshape((60000, 784))
X_TE = X_TE.reshape((10000, 784))

# 训练决策树
clf.fit(X_T, Y_T)

# 测试模型
y_pred = clf.predict(X_TE)
print("Accuracy of the Decision Trees is :", metrics.accuracy_score(Y_TE, y_pred) * 100)