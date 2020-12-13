from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np

# 导入MNIST数据集
mnist = datasets.load_digits()

# 训练集与测试集划分，25%的数据作为测试集
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target, test_size=0.25, random_state=42)

# 在训练集中，划出10%的数据用作验证
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1, random_state=84)

# KNN方法中的参数K
kVals = range(1, 30, 3)
accuracies = []

# 尝试不同的参数K的值
for k in range(1, 30, 3):
    # train the classifier with the current value of `k`
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    score = model.score(valData, valLabels)
    print("k = %d, accuracy = %.3f%%" % (k, score * 100))
    accuracies.append(score)

# 选出准确率最高的参数K的值
i = np.argmax(accuracies)
print("k = %d达到了最好的效果，准确率%.3f%%" % (kVals[i],
    accuracies[i] * 100))

# 用选出的这个K值重训练
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData, trainLabels)

# 测试
predictions = model.predict(testData)

# 输出测试结果
print(classification_report(testLabels, predictions))