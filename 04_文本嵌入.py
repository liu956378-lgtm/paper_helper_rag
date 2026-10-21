
# 变量类型	语法	作用域	生命周期
# 局部变量	weights = ...	仅在当前方法内	方法执行结束后消失
# 实例变量	self.weights = ...	整个类的所有方法	对象存在期间一直保留


from langchain_community.chat_models import ChatTongyi

from langchain_community.embeddings import DashScopeEmbeddings
import numpy as np

# 使用时指定模型
embeddings = DashScopeEmbeddings(
    model="qwen3.7-text-embedding",
    dashscope_api_key='sk-2c7d93a45c384a389d70bf757d8d5a27'
)


comment=['这个产品太棒了','垃圾']
vector=embeddings.embed_documents(comment)
labels=[1,0]  ##1表示正面，0表示负面

test_comment=['这个产品太垃圾了']
test_vector=embeddings.embed_documents(test_comment)




import numpy as np

class LogisticRegressionFromScratch:
    """
    手写逻辑回归分类器
    使用梯度下降优化交叉熵损失
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization  # L2正则化系数
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def _sigmoid(self, z):
        """
        Sigmoid激活函数
        将线性输出映射到 [0, 1] 概率区间
        """
        # 防止溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, X, y):
        """
        计算二分类交叉熵损失
        """
        n_samples = X.shape[0]
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(linear_output)
        
        # 交叉熵损失
        epsilon = 1e-10  # 防止log(0)
        loss = -np.mean(y * np.log(predictions + epsilon) + 
                        (1 - y) * np.log(1 - predictions + epsilon))
        
        # 添加L2正则化项
        loss += (self.regularization / (2 * n_samples)) * np.sum(self.weights ** 2)
        
        return loss
    
    def fit(self, X, y, verbose=True):
        """
        使用梯度下降训练逻辑回归
        X: 特征矩阵，可以是 list 或 numpy array，shape=(n_samples, n_features)
        y: 标签向量，可以是 list 或 numpy array，shape=(n_samples,)，取值0或1
        """
        # 将输入转换为 numpy 数组
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # 确保 y 是一维数组
        if len(y.shape) > 1:
            y = y.flatten()
        
        n_samples, n_features = X.shape
        
        # 初始化权重和偏置
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        
        # 特征标准化（可选，但建议加上）
        # self.X_mean = np.mean(X, axis=0)
        # self.X_std = np.std(X, axis=0) + 1e-10
        # X = (X - self.X_mean) / self.X_std
        
        for iteration in range(self.n_iterations):
            # 前向传播：计算预测值
            linear_output = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_output)
            
            # 计算梯度
            # 损失对权重的梯度: (1/n) * X^T * (predictions - y) + regularization * weights
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y)) + (self.regularization / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # 更新参数（梯度下降）
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 记录损失
            if iteration % 100 == 0:
                loss = self._compute_loss(X, y)
                self.loss_history.append(loss)
                if verbose:
                    print(f"迭代 {iteration}, 损失: {loss:.6f}")
        
        if verbose and len(self.loss_history) > 0:
            print(f"训练完成，最终损失: {self.loss_history[-1]:.6f}")
    
    def predict_proba(self, X):
        """
        预测概率
        X: 特征矩阵，可以是 list 或 numpy array
        返回: 属于类别1的概率
        """
        X = np.array(X, dtype=np.float64)
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)
    
    def predict(self, X, threshold=0.5):
        """
        预测类别
        X: 特征矩阵
        返回: 预测标签，取值0或1
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def predict_with_confidence(self, X):
        """
        预测并返回置信度
        返回: (预测标签, 置信度)
        """
        prob = self.predict_proba(X)
        pred = (prob >= 0.5).astype(int)
        confidence = np.where(pred == 1, prob, 1 - prob)
        return pred, confidence
# 使用示例
clf = LogisticRegressionFromScratch(learning_rate=0.1, n_iterations=1000)
clf.fit(vector, labels)
prediction = clf.predict(test_vector)
print(f"预测结果: {prediction[0]}")  # 输出预测标签

         