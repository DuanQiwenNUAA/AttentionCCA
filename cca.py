import numpy as np
from scipy.linalg import eigh

class CCA:
    """
    典型相关分析(CCA)实现类
    用于计算两组变量之间的典型相关
    """
    def __init__(self, n_components=None):
        """
        初始化CCA模型
        
        参数:
            n_components: 保留的典型变量数量，默认为None表示保留所有
        """
        self.n_components = n_components
        self.w_x = None  # X视图的投影矩阵
        self.w_y = None  # Y视图的投影矩阵
        self.corrs = None  # 典型相关系数
    
    def fit(self, X, Y):
        """
        拟合CCA模型
        
        参数:
            X: 第一个视图的数据，形状为[n_samples, n_features1]
            Y: 第二个视图的数据，形状为[n_samples, n_features2]
        """
        # 中心化数据
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)
        
        # 计算协方差矩阵
        cov_xx = np.cov(X, rowvar=False)
        cov_yy = np.cov(Y, rowvar=False)
        cov_xy = np.cov(X, Y, rowvar=False)[:X.shape[1], X.shape[1]:]
        cov_yx = cov_xy.T
        
        # 计算矩阵A和B
        inv_cov_xx = np.linalg.pinv(cov_xx)
        inv_cov_yy = np.linalg.pinv(cov_yy)
        A = np.dot(np.dot(inv_cov_xx, cov_xy), np.dot(inv_cov_yy, cov_yx))
        B = np.dot(np.dot(inv_cov_yy, cov_yx), np.dot(inv_cov_xx, cov_xy))
        
        # 计算特征值和特征向量
        eigvals_x, eigvecs_x = eigh(A)
        eigvals_y, eigvecs_y = eigh(B)
        
        # 排序特征值和特征向量(降序)
        idx_x = np.argsort(eigvals_x)[::-1]
        idx_y = np.argsort(eigvals_y)[::-1]
        
        eigvals_x = eigvals_x[idx_x]
        eigvecs_x = eigvecs_x[:, idx_x]
        eigvals_y = eigvals_y[idx_y]
        eigvecs_y = eigvecs_y[:, idx_y]
        
        # 计算典型相关系数
        self.corrs = np.sqrt(eigvals_x)
        
        # 计算投影矩阵
        self.w_x = eigvecs_x
        self.w_y = np.dot(np.dot(inv_cov_yy, cov_yx), eigvecs_x) / self.corrs
        
        # 如果指定了n_components，则截断
        if self.n_components is not None:
            self.w_x = self.w_x[:, :self.n_components]
            self.w_y = self.w_y[:, :self.n_components]
            self.corrs = self.corrs[:self.n_components]
    
    def transform(self, X, Y):
        """
        将数据投影到典型变量空间
        
        参数:
            X: 第一个视图的数据，形状为[n_samples, n_features1]
            Y: 第二个视图的数据，形状为[n_samples, n_features2]
            
        返回:
            X_transformed: X视图的典型变量
            Y_transformed: Y视图的典型变量
        """
        if self.w_x is None or self.w_y is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        # 中心化数据
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)
        
        # 计算典型变量
        X_transformed = np.dot(X, self.w_x)
        Y_transformed = np.dot(Y, self.w_y)
        
        return X_transformed, Y_transformed
    
    def fit_transform(self, X, Y):
        """
        拟合模型并返回转换后的数据
        """
        self.fit(X, Y)
        return self.transform(X, Y)
    
    def get_correlation_coefficients(self):
        """
        获取典型相关系数
        """
        return self.corrs