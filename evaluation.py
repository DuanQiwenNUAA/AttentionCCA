import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score

def evaluate_kmeans_clustering(view1_data, view2_data, n_clusters=3, true_labels1=None, true_labels2=None):
    """
    评估KMeans聚类效果，计算聚类指标
    
    参数:
        view1_data: 第一个视图的数据
        view2_data: 第二个视图的数据
        n_clusters: 聚类数量，默认为3
        true_labels1: 视图1的真实标签(可选)
        true_labels2: 视图2的真实标签(可选)
    
    返回:
        dict: 包含聚类评估指标的字典
    """
    # 确保数据是numpy数组
    if isinstance(view1_data, torch.Tensor):
        view1_data = view1_data.cpu().numpy()  # 将PyTorch张量转换为numpy数组
    if isinstance(view2_data, torch.Tensor):
        view2_data = view2_data.cpu().numpy()  # 将PyTorch张量转换为numpy数组
    
    # 如果数据是三维的，展平为二维数组
    if view1_data.ndim > 2:
        view1_data = view1_data.reshape(view1_data.shape[0], -1)  # 展平为(样本数, 特征数)格式
    if view2_data.ndim > 2:
        view2_data = view2_data.reshape(view2_data.shape[0], -1)  # 展平为(样本数, 特征数)格式
    
    # 执行KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # 初始化KMeans模型，固定随机种子保证可重复性
    
    # 分别计算每个视图的聚类指标
    view1_labels = kmeans.fit_predict(view1_data)  # 对视图1数据进行聚类并获取标签
    view1_silhouette = silhouette_score(view1_data, view1_labels)  # 计算视图1轮廓系数
    view1_calinski = calinski_harabasz_score(view1_data, view1_labels)  # 计算视图1Calinski-Harabasz指数
    
    view2_labels = kmeans.fit_predict(view2_data)  # 对视图2数据进行聚类并获取标签
    view2_silhouette = silhouette_score(view2_data, view2_labels)  # 计算视图2轮廓系数
    view2_calinski = calinski_harabasz_score(view2_data, view2_labels)  # 计算视图2Calinski-Harabasz指数
    
    # 计算联合聚类指标
    joint_data = np.concatenate([view1_data, view2_data], axis=1)  # 将两个视图数据在特征维度上拼接
    joint_labels = kmeans.fit_predict(joint_data)  # 对联合数据进行聚类
    joint_silhouette = silhouette_score(joint_data, joint_labels)  # 计算联合轮廓系数
    
    # 返回评估结果
    return {
        'n_clusters': n_clusters,  # 聚类数量

        # 内部指标，无需外部真实标签，仅基于数据集自身的特征和聚类结果来评估聚类的紧密度和分离度
        'view1_silhouette': view1_silhouette,  # 视图1轮廓系数，衡量聚类质量，值越大表示聚类效果越好
        'view2_silhouette': view2_silhouette,  # 视图2轮廓系数
        'joint_silhouette': joint_silhouette,  # 联合视图轮廓系数
        'view1_calinski_harabasz_score': view1_calinski,  # 视图1Calinski-Harabasz指数，值越大表示聚类效果越好
        'view2_calinski_harabasz_score': view2_calinski,  # 视图2Calinski-Harabasz指数
        'view1_davies_bouldin_score': davies_bouldin_score(view1_data, view1_labels),  # 视图1Davies-Bouldin指数，值越小表示聚类效果越好
        'view2_davies_bouldin_score': davies_bouldin_score(view2_data, view2_labels),  # 视图2Davies-Bouldin指数

        # 外部指标，当有数据的真实类别标签时，可以使用这些指标，它们类似于分类任务中的准确率、召回率等
        'view1_adjusted_rand_score': adjusted_rand_score(true_labels1, view1_labels),  # 视图1调整兰德指数
        'view2_adjusted_rand_score': adjusted_rand_score(true_labels2, view2_labels),  # 视图2调整兰德指数
        'view1_adjusted_mutual_info_score': adjusted_mutual_info_score(true_labels1, view1_labels),  # 视图1调整互信息分数
        'view2_adjusted_mutual_info_score': adjusted_mutual_info_score(true_labels2, view2_labels),  # 视图2调整互信息分数
    }



def evaluate_attention_effect(view1_data, view2_data):
    """
    评估自注意力机制的效果，计算两个视图之间的相关性
    
    参数:
        view1_data: 第一个视图的数据
        view2_data: 第二个视图的数据
    
    返回:
        dict: 包含视图间相关性评估指标的字典
    """
    # 确保数据是numpy数组
    if isinstance(view1_data, torch.Tensor):
        view1_data = view1_data.cpu().numpy()
    if isinstance(view2_data, torch.Tensor):
        view2_data = view2_data.cpu().numpy()
        
    # 计算数据形状
    view1_shape = view1_data.shape
    view2_shape = view2_data.shape
    
    # 计算数据统计量
    view1_mean = np.mean(view1_data)
    view2_mean = np.mean(view2_data)
    view1_var = np.var(view1_data)
    view2_var = np.var(view2_data)
    
    # 计算两个视图之间的相关性
    # 展平数据以计算相关性
    view1_flat = view1_data.flatten()
    view2_flat = view2_data.flatten()
    cross_correlation = np.corrcoef(view1_flat, view2_flat)[0, 1]
    
    # 返回评估结果
    return {
        'view1_shape': view1_shape,
        'view2_shape': view2_shape,
        'view1_mean': view1_mean,
        'view2_mean': view2_mean,
        'view1_variance': view1_var,
        'view2_variance': view2_var,
        'cross_correlation': cross_correlation
    }