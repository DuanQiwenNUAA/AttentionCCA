import numpy as np
import torch

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