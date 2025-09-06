import torch
import torch.optim as optim
import numpy as np
from self_attention import SelfAttention, MultiHeadSelfAttention, apply_self_attention
from data_preprocessing import (
    load_multi_view_data,
    normalize_data,
    prepare_for_attention,
    convert_to_tensor,
    split_train_test,
    batch_data
)
from evaluation import evaluate_attention_effect


class AttentionCCA:
    """
    注意力机制结合CCA的主类
    先对每个视图数据进行自注意力机制处理，得到新的向量表示
    然后可以进行后续的CCA处理
    """
    def __init__(self, config=None):
        """
        初始化AttentionCCA模型
        
        参数:
            config: 配置字典，包含模型参数
        """
        # 默认配置
        self.config = {
            'view1_input_dim': 100,  # 第一个视图的输入维度
            'view2_input_dim': 100,  # 第二个视图的输入维度
            'view1_output_dim': None,  # 第一个视图的输出维度，默认为输入维度
            'view2_output_dim': None,  # 第二个视图的输出维度，默认为输入维度
            'attention_type': 'multihead',  # 'single' 或 'multihead'
            'num_heads': 4,  # 多头自注意力的头数
            'hidden_dim': 128,  # 隐藏层维度
            'use_gpu': False,  # 是否使用GPU
        }
        
        # 更新配置
        if config is not None:
            self.config.update(config)
        
        # 初始化设备
        self.device = torch.device('cuda' if self.config['use_gpu'] and torch.cuda.is_available() else 'cpu')
        
        # 初始化自注意力模型
        self._init_attention_models()
        
    def _init_attention_models(self):
        """
        初始化自注意力模型
        """
        if self.config['attention_type'] == 'single':
            # 单头自注意力
            self.view1_attention = SelfAttention(
                input_dim=self.config['view1_input_dim'],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['view1_output_dim']
            )
            self.view2_attention = SelfAttention(
                input_dim=self.config['view2_input_dim'],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['view2_output_dim']
            )
        else:
            # 多头自注意力
            self.view1_attention = MultiHeadSelfAttention(
                input_dim=self.config['view1_input_dim'],
                num_heads=self.config['num_heads'],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['view1_output_dim']
            )
            self.view2_attention = MultiHeadSelfAttention(
                input_dim=self.config['view2_input_dim'],
                num_heads=self.config['num_heads'],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['view2_output_dim']
            )
            
    def process_views(self, view1_data, view2_data, sequence_length1=None, sequence_length2=None):
        """
        处理两个视图数据，应用自注意力机制
        
        参数:
            view1_data: 第一个视图的数据
            view2_data: 第二个视图的数据
            sequence_length1: 第一个视图的序列长度
            sequence_length2: 第二个视图的序列长度
        
        返回:
            tuple: (processed_view1, processed_view2)，处理后的两个视图数据
        """
        # 准备数据格式
        prepared_view1 = prepare_for_attention(view1_data, sequence_length1)
        prepared_view2 = prepare_for_attention(view2_data, sequence_length2)
        
        # 转换为张量
        tensor_view1 = convert_to_tensor(prepared_view1)
        tensor_view2 = convert_to_tensor(prepared_view2)
        
        # 应用自注意力机制
        processed_view1 = apply_self_attention(tensor_view1, self.view1_attention, self.device)
        processed_view2 = apply_self_attention(tensor_view2, self.view2_attention, self.device)
        
        # 将结果转换回numpy数组（如果需要）
        if not isinstance(view1_data, torch.Tensor):
            processed_view1 = processed_view1.cpu().numpy()
            processed_view2 = processed_view2.cpu().numpy()
        
        return processed_view1, processed_view2

    def save_models(self, view1_path, view2_path):
        """
        保存自注意力模型
        
        参数:
            view1_path: 第一个视图的模型保存路径
            view2_path: 第二个视图的模型保存路径
        """
        torch.save(self.view1_attention.state_dict(), view1_path)
        torch.save(self.view2_attention.state_dict(), view2_path)
        
    def load_models(self, view1_path, view2_path):
        """
        加载自注意力模型
        
        参数:
            view1_path: 第一个视图的模型加载路径
            view2_path: 第二个视图的模型加载路径
        """
        self.view1_attention.load_state_dict(torch.load(view1_path, map_location=self.device))
        self.view2_attention.load_state_dict(torch.load(view2_path, map_location=self.device))
        
        # 设置为评估模式
        self.view1_attention.eval()
        self.view2_attention.eval()
        
    def _correlation_loss(self, view1_features, view2_features):
        """
        计算两个视图特征之间的相关性损失
        目标是最大化或最小化两个视图之间的相关性
        
        参数:
            view1_features: 第一个视图处理后的特征
            view2_features: 第二个视图处理后的特征
        
        返回:
            loss: 相关性损失值
        """
        # 对特征进行平均池化，减少序列维度
        view1_mean = torch.mean(view1_features, dim=1)  # [batch_size, output_dim]
        view2_mean = torch.mean(view2_features, dim=1)  # [batch_size, output_dim]
        
        # 计算协方差矩阵
        batch_size = view1_mean.size(0)
        centered_view1 = view1_mean - torch.mean(view1_mean, dim=0, keepdim=True)
        centered_view2 = view2_mean - torch.mean(view2_mean, dim=0, keepdim=True)
        
        # 归一化特征以计算相关性
        view1_norm = torch.norm(centered_view1, dim=1, keepdim=True) + 1e-8
        view2_norm = torch.norm(centered_view2, dim=1, keepdim=True) + 1e-8
        
        normalized_view1 = centered_view1 / view1_norm
        normalized_view2 = centered_view2 / view2_norm
        
        # 计算视图间的相关性
        correlation = torch.mean(torch.sum(normalized_view1 * normalized_view2, dim=1))
        
        # 如果我们想最大化相关性，使用1 - correlation作为损失
        # 如果我们想最小化相关性，直接使用correlation作为损失
        # 这里我们选择最大化视图间的相关性
        loss = 1 - correlation
        
        return loss
        
    def train_model(self, train_data, num_epochs=100, batch_size=32, learning_rate=0.001):
        """
        训练AttentionCCA模型
        
        参数:
            train_data: 训练数据，包含(view1_data, view2_data)元组
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        
        返回:
            loss_history: 训练过程中的损失历史
        """
        # 解包训练数据
        view1_train, view2_train = train_data
        
        # 准备数据格式
        view1_data = prepare_for_attention(view1_train)
        view2_data = prepare_for_attention(view2_train)
        
        # 转换为张量
        tensor_view1 = convert_to_tensor(view1_data)
        tensor_view2 = convert_to_tensor(view2_data)
        
        # 创建批次数据并转换为列表以便获取长度
        train_batches = list(batch_data(tensor_view1, tensor_view2, batch_size))
        
        # 设置优化器
        params = list(self.view1_attention.parameters()) + list(self.view2_attention.parameters())
        optimizer = optim.Adam(params, lr=learning_rate)
        
        # 记录损失历史
        loss_history = []
        
        # 开始训练循环
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch in train_batches:
                # 解包批次数据
                batch_view1, batch_view2 = batch
                
                # 移动到指定设备
                batch_view1 = batch_view1.to(self.device)
                batch_view2 = batch_view2.to(self.device)
                
                # 前向传播 - 使用训练模式
                processed_view1 = apply_self_attention(batch_view1, self.view1_attention, self.device, train_mode=True)
                processed_view2 = apply_self_attention(batch_view2, self.view2_attention, self.device, train_mode=True)
                
                # 计算损失
                loss = self._correlation_loss(processed_view1, processed_view2)
                
                # 反向传播和参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 累计损失
                epoch_loss += loss.item()
            
            # 计算平均损失
            avg_epoch_loss = epoch_loss / len(train_batches)
            loss_history.append(avg_epoch_loss)
            
            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
        
        return loss_history


# 用于评估的辅助函数

# 示例用法函数
def demo_attention_cca():
    """
    演示AttentionCCA的使用方法，包括模型训练过程
    """
    # 创建模拟数据
    np.random.seed(42)
    view1_data = np.random.rand(100, 100)  # 100个样本，每个样本100维
    view2_data = np.random.rand(100, 100)  # 100个样本，每个样本100维
    
    # 创建配置
    config = {
        'view1_input_dim': 100,
        'view2_input_dim': 100,
        'view1_output_dim': 50,  # 指定降维后的输出维度
        'view2_output_dim': 50,  # 指定降维后的输出维度
        'attention_type': 'multihead',
        'num_heads': 4,
        'hidden_dim': 128,
        'use_gpu': False
    }
    
    # 初始化模型
    model = AttentionCCA(config)
    
    # 使用未训练的模型处理数据
    print("===== 未训练模型的处理结果 =====")
    untrained_view1, untrained_view2 = model.process_views(view1_data, view2_data)
    
    # 准备训练数据
    print("\n===== 开始训练模型 =====")
    # 分割训练和测试数据
    view1_train, view1_test, view2_train, view2_test = split_train_test(view1_data, view2_data, test_ratio=0.2)
    train_data = (view1_train, view2_train)
    test_data = (view1_test, view2_test)
    
    # 训练模型
    loss_history = model.train_model(
        train_data=train_data,
        num_epochs=50,  # 训练轮数
        batch_size=32,  # 批次大小
        learning_rate=0.001  # 学习率
    )
    
    # 保存训练后的模型
    model.save_models('view1_attention_model.pth', 'view2_attention_model.pth')
    print("\n模型已保存到view1_attention_model.pth和view2_attention_model.pth")
    
    # 使用训练后的模型处理数据
    print("\n===== 训练后模型的处理结果 =====")
    trained_view1, trained_view2 = model.process_views(view1_test, view2_test)
    
    # 打印结果形状
    print(f"\n测试数据形状:")
    print(f"  视图1形状: {view1_test.shape}")
    print(f"  视图2形状: {view2_test.shape}")
    print(f"训练后处理结果形状:")
    print(f"  视图1形状: {trained_view1.shape}")
    print(f"  视图2形状: {trained_view2.shape}")
    
    # 评估处理前后视图之间的相关性
    print("\n评估处理前后视图之间的相关性:")
    print("=========================")
    
    # 先确保数据是numpy数组
    if isinstance(view1_test, torch.Tensor):
        view1_test = view1_test.cpu().numpy()
    if isinstance(view2_test, torch.Tensor):
        view2_test = view2_test.cpu().numpy()
    if isinstance(trained_view1, torch.Tensor):
        trained_view1 = trained_view1.cpu().numpy()
    if isinstance(trained_view2, torch.Tensor):
        trained_view2 = trained_view2.cpu().numpy()
    
    # 评估原始视图间相关性
    original_cross_correlation_result = evaluate_attention_effect(view1_test, view2_test)
    original_cross_correlation = original_cross_correlation_result['cross_correlation']
    
    # 评估处理后视图间相关性
    processed_cross_correlation_result = evaluate_attention_effect(trained_view1, trained_view2)
    processed_cross_correlation = processed_cross_correlation_result['cross_correlation']
    
    print("\n原始视图间相关性:")
    print(f"  视图1形状: {original_cross_correlation_result['view1_shape']}")
    print(f"  视图2形状: {original_cross_correlation_result['view2_shape']}")
    print(f"  视图1均值: {original_cross_correlation_result['view1_mean']:.4f}")
    print(f"  视图2均值: {original_cross_correlation_result['view2_mean']:.4f}")
    print(f"  视图1方差: {original_cross_correlation_result['view1_variance']:.4f}")
    print(f"  视图2方差: {original_cross_correlation_result['view2_variance']:.4f}")
    print(f"  视图间相关性: {original_cross_correlation:.4f}")
    
    print("\n处理后视图间相关性:")
    print(f"  视图1形状: {processed_cross_correlation_result['view1_shape']}")
    print(f"  视图2形状: {processed_cross_correlation_result['view2_shape']}")
    print(f"  视图1均值: {processed_cross_correlation_result['view1_mean']:.4f}")
    print(f"  视图2均值: {processed_cross_correlation_result['view2_mean']:.4f}")
    print(f"  视图1方差: {processed_cross_correlation_result['view1_variance']:.4f}")
    print(f"  视图2方差: {processed_cross_correlation_result['view2_variance']:.4f}")
    print(f"  视图间相关性: {processed_cross_correlation:.4f}")
    
    print("\n相关性比较:")
    print(f"  相关性变化: {processed_cross_correlation - original_cross_correlation:.4f} ({(processed_cross_correlation - original_cross_correlation) / original_cross_correlation * 100:.2f}%)")
    
    return trained_view1, trained_view2


if __name__ == "__main__":
    # 运行演示
    demo_attention_cca()