""" 
扩散吸取网络推理部署脚本
作者: dingtao huang
用于加载训练好的模型进行吸取点预测推理
"""
import os    
import sys
import numpy as np
import torch
import argparse
import time
from typing import Dict, List, Tuple, Optional
import json

# 设置项目路径
FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.dirname(FILE_PATH)
print(f"当前文件所在文件夹路径: {FILE_DIR}")
ROOT_DIR = os.path.dirname(FILE_DIR)
print(f"项目根目录路径: {ROOT_DIR}")

# 添加模块路径
sys.path.append(ROOT_DIR)
sys.path.append(FILE_DIR)  # 添加当前目录以导入 preprocess 模块
thirdparty_path = os.path.join(ROOT_DIR, "thirdparty", "Diffusion_Suction")
diffusion_model_path = os.path.join(ROOT_DIR, "thirdparty", "Diffusion_Suction", "diffusion_suctionnet_model")
sys.path.append(thirdparty_path)
sys.path.append(diffusion_model_path)

try:
    from diffusion_suctionnet_model.model import dsnet
    from utils import fps_subsample, pc_normalize
    from preprocess import preprocess  # 导入预处理函数
    print("成功导入模型和工具函数")
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("尝试从当前目录导入...")
    try:
        import sys
        sys.path.append(os.path.join(FILE_DIR, "thirdparty", "Diffusion_Suction", "diffusion_suctionnet_model"))
        from model import dsnet
        from utils import fps_subsample, pc_normalize
        from preprocess import preprocess  # 导入预处理函数
        print("成功从当前目录导入模型和工具函数")
    except ImportError as e2:
        print(f"从当前目录导入也失败: {e2}")
        sys.exit(1)


class SuctionNetInference:
    """
    扩散吸取网络推理类
    用于加载训练好的模型并进行吸取点预测
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 device: str = 'cuda',
                 num_points: int = 16384,
                 diffusion_steps: int = 50):
        """
        初始化推理器
        
        参数:
            checkpoint_path: 模型检查点文件路径
            device: 计算设备 ('cuda' 或 'cpu')
            num_points: 点云采样点数
            diffusion_steps: 扩散推理步数
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_points = num_points
        self.diffusion_steps = diffusion_steps
        
        print(f"使用设备: {self.device}")
        print(f"点云采样点数: {self.num_points}")
        print(f"扩散推理步数: {self.diffusion_steps}")
        
        # 加载模型
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        print("模型加载完成，已设置为推理模式")
    
    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """
        加载训练好的模型
        
        参数:
            checkpoint_path: 检查点文件路径
            
        返回:
            model: 加载的模型
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        print(f"正在加载模型检查点: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 创建模型实例
        model = dsnet(
            use_vis_branch=False,  # 推理模式不需要可见性分支
            return_loss=False,     # 推理模式不返回损失
        )
        
        # 设置扩散推理步数
        model.diffusion_inference_steps = self.diffusion_steps
        
        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载模型权重 (epoch: {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("成功加载模型权重")
        
        model.to(self.device)
        return model
    
    def preprocess_point_cloud(self, 
                             point_cloud: np.ndarray,
                             normals: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        """
        预处理输入点云数据
        
        参数:
            point_cloud: 点云坐标 (N, 3) numpy数组
            normals: 点云法向量 (N, 3) numpy数组，可选
            
        返回:
            inputs: 模型输入字典
        """
        print(f"输入点云形状: {point_cloud.shape}")
        
        # 点云归一化
        point_cloud = pc_normalize(point_cloud)
        
        # 如果点数超过目标数量，进行下采样
        if point_cloud.shape[0] > self.num_points:
            # 使用FPS采样
            indices = fps_subsample(torch.from_numpy(point_cloud).float(), self.num_points)
            point_cloud = point_cloud[indices]
            if normals is not None:
                normals = normals[indices]
        elif point_cloud.shape[0] < self.num_points:
            # 如果点数不足，进行重复采样
            repeat_times = (self.num_points + point_cloud.shape[0] - 1) // point_cloud.shape[0]
            point_cloud = np.tile(point_cloud, (repeat_times, 1))[:self.num_points]
            if normals is not None:
                normals = np.tile(normals, (repeat_times, 1))[:self.num_points]
        
        # 如果没有提供法向量，创建零向量
        if normals is None:
            normals = np.zeros((self.num_points, 3), dtype=np.float32)
            print("警告: 未提供法向量，使用零向量")
        
        print(f"预处理后点云形状: {point_cloud.shape}")
        print(f"法向量形状: {normals.shape}")
        
        # 转换为tensor并添加batch维度
        point_cloud_tensor = torch.from_numpy(point_cloud).float().unsqueeze(0).to(self.device)
        normals_tensor = torch.from_numpy(normals).float().unsqueeze(0).to(self.device)
        
        # 创建输入字典
        inputs = {
            'point_clouds': point_cloud_tensor,  # (1, num_points, 3)
            'labels': {
                'suction_or': normals_tensor  # (1, num_points, 3)
            }
        }
        
        return inputs
    
    def predict(self, point_cloud: np.ndarray, 
                normals: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        对输入点云进行吸取点预测
        
        参数:
            point_cloud: 点云坐标 (N, 3) numpy数组
            normals: 点云法向量 (N, 3) numpy数组，可选
            
        返回:
            results: 预测结果字典，包含各种吸取评分
            preprocessed_pc: 预处理后的点云坐标
        """
        print("开始进行吸取点预测...")
        start_time = time.time()
        
        # 预处理输入数据
        inputs = self.preprocess_point_cloud(point_cloud, normals)
        
        # 获取预处理后的点云坐标（用于后续的最佳点定位）
        preprocessed_pc = inputs['point_clouds'].squeeze(0).cpu().numpy()  # (num_points, 3)
        
        # 临时修改模型的pipeline调用，使用动态点数
        original_forward = self.model.forward
        
        def patched_forward(inputs_dict):
            """修补的forward函数，支持动态点数"""
            batch_size = inputs_dict['point_clouds'].shape[0]
            num_point = inputs_dict['point_clouds'].shape[1]
            
            # pointnet++提取堆叠场景点云
            input_points = inputs_dict['point_clouds']
            input_points = torch.cat((input_points, inputs_dict['labels']['suction_or']), dim=2)
            features, global_features = self.model.backbone(input_points)

            # 推理模式
            pred_results = self.model.pipeline(   
                batch_size=batch_size,
                device=features.device,
                dtype=features.dtype,
                shape=(num_point, 4),  # 使用动态点数
                features=features,
                num_inference_steps=self.model.diffusion_inference_steps,
            )
            ddim_loss = None
            return pred_results, ddim_loss
        
        # 暂时替换forward方法
        self.model.forward = patched_forward
        
        try:
            # 模型推理
            with torch.no_grad():
                pred_results, _ = self.model(inputs)
        finally:
            # 恢复原始forward方法
            self.model.forward = original_forward
        
        inference_time = time.time() - start_time
        print(f"推理完成，用时: {inference_time:.3f}秒")
        
        # 后处理结果
        results = self.postprocess_results(pred_results)
        
        return results, preprocessed_pc
    
    def postprocess_results(self, pred_results: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        后处理模型预测结果
        
        参数:
            pred_results: 模型输出张量 (1, num_points, 4)
            
        返回:
            results: 处理后的结果字典
        """
        # 移除batch维度并转换为numpy
        pred_results = pred_results.squeeze(0).cpu().numpy()  # (num_points, 4)
        
        print(f"预测结果形状: {pred_results.shape}")
        print(f"各评分范围:")
        
        results = {
            'suction_seal_scores': pred_results[:, 0],      # 密封评分
            'suction_wrench_scores': pred_results[:, 1],    # 扭矩评分  
            'suction_feasibility_scores': pred_results[:, 2], # 可行性评分
            'object_size_scores': pred_results[:, 3]         # 物体尺寸评分
        }
        
        # 打印各评分的统计信息
        for key, values in results.items():
            print(f"  {key}: [{values.min():.3f}, {values.max():.3f}], 均值: {values.mean():.3f}")
        
        return results
    
    def get_best_suction_points(self, 
                               results: Dict[str, np.ndarray],
                               preprocessed_point_cloud: np.ndarray,  # 使用预处理后的点云
                               top_k: int = 10,
                               score_weights: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        获取最佳吸取点
        
        参数:
            results: 预测结果字典
            preprocessed_point_cloud: 预处理后的点云坐标（与预测结果对应）
            top_k: 返回前k个最佳点
            score_weights: 各评分权重
            
        返回:
            best_points: 最佳吸取点列表
        """
        if score_weights is None:
            score_weights = {
                'suction_seal_scores': 0.3,
                'suction_wrench_scores': 0.3,
                'suction_feasibility_scores': 0.3,
                'object_size_scores': 0.1
            }
        
        # 计算综合评分
        composite_score = np.zeros(len(results['suction_seal_scores']))
        for score_name, weight in score_weights.items():
            if score_name in results:
                composite_score += weight * results[score_name]
        
        # 获取top-k索引
        top_indices = np.argsort(composite_score)[-top_k:][::-1]
        
        best_points = []
        for i, idx in enumerate(top_indices):
            point_info = {
                'rank': i + 1,
                'index': int(idx),
                'position': preprocessed_point_cloud[idx].tolist(),  # 使用预处理后的点云
                'composite_score': float(composite_score[idx]),
                'suction_seal_score': float(results['suction_seal_scores'][idx]),
                'suction_wrench_score': float(results['suction_wrench_scores'][idx]),
                'suction_feasibility_score': float(results['suction_feasibility_scores'][idx]),
                'object_size_score': float(results['object_size_scores'][idx])
            }
            best_points.append(point_info)
        
        return best_points
    
    def save_results(self, 
                    results: Dict[str, np.ndarray],
                    point_cloud: np.ndarray,
                    output_path: str,
                    best_points: Optional[List[Dict]] = None):
        """
        保存预测结果到文件
        
        参数:
            results: 预测结果
            point_cloud: 点云坐标
            output_path: 输出文件路径
            best_points: 最佳吸取点信息
        """
        print(f"正在保存结果到: {output_path}")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为numpy格式
        np.savez(output_path, 
                point_cloud=point_cloud,
                **results)
        
        # 如果有最佳点信息，保存为JSON
        if best_points is not None:
            json_path = output_path.replace('.npz', '_best_points.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(best_points, f, indent=2, ensure_ascii=False)
            print(f"最佳吸取点信息已保存到: {json_path}")
        
        print("结果保存完成")


def load_data_from_images(rgb_path: str, depth_path: str, mask_path: str, camera_info_path: str, params_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    从RGB图像、深度图像、掩码图像和相机信息文件加载点云数据
    
    参数:
        rgb_path: RGB图像文件路径
        depth_path: 深度图像文件路径  
        mask_path: 掩码图像文件路径
        camera_info_path: 相机信息文件路径
        params_path: 参数配置文件路径，可选
        
    返回:
        point_cloud: 点云坐标 (N, 3)
        normals: 法向量 (N, 3)
    """
    print(f"正在从图像文件加载数据:")
    print(f"  RGB图像: {rgb_path}")
    print(f"  深度图像: {depth_path}")
    print(f"  掩码图像: {mask_path}")
    print(f"  相机信息: {camera_info_path}")
    
    # 检查文件是否存在
    for file_path in [rgb_path, depth_path, mask_path, camera_info_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 使用预处理函数生成点云和法向量
    try:
        point_cloud, normals = preprocess(rgb_path, depth_path, mask_path, camera_info_path, params_path)
        print(f"成功生成点云: {point_cloud.shape}, 法向量: {normals.shape}")
        return point_cloud, normals
    except Exception as e:
        raise RuntimeError(f"预处理失败: {e}")





def main():
    """主函数 - 命令行推理接口"""
    parser = argparse.ArgumentParser(description='扩散吸取网络推理')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点文件路径')
    
    # 输入图像参数
    parser.add_argument('--rgb', type=str, required=True,
                       help='RGB图像文件路径')
    parser.add_argument('--depth', type=str, required=True,
                       help='深度图像文件路径')
    parser.add_argument('--mask', type=str, required=True,
                       help='掩码图像文件路径')
    parser.add_argument('--camera_info', type=str, required=True,
                       help='相机信息文件路径')
    parser.add_argument('--params', type=str, 
                       help='参数配置文件路径')
    
    parser.add_argument('--output', type=str, default='./inference_results.npz',
                       help='输出结果文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--num_points', type=int, default=16384,
                       help='点云采样点数')
    parser.add_argument('--diffusion_steps', type=int, default=50,
                       help='扩散推理步数')
    parser.add_argument('--top_k', type=int, default=10,
                       help='返回前k个最佳吸取点')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("扩散吸取网络推理系统")
    print("=" * 60)
    
    # 创建推理器
    inferencer = SuctionNetInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_points=args.num_points,
        diffusion_steps=args.diffusion_steps
    )
    
    # 从图像文件加载数据
    print(f"\n正在从图像文件加载数据...")
    point_cloud, normals = load_data_from_images(
        args.rgb, args.depth, args.mask, args.camera_info, args.params
    )
    
    # 执行推理
    print(f"\n开始推理...")
    results, preprocessed_pc = inferencer.predict(point_cloud, normals)
    
    # 获取最佳吸取点
    print(f"\n计算最佳吸取点 (top-{args.top_k})...")
    best_points = inferencer.get_best_suction_points(
        results, preprocessed_pc, top_k=args.top_k
    )
    
    # 打印最佳吸取点
    print(f"\n最佳吸取点:")
    for point in best_points[:5]:  # 只打印前5个
        # 查找对应的点云位置
        position = point['position']
        index = np.argmin(np.sum((preprocessed_pc - position) ** 2, axis=1))
        
        print(f"  第{point['rank']}名: 位置{point['position']}, 序号{index}, 实际位置: {point_cloud[index]}, "
              f"综合评分: {point['composite_score']:.3f}")
    
    # 保存结果
    print(f"\n保存结果...")
    inferencer.save_results(results, preprocessed_pc, args.output, best_points)
    
    # 保存原始点云为txt文件
    output_dir = os.path.dirname(args.output)
    output_name = os.path.splitext(os.path.basename(args.output))[0]
    point_cloud_txt = os.path.join(output_dir, f"{output_name}_original_pointcloud.txt")
    
    print(f"保存原始点云到: {point_cloud_txt}")
    # 保存格式: x y z nx ny nz (坐标 + 法向量)
    if normals is not None:
        combined_data = np.hstack([point_cloud, normals])
        header = "x y z nx ny nz"
    else:
        combined_data = point_cloud
        header = "x y z"
    
    np.savetxt(point_cloud_txt, combined_data, fmt='%.6f', header=header, comments='# ')
    print(f"原始点云已保存到: {point_cloud_txt}")
    
    print(f"\n推理完成！结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
