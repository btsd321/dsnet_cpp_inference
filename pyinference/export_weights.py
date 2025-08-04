#!/usr/bin/env python3
"""
DSNet权重导出脚本
从checkpoint.tar文件中提取模型权重并保存为C++可读的格式
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# 添加模型路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "diffusion_suctionnet_model"))

try:
    from diffusion_suctionnet_model.model import dsnet
    print("成功导入模型")
except ImportError as e:
    print(f"导入模型失败: {e}")
    sys.exit(1)


def save_tensor_as_binary(tensor: torch.Tensor, filepath: str):
    """保存tensor为二进制文件"""
    if tensor.dtype == torch.float32:
        np_array = tensor.detach().cpu().numpy().astype(np.float32)
    else:
        np_array = tensor.detach().cpu().float().numpy().astype(np.float32)
    
    with open(filepath, 'wb') as f:
        # 写入形状信息
        shape = np.array(np_array.shape, dtype=np.int32)
        f.write(shape.tobytes())
        # 写入数据
        f.write(np_array.tobytes())
    
    return np_array.shape


def save_tensor_as_text(tensor: torch.Tensor, filepath: str):
    """保存tensor为文本文件（用于调试）"""
    if tensor.dtype == torch.float32:
        np_array = tensor.detach().cpu().numpy()
    else:
        np_array = tensor.detach().cpu().float().numpy()
    
    np.savetxt(filepath, np_array.flatten(), fmt='%.8f')
    return np_array.shape


def extract_pointnet_weights(model: dsnet, output_dir: str):
    """提取PointNet++骨干网络权重"""
    print("提取PointNet++骨干网络权重...")
    
    pointnet_dir = os.path.join(output_dir, "pointnet")
    os.makedirs(pointnet_dir, exist_ok=True)
    
    weights_info = {}
    
    # 提取backbone权重
    backbone = model.backbone
    state_dict = backbone.state_dict()
    
    # 保存所有权重
    for name, param in state_dict.items():
        print(f"  提取权重: {name}, 形状: {param.shape}")
        
        # 保存为二进制文件
        bin_path = os.path.join(pointnet_dir, f"{name.replace('.', '_')}.bin")
        shape = save_tensor_as_binary(param, bin_path)
        
        # 保存为文本文件（调试用）
        txt_path = os.path.join(pointnet_dir, f"{name.replace('.', '_')}.txt")
        save_tensor_as_text(param, txt_path)
        
        weights_info[name] = {
            "shape": list(shape),
            "file": f"{name.replace('.', '_')}.bin"
        }
    
    # 保存权重信息
    with open(os.path.join(pointnet_dir, "weights_info.json"), 'w') as f:
        json.dump(weights_info, f, indent=2)
    
    print(f"PointNet++权重保存完成，共{len(weights_info)}个权重文件")
    return weights_info


def extract_diffusion_weights(model: dsnet, output_dir: str):
    """提取扩散模型权重"""
    print("提取扩散模型权重...")
    
    diffusion_dir = os.path.join(output_dir, "diffusion")
    os.makedirs(diffusion_dir, exist_ok=True)
    
    weights_info = {}
    
    # 提取ScheduledCNNRefine模型权重
    diffusion_model = model.model
    state_dict = diffusion_model.state_dict()
    
    for name, param in state_dict.items():
        print(f"  提取权重: {name}, 形状: {param.shape}")
        
        # 保存为二进制文件
        bin_path = os.path.join(diffusion_dir, f"{name.replace('.', '_')}.bin")
        shape = save_tensor_as_binary(param, bin_path)
        
        # 保存为文本文件（调试用）
        txt_path = os.path.join(diffusion_dir, f"{name.replace('.', '_')}.txt")
        save_tensor_as_text(param, txt_path)
        
        weights_info[name] = {
            "shape": list(shape),
            "file": f"{name.replace('.', '_')}.bin"
        }
    
    # 保存权重信息
    with open(os.path.join(diffusion_dir, "weights_info.json"), 'w') as f:
        json.dump(weights_info, f, indent=2)
    
    print(f"扩散模型权重保存完成，共{len(weights_info)}个权重文件")
    return weights_info


def extract_model_config(model: dsnet, output_dir: str):
    """提取模型配置信息"""
    print("提取模型配置...")
    
    config = {
        "pointnet_config": {
            "npoint_per_layer": [4096, 1024, 256, 64],
            "radius_per_layer": [[10, 20, 30], [30, 45, 60], [60, 80, 120], [120, 160, 240]],
            "input_feature_dims": 3,
            "backbone_feature_dim": 128
        },
        "diffusion_config": {
            "channels_in": 128,
            "channels_noise": 4,
            "diffusion_inference_steps": model.diffusion_inference_steps,
            "num_train_timesteps": 1000,
            "bit_scale": model.bit_scale
        },
        "scheduler_config": {
            "num_train_timesteps": 1000,
            "clip_sample": False
        }
    }
    
    with open(os.path.join(output_dir, "model_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("模型配置保存完成")
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DSNet权重导出工具')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='checkpoint.tar文件路径')
    parser.add_argument('--output', type=str, default='./exported_weights',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'], help='设备类型')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.checkpoint):
        print(f"错误: checkpoint文件不存在: {args.checkpoint}")
        return 1
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DSNet权重导出工具")
    print("=" * 60)
    print(f"输入文件: {args.checkpoint}")
    print(f"输出目录: {args.output}")
    print(f"设备: {args.device}")
    
    try:
        # 加载模型
        print("\n加载模型...")
        device = torch.device(args.device)
        
        # 创建模型实例
        model = dsnet(
            use_vis_branch=False,
            return_loss=False
        )
        
        # 加载权重
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载模型权重 (epoch: {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("成功加载模型权重")
        
        model.eval()
        model.to(device)
        
        # 提取权重
        print("\n开始提取权重...")
        
        # 提取PointNet++权重
        pointnet_info = extract_pointnet_weights(model, args.output)
        
        # 提取扩散模型权重
        diffusion_info = extract_diffusion_weights(model, args.output)
        
        # 提取模型配置
        config_info = extract_model_config(model, args.output)
        
        # 保存总体信息
        export_info = {
            "export_timestamp": str(Path().cwd()),
            "checkpoint_path": args.checkpoint,
            "total_pointnet_weights": len(pointnet_info),
            "total_diffusion_weights": len(diffusion_info),
            "model_config": config_info
        }
        
        with open(os.path.join(args.output, "export_info.json"), 'w') as f:
            json.dump(export_info, f, indent=2)
        
        print("\n" + "=" * 60)
        print("权重导出完成!")
        print(f"PointNet++权重: {len(pointnet_info)}个文件")
        print(f"扩散模型权重: {len(diffusion_info)}个文件")
        print(f"输出目录: {args.output}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n错误: 权重导出失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
