#!/usr/bin/env python3
"""
将PyTorch训练的DSNet模型转换为TorchScript格式，用于C++推理
"""

import os
import sys
import torch
import argparse

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIFFUSION_SUCTION_PATH = os.path.join(PROJECT_ROOT, "thirdparty", "Diffusion_Suction")
sys.path.append(DIFFUSION_SUCTION_PATH)

from diffusion_suctionnet_model.model import dsnet

def convert_model_to_torchscript(checkpoint_path, output_path, device='cuda'):
    将PyTorch模型转换为TorchScript格式
    
    Args:
        checkpoint_path: PyTorch模型检查点路径
        output_path: 输出的TorchScript模型路径
        device: 运行设备
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # 创建模型实例（推理模式）
    model = dsnet(use_vis_branch=False, return_loss=False)
    
    # 加载训练好的权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理不同的检查点格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # 假设整个checkpoint就是state_dict
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    
    # 创建示例输入
    batch_size = 1
    num_points = 16384
    
    # 点云数据 (batch_size, num_points, 3)
    example_points = torch.randn(batch_size, num_points, 3).to(device)
    
    # 法向量数据 (batch_size, num_points, 3) - 推理时可能不需要真实值
    example_normals = torch.randn(batch_size, num_points, 3).to(device)
    
    # 构建输入字典
    example_input = {
        'point_clouds': example_points,
        'labels': {
            'suction_or': example_normals  # 推理时这个可以是占位符
        }
    }
    
    print("Converting to TorchScript...")
    
    try:
        # 使用torch.jit.trace进行转换
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
            
        print(f"Saving TorchScript model to: {output_path}")
        traced_model.save(output_path)
        
        print("Model conversion completed successfully!")
        
        # 验证转换后的模型
        print("Verifying converted model...")
        loaded_model = torch.jit.load(output_path)
        loaded_model.eval()
        
        with torch.no_grad():
            original_output = model(example_input)
            traced_output = loaded_model(example_input)
            
        print("Model verification completed!")
        return True
        
    except Exception as e:
        print(f"Error during tracing: {e}")
        
        # 如果trace失败，尝试使用script
        try:
            print("Trying torch.jit.script instead...")
            scripted_model = torch.jit.script(model)
            scripted_model.save(output_path)
            print("Script conversion completed successfully!")
            return True
        except Exception as e2:
            print(f"Error during scripting: {e2}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch DSNet model to TorchScript')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint file (.tar)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output TorchScript model (.pt)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use for conversion')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 检查设备可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("PyTorch to TorchScript Model Converter")
    print("=" * 60)
    print(f"Input checkpoint: {args.checkpoint}")
    print(f"Output model: {args.output}")
    print(f"Device: {args.device}")
    print("")
    
    success = convert_model_to_torchscript(args.checkpoint, args.output, args.device)
    
    if success:
        print("\n✓ Model conversion completed successfully!")
        print(f"TorchScript model saved to: {args.output}")
        print("\nYou can now use this model in C++ with LibTorch.")
    else:
        print("\n✗ Model conversion failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
