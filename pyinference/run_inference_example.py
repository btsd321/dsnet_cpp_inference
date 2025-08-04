#!/usr/bin/env python3
"""
扩散吸取网络推理示例脚本
演示如何使用修改后的推理系统
"""

import os
import subprocess
import sys

def run_inference_from_images():
    """使用图像文件进行推理的示例"""
    
    # 设置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输入文件路径
    rgb_path = os.path.join(current_dir, "input/rgb_images/test_bgr_30-30.png")
    depth_path = os.path.join(current_dir, "input/depth_images/test_depth_30-30.png")
    mask_path = os.path.join(current_dir, "input/segment_single_images/test_segment_30-30-26.png")
    camera_info_path = os.path.join(current_dir, "input/camera_info.yaml")
    params_path = os.path.join(current_dir, "input/parameter.json")
    
    # 模型检查点路径 (需要根据实际情况修改)
    checkpoint_path = os.path.join(current_dir, "input/model/checkpoint.tar")
    
    # 输出路径
    output_path = os.path.join(current_dir, "output/inference_results_from_images.npz")
    
    # 检查输入文件是否存在
    required_files = [rgb_path, depth_path, mask_path, camera_info_path, params_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("错误：以下必需文件不存在：")
        for f in missing_files:
            print(f"  - {f}")
        return False
        
    if not os.path.exists(checkpoint_path):
        print(f"警告：模型检查点文件不存在: {checkpoint_path}")
        print("请将训练好的模型文件放在该位置，或修改路径")
        return False
    
    # 构建推理命令
    inference_script = os.path.join(current_dir, "inference.py")
    cmd = [
        sys.executable, inference_script,
        "--checkpoint", checkpoint_path,
        "--rgb", rgb_path,
        "--depth", depth_path,
        "--mask", mask_path,
        "--camera_info", camera_info_path,
        "--params", params_path,
        "--output", output_path,
        "--device", "cuda",
        "--num_points", "16384",
        "--diffusion_steps", "50",
        "--top_k", "10"
    ]
    
    print("执行推理命令:")
    print(" ".join(cmd))
    print()
    
    try:
        # 执行推理
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("推理成功完成！")
            print("输出:")
            print(result.stdout)
            return True
        else:
            print("推理失败！")
            print("错误信息:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"执行推理时出错: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("扩散吸取网络推理示例")
    print("=" * 60)
    
    # 创建输出目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n正在从图像文件进行推理...")
    success = run_inference_from_images()
    
    print("\n" + "=" * 60)
    print("推理示例完成")
    print(f"图像推理: {'成功' if success else '失败'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
