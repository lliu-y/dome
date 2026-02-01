"""
统一评估脚本 - 计算 FID, LPIPS, PSNR, SSIM 指标

先生成评估样本:
python main.py --config configs/A0-Baseline.yaml 
    --sample_to_eval 
    --gpu_ids 0 
    --resume_model results\\anime_colorization\\A0-Baseline\\checkpoint\\early_stop_model.pth         

使用示例:
    python evaluation/evaluate_all.py \
        --result_dir results/A0-Baseline/sample_to_eval/200 \
        --gt_dir results/A0-Baseline/sample_to_eval/ground_truth \
        --metrics fid lpips psnr ssim \
        --output results/A0-Baseline/metrics.json
"""

import os
import json
import argparse
from datetime import datetime
from PIL import Image
import numpy as np
from tqdm import tqdm


# ==================== PSNR / SSIM 计算 ====================

def calc_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算 PSNR (Peak Signal-to-Noise Ratio)
    
    Args:
        img1, img2: [H, W, C] uint8 numpy arrays, 范围 [0, 255]
    Returns:
        psnr: float, 单位 dB
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calc_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算 SSIM (Structural Similarity Index)
    
    Args:
        img1, img2: [H, W, C] uint8 numpy arrays
    Returns:
        ssim: float, 范围 [-1, 1], 越高越好
    """
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, channel_axis=-1, data_range=255)


def calc_psnr_ssim_batch(result_dir: str, gt_dir: str) -> tuple:
    """
    批量计算目录下所有图像的 PSNR 和 SSIM
    
    Returns:
        (avg_psnr, avg_ssim): 平均值
    """
    # 获取文件列表 (按文件名匹配)
    result_files = {f for f in os.listdir(result_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    gt_files = {f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    common_files = sorted(result_files & gt_files)
    
    if not common_files:
        raise ValueError(f"No matching files between {result_dir} and {gt_dir}")
    
    if len(common_files) < len(result_files):
        print(f"[Warning] {len(result_files) - len(common_files)} files in result_dir have no matching GT")
    
    psnr_list, ssim_list = [], []
    
    for fname in tqdm(common_files, desc='PSNR/SSIM'):
        img_result = np.array(Image.open(os.path.join(result_dir, fname)).convert('RGB'))
        img_gt = np.array(Image.open(os.path.join(gt_dir, fname)).convert('RGB'))
        
        # 检查尺寸一致性
        if img_result.shape != img_gt.shape:
            print(f"[Warning] Size mismatch for {fname}: {img_result.shape} vs {img_gt.shape}, skipping")
            continue
        
        psnr_list.append(calc_psnr(img_result, img_gt))
        ssim_list.append(calc_ssim(img_result, img_gt))
    
    return np.mean(psnr_list), np.mean(ssim_list)


# ==================== FID / LPIPS 调用 ====================

def calc_fid(result_dir: str, gt_dir: str) -> float:
    """调用现有 FID 实现"""
    import sys
    from pathlib import Path
    # 添加项目根目录到 sys.path
    root_dir = Path(__file__).parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    
    from evaluation.FID import calc_FID
    return calc_FID(result_dir, gt_dir)


def calc_lpips_batch(result_dir: str, gt_dir: str) -> float:
    """
    计算 LPIPS (按文件名匹配，而非数字索引)
    
    注: 原 LPIPS.py 使用数字索引命名，这里改为文件名匹配以兼容任意命名
    """
    import torch
    import lpips
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss_fn = lpips.LPIPS(net='alex', version='0.1').to(device)
    
    # 获取匹配的文件列表
    result_files = {f for f in os.listdir(result_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    gt_files = {f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    common_files = sorted(result_files & gt_files)
    
    if not common_files:
        raise ValueError(f"No matching files between {result_dir} and {gt_dir}")
    
    total_lpips = 0.0
    with torch.no_grad():
        for fname in tqdm(common_files, desc='LPIPS'):
            img_result = lpips.im2tensor(lpips.load_image(os.path.join(result_dir, fname))).to(device)
            img_gt = lpips.im2tensor(lpips.load_image(os.path.join(gt_dir, fname))).to(device)
            total_lpips += loss_fn(img_result, img_gt).item()
    
    return total_lpips / len(common_files)


# ==================== 主函数 ====================

def evaluate(result_dir: str, gt_dir: str, metrics: list) -> dict:
    """
    计算指定的评估指标
    
    Args:
        result_dir: 生成结果目录
        gt_dir: Ground Truth 目录
        metrics: 要计算的指标列表, 可选 ['fid', 'lpips', 'psnr', 'ssim']
    
    Returns:
        dict: {metric_name: value}
    """
    results = {}
    metrics = [m.lower() for m in metrics]
    
    # FID
    if 'fid' in metrics:
        print("\n[1/4] Calculating FID...")
        results['FID'] = calc_fid(result_dir, gt_dir)
    
    # LPIPS
    if 'lpips' in metrics:
        print("\n[2/4] Calculating LPIPS...")
        results['LPIPS'] = calc_lpips_batch(result_dir, gt_dir)
    
    # PSNR / SSIM (一起计算更高效)
    if 'psnr' in metrics or 'ssim' in metrics:
        print("\n[3/4] Calculating PSNR/SSIM...")
        psnr, ssim = calc_psnr_ssim_batch(result_dir, gt_dir)
        if 'psnr' in metrics:
            results['PSNR'] = psnr
        if 'ssim' in metrics:
            results['SSIM'] = ssim
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Unified evaluation script for image generation')
    parser.add_argument('--result_dir', type=str, required=True, help='生成结果目录')
    parser.add_argument('--gt_dir', type=str, required=True, help='Ground Truth 目录')
    parser.add_argument('--metrics', nargs='+', default=['fid', 'lpips', 'psnr', 'ssim'],
                        choices=['fid', 'lpips', 'psnr', 'ssim'], help='要计算的指标')
    parser.add_argument('--output', type=str, default=None, help='保存结果的 JSON 文件路径')
    args = parser.parse_args()
    
    # 目录校验
    if not os.path.isdir(args.result_dir):
        raise FileNotFoundError(f"Result directory not found: {args.result_dir}")
    if not os.path.isdir(args.gt_dir):
        raise FileNotFoundError(f"GT directory not found: {args.gt_dir}")
    
    # 计算指标
    results = evaluate(args.result_dir, args.gt_dir, args.metrics)
    
    # 打印结果
    print("\n" + "=" * 40)
    print("         Evaluation Results")
    print("=" * 40)
    for metric, value in results.items():
        print(f"  {metric:8s}: {value:.4f}")
    print("=" * 40)
    
    # 保存到 JSON
    if args.output:
        output_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'result_dir': args.result_dir,
            'gt_dir': args.gt_dir,
            'metrics': results
        }
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
