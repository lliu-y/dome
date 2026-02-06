# ========================================
# A0-Baseline: 线稿Concat输入（无参考图）
# ========================================

## A0 训练（从头开始）
python main.py --config configs/A0-Baseline.yaml --train --sample_at_start --gpu_ids 0

## A0 断点重训（从checkpoint继续）
python main.py --config configs/A0-Baseline.yaml --train --gpu_ids 0 \
--resume_model results/anime_colorization/A0-Baseline/checkpoint/last_model.pth \
--resume_optim results/anime_colorization/A0-Baseline/checkpoint/last_optim_sche.pth

## A0 断点重训（从早停模型继续）
python main.py --config configs/A0-Baseline.yaml --train --gpu_ids 0 \
--resume_model results/anime_colorization/A0-Baseline/checkpoint/early_stop_model.pth \
--resume_optim results/anime_colorization/A0-Baseline/checkpoint/early_stop_optim_sche.pth

## A0 推理采样（生成测试集结果）
python main.py --config configs/A0-Baseline.yaml --sample_to_eval --gpu_ids 0 \
--resume_model results/anime_colorization/A0-Baseline/checkpoint/early_stop_model.pth

## A0 推理采样（指定checkpoint）
python main.py --config configs/A0-Baseline.yaml --sample_to_eval --gpu_ids 0 \
--resume_model results/anime_colorization/A0-Baseline/checkpoint/model_epoch_XX.pth


# ========================================
# A1-CrossAttn: A0 + 参考图Cross-Attention
# ========================================

## A1 训练（从头开始）
python main.py --config configs/A1-CrossAttn.yaml --train --sample_at_start --gpu_ids 0

## A1 断点重训（从checkpoint继续）
python main.py --config configs/A1-CrossAttn.yaml --train --gpu_ids 0 \
--resume_model results/anime_colorization/A1-CrossAttn/checkpoint/last_model.pth \
--resume_optim results/anime_colorization/A1-CrossAttn/checkpoint/last_optim_sche.pth

## A1 断点重训（从早停模型继续）
python main.py --config configs/A1-CrossAttn.yaml --train --gpu_ids 0 \
--resume_model results/anime_colorization/A1-CrossAttn/checkpoint/early_stop_model.pth \
--resume_optim results/anime_colorization/A1-CrossAttn/checkpoint/early_stop_optim_sche.pth

## A1 推理采样（生成测试集结果）
python main.py --config configs/A1-CrossAttn.yaml --sample_to_eval --gpu_ids 0 \
--resume_model results/anime_colorization/A1-CrossAttn/checkpoint/early_stop_model.pth

## A1 推理采样（指定checkpoint）
python main.py --config configs/A1-CrossAttn.yaml --sample_to_eval --gpu_ids 0 \
--resume_model results/anime_colorization/A1-CrossAttn/checkpoint/model_epoch_XX.pth


# ========================================
# 评估指标计算（统一脚本）
# ========================================

## A0 完整评估（FID + LPIPS + PSNR + SSIM）
python evaluation/evaluate_all.py \
  --result_dir results/anime_colorization/A0-Baseline/sample_to_eval/200 \
  --gt_dir results/anime_colorization/A0-Baseline/sample_to_eval/ground_truth \
  --metrics fid lpips psnr ssim \
  --output results/anime_colorization/A0-Baseline/metrics.json

## A1 完整评估（FID + LPIPS + PSNR + SSIM）
python evaluation/evaluate_all.py \
  --result_dir results/anime_colorization/A1-CrossAttn/sample_to_eval/200 \
  --gt_dir results/anime_colorization/A1-CrossAttn/sample_to_eval/ground_truth \
  --metrics fid lpips psnr ssim \
  --output results/anime_colorization/A1-CrossAttn/metrics.json

## 仅计算特定指标（示例）
# 仅FID
python evaluation/evaluate_all.py \
  --result_dir results/anime_colorization/A0-Baseline/sample_to_eval/200 \
  --gt_dir results/anime_colorization/A0-Baseline/sample_to_eval/ground_truth \
  --metrics fid

# 仅LPIPS + PSNR
python evaluation/evaluate_all.py \
  --result_dir results/anime_colorization/A0-Baseline/sample_to_eval/200 \
  --gt_dir results/anime_colorization/A0-Baseline/sample_to_eval/ground_truth \
  --metrics lpips psnr


# ========================================
# 实用工具
# ========================================

## 重命名样本文件
python preprocess_and_evaluation.py -f rename_samples \
  -r results/anime_colorization/A0-Baseline \
  -s sample_to_eval/200 -t renamed_samples

## 复制样本文件
python preprocess_and_evaluation.py -f copy_samples \
  -r results/anime_colorization/A0-Baseline \
  -s sample_to_eval/200 -t copied_samples


# ========================================
# 消融对比可视化（A0 vs A1）
# ========================================

## 将A0和A1结果放在同一目录下对比
# mkdir -p comparison/A0 comparison/A1 comparison/GT
# cp results/anime_colorization/A0-Baseline/sample_to_eval/200/*.png comparison/A0/
# cp results/anime_colorization/A1-CrossAttn/sample_to_eval/200/*.png comparison/A1/
# cp dataset/test/color/*.png comparison/GT/


#autodl 训练相关

# 关闭默认的 tensorboard
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
#启动 tensorboard
tensorboard --port 6007 --logdir xxx

#自动关闭

# 假设您的程序原执行命令为
python train.py
# 那么可以在您的程序后跟上shutdown命令
python train.py; /usr/bin/shutdown      # 用;拼接意味着前边的指令不管执行成功与否，都会执行shutdown命令
python train.py && /usr/bin/shutdown    # 用&&拼接表示前边的命令执行成功后才会执行shutdown。请根据自己的需要选择

#使用screen
# 创建一个新的screen会话，命名为my_session
screen -S my_session
# 在screen会话中执行训练命令, 并输出到日志文件
python train.py > train.log 2>&1
# 退出screen会话（按Ctrl + A + D）
# 实时查看日志文件
tail -f train.log