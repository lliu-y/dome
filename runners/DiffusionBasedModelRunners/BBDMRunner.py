import os

import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

from PIL import Image
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from tqdm.autonotebook import tqdm
from torchsummary import summary


@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def _unpack_batch(self, batch):
        """
        统一解包batch，兼容tuple和dict两种格式
        
        Returns:
            gt: 目标彩图 [B, 3, H, W]
            lineart: 线稿 [B, 1, H, W] 或 [B, 3, H, W]
            ref: 参考图 [B, 3, H, W] 或 None
            name: 文件名
        """
        if isinstance(batch, dict):
            # 新数据集格式 (anime_colorization)
            gt = batch['GT']
            lineart = batch['lineart']
            ref = batch.get('distorted', None)
            name = batch['name']
        else:
            # 原有数据集格式 (x, x_name), (x_cond, x_cond_name)
            (gt, name), (lineart, _) = batch
            ref = None
        return gt, lineart, ref, name

    def _prepare_cond(self, lineart):
        """
        准备条件输入：将线稿转换为布朗桥起点 y（采样端点）
        
        Args:
            lineart: [B, C, H, W] 线稿，C=1或3
        Returns:
            y_cond: [B, 3, H, W] 布朗桥起点（采样时从此出发，t→T时到达）
        """
        if lineart.shape[1] == 1:
            # 1通道灰度线稿，repeat为3通道
            return lineart.repeat(1, 3, 1, 1)
        else:
            # 已经是3通道
            return lineart

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM":
            bbdmnet = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        self.logger("Total Number of parameter: %.2fM" % (total_num / 1e6))
        self.logger("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        
        lr_config = config.model.BB.lr_scheduler
        scheduler_type = getattr(lr_config, 'type', 'ReduceLROnPlateau')
        
        if scheduler_type == 'CosineAnnealingLR':
            T_max = getattr(lr_config, 'T_max', -1)
            eta_min = getattr(lr_config, 'eta_min', 5e-7)
            
            if T_max == -1:
                # 延迟初始化：返回配置信息，在 train() 中计算实际 T_max 后创建调度器
                # 使用占位符标记需要延迟初始化
                scheduler = {
                    '_deferred_': True,
                    'type': 'CosineAnnealingLR',
                    'optimizer': optimizer,
                    'eta_min': eta_min
                }
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=T_max,
                    eta_min=eta_min
                )
        else:
            # 默认使用 ReduceLROnPlateau（向后兼容）
            # 过滤掉 type 字段，只保留 ReduceLROnPlateau 支持的参数
            plateau_params = {k: v for k, v in vars(lr_config).items() 
                              if k not in ['type', 'T_max', 'eta_min']}
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='min',
                threshold_mode='rel',
                **plateau_params
            )
        
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            if self.config.training.use_DDP:
                model_states['ori_latent_mean'] = self.net.module.ori_latent_mean
                model_states['ori_latent_std'] = self.net.module.ori_latent_std
                model_states['cond_latent_mean'] = self.net.module.cond_latent_mean
                model_states['cond_latent_std'] = self.net.module.cond_latent_std
            else:
                model_states['ori_latent_mean'] = self.net.ori_latent_mean
                model_states['ori_latent_std'] = self.net.ori_latent_std
                model_states['cond_latent_mean'] = self.net.cond_latent_mean
                model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        self.logger(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        self.logger(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)
            # break

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        self.logger(self.net.ori_latent_mean)
        self.logger(self.net.ori_latent_std)
        self.logger(self.net.cond_latent_mean)
        self.logger(self.net.cond_latent_std)

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        # 统一解包batch (兼容tuple和dict格式)
        gt, lineart, ref, name = self._unpack_batch(batch)
        
        # 移动到设备
        x = gt.to(self.config.training.device[0])
        x_cond = self._prepare_cond(lineart).to(self.config.training.device[0])
        
        # A1+: 提前编码参考图为context（避免在forward中重复编码）
        context = None
        if ref is not None and hasattr(net, 'ref_encoder') and net.ref_encoder is not None:
            ref = ref.to(self.config.training.device[0])
            context = net.ref_encoder(ref)
            
            # Reference Dropout (10%概率): Classifier-Free Guidance 训练策略
            # 关键改进:
            # 1. 样本级别dropout (已实现): 每个样本独立决定是否drop，批次内混合状态
            # 2. 可学习null_embed (新增): 用语义中性的embedding替代零向量
            if stage == 'train':
                drop_prob = 0.1
                # drop_mask: [B, 1, 1] -> 广播到 [B, M, C]，每个样本独立drop
                drop_mask = torch.rand(context.shape[0], 1, 1, device=context.device) < drop_prob
                
                # 获取可学习的"无条件"嵌入 (优于零向量)
                null_context = net.ref_encoder.get_null_context(context.shape[0])
                
                # 使用torch.where替换: drop的样本用null_embed，保留的用真实context
                context = torch.where(drop_mask, null_context, context)

        # 前向传播计算损失
        # A0: context=None, condition_key="nocond"
        # A1+: context=[B,M,C], condition_key="crossattn" (10% dropout训练)
        loss, additional_info = net(x, x_cond, context=context)
        
        if write and self.is_main_process:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            if additional_info.__contains__('recloss_noise'):
                self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
            if additional_info.__contains__('recloss_xy'):
                self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        reverse_sample_path = make_dir(os.path.join(sample_path, 'reverse_sample'))
        reverse_one_step_path = make_dir(os.path.join(sample_path, 'reverse_one_step_samples'))

        print(sample_path)

        # 统一解包batch (兼容tuple和dict格式)
        gt, lineart, ref, name = self._unpack_batch(batch)
        x = gt
        x_cond = self._prepare_cond(lineart)

        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond = x_cond[0:batch_size].to(self.config.training.device[0])
        
        # 保存原始线稿用于可视化 (如果是1通道)
        lineart_vis = lineart[0:batch_size].to(self.config.training.device[0])

        grid_size = 4

        # samples, one_step_samples = net.sample(x_cond,
        #                                        clip_denoised=self.config.testing.clip_denoised,
        #                                        sample_mid_step=True)
        # self.save_images(samples, reverse_sample_path, grid_size, save_interval=200,
        #                  writer_tag=f'{stage}_sample' if stage != 'test' else None)
        #
        # self.save_images(one_step_samples, reverse_one_step_path, grid_size, save_interval=200,
        #                  writer_tag=f'{stage}_one_step_sample' if stage != 'test' else None)
        #
        # sample = samples[-1]
        
        # 生成 context（A1: 用参考图编码特征，A0: None）
        context = None
        if ref is not None and hasattr(net, 'ref_encoder') and net.ref_encoder is not None:
            context = net.ref_encoder(ref[0:batch_size].to(self.config.training.device[0]))
        
        sample = net.sample(x_cond, context=context, clip_denoised=self.config.testing.clip_denoised).to('cpu')
        image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x_cond.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'condition.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')
        
        # 保存线稿可视化 (如果是1通道，repeat为3通道用于显示)
        lineart_for_vis = lineart_vis.repeat(1, 3, 1, 1) if lineart_vis.shape[1] == 1 else lineart_vis
        image_grid = get_image_grid(lineart_for_vis.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'lineart.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_lineart', image_grid, self.global_step, dataformats='HWC')

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        condition_path = make_dir(os.path.join(sample_path, f'condition'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))
        result_path = make_dir(os.path.join(sample_path, str(self.config.model.BB.params.sample_step)))

        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        batch_size = self.config.data.test.batch_size
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = self.config.testing.sample_num
        for test_batch in pbar:
            # 统一解包batch (兼容tuple和dict格式)
            gt, lineart, ref, name = self._unpack_batch(test_batch)
            x = gt.to(self.config.training.device[0])
            x_cond = self._prepare_cond(lineart).to(self.config.training.device[0])
            
            # A1+: 提前编码参考图为context
            context = None
            if ref is not None and hasattr(net, 'ref_encoder') and net.ref_encoder is not None:
                context = net.ref_encoder(ref.to(self.config.training.device[0]))
            
            # 获取文件名列表
            if isinstance(name, (list, tuple)):
                x_name = name
            else:
                x_name = [name] * batch_size if isinstance(name, str) else name

            for j in range(sample_num):
                sample = net.sample(x_cond, context=context, clip_denoised=False)
                # sample = net.sample_vqgan(x)
                for i in range(batch_size):
                    condition = x_cond[i].detach().clone()
                    gt_img = x[i]
                    result = sample[i]
                    img_name = x_name[i] if i < len(x_name) else f'img_{i}'
                    if j == 0:
                        save_single_image(condition, condition_path, f'{img_name}.png', to_normal=to_normal)
                        save_single_image(gt_img, gt_path, f'{img_name}.png', to_normal=to_normal)
                    if sample_num > 1:
                        result_path_i = make_dir(os.path.join(result_path, img_name))
                        save_single_image(result, result_path_i, f'output_{j}.png', to_normal=to_normal)
                    else:
                        save_single_image(result, result_path, f'{img_name}.png', to_normal=to_normal)
