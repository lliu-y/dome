import argparse
import datetime
import pdb
import time

import yaml
import os
import traceback

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm

# from evaluation.FID import calc_FID
# from evaluation.LPIPS import calc_LPIPS
from runners.base.EMA import EMA
from runners.utils import make_save_dirs, make_dir, get_dataset, remove_file


class BaseRunner(ABC):
    def __init__(self, config):
        self.net = None  # Neural Network
        self.optimizer = None  # optimizer
        self.scheduler = None  # scheduler
        self.config = config  # config from configuration file
        self.is_main_process = True if (not self.config.training.use_DDP) or self.config.training.local_rank == 0 else False
        # set training params
        self.global_epoch = 0  # global epoch
        if config.args.sample_at_start:
            self.global_step = -1  # global step
        else:
            self.global_step = 0

        self.GAN_buffer = {}  # GAN buffer for Generative Adversarial Network
        self.topk_checkpoints = {}  # Top K checkpoints
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_triggered = False
        self.best_model_path = None
        
        # Parse early stopping config with defaults for backward compatibility
        if hasattr(self.config.training, 'early_stopping'):
            es_config = self.config.training.early_stopping
            self.early_stopping_enabled = getattr(es_config, 'enabled', False)
            self.early_stopping_patience = getattr(es_config, 'patience', 10)
            self.early_stopping_min_delta = getattr(es_config, 'min_delta', 0.0)
        else:
            self.early_stopping_enabled = False
            self.early_stopping_patience = 10
            self.early_stopping_min_delta = 0.0

        # set log and save destination
        self.config.result = argparse.Namespace()
        self.config.result.result_path, \
        self.config.result.image_path, \
        self.config.result.ckpt_path, \
        self.config.result.log_path, \
        self.config.result.sample_path, \
        self.config.result.sample_to_eval_path = make_save_dirs(self.config.args,
                                                                prefix=self.config.data.dataset_name,
                                                                suffix=self.config.model.model_name)

        self.logger("save training results to " + self.config.result.result_path)

        self.save_config()  # save configuration file
        self.writer = SummaryWriter(self.config.result.log_path)  # initialize SummaryWriter

        # initialize model
        self.net, self.optimizer, self.scheduler = self.initialize_model_optimizer_scheduler(self.config)

        self.print_model_summary(self.net)

        # initialize EMA
        self.use_ema = False if not self.config.model.__contains__('EMA') else self.config.model.EMA.use_ema
        if self.use_ema:
            self.ema = EMA(self.config.model.EMA.ema_decay)
            self.update_ema_interval = self.config.model.EMA.update_ema_interval
            self.start_ema_step = self.config.model.EMA.start_ema_step
            self.ema.register(self.net)

        # load model from checkpoint
        self.load_model_from_checkpoint()

        # initialize DDP
        if self.config.training.use_DDP:
            self.net = DDP(self.net, device_ids=[self.config.training.local_rank], output_device=self.config.training.local_rank)
        else:
            self.net = self.net.to(self.config.training.device[0])
        # self.ema.reset_device(self.net)

    # print msg
    def logger(self, msg, **kwargs):
        if self.is_main_process:
            print(msg, **kwargs)

    # save configuration file
    def save_config(self):
        if self.is_main_process:
            save_path = os.path.join(self.config.result.ckpt_path, 'config.yaml')
            save_config = self.config
            with open(save_path, 'w') as f:
                yaml.dump(save_config, f)

    def initialize_model_optimizer_scheduler(self, config, is_test=False):
        """
        get model, optimizer, scheduler
        :param args: args
        :param config: config
        :param is_test: is_test
        :return: net: Neural Network, nn.Module;
                 optimizer: a list of optimizers;
                 scheduler: a list of schedulers or None;
        """
        net = self.initialize_model(config)
        optimizer, scheduler = None, None
        if not is_test:
            optimizer, scheduler = self.initialize_optimizer_scheduler(net, config)
        return net, optimizer, scheduler

    # load model, EMA, optimizer, scheduler from checkpoint
    def load_model_from_checkpoint(self):
        model_states = None
        if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
            self.logger(f"load model {self.config.model.model_name} from {self.config.model.model_load_path}")
            model_states = torch.load(self.config.model.model_load_path, map_location='cpu')

            self.global_epoch = model_states['epoch']
            self.global_step = model_states['step']

            # load model
            self.net.load_state_dict(model_states['model'])

            # load ema
            if self.use_ema:
                self.ema.shadow = model_states['ema']
                self.ema.reset_device(self.net)
            
            # load early stopping state (for resume training)
            if 'best_val_loss' in model_states:
                self.best_val_loss = model_states['best_val_loss']
                self.patience_counter = model_states.get('patience_counter', 0)
                self.best_model_path = model_states.get('best_model_path', None)
                self.logger(f"Restored early stopping state: best_val_loss={self.best_val_loss:.6f}, patience_counter={self.patience_counter}")

            # load optimizer and scheduler
            if self.config.args.train:
                if self.config.model.__contains__('optim_sche_load_path') and self.config.model.optim_sche_load_path is not None:
                    optimizer_scheduler_states = torch.load(self.config.model.optim_sche_load_path, map_location='cpu')
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(optimizer_scheduler_states['optimizer'][i])

                    if self.scheduler is not None:
                        for i in range(len(self.optimizer)):
                            # 跳过延迟初始化的scheduler（dict占位符）
                            if isinstance(self.scheduler[i], dict) and self.scheduler[i].get('_deferred_'):
                                self.logger(f"Skipping scheduler[{i}] state loading (deferred initialization)")
                                continue
                            self.scheduler[i].load_state_dict(optimizer_scheduler_states['scheduler'][i])
        return model_states

    def _check_early_stopping(self, val_loss, epoch):
        """
        Check if early stopping should be triggered.
        
        Args:
            val_loss: Current validation loss
            epoch: Current epoch number
            
        Returns:
            bool: True if early stopping is triggered
        """
        if not self.early_stopping_enabled:
            return False
        
        # Check if validation loss improved
        if val_loss < self.best_val_loss - self.early_stopping_min_delta:
            # Improved: reset counter and save best model
            self.best_val_loss = val_loss
            self.patience_counter = 0
            
            # Save best model
            if self.is_main_process:
                self.best_model_path = os.path.join(self.config.result.ckpt_path, 'early_stop_best_model.pth')
                model_states, _ = self.get_checkpoint_states(stage='epoch_end')
                model_states['best_val_loss'] = self.best_val_loss
                model_states['patience_counter'] = self.patience_counter
                model_states['best_model_path'] = self.best_model_path
                torch.save(model_states, self.best_model_path)
                self.logger(f"[Early Stopping] New best model saved: val_loss={val_loss:.6f} (epoch {epoch + 1})")
        else:
            # No improvement: increment counter
            self.patience_counter += 1
            self.logger(f"[Early Stopping] No improvement: patience {self.patience_counter}/{self.early_stopping_patience} (best={self.best_val_loss:.6f}, current={val_loss:.6f})")
            
            if self.patience_counter >= self.early_stopping_patience:
                self.early_stop_triggered = True
                self.logger(f"[Early Stopping] Triggered! Stopping training at epoch {epoch + 1}")
                return True
        
        return False

    def get_checkpoint_states(self, stage='epoch_end'):
        optimizer_state = []
        for i in range(len(self.optimizer)):
            optimizer_state.append(self.optimizer[i].state_dict())

        scheduler_state = []
        for i in range(len(self.scheduler)):
            # 跳过延迟初始化的scheduler（dict占位符）
            if isinstance(self.scheduler[i], dict) and self.scheduler[i].get('_deferred_'):
                scheduler_state.append({})  # 保存空dict占位
            else:
                scheduler_state.append(self.scheduler[i].state_dict())

        optimizer_scheduler_states = {
            'optimizer': optimizer_state,
            'scheduler': scheduler_state
        }

        model_states = {
            'step': self.global_step,
            # Save early stopping state
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'best_model_path': self.best_model_path,
        }

        if self.config.training.use_DDP:
            model_states['model'] = self.net.module.state_dict()
        else:
            model_states['model'] = self.net.state_dict()

        if stage == 'exception':
            model_states['epoch'] = self.global_epoch
        else:
            model_states['epoch'] = self.global_epoch + 1

        if self.use_ema:
            model_states['ema'] = self.ema.shadow
        return model_states, optimizer_scheduler_states

    # EMA part
    def step_ema(self):
        with_decay = False if self.global_step < self.start_ema_step else True
        if self.config.training.use_DDP:
            self.ema.update(self.net.module, with_decay=with_decay)
        else:
            self.ema.update(self.net, with_decay=with_decay)

    def apply_ema(self):
        if self.use_ema:
            if self.config.training.use_DDP:
                self.ema.apply_shadow(self.net.module)
            else:
                self.ema.apply_shadow(self.net)

    def restore_ema(self):
        if self.use_ema:
            if self.config.training.use_DDP:
                self.ema.restore(self.net.module)
            else:
                self.ema.restore(self.net)

    # Evaluation and sample part
    @torch.no_grad()
    def validation_step(self, val_batch, epoch, step):
        self.apply_ema()
        self.net.eval()
        loss = self.loss_fn(net=self.net,
                            batch=val_batch,
                            epoch=epoch,
                            step=step,
                            opt_idx=0,
                            stage='val_step')
        if len(self.optimizer) > 1:
            loss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=1,
                                stage='val_step')
        self.restore_ema()

    @torch.no_grad()
    def validation_epoch(self, val_loader, epoch):
        self.apply_ema()
        self.net.eval()

        pbar = tqdm(val_loader, total=len(val_loader), smoothing=0.01, disable=not self.is_main_process)
        step = 0
        loss_sum = 0.
        dloss_sum = 0.
        for val_batch in pbar:
            loss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=0,
                                stage='val',
                                write=False)
            loss_sum += loss
            if len(self.optimizer) > 1:
                loss = self.loss_fn(net=self.net,
                                    batch=val_batch,
                                    epoch=epoch,
                                    step=step,
                                    opt_idx=1,
                                    stage='val',
                                    write=False)
                dloss_sum += loss
            step += 1
        average_loss = loss_sum / step
        if self.is_main_process:
            self.writer.add_scalar(f'val_epoch/loss', average_loss, epoch)
            if len(self.optimizer) > 1:
                average_dloss = dloss_sum / step
                self.writer.add_scalar(f'val_dloss_epoch/loss', average_dloss, epoch)
        self.restore_ema()
        return average_loss

    @torch.no_grad()
    def sample_step(self, train_batch, val_batch):
        self.apply_ema()
        self.net.eval()
        sample_path = make_dir(os.path.join(self.config.result.image_path, str(self.global_step)))
        if self.config.training.use_DDP:
            self.sample(self.net.module, train_batch, sample_path, stage='train')
            self.sample(self.net.module, val_batch, sample_path, stage='val')
        else:
            self.sample(self.net, train_batch, sample_path, stage='train')
            self.sample(self.net, val_batch, sample_path, stage='val')
        self.restore_ema()

    # abstract methods
    @abstractmethod
    def print_model_summary(self, net):
        pass

    @abstractmethod
    def initialize_model(self, config):
        """
        initialize model
        :param config: config
        :return: nn.Module
        """
        pass

    @abstractmethod
    def initialize_optimizer_scheduler(self, net, config):
        """
        initialize optimizer and scheduler
        :param net: nn.Module
        :param config: config
        :return: a list of optimizers; a list of schedulers
        """
        pass

    @abstractmethod
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        """
        loss function
        :param net: nn.Module
        :param batch: batch
        :param epoch: global epoch
        :param step: global step
        :param opt_idx: optimizer index, default is 0; set it to 1 for GAN discriminator
        :param stage: train, val, test
        :param write: write loss information to SummaryWriter
        :return: a scalar of loss
        """
        pass

    @abstractmethod
    def sample(self, net, batch, sample_path, stage='train'):
        """
        sample a single batch
        :param net: nn.Module
        :param batch: batch
        :param sample_path: path to save samples
        :param stage: train, val, test
        :return:
        """
        pass

    @abstractmethod
    def sample_to_eval(self, net, test_loader, sample_path):
        """
        sample among the test dataset to calculate evaluation metrics
        :param net: nn.Module
        :param test_loader: test dataloader
        :param sample_path: path to save samples
        :return:
        """
        pass

    def on_save_checkpoint(self, net, train_loader, val_loader, epoch, step):
        """
        additional operations whilst saving checkpoint
        :param net: nn.Module
        :param train_loader: train data loader
        :param val_loader: val data loader
        :param epoch: epoch
        :param step: step
        :return:
        """
        pass

    def train(self):
        self.logger(self.__class__.__name__)

        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_sampler = None
        val_sampler = None
        test_sampler = None
        if self.config.training.use_DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.config.data.train.batch_size,
                                      num_workers=8,
                                      drop_last=True,
                                      sampler=train_sampler)
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.config.data.val.batch_size,
                                    num_workers=8,
                                    drop_last=True,
                                    sampler=val_sampler)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     num_workers=8,
                                     drop_last=True,
                                     sampler=test_sampler)
        else:
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.config.data.train.batch_size,
                                      shuffle=self.config.data.train.shuffle,
                                      num_workers=8,
                                      drop_last=True)
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.config.data.val.batch_size,
                                    shuffle=self.config.data.val.shuffle,
                                    num_workers=8,
                                    drop_last=True)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=8,
                                     drop_last=True)

        epoch_length = len(train_loader)
        start_epoch = self.global_epoch
        self.logger(f"start training {self.config.model.model_name} on {self.config.data.dataset_name}, {len(train_loader)} iters per epoch")

        # 延迟初始化调度器（如果需要）
        # 检查是否有调度器需要延迟初始化（T_max=-1 的情况）
        if self.scheduler is not None:
            for i, sched in enumerate(self.scheduler):
                if isinstance(sched, dict) and sched.get('_deferred_', False):
                    # 计算实际的 T_max = epochs × steps_per_epoch / accumulate_grad_batches
                    accumulate = getattr(self.config.training, 'accumulate_grad_batches', 1)
                    T_max = epoch_length * self.config.training.n_epochs // accumulate
                    self.logger(f"Deferred scheduler initialization: T_max = {epoch_length} * {self.config.training.n_epochs} // {accumulate} = {T_max}")
                    
                    # 创建实际的调度器
                    self.scheduler[i] = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=sched['optimizer'],
                        T_max=T_max,
                        eta_min=sched.get('eta_min', 5e-7)
                    )

        try:
            accumulate_grad_batches = self.config.training.accumulate_grad_batches
            for epoch in range(start_epoch, self.config.training.n_epochs):
                if self.global_step > self.config.training.n_steps:
                    break

                # 每个 epoch 开始前清理显存碎片，防止长时间训练导致显存碎片化
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 确保之前的操作都完成

                if self.config.training.use_DDP:
                    train_sampler.set_epoch(epoch)
                    val_sampler.set_epoch(epoch)

                pbar = tqdm(train_loader, total=len(train_loader), smoothing=0.01, disable=not self.is_main_process)
                self.global_epoch = epoch
                start_time = time.time()
                for train_batch in pbar:
                    self.global_step += 1
                    self.net.train()

                    losses = []
                    for i in range(len(self.optimizer)):
                        # pdb.set_trace()
                        loss = self.loss_fn(net=self.net,
                                            batch=train_batch,
                                            epoch=epoch,
                                            step=self.global_step,
                                            opt_idx=i,
                                            stage='train')

                        loss.backward()
                        if self.global_step % accumulate_grad_batches == 0:
                            self.optimizer[i].step()
                            self.optimizer[i].zero_grad()
                            if self.scheduler is not None:
                                # 根据调度器类型决定 step() 调用方式
                                if isinstance(self.scheduler[i], torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    # ReduceLROnPlateau 需要传入 metric (loss)
                                    self.scheduler[i].step(loss)
                                else:
                                    # CosineAnnealingLR 等其他调度器不需要参数
                                    self.scheduler[i].step()
                        if self.config.training.use_DDP:
                            dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
                        losses.append(loss.detach().mean())

                    if self.use_ema and self.global_step % (self.update_ema_interval*accumulate_grad_batches) == 0:
                        self.step_ema()

                    if len(self.optimizer) > 1:
                        pbar.set_description(
                            (
                                f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                f'iter: {self.global_step} loss-1: {losses[0]:.4f} loss-2: {losses[1]:.4f}'
                            )
                        )
                    else:
                        pbar.set_description(
                            (
                                f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                f'iter: {self.global_step} loss: {losses[0]:.4f}'
                            )
                        )

                    with torch.no_grad():
                        if self.global_step % 50 == 0:
                            val_batch = next(iter(val_loader))
                            self.validation_step(val_batch=val_batch, epoch=epoch, step=self.global_step)

                        if self.global_step % int(self.config.training.sample_interval * epoch_length) == 0:
                            # val_batch = next(iter(val_loader))
                            # self.validation_step(val_batch=val_batch, epoch=epoch, step=self.global_step)

                            if self.is_main_process:
                                val_batch = next(iter(val_loader))
                                self.sample_step(val_batch=val_batch, train_batch=train_batch)
                                torch.cuda.empty_cache()

                end_time = time.time()
                elapsed_rounded = int(round((end_time-start_time)))
                self.logger("training time: " + str(datetime.timedelta(seconds=elapsed_rounded)))

                # validation
                if (epoch + 1) % self.config.training.validation_interval == 0 or (
                        epoch + 1) == self.config.training.n_epochs:
                    # if self.is_main_process == 0:
                    with torch.no_grad():
                        self.logger("validating epoch...")
                        average_loss = self.validation_epoch(val_loader, epoch)
                        torch.cuda.empty_cache()
                        self.logger("validating epoch success")
                        
                        # Early stopping check
                        if self._check_early_stopping(average_loss, epoch):
                            break  # Exit epoch loop

                # save checkpoint
                if (epoch + 1) % self.config.training.save_interval == 0 or \
                        (epoch + 1) == self.config.training.n_epochs or \
                        self.global_step > self.config.training.n_steps:
                    if self.is_main_process:
                        with torch.no_grad():
                            self.logger("saving latest checkpoint...")
                            self.on_save_checkpoint(self.net, train_loader, val_loader, epoch, self.global_step)
                            model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')

                            # save latest checkpoint
                            temp = 0
                            while temp < epoch + 1:
                                remove_file(os.path.join(self.config.result.ckpt_path, f'latest_model_{temp}.pth'))
                                remove_file(
                                    os.path.join(self.config.result.ckpt_path, f'latest_optim_sche_{temp}.pth'))
                                temp += 1
                            torch.save(model_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'latest_model_{epoch + 1}.pth'))
                            torch.save(optimizer_scheduler_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'latest_optim_sche_{epoch + 1}.pth'))
                            torch.save(model_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'last_model.pth'))
                            torch.save(optimizer_scheduler_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'last_optim_sche.pth'))

                            # save top_k checkpoints
                            model_ckpt_name = f'top_model_epoch_{epoch + 1}.pth'
                            optim_sche_ckpt_name = f'top_optim_sche_epoch_{epoch + 1}.pth'

                            if self.config.args.save_top:
                                print("save top model start...")
                                top_key = 'top'
                                if top_key not in self.topk_checkpoints:
                                    print('top key not in topk_checkpoints')
                                    self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                      'model_ckpt_name': model_ckpt_name,
                                                                      'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                    print(f"saving top checkpoint: average_loss={average_loss} epoch={epoch + 1}")
                                    torch.save(model_states,
                                               os.path.join(self.config.result.ckpt_path, model_ckpt_name))
                                    torch.save(optimizer_scheduler_states,
                                               os.path.join(self.config.result.ckpt_path, optim_sche_ckpt_name))
                                else:
                                    if average_loss < self.topk_checkpoints[top_key]["loss"]:
                                        print("remove " + self.topk_checkpoints[top_key]["model_ckpt_name"])
                                        remove_file(os.path.join(self.config.result.ckpt_path,
                                                                 self.topk_checkpoints[top_key]['model_ckpt_name']))
                                        remove_file(os.path.join(self.config.result.ckpt_path,
                                                                 self.topk_checkpoints[top_key]['optim_sche_ckpt_name']))

                                        print(
                                            f"saving top checkpoint: average_loss={average_loss} epoch={epoch + 1}")

                                        self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                          'model_ckpt_name': model_ckpt_name,
                                                                          'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                        torch.save(model_states,
                                                   os.path.join(self.config.result.ckpt_path, model_ckpt_name))
                                        torch.save(optimizer_scheduler_states,
                                                   os.path.join(self.config.result.ckpt_path, optim_sche_ckpt_name))
                if self.config.training.use_DDP:
                    dist.barrier()
            
            # Training completed (normally or via early stopping)
            # Load best model and run test if early stopping was used
            if self.is_main_process and self.early_stopping_enabled and self.best_model_path is not None:
                if os.path.exists(self.best_model_path):
                    self.logger(f"[Early Stopping] Training finished. Loading best model from {self.best_model_path}")
                    best_states = torch.load(self.best_model_path, map_location='cpu')
                    if self.config.training.use_DDP:
                        self.net.module.load_state_dict(best_states['model'])
                    else:
                        self.net.load_state_dict(best_states['model'])
                    if self.use_ema and 'ema' in best_states:
                        self.ema.shadow = best_states['ema']
                    self.logger(f"[Early Stopping] Best model loaded (val_loss={best_states.get('best_val_loss', 'N/A')}). Running test...")
                    self.test()
                    self.logger("[Early Stopping] Test completed.")
                    
        except BaseException as e:
            if self.is_main_process == 0:
                print("exception save model start....")
                print(self.__class__.__name__)
                model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='exception')
                torch.save(model_states,
                           os.path.join(self.config.result.ckpt_path, f'last_model.pth'))
                torch.save(optimizer_scheduler_states,
                           os.path.join(self.config.result.ckpt_path, f'last_optim_sche.pth'))

                print("exception save model success!")

            print('str(Exception):\t', str(Exception))
            print('str(e):\t\t', str(e))
            print('repr(e):\t', repr(e))
            print('traceback.print_exc():')
            traceback.print_exc()
            print('traceback.format_exc():\n%s' % traceback.format_exc())

    @torch.no_grad()
    def test(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        if test_dataset is None:
            test_dataset = val_dataset
        # test_dataset = val_dataset
        if self.config.training.use_DDP:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=True,
                                     sampler=test_sampler)
        else:
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=True)

        if self.use_ema:
            self.apply_ema()

        self.net.eval()
        if self.config.args.sample_to_eval:
            sample_path = self.config.result.sample_to_eval_path
            if self.config.training.use_DDP:
                self.sample_to_eval(self.net.module, test_loader, sample_path)
            else:
                self.sample_to_eval(self.net, test_loader, sample_path)
        else:
            test_iter = iter(test_loader)
            for i in tqdm(range(1), initial=0, dynamic_ncols=True, smoothing=0.01):
                test_batch = next(test_iter)
                sample_path = os.path.join(self.config.result.sample_path, str(i))
                if self.config.training.use_DDP:
                    self.sample(self.net.module, test_batch, sample_path, stage='test')
                else:
                    self.sample(self.net, test_batch, sample_path, stage='test')
