# 文件名: run_multimodal.py
import torch
import os
import pandas as pd
import argparse
import time
import numpy as np

from loader import batch_loader 
from dataset_multimodal import CovidVaccineMultiModal, RNAMultiModal
from model_multimodal import EndToEndRNAFusionModel
from utils import set_seed, mcrmse_loss, EarlyStopping, pearson_correlation, spearman_correlation, r2_score

LOG_INTERVAL = 10

# --- 加权 Loss 函数 ---
def weighted_mcrmse_loss(input, target, mask=None, weight_factor=2.0, threshold=0.5):
    if mask is not None:
        input = input[mask]
        target = target[mask]
    
    mse = (input - target) ** 2
    weights = torch.ones_like(target)
    high_val_mask = target.abs() > threshold 
    weights[high_val_mask] = weight_factor
    mse = mse * weights
    mse = torch.mean(mse, dim=0) 
    columnwise_rmse = torch.sqrt(mse + 1e-8) 
    return torch.mean(columnwise_rmse)


class Trainer():
    def __init__(self, args, seed, save_path=None):
        self.seed = seed
        self.save_path = save_path
        set_seed(self.seed)
        print(f'Seed: {self.seed}')
        
        self.lr = args.lr
        self.epochs = args.epochs
        self.device = args.device
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        self.dataset_name = args.dataset.lower()

        if 'covid' in self.dataset_name:
            self.out_channels = 3
            self.task = 'node'
            self.in_channels_2d = 13 
            self.in_channels_3d = 6
            self.max_seq_len_1d = 130
            dataset_cls = CovidVaccineMultiModal
            self.loss_threshold = 0.5 
            
        elif 'tcribo' in self.dataset_name:
            self.out_channels = 1
            self.task = 'graph'
            self.in_channels_2d = 6  
            self.in_channels_3d = 6
            self.max_seq_len_1d = 130 
            dataset_cls = lambda root: RNAMultiModal(root, name='tcribo')
            self.loss_threshold = 1.0

        elif 'fungal' in self.dataset_name:
            self.out_channels = 1
            self.task = 'graph'
            self.in_channels_2d = 6
            self.in_channels_3d = 6
            self.max_seq_len_1d = 3100 
            dataset_cls = lambda root: RNAMultiModal(root, name='fungal')
            self.loss_threshold = 1.0 
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented.")

        print(f"Loading dataset: {self.dataset_name}")
        self.dataset = dataset_cls(root='./data')
        
        self.train_loader, self.val_loader, self.test_loader = batch_loader(
            dataset_name=self.dataset_name, 
            dataset=self.dataset,
            batch_size=self.batch_size,
            train_ratio=args.train_ratio, 
            val_ratio=args.val_ratio, 
            test_ratio=args.test_ratio,
        )
        
        self.model = EndToEndRNAFusionModel(
            d_model_1d=args.d_model_1d, nhead_1d=args.nhead_1d, num_layers_1d=args.num_layers_1d, dim_ff_1d=args.dim_ff_1d, max_seq_len_1d=self.max_seq_len_1d,
            in_channels_2d=self.in_channels_2d, hidden_2d=args.hidden_2d, L_2d=args.L_2d,
            in_channels_3d=self.in_channels_3d, hidden_3d=args.hidden_3d, L_atom_3d=args.L_atom_3d, L_nt_3d=args.L_nt_3d, thres_atom_3d=args.thres_atom_3d, thres_nt_3d=args.thres_nt_3d,
            fusion_dim=args.fusion_dim, out_channels=self.out_channels, task=self.task, pool=args.pool
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_func = weighted_mcrmse_loss 
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params:,}")

    def _train(self):
        self.model.train()
        train_total_loss = 0
        accumulation_steps = 16 
        self.optimizer.zero_grad() 
        
        for i, data in enumerate(self.train_loader):
            data = data.to(self.device)
            
            # [修改] 解包主输出和辅助输出
            logits, aux_logits_3d = self.model(data)
            
            mask = data.mask if self.task == 'node' else None
            
            # 1. 主任务 Loss
            loss_main = self.loss_func(
                input=logits, 
                target=data.y, 
                mask=mask, 
                threshold=self.loss_threshold
            )
            
            # 2. [新增] 辅助任务 Loss (强迫训练 EGNN)
            loss_aux = self.loss_func(
                input=aux_logits_3d,
                target=data.y,
                mask=mask,
                threshold=self.loss_threshold
            )
            
            # 联合 Loss
            loss = loss_main + 0.5 * loss_aux
            
            loss = loss / accumulation_steps 
            loss.backward()
            
            # [监控] 3D 梯度检查
            if i % 100 == 0:
                grad_3d = self.model.proj_3d.weight.grad
                if grad_3d is not None:
                    # print(f"Step {i}: 3D Projection Grad Norm: {grad_3d.norm().item():.6f}")
                    pass
                else:
                    print(f"Step {i}: 3D Projection has NO GRADIENT!")

            train_total_loss += loss.item() * accumulation_steps

            if ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(self.train_loader)):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return train_total_loss / len(self.train_loader)

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss = 0
        
        for data in self.val_loader:
            data = data.to(self.device)
            # [修改] 忽略辅助输出
            logits, _ = self.model(data)
            mask = data.mask if self.task == 'node' else None
            
            loss = self.loss_func(logits, data.y, mask, threshold=self.loss_threshold)
            total_loss += loss.item()
            
            if mask is not None:
                all_preds.append(logits[mask].cpu())
                all_targets.append(data.y[mask].cpu())
            else:
                all_preds.append(logits.cpu())
                all_targets.append(data.y.cpu())

        if len(all_preds) == 0: return {'loss': float('inf'), 'pcc': 0, 'spearman': 0, 'r2': 0}
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = {}
        metrics['loss'] = total_loss / len(self.val_loader)
        metrics['pcc'] = pearson_correlation(all_preds, all_targets).item()
        metrics['spearman'] = spearman_correlation(all_preds, all_targets).item()
        metrics['r2'] = r2_score(all_preds, all_targets).item()
        return metrics
        
    @torch.no_grad()
    def _test(self):
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss = 0
        
        for data in self.test_loader:
            data = data.to(self.device)
            # [修改] 忽略辅助输出
            logits, _ = self.model(data)
            mask = data.mask if self.task == 'node' else None
            
            loss = self.loss_func(logits, data.y, mask, threshold=self.loss_threshold)
            total_loss += loss.item()
            
            if mask is not None:
                all_preds.append(logits[mask].cpu())
                all_targets.append(data.y[mask].cpu())
            else:
                all_preds.append(logits.cpu())
                all_targets.append(data.y.cpu())

        if len(all_preds) == 0:
            return {'loss': float('inf'), 'pcc': 0, 'spearman': 0, 'r2': 0}

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = {}
        metrics['loss'] = total_loss / len(self.test_loader)
        metrics['pcc'] = pearson_correlation(all_preds, all_targets).item()
        metrics['spearman'] = spearman_correlation(all_preds, all_targets).item()
        metrics['r2'] = r2_score(all_preds, all_targets).item()
        return metrics

    def train(self):
        best_val_rmse = float('inf')
        best_test_metrics = {}
        
        start_time = time.time()
        early_stopping = EarlyStopping(patience=50) 

        print("Start Training...")
        for epoch in range(self.epochs + 1):
            train_loss = self._train()
            val_metrics = self._evaluate()
            val_rmse = val_metrics['loss']
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_test_metrics = self._test()
                if self.save_path:
                    os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), self.save_path)

            if epoch % LOG_INTERVAL == 0:
                print(f'Epoch: {epoch:03d}, Train: {train_loss:.4f}, Val Loss: {val_rmse:.4f}, '
                      f'Test PCC: {best_test_metrics.get("pcc", 0):.4f}')
            
            early_stopping(val_rmse)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        end_time = time.time()
        res = {
            'best_val_rmse': best_val_rmse, 
            'best_test_rmse': best_test_metrics.get('loss'),
            'best_test_pcc': best_test_metrics.get('pcc'),
            'best_test_spearman': best_test_metrics.get('spearman'),
            'best_test_r2': best_test_metrics.get('r2'),
            'time': end_time - start_time
        }
        return res
    

def run_train_multimodal(args):
    save_dir = os.path.join('./checkpoints', args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results = []
    for seed in args.seeds:
        print(f"\n{'='*20} Running Seed {seed} {'='*20}")
        save_path = os.path.join(save_dir, f"model_full_seed{seed}.pth")
        
        trainer = Trainer(args, seed, save_path=save_path)
        res = trainer.train()
        results.append(res)
        
        print(f"Seed {seed} Result: Test RMSE={res['best_test_rmse']:.4f}, PCC={res['best_test_pcc']:.4f}")

    df = pd.DataFrame(results)
    print(f"\n{'='*20} Final Results ({len(args.seeds)} seeds) {'='*20}")
    print(df.mean())

def parser_add_main_args(parser):
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--dataset', type=str, default='covid') 
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4) 
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--pool', type=str, default='mean', choices=['mean', 'add', 'max'])
    parser.add_argument('--fusion_dim', type=int, default=128)
    parser.add_argument('--d_model_1d', type=int, default=128)
    parser.add_argument('--nhead_1d', type=int, default=8)
    parser.add_argument('--num_layers_1d', type=int, default=4)
    parser.add_argument('--dim_ff_1d', type=int, default=128)
    parser.add_argument('--hidden_2d', type=int, default=128)
    parser.add_argument('--L_2d', type=int, default=4)
    parser.add_argument('--hidden_3d', type=int, default=128)
    parser.add_argument('--L_atom_3d', type=int, default=3)
    parser.add_argument('--L_nt_3d', type=int, default=4)
    
    # [修改] 默认阈值放宽到 3.0，确保 3D 图连通
    parser.add_argument('--thres_atom_3d', type=float, default=3.0)
    parser.add_argument('--thres_nt_3d', type=float, default=22)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser_add_main_args(parser)
    args = parser.parse_args()
    args.noise_level = None 
    run_train_multimodal(args)