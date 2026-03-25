# 文件名: run_ablation.py
import torch
import pandas as pd
import argparse
import os
import shutil
from run_multimodal import Trainer, parser_add_main_args
from utils import set_seed

class AblationTrainer(Trainer):
    def __init__(self, args, seed, ablation_mode, save_path=None):
        super().__init__(args, seed, save_path=save_path)
        self.ablation_mode = ablation_mode
        print(f"\n>>> [Ablation] Mode: {self.ablation_mode} | Seed: {seed}")
        if save_path:
            print(f">>> Model will be saved to: {save_path}")

    def _forward(self, data):
        logits = self.model(data, ablation_mode=self.ablation_mode)
        return logits

def run_ablation_study(args):
    # 结果保存路径
    save_dir = f"./checkpoints/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)

    # 定义消融模式
    modes = [
        'full',      # 完整项目 (会保存模型)
        'no_3d',     # 无 3D
        'no_2d',     # 无 2D
        'only_1d',   # 仅 1D
        'no_1d', 'only_2d', 'only_3d' # 可选
    ]
    
    results = []

    for mode in modes:
        print(f"\n{'='*40}\n STARTING ABLATION MODE: {mode.upper()} \n{'='*40}")
        
        seed_metrics = []
        for seed in args.seeds:
            # 只有 Full 模式保存模型
            if mode == 'full':
                model_filename = f"model_full_seed{seed}.pth"
                save_path = os.path.join(save_dir, model_filename)
            else:
                save_path = None 

            trainer = AblationTrainer(args, seed, ablation_mode=mode, save_path=save_path)
            
            # 训练
            res = trainer.train() 
            res['mode'] = mode
            res['seed'] = seed
            seed_metrics.append(res)
            
            del trainer
            torch.cuda.empty_cache()
        
        # 统计均值
        df_seed = pd.DataFrame(seed_metrics)
        avg_res = df_seed.mean(numeric_only=True).to_dict()
        avg_res['mode'] = mode
        
        print(f"\n>>> Mode {mode} Finished.")
        results.append(avg_res)

    # --- 最终结果展示 ---
    print("\n\n" + "="*50)
    print("FINAL ABLATION STUDY REPORT")
    print("="*50)
    final_df = pd.DataFrame(results)
    
    # --- [修正] 添加 best_test_spearman 到输出列 ---
    target_cols = ['mode', 'best_test_rmse', 'best_test_pcc', 'best_test_spearman', 'best_test_r2', 'time']
    
    # 自动匹配存在的列（防止有些指标没计算导致报错）
    final_cols = [c for c in target_cols if c in final_df.columns]
    # 把剩余的列（如果有）也加在后面
    remaining_cols = [c for c in final_df.columns if c not in final_cols]
    
    final_df = final_df[final_cols + remaining_cols]
    
    # 打印和保存
    print(final_df)
    csv_name = f'ablation_full_report_{args.dataset}.csv'
    final_df.to_csv(csv_name, index=False)
    print(f"\nResults saved to: {csv_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser_add_main_args(parser)
    args = parser.parse_args()
    
    print(f"Running Ablation on {args.dataset} | Epochs: {args.epochs}")
    run_ablation_study(args)