# 文件名: analysis_viz.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
import matplotlib.font_manager as fm 
import warnings

# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

from dataset_multimodal import CovidVaccineMultiModal, RNAMultiModal
from model_multimodal import EndToEndRNAFusionModel
from loader import batch_loader
from utils import set_seed

# ==========================================
# 1. 论文级绘图风格设置
# ==========================================
def set_paper_style():
    font_dir = './fonts' 
    font_files = []
    
    if os.path.exists(font_dir):
        for file in os.listdir(font_dir):
            if file.lower().endswith('.ttf'):
                font_path = os.path.join(font_dir, file)
                fm.fontManager.addfont(font_path)
                font_files.append(font_path)
        
        if len(font_files) > 0:
            print(f">>> [Font] Loaded {len(font_files)} local font files from {font_dir}")
    
    plt.rcParams['font.family'] = 'sans-serif' 
    plt.rcParams['font.family'] = 'Times New Roman' 
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.dpi'] = 300 
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

def visualize_results(args):
    set_paper_style()
    set_seed(2025)
    device = torch.device(args.device)
    
    print(f"\n>>> Loading Dataset: {args.dataset}...")
    dataset_name = args.dataset.lower()
    
    if 'covid' in dataset_name:
        dataset = CovidVaccineMultiModal(root='./data')
        in_channels_2d = 13; out_channels = 3; task = 'node'; max_seq_len = 130; pool = 'mean'
    elif 'tcribo' in dataset_name:
        dataset = RNAMultiModal(root='./data', name='tcribo')
        in_channels_2d = 6; out_channels = 1; task = 'graph'; max_seq_len = 130; pool = 'mean'
    elif 'fungal' in dataset_name:
        dataset = RNAMultiModal(root='./data', name='fungal')
        in_channels_2d = 6; out_channels = 1; task = 'graph'; max_seq_len = 3100; pool = 'mean'
    else:
        raise ValueError("Unknown dataset")

    # 获取少量测试集数据
    _, _, test_loader = batch_loader(args.dataset, dataset, batch_size=1, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    print(">>> Initializing Model...")
    # 注意：这里参数要和 run_multimodal.py 里的默认参数或者你训练时的参数保持一致
    # 特别是 thres_atom_3d 现在默认是 3.0
    model = EndToEndRNAFusionModel(
        d_model_1d=128, nhead_1d=16, num_layers_1d=4, dim_ff_1d=128, max_seq_len_1d=max_seq_len,
        in_channels_2d=in_channels_2d, hidden_2d=128, L_2d=4,
        in_channels_3d=6, hidden_3d=128, L_atom_3d=3, L_nt_3d=4, 
        thres_atom_3d=3.0, # <--- 确保这里也是 3.0
        thres_nt_3d=22,
        fusion_dim=128, out_channels=out_channels, task=task, pool=pool
    ).to(device)
    
    model_path = args.model_path
    if not model_path:
        possible_paths = [
            f"./checkpoints/{args.dataset}/model_full_seed0.pth",
            f"./checkpoints_1/{args.dataset}/model_full_seed0.pth"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f">>> Auto-detected trained model: {model_path}")
                break
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(">>> Model weights loaded successfully.")
    
    model.eval()
    
    # 获取可视化样本
    target_sample = None
    for data in test_loader:
        if task == 'node' and data.mask.sum() > 10: 
            target_sample = data; break
        elif task == 'graph':
            target_sample = data; break
    
    if target_sample is None:
        print("Error: No sample found.")
        return

    target_sample = target_sample.to(device)
    
    with torch.no_grad():
        # === [关键修改] ===
        # 新模型返回 3 个值：(preds, aux_preds, attn_dict)
        # 我们用 _ 忽略中间的辅助预测
        preds, _, attn_dict = model(target_sample, return_attn=True)
        # =================

    # --- 绘图 1: 散点图 ---
    print(">>> Generating Scatter Plot...")
    plt.figure(figsize=(7, 7))
    
    if task == 'node':
        y_true = target_sample.y.cpu().numpy()[target_sample.mask.cpu().numpy()]
        y_pred = preds.cpu().numpy()[target_sample.mask.cpu().numpy()]
        
        # 展平以便绘制
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        plt.scatter(y_true, y_pred, alpha=0.6, s=20, color='#2878B5')
    elif task == 'graph':
        plt.scatter(target_sample.y.cpu(), preds.cpu(), s=100, color='#C82423')
        y_true = target_sample.y.cpu().numpy().flatten()
        y_pred = preds.cpu().numpy().flatten()
    
    if len(y_true) > 0:
        min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        plt.plot([min_v, max_v], [min_v, max_v], 'k--', label='Ideal Fit')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title(f'Performance on {args.dataset}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'viz_scatter_{args.dataset}.png')

    # --- 绘图 2: 热力图 ---
    print(">>> Generating Heatmaps...")
    attn_3d = attn_dict['attn_3d'][0].cpu().numpy() 
    attn_2d = attn_dict['attn_2d'][0].cpu().numpy()
    
    viz_len = min(attn_3d.shape[0], 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sns.heatmap(attn_2d[:viz_len, :viz_len], ax=axes[0], cmap='viridis', square=True)
    axes[0].set_title('(a) Attention to 2D Structure')
    axes[0].set_ylabel('Query (1D Sequence)')
    
    sns.heatmap(attn_3d[:viz_len, :viz_len], ax=axes[1], cmap='plasma', square=True)
    axes[1].set_title('(b) Attention to 3D Structure')
    axes[1].set_ylabel('Query (1D Sequence)')
    
    plt.tight_layout()
    plt.savefig(f'viz_attention_{args.dataset}.png')
    print(">>> Done. Images saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='covid')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_path', type=str, default='', help='Path to .pth file')
    args = parser.parse_args()
    
    visualize_results(args)