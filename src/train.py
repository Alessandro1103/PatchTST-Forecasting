import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import yaml
import types
from src.transformer_blackbone import Transformer_Backbone
from src.dataset import get_dataloader


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file non trovato: {config_path}")
        
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    config_flat = {}
    for section in config_dict:
        for key, value in config_dict[section].items():
            config_flat[key] = value
            
    return types.SimpleNamespace(**config_flat)


class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def main(args):

    device = torch.device("cpu")
    
    # Type defining
    is_pretraining = args.mask_ratio > 0
    mode_str = "Pre-training" if is_pretraining else "Forecasting"
    print(f"Mode: {mode_str} (Mask Ratio: {args.mask_ratio})")

    # Data Loaders
    _, train_loader = get_dataloader(args.root_path, args.data_path, 'train', args.seq_len, args.pred_len, args.batch_size)
    _, val_loader = get_dataloader(args.root_path, args.data_path, 'val', args.seq_len, args.pred_len, args.batch_size)
    _, test_loader = get_dataloader(args.root_path, args.data_path, 'test', args.seq_len, args.pred_len, args.batch_size)

    # Model
    model = Transformer_Backbone(
        patch_len=args.patch_len,
        stride=args.stride,
        n_channels=args.n_channels,
        seq_len=args.seq_len,
        prediction_len=args.pred_len,
        latent_dimension=args.d_model,
        dropout=args.dropout
    ).to(device)

    if not is_pretraining and hasattr(args, 'pretrained_model_path') and args.pretrained_model_path:
        print(f"Loading pre-trained backbone from: {args.pretrained_model_path}")
        
        checkpoint = torch.load(args.pretrained_model_path, map_location=device)
        
        # Load everything except for the heads
        model_dict = model.state_dict()
        
        # head_pretrain and head_forecast don't belong to the backbone
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and 'head' not in k}
        
        # Check
        if len(pretrained_dict) == 0:
            print("Attention: No weight uploaded")
        else:
            print(f"Loaded {len(pretrained_dict)} layer from pre-train.")

        # Update dictionary
        model_dict.update(pretrained_dict) 
        
        # Load the weights
        model.load_state_dict(model_dict, strict=False) # strict=False because the heads are missing
        
        print("Backbone weights loaded successfully. Prediction head initialized from scratch.")

        # Congeliamo tutti i parametri tranne la head_forecast
        print("\n[INFO] Attivazione Linear Probing: Congelamento della Backbone.")
        for name, param in model.named_parameters():
            if "head_forecast" not in name:
                param.requires_grad = False
            else:
                print(f"Trainable: {name}")
        
        print("Backbone weights loaded successfully. Prediction head initialized from scratch.")

    # Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.learning_rate, 
        total_steps=total_steps,
        pct_start=0.3 # 30% warmup
    )
    # Loss Function
    criterion = nn.MSELoss(reduction='none') if is_pretraining else nn.MSELoss()

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    
    # Crea ID univoco
    dataset_name = args.data_path.split('.')[0]
    task_tag = "PT" if is_pretraining else "FC" # PT=PreTraining, FC=ForeCasting
    
    if not hasattr(args, 'model_id') or args.model_id is None:
        model_id = f"{dataset_name}_{task_tag}_L{args.seq_len}_T{args.pred_len}_M{args.mask_ratio}"
    else:
        model_id = args.model_id

    print(f"Saving model to: ./checkpoints/{model_id}.pth")
    save_path = f'./checkpoints/{model_id}.pth'
    early_stopping = EarlyStopping(patience=args.patience, path=save_path)

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # [Batch, Seq, Chan] -> [Batch, Chan, Seq]
            batch_x = batch_x.float().to(device).permute(0, 2, 1)
            
            if is_pretraining:
                mean_x = batch_x.mean(dim=2, keepdim=True).detach()
                std_x = batch_x.std(dim=2, keepdim=True).detach()
                batch_x_norm = (batch_x - mean_x) / (std_x + 1e-5)
                
                outputs, mask = model(batch_x, mask_ratio=args.mask_ratio)
                
                with torch.no_grad():
                    ground_truth_patches, _ = model.patcher(batch_x_norm, mask_ratio=0.0)
                
                # Loss between masked and clean patches
                loss_elementwise = criterion(outputs, ground_truth_patches)
                
                # mask shape: [B, C, N] -> [B, C, N, 1]
                mask = mask.unsqueeze(-1)

                loss = (loss_elementwise * mask).sum() / (mask.sum() + 1e-7)

            # Forecasting  
            else:

                batch_y = batch_y.float().to(device)
                
                outputs = model(batch_x, mask_ratio=0.0)
                outputs = outputs.permute(0, 2, 1) # [B, L, C]
                
                loss = criterion(outputs, batch_y)
            
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Validation
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.float().to(device).permute(0, 2, 1)
                
                if is_pretraining:
                    # Validation Pre-training                    
                    mean_x = batch_x.mean(dim=2, keepdim=True).detach()
                    std_x = batch_x.std(dim=2, keepdim=True).detach()
                    batch_x_norm = (batch_x - mean_x) / (std_x + 1e-5)

                    outputs, mask = model(batch_x, mask_ratio=args.mask_ratio)
                    ground_truth_patches, _ = model.patcher(batch_x_norm, mask_ratio=0.0)
                    
                    loss_elementwise = criterion(outputs, ground_truth_patches)
                    mask = mask.unsqueeze(-1)
                    loss = (loss_elementwise * mask).sum() / (mask.sum() + 1e-7)
                    val_loss.append(loss.item())
                else:
                    # Validazione Forecasting
                    batch_y = batch_y.float().to(device)
                    outputs = model(batch_x, mask_ratio=0.0).permute(0, 2, 1)
                    val_loss.append(criterion(outputs, batch_y).item())

        val_loss = np.mean(val_loss)
        print(f"Epoch {epoch+1}: Train Loss {np.mean(train_loss):.5f} | Val Loss {val_loss:.5f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping.")
            break

    # Testing
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    print(f"Training completato. Modello salvato in {save_path}")

if __name__ == "__main__":

    # Default choice
    CONFIG_FILE = './configs/default.yaml'
    args = load_config(CONFIG_FILE)
    
    main(args)