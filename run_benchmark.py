import copy
import os
from src.train import main, load_config

def run_benchmark():

    # datasets = ['ETTh1.csv', 'ETTh2.csv', 'ETTm1.csv', 'ETTm2.csv']
    # horizons = [96, 192, 336, 720]
    # seq_lens = [336, 512]

    datasets = ['ETTh1.csv']
    horizons = [96]
    seq_lens = [720]
    
    config_path = './configs/default.yaml'
    base_args = load_config(config_path)


    for data_path in datasets:

        for seq_len in seq_lens:

            for pred_len in horizons:
                
                print(f"\n{'='*50}")
                print(f"BENCHMARK: {data_path} | L={seq_len} | T={pred_len}")
                print(f"{'='*50}")
                
                dataset_name = data_path.split('.')[0]

                print(f"\n[1/2] Pre-training ({base_args.pretrain['epochs']} epochs)")
                
                args_pretrain = copy.copy(base_args)

                # Setup Base
                args_pretrain.data_path = data_path
                args_pretrain.seq_len = seq_len
                args_pretrain.pred_len = pred_len

                vars(args_pretrain).update(base_args.pretrain)
                
                args_pretrain.model_id = f"{dataset_name}_L{seq_len}{args_pretrain.model_id_suffix}"
                
                # Run
                main(args_pretrain)
                
                ckpt_name = f"{args_pretrain.model_id}.pth"
                pretrained_ckpt = f"./checkpoints/{ckpt_name}"

                print(f"\n[2/2] Fine-Tuning ({base_args.forecast['epochs']} epochs)")

                if os.path.exists(pretrained_ckpt):
                    args_finetune = copy.copy(base_args)
                    
                    # Setup Base
                    args_finetune.data_path = data_path
                    args_finetune.seq_len = seq_len
                    args_finetune.pred_len = pred_len
                    
                    vars(args_finetune).update(base_args.forecast)
                    
                    args_finetune.pretrained_model_path = pretrained_ckpt
                    args_finetune.model_id = f"{dataset_name}_L{seq_len}_T{pred_len}{args_finetune.model_id_suffix}"
                    
                    # Run
                    main(args_finetune)
                    
                else:
                    print(f"Checkpoint not found: {pretrained_ckpt}")

if __name__ == "__main__":
    run_benchmark()