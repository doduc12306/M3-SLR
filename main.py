import argparse
import sys
from utils.misc import load_config
from dataset.dataloader import build_dataloader
from dataset.videoLoader import get_selected_indexs, pad_index
import random
import os
import numpy as np
import torch
import logging
from pathlib import Path
from utils.utils import load_criterion,load_lr_scheduler,load_optimizer,load_model
from utils.video_augmentation import Compose, Resize, ToFloatTensor, PermuteImage, Normalize
from decord import VideoReader
from trainer.trainer import Trainer
import wandb
from sklearn.model_selection import KFold
import pandas as pd

# Optional login for online tracking only.
# By default this is skipped to avoid interactive prompts during local runs.
if os.environ.get("ENABLE_WANDB_LOGIN", "0") == "1":
    wandb.login(key=os.environ.get("WANDB_API_KEY", ""))


def build_eval_transform(cfg):
    return Compose(
        Resize(cfg['data']['vid_transform']['IMAGE_SIZE']),
        ToFloatTensor(),
        PermuteImage(),
        Normalize(
            cfg['data']['vid_transform']['NORM_MEAN_IMGNET'],
            cfg['data']['vid_transform']['NORM_STD_IMGNET']
        )
    )


def run_single_video_inference(model, cfg, video_path, device, top_k=5, lookup_table_path=None):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    vr = VideoReader(str(video_path), width=320, height=256)
    vlen = len(vr)

    index_setting = cfg['data']['transform_cfg'].get('index_setting', ['segment', 'pad', 'segment', 'pad'])
    selected_index, pad = get_selected_indexs(
        vlen,
        cfg['data']['num_output_frames'],
        is_train=False,
        setting=index_setting,
        temporal_stride=cfg['data']['temporal_stride']
    )
    if pad is not None:
        selected_index = pad_index(selected_index, pad).tolist()

    transform = build_eval_transform(cfg)
    frames = vr.get_batch(selected_index).asnumpy()
    clip = [transform(frame) for frame in frames]
    clip = torch.stack(clip, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(device)

    label_map = {}
    if lookup_table_path:
        lut_path = Path(lookup_table_path)
        if lut_path.exists():
            df = pd.read_csv(lut_path)
            if {'id_label_in_documents', 'name'}.issubset(df.columns):
                for _, row in df.iterrows():
                    label_map[int(row['id_label_in_documents'])] = str(row['name'])

    model.eval()
    with torch.no_grad():
        if cfg['data'].get('model_name') == 'UsimKD':
            outputs = model(rgb_center=clip)
        else:
            outputs = model(clip=clip)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)
        k = min(top_k, probs.shape[-1])
        top_probs, top_indices = torch.topk(probs[0], k=k)

    print("\nSingle-video inference results:")
    print(f"Video: {video_path}")
    for rank, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist()), start=1):
        # Lookup table in this repo is typically 1-based while model indices are 0-based.
        label_name = label_map.get(idx + 1, label_map.get(idx, "N/A"))
        print(f"Top-{rank}: class_id={idx} prob={prob:.4f} label={label_name}")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline")
    parser.add_argument(
        "--config",
        default="configs/vtn_att_poseflow/vtn_att_poseflow_autsl.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Load model and pretrained checkpoint only, then exit without building datasets/dataloaders.",
    )
    parser.add_argument(
        "--infer-video",
        type=str,
        default=None,
        help="Path to a single video for inference. When set, dataset loaders are skipped.",
    )
    parser.add_argument(
        "--infer-topk",
        type=int,
        default=5,
        help="Top-k predictions to print in single-video inference mode.",
    )
    parser.add_argument(
        "--lookup-table",
        type=str,
        default="data/MultiVSL200/lookuptable.csv",
        help="CSV file to map predicted class IDs to human-readable labels.",
    )
    args = parser.parse_args()
    
    cfg = load_config(args.config)

    is_test = cfg['training'].get('test', False)
    if not is_test:
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"{cfg['data']['model_name']}",
            # mode = 'disabled',
            # Track hyperparameters and run metadata
            config={
            **cfg
            })
    else:
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"{cfg['data']['model_name']}",
            mode = 'disabled',
            # Track hyperparameters and run metadata
            config={
            **cfg
            })


    seed_everything(cfg['training']['random_seed'])
    log_directory = f"log/{cfg['data']['model_name']}"
    os.makedirs(log_directory, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"log/{cfg['data']['model_name']}/{cfg['training']['experiment_name']}" + ".log")
        ]
    )
    Path(f"checkpoints/{cfg['data']['model_name']}/{cfg['training']['experiment_name']}").mkdir(parents=True, exist_ok=True)
    Path(f"out-img/{cfg['data']['model_name']}/{cfg['training']['experiment_name']}").mkdir(parents=True, exist_ok=True)
    
    print("Starting " + f"{cfg['data']['model_name']}/{cfg['training']['experiment_name']}" + "...\n\n")
    logging.info("Starting " + f"{cfg['data']['model_name']}/{cfg['training']['experiment_name']}" + "...\n\n")
    
    num_folds = cfg['training'].get('num_folds', 1)
    if num_folds > 1:
        
        k_fold_loss = []
        k_fold_accuracy = []   
        
        print("K-Fold with k =",num_folds)
        kfold = KFold(n_splits=num_folds, shuffle=True,random_state = cfg['training']['random_seed'])
        
        train = pd.read_csv(os.path.join(cfg['data']['base_url'],f"{cfg['data']['label_folder']}/train_{cfg['data']['data_type']}.csv"),sep=',')
        val = pd.read_csv(os.path.join(cfg['data']['base_url'],f"{cfg['data']['label_folder']}/val_{cfg['data']['data_type']}.csv"),sep=',')

        full = pd.concat([train,val],ignore_index=True)
        signer = []
        for idx,row in full.iterrows():
            name = row['name'].split('_center_ord1')[0].split('_')[-1]
            if name not in signer:
                signer.append(name)

        for idx,(train_idx, val_idx) in enumerate(kfold.split(signer)):
            train_signers = [signer[i] for i in train_idx]
            val_signers = [signer[i] for i in val_idx]
            train_csv = full[(full['name'].str.contains('|'.join(train_signers))) ]
            val_csv = full[(full['name'].str.contains('|'.join(val_signers))) ]
            
            total_1 = len(train_signers) + len(val_signers)
            total_2 = len(set(val_signers + train_signers))
            
            assert total_1 == total_2
            
            logging.info(f"Starting Fold {idx+1}: \n Train: {train_signers} \n Val {val_signers} \n\n " )

            device = torch.device(cfg['training']['device'])
                
            model = load_model(cfg)
            model.to(device)
            
            criterion = load_criterion(cfg['training'])
            optimizer = load_optimizer(cfg['training'],model)
            lr_scheduler = load_lr_scheduler(cfg['training'],optimizer)

            # create dataloader
            train_loader = build_dataloader(cfg,'train',is_train=True,model = model,labels = train_csv)
            val_loader = build_dataloader(cfg,'val',is_train=False,model = model,labels = val_csv)
            test_loader = build_dataloader(cfg,'test',is_train=False,model = model)

            trainer = Trainer(model,criterion,optimizer,device,lr_scheduler,cfg['training']['top_k'],
                        cfg['training']['total_epoch'],logging=logging,cfg=cfg,
                        num_accumulation_steps = cfg['training']['num_accumulation_steps'],
                        patience = cfg['training']['patience'],verbose =  cfg['training']['verbose'],
                        delta = cfg['training']['delta'],
                        is_early_stopping = cfg['training']['is_early_stopping'],
                        gradient_clip_val = cfg['training']['gradient_clip_val'],
                        log_train_step=cfg['training']['log_train_step'],
                        log_steps= cfg['training']['log_steps'],
                        evaluate_step= cfg['training']['evaluate_step'],
                        evaluate_strategy=cfg['training']['evaluate_strategy'],
                        wandb = wandb,
                        k_fold = idx + 1
                        )
    
            if cfg['model']['num_classes'] != 0:
                trainer.train(train_loader,val_loader,test_loader) # train
            else:
                trainer.train(train_loader,None,None) # pretrained video mae
                
            # add accuracy and loss 
            k_fold_loss.append(abs(trainer.early_stopping.best_score)) # abs due to I use -loss in EarlyStopping to save model
            k_fold_accuracy.append(trainer.test_accuracy)

        logging.info(f"K Fold Loss: {k_fold_loss} " )
        logging.info(f"K Fold Accuracy: {k_fold_accuracy} " )
        
        best_idx = np.argmin(np.array(k_fold_loss))
        logging.info(f"Best Fold Loss: {k_fold_loss[best_idx]} " )
        logging.info(f"BestFold Accuracy: {k_fold_accuracy[best_idx]} " )
        wandb.run.finish()
    else:
        device = torch.device(cfg['training']['device'])
                
        model = load_model(cfg)
        model.to(device)

        if args.load_only:
            print("Load-only mode: model and pretrained checkpoint loaded successfully. Exiting without dataset.")
            logging.info("Load-only mode: model and pretrained checkpoint loaded successfully. Exiting without dataset.")
            wandb.run.finish()
            sys.exit(0)

        if args.infer_video is not None:
            run_single_video_inference(
                model=model,
                cfg=cfg,
                video_path=args.infer_video,
                device=device,
                top_k=args.infer_topk,
                lookup_table_path=args.lookup_table,
            )
            wandb.run.finish()
            sys.exit(0)
        
        criterion = load_criterion(cfg['training'])
        optimizer = load_optimizer(cfg['training'],model,criterion)
        lr_scheduler = load_lr_scheduler(cfg['training'],optimizer)

        # create dataloader
        train_loader = build_dataloader(cfg,'train',is_train=True,model = model)
        val_loader = build_dataloader(cfg,'val',is_train=False,model = model)
        test_loader = build_dataloader(cfg,'test',is_train=False,model = model)

        trainer = Trainer(model,criterion,optimizer,device,lr_scheduler,cfg['training']['top_k'],
                    cfg['training']['total_epoch'],logging=logging,cfg=cfg,
                    num_accumulation_steps = cfg['training']['num_accumulation_steps'],
                    patience = cfg['training']['patience'],verbose =  cfg['training']['verbose'],
                    delta = cfg['training']['delta'],
                    is_early_stopping = cfg['training']['is_early_stopping'],
                    gradient_clip_val = cfg['training']['gradient_clip_val'],
                    log_train_step=cfg['training']['log_train_step'],
                    log_steps= cfg['training']['log_steps'],
                    evaluate_step= cfg['training']['evaluate_step'],
                    evaluate_strategy=cfg['training']['evaluate_strategy'],
                    wandb = wandb,
                    )
        if not is_test:
            if cfg['model']['num_classes'] != 0:
                trainer.train(train_loader,val_loader,test_loader) # train
            else:
                trainer.train(train_loader,None,None) # maskfeat
        else:
            scores = []
            scores_top_k=[]
            scores_class = []
            scores_top_k_class=[]

            # evaluation: 5 times
            for i in range(5): 
                trainer.model.eval()

                _,_,  pred_correct, pred_all, acc = trainer.evaluate(test_loader)
                scores.append(acc)
                print(acc)
                logging.info(f"Idx:{i+1} - Total: {pred_all} - Correct: {pred_correct} -Top 1 Accuracy per Instance: {acc} " )

                pred_correct, pred_all, acc= trainer.evaluate_top_k(test_loader)
                scores_top_k.append(acc)
                print(acc)
                logging.info(f"Idx:{i+1} - Total: {pred_all} - Correct: {pred_correct} -Top 5 Accuracy per Instance: {acc} " )


                
            logging.info(f"--------------------------------Per Instance Statistic------------------------------" )    
            scores = torch.tensor(scores)
            mean = scores.mean()
            std = scores.std()
            logging.info(f"Mean top 1: {mean} - Std top 1: {std}" )
            
            # evaluation top k
            scores_top_k = torch.tensor(scores_top_k)
            meank = scores_top_k.mean()
            stdk = scores_top_k.std()
            logging.info(f"Mean Top 5: {meank} - Std Top 5: {stdk}" )


          
