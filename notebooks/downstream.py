# Code to run xperiments on Fold and EC datasets in our paper 
# "Learning Hierarchical Protein Representations via Complete 3D Graph Networks" 
# (https://openreview.net/forum?id=9X-hgLDLYkQ)

##################################### Default hyperparameters for ECdataset #####################################
# device=0
# dataset='func'
# dataset_path='dataset/' # make sure that the folder 'ProtFunct' is under this path
# cutoff=10.0
# batch_size=32
# eval_batch_size=32

# level='backbone'
# num_blocks=4
# hidden_channels=128
# out_channels=384

# epochs=400
# lr=0.0005
# lr_decay_step_size=60
# lr_decay_factor=0.5

# mask_aatype=0.2
# dropout=0.3
# num_workers=5

# python run_ProNet.py --device $device --dataset $dataset --dataset_path $dataset_path --cutoff $cutoff \
# --batch_size $batch_size --eval_batch_size $eval_batch_size \
# --level $level --num_blocks $num_blocks --hidden_channels $hidden_channels --out_channels $out_channels \
# --epochs $epochs \
# --lr $lr --lr_decay_step_size $lr_decay_step_size --lr_decay_factor $lr_decay_factor \
# --mask_aatype $mask_aatype --dropout $dropout \
# --num_workers $num_workers \
# --mask --noise --deform --euler_noise --data_augment_eachlayer

##################################### Default hyperparameters for ECdataset #####################################
# device=0
# dataset='fold'
# dataset_path='dataset/' # make sure that the folder 'HomologyTAPE' is under this path
# cutoff=10.0
# batch_size=32
# eval_batch_size=32

# level='backbone'
# num_blocks=4
# hidden_channels=128
# out_channels=1195

# epochs=1000
# lr=0.0005
# lr_decay_step_size=150
# lr_decay_factor=0.5

# mask_aatype=0.2
# dropout=0.3
# num_workers=5

# python run_ProNet.py --device $device --dataset $dataset --dataset_path $dataset_path --cutoff $cutoff \
# --batch_size $batch_size --eval_batch_size $eval_batch_size \
# --level $level --num_blocks $num_blocks --hidden_channels $hidden_channels --out_channels $out_channels \
# --epochs $epochs \
# --lr $lr --lr_decay_step_size $lr_decay_step_size --lr_decay_factor $lr_decay_factor \
# --mask_aatype $mask_aatype --dropout $dropout \
# --num_workers $num_workers \
# --mask --noise --deform --euler_noise --data_augment_eachlayer

import os

import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import argparse

import torch
import torch.optim as optim
from torch import nn 
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
from pronet import ProNet

from dig.threedgraph.dataset import ECdataset
from dig.threedgraph.dataset import FOLDdataset
from torch_geometric.data import DataLoader




import warnings
warnings.filterwarnings("ignore")

criterion = nn.CrossEntropyLoss()

num_fold = 1195
num_func = 384


def train(args, model, loader, optimizer, device, featuriser):
    model.train()

    loss_accum = 0
    preds = []
    functions = []
    for step, batch in enumerate(tqdm(loader, disable=args.disable_tqdm)):
        batch = featuriser(batch)
        if args.mask:
            # random mask node aatype
            mask_indice = torch.tensor(np.random.choice(batch.num_nodes, int(batch.num_nodes * args.mask_aatype), replace=False))
            batch.x[:, 0][mask_indice] = 25
        if args.noise:
            # add gaussian noise to atom coords
            gaussian_noise = torch.clip(torch.normal(mean=0.0, std=0.1, size=batch.coords_ca.shape), min=-0.3, max=0.3)
            batch.coords_ca += gaussian_noise
            if args.level != 'aminoacid':
                batch.coords_n += gaussian_noise
                batch.coords_c += gaussian_noise
        if args.deform:
            # Anisotropic scale
            deform = torch.clip(torch.normal(mean=1.0, std=0.1, size=(1, 3)), min=0.9, max=1.1)
            batch.coords_ca *= deform
            if args.level != 'aminoacid':
                batch.coords_n *= deform
                batch.coords_c *= deform
        batch = batch.to(device)
        batch.x = batch.x.to(device)
        try:
            pred = model(batch) 
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): 
                print('\n forward error \n')
                raise(e)
            else:
                print('OOM')
            #torch.cuda.empty_cache()
            continue
        preds.append(torch.argmax(pred, dim=1))        
        function = batch.graph_y.to(device)
        functions.append(function)
        optimizer.zero_grad()
        loss = criterion(pred, function)
        loss.backward()
        optimizer.step()

        loss_accum += loss.item()        

    functions = torch.cat(functions, dim=0)
    preds = torch.cat(preds, dim=0)
    acc = torch.sum(preds==functions)/functions.shape[0]
    
    return loss_accum/(step + 1), acc.item()


def evaluation(args, model, loader, device, featuriser):    
    model.eval()
    
    loss_accum = 0
    preds = []
    functions = []
    for step, batch in enumerate(loader):
        batch = featuriser(batch)
        batch = batch.to(device)
        # pred = model(batch)
        try:
            pred = model(batch) 
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): 
                print('\n forward error \n')
                raise(e)
            else:
                print('evaluation OOM')
            #torch.cuda.empty_cache()
            continue
        preds.append(torch.argmax(pred, dim=1))
        
        function = batch.graph_y.to(device)
        functions.append(function)
        loss = criterion(pred, function)
        loss_accum += loss.item()    
            
    functions = torch.cat(functions, dim=0)
    preds = torch.cat(preds, dim=0)
    acc = torch.sum(preds==functions)/functions.shape[0]
    
    return loss_accum/(step + 1), acc.item()

    
def main():
    ### Args
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=0, help='Device to use')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of workers in Dataloader')

    ### Data
    parser.add_argument('--dataset', type=str, default='func', help='Func or fold')
    parser.add_argument('--dataset_path', type=str, default='/home/zhang/Projects/3d/dataset', help='path to load and process the data')
    
    # data augmentation tricks, see appendix E in the paper (https://openreview.net/pdf?id=9X-hgLDLYkQ)
    parser.add_argument('--mask', default=False, action='store_true', help='Random mask some node type')
    parser.add_argument('--noise', default=False, action='store_true', help='Add Gaussian noise to node coords')
    parser.add_argument('--deform', default=False, action='store_true', help='Deform node coords')
    parser.add_argument('--data_augment_eachlayer', default=True, action='store_true', help='Add Gaussian noise to features')
    parser.add_argument('--euler_noise', default=False, action='store_true', help='Add Gaussian noise Euler angles')
    parser.add_argument('--mask_aatype', type=float, default=0.2, help='Random mask aatype to 25(unknown:X) ratio')
    
    ### Model
    parser.add_argument('--level', type=str, default='aminoacid', help='Choose from \'aminoacid\', \'backbone\', and \'allatom\' levels')
    parser.add_argument('--num_blocks', type=int, default=4, help='Model layers')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--out_channels', type=int, default=384, help='Number of classes, 1195 for the fold data, 384 for the ECdata')
    parser.add_argument('--fix_dist', action='store_true')  
    parser.add_argument('--cutoff', type=float, default=10, help='Distance constraint for building the protein graph') 
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout')

    ### Training hyperparameter
    parser.add_argument('--epochs', type=int, default=800, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--lr_decay_step_size', type=int, default=150, help='Learning rate step size')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='Learning rate factor') 
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight Decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size during training')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Batch size')
    
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--save_dir', type=str, default=None, help='Trained model path')
    parser.add_argument('--home_dir', type=str, default=".", help='Trained model path')

    parser.add_argument('--path_to_pretrained_model', type=str, default=None, #default="/home/michail/datadisk/PretrainDas/pretrained_models_func/backbone/layer4_cutoff10_hidden128_batch32_lr0.0005_0.5_60_dropout0.3__2023-11-17 17:13:01.594391/epoch10.pt",
                        help='Pretrained model path')
    
    parser.add_argument('--disable_tqdm', default=False, action='store_true')
    args = parser.parse_args()
    
    
        
    

    if(args.dataset == 'func'):
        args.out_channels = num_func
        args.lr_decay_step_size = 60
        
    else:
        args.out_channels = num_fold
        args.epochs = 1000

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if(args.device==1):
        torch.cuda.set_device(1)
    ##### load datasets
    print('Loading Train & Val & Test Data...')
    if args.dataset == 'func':
        try:
            train_set = ECdataset(root=args.dataset_path + '/ProtFunct', split='Train')
            val_set = ECdataset(root=args.dataset_path + '/ProtFunct', split='Val')
            test_set = ECdataset(root=args.dataset_path + '/ProtFunct', split='Test')
        except FileNotFoundError: 
            print('\n Please download data firstly, following https://github.com/divelab/DIG/tree/dig-stable/dig/threedgraph/dataset#ecdataset-and-folddataset and https://github.com/phermosilla/IEConv_proteins#download-the-preprocessed-datasets \n')
            raise(FileNotFoundError)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        print('Done!')
        print('Train, val, test:', train_set, val_set, test_set)
    elif args.dataset == 'fold':
        try:
            train_set = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='training')
            val_set = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='validation')
            test_fold = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='test_fold')
            test_super = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='test_superfamily')
            test_family = FOLDdataset(root=args.dataset_path + '/HomologyTAPE', split='test_family')
        except FileNotFoundError: 
            print('\n Please download data firstly, following https://github.com/divelab/DIG/tree/dig-stable/dig/threedgraph/dataset#ecdataset-and-folddataset and https://github.com/phermosilla/IEConv_proteins#download-the-preprocessed-datasets \n')
            raise(FileNotFoundError)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        test_fold_loader = DataLoader(test_fold, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        test_super_loader = DataLoader(test_super, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        test_family_loader = DataLoader(test_family, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        print('Done!')
        print('Train, val, test (fold, superfamily, family):', train_set, val_set, test_fold, test_super, test_family)
    else:
        print('not supported dataset')

    # print(len(train_set))
    # print(train_set[0])
    # for i in train_loader:
    #     print(i)
    #     break
    # exit() # Data(side_chain_embs=[146, 8], bb_embs=[146, 6], x=[146, 1], coords_ca=[146, 3], coords_n=[146, 3], coords_c=[146, 3], id='d1v4wb_', y=[1])
    #         #DataBatch(side_chain_embs=[5088, 8], bb_embs=[5088, 6], x=[5088, 1], coords_ca=[5088, 3], coords_n=[5088, 3], coords_c=[5088, 3], id=[32], y=[32], batch=[5088], ptr=[33])

    # Misc. tools
    import os

    # Hydra tools
    import hydra

    from hydra.compose import GlobalHydra
    from hydra.core.hydra_config import HydraConfig

    from proteinworkshop.constants import HYDRA_CONFIG_PATH
    from proteinworkshop.utils.notebook import init_hydra_singleton

    version_base = "1.2"  # Note: Need to update whenever Hydra is upgraded
    init_hydra_singleton(reload=True, version_base=version_base)

    path = HYDRA_CONFIG_PATH
    rel_path = os.path.relpath(path, start=".")
    # print(rel_path)
    GlobalHydra.instance().clear()
    hydra.initialize(rel_path, version_base=version_base)

    cfg = hydra.compose(
        config_name="train",
        overrides=[
            "encoder=pronet",
            "encoder.level='aminoacid'",
            "encoder.num_blocks=4",
            "encoder.hidden_channels=128",
            "encoder.out_channels=384",
            "encoder.mid_emb=64",
            "encoder.num_radial=6",
            "encoder.num_spherical=2",
            "encoder.cutoff=10.0",
            "encoder.max_num_neighbors=32",
            "encoder.int_emb_layers=3",
            "encoder.out_layers=2",
            "encoder.num_pos_emb=16",
            "encoder.dropout=0.3",
            "encoder.data_augment_eachlayer=True",
            "encoder.euler_noise=False",
            "encoder.pretraining=False",
            "encoder.node_embedding=False",

            "decoder.graph_label.dummy=True",

            "task=multiclass_graph_classification",
            "dataset=ec_reaction",
            "dataset.datamodule.batch_size=32",
            "features=ca_base", 
            "+aux_task=none",
            
            "trainer.max_epochs=400",
            "optimiser=adam",
            "optimiser.optimizer.lr=5e-4",
            "callbacks.early_stopping.patience=200",
            "test=False",
            "scheduler=steplr",

            ## for test ONLY
            # "task_name=test",  # here
            # "ckpt_path_test=/home/zhang/Projects/3d/ProteinWorkshop/notebooks/outputs/checkpoints/epoch_275.ckpt", # here
            # "optimizer.weight_decay=0.5"
        ],
        return_hydra_config=True,
    )

    # Note: Customize as needed e.g., when running a sweep
    cfg.hydra.job.num = 0
    cfg.hydra.job.id = 0
    cfg.hydra.hydra_help.hydra_help = False
    cfg.hydra.runtime.output_dir = "outputs"

    HydraConfig.instance().set_config(cfg)
    from proteinworkshop.datasets.atom3d_datamodule import ATOM3DDataModule
    from omegaconf import OmegaConf
    from proteinworkshop.configs import config

    cfg = config.validate_config(cfg)
    # print("Original config:\n", OmegaConf.to_yaml(cfg))
    mutable_cfg = OmegaConf.to_container(cfg.dataset.datamodule, resolve=True)
    mutable_cfg = OmegaConf.create(mutable_cfg)
    # print("Cloned config:\n", OmegaConf.to_yaml(mutable_cfg))
    # Instantiate the datamodule with the mutable configuration
    datamodule = hydra.utils.instantiate(mutable_cfg)
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    from torch import nn
    featuriser: nn.Module = hydra.utils.instantiate(cfg.features)

    ##### set up model
    model = ProNet(num_blocks=args.num_blocks, hidden_channels=args.hidden_channels, out_channels=args.out_channels,
            cutoff=args.cutoff, dropout=args.dropout,
            data_augment_eachlayer=args.data_augment_eachlayer,
            euler_noise = args.euler_noise, level=args.level)
    is_pretrained = "Not_Pretrained"
    if(args.path_to_pretrained_model is not None):
        is_pretrained = "Pretrained"
        print('Loading pretrained model')
        checkpoint = torch.load(args.path_to_pretrained_model,map_location=f'cuda:{args.device}')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.lin_out = nn.Linear(args.hidden_channels, args.out_channels)
    model.pretraining = False
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor)
    
    
    if args.continue_training:
        save_dir = args.save_dir
        checkpoint = torch.load(save_dir + '/best_val.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        save_dir = './downstream_trained_models_{dataset}/{level}/{is_pretrained}__{time}'.format(
            dataset=args.dataset, level=args.level, is_pretrained=is_pretrained, time=datetime.now())
        print('saving to...', save_dir)
        start_epoch = 1
        
    num_params = sum(p.numel() for p in model.parameters()) 
    print('num_parameters:', num_params)
   
    if not save_dir == "" and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(save_dir + "/args.txt", 'w') as f:
        print(args,file=f)
        
    if args.dataset == 'func':
        writer = SummaryWriter(log_dir=save_dir)
        best_val_acc = 0
        test_at_best_val_acc = 0
        
        for epoch in range(start_epoch, args.epochs+1):
            print('==== Epoch {} ===='.format(epoch))
            t_start = time.perf_counter()
            
            train_loss, train_acc = train(args, model, train_loader, optimizer, device, featuriser)
            t_end_train = time.perf_counter()
            val_loss, val_acc = evaluation(args, model, val_loader, device, featuriser)
            t_start_test = time.perf_counter()
            test_loss, test_acc = evaluation(args, model, test_loader, device, featuriser)
            t_end_test = time.perf_counter() 

            if not save_dir == "" and val_acc > best_val_acc:
                print('Saving best val checkpoint ...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
                torch.save(checkpoint, save_dir + '/best_val.pt')
                best_val_acc = val_acc    
                test_at_best_val_acc = test_acc       

            t_end = time.perf_counter()
            print('Train: Loss:{:.6f} Acc:{:.4f}, Validation: Loss:{:.6f} Acc:{:.4f}, Test: Loss:{:.6f} Acc:{:.4f}, test_acc@best_val:{:.4f}, time:{}, train_time:{}, test_time:{}'.format(
                train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, test_at_best_val_acc, t_end-t_start, t_end_train-t_start, t_end_test-t_start_test))
            
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_acc', val_acc, epoch)
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)
            writer.add_scalar('test_at_best_val_acc', test_at_best_val_acc, epoch)

            scheduler.step()   
        
        writer.close()    
        # Save last model
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
        torch.save(checkpoint, save_dir + "/epoch{}.pt".format(epoch))
        with open(args.home_dir + '/results.txt', 'a') as f:
            print(args,file=f)
            print("test_at_best_val_acc",test_at_best_val_acc,"\n",file=f)
        
        
        
    elif args.dataset == 'fold':
        writer = SummaryWriter(log_dir=save_dir)
        best_val_acc = 0
        test_fold_at_best_val_acc = 0
        test_super_at_best_val_acc = 0
        test_family_at_best_val_acc = 0
        
        for epoch in range(start_epoch, args.epochs+1):
            print('==== Epoch {} ===='.format(epoch))
            t_start = time.perf_counter()
            
            train_loss, train_acc = train(args, model, train_loader, optimizer, device)
            t_end_train = time.perf_counter()
            val_loss, val_acc = evaluation(args, model, val_loader, device)
            t_start_test = time.perf_counter()
            test_fold_loss, test_fold_acc = evaluation(args, model, test_fold_loader, device)
            test_super_loss, test_super_acc = evaluation(args, model, test_super_loader, device)
            test_family_loss, test_family_acc = evaluation(args, model, test_family_loader, device)
            t_end_test = time.perf_counter()

            if not save_dir == "" and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if not save_dir == "" and val_acc > best_val_acc:
                print('Saving best val checkpoint ...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
                torch.save(checkpoint, save_dir + '/best_val.pt')
                best_val_acc = val_acc    
                test_fold_at_best_val_acc = test_fold_acc
                test_super_at_best_val_acc = test_super_acc
                test_family_at_best_val_acc = test_family_acc       

            t_end = time.perf_counter()
            print('Train: Loss:{:.6f} Acc:{:.4f}, Validation: Loss:{:.6f} Acc:{:.4f}, '\
                'Test_fold: Loss:{:.6f} Acc:{:.4f}, Test_super: Loss:{:.6f} Acc:{:.4f}, Test_family: Loss:{:.6f} Acc:{:.4f}, '\
                'test_fold_acc@best_val:{:.4f}, test_super_acc@best_val:{:.4f}, test_family_acc@best_val:{:.4f}, '\
                'time:{}, train_time:{}, test_time:{}'.format(
                train_loss, train_acc, val_loss, val_acc, 
                test_fold_loss, test_fold_acc, test_super_loss, test_super_acc, test_family_loss, test_family_acc, 
                test_fold_at_best_val_acc, test_super_at_best_val_acc, test_family_at_best_val_acc, 
                t_end-t_start, t_end_train-t_start, t_end_test-t_start_test))
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_acc', val_acc, epoch)
            writer.add_scalar('test_fold_loss', test_fold_loss, epoch)
            writer.add_scalar('test_fold_acc', test_fold_acc, epoch)
            writer.add_scalar('test_super_loss', test_super_loss, epoch)
            writer.add_scalar('test_super_acc', test_super_acc, epoch)
            writer.add_scalar('test_family_loss', test_family_loss, epoch)
            writer.add_scalar('test_family_acc', test_family_acc, epoch)
            writer.add_scalar('test_fold_at_best_val_acc', test_fold_at_best_val_acc, epoch)
            writer.add_scalar('test_super_at_best_val_acc', test_super_at_best_val_acc, epoch)
            writer.add_scalar('test_family_at_best_val_acc', test_family_at_best_val_acc, epoch)

            scheduler.step()   
            
        writer.close()
        # Save last model
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
        torch.save(checkpoint, save_dir + "/epoch{}.pt".format(epoch))
        with open(args.home_dir + '/results.txt', 'a') as f:
            print(args,file=f)
            print("test_fold_at_best_val_acc",test_fold_at_best_val_acc,
                "test_super_at_best_val_acc",test_super_at_best_val_acc,"test_family_at_best_val_acc",test_family_at_best_val_acc,"\n",file=f)
            
     
        
if __name__ == "__main__":  
    main()