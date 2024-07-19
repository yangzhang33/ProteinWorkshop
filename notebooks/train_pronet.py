# Misc. tools
import os

# Hydra tools
import hydra

from hydra.compose import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from proteinworkshop.constants import HYDRA_CONFIG_PATH
from proteinworkshop.utils.notebook import init_hydra_singleton

init_hydra_singleton(reload=True)

path = HYDRA_CONFIG_PATH
rel_path = os.path.relpath(path, start=".")

GlobalHydra.instance().clear()
hydra.initialize(rel_path)

cfg = hydra.compose(
    "train",
    overrides=[
        "dataset=afdb_swissprot_v4",
        "dataset.datamodule.batch_size=32",
        "dataset.datamodule.train_split=0.02", # here
        "dataset.datamodule.val_split=0.001", # here
        "features=fe_subgraph",

        "task=subgraph_distance_prediction", # here
        ],
    return_hydra_config=False,
)

from proteinworkshop.configs import config

cfg = config.validate_config(cfg)

print(cfg.keys())
for key in cfg.keys():
    print(key)
    print(cfg[key])


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
dl = datamodule.train_dataloader()

for i in dl:
    print(i)
    break

from torch import nn
featuriser: nn.Module = hydra.utils.instantiate(cfg.features)

for i in dl:
    batch = featuriser(i)
    print(batch)
    break

from tqdm import tqdm
import numpy as np
import time
from torch_scatter import scatter_mean, scatter
import torch
def train(args, model, mlp_pred_dist, train_loader,  criterion, optimizer, device):
    model.train()
    mlp_pred_dist.train()
    loss_accum = 0
   
    # shuffle the train batches and all_subgraphs
    #random_idx = np.random.permutation(len(train_batches))
    #train_batches = [train_batches[i] for i in random_idx]
    
    
    #for step, batch in enumerate(tqdm(loader, disable=args.disable_tqdm)):
    for step, batch in enumerate(tqdm(train_loader, disable=args.disable_tqdm)):
        batch = featuriser(batch)
        #init_idx = random_idx[step]
        init_idx = step
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
                      
        pred = model(batch) 
        
        subgraphs = batch.subgraphs
        dist = batch.subgraph_distances.to(device)

        #aggregate node representations of each subgraph

        pooled_subgraphs = []
        for i in range(len(subgraphs)):
            pooled_subgraphs.append(torch.sum(pred[subgraphs[i]], dim=0))

        pooled_subgraphs = torch.stack(pooled_subgraphs)
        graph_repr = scatter(pred, batch.batch, dim=0)
        # for lin in model.lins_out:
        #     pooled_subgraphs = model.relu(lin(pooled_subgraphs))
        #     pooled_subgraphs = model.dropout(pooled_subgraphs)     
        
        #protein representations
        #compute the center of the subgraphs based on the coordinates
       
        #compute the center of the protein based on the coordinates
        # compute the distance between the center of the subgraphs and the center of the proteins
        # we have to compute the distance only between the subgraph and the corresponding protein
        # repeat the center of the protein for each subgraph and the perform the distance computation
        #G_c = G_c.repeat(num_subgraphs_per_protein,1)

        
        #dist = torch.norm(G_c-S_c,dim=1)
        #concat the subgraph and the protein representations. find the graph_repr that corresponds to the subgraph of the protein
        fused_repr = []
        for i in range(len(pooled_subgraphs)):
            fused_repr.append(torch.cat((pooled_subgraphs[i],graph_repr[batch.batch[subgraphs[i][0]]])))
        fused_repr = torch.stack(fused_repr)
        #predict the distance
        pred_dist = mlp_pred_dist(fused_repr)
        pred_dist = pred_dist.squeeze()
        #normalize the distance (note maybe we should normalize in the whole dataset and not in each bach)
        y_dist = dist/torch.max(dist)
        #mse loss
        optimizer.zero_grad()

        loss = criterion(pred_dist, y_dist)
        loss.backward()
        optimizer.step()
        loss_accum += loss.item()
        if(step%300==0):
            print(loss_accum/(step + 1))
        #### end pretraining
        ######
    print(loss_accum/(step + 1) )
    return loss_accum/(step + 1)

def evaluation(args, model, mlp_pred_dist, loader, criterion, device):    
    model.eval()
    
    loss_accum = 0
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
                     
        
        pred = model(batch) 
        
        

        subgraphs = batch.subgraphs
        dist = batch.subgraph_distances.to(device)
        #### pretraining
        #aggregate node representations of each subgraph


        pooled_subgraphs = []
        for i in range(len(subgraphs)):
            pooled_subgraphs.append(torch.sum(pred[subgraphs[i]], dim=0))
      
        pooled_subgraphs = torch.stack(pooled_subgraphs)

        # for lin in model.lins_out:
        #     pooled_subgraphs = model.relu(lin(pooled_subgraphs))
        #     pooled_subgraphs = model.dropout(pooled_subgraphs)    
        #protein representations
        #compute the center of the subgraphs based on the coordinates

        #compute the center of the protein based on the coordinates
        # compute the distance between the center of the subgraphs and the center of the proteins
        # we have to compute the distance only between the subgraph and the corresponding protein
        # repeat the center of the protein for each subgraph and the perform the distance computation
        #G_c = G_c.repeat(num_subgraphs_per_protein,1)

        
        #dist = torch.norm(G_c-S_c,dim=1)
        #normalize the distance (note maybe we should normalize in the whole dataset and not in each bach)
        graph_repr = scatter(pred, batch.batch, dim=0)
        fused_repr = []
        for i in range(len(pooled_subgraphs)):
            fused_repr.append(torch.cat((pooled_subgraphs[i],graph_repr[batch.batch[subgraphs[i][0]]])))
        fused_repr = torch.stack(fused_repr)
        y_dist = dist/torch.max(dist)
        #predict the distance
        pred_dist = mlp_pred_dist(fused_repr)
        pred_dist = pred_dist.squeeze()
        #mse loss
        loss = criterion(pred_dist, y_dist)
        loss_accum += loss.item()
        if(step %100 == 0):
            print(loss_accum/(step + 1))
    return loss_accum/(step + 1) 

#     ### Args
import argparse
import sys
import torch
from pronet import ProNet
import torch.optim as optim
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
import time

sys.argv = ['notebook']
parser = argparse.ArgumentParser()

parser.add_argument('--device', type=int, default=0, help='Device to use')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers in Dataloader')

### Data
# parser.add_argument('--dataset', type=str, default='alphafold', help='Func or fold or all')
# parser.add_argument('--dataset_path', type=str, default='/datalake/datastore2/alphafold_v4_pronet_processed', help='path to load and process the data')
# parser.add_argument('--annot_fn', type=str, default="/home/michail/datadisk/PretrainDas/data/GO_EC_labels_deepfri/nrPDB-GO_2019.06.18_annot.tsv")
# parser.add_argument('--ontology', type=str, default="ec")

# data augmentation tricks, see appendix E in the paper (https://openreview.net/pdf?id=9X-hgLDLYkQ)
parser.add_argument('--mask', action='store_true', help='Random mask some node type')
parser.add_argument('--noise', default=False, action='store_true', help='Add Gaussian noise to node coords')
parser.add_argument('--deform', default=False, action='store_true', help='Deform node coords')
parser.add_argument('--data_augment_eachlayer', default=True, action='store_true', help='Add Gaussian noise to features')
parser.add_argument('--euler_noise', default=False, action='store_true', help='Add Gaussian noise Euler angles')
parser.add_argument('--mask_aatype', type=float, default=0.1, help='Random mask aatype to 25(unknown:X) ratio')

### Model
parser.add_argument('--model_name', type=str, default='pronet', help='rgcn,pronet')
#for pronet
parser.add_argument('--level', type=str, default='aminoacid', help='Choose from \'aminoacid\', \'backbone\', and \'allatom\' levels')
parser.add_argument('--num_blocks', type=int, default=4, help='Model layers')
parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden dimension')
parser.add_argument('--out_channels', type=int, default=384, help='Number of classes, 1195 for the fold data, 384 for the ECdata')
parser.add_argument('--fix_dist', action='store_true')  
parser.add_argument('--cutoff', type=float, default=10, help='Distance constraint for building the protein graph') 
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout')
parser.add_argument('--precompute_subgraphs', type=int, default=0, help='Compute the subgraphs')

## Training hyperparameter
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--lr_decay_step_size', type=int, default=150, help='Learning rate step size')
parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='Learning rate factor') 
parser.add_argument('--weight_decay', type=float, default=0, help='Weight Decay')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size during training')
parser.add_argument('--eval_batch_size', type=int, default=64, help='Batch size')



parser.add_argument('--continue_training', action='store_true')
parser.add_argument('--save_dir', type=str, default="./logs", help='Trained model path')

parser.add_argument('--disable_tqdm', default=False, action='store_true')
args = parser.parse_args()

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
# if(args.device == 1):
#     torch.cuda.set_device(1)

##### load datasets
print('Loading Train & Val & Test Data...')




train_loader = datamodule.train_dataloader()

#check if file exists


##### set up model
if(args.model_name == "pronet"):
    model = ProNet(num_blocks=args.num_blocks, hidden_channels=args.hidden_channels, out_channels=args.out_channels,
            cutoff=args.cutoff, dropout=args.dropout,
            data_augment_eachlayer=args.data_augment_eachlayer,
            euler_noise = args.euler_noise, level=args.level, pretraining=True)
else:
    model = RGCN(input_dim=input_dim, hidden_dim=args.hidden_channels, n_layers=6, emb_dim=args.out_channels, dropout=args.dropout, pretraining=True)
    
model.to(device)

mlp_pred_dist = torch.nn.Sequential(
    torch.nn.Linear(2*args.hidden_channels, args.hidden_channels),
    torch.nn.ReLU(),
    torch.nn.Linear(args.hidden_channels, 1)
).to(device)

#linear_pred_dist= torch.nn.Linear(2*args.hidden_channels, 1).to(device)


optimizer = optim.Adam(list(model.parameters())+list(mlp_pred_dist.parameters()), lr=args.lr, weight_decay=args.weight_decay) 
criterion = torch.nn.MSELoss()

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor)


if args.continue_training:
    save_dir = args.save_dir
    checkpoint = torch.load(save_dir + '/best_val.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
else:
    # save_dir = './pretrained_models_{dataset}/{level}/layer{num_blocks}_cutoff{cutoff}_hidden{hidden_channels}_batch{batch_size}_lr{lr}_{lr_decay_factor}_{lr_decay_step_size}_dropout{dropout}__{time}'.format(
    #     dataset=args.dataset, level=args.level, 
    #     num_blocks=args.num_blocks, cutoff=args.cutoff, hidden_channels=args.hidden_channels, batch_size=args.batch_size, 
    #     lr=args.lr, lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size, dropout=args.dropout, time=datetime.now())
    # print('saving to...', save_dir)
    start_epoch = 1
    
num_params = sum(p.numel() for p in model.parameters()) 
print('num_parameters:', num_params)


# writer = SummaryWriter(log_dir=save_dir)
#  best_val_loss = 1000
# test_at_best_val_loss = 1000

    
# print("loading edge_index")
# with open("edge_index_pronet_64.pkl","rb") as f:
#     edge_index = pickle.load(f)
# print("edge_index loaded")


    
print(len(train_loader))
# exit()

# print("Loading subgraphs")
# if(args.precompute_subgraphs==1):
#     print("Preprocessing - Compute Subgraphs")
#     train_subgraphs,train_dist = compute_subgraphs(train_loader, args=args, device=device)
#     with open(f'./subgraphs/alphafold_train_subgraphs_{args.batch_size}_490k.pkl', 'wb') as f:
#         pickle.dump(train_subgraphs, f)
#     with open(f'./subgraphs/alphafold_train_dist_{args.batch_size}_490k.pkl', 'wb') as f:
#         pickle.dump(train_dist, f)
# else:
#     with open(f'./subgraphs/alphafold_train_subgraphs_{args.batch_size}_490k.pkl', 'rb') as f:
#         train_subgraphs = pickle.load(f)
#     with open(f'./subgraphs/alphafold_train_dist_{args.batch_size}_490k.pkl', 'rb') as f:
#         train_dist = pickle.load(f)
    


print("Loaded subgraphs")


for epoch in range(start_epoch, args.epochs+1):
    print('==== Epoch {} ===='.format(epoch))
    t_start = time.perf_counter()
    
    train_loss = train(args, model, mlp_pred_dist, train_loader, criterion, optimizer, device)
    t_end_train = time.perf_counter()
    # val_loss = evaluation(args, model, linear_pred_dist, val_loader, criterion, device)
    # t_start_test = time.perf_counter()
    # test_loss = evaluation(args, model, linear_pred_dist, test_loader, criterion, device)
    
    
    # t_end_test = time.perf_counter() 

    # if not save_dir == "" and not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    t_end = time.perf_counter()
    print('Train: Loss:{:.6f}, time:{}, train_time:{}'.format(
        train_loss, t_end-t_start, t_end_train-t_start))
    
    # writer.add_scalar('train_loss', train_loss, epoch)

    # scheduler.step()   

    # writer.close()    
    print("Train Loss", train_loss)
    # Save last model
    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()} #'scheduler_state_dict': scheduler.state_dict()}
    # torch.save(checkpoint, save_dir + "/epoch{}.pt".format(epoch))