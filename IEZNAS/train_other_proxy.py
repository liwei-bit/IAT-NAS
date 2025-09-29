#!/usr/bin/env python
"""
train_other_proxy.py – 统一训练并评估多种 NAS 架构（搜索结果或 baseline）
--------------------------------------------------------------------------
• 支持不平衡 CIFAR10 / CIFAR100（自定义 get_dataloader）
• 支持 PathMNIST / OrganAMNIST / DermaMNIST（三元组返回）
• 自动统计均值 ± 方差、每类精度
"""
import os, json, argparse, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models as tv_models
from tqdm import tqdm

from medmnist_imbalance import get_medmnist_loader
from Dataset_partitioning import get_dataloader            # CIFAR 专用
from lib.models.nas201_model import build_model_from_arch_str

# reproducibility
def set_seed(sd=0):
    random.seed(sd); np.random.seed(sd)
    torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# simple baseline builder
def build_baseline(name: str, in_ch: int, n_cls: int):
    name = name.lower()
    if name in {"resnet18","resnet34"}:
        net = getattr(tv_models, name)(num_classes=n_cls)
        net.conv1 = nn.Conv2d(in_ch,64,3,1,1,bias=False); net.maxpool = nn.Identity(); return net
    if name == "mobilenet_v2":
        net = tv_models.mobilenet_v2(num_classes=n_cls)
        net.features[0][0] = nn.Conv2d(in_ch,32,3,1,1,bias=False); return net
    raise ValueError(name)

# one epoch
def run_epoch(model, loader, crit, opt, dev, train=True):
    model.train() if train else model.eval()
    tot, cor = 0, 0
    cls_cor = np.zeros(crit.weight.numel()); cls_tot = np.zeros_like(cls_cor)
    for x,y in loader:
        x,y = x.to(dev), y.squeeze().to(dev)
        if train: opt.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(x); loss = crit(out,y)
            if train: loss.backward(); opt.step()
        pred = out.argmax(1)
        tot += y.size(0); cor += pred.eq(y).sum().item()
        for t,p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cls_tot[t]+=1; cls_cor[t]+= (t==p)
    acc = 100.*cor/tot
    per_cls = 100.*cls_cor/(cls_tot+1e-9)
    return acc, per_cls

# train + eval
def train_and_eval(model, loaders, epochs, dev, n_cls):
    train_loader, val_loader = loaders
    # class weights
    base_ds = train_loader.dataset.dataset if isinstance(train_loader.dataset, Subset) else train_loader.dataset
    labels = np.array(base_ds.targets if hasattr(base_ds,'targets') else base_ds.labels)
    if isinstance(train_loader.dataset, Subset):
        labels = labels[train_loader.dataset.indices]
    counts = np.bincount(labels.squeeze(), minlength=n_cls).astype(float)
    w = 1./(counts+1e-6); w = np.clip(w,0,50); w = w/w.sum()*len(w)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w,device=dev,dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(),1e-3,weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best_acc, best_pc = 0., None
    for ep in range(1,epochs+1):
        run_epoch(model,train_loader,crit,opt,dev,True)
        acc, pc = run_epoch(model,val_loader,crit,opt,dev,False)
        sch.step()
        if acc>best_acc: best_acc, best_pc = acc, pc
        if ep==1 or ep==epochs or ep%max(1,epochs//5)==0:
            print(f"  [ep {ep}/{epochs}] val_acc={acc:.2f}")
    return best_acc, best_pc

# main

def main(args):
    with open(args.arch_json) as f:
        archs=list(json.load(f).keys()) if isinstance(json.load(open(args.arch_json)),dict) else json.load(f)

    # dataloader
    if args.dataset in {'pathmnist','organamnist','dermamnist'}:
        tr_loader,in_ch,n_cls = get_medmnist_loader(args.dataset,batch_size=args.batch_size,split='train')
        val_loader,_,_        = get_medmnist_loader(args.dataset,batch_size=args.batch_size,split='test')
    else:
        tr_loader = get_dataloader(dataset=args.dataset,imbalance_type=args.imb_type,imbalance_ratio=args.imb_ratio,batch_size=args.batch_size)
        in_ch = 3; n_cls = 10 if args.dataset=='cifar10' else 100
        tf_val = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
        val_ds = getattr(datasets,args.dataset.upper())(root='./datasets',train=False,download=True,transform=tf_val)
        val_loader = DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=2)
    loaders=(tr_loader,val_loader)

    seeds=[int(s) for s in args.seeds.split(',')]
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results=[]
    for arch in archs:
        accs,pcs=[],[]
        print(f"\n==== {arch} ====")
        for sd in seeds:
            set_seed(sd)
            model = build_model_from_arch_str(arch,in_channels=in_ch,num_classes=n_cls) if '+' in arch else build_baseline(arch,in_ch,n_cls)
            model.to(dev)
            acc,pc = train_and_eval(model,loaders,args.epochs,dev,n_cls)
            accs.append(acc); pcs.append(pc)
        accs,pcs=np.array(accs),np.array(pcs)
        results.append({'arch':arch,'mean_acc':float(accs.mean()),'std_acc':float(accs.std()),'mean_per_class':pcs.mean(0).tolist(),'std_per_class':pcs.std(0).tolist()})
        print(f"=> {arch}: {accs.mean():.2f} ± {accs.std():.2f}%")

    results.sort(key=lambda d:d['mean_acc'],reverse=True)
    os.makedirs('results',exist_ok=True)
    out=f"results/{args.dataset}_arch_90.json"; json.dump(results,open(out,'w'),indent=2)
    print(f"\nSaved → {out}")

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--arch_json',required=True)
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','pathmnist','organamnist','dermamnist'])
    ap.add_argument('--seeds',default='0,1,2,3,4')
    ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--batch_size',type=int,default=128)
    ap.add_argument('--imb_type',default='exp'); ap.add_argument('--imb_ratio',type=float,default=0.05)
    args=ap.parse_args(); main(args)


#  python train_other_proxy.py --arch_json arch_selected.json --dataset cifar10 --seeds 0,1,2 --epochs 10