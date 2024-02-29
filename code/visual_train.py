#from model.bn_inception import bn_inception
from tqdm import tqdm

import dataset
import torch
from sklearn import datasets
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from net.bn_inception import *
from dataset import sampler
from sklearn.manifold import TSNE
import random
import os

seed = 56
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def plot_embedding(data,label):
    x_min,x_max=np.min(data,0),np.max(data,0)
    data = (data - x_min)/(x_max - x_min)
    fig=plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i,0],data[i,1],str(label[i]),color=plt.cm.Set3(label[i]-90),fontdict={'weight':'bold','size':9})
    plt.xticks([])
    plt.yticks([])
    return fig
def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        for batch in dataloader:
            for i, J in enumerate(batch):
                if i == 0:
                    J = model(J.cuda())
                for j in J:
                    A[i].append(j)
            break
    model.train()
    model.train(model_is_training)
    return [torch.stack(A[i]) for i in range(len(A))]

trn_dataset = dataset.load(
    name='cub',
    root='/home/wangxinru/program/Proxy-Anchor-CVPR2020-master/data',
    mode='train',
    transform=dataset.utils.make_transform(
        is_train=True,
        is_inception=True
    ))
balanced_sampler = sampler.BalancedSampler(trn_dataset, batch_size=200, images_per_class = 20)
batch_sampler = BatchSampler(balanced_sampler, batch_size = 200, drop_last = True)
dl_tr = torch.utils.data.DataLoader(
    trn_dataset,
    # batch_size=args.sz_batch,
    # shuffle=True,
    num_workers=4,
    # drop_last=True,
    pin_memory=True,
    batch_sampler = batch_sampler
)

#saved_model_dir = r"/resave/log_c/cub_bn_inception_best.pth"
# saved_model_dir=r"/home/wangxinru/program/Proxy-Anchor-CVPR2020-master/resave/log_c/cub_bn_inception_best_original.pth"
saved_model_dir = r"/home/wangxinru/program/Proxy-Anchor-CVPR2020-master/logs/logs_cub/bn_inception_Proxy_Anchor_embedding512_alpha32_mrg0.1_adamw_lr0.0001_batch180/cub_bn_inception_best.pth"
model = bn_inception(embedding_size=512, pretrained=True, is_norm=1, bn_freeze = 1)
# for name in model.state_dict():
#    print(name)
model.load_state_dict(torch.load(saved_model_dir)['model_state_dict'])
model = model.cuda()
X, T = predict_batchwise(model, dl_tr)
print(X)

tsne = TSNE(n_components=2,init='pca',random_state=0)
result = tsne.fit_transform(X.cpu().detach().numpy())
print(T.cpu().detach().numpy())
fig = plot_embedding(result,T.cpu().detach().numpy())
# proxies = torch.load("/home/wangxinru/program/Proxy-Anchor-CVPR2020-master/proxies")['proxy_dict']['params'][0]
L = T.cpu().detach().numpy()[0]
plt.savefig("/home/wangxinru/program/Proxy-Anchor-CVPR2020-master/resave_image/{}_train".format(L))
plt.show()




# print(proxies)
# print(proxies.size())

