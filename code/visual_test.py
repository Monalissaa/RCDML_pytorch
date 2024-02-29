#from model.bn_inception import bn_inception
from tqdm import tqdm
import seaborn as sns
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
    # print(data.shape[0])
    # print(label)
    # for i in range(data.shape[0]):
    #     #plt.text(data[i,0],data[i,1],str(label[i]),color=plt.cm.Set3(label[i]-100),fontdict={'weight':'bold','size':9})
    #     plt.scatter(data[i,0],data[i,1], color=plt.cm.Set3(label[i]-100))
    vis_x = data[:, 0]
    vis_y = data[:, 1]
    sns.scatterplot(x = vis_x,y = vis_y,hue=label,palette='deep')
    #plt.scatter(vis_x, vis_y, c=label, cmap=plt.cm.get_cmap("jet", 10), marker='.', alpha=0.6)
    # plt.colorbar(ticks=range(10))
    plt.savefig("hrtsne.png")
    plt.show()
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

ev_dataset = dataset.load(
    name='cub',
    root='/home/wangxinru/program/Proxy-Anchor-CVPR2020-master/data',
    mode='eval',
    transform=dataset.utils.make_transform(
        is_train=False,
        is_inception=True
    ))

balanced_sampler = sampler.BalancedSampler(ev_dataset, batch_size=300, images_per_class = 30)
batch_sampler = BatchSampler(balanced_sampler, batch_size = 200, drop_last = True)
dl_ev = torch.utils.data.DataLoader(
    ev_dataset,
    num_workers = 4,
    pin_memory = True,
    batch_sampler = batch_sampler
)

#saved_model_dir = r"/resave/log_c/cub_bn_inception_best.pth"
#saved_model_dir=r"/home/wangxinru/program/Proxy-Anchor-CVPR2020-master/bn_inception-52deb4733.pth"
saved_model_dir = r"/home/wangxinru/program/Proxy-Anchor-CVPR2020-master/logs/logs_cub/bn_inception_Proxy_Anchor_embedding512_alpha32_mrg0.1_adamw_lr0.0001_batch180/cub_bn_inception_best.pth"
model = bn_inception(embedding_size=512, pretrained=True, is_norm=1, bn_freeze = 1)
# for name in model.state_dict():
#    print(name)
#model.load_state_dict(torch.load(saved_model_dir)['model_state_dict'])
model = model.cuda()
X, T = predict_batchwise(model, dl_ev)
# print(X.size())
# print(T.size())

tsne = TSNE(n_components=2,init='pca',random_state=0)
result = tsne.fit_transform(X.cpu().detach().numpy())
#print(result.type())
# re = torch.from_numpy(result)
# print(re.size())
fig = plot_embedding(result,T.cpu().detach().numpy())
# proxies = torch.load("/home/wangxinru/program/Proxy-Anchor-CVPR2020-master/proxies")['proxy_dict']['params'][0]
#plt.show()
# plt.savefig(r"/home/wangxinru/program/Proxy-Anchor-CVPR2020-master/image/image2.png")

# print(proxies)
# print(proxies.size())

