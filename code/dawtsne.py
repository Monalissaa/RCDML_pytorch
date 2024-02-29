import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import sys
import numpy as np
# from myNetwork import network
# from iCIFAR100 import iCIFAR100
# from autoaugment import RandAugment
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

def compute_embed(loader, model):
    total_embeddings = np.zeros((len(loader) * loader.batch_size, 512))
    total_labels = np.zeros(len(loader) * loader.batch_size)
    for idx, (_, images, _, _,_,_,labels) in enumerate(loader):
        #print(labels)
        images = images.cuda()
        bsz = labels.shape[0]
        embed =  model.feature_extractor(images)
        #embed = model.projection(images)
        #embed = model.mlp(embed)
        total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
        total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()
        del images, labels, embed
        torch.cuda.empty_cache()
    return np.float32(total_embeddings), total_labels.astype(int)

train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.24705882352941178),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
test_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

train_dataset = iCIFAR100('./dataset', transform=train_transform, transform2=train_transform,download=True)
test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)

numclass = 10
classes = [0, numclass]
#classes = [numclass - self.task_size, self.numclass]
train_dataset.getTrainData(classes)
train_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=64)

cuda_index = 'cuda:' + '0'
device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
print(device)
filename = 'pa_model_saved_check/cifar100_10_9*10/100_model_PASS.pkl'
#filename = 'model_saved_check/cifar100_10_9*10/10_model.pkl'
model = torch.load(filename)
model.to(device)
model.eval()

embeddings, labels = compute_embed(train_loader, model)
print(labels)

print('Embedding Complit!')
# embeddings_tsne = TSNE.fit_transform(embeddings,y=None)
print('Tsneing working!')
embeddings_tsne = TSNE()._fit(embeddings)
vis_x = embeddings_tsne[:, 0]
vis_y = embeddings_tsne[:, 1]
plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap("jet", numclass), marker='.',alpha=0.6)
plt.colorbar(ticks=range(numclass))
plt.savefig("hrtsne.png")
print('Drawing!')
plt.show()

