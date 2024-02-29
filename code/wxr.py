from numpy import load
from os.path import join as join
data = load('/home/wangxinru/.cache/torch/hub/checkpoints/ViT-B_16.npz')
lst = data.files
for item in lst:
    print(item)
    #print(data[item])

