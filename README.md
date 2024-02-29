
# Relationship Constraint Deep Metric Learning

Official PyTorch implementation of Relationship Constraint Deep Metric Learning. 

## Requirements

- Python3
- PyTorch (> 1.0)
- NumPy
- tqdm
- wandb
- [Pytorch-Metric-Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)



## Datasets

1. Download four public benchmarks for deep metric learning
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   - Cars-196 ([Img](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotation](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
   - Stanford Online Products ([Link](https://cvgl.stanford.edu/projects/lifted_struct/))

2. Extract the tgz or zip file into `./data/` (Exceptionally, for Cars-196, put the files in a `./data/cars196`)

## Training Embedding Network

### CUB-200-2011


```bash
python resave/train_best.py --seed 2412 \
                --src 0.01 \  # the weight of Sample Relationship Constraint
                --prc 0.05 \  # the weight of Proxy Relationship Constraint
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 128 \
                --lr 1e-4 \
                --dataset cub 
```


## Evaluating Image Retrieval

Follow the below steps to evaluate the provided pretrained model or your trained model. 

Trained best model will be saved in the `./logs/folder_name`.

```bash
# The parameters should be changed according to the model to be evaluated.
python code/evaluate.py --gpu-id 0 \
                   --batch-size 128 \
                   --model bn_inception \
                   --embedding-size 512 \
                   --dataset cub \
                   --resume /set/your/model/path/best_model.pth
```



## Acknowledgements

Our code is modified and adapted on these great repositories:

- [No Fuss Distance Metric Learning using Proxies](https://github.com/dichotomies/proxy-nca)
- [PyTorch Metric learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
- [Proxy Anchor Loss for Deep Metric Learning](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020)
