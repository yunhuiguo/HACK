# Unsupervised Feature Learning with Emergent Data-Driven Prototypicality


> [**Unsupervised Feature Learning with Emergent Data-Driven Prototypicality**]([https://arxiv.org/pdf/2403.10663](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_Unsupervised_Feature_Learning_with_Emergent_Data-Driven_Prototypicality_CVPR_2024_paper.pdf))            
> [Yunhui Guo, Youren Zhang, Yubei Chen, Stella X Yu]         
> CVPR 2024


## HACK

1. Generate particle embeddings on MNIST

```bash
python particle_simulation.py --dataset mnist --class_idx 0 --use_hyperbolic_dist --use_nonlinear_repulsion_loss --lr 0.01  --dim 2 --k 1.55 --epochs 600  --batch-size 1024 --c 1.0   --max_clip_norm 15.0
```

Generate particle embeddings on CIFAR10

```bash
python particle_simulation.py  --dataset cifar10 --class_idx 0 --use_hyperbolic_dist --use_nonlinear_repulsion_loss --lr 0.01  --dim 2 --k 1.55 --epochs 600  --batch-size 1024 --c 1.0   --max_clip_norm 15.0

```

2. Run assignment on MNIST

```bash
python hyperbolic_assignment.py --dataset mnist --class_idx 0  --lr 0.1 --epochs 200  --batch-size 1024 --c 1.0   --max_clip_norm 15
```

Run assignment on CIFAR10

```bash
python hyperbolic_assignment.py --dataset cifar10 --epochs 200 --class_idx 0  --lr 0.1  --batch-size 1024 --c 1.0   --max_clip_norm 15.0
```


## Citation

If our work has been helpful to you, we would greatly appreciate a citation.

```
@inproceedings{guo2024unsupervised,
  title={Unsupervised feature learning with emergent data-driven prototypicality},
  author={Guo, Yunhui and Zhang, Youren and Chen, Yubei and Yu, Stella X},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23199--23208},
  year={2024}
}
```
