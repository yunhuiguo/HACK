import torch
import shutil
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import math

import importlib



def adjust_learning_rate_200_epochs(optimizer, epoch, args):

    if epoch > 190:
        
        lr = args.lr*0.001
    
    elif epoch > 180:

        lr = args.lr*0.01
    
    elif epoch > 160:

        lr = args.lr*0.1

    else:

        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print ("lr: ", lr)




def adjust_learning_rate_1000_epochs(optimizer, epoch, args):


    if epoch > 800:

        lr = args.lr*0.01
    
    elif epoch > 400:

        lr = args.lr*0.1

    else:

        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print ("lr: ", lr)



def cosine_anneal_schedule(optimizer, t, args):
    T = 201
    M = 4
    alpha_zero = args.lr

    cos_inner = np.pi * (t % (T // M))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1

    lr = float(alpha_zero / 2 * cos_out)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print ("lr: ", lr)  


def adjust_cosine_learning_rate(optimizer, epoch, args):
    lr = args.lr
    lr_decay_rate = 0.1

    eta_min = lr * (lr_decay_rate ** 3)
    
    lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / args.epochs)) / 2


    print ("LR: ", lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_contrastive_learning_rate(args, optimizer, epoch):
    
    lr = args.contrastive_lr
    
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_contrastive_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    
    if args.warm and epoch <= args.warm_epochs:
        
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)

        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

def calc_confusion_mat(val_loader, model, args):
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cf = confusion_matrix(all_targets, all_preds).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt

    print('Class Accuracy : ')
    print(cls_acc)
    classes = [str(x) for x in args.cls_num_list]
    plot_confusion_matrix(all_targets, all_preds, classes)
    plt.savefig(os.path.join(args.root_log, args.store_name, 'confusion_matrix.png'))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def prepare_folders(args):
    
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(args, state, is_best):
    
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_model(model_dir, best_model_weights):

    model_states = {'state_dict_best': best_model_weights}

    model_dir = os.path.join(model_dir)

    torch.save(model_states, model_dir)


def load_model_simclr(model, model_dir=None):
   
    print('Loading model from %s' % (model_dir))
        
    checkpoint = torch.load(model_dir)          
    model_state = checkpoint['state_dict_best']

    model.load_state_dict(model_state)   

    return model

def load_model(model, head, model_dir=None, load_feature_model=False, load_head=False):
   
    print('Loading model from %s' % (model_dir))
        
    checkpoint = torch.load(model_dir)          
    model_state = checkpoint['state_dict_best']
    

    if load_feature_model:

        weights = model_state['feature_model']
        model.load_state_dict(weights)

    if load_head:

        weights = model_state['head']
        head.load_state_dict(weights)    

    return model, head