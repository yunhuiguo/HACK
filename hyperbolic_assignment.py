from __future__ import print_function, absolute_import 

import os
import argparse
import time
import pickle
import math
import copy
import numpy as np
from sklearn.metrics import confusion_matrix
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms.functional as F_t
from torch import Tensor

import hyptorch.nn as hypnn
import hyptorch.pmath as pmath

from utils import source_import, load_model, load_model_simclr, adjust_cosine_learning_rate, save_model, adjust_learning_rate_200_epochs, adjust_learning_rate_1000_epochs, AverageMeter

from data_utils import CongealingMNIST, IMBALANCEMNIST, IMBALANCEMNIST_plain, ThreeCropTransform, IMBALANCECIFAR10

from model import Particles, feature_Net, Projection_Head, Hyperbolic_Classifier, Euclidean_Classifier, resnet20



torch.autograd.set_detect_anomaly(True)

import matplotlib.pyplot as plt

import scipy
import scipy.optimize

from lapsolver import solve_dense


def calc_optimal_target_permutation(feats: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute the new target assignment that minimises the SSE between the mini-batch feature space and the targets.
    :param feats: the learnt features (given some input images)
    :param targets: the currently assigned targets.
    :return: the targets reassigned such that the SSE between features and targets is minimised for the batch.
    """
    # Compute cost matrix
    
    cost_matrix = np.zeros([feats.shape[0], targets.shape[0]])
        
        
    cost_matrix = pmath.pairwise_poincare_distances(torch.tensor(feats).cuda(), torch.tensor(targets).cuda())


    cost_matrix = cost_matrix.cpu().data.numpy()

    _, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    
    #_, col_ind = solve_dense(cost_matrix)

    # Permute the targets based on hungarian algorithm optimisation
    targets[range(feats.shape[0])] = targets[col_ind]
    
    return targets




def assigment_loss(features, targets):

    dist_ =  pmath.dist(features, targets).mean()
    return dist_


def train_Constrastive(args, train_loader, model, hyperbolic_projection, optimizer, epoch):
    """one epoch training"""

    criterion = SupConLoss().cuda()


    model.train()
    hyperbolic_projection.train()


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    

    all_targets  = []
    all_images   = []
    all_features = []


    for idx, (index, images, labels, nat)in enumerate(train_loader):
        
        cur_img = images[0]

        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1], images[2]], dim=0)
        
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        
        bsz = labels.shape[0]

        # compute loss

        normalized_fea, unnormalized_fea = model(images)
        
            
        f1, f2, f3 = torch.split(normalized_fea, [bsz, bsz, bsz], dim=0)

        un_f1, un_f2, un_f3 = torch.split(unnormalized_fea, [bsz, bsz, bsz], dim=0)


        contrastive_features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
            
        loss = criterion(contrastive_features)


        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        #####################################
        all_targets.extend(labels.cpu().detach().numpy())

        all_features.extend(un_f1.cpu().detach().numpy())

        all_images.extend(cur_img.cpu().detach().numpy())

        #####################################

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 1 == 0:
            
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

        #####################################


    np.save("./unsupervised_embeddings/label" + ".npy", all_targets)
    np.save("./unsupervised_embeddings/feature" + ".npy", all_features)
    np.save("./unsupervised_embeddings/images" + ".npy", all_images)


    return losses.avg



class Samples_Queue():

    """

    """
    def __init__(self):

        self.features = None
        self.targets = None

        self.images_0 = None
        self.images_1 = None
        self.images_2 = None

        self.index = None

    @torch.no_grad()
    def dist_to_target(self, args, images, features, index, targets):
        
 
        all_features   = features.cpu().detach().numpy()

        all_images_0   = images[0].cpu().detach().numpy()
        all_images_1   = images[1].cpu().detach().numpy()
        all_images_2   = images[2].cpu().detach().numpy()

        all_index    = index.cpu().detach().numpy()
        all_targets  = targets.cpu().detach().numpy()


        ########################################################
        dist_ =  -pmath.dist(torch.from_numpy(all_features), torch.from_numpy(all_targets))

        sorted_ = torch.argsort(dist_, dim=0)


        print (dist_[sorted_[:int(args.batch_size/2)]][0])


        self.targets    = all_targets[sorted_[:int(args.batch_size/2)]]
        self.images_0   = all_images_0[sorted_[:int(args.batch_size/2)]]
        self.images_1   = all_images_1[sorted_[:int(args.batch_size/2)]]
        self.images_2   = all_images_2[sorted_[:int(args.batch_size/2)]]

        self.index      = all_index[sorted_[:int(args.batch_size/2)]]

        ########################################################


def train_assignment(args, train_dataset, train_loader, model, hyperbolic_projection, device,  optimizer, epoch):
    """one epoch training"""

    model.train()
    hyperbolic_projection.train()


    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    losses = AverageMeter()
    aug_losses =  AverageMeter()


    end = time.time()
    

    all_targets = []
    all_images  = []
    all_features = []


    update_targets = bool(((epoch+1) % 2) == 0)


    #for idx, (index, images, labels, nat, congealing_index) in enumerate(train_loader):
    for idx, (index, images, labels, nat) in enumerate(train_loader):
        
        images_list = images
        

        '''
        ##########################################################
        if queue.index is not None:

            images_0 = torch.cat((torch.from_numpy(queue.images_0), images_list[0]), dim=0)
            images_1 = torch.cat((torch.from_numpy(queue.images_1), images_list[1]), dim=0)
            images_2 = torch.cat((torch.from_numpy(queue.images_2), images_list[2]), dim=0)

            index = torch.cat((torch.from_numpy(queue.index), index), dim=0)
            
            nat   = torch.cat((torch.from_numpy(queue.targets), nat))

            images_list = [images_0, images_1, images_2]
        '''

        ##########################################################
    
        data_time.update(time.time() - end)

        images = torch.cat([images_list[0], images_list[1], images_list[2]], dim=0)
        
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
            
        ##########################################################

        bsz = nat.size()[0]
        # compute loss

        normalized_fea, unnormalized_fea = model(images)
        
        features = hyperbolic_projection(unnormalized_fea)

        f1, f2, f3 = torch.split(features, [bsz, bsz, bsz], dim=0)
          
        ##########################################################
        
        if update_targets:

            print ("update targets")

            e_targets = nat.cpu().numpy()
            e_out     = f1.cpu().data.numpy()

            new_targets = calc_optimal_target_permutation(e_out, e_targets)
            
            # update.
            train_dataset.update_targets(index, new_targets)

            nat = torch.FloatTensor(new_targets)
    

        nat = nat.cuda()

        ##########################################################
        # store samples

        #queue.dist_to_target(args, images_list, f1, index, nat)

        ##########################################################

        if args.with_augmentation:

            aug_loss = assigment_loss(f2, f3) 


        assign_loss =  assigment_loss(f1, nat)
        

        # update metric
        losses.update(assign_loss.item(), bsz)

        if args.with_augmentation:

            aug_losses.update(aug_loss.item(), bsz)


        if args.with_augmentation:
        
            loss = assign_loss + aug_loss

        else:
        
            loss = assign_loss
        
        #####################################

        # SGD

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        #####################################

        '''
        #####################################
        all_targets.extend(labels.cpu().detach().numpy())
        
        all_features.extend(f1.cpu().detach().numpy())
    
        all_images.extend(cur_img.cpu().detach().numpy())

        #####################################
        '''

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 1 == 0:
            
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

        #####################################

    
    return train_dataset, losses.avg, aug_losses.avg



def save_embeddings(args, unorm, train_loader, model, hyperbolic_projection, device,  optimizer, epoch):

    model.eval()
    hyperbolic_projection.eval()


    end = time.time()
    
    all_targets = []
    all_images = []
    all_features = []

    all_congealing_index = []
    all_nats = []

    #for idx, (index, images, labels, nat, congealing_index) in enumerate(train_loader):
    for idx, (index, images, labels, nat) in enumerate(train_loader):
        
        e_targets = nat.numpy()

        cur_img = images[0]

        ##########################################################

        images = torch.cat([images[0], images[1], images[2]], dim=0)
        
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
        ##########################################################

        bsz = labels.shape[0]
        # compute loss

        normalized_fea, unnormalized_fea = model(images)
        
        features = hyperbolic_projection(unnormalized_fea)

        f1, f2, f3 = torch.split(features, [bsz, bsz, bsz], dim=0)
          
        #####################################
        all_targets.extend(labels.cpu().detach().numpy())
        
        all_features.extend(f1.cpu().detach().numpy())
        
        #all_congealing_index.extend(congealing_index.cpu().detach().numpy())

        unnorm_img = unorm(cur_img)
        #unnorm_img = cur_img

        all_images.extend(unnorm_img.cpu().detach().numpy())
            
        all_nats.extend(nat.cpu().detach().numpy())

        #####################################

    if args.with_augmentation:
        
        if args.across_batches:

            directory =  "./assignment_embeddings/across_batches/" + args.dataset + "/with_augmentation/"

        else:

            directory =  "./assignment_embeddings/" + args.dataset + "/with_augmentation/"
    
    else:
        
        directory =  "./assignment_embeddings/" + args.dataset + "/no_augmentation/"


    if not args.all_classes:

        directory = directory + str(args.class_idx) + "/"
    
    else:

        directory = directory + "/all_classes/"


    import os

    if not os.path.exists(directory):
        os.makedirs(directory)

    print (directory)

    #all_images = np.concatenate(all_images, axis=0)

    np.save(directory + "label_"   + str(epoch) + ".npy", all_targets)
    np.save(directory + "feature_" + str(epoch) + ".npy", all_features)
    np.save(directory + "images_"  + str(epoch) + ".npy", all_images)
    np.save(directory + "nats_"    + str(epoch) + ".npy", all_nats)
    #np.save(directory + "congealing_indicator_"    + str(epoch) + ".npy", all_congealing_index)


    return directory



def main():

    # Training settings
    parser = argparse.ArgumentParser(description="Sphere Packing")
    parser.add_argument('--dataset', default='mnist', help='dataset setting')

    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="num of classes",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=10000,
        metavar="N",
        help="input batch size for testing (default: 10000)",
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )


    parser.add_argument(
        "--seed", type=int, default=8888, metavar="S", help="random seed (default: 1)"
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )


    parser.add_argument(
        "--c", type=float, default=1.0, help="Curvature of the Poincare ball"
    )
    
    parser.add_argument(
        "--max_clip_norm", type=float, default=15.0, help="Max clip norm of the Euclidean embedding"
    )


    parser.add_argument("--use-hyperbolic", action="store_true", default=False)
        

    parser.add_argument(
        "--dim", type=int, default=2, help="Dimension of the Poincare ball"
    )

    parser.add_argument(
        "--model_dir", type=str, default="./saved_model_som/", help="directory for saving models"
    )

    parser.add_argument(
        "--save_dir", type=str, default="./saved_embedding/", help="directory for saving embeddings"
    )

    parser.add_argument(
        "--train_x",
        action="store_true",
        default=False,
        help="train the exponential map origin",
    )
    
    parser.add_argument(
        "--train_c",
        action="store_true",
        default=False,
        help="train the Poincare ball curvature",
    )

    parser.add_argument(
        "--save_embedding",
        action="store_true",
        default=False,
        help="save embedding or not",
    )


    parser.add_argument(
        "--num_of_points", type=int, default=1000, help="Number of points used in experiments"
    )

    #########################################

    parser.add_argument(
        "--k", type=float, default=0.5, help="c(k) in repulsion loss"
    )


    parser.add_argument(
        "--class_idx", type=int, default=0, help="c(k) in repulsion loss"
    )


    parser.add_argument("--across_batches", action="store_true", default=False)


    parser.add_argument("--with_augmentation", action="store_true", default=False)


    parser.add_argument("--all_classes", action="store_true", default=False)

    #########################################

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.deterministic=True


    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}


    ##################################################

    if args.dataset == "mnist" or args.dataset == "congealing_mnist":

        model = feature_Net(args).cuda()
        hyperbolic_projection = Projection_Head(args).cuda()

        mean = (0.1307,)
        std = (0.3081,)

        # Use MNIST
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


        '''
        val_transorm =  transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )
        '''
    
        val_transorm =  transforms.Compose(
                [transforms.ToTensor()]
            )


        unorm = transforms.Compose([ transforms.Normalize(mean = [ 0. ],
                                                     std = [ 1/0.3081]),

                                transforms.Normalize(mean = [ -0.1307],
                                                     std = [ 1]),
                               ])


    elif args.dataset == "cifar10":

        model = resnet20().cuda()

        hyperbolic_projection = Projection_Head(args, input_dim=10).cuda()


        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        normalize = transforms.Normalize(mean=mean, std=std)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])


        val_transorm =  transforms.Compose(
                [transforms.ToTensor(), normalize]
            )


        unorm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),

                                transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
        
        #######################################################


    #model                 = nn.DataParallel(model)
    #hyperbolic_projection = nn.DataParallel(hyperbolic_projection)
    


    #######################################################
    if not args.all_classes:

        if args.dataset == "cifar10":

            hyperbolic_embeddings = np.load("./particle_embedding/" + args.dataset + "/hyperbolic/1.55/class_0" + ".npy")
        
        else:

            hyperbolic_embeddings = np.load("./particle_embedding/" + "mnist" + "/hyperbolic/1.55/class_" + str(args.class_idx) + ".npy")

    else:

        hyperbolic_embeddings = np.load("./particle_embedding/hyperbolic/1.43/all_classes.npy")



    print ("shape of embeddings: ", hyperbolic_embeddings.shape)

    #######################################################

    if args.dataset == "mnist":

        train_dataset = IMBALANCEMNIST(args, root='../data', class_idx=args.class_idx, hyperbolic_embeddings=hyperbolic_embeddings, rand_number=args.rand_number, train=True, download=True, transform=ThreeCropTransform(train_transform, val_transorm))

    elif args.dataset == "congealing_mnist":

        train_dataset = CongealingMNIST(args=args, hyperbolic_embeddings=hyperbolic_embeddings, rand_number=args.rand_number, train=True, download=False, transform=ThreeCropTransform(train_transform, val_transorm))

    elif args.dataset == "cifar10":

        train_dataset = IMBALANCECIFAR10(args, root='../data', class_idx=args.class_idx, hyperbolic_embeddings=hyperbolic_embeddings, rand_number=args.rand_number, train=True, download=True, transform=ThreeCropTransform(train_transform, val_transorm))


    print ("size of train set: ", len(train_dataset))


    args.num_of_points = len(train_dataset)
    

    #######################################################
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=None,
        **kwargs
    )


    # print cls_num_list
    #######################################################
    if args.dataset != "congealing_mnist":
        cls_num_list = train_dataset.get_cls_num_list()
        print('cls num list: ', cls_num_list)
        args.cls_num_list = cls_num_list

    #######################################################

    model_optim_params_list = list(model.parameters()) + list(hyperbolic_projection.parameters())
 
    optimizer = optim.SGD(model_optim_params_list, lr=args.lr, momentum=0.0)

    ###################################################

    assign_loss_epochs = []
    aug_loss_epochs = []


    for epoch in range(1, args.epochs + 1):
        
        print ("Epoch: ", epoch, )

        adjust_cosine_learning_rate(optimizer, epoch, args)

        ###################################################
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            sampler=None,
            **kwargs
        )   

        if epoch == 1 or epoch % 5 == 0:
            
            print ("save_embeddings")
            directory = save_embeddings(args, unorm, train_loader, model, hyperbolic_projection, device, optimizer, epoch)

    
        ###################################################
        train_dataset, ave_loss, ave_aug_loss = train_assignment(args, train_dataset, train_loader, model, hyperbolic_projection, device,  optimizer, epoch)


        assign_loss_epochs.append(ave_loss)
        aug_loss_epochs.append(ave_aug_loss)

        ###################################################



    print (directory)
    np.save(directory + "per_epoch_assign_loss.npy", assign_loss_epochs) 

    if args.with_augmentation:

        np.save(directory + "per_epoch_aug_loss.npy", aug_loss_epochs) 


if __name__ == "__main__":
    main()
