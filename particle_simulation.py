from __future__ import print_function
import os
import argparse
import time
import pickle
import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms.functional as F_t

import hyptorch.nn as hypnn
import hyptorch.pmath as pmath
from utils import source_import, load_model, save_model, adjust_learning_rate_200_epochs, adjust_cosine_learning_rate

from losses import  augmentation_loss,  hyperbolic_uniform_loss, repulsion_loss, boundary_loss

from model import Particles



torch.autograd.set_detect_anomaly(True)


from data_utils import IMBALANCEMNIST, IMBALANCEMNIST_plain, ThreeCropTransform, IMBALANCECIFAR10_plain


def train_particle_simulation(args, network, device, optimizer, epoch, r_per_point, boundary_r, loss_epochs):
    
    network.train()

    all_Poincare_embedding = []

    if args.use_hyperbolic_dist:

        tp = hypnn.ToPoincare(c=args.c, train_x=args.train_x, train_c=args.train_c, ball_dim=args.dim, clip_norm=args.max_clip_norm)

    #################################################################
    idx = torch.tensor(list(range(args.num_of_points)))


    f1 = network(idx)

    loss_1 =  boundary_loss(args, f1, boundary_r, r_per_point)

    #################################################################
    
    print ("boundary loss: ", loss_1)

    if args.use_euclidean_dist:

        loss_2 =  Euclidean_uniform_loss(args, f1, r_per_point)

    elif args.use_hyperbolic_dist:
        
        loss_2 =  hyperbolic_uniform_loss(args, tp(f1), r_per_point)

    #################################################################


    print ("uniform loss: ", loss_2)

    loss = args.boundary_loss_weight * loss_1 + loss_2

    loss_epochs.append(loss.item())


    print ("total loss: ", loss)
   
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #################################################################
    if args.use_hyperbolic_dist:
        
        f1 = tp(f1)

    all_Poincare_embedding.extend(f1.cpu().detach().numpy())
    

    if not args.all_classes:

        if args.use_hyperbolic_dist:

            np.save( args.save_dir + 'class_' + str(args.class_idx) + '.npy', all_Poincare_embedding)
        
        else:

            np.save( args.save_dir + 'class_' + str(args.class_idx) + '.npy', all_Poincare_embedding)

    else:

        np.save( args.save_dir+ 'all_classes.npy', all_Poincare_embedding)


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
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--c", type=float, default=1.0, help="Curvature of the Poincare ball"
    )
    
    parser.add_argument(
        "--max_clip_norm", type=float, default=15.0, help="Max clip norm of the Euclidean embedding"
    )


    parser.add_argument(
        "--class_idx", type=int, default=0, help="c(k) in repulsion loss"
    )



    parser.add_argument("--use-hyperbolic", action="store_true", default=False)
        

    parser.add_argument(
        "--save_dir", type=str, default="./", help="directory for saving embeddings"
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
        "--num_of_points", type=int, default=0, help="Number of points used in experiments"
    )

    parser.add_argument(
        "--dim", type=int, default=2, help="Dimension of the Poincare ball"
    )

    #########################################

    parser.add_argument(
        "--k", type=float, default=0.5, help="c(k) in repulsion loss"
    )

    ##########################################
    parser.add_argument("--particle_simulation", action="store_true", default=False)
    parser.add_argument("--use_euclidean_dist", action="store_true", default=False)
    parser.add_argument("--use_hyperbolic_dist", action="store_true", default=False)
    ##########################################


    parser.add_argument("--use_nonlinear_repulsion_loss", action="store_true", default=False)
    parser.add_argument("--use_linear_repulsion_loss", action="store_true", default=False)

    parser.add_argument("--use_boundary_loss", action="store_true", default=False)

    parser.add_argument(
        "--boundary_loss_weight", type=float, default=100.0, help=""
    )

    parser.add_argument("--use_uniform_loss", action="store_true", default=False)


    parser.add_argument("--all_classes", action="store_true", default=False)

    #########################################
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    
    #########################################

    #######################################################


    if not args.all_classes:

        if args.dataset == "mnist":


            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=28),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])


            val_transorm =  transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                )

            train_dataset = IMBALANCEMNIST_plain(root='../data', class_idx=args.class_idx, rand_number=args.rand_number, train=True, download=True, transform=ThreeCropTransform(train_transform, val_transorm))

            print ("size of train set: ", len(train_dataset))

            args.num_of_points = len(train_dataset)


        elif args.dataset == "cifar10":

            train_dataset = IMBALANCECIFAR10_plain(root='../data', class_idx=args.class_idx, rand_number=args.rand_number, train=True, download=True)

            print ("size of train set: ", len(train_dataset))


            args.num_of_points = len(train_dataset)

    else:

        args.num_of_points = 200


    print ("num_of_points: ", args.num_of_points)
    #######################################################

    # Compute the radius for each point
    Euclidean_r = math.tanh(math.sqrt(args.c) * args.max_clip_norm)
    
    print ("Euclidean_r: ", Euclidean_r)

    if args.use_hyperbolic_dist:

        Hyperbolic_r = pmath.Eculidean_to_Hyperbolic(Euclidean_r, args.c)

        print ("Hyperbolic_r: ", Hyperbolic_r)
        ###################################

        all_area = pmath.h_area(Hyperbolic_r, args.c)

        print ("All Hyperbolic area: ", all_area)

        # area_per_point
        area_per_point =  all_area / args.num_of_points
            
        # radius for the ball
        r_per_point = pmath.area_to_r(area_per_point, args.c)

        print ("hyperbolic_r_per_point: ",  r_per_point)

        print ("hyperbolic_area_per_point: ", area_per_point)

        recomputed_r = pmath.h_area( r_per_point, args.c)
        print ("recomputed_hyperbolic_r: ", recomputed_r)
        ###################################


    elif  args.use_euclidean_dist:

        all_area = math.pi * Euclidean_r * Euclidean_r

        area_per_point =  all_area / args.num_of_points

        r_per_point = math.sqrt(area_per_point / math.pi)

        print ("Euclidean_r_per_point: ",  r_per_point)


    #####################################################
    args.Euclidean_r = Euclidean_r

    args.save_dir = args.save_dir + "particle_embedding/" 


    args.save_dir = args.save_dir + args.dataset + "/"

    if args.use_euclidean_dist:

        args.save_dir  = args.save_dir  + "euclidean/"

    elif args.use_hyperbolic_dist:

        args.save_dir  = args.save_dir  + "hyperbolic/"
 
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)



    ##################################################

    network = Particles(args)

    model_optim_params_list = []

    model_optim_params_list += list(filter(lambda p: p.requires_grad, network.parameters()))

    ##################################################

    optimizer = optim.SGD(model_optim_params_list, lr=args.lr, momentum=0.0)
    
    ##################################################

    loss_epochs = []

    final_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        
        print ("Epoch: ", epoch, )
        ##################################################
        
        #adjust_learning_rate_200_epochs(optimizer, epoch, args)
        adjust_cosine_learning_rate(optimizer, epoch, args)

        train_particle_simulation(args, network, device, optimizer, epoch, r_per_point, args.Euclidean_r, loss_epochs)

        print ("\n")
        ##################################################

if __name__ == "__main__":
    main()
