import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import hyptorch.pmath as pmath

import numpy as np



class HyperbolicKmeans:
    def __init__(self, args, k, iters):
        self.k = k
        self.iters = iters
        self.args = args
    def fit(self, embeddings):
        c = 1.0
        
        embeddings = pmath.expmap0(embeddings, c=c, clip_norm=self.args.max_clip_norm)
        

        self.centroids = torch.zeros((self.k, embeddings.shape[-1]))
        
        for i in range(self.k):
            self.centroids[i] = embeddings[i]
        
        for iter_ in range(self.iters):
            print (iter_)

            dists = pmath.dist_matrix(embeddings, self.centroids, c=c)
            
            self.labels = torch.argmin(dists, 1)
            
            for i in range(self.k):
                self.centroids[i] = pmath.poincare_mean(embeddings[self.labels==i], c=c)



class HyperbolicMLR(nn.Module):
    r"""
    Module which performs softmax classification
    in Hyperbolic space.
    """

    def __init__(self, ball_dim, n_classes, c, dataset, max_clip_norm, save_embedding, zero_bias=False, normalize_a_val=False):
        super(HyperbolicMLR, self).__init__()
        
        self.a_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.p_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))

        ########################################
        self.zero_bias = zero_bias
        self.normalize_a_val = normalize_a_val

        ########################################

        self.c = c
        self.n_classes = n_classes
        self.ball_dim = ball_dim
        self.reset_parameters()

        self.dataset = dataset
        self.max_clip_norm = max_clip_norm
        self.save_embedding = save_embedding


    def forward(self, x, save_dir=None, c=None, last_epoch=False):
        if c is None:
            c = torch.as_tensor(self.c).type_as(x)
        else:
            c = torch.as_tensor(c).type_as(x)
    
        if self.zero_bias:
            
            p_vals_poincare = pmath.expmap0(self.p_vals, c=c, clip_norm=15.0)

            p_vals_poincare = torch.zeros(p_vals_poincare.size()).cuda().detach()

        else:
            p_vals_poincare = pmath.expmap0(self.p_vals, c=c, clip_norm=15.0)
        
        '''
        if last_epoch and self.save_embedding:
            #print ("Save p_vals_poincare.")
            np.save(save_dir + '_p_vals.npy', p_vals_poincare.cpu().detach().numpy())
        '''


        conformal_factor = 1 - c * p_vals_poincare.pow(2).sum(dim=1, keepdim=True)
        a_vals_poincare = self.a_vals * conformal_factor
            

        '''
        if last_epoch and self.save_embedding:
            #print ("Save a_vals_poincare.")
            np.save(save_dir + '_a_vals.npy', a_vals_poincare.cpu().detach().numpy())
        '''

        logits = pmath._hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, c)
        return logits


    def extra_repr(self):
        return "Poincare ball dim={}, n_classes={}, c={}".format(
            self.ball_dim, self.n_classes, self.c
        )

    def reset_parameters(self):
        init.kaiming_uniform_(self.a_vals, a=math.sqrt(5))
        init.kaiming_uniform_(self.p_vals, a=math.sqrt(5))


class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, c, bias=True):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c=None):
        if c is None:
            c = self.c
        mv = pmath.mobius_matvec(self.weight, x, c=c)
        if self.bias is None:
            return pmath.project(mv, c=c)
        else:
            bias = pmath.expmap0(self.bias, c=c)
            return pmath.project(pmath.mobius_add(mv, bias), c=c)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, c={}".format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )


class ConcatPoincareLayer(nn.Module):
    def __init__(self, d1, d2, d_out, c):
        super(ConcatPoincareLayer, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.d_out = d_out

        self.l1 = HypLinear(d1, d_out, bias=False, c=c)
        self.l2 = HypLinear(d2, d_out, bias=False, c=c)
        self.c = c

    def forward(self, x1, x2, c=None):
        if c is None:
            c = self.c
        return pmath.mobius_add(self.l1(x1), self.l2(x2), c=c)

    def extra_repr(self):
        return "dims {} and {} ---> dim {}".format(self.d1, self.d2, self.d_out)


class HyperbolicDistanceLayer(nn.Module):
    def __init__(self, c):
        super(HyperbolicDistanceLayer, self).__init__()
        self.c = c

    def forward(self, x1, x2, c=None):
        if c is None:
            c = self.c
        return pmath.dist(x1, x2, c=c, keepdim=True)

    def extra_repr(self):
        return "c={}".format(self.c)


class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None, clip_norm=15.0, riemannian=True):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)


        self.train_c = train_c

        if self.train_c:
            self.log_c = nn.Parameter(torch.Tensor([math.log(c),]))

        else:
            self.log_c = nn.Parameter(torch.Tensor([math.log(c),])).detach()


        self.train_x = train_x
        self.clip_norm = clip_norm

        
        self.riemannian = pmath.RiemannianGradient


    def forward(self, x, train=None, feature_embedding=None):

        self.c = self.log_c.exp()
            
        ##################
        self.riemannian.c = self.c
        self.grad_fix = lambda x: self.riemannian.apply(x)
        ##################

        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return self.grad_fix(pmath.project(pmath.expmap(xp, x, c=self.c), c=self.c))
        
        Poincare_points = pmath.expmap0(x, c=self.c, clip_norm=self.clip_norm)

        return self.grad_fix(pmath.project(Poincare_points, c=self.c))
        #return pmath.project(Poincare_points, c=self.c)

    '''
    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)
    '''

class FromPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Poincare ball
    to n-dim Euclidean space
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None):

        super(FromPoincare, self).__init__()

        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c

        self.train_c = train_c
        self.train_x = train_x

    def forward(self, x):
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return pmath.logmap(xp, x, c=self.c)
        return pmath.logmap0(x, c=self.c)

    def extra_repr(self):
        return "train_c={}, train_x={}".format(self.train_c, self.train_x)