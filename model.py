
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import hyptorch.nn as hypnn
import hyptorch.pmath as pmath


class Particles(nn.Module):
    
    def __init__(self, args):
        super(Particles, self).__init__()

        self.embedding = nn.Embedding(args.num_of_points, args.dim)

        self.embedding.weight.data.uniform_(-0.01, 0.01)

    def forward(self, idx):

        return self.embedding(idx)


def init_models(args, model_dir, load_feature_model=False, load_classifer=False, fix_feature_model=False):
    networks = {}
    model_optim_params_list = []
    
    feature_model = feature_Net(args)

    head = Projection_Head(args)

    networks["feature_model"] = feature_model
    networks["head"]    = head

    if load_feature_model or load_classifer:
        load_model(networks, model_dir, load_feature_model, load_classifer)

    if fix_feature_model:

        for param_name, param in networks["feature_model"].named_parameters():
            param.requires_grad = False

    return networks



class feature_Net(nn.Module):

    def __init__(self, args):
        super(feature_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 100)
            
        self.fc2 = nn.Linear(100, 100)

        self.fc3 = nn.Linear(100, 64)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        feat = F.normalize(x, dim=1)

        return feat, x


class Projection_Head(nn.Module):
    def __init__(self, opt, input_dim=64):

        super(Projection_Head, self).__init__()
            
        self.opt = opt

        self.project = nn.Linear(input_dim, opt.dim)

        self.tp = hypnn.ToPoincare(c=opt.c, train_x=opt.train_x, train_c=opt.train_c, ball_dim=opt.dim, clip_norm=opt.max_clip_norm)


    def forward(self, x):

        if self.opt.dim == 2:

            x = self.project(x)


        x = self.tp(x)

        return x



class Euclidean_Classifier(nn.Module):
    def __init__(self, args):

        super(Euclidean_Classifier, self).__init__()


        self.mlr =  nn.Linear(64, args.num_classes)

    def forward(self, x):

        x = self.mlr(x)

        return x


class Hyperbolic_Classifier(nn.Module):
    def __init__(self, args):

        super(Hyperbolic_Classifier, self).__init__()

        self.curvature = args.c
        self.dim = args.dim
        self.train_x = args.train_x
        self.train_c = args.train_c

        self.mlr = hypnn.HyperbolicMLR(ball_dim=self.dim, n_classes=args.num_classes, c=self.curvature, dataset=args.dataset, max_clip_norm=args.max_clip_norm, save_embedding=args.save_embedding)


    def forward(self, x):

        x = self.mlr(x)

        return x


__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False):
        super(ResNet_s, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if use_norm:
            self.linear = NormedLinear(64, num_classes)
        else:
            self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        feat = F.normalize(out, dim=1)


        return feat, out


def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])


def resnet32(num_classes=10, use_norm=False):
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])

