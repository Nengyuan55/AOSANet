import torch.nn as nn
import torch.nn.functional as F
import torch

class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 9 + 3, [64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 32, 128 + 3, [128, 256], False)
        self.sa3 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [512, 1024], False)
        #self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp3 = PointNetFeaturePropagation(1280, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 128])
        # self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        self.mfe1 = mfe()
        self.mfe2 = mfe()
        self.mfe3 = mfe()


    def forward(self, xyz,trans=None):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) #2,3,1024  2,128,1024
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) #2,3,256   2,256,256
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) #2,3,128   2,1024,128
        # famodel
        l1_points_max = self.mfe1(l1_points)  # 2,1,1024
        l1_points = l1_points * l1_points_max  # 2,128,1024
        l2_points_max = self.mfe2(l2_points)  # 2,1,256
        l2_points = l2_points * l2_points_max  # 2,256,256
        l3_points_max = self.mfe3(l3_points)  # 2,1,128
        l3_points = l3_points * l3_points_max  # 2,1024,128


        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)#2,256,256
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)#2,128,1024
        #l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  #2,128,4096

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


class mfe(nn.Module):
    def __init__(self):
        super(mfe, self).__init__()

    def forward(self, x):
        B,C,N=x.shape
        x=torch.max(x,dim=1,keepdim=True)[0]
        return x




class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.attention1 = AOSA(mlp[1])
    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))     #new_points=2,256,128
        new_points=self.attention1(new_points)
        return new_points   #2,256,128




class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.inchannel=in_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.act=nn.LeakyReLU(0.2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel , mlp[0], 1, bias=False), nn.BatchNorm2d(mlp[0]), self.act,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mlp[0], mlp[1], 1, bias=False), nn.BatchNorm2d(mlp[1]), self.act,
        )




        self.attention1 = AOSA(mlp[1])





        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points  = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points0 = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]   #2,9,32,512
        mean = torch.mean(new_points0, dim=1,keepdim=True)  # 2,1,32,512
        #new_points=2,6,32,512


        new_points1=self.conv1(new_points0) #2,64,32,512
        new_points1 = self.conv2(new_points1)  # 2,64,32,512
        new_points1 = torch.max(new_points1, 2)[0] #2,128,512
        mean = torch.max(mean, 2)[0]  # 2,1,512

        new_points1 = self.attention1(new_points1)  #2,128,512

        new_points = (new_points1 - mean)  #2,128,512




        #new_points=new_points+original
        #new_points=torch.cat((new_points1,new_points2),dim=1)

        #new_points = torch.max(new_points0, 2)[0]   #new_points=2,128,512
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points



class AOSA(nn.Module):
    def __init__(self, channels):
        super(AOSA,self).__init__()

        self.q_conv = nn.Conv1d(channels, channels , 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels , 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.affine_alpha = nn.Parameter(torch.ones([1, channels,1]))
        self.affine_beta = nn.Parameter(torch.zeros([1, channels,1]))
    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  #  b, n, c 2,256,64
        x_k = self.k_conv(x)  # b, c, n  2,64,256
        x_v = self.v_conv(x)  # b,n,n 2,256,256
        #@代表数学上的相乘
        energy = x_q @ x_k  # b, n, n 2,256,256
        attention = self.softmax(energy) #2,256,256
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True)) #2,256,256  #l1_norem操作，得到attention_map
        x_r = x_v @ attention  # b, c, n 2,256,256
        x_z=self.affine_alpha * self.trans_conv(x - x_r) + self.affine_beta
        x_r = self.act(self.after_norm(x_z))  #2,256,256
        # 进行LBR       ReLu层           BN层                liner层
        x = x + x_r  #2,256,256  相加
        return x



def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx



def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # import ipdb; ipdb.set_trace()
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = torch.zeros(B, dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist





class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    print(model(xyz))