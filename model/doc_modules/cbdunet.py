import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.fcn(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.inc = nn.Sequential(
            single_conv(9, 64),
            # single_conv(3, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, 3)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fcn = FCN()
       
        
        self.unet1 = UNet()
    
    def forward(self, x):
        
        
        x_in = x
        noise_level = self.fcn(x_in)
        out = self.unet1(torch.cat([x_in, noise_level], dim=1))
       

        return out + x
    
    
class NetworkT(nn.Module):
    def __init__(self):
        super(NetworkT, self).__init__()
        
        
        self.unet2 = UNet()
        self.acf = nn.Hardtanh(-1,1)
    
    def forward(self, x, mp1, score):
        
        
        x_in = x
       
        mp2 = self.unet2(torch.cat([x_in,mp1,score], dim=1))
        
        out = mp2 + mp1
        out = out.tanh()
        return out + x, out
       
  


class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):
        l2_loss = F.mse_loss(out_image, gt_image)

        asym_loss = torch.mean(if_asym * torch.abs(0.3 - torch.lt(gt_noise, est_noise).float()) * torch.pow(est_noise - gt_noise, 2))

        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, : ,1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x-1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        loss = l2_loss +  0.5 * asym_loss + 0.05 * tvloss

        return loss

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
    
    
class ResidualBlock(nn.Module):
    '''Residual block with residual connections
    ---Conv-IN-ReLU-Conv-IN-x-+-
     |________________________|
    '''
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        # nn.ReLU(inplace=True),
                        nn.Tanh(),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class ConGenerator_S2F(nn.Module):
    '''Coarse Deshadow Network
    '''
    def __init__(self,num,init_weights=False):
        super(ConGenerator_S2F, self).__init__()
        # Initial convolution block
        self.conv1_0 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(num, 32, 7))
        self.conv1_1 = nn.Sequential(ResidualBlock(32))
        self.conv1_2 = nn.Sequential(ResidualBlock(32))
        self.pool1 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode='reflect'))
        self.conv2_1 = nn.Sequential(ResidualBlock(64))
        self.conv2_2 = nn.Sequential(ResidualBlock(64))
        self.pool2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode='reflect'))
        self.conv3_1 = nn.Sequential(ResidualBlock(128))
        self.conv3_2 = nn.Sequential(ResidualBlock(128))
        self.conv3_3 = nn.Sequential(ResidualBlock(128))
        self.conv3_4 = nn.Sequential(ResidualBlock(128))
        self.conv3_5 = nn.Sequential(ResidualBlock(128))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1))
        self.conv4_1 = nn.Sequential(ResidualBlock(64))
        self.conv4_2 = nn.Sequential(ResidualBlock(64))
        self.up5 = nn.Sequential(nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1))
        self.conv5_1 = nn.Sequential(ResidualBlock(32))
        self.conv5_2 = nn.Sequential(ResidualBlock(32))
        self.conv5_3 = nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(32, 3, 7))
        
        self.acf = nn.Hardtanh(-1,1)
        
        if init_weights:
            self.apply(weights_init_normal)
    
    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = ConGenerator_S2F(init_weights=True)
        return model



    
    def forward(self,xin):
        x = self.conv1_0(xin)
        
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv3_5(x)
        x = self.up4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.up5(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = x.tanh()
       
        xout = x + xin
       
        return xout, x
    
    
class ConGenerator_S2FF(nn.Module):
    '''Coarse Deshadow Network
    '''
    def __init__(self,num,init_weights=False):
        super(ConGenerator_S2FF, self).__init__()
        # Initial convolution block
        self.conv1_0 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(num, 32, 7))
        self.conv1_1 = nn.Sequential(ResidualBlock(32))
        self.conv1_2 = nn.Sequential(ResidualBlock(32))
        self.pool1 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode='reflect'))
        self.conv2_1 = nn.Sequential(ResidualBlock(64))
        self.conv2_2 = nn.Sequential(ResidualBlock(64))
        self.pool2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode='reflect'))
        self.conv3_1 = nn.Sequential(ResidualBlock(128))
        self.conv3_2 = nn.Sequential(ResidualBlock(128))
        self.conv3_3 = nn.Sequential(ResidualBlock(128))
        self.conv3_4 = nn.Sequential(ResidualBlock(128))
        self.conv3_5 = nn.Sequential(ResidualBlock(128))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1))
        self.conv4_1 = nn.Sequential(ResidualBlock(64))
        self.conv4_2 = nn.Sequential(ResidualBlock(64))
        self.up5 = nn.Sequential(nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1))
        self.conv5_1 = nn.Sequential(ResidualBlock(32))
        self.conv5_2 = nn.Sequential(ResidualBlock(32))
        self.conv5_3 = nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(32, 3, 7))
        
        self.acf = nn.Hardtanh(-1,1)
        
        if init_weights:
            self.apply(weights_init_normal)
    
    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = ConGenerator_S2F(init_weights=True)
        return model


    
    
    def forward(self,xin,mp,score):
        # x = self.conv1_0(xin)
        x = self.conv1_0(torch.cat((xin,mp,score),dim=1))
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv3_5(x)
        x = self.up4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.up5(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        
        x = x.tanh()
        
        xout = x + xin
        
        return xout, x
class Condition(nn.Module):
    ''' Compute the region style of non-shadow regions'''

    def __init__(self, in_nc=3, nf=128):
        super(Condition, self).__init__()
        stride = 1
        pad = 0
        self.conv1 = nn.Conv2d(in_nc, nf//4, 1, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf//4, nf//2, 1, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf//2, nf, 1, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.act(self.conv3(out))

        mask = F.interpolate(mask.detach(), size=out.size()[2:], mode='nearest')  ## use the dilated mask to get the condition  
        zero = torch.zeros_like(mask)
        one = torch.ones_like(mask)
        mask = torch.where(mask >= 1.0, one, zero)
        cond = out*(1.0-mask)
        cond = torch.mean(cond, dim=[2, 3], keepdim=False)
        
        return cond

class RN(nn.Module):
    '''Compute the region normalization within the foreground-background region
    '''
    def __init__(self, dims_in, eps=1e-5):
        super(RN, self).__init__()
        self.eps = eps

    def forward(self, x, mask):
        mean_back, std_back = self.get_foreground_mean_std(x * (1-mask), 1 - mask) # the background features
        normalized = (x - mean_back) / std_back
        normalized_background = normalized * (1 - mask)

        mean_fore, std_fore = self.get_foreground_mean_std(x * mask, mask) # the background features
        normalized = (x - mean_fore) / std_fore 
        normalized_foreground = normalized * mask

        return normalized_foreground + normalized_background

    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])     # (B, C)
        num = torch.sum(mask, dim=[2, 3])       # (B, C)
        mu = sum / (num + self.eps)
        mean = mu[:, :, None, None]
        var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + self.eps)
        var = var[:, :, None, None]
        return mean, torch.sqrt(var+self.eps)
        
class SINLayer(nn.Module):
    '''Compute the spatial region denormalization within the foreground-background region
    '''
    def __init__(self, dims_in=256):
         super(SINLayer, self).__init__() 
         self.gamma_conv0 = nn.Conv2d(dims_in+1, dims_in//2, 1)
         self.gamma_conv1 = nn.Conv2d(dims_in//2, dims_in, 1)
         self.gamma_conv2 = nn.Conv2d(dims_in, dims_in, 1)
         self.bate_conv0 = nn.Conv2d(dims_in+1, dims_in//2, 1)
         self.bate_conv1 = nn.Conv2d(dims_in//2, dims_in, 1)
         self.bate_conv2 = nn.Conv2d(dims_in, dims_in, 1)
    def forward(self, x, cond_f, mask): 
        m_cond_f = torch.cat((mask * cond_f, mask*2.0-1.0), dim=1)
        gamma = self.gamma_conv2(self.gamma_conv1(F.leaky_relu(self.gamma_conv0(m_cond_f), 0.2, inplace=True)))
        beta = self.bate_conv2(self.bate_conv1(F.leaky_relu(self.bate_conv0(m_cond_f), 0.2, inplace=True)))

        return x * (gamma) + beta

class ResidualBlock_SIN(nn.Module):
    '''Residual block with spatially region-aware prototypical normalization
    ---Conv-SRPNorm-ReLU-Conv-SRPNorm-x-+-
     |__________________________________|
    '''

    def __init__(self, in_features=256, cond_dim=128):
        super(ResidualBlock_SIN, self).__init__()
        self.conv0 = nn.Sequential(nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3))
        self.local_scale0 = nn.Sequential(
            nn.Linear(cond_dim, in_features//16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features, bias=False)
            )
        self.local_shift0 = nn.Sequential(
            nn.Linear(cond_dim, in_features//16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features, bias=False)
            )

        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3))
        self.local_scale1 = nn.Sequential(
            nn.Linear(cond_dim, in_features//16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features, bias=False)
            )
        self.local_shift1 = nn.Sequential(
            nn.Linear(cond_dim, in_features//16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features, bias=False)
            )
        self.in_features = in_features
        self.act = nn.ReLU(inplace=True)
        self.RN0 = RN(in_features)
        self.RN1 = RN(in_features)
        self.SIN0 = SINLayer(in_features)
        self.SIN1 = SINLayer(in_features)

    def forward(self, x):
        identity = x[0]
        cond = x[1]
        mask = x[2]
        mask = F.interpolate(mask.detach(), size=identity.size()[2:], mode='nearest')

        local_scale_0 = self.local_scale0(cond)
        local_scale_1 = self.local_scale1(cond)
        local_shift_0 = self.local_shift0(cond)
        local_shift_1 = self.local_shift1(cond)
        # - Conv -SRPNorm - Relu
        out = self.conv0(identity)        
        out = self.RN0(out, mask) # with no extra params
        cond_f0 = out * (local_scale_0.view(-1, self.in_features, 1, 1)) + local_shift_0.view(-1, self.in_features, 1, 1)
        out = self.SIN0(out, cond_f0, mask)
        out = self.act(out)
        # - Conv -SRPNorm 
        out = self.conv1(out)
        out = self.RN1(out, mask)
        cond_f1 = out * (local_scale_1.view(-1, self.in_features, 1, 1)) + local_shift_1.view(-1, self.in_features, 1, 1)
        out = self.SIN1(out, cond_f1, mask)
        #  shortcut
        out += identity

        return out, cond
    
    
    
    
class ConRefineNet(nn.Module):
    '''Coarse Deshadow Network
    '''
    def __init__(self,init_weights=False):
        super(ConRefineNet, self).__init__()
        # Initial convolution block
        self.conv1_0 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(6, 32, 7))
        self.conv1_1 = nn.Sequential(ResidualBlock(32))
        self.conv1_2 = nn.Sequential(ResidualBlock(32))
        self.pool1 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode='reflect'))
        self.conv2_1 = nn.Sequential(ResidualBlock(64))
        self.conv2_2 = nn.Sequential(ResidualBlock(64))
        self.pool2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode='reflect'))
        self.conv3_1 = nn.Sequential(ResidualBlock(128))
        self.conv3_2 = nn.Sequential(ResidualBlock(128))
        self.conv3_3 = nn.Sequential(ResidualBlock(128))
        self.conv3_4 = nn.Sequential(ResidualBlock(128))
        self.conv3_5 = nn.Sequential(ResidualBlock(128))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1))
        self.conv4_1 = nn.Sequential(ResidualBlock(64))
        self.conv4_2 = nn.Sequential(ResidualBlock(64))
        self.up5 = nn.Sequential(nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1))
        self.conv5_1 = nn.Sequential(ResidualBlock(32))
        self.conv5_2 = nn.Sequential(ResidualBlock(32))
        self.conv5_3 = nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(32, 3, 7))
        
        if init_weights:
            self.apply(weights_init_normal)
    
    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = ConGenerator_S2F(init_weights=True)
        return model


    def forward(self,xin,noise):
        x = self.conv1_0(torch.cat((xin,noise), dim=1))
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv3_5(x)
        x = self.up4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.up5(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        xout = x + xin
        return xout, x
    
    
    
    
class ShadowRemove(nn.Module):
    
    def __init__(self,num_pre,num_pos):
        super(ShadowRemove,self).__init__()
        self.pre = ConGenerator_S2F(num_pre)
        self.pos = ConGenerator_S2FF(num_pos)
       
        
    def forward(self,xin,score):
        xout1,mp1 = self.pre(xin)
        xout2,out = self.pos(xin,mp1,score)
        
        
        return xout1,xout2,mp1,out
        
    
   
        
