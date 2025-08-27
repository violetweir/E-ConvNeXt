from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .convnext import ClasHead, _load_pretrained
import paddle 
#import paddle.fluid as fluid
import paddle.nn.functional as F
import paddle.nn as nn
from paddle import ParamAttr
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url


trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)

MODEL_URLS = {
    "ConvNext_tiny":
    "https://passl.bj.bcebos.com/models/convnext_tiny_1k_224.pdparams",
    "ConvNext_small":
    "https://passl.bj.bcebos.com/models/convnext_small_1k_224.pdparams",
}

__all__ = list(MODEL_URLS.keys())


class Identity(nn.Layer):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class DropBlock(nn.Layer):
    def __init__(self, block_size, keep_prob, name=None, data_format='NCHW'):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890
        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size**2)
            if self.data_format == 'NCHW':
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = paddle.cast(paddle.rand(x.shape) < gamma, x.dtype)
            mask_inv = F.max_pool2d(
                matrix,
                self.block_size,
                stride=1,
                padding=self.block_size // 2,
                data_format=self.data_format)
            mask = 1. - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y


class Block(nn.Layer):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, kernel_size=7, padding=3,
                                groups=dim)  # depthwise conv
        self.norm =nn.BatchNorm2D(dim)
        self.pwconv1 = nn.Conv2D(
            dim, 4 * dim, 1)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2D(4 * dim, dim, 1)
        self.ese = EffectiveSELayer(dim, dim)
        self.norm2 =nn.BatchNorm2D(dim)
        self.gamma =  paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(
                value=1.0)
        ) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # x = x.transpose([0, 2, 3, 1])  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        # x = self.ese(x)
        # x = self.norm2(x)
        x = self.pwconv2(x)
        x = self.norm2(x)
        x = self.ese(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        # x = x.transpose([0, 3, 1, 2])  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.gamma * x
        return x
        
class Block2(nn.Layer):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, 2*dim, kernel_size=3, padding=1,)  # depthwise conv
        self.norm =nn.BatchNorm2D(42*dim)
        # self.pwconv1 = nn.Conv2D(
        #     dim*4,dim, 1)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2D(2 * dim, dim, 1)
        self.ese = EffectiveSELayer(2*dim, 2*dim)
        # self.norm2 =nn.BatchNorm2D(dim)
        self.gamma =  paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(
                value=1.0)
        ) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # x = x.transpose([0, 2, 3, 1])  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.ese(x)
        x = self.act(x)
        # x = self.ese(x)
        # x = self.norm2(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        # x = x.transpose([0, 3, 1, 2])  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.gamma * x
        return x



class LayerNorm(nn.Layer):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 epsilon=1e-6,
                 data_format="channels_last"):
        super().__init__()

        self.weight = paddle.create_parameter(shape=[normalized_shape],
                                              dtype='float32',
                                              default_initializer=ones_)

        self.bias = paddle.create_parameter(shape=[normalized_shape],
                                            dtype='float32',
                                            default_initializer=zeros_)

        self.epsilon = epsilon
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.epsilon)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / paddle.sqrt(s + self.epsilon)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class L2Decay(paddle.regularizer.L2Decay):
    def __init__(self, coeff=0.0):
        super(L2Decay, self).__init__(coeff)

class EffectiveSELayer(nn.Layer):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2D(channels, channels, kernel_size=1, padding=0)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)



class ConvBNLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups)

        self.bn = nn.BatchNorm2D(
            ch_out,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class CSPStage(nn.Layer):
    def __init__(self,
                block_fn,
                ch_in,
                ch_out,
                n,
                stride,
                p_rates,
                layer_scale_init_value=1e-6,
                act=nn.GELU,
                attn='eca',
                block_former = 1):
        super().__init__()
        ch_mid = (ch_in+ch_out)//2
        if stride == 2:
            self.down = nn.Sequential(ConvBNLayer(ch_in, ch_mid , 2, stride=2,  act=act))
        else:
            self.down = nn.Sequential(
                nn.Conv2D(ch_in, ch_mid, kernel_size=3, stride=1,padding=1),      
                LayerNorm(ch_mid, epsilon=1e-6, data_format="channels_first"),
                nn.GELU()
            )
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2, drop_path=p_rates[i],layer_scale_init_value=layer_scale_init_value)
            for i in range(n)
        ])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = paddle.concat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y

class CSPConvNext(nn.Layer):
    def __init__(
        self,
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[64,128,256,512,1024],
        drop_path_rate=0.,
        layer_scale_init_value=1e-6,
        stride=[2,2,2,2],
        return_idx=[1,2,3],
        depth_mult = 1.0,
        width_mult = 1.0,
    ):
        super().__init__()
        block_former = [Block2,Block,Block,Block]
        depths = [int(i*depth_mult) for i in depths]
        dims = [int(i*width_mult)  for i in dims]
        act = nn.GELU()
        self.Down_Conv = nn.Sequential(
                ('conv1', ConvBNLayer(
                    3, dims[0]//2 , 2, stride=2,  act=act)),
                ('conv2', ConvBNLayer(
                    dims[0]//2, dims[0]//2 , 3, stride=1,padding=1,  act=act)),
                ('conv3', ConvBNLayer(
                    dims[0]//2, dims[0] , 3, stride=1,padding=1, act=act)),
)
        dp_rates = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]
        n = len(depths)

        self.stages = nn.Sequential(*[(str(i), CSPStage(
            block_former[i], dims[i], dims[i + 1], depths[i], stride[i], dp_rates[sum(depths[:i]) : sum(depths[:i+1])],act=nn.GELU))
                                      for i in range(n)])
        self.norm = nn.BatchNorm(dims[-1])

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            try:
                trunc_normal_(m.weight)
                zeros_(m.bias)
            except:
                print(m)
    
    
    def forward(self, inputs):
        x = inputs
        x = self.Down_Conv(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            # if idx in self.return_idx:
            #     outs.append(x)
        return self.norm(x.mean([-2, -1]))

class Model(paddle.nn.Layer):
    def __init__(self, class_num=1000,):
        super().__init__()
        self.backbone = CSPConvNext()
        self.head = ClasHead(with_avg_pool=False, in_channels=1024, num_classes=class_num)

    def forward(self, x):

        x = self.backbone(x)
        x = self.head(x)
        return x

def CSPConvNext_tiny(pretrained=False, use_ssld=False, **kwargs):
    model = Model( **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNext_tiny"], use_ssld=use_ssld)
    return model

if __name__=="__main__":
     model  = CSPConvNext()
     # Total Flops: 1189500624     Total Params: 8688640
     GFlops = paddle.flops(model,(1,3,224,224),print_detail=True)



