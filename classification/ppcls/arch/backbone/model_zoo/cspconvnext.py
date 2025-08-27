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


# class Identity(nn.Layer):

#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x

# def drop_path(x, drop_prob=0., training=False):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
#     shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
#     random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
#     random_tensor = paddle.floor(random_tensor)  # binarize
#     output = x.divide(keep_prob) * random_tensor
#     return output


# class DropPath(nn.Layer):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """

#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)


# class Block(nn.Layer):
#     """ ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """

#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2D(dim, dim, kernel_size=7, padding=3,
#                                 groups=dim)  # depthwise conv
#         self.norm =nn.BatchNorm2D(dim)
#         self.pwconv1 = nn.Conv2D(
#             dim, 4 * dim, 1)  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Conv2D(4 * dim, dim, 1)
#         self.ese = EffectiveSELayer(dim, dim)
#         self.norm2 =nn.BatchNorm2D(dim)
#         self.gamma =  paddle.create_parameter(
#             shape=[1],
#             dtype='float32',
#             default_initializer=nn.initializer.Constant(
#                 value=1.0)
#         ) if layer_scale_init_value > 0 else None


#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         # x = x.transpose([0, 2, 3, 1])  # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         # x = self.ese(x)
#         # x = self.norm2(x)
#         x = self.pwconv2(x)
#         x = self.norm2(x)
#         x = self.ese(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         # x = x.transpose([0, 3, 1, 2])  # (N, H, W, C) -> (N, C, H, W)

#         x = input +  x
#         return x
        

# class LayerNorm(nn.Layer):
#     """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs
#     with shape (batch_size, channels, height, width).
#     """

#     def __init__(self,
#                  normalized_shape,
#                  epsilon=1e-6,
#                  data_format="channels_last"):
#         super().__init__()

#         self.weight = paddle.create_parameter(shape=[normalized_shape],
#                                               dtype='float32',
#                                               default_initializer=ones_)

#         self.bias = paddle.create_parameter(shape=[normalized_shape],
#                                             dtype='float32',
#                                             default_initializer=zeros_)

#         self.epsilon = epsilon
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError
#         self.normalized_shape = (normalized_shape, )

#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight,
#                                 self.bias, self.epsilon)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / paddle.sqrt(s + self.epsilon)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x


# class L2Decay(paddle.regularizer.L2Decay):
#     def __init__(self, coeff=0.0):
#         super(L2Decay, self).__init__(coeff)

# class EffectiveSELayer(nn.Layer):
#     """ Effective Squeeze-Excitation
#     From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
#     """

#     def __init__(self, channels, act='hardsigmoid'):
#         super(EffectiveSELayer, self).__init__()
#         self.fc = nn.Conv2D(channels, channels, kernel_size=1, padding=0)
#         self.act = nn.Hardsigmoid()

#     def forward(self, x):
#         x_se = x.mean((2, 3), keepdim=True)
#         x_se = self.fc(x_se)
#         return x * self.act(x_se)



# class ConvBNLayer(nn.Layer):
#     def __init__(self,
#                  ch_in,
#                  ch_out,
#                  filter_size=3,
#                  stride=1,
#                  groups=1,
#                  padding=0,
#                  act=None):
#         super(ConvBNLayer, self).__init__()

#         self.conv = nn.Conv2D(
#             in_channels=ch_in,
#             out_channels=ch_out,
#             kernel_size=filter_size,
#             stride=stride,
#             padding=padding,
#             groups=groups)

#         self.bn = nn.BatchNorm2D(
#             ch_out,
#             weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
#             bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
#         self.act = nn.GELU()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.act(x)

#         return x


# class CSPStage(nn.Layer):
#     def __init__(self,
#                 block_fn,
#                 ch_in,
#                 ch_out,
#                 n,
#                 stride,
#                 p_rates,
#                 layer_scale_init_value=1e-6,
#                 act=nn.GELU,
#                 attn='eca',
#                 block_former = 1):
#         super().__init__()
#         ch_mid = (ch_in+ch_out)//2
#         if stride == 2:
#             self.down = nn.Sequential(ConvBNLayer(ch_in, ch_mid , 2, stride=2,  act=act))
#         else:
#             self.down = nn.Sequential(
#                 nn.Conv2D(ch_in, ch_mid, kernel_size=3, stride=1,padding=1),      
#                 LayerNorm(ch_mid, epsilon=1e-6, data_format="channels_first"),
#                 nn.GELU()
#             )
#         self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
#         self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
#         self.blocks = nn.Sequential(*[
#             block_fn(
#                 ch_mid // 2, drop_path=p_rates[i],layer_scale_init_value=layer_scale_init_value)
#             for i in range(n)
#         ])
#         if attn:
#             self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
#         else:
#             self.attn = None

#         self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

#     def forward(self, x):
#         if self.down is not None:
#             x = self.down(x)
#         y1 = self.conv1(x)
#         y2 = self.blocks(self.conv2(x))
#         y = paddle.concat([y1, y2], axis=1)
#         if self.attn is not None:
#             y = self.attn(y)
#         y = self.conv3(y)
#         return y

# class CSPConvNext(nn.Layer):
#     def __init__(
#         self,
#         in_chans=3,
#         depths=[3, 3, 9, 3],
#         dims=[64,128,256,512,1024],
#         drop_path_rate=0.2,
#         layer_scale_init_value=1e-6,
#         stride=[2,2,2,2],
#         return_idx=[1,2,3],

#     ):
#         super().__init__()
#         block_former = [Block,Block,Block,Block]
#         act = nn.GELU()
#         self.Down_Conv = nn.Sequential(
#                 ('conv1', ConvBNLayer(
#                     3, dims[0]//2 , 2, stride=2,  act=act)),
#                 ('conv2', ConvBNLayer(
#                     dims[0]//2, dims[0]//2 , 3, stride=1,padding=1,  act=act)),
#                 ('conv3', ConvBNLayer(
#                     dims[0]//2, dims[0] , 3, stride=1,padding=1, act=act)),
# )
#         dp_rates = [
#             x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
#         ]
#         n = len(depths)

#         self.stages = nn.Sequential(*[(str(i), CSPStage(
#             block_former[i], dims[i], dims[i + 1], depths[i], stride[i], dp_rates[sum(depths[:i]) : sum(depths[:i+1])],act=nn.GELU))
#                                       for i in range(n)])
#         self.norm = nn.BatchNorm(dims[-1])

#         self.apply(self._init_weights)


#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2D, nn.Linear)):
#             try:
#                 trunc_normal_(m.weight)
#                 zeros_(m.bias)
#             except:
#                 print(m)
    
    
#     def forward(self, inputs):
#         x = inputs
#         x = self.Down_Conv(x)
#         outs = []
#         for idx, stage in enumerate(self.stages):
#             x = stage(x)
#             # if idx in self.return_idx:
#             #     outs.append(x)
#         return self.norm(x.mean([-2, -1]))

# class Model(paddle.nn.Layer):
#     def __init__(self, class_num=1000,        in_chans=3,
#         depths=[3, 3, 9, 3],
#         dims=[64,128,256,512,1024],
#         drop_path_rate=0.2,
#         layer_scale_init_value=1e-6,
#         stride=[2,2,2,2],
#         return_idx=[1,2,3],):
#         super().__init__()
#         self.backbone = CSPConvNext(in_chans=in_chans,
#         depths=depths,
#         dims=dims,
#         drop_path_rate=drop_path_rate,
#         layer_scale_init_value=layer_scale_init_value,
#         stride=stride,)
#         self.head = ClasHead(with_avg_pool=False, in_channels=1024, num_classes=class_num)

#     def forward(self, x):

#         x = self.backbone(x)
#         x = self.head(x)
#         return x
    

# def CSPConvNeXt_tiny(pretrained=False, use_ssld=False, **kwargs):
#     model = Model( **kwargs)
#     _load_pretrained(
#         pretrained, model, MODEL_URLS["ConvNext_tiny"], use_ssld=use_ssld)
#     return model

# def CSPConvNeXt_tiny(pretrained=False, use_ssld=False, **kwargs):
#     model = Model( **kwargs)
#     _load_pretrained(
#         pretrained, model, MODEL_URLS["ConvNext_tiny"], use_ssld=use_ssld)
#     return model

# if __name__=="__main__":
#      model  = CSPConvNeXt_tiny()
#      # Total Flops: 1189500624     Total Params: 8688640
#      GFlops = paddle.flops(model,(1,3,224,224),print_detail=True)




import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant

# import numpy as np

# from ppdet.core.workspace import register, serializable
# from ..shape_spec import ShapeSpec
# from .transformer_utils import DropPath, trunc_normal_, zeros_

__all__ = ['CSPConvNeXt']

class Identity(nn.Layer):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

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

class Block(nn.Layer):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim,size, kernel_size=7, if_gourp=1,drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        if if_gourp == 1:
            groups = dim
        else:
            groups = 1
        self.dwconv = nn.Conv2D(dim, dim, kernel_size=kernel_size, padding=kernel_size//2,
                                groups=groups)  # depthwise conv
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

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()


    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        # x = x.transpose([0, 2, 3, 1])  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
       # x = self.ese(x)
        x = self.act(x)
        # self.grn(x)
        x = self.pwconv2(x)
        x = self.norm2(x)
        x = self.ese(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        # x = x.transpose([0, 3, 1, 2])  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

class L2Decay(paddle.regularizer.L2Decay):
    def __init__(self, coeff=0.0):
        super(L2Decay, self).__init__(coeff)

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
                size,
                kernel_size=7,
                if_group=1,
                layer_scale_init_value=1e-6,
                act=nn.GELU,
                attn='eca',
                block_former = 1):
        super().__init__()
        ch_mid = (ch_in+ch_out)//2
        if stride == 2:
            self.down = nn.Sequential(ConvBNLayer(ch_in, ch_mid , 2, stride=2,  act=act))
        else:
            self.down = Identity()
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2,size,kernel_size, if_group,drop_path=p_rates[i],layer_scale_init_value=layer_scale_init_value)
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

@register
@serializable
class CSPConvNeXt(nn.Layer):
    arch_settings = {
        'mini': {
            'depths': [3, 3, 9, 3],
            'dims': [48,96,192,384,768],
            'stem': 'va',
            'stride': [1,2,2,2]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'dims': [64,128,256,512,1024],
            'stem': 'vb',
            'stride': [2,2,2,2]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'dims': [64,128,256,512,1024],
            'stem': 'vb',
            'stride': [2,2,2,2]
        },
    }

    def __init__(
            self,
            arch='tiny',
            in_chans=3,
            drop_path_rate=0.,
            layer_scale_init_value=1e-6,
            return_idx=[1, 2, 3],
            norm_output=True,
            pretrained=None,
            class_num=1000,
             kernel_size=7,
            if_group=1,
            depth_mult = 1.0,
            width_mult = 1.0,
            stem = "vb" ):
        super().__init__()
        depths = self.arch_settings[arch]['depths']
        dims = self.arch_settings[arch]['dims']
        stem = self.arch_settings[arch]['stem']
        stride = self.arch_settings[arch]["stride"]
        block_former = [Block,Block,Block,Block]
        depths = [int(i*depth_mult) for i in depths]
        dims = [int(i*width_mult)  for i in dims]
        self.dims = dims[2:]
        act = nn.GELU()
        self.return_idx = return_idx

        if stem == "va":
            self.Down_Conv = nn.Sequential(
                ('conv1', ConvBNLayer(
                    in_chans,(dims[0]+dims[1])//2 , 4, stride=4,  act=act)),
            )
        if stem == "vb":
            self.Down_Conv = nn.Sequential(
                ('conv1', ConvBNLayer(
                    in_chans, dims[0]//2 , 2, stride=2,  act=act)),
                ('conv2', ConvBNLayer(
                    dims[0]//2, dims[0]//2 , 3, stride=1,padding=1,  act=act)),
                ('conv3', ConvBNLayer(
                    dims[0]//2, dims[0] , 3, stride=1,padding=1, act=act)),
            )
        
        dp_rates = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]
        n = len(depths)
        sizes = [224//4,224//8,224//16,224//32]
        self.stages = nn.Sequential(*[(str(i), CSPStage(
            block_former[i], 
            dims[i], 
            dims[i + 1], 
            depths[i], 
            stride[i],
            dp_rates[sum(depths[:i]) : sum(depths[:i+1])],
            kernel_size=kernel_size,
            if_group=if_group,
            size = sizes[i], 
            act=nn.GELU))
                                      for i in range(n)])
        self.norm = nn.BatchNorm(dims[-1])
        self.norm_output = norm_output
        if norm_output:
            self.norms = nn.LayerList([
                nn.BatchNorm2D(
                    c,)
                for c in self.dims
            ])

        # self.avgpool_pre_head = nn.Sequential(
        #         nn.AdaptiveAvgPool2D(1),
        #         nn.Conv2D(dims[-1], 1280, 1),
        #         nn.GELU()
        #     )
        # self.head = nn.Linear(1280, class_num)
        self.head = ClasHead(with_avg_pool=False, in_channels=dims[-1], num_classes=class_num)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            try:
                trunc_normal_(m.weight)
                zeros_(m.bias)
            except:
                print(m)
    
    
    def forward_body(self, inputs):
        x = inputs
        x = self.Down_Conv(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            # if idx in self.return_idx:
            #     outs.append(x)
        return self.norm(x.mean([-2, -1]))

    def forward_features(self, x):
        output = []
        x = self.Down_Conv(x)
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            output.append(x)

        outputs = [output[i] for i in self.return_idx]
        if self.norm_output:
            outputs = [self.norms[i](out) for i, out in enumerate(outputs)]

        return outputs

    def forward(self, x):
        x = self.forward_body(x)
        x = self.head(x)
        return x
    

class ClasHead(nn.Layer):
    """Simple classifier head.
    """
    def __init__(self, with_avg_pool=False, in_channels=1024, num_classes=1000):
        super(ClasHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc_cls = nn.Linear(in_channels, num_classes)
        # reset_parameters(self.fc_cls)

    def forward(self, x):
        if self.with_avg_pool:
            assert x.dim() == 4, \
                "Tensor must has 4 dims, got: {}".format(x.dim())
            x = self.avg_pool(x)
        # x = x.reshape([x.shape[0], -1])
        cls_score = self.fc_cls(x)
        return cls_score
    

def EConvNeXt_mini(pretrained=False, use_ssld=False, **kwargs):
    model = CSPConvNeXt( arch='mini' **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNext_tiny"], use_ssld=use_ssld)
    return model

def EConvNeXt_tiny(pretrained=False, use_ssld=False, **kwargs):
    model = CSPConvNeXt( arch='tiny' **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNext_tiny"], use_ssld=use_ssld)
    return model

def EConvNeXt_small(pretrained=False, use_ssld=False, **kwargs):
    model = CSPConvNeXt( arch='small' **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNext_tiny"], use_ssld=use_ssld)
    return model



