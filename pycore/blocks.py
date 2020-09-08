
from .tikzeng import *

#define new block
def block_2ConvPool( name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ):
    return [
    to_ConvConvRelu( 
        name="ccr_{}".format( name ),
        s_filer=str(s_filer), 
        n_filer=(n_filer,n_filer), 
        offset=offset, 
        to="({}-east)".format( botton ), 
        width=(size[2],size[2]), 
        height=size[0], 
        depth=size[1],   
        ),    
    to_Pool(         
        name="{}".format( top ), 
        offset="(0,0,0)", 
        to="(ccr_{}-east)".format( name ),  
        width=1,         
        height=size[0] - int(size[0]/4), 
        depth=size[1] - int(size[0]/4), 
        opacity=opacity, ),
    to_connection( 
        "{}".format( botton ), 
        "ccr_{}".format( name )
        )
    ]

def block_ConvBnRelu6(name, top, first_block, bottom, s_filter=256, n_filter=64,
                      offset="(0.2,0,0)", size=(32,32,3.5), opacity=1.0):
    if first_block:
        anchor = "(0,0,0)"
    else:
        anchor = "({}-east)".format(bottom)

    return [
    to_Conv(
        name="{}_ccr".format(name),
        s_filter=str(s_filter),
        n_filter=(n_filter, n_filter),
        offset=offset,
        to=anchor,
        width=size[2],
        height=size[0],
        depth=size[1],
        ),
    to_BatchNorm(
        name="{}_bn".format(name),
        offset=offset,
        to="({}_ccr-east)".format(name),
        width=1.0,
        height=size[0],
        depth=size[1],
        opacity=opacity),
     to_Relu6(top, # "{}_relu6".format(name)
     offset=offset,
     to="({}_bn-east)".format(name),
     width=1.0,
     height=size[0],
     depth=size[1],
     opacity=opacity)
    ]


def block_Unconv( name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ):
    return [
        to_UnPool(  name='unpool_{}'.format(name),    offset=offset,    to="({}-east)".format(botton),         width=1,              height=size[0],       depth=size[1], opacity=opacity ),
        to_ConvRes( name='ccr_res_{}'.format(name),   offset="(0,0,0)", to="(unpool_{}-east)".format(name),    s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1], opacity=opacity ),       
        to_Conv(    name='ccr_{}'.format(name),       offset="(0,0,0)", to="(ccr_res_{}-east)".format(name),   s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_ConvRes( name='ccr_res_c_{}'.format(name), offset="(0,0,0)", to="(ccr_{}-east)".format(name),       s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1], opacity=opacity ),       
        to_Conv(    name='{}'.format(top),            offset="(0,0,0)", to="(ccr_res_c_{}-east)".format(name), s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_connection( 
            "{}".format( botton ), 
            "unpool_{}".format( name ) 
            )
    ]


def block_Res( num, name, bottom, top, s_filer=256, n_filer=64, offset="(0,0,0)", size=(32,32,3.5), opacity=0.5 ):
    lys = []
    layers = [ *[ '{}_{}'.format(name,i) for i in range(num-1) ], top]
    for name in layers:        
        ly = [ to_Conv( 
            name='{}'.format(name),       
            offset=offset, 
            to="({}-east)".format( bottom ),
            s_filer=str(s_filer), 
            n_filer=str(n_filer), 
            width=size[2],
            height=size[0],
            depth=size[1]
            ),
            to_connection( 
                "{}".format( bottom  ),
                "{}".format( name ) 
                )
            ]
        bottom = name
        lys+=ly
    
    lys += [
        to_skip( of=layers[1], to=layers[-2], pos=1.25),
    ]
    return lys


# def inv_block(num, name, bottom, top, s_filter=256, n_filter=64, offset="(0,0,0)", size=(32, 32, 3.5)):
#     lys = []
#     layers = [*['{}_{}'.format(name, i) for i in range(num - 1)], top]
#     for name in layers:
#         ly = [
#             conv_bn_relu6(
#         name='{}'.format(name),
#         offset=offset,
#         to="({}-east)".format(bottom),
#         s_filter=s_filter,
#         n_filter=n_filter,
#         width=size[2],
#         height=size[0],
#         depth=size[1])
#     ]
#         bottom = name
#         lys += ly
#
#     return lys, layers[-1]


def conv_bn_relu6(num, name, bottom, out_lin=True, s_filter=256, n_filter=64, offset="(0,0,0)", size=(32, 32, 10),
                  caption=" ", opacity=0.5):
    lys = []
    layers = [*['{}_{}'.format(name, i) for i in range(num)]]
    for idx, name in enumerate(layers):
        name_l = '_'.join([name, 'conv'])

        if idx == 0:
            first_layer = name_l
            stacked_offset = "(1,0,0)"
        else:
            offset = "(0,0,0)"

        lys.append(
            to_Conv(name_l, s_filter=s_filter, n_filter=n_filter, offset=offset, to="({}-east)".format(bottom),
                    height=size[0], depth=size[1], width=size[2], opacity=opacity,caption=caption))
        bottom = name_l

    name_l = '_'.join([name, 'bn'])
    lys.append(to_BatchNorm(name_l, offset="(0.15,0,0)", to="({}-east)".format(bottom), height=size[0], depth=size[1],
                            width=1, opacity=opacity, caption=caption))
    bottom = name_l
    if out_lin:
        return lys, name_l, first_layer

    name_l = '_'.join([name, 'relu6'])
    lys.append(to_Relu6(name_l, offset="(0.15,0,0)", to="({}-east)".format(bottom), height=size[0],
                        depth=size[1], width=1, opacity=opacity, caption=caption))
    return lys, name_l, first_layer


def inverted_residual(block_name, bottom, top, size=(32, 32, 8), opacity=0.75):
    previous_block_bottom = bottom
    inverted_layers = ["1x1_relu6", "3x3_relu6_dwise"]
    layers_name = ['_'.join([block_name, l]) for l in inverted_layers]

    lys = []
    expand, last_layer_1, first_layer = conv_bn_relu6(8, layers_name[0], bottom, out_lin=False, s_filter=256, n_filter=1, offset="(5,0,0)",
                                            size=(size[0], size[1], 2.0), opacity=opacity)
    lys += expand
    lys += to_connection(bottom, first_layer)
    new_bottom = last_layer_1
    dwise, last_layer_2, first_layer = conv_bn_relu6(1, layers_name[1], new_bottom, out_lin=False, s_filter=256, n_filter=32, offset="(2.5,0,0)",
                                                     size=(size[0], size[1], size[2]*2), opacity=opacity)
    lys += dwise
    lys += to_connection(last_layer_1, first_layer)
    new_bottom = last_layer_2

    out_conv_bn, last_layer_3, first_layer = conv_bn_relu6(1, top, bottom=new_bottom, out_lin=True, s_filter=256,
                                                    n_filter=64, offset="(2.5,0,0)", size=size, opacity=opacity)
    lys += out_conv_bn
    lys += to_connection(last_layer_2, first_layer)

    lys += [
        to_skip(of=previous_block_bottom, to=last_layer_3, pos=3),
    ]
    return lys


def transition_block(name, top, first_block, bottom, stride, s_filter=256, n_filter=64,
                      offset="(0.2,0,0)", size=(32,32,3.5), opacity=1.0):
    if first_block:
        anchor = "(0,0,0)"
    else:
        anchor = "({}-east)".format(bottom)

    first_layer = "{}_ccr".format(name)
    return [
    to_Conv(
        name=first_layer,
        s_filter=str(s_filter),
        n_filter=(n_filter, n_filter),
        offset=offset,
        to=anchor,
        width=size[2],
        height=size[0],
        depth=size[1]
        ),
    to_BatchNorm(
        name="{}_bn".format(name),
        offset="(0.15,0,0)",
        to="({}_ccr-east)".format(name),
        width=1.0,
        height=size[0],
        depth=size[1],
        opacity=opacity),
    to_Relu6(top,
             offset="(0.15,0,0)",
             to="({}_bn-east)".format(name),
             width=1.0,
             height=size[0],
             depth=size[1],
             opacity=opacity),
    to_connection(bottom, first_layer)
    ]
