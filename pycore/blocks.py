
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


def inv_block(num, name, bottom, top, s_filer=256, n_filer=64, offset="(0,0,0)", size=(32, 32, 3.5), opacity=0.5):
    lys = []
    layers = [*['{}_{}'.format(name, i) for i in range(num - 1)], top]
    for name in layers:
        ly = [to_Conv(
        name='{}'.format(name),
        offset=offset,
        to="({}-east)".format(bottom),
        s_filer=str(s_filer),
        n_filer=str(n_filer),
        width=size[2],
        height=size[0],
        depth=size[1])
    ]
        bottom = name
        lys += ly

    return lys, layers[-1]


def inverted_residual(in_name, bottom, top):
    lys = []
    dwise, last_layer = inv_block(64, in_name[0], bottom, top, s_filer=256, n_filer=1, offset="(0,0,0)",
                                    size=(32, 32, 0.2), opacity=0.5)
    lys += dwise
    bottom = last_layer
    dadd, last_layer = inv_block(1, in_name[1], bottom, top, s_filer=256, n_filer=64, offset="(1,0,0)",
                                    size=(32, 32, 10), opacity=0.5)
    lys += dadd
    # lys += [
    #     to_skip(of=layers[1], to=layers[-2], pos=1.25),
    # ]
    return lys


def transition_block(in_name, bottom, top):
    lys = []
    dwise, last_layer = inv_block(64, in_name[0], bottom, top, s_filer=256, n_filer=1, offset="(0,0,0)",
                                    size=(32, 32, 0.2), opacity=0.5)
    lys += dwise
    bottom = last_layer
    dadd, last_layer = inv_block(1, in_name[1], bottom, top, s_filer=256, n_filer=64, offset="(1,0,0)",
                                    size=(32, 32, 10), opacity=0.5)
    lys += dadd
    # lys += [
    #     to_skip(of=layers[1], to=layers[-2], pos=1.25),
    # ]
    return lys
