import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import block_ConvBnRelu6, inverted_residual


# defined your arch
last_layer = 'init_conv'
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    *block_ConvBnRelu6("init_conv", first_block=True, bottom="(0,0,0)", s_filter=256,
                       n_filter=64, offset="(0.15,0,0)", size=(32,32,3.5), opacity=1.0),
    *inverted_residual(["1x1_relu6_conv, 3x3_relu6_dwise", '1x1_linear_conv'], 'init_conv_relu6', 'out_inv_block_1')
    # to_Conv(last_layer, 3, 8, offset="(0,0,0)", to="(0,0,0)", height=32, depth=32, width=1),
    # to_BatchNorm('batchnorm', offset="(0.2,0,0)", to="(init_conv-east)", height=32, depth=32, width=1),
    # to_Relu6('relu6', offset="(0.2,0,0)", to="(batchnorm-east)", height=32, depth=32, width=1)
    ]

# for i in range(3):
#     arch.extend(*inverted_residual(["dwise", "1x1"], last_layer, 'out'))

arch.append(to_end())


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
