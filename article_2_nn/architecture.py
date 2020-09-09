import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import block_ConvBnRelu6, inverted_residual, transition_block


# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    *block_ConvBnRelu6("init_conv", top="out_init_conv", first_block=True, bottom="(0,0,0)", s_filter=256,
                       n_filter=64, offset="(0.15,0,0)", size=(32,32,8), opacity=0.75),
    *inverted_residual("inverted_block_1", bottom='out_init_conv', top='inverted_block_1_out', size=(32, 32, 8), opacity=0.75),
    *transition_block("transition_block_1", top="out_trans_block_1", first_block=False, bottom='inverted_block_1_out_0_bn', stride=2,
                        s_filter=256, n_filter=64, offset="(2.5,0,0)", size=(26,26,8), opacity=0.75),
    *inverted_residual("inverted_block_2", bottom='out_trans_block_1', top='inverted_block_2_out', size=(26, 26, 10), opacity=0.75),
    *transition_block("transition_block_2", top="out_trans_block_2", first_block=False,
                      bottom='inverted_block_2_out_0_bn', stride=2,
                      s_filter=256, n_filter=64, offset="(2.5,0,0)", size=(20, 20, 12), opacity=0.75),
    *inverted_residual("inverted_block_3", bottom='out_trans_block_2', top='inverted_block_3_out', size=(20, 20, 12), opacity=0.75),
    SelfAttention("SA", offset="(2.5,0,0)", to="(inverted_block_3_out_0_bn-east)", width=12, height=20, depth=20, opacity=1.0, caption=" "),
    to_connection('inverted_block_3_out_0_bn', 'SA'),
    to_Pool('global_pooling', offset="(2.5,0,0)", to="(SA-east)", width=1, height=20, depth=20, opacity=1.0, caption=" "),
    to_connection('SA', 'global_pooling'),
    to_FC('linear', offset="(2.5,0,0)", to="(global_pooling-east)", width=1, height=1, depth=14, opacity=1.0, caption=" "),
    to_connection('global_pooling', 'linear'),
    to_SoftMax('out_softmax', s_filer=10, offset="(0.15,0,0)", to="(linear-east)", width=1, height=1, depth=14, opacity=1.0, caption=" " )
    ]

# for i in range(3):
#     arch.extend(*inverted_residual(["dwise", "1x1"], last_layer, 'out'))

arch.append(to_end())


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
