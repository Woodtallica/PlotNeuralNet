import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import block_ConvBnRelu6, inverted_residual, transition_block


# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    SelfAttention("SA", offset="(0,0,0)", to="(0,0,0)", width=12, height=20, depth=20, opacity=1.0, caption=" "),

    # Positionnal Segmented Self-Attention
    to_Conv('pos_Q', s_filter=256, n_filter=64, offset="(2.5,21,0)", to="(SA-east)", width=6, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('pos_Q_extracted', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(pos_Q-east)", width=6, height=5, depth=5, opacity=1.0, caption=" "),
    to_Conv('pos_Q_reshaped', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(pos_Q_extracted-east)", width=0, height=25, depth=6, opacity=1.0, caption=" "),

    to_Conv('pos_K', s_filter=256, n_filter=64, offset="(2.5,14,0)", to="(SA-east)", width=6, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('pos_K_extracted', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(pos_K-east)", width=6, height=5, depth=5, opacity=1.0, caption=" "),
    to_Conv('pos_K_reshaped', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(pos_K_extracted-east)", width=0, height=6, depth=25, opacity=1.0, caption=" "),

    to_Conv('pos_QK', s_filter=256, n_filter=64, offset="(7,3,0)", to="(pos_K_reshaped-east)", width=0, height=25, depth=25, opacity=1.0, caption=" "),

    to_Conv('pos_V', s_filter=256, n_filter=64, offset="(2.5,7,0)", to="(SA-east)", width=6, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('pos_V_extracted', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(pos_V-east)", width=6, height=5, depth=5, opacity=1.0, caption=" "),
    to_Conv('pos_V_reshaped', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(pos_V_extracted-east)", width=0, height=6, depth=25, opacity=1.0, caption=" "),

    to_Conv('pos_QKV', s_filter=256, n_filter=64, offset="(14,3.5,0)", to="(pos_V_reshaped-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),

    # Channel Segmented Self-Attention
    to_Conv('chan_Q', s_filter=256, n_filter=64, offset="(2.5,-21,0)", to="(SA-east)", width=6, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('chan_Q_extracted', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(chan_Q-east)", width=6, height=5, depth=5, opacity=1.0, caption=" "),
    to_Conv('chan_Q_reshaped', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(chan_Q_extracted-east)", width=0, height=25, depth=6, opacity=1.0, caption=" "),

    to_Conv('chan_K', s_filter=256, n_filter=64, offset="(2.5,-14,0)", to="(SA-east)", width=6, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('chan_K_extracted', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(chan_K-east)", width=6, height=5, depth=5, opacity=1.0, caption=" "),
    to_Conv('chan_K_reshaped', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(chan_K_extracted-east)", width=0, height=6, depth=25, opacity=1.0, caption=" "),

    to_Conv('chan_QK', s_filter=256, n_filter=64, offset="(7,-3.5,0)", to="(chan_K_reshaped-east)", width=0, height=25, depth=25, opacity=1.0, caption=" "),

    to_Conv('chan_V', s_filter=256, n_filter=64, offset="(2.5,-7,0)", to="(SA-east)", width=6, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('chan_V_extracted', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(chan_V-east)", width=6, height=5, depth=5, opacity=1.0, caption=" "),
    to_Conv('chan_V_reshaped', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(chan_V_extracted-east)", width=0, height=6, depth=25, opacity=1.0, caption=" "),

    to_Conv('chan_QKV', s_filter=256, n_filter=64, offset="(14,-3.5,0)", to="(chan_V_reshaped-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),

    to_Conv('G', s_filter=256, n_filter=64, offset="(4,10.5,0)", to="(chan_QKV-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),

    to_Conv('concat_0', s_filter=256, n_filter=64, offset="(4,0,0)", to="(G-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('concat_1', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_0-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('concat_2', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_1-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('concat_3', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_2-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('concat_4', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_3-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('concat_5', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_4-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('concat_6', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_5-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('concat_7', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_6-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('concat_8', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_7-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('concat_9', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_8-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('concat_10', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_9-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('concat_11', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_10-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),
    to_Conv('concat_12', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_11-east)", width=0, height=20, depth=20, opacity=1.0, caption=" "),

    to_Conv('SA_out', s_filter=256, n_filter=64, offset="(5,0,0)", to="(concat_4-east)", width=12, height=20, depth=20, opacity=1.0, caption=" "),
    ]

# for i in range(3):
#     arch.extend(*inverted_residual(["dwise", "1x1"], last_layer, 'out'))

arch.append(to_end())


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
