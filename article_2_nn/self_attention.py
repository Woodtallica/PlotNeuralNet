import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import block_ConvBnRelu6, inverted_residual, transition_block


# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    SelfAttention("SA", offset="(0,0,0)", to="(0,0,0)", width=20, height=30, depth=30, opacity=1.0, caption=" "),

    # Positionnal Segmented Self-Attention
    to_Conv('pos_Q', s_filter=256, n_filter=64, offset="(2.5,38,0)", to="(SA-east)", width=10, height=45, depth=45, opacity=0.75, caption=" "),
    to_Conv('pos_Q_embedded', s_filter=256, n_filter=64, offset="(-4.81,1,-3.5)", to="(pos_Q-east)", width=10, height=7, depth=7, opacity=0.75, caption=" "),
    to_Conv('pos_Q_extracted', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(pos_Q-east)", width=10, height=7, depth=7, opacity=1.0, caption=" "),
    to_Conv('pos_Q_reshaped', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(pos_Q_extracted-east)", width=0, height=45, depth=10, opacity=1.0, caption=" "),

    to_Conv('pos_K', s_filter=256, n_filter=64, offset="(2.5,25,0)", to="(SA-east)", width=10, height=45, depth=45, opacity=1.0, caption=" "),
    to_Conv('pos_K_embedded', s_filter=256, n_filter=64, offset="(-4.81,1,-3.5)", to="(pos_K-east)", width=10, height=7, depth=7, opacity=0.75, caption=" "),
    to_Conv('pos_K_extracted', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(pos_K-east)", width=10, height=7, depth=7, opacity=1.0, caption=" "),
    to_Conv('pos_K_reshaped', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(pos_K_extracted-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),

    to_Conv('pos_QK', s_filter=256, n_filter=64, offset="(10,6.5,0)", to="(pos_K_reshaped-east)", width=0, height=45, depth=45, opacity=1.0, caption=" "),

    to_Conv('pos_V', s_filter=256, n_filter=64, offset="(2.5,12,0)", to="(SA-east)", width=10, height=45, depth=45, opacity=1.0, caption=" "),
    to_Conv('pos_V_embedded', s_filter=256, n_filter=64, offset="(-4.81,1,-3.5)", to="(pos_V-east)", width=10, height=7, depth=7, opacity=0.75, caption=" "),
    to_Conv('pos_V_extracted', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(pos_V-east)", width=10, height=7, depth=7, opacity=1.0, caption=" "),
    to_Conv('pos_V_reshaped', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(pos_V_extracted-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),

    to_Conv('pos_QKV', s_filter=256, n_filter=64, offset="(14,9.75,0)", to="(pos_V_reshaped-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),

    # Channel Segmented Self-Attention
    to_Conv('chan_Q', s_filter=256, n_filter=64, offset="(2.5,-35,0)", to="(SA-east)", width=10, height=45, depth=45, opacity=1.0, caption=" "),
    to_Conv('chan_Q_embedded', s_filter=256, n_filter=64, offset="(-4.81,1,-3.5)", to="(chan_Q-east)", width=10, height=7, depth=7, opacity=0.75, caption=" "),
    to_Conv('chan_Q_extracted', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(chan_Q-east)", width=10, height=7, depth=7, opacity=1.0, caption=" "),
    to_Conv('chan_Q_reshaped', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(chan_Q_extracted-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),

    to_Conv('chan_K', s_filter=256, n_filter=64, offset="(2.5,-22,0)", to="(SA-east)", width=10, height=45, depth=45, opacity=1.0, caption=" "),
    to_Conv('chan_K_embedded', s_filter=256, n_filter=64, offset="(-4.81,1,-3.5)", to="(chan_K-east)", width=10, height=7, depth=7, opacity=0.75, caption=" "),
    to_Conv('chan_K_extracted', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(chan_K-east)", width=10, height=7, depth=7, opacity=1.0, caption=" "),
    to_Conv('chan_K_reshaped', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(chan_K_extracted-east)", width=0, height=45, depth=10, opacity=1.0, caption=" "),

    to_Conv('chan_QK', s_filter=256, n_filter=64, offset="(7,-6.5,0)", to="(chan_K_reshaped-east)", width=0, height=10, depth=10, opacity=1.0, caption=" "),

    to_Conv('chan_V', s_filter=256, n_filter=64, offset="(2.5,-9,0)", to="(SA-east)", width=10, height=45, depth=45, opacity=1.0, caption=" "),
    to_Conv('chan_V_embedded', s_filter=256, n_filter=64, offset="(-4.81,1,-3.5)", to="(chan_V-east)", width=10, height=7, depth=7, opacity=0.75, caption=" "),
    to_Conv('chan_V_extracted', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(chan_V-east)", width=10, height=7, depth=7, opacity=1.0, caption=" "),
    to_Conv('chan_V_reshaped', s_filter=256, n_filter=64, offset="(2.5,0,0)", to="(chan_V_extracted-east)", width=0, height=45, depth=10, opacity=1.0, caption=" "),

    to_Conv('chan_QKV', s_filter=256, n_filter=64, offset="(14,-9.75,0)", to="(chan_V_reshaped-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),

    to_Conv('G', s_filter=256, n_filter=64, offset="(5,16.5,0)", to="(chan_QKV-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),

    to_Conv('concat_0', s_filter=256, n_filter=64, offset="(4,0,0)", to="(G-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_1', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_0-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_2', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_1-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_3', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_2-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_4', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_3-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_5', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_4-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_6', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_5-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_7', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_6-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_8', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_7-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_9', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_8-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_10', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_9-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_11', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_10-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_12', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_11-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_13', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_12-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_14', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_13-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_15', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_14-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_16', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_15-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),
    to_Conv('concat_17', s_filter=256, n_filter=64, offset="(0.15,0,0)", to="(concat_16-east)", width=0, height=10, depth=45, opacity=1.0, caption=" "),


    to_Conv('reshaped_G', s_filter=256, n_filter=64, offset="(5,0,0)", to="(concat_17-east)", width=10, height=45, depth=45, opacity=1.0, caption=" "),
    to_Conv('SA_out', s_filter=256, n_filter=64, offset="(5,0,0)", to="(reshaped_G-east)", width=20, height=45, depth=45, opacity=1.0, caption=" "),
    ]

# for i in range(3):
#     arch.extend(*inverted_residual(["dwise", "1x1"], last_layer, 'out'))

arch.append(to_end())


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
