import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import inverted_residual


# defined your arch
last_layer = 'init_conv'
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_Conv(last_layer, 3, 8, offset="(0,0,0)", to="(0,0,0)", height=32, depth=32, width=1),
    to_Relu6('yess', 3, offset="(0.2,0,0)", to="(init_conv-east)", height=32, depth=32, width=1)
    ]

# for i in range(3):
#     arch.extend(*inverted_residual(["dwise", "1x1"], last_layer, 'out'))

arch.append(to_end())


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
