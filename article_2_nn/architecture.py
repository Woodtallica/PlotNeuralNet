import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import inverted_residual

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_Conv("init_conv", 3, 8, offset="(0,0,0)", to="(0,0,0)", height=32, depth=32, width=1),
    *inverted_residual(["dwise", "1x1"], 'init_conv', 'out'),

    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
