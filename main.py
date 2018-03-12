import sys
import argparse
from managers.shallow import main as shallow_main
from managers.deep import main as deep_main
def main():

    parser = argparse.ArgumentParser(description="Fake News Identification")

    parser.add_argument('--shallow', help="Use Shallow Learning", action='store_true')
    parser.add_argument('--deep', help="Use Deep Learning (LSTM)", action='store_true')

    args = parser.parse_args()
    args = vars(args)

    use_shallow = args.get('shallow')
    use_deep = args.get('deep')


    if(use_deep):
        deep_main()
    
    if(use_shallow):
        shallow_main()
    

if __name__ == '__main__':
    main()