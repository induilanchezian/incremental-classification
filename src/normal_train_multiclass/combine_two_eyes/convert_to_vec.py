import numpy as np 
import argparse

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--src', type=str, help='File with left right predictions')
    parser.add_argument('--dest', type=str, help='Destination file')

    args = parser.parse_args()

    x = np.load(args.src)
    x = x[:,:2]

    left = x[:,0]
    right = x[:,1]

    preds = np.append(left, right)
    print(preds.shape)

    np.save(args.dest, preds)

if __name__ == '__main__':
    main()
