import numpy as np


def main():
    size = 500
    with open("input.txt", 'w') as fout:
        fout.write(str(size)+'\n')
        for i in range(size):
            fout.write(str(np.random.randint(200))+' '+str(np.random.randint(200))+' '+str(np.random.randint(200))+'\n')


main()