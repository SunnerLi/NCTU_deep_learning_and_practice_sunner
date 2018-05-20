from matplotlib import pyplot as plt
import numpy as np
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default = './after_record.txt', type = str, help = 'The path of output log file')
    parser.add_argument('--title', default = 'The curve of TD before-state with pattern 2', type = str, help = 'The title of the image')   
    return parser.parse_args()

def draw(args):
    # Get the whole candidate sentence
    candidate_list = []
    for string in open(args.log, 'r').readlines():
        if "max" in string:
            candidate_list.append(string)

    # Split the mean and max value and record
    mean_list = []
    max_list = []
    for string in candidate_list:
        mean_list.append(float(string.split('\t')[1].split()[-1]))
        max_list.append(int(string.split('\t')[-1].split()[-1]))

    # Draw
    plt.plot(range(len(mean_list)), mean_list, '-', label = "The mean of total reward")
    plt.plot(range(len(max_list)), max_list, '-', label = "The maximun of total reward")
    plt.title(args.title)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    args = parse()
    draw(args)