from models.downsampler import Downsampler
from skimage.measure import compare_psnr
from torch.autograd import Variable
from skimage import io, transform
from utils.sr_utils import *
from models import *
import matplotlib.pyplot as plt
import argparse

loss_image = []
loss_image_noise = []
loss_image_shuffled = []
loss_pure_noise = []
iteration = 0

def getNet():
    net = skip(
        num_input_channels = 3,
        num_output_channels = 3,
        num_channels_down = [8, 16, 32, 64, 128],
        num_channels_up = [8, 16, 32, 64, 128],
        num_channels_skip = [0, 0, 0, 4, 4],
        upsample_mode = 'bilinear',
        need_sigmoid = True,
        need_bias = True,
        pad = 'reflection',
        act_fun = 'LeakyReLU'
    )
    net = net.float()
    net = net.cuda() if torch.cuda.is_available() else net
    return net

def imageShuffle(img, ratio = 5):
    flat_img = np.reshape(np.copy(img), [-1, 3])
    shuffle_time = int(np.shape(flat_img)[0] * ratio)
    for i in range(shuffle_time):
        idx1, idx2 = np.random.randint(0, np.shape(flat_img)[0] - 1, 2)
        flat_img[idx1], flat_img[idx2] = flat_img[idx2], flat_img[idx1]
    return np.reshape(flat_img, np.shape(img))

def train(args, img):
    global loss_image
    global loss_image_noise
    global loss_image_shuffled
    global loss_pure_noise
    global iteration

    # Define closure and optimizer
    criterion = nn.MSELoss()
    def closure(loss_list):
        global iteration, psnr_HR
        net_input = torch.cuda.FloatTensor(target_var.size()).normal_()
        net_out = net(net_input)
        total_loss = criterion(net_out, target_var) 
        total_loss.backward()

        # Log
        print('Iteration %05d   Loss %3f' % (iteration, total_loss.data[0]), '\r', end = '')
        loss_list.append(total_loss.data.cpu().numpy()[0])
        iteration += 1
        return total_loss

    # -------------------------------------------------------------
    # Blue curve
    # -------------------------------------------------------------
    # define target and net
    net = getNet()    
    target_var = np_to_var(img).float() / 255.
    target_var = target_var.transpose(2, 3).transpose(1, 2)
    target_var = target_var.cuda() if torch.cuda.is_available() else target_var
    net_input = get_noise(3, 'noise', (target_var.size()[2], target_var.size()[3])).float().detach()
    net_input = net_input.cuda() if torch.cuda.is_available() else net_input

    # Optimize
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    for j in range(args.epoch):
        optimizer.zero_grad()
        closure(loss_image)
        optimizer.step()
    iteration = 0

    # -------------------------------------------------------------
    # Green curve
    # -------------------------------------------------------------
    # define target and net
    net = getNet()
    noise_factor = 0.2
    target_var = np_to_var(img + noise_factor * np.random.randint(0, 255, np.shape(img))).float() / 255.
    target_var = target_var.transpose(2, 3).transpose(1, 2)
    target_var = target_var.cuda() if torch.cuda.is_available() else target_var

    # Optimize
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    for j in range(args.epoch):
        optimizer.zero_grad()
        closure(loss_image_noise)
        optimizer.step()
    iteration = 0

    # -------------------------------------------------------------
    # Red curve
    # -------------------------------------------------------------
    # define target and net
    net = getNet()
    noise_factor = 0.3
    shuffle_img = imageShuffle(img)
    target_var = np_to_var(shuffle_img).float() / 255.
    target_var = target_var.transpose(2, 3).transpose(1, 2)
    target_var = target_var.cuda() if torch.cuda.is_available() else target_var

    # Optimize
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    for j in range(args.epoch):
        optimizer.zero_grad()
        closure(loss_image_shuffled)
        optimizer.step()
    iteration = 0

    # -------------------------------------------------------------
    # Purple curve
    # -------------------------------------------------------------
    # define target and net
    net = getNet()
    _, channel, height, width = target_var.size()
    target_var = np_to_var(np.random.uniform(size = [channel, height, width])).float()
    target_var = target_var.cuda() if torch.cuda.is_available() else target_var

    # Optimize
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for j in range(args.epoch):
        optimizer.zero_grad()
        closure(loss_pure_noise)
        optimizer.step()
    iteration = 0

    # Draw and show
    plt.plot(range(len(loss_image)), loss_image, label = 'Image')
    plt.plot(range(len(loss_image_noise)), loss_image_noise, label = 'Image + noise')
    plt.plot(range(len(loss_image_shuffled)), loss_image_shuffled, label = 'Image shuffled')
    plt.plot(range(len(loss_pure_noise)), loss_pure_noise, label = 'U(0, 1) noise')
    plt.legend()
    plt.savefig('requirement1.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default = 1, type = int, help = 'The value of epoch')
    parser.add_argument('--image_path', default = './images/SR_GT.png', type = str, help = 'The path of natural image')
    parser.add_argument('--model_path', default = './sr.ckpt', type = str, help = 'The path of trained model')
    args = parser.parse_args()
    train(args, io.imread(args.image_path))