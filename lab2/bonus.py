from models.downsampler import Downsampler
from skimage.measure import compare_psnr
from torch.autograd import Variable
from skimage import io, transform
from utils.sr_utils import *
from models import *
import matplotlib.pyplot as plt
import argparse

iteration = 0

def loadImageAndMask(args):
    img_pil, img_np = get_image(args.image_path, -1)
    img_mask_pil, img_mask_np = get_image(args.mask_path, -1)

    # Crop to center
    img_mask_pil = crop_image(img_mask_pil, 64)
    img_pil      = crop_image(img_pil,      64)
    img_np      = pil_to_np(img_pil)
    img_mask_np = pil_to_np(img_mask_pil)

    # Visualize
    img_mask_var = np_to_var(img_mask_np).float()
    # plot_image_grid([img_np, img_mask_np, img_mask_np*img_np], 3,11)
    return img_np, img_mask_np

def train(args, img, mask):
    # Load model
    net = skip(
        num_input_channels = 2,
        num_output_channels = 3,
        num_channels_down = [16, 32, 64, 128, 128],
        num_channels_up = [16, 32, 64, 128, 128],
        num_channels_skip = [0, 0, 0, 0, 0],
        upsample_mode = 'nearest',
        need_sigmoid = True,
        need_bias = True,
        pad = 'reflection',
        act_fun = 'LeakyReLU'
    )
    net = net.float()
    net = net.cuda() if torch.cuda.is_available() else net

    # Compute the number of parameters
    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    print('Number of parameters: ', s)
    criterion = nn.MSELoss()

    # Define input and target, and deploy into GPU
    net_input = get_noise(2, 'noise', (np.shape(img)[1], np.shape(img)[2])).float().detach()
    img_var = np_to_var(img).float()
    mask_var = np_to_var(mask).float()
    net_input = net_input.cuda() if torch.cuda.is_available() else net_input
    img_var = img_var.cuda() if torch.cuda.is_available() else img_var
    mask_var = mask_var.cuda() if torch.cuda.is_available() else mask_var

    # Define closure
    loss_list = []
    def closure():
        global iteration
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n.data += n.data.clone().normal_()*n.data.std()/50
        out = net(net_input)
        total_loss = criterion(out * mask_var, img_var * mask_var)
        total_loss.backward()

        # Log
        print ('Iteration %05d    Loss %f' % (iteration, total_loss.data[0]), '\r', end='')
        loss_list.append(total_loss.data[0])
        if iteration % 1000 == 0 or iteration == args.epoch - 1:
            out_np = var_to_np(out)
            # plot_image_grid([np.clip(out_np, 0, 1)], factor=5, nrow=1)
            saved_img = np.transpose(np.clip(out_np, 0, 1), [1, 2, 0])
            io.imsave('bonus_' + str(iteration) + '.png', saved_img)
            # io.imsave('training_result.png', saved_img)
        iteration += 1
        return total_loss

    # Optimize (Adam with first 100 epoch and LBFGS with rest)
    p = get_params('net', net, net_input)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    for j in range(args.epoch):
        optimizer.zero_grad()
        closure()
        optimizer.step()

    # Plot the loss curve
    plt.plot(range(len(loss_list)), loss_list)
    plt.savefig('bonus_loss_curve.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default = 1, type = int, help = 'The value of epoch')
    parser.add_argument('--image_path', default = './images/bonus/2.png', type = str, help = 'The path of natural image')
    parser.add_argument('--mask_path', default = './images/bonus/2_mask.png', type = str, help = 'The path of natural image')
    parser.add_argument('--model_path', default = './sr.ckpt', type = str, help = 'The path of trained model')
    args = parser.parse_args()
    img, mask = loadImageAndMask(args)
    train(args, img, mask)