from skimage.measure import compare_psnr
from torch.autograd import Variable
from utils.denoising_utils import *
from skimage import io
from models import *
import argparse

iteration = 0
final_psnr = 0.

def load_img(args):
    if args.noise_img_gt is None and args.noise_img is None:
        raise Exception('You should at least assign one path of image...')
    img_pil, img_np = None, None
    if args.noise_img_gt is not None:
        img_pil = crop_image(get_image(args.noise_img_gt)[0])
        img_np = pil_to_np(img_pil)
        img_noisy_pil, img_noisy_np = get_noisy_image(img_np, 25/255.)
    if args.noise_img is not None:
        img_noisy_pil = crop_image(get_image(args.noise_img)[0])
        img_noisy_np = pil_to_np(img_noisy_pil)
    img_pil = img_noisy_pil if img_pil is None else img_pil
    img_np = img_noisy_np if img_np is None else img_np
    # plot_image_grid([img_np, img_noisy_np], 4, 6)
    return img_noisy_np, img_np

def train(args, img_noisy_np, img_np):
    # Load model
    net = skip(
        num_input_channels = 3,
        num_output_channels = 3,
        num_channels_down = [128, 128, 128, 128, 128],
        num_channels_up = [128, 128, 128, 128, 128],
        num_channels_skip = [4, 4, 4, 4, 4],
        upsample_mode = 'bilinear',
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
    img_noisy_var = np_to_var(img_noisy_np).float()

    # define input
    net_input = get_noise(3, 'noise', (np.shape(img_np)[1], np.shape(img_np)[2])).float().detach()
    img_noisy_var = np_to_var(img_noisy_np).float()
    img_noisy_var = img_noisy_var.cuda() if torch.cuda.is_available() else img_noisy_var
    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()

    # Define closure
    reg_noise_std = 1./30.
    net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)
    net_input = net_input.cuda() if torch.cuda.is_available() else net_input
    def closure():
        global iteration
        global final_psnr
        out = net(net_input)
        total_loss = criterion(out, img_noisy_var)
        total_loss.backward()
        psnr = compare_psnr(img_np, np.clip(var_to_np(out), 0, 1))
        final_psnr = psnr
        print ('Iteration %5d    Loss %f   PSNR %f' % (iteration, total_loss.data[0], psnr), '\r', end='')
        if  iteration % 300 == 0 or iteration == args.epoch - 1:
            out_np = var_to_np(out)
            # plot_image_grid([np.clip(out_np, 0, 1)], factor=5, nrow=1)
            saved_img = np.transpose(np.clip(out_np, 0, 1), [1, 2, 0])
            io.imsave('denoise_' + str(iteration) + '.png', saved_img)
        iteration += 1
        return total_loss

    # Work
    p = get_params('net', net, net_input)
    optimize('LBFGS', p, closure, 0.01, args.epoch)
    print('Final PSNR: ', final_psnr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default = 1, type = int, help = 'The value of epoch')
    parser.add_argument('--noise_img', default = None, type = str, help = 'The path of noise image')
    parser.add_argument('--noise_img_gt', default = None, type = str, help = 'The path of ground truth')
    parser.add_argument('--model_path', default = './images/denoising.ckpt', type = str, help = 'The path of trained model')
    args = parser.parse_args()
    img_noisy_np, img_np = load_img(args)
    train(args, img_noisy_np, img_np)