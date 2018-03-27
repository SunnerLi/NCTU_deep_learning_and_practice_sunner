from models.downsampler import Downsampler
from skimage.measure import compare_psnr
from torch.autograd import Variable
from skimage import io, transform
from utils.sr_utils import *
from models import *
import argparse

iteration = 0
final_psnr = 0.
psnr_HR = 0.

def load_img(args):
    """
    # Load image
    imgs = dict()
    imgs['HR_np'] = io.imread(args.gt)
    height, width, _ = np.shape(imgs['HR_np'])
    imgs['LR_np'] = transform.resize(imgs['HR_np'], (height // args.factor, width // args.factor))
    imgs['nearest_np'] = transform.resize(imgs['LR_np'], (height, width), order = 0)
    imgs['bicubic_np'] = transform.resize(imgs['LR_np'], (height, width), order = 3)
    imgs['HR_np'] = (imgs['HR_np'] / 255.)

    # Transpose as pytorch format and print log
    imgs['HR_np'] = np.transpose(imgs['HR_np'], [2, 0, 1]).astype(np.float32)
    imgs['LR_np'] = np.transpose(imgs['LR_np'], [2, 0, 1]).astype(np.float32)
    imgs['nearest_np'] = np.transpose(imgs['nearest_np'], [2, 0, 1]).astype(np.float32)
    imgs['bicubic_np'] = np.transpose(imgs['bicubic_np'], [2, 0, 1]).astype(np.float32)
    print ('PSNR bicubic: %.4f   PSNR nearest: %.4f' %  (
                    compare_psnr(imgs['HR_np'], imgs['bicubic_np']), 
                    compare_psnr(imgs['HR_np'], imgs['nearest_np'])))
    """
    imgs = load_LR_HR_imgs_sr(args.gt , -1, 4, 'CROP')
    imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])
    print ('PSNR bicubic: %.4f   PSNR nearest: %.4f' %  (
                                        compare_psnr(imgs['HR_np'], imgs['bicubic_np']), 
                                        compare_psnr(imgs['HR_np'], imgs['nearest_np'])))
    return imgs

def train(args, imgs):
    # Load model
    net = skip(
        num_input_channels = 32,
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

    # define input
    net_input = get_noise(32, 'noise', (np.shape(imgs['HR_np'])[1], np.shape(imgs['HR_np'])[2])).float().detach()
    img_LR_var = np_to_var(imgs['LR_np']).float()
    downsampler = Downsampler(n_planes=3, factor=args.factor, kernel_type='lanczos2', phase=0.5, preserve_size=True).float()
    downsampler = downsampler.cuda() if torch.cuda.is_available() else downsampler
    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()

    # Define closure
    reg_noise_std = 0.0
    tv_weight = 0.0   
    net_input = net_input.cuda() if torch.cuda.is_available() else net_input
    img_LR_var = img_LR_var.cuda() if torch.cuda.is_available() else img_LR_var
    net_input_saved = net_input_saved.cuda() if torch.cuda.is_available() else net_input_saved
    def closure():
        global iteration, psnr_HR
        if reg_noise_std > 0:
            net_input.data = net_input_saved + (torch.cuda.FloatTensor(noise.size()).normal_() * reg_noise_std)       
        out_HR = net(net_input)
        out_LR = downsampler(out_HR)
        total_loss = criterion(out_LR, img_LR_var) 
        if tv_weight > 0:
            total_loss += tv_weight * tv_loss(out_HR)
        total_loss.backward()

        # Log
        psnr_LR = compare_psnr(imgs['LR_np'], out_LR.data.cpu().numpy()[0])
        psnr_HR = compare_psnr(imgs['HR_np'], out_HR.data.cpu().numpy()[0])
        print('Iteration %05d   Loss %3f    PSNR_LR %.3f   PSNR_HR %.3f' % (iteration, total_loss.data[0], psnr_LR, psnr_HR), '\r', end = '')
        if iteration % 100 == 0:
            out_HR_np = var_to_np(out_HR)
            # plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], np.clip(out_HR_np, 0, 1)], factor=13, nrow=3)
            io.imsave('sr_' + str(iteration) + '.png', np.transpose(np.clip(out_HR_np, 0, 1), [1, 2, 0])[:, :, :3])
        iteration += 1
        return total_loss

    # Optimize (Adam with first 100 epoch and LBFGS with rest)
    p = get_params('net', net, net_input)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay = 0.0001)
    for j in range(100):
        optimizer.zero_grad()
        closure()
        optimizer.step()
    print('Starting optimization with LBFGS')        
    if args.epoch > 100:
        optimizer = torch.optim.LBFGS(net.parameters(), max_iter=args.epoch - 100, lr=0.01, tolerance_grad=-1, tolerance_change=-1)
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer.step(closure2)

    # Show final result
    out_HR_np = np.clip(var_to_np(net(net_input)), 0, 1)
    result_deep_prior = put_in_center(out_HR_np, imgs['HR_np'].shape[1:])
    plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], out_HR_np], factor=4, nrow=1)
    print('\nFinal PSNR: ', psnr_HR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default = 1, type = int, help = 'The value of epoch')
    parser.add_argument('--factor', default = 4, type = int, help = 'The factor of scale')
    parser.add_argument('--gt', default = './images/SR_GT.png', type = str, help = 'The path of ground truth')
    parser.add_argument('--model_path', default = './sr.ckpt', type = str, help = 'The path of trained model')
    args = parser.parse_args()
    imgs = load_img(args)
    train(args, imgs)