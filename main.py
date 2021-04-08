import os
import argparse

from SNP_pytorch import SNP
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from data_generator import GPCurvesReader
from tqdm import tqdm
from plotting import plot_functions_1d

# filter warnings
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning)

log_dir = os.path.join('logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Arguments potentially from shell
parser = argparse.ArgumentParser(description='SNP for regression tasks')
# models
parser.add_argument('--HIDDEN_SIZE', type=int, default=128, metavar='N',
                    help='hidden unit size of network')
parser.add_argument('--beta', type=float, default=1.0, metavar='N',
                    help='weight to kl loss term')
# dataset
parser.add_argument('--task_type', type=int, default=1, metavar='N',
                    help="three types, {1|2|3}", choices=[1,2,3])
parser.add_argument('--MAX_CXT_PNTS', type=int, default=500, metavar='N',
                    help='max context size at each time-steps')
parser.add_argument('--LEN_SEQ', type=int, default=20, metavar='N',
                    help='sequence length')
parser.add_argument('--LEN_GIVEN', type=int, default=10, metavar='N',
                    help='given context length')
parser.add_argument('--LEN_GEN', type=int, default=10, metavar='N',
                    help='generalization test sequence length')
# gaussian process hyperparams
parser.add_argument('--l1_min', type=float, default=0.7, metavar='N',
                    help='l1 initial boundary')
parser.add_argument('--l1_max', type=float, default=1.2, metavar='N',
                    help='l1 initial boundary')
parser.add_argument('--l1_vel', type=float, default=0.03, metavar='N',
                    help='l1 kernel parameter dynamics')
parser.add_argument('--sigma_min', type=float, default=1.0, metavar='N',
                    help='sigma initial boundary')
parser.add_argument('--sigma_max', type=float, default=1.6, metavar='N',
                    help='sigma initial boundary')
parser.add_argument('--sigma_vel', type=float, default=0.05, metavar='N',
                    help='sigma kernel parameter dynamics')
# training
parser.add_argument('--TRAINING_ITERATIONS', type=int, default=1000000, metavar='N',
                    help='training iteration')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size')
parser.add_argument('--PLOT_AFTER', type=int, default=1000, metavar='N',
                    help='plot iteration')
parser.add_argument('--log_folder', type=str, default='logs', metavar='N',
                    help='log folder')
parser.add_argument('--resume', type=bool, default=False, metavar='N',
                    help='log folder')
# other
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# torch.manual_seed(args.seed)
# random.seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

latent_encoder_output_sizes = [args.HIDDEN_SIZE]*4
num_latents = args.HIDDEN_SIZE
deterministic_encoder_output_sizes= [args.HIDDEN_SIZE]*4
decoder_output_sizes = [args.HIDDEN_SIZE]*2 + [2]
representation_size = args.HIDDEN_SIZE * 2

setattr(args, 'latent_encoder_output_sizes', latent_encoder_output_sizes)
setattr(args, 'num_latents', num_latents)
setattr(args, 'deterministic_encoder_output_sizes', deterministic_encoder_output_sizes)
setattr(args, 'decoder_output_sizes', decoder_output_sizes)
setattr(args, 'representation_size', representation_size)

dataset_train = GPCurvesReader(
    batch_size=args.batch_size, max_num_cxt=args.MAX_CXT_PNTS,
    len_seq=args.LEN_SEQ, len_given=args.LEN_GIVEN, len_gen=args.LEN_GEN,
    l1_min=args.l1_min, l1_max=args.l1_max, l1_vel=args.l1_vel,
    sigma_min=args.sigma_min, sigma_max=args.sigma_max, sigma_vel=args.sigma_vel,
    task_type=args.task_type
)
# data_train = dataset_train.generate_temporal_curves()

dataset_test = GPCurvesReader(
        batch_size=args.batch_size, max_num_cxt=args.MAX_CXT_PNTS,
        len_seq=args.LEN_SEQ, len_given=args.LEN_GIVEN, len_gen=args.LEN_GEN,
        l1_min=args.l1_min, l1_max=args.l1_max, l1_vel=args.l1_vel,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max, sigma_vel=args.sigma_vel,
        testing=True, task_type=args.task_type
    )
data_test = dataset_test.generate_temporal_curves(seed=None)

model = SNP(args).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# generate training dataset: 100 batches with 16 training_examples / batch
data_train = []
for i in range(100):
    data_train.append(dataset_train.generate_temporal_curves()) 

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx in range(100):
        query, trgt_y, num_tot_pnts, num_cxt_pnts, _ = data_train[batch_idx]
        (cxt_x, cxt_y), trgt_x = query
        new_cxt_x, new_cxt_y, new_trgt_x, new_trgt_y = [], [], [], []
        for ta, tb, tc, td in zip(cxt_x, cxt_y, trgt_x, trgt_y):
            ta, tb, tc, td = ta.to(device), tb.to(device), tc.to(device), td.to(device)
            new_cxt_x.append(ta)
            new_cxt_y.append(tb)
            new_trgt_x.append(tc)
            new_trgt_y.append(td)

        query = ((new_cxt_x, new_cxt_y), new_trgt_x)
        trgt_y = new_trgt_y

        optimizer.zero_grad()
        trgt_dist_list, trgt_mu_n_sigma, prior_dist, post_dist = model(query, trgt_y, num_tot_pnts, num_cxt_pnts)

        loss = model.snp_loss(trgt_dist_list, trgt_mu_n_sigma, prior_dist, post_dist, args.LEN_SEQ, trgt_y)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        writer.add_scalar('Loss/train', loss.item(), batch_idx)

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * 16, 1600,
                    100. * batch_idx / 100,
                    loss.item() / 16))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / 1600))

def test(epoch):
    model.eval()

    test_loss = 0
    with torch.no_grad():
        query, trgt_y, num_tot_pnts, num_cxt_pnts, _ = data_test
        (cxt_x, cxt_y), trgt_x = query
        new_cxt_x, new_cxt_y, new_trgt_x, new_trgt_y = [], [], [], []
        for ta, tb, tc, td in zip(cxt_x, cxt_y, trgt_x, trgt_y):
            ta, tb, tc, td = ta.to(device), tb.to(device), tc.to(device), td.to(device)
            new_cxt_x.append(ta)
            new_cxt_y.append(tb)
            new_trgt_x.append(tc)
            new_trgt_y.append(td)

        query = ((new_cxt_x, new_cxt_y), new_trgt_x)
        trgt_y = new_trgt_y

        optimizer.zero_grad()
        trgt_dist_list, trgt_mu_n_sigma, prior_dist, post_dist = model(query, trgt_y, num_tot_pnts, num_cxt_pnts)

        loss = model.snp_loss(trgt_dist_list, trgt_mu_n_sigma, prior_dist, post_dist, args.LEN_SEQ+args.LEN_GEN, trgt_y)
        test_loss += loss.item()
        
        log_p = model.get_debug_metrics(trgt_dist_list, trgt_mu_n_sigma, prior_dist, post_dist, args.LEN_SEQ+args.LEN_GEN, trgt_y, num_cxt_pnts )
        
        writer.add_scalar('Loss/test', loss.item(), epoch)
        writer.add_scalar('TargetNLL/Nll', log_p, epoch)

    if epoch%1000 == 0:
        plot_data = (trgt_x, trgt_y, cxt_x, cxt_y, trgt_mu_n_sigma["mu"], trgt_mu_n_sigma["sigma"])
        plot_functions_1d(args.LEN_SEQ, args.LEN_GIVEN, args.LEN_GEN, log_dir, plot_data)

    test_loss /= 16
    print('====> Test set loss: {:.4f}'.format(test_loss))


writer = SummaryWriter()

start_epoch = 0
if not os.path.isdir("ckpnt"):
    os.mkdir("ckpnt")

if args.resume:
    assert os.path.isfile('ckpnt/ckpnt_file.pth'), "checkpoint not found"
    ckpnt = torch.load('ckpnt/ckpnt_file.pth')
    start_epoch = ckpnt['epoch']+1
    model.load_state_dict(ckpnt['model'])
    optimizer.load_state_dict(ckpnt['optimizer'])


for epoch in range(start_epoch, int(args.TRAINING_ITERATIONS / 100)):
    if epoch % 10 == 0:
        ckpnt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(ckpnt, './ckpnt/ckpnt_file.pth')

    train(epoch)
    test(epoch)


def plot_gen_curve(task_type):
    '''
    # plot generated 1D regression curve after training
    '''
    assert task_type in {'a', 'b', 'c'}
    path  = 'ckpnt/task_'+task_type+'/ckpnt_file.pth'
    ckpnt = torch.load(path)
    model.load_state_dict(ckpnt['model'])
    optimizer.load_state_dict(ckpnt['optimizer'])


    model.eval()

    test_loss = 0
    with torch.no_grad():
        query, trgt_y, num_tot_pnts, num_cxt_pnts, _ = data_test
        (cxt_x, cxt_y), trgt_x = query
        new_cxt_x, new_cxt_y, new_trgt_x, new_trgt_y = [], [], [], []
        for ta, tb, tc, td in zip(cxt_x, cxt_y, trgt_x, trgt_y):
            ta, tb, tc, td = ta.to(device), tb.to(device), tc.to(device), td.to(device)
            new_cxt_x.append(ta)
            new_cxt_y.append(tb)
            new_trgt_x.append(tc)
            new_trgt_y.append(td)

        query = ((new_cxt_x, new_cxt_y), new_trgt_x)
        trgt_y = new_trgt_y

        optimizer.zero_grad()
        trgt_dist_list, trgt_mu_n_sigma, prior_dist, post_dist = model(query, trgt_y, num_tot_pnts, num_cxt_pnts)
        
        log_p = model.get_debug_metrics(trgt_dist_list, trgt_mu_n_sigma, prior_dist, post_dist, args.LEN_SEQ+args.LEN_GEN, trgt_y, num_cxt_pnts )

        plot_data = (trgt_x, trgt_y, cxt_x, cxt_y, trgt_mu_n_sigma["mu"], trgt_mu_n_sigma["sigma"])
        plot_functions_1d(args.LEN_SEQ, args.LEN_GIVEN, args.LEN_GEN, log_dir, plot_data)

plot_gen_curve(args.task_type)
