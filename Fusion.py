 #encoding=utf8
import torch
import argparse
from Physiological_Signal_Encoder.CNxF.utils import *
from torch.utils.data import DataLoader
from Physiological_Signal_Encoder.CNxF import train
from Physiological_Signal_Encoder.CNxF import test


parameter = argparse.ArgumentParser(description='CNxF Emotion Recognition')
parameter.add_argument('-f', default='', type=str)

# Model Name
parameter.add_argument('--model', type=str, default='CNxF',
                    help='name of the model to use (default: CNxF)')

# Tasks
parameter.add_argument('--dataset', type=str, default='DEAP',
                    help='dataset to use (default: WESAD)')
parameter.add_argument('--data_path', type=str, default='A_Origin_Multimodal_Data',
                    help='path for storing the dataset')

# Model Parameter
parameter.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 1024)') 
parameter.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parameter.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-4)')
parameter.add_argument('--optim', type=str, default='AdamW',
                    help='optimizer to use (default: AdamW)')
parameter.add_argument('--num_epochs', type=int, default=1,
                    help='number of epochs (default: 20)')
parameter.add_argument('--when', type=int, default=10000,
                    help='when to decay learning rate (default: 5)')
parameter.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')
parameter.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parameter.add_argument('--seed', type=int, default=1111,
                    help='random seed')

#ConvNext
parameter.add_argument('--weight_decay', type=float, default=0.05,
                      help='Weight_decay (default: 0.05)')
parameter.add_argument('--layer_scale_init_value', type=float, default=1e-6,
                      help='layer_scale_init_value (default: 1e-6)')
parameter.add_argument('--head_init_scale', type=float, default=1.,
                      help='head_init_scale (default: 1.)')

#Other
parameter.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parameter.add_argument('--name', type=str, default='cnxf',
                    help='name of the trial (default: "cnxf")')
parameter.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
args = parameter.parse_args()
 
torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
use_cuda = False

output_dim_dict = {
    'CNxF': 1
}

criterion_dict = {
    'CNxF': 'CrossEntropyLoss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("no cuda so using cpu lol")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True
        
print("Welcome to the journey of CNxF, I'm no one but a student of AHU (-^-)")

print("Loading Data....")

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')
   

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda:0'))
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda:0'))
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device='cuda:0'))

print('Finish!')

hyp_params = args
hyp_params.orig_d_m1, hyp_params.orig_d_m2, hyp_params.orig_d_m3,hyp_params.orig_d_m4 = train_data.get_dim()
hyp_params.m1_len, hyp_params.m2_len, hyp_params.m3_len, hyp_params.m4_len = train_data.get_seq_len()
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')

hyp_params.weight_decay = args.weight_decay
hyp_params.layer_scale_init_value = args.layer_scale_init_value
hyp_params.head_init_scale = args.head_init_scale

if __name__ == '__main__':
    if args.eval:
        test = test.eval(hyp_params, test_loader)
    else:
        test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)
