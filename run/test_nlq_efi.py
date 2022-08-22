import os
import sys
import tqdm
import argparse
import numpy as np
import transformers
from sacred import Experiment

import torch
import model.metric as module_metric
import data_loader.data_loader as module_data
from utils import state_dict_data_parallel_fix
from parse_config import ConfigParser

# ex = Experiment('test')

# @ex.main
def main(args ,config):
    # setup data_loader instances
    config._config['data_loader']['type'] = 'TextVideoDataLoader'
    config._config['data_loader']['args']['split'] = args.split
    config._config['data_loader']['args']['tsfm_split'] = 'test'  # set transform to test split to remove augmentations
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['batch_size'] = args.batch_size
    config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride

    # world_size = int(os.environ["SLURM_NPROCS"])
    # rank = int(os.environ["SLURM_PROCID"])
    config._config['data_loader']['args']['n_jobs'] = args.n_jobs
    config._config['data_loader']['args']['job_id'] = args.job_id
    # if "SLURM_ARRAY_TASK_ID" in os.environ:
    #     rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
    #     config._config['data_loader']['args']['job_id'] = rank


    save_feats = config._config['data_loader']['args']['save_feats_dir']

    data_loader = config.initialize('data_loader', module_data)

    # build model architecture
    import model.model as module_arch
    model = config.initialize('arch', module_arch)

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))

    if os.path.exists(config.resume):
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=True)
    else:
        print('Using random weights')

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(len(data_loader))

    if not os.path.exists(save_feats):
        os.mkdir(save_feats)

    num_frame = config.config['data_loader']['args']['video_params']['num_frames']
    emb_dim = 768
    proj_dim = config.config['arch']['args']['projection_dim']
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            # leave this for now since not doing anything on the gpu
            # pdb.set_trace()

            # if os.path.exists(os.path.join(args.save_feats, data['meta']['clip_uid'][0]+'.pt')):
            #     print(f"{data['meta']['clip_uid']} is already.")
            #     continue

            # this implementation is cautious, we use 4f video-encoder to extract featurs of whole clip.
            f, c, h, w = data['video'].shape[1], data['video'].shape[2], data['video'].shape[3], data['video'].shape[4]

            data['video'] = data['video'][0][:(f // num_frame * num_frame), ]
            data['video'] = data['video'].reshape(-1, num_frame, c, h, w)

            # data['video'] = data['video'].to(device)
            outs_emb = torch.zeros(data['video'].shape[0], emb_dim)
            outs_proj = torch.zeros(data['video'].shape[0], proj_dim)

            batch = 16
            times = data['video'].shape[0] // batch + ( 1 if (data['video'].shape[0] % batch) else 0 )
            for j in tqdm.tqdm(range(times)):
                start = j*batch
                if (j+1) * batch > data['video'].shape[0]:
                    end = data['video'].shape[0]
                else:
                    end = (j+1)*batch

                video_embd, video_proj = model.compute_video_embeddings_and_projection(data['video'][start:end,].to(device))

                outs_emb[start:end,] = video_embd
                outs_proj[start:end,] = video_proj

            video_uid = data['meta']['video_uid'][0]

            os.makedirs( os.path.join(save_feats, 'embeddings_768d' ), exist_ok=True )
            os.makedirs( os.path.join(save_feats, 'projections_256d' ), exist_ok=True )

            torch.save(outs_emb, save_feats+f'/embeddings_768d/{video_uid}.pt')
            torch.save(outs_proj, save_feats+f'/projections_256d/{video_uid}.pt')

            # video_embd  = video_embd.cpu().numpy()
            # video_proj  = video_proj.cpu().numpy()
            # all_video_embd.append(video_embd)
            # all_video_proj.append(video_proj)
            # all_video_idx.append(data['meta']['idx'])

def parse_args():
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume',
                      default='/private/home/afourast/EgoVLP/egovlp.pth',
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('-subsample', '--subsample', default='text', type=str, # 0 for vidoe while 1 for text.
                      help='source data from video or text.')
    args.add_argument('--token', default=False, type=bool,
                      help='whether use token features to represent sentence.')
    args.add_argument('--split', default='train', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=1, type=int,
                      help='size of batch')
    args.add_argument('-gpu', '--gpu', default=0, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--n_jobs', default=1, type=int,)
    args.add_argument('--job_id', default=0, type=int,)
    # hack to get sliding into config
    config = ConfigParser(args, test=True, eval_mode='efi_nlq')

    args = args.parse_args()

    args.checkpoint_dir = config._config['data_loader']['args']['save_feats_dir']

    config._config['sliding_window_stride'] = args.sliding_window_stride
    return args, config

if __name__ == '__main__':
    args, config = parse_args()
    # ex.add_config(config.config)

    os.environ["CUDA_VISIBLE_DEVICES"] =  ""+str(args.gpu)
    main(args, config)