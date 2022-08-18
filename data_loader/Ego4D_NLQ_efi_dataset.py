import os
import pdb
import sys
import json
import pandas as pd
sys.path.append('/apdcephfs/private_qinghonglin/video_codebase/EgoVLP/')

from base.base_dataset import TextVideoDataset
from data_loader.transforms import init_transform_dict, init_video_transform_dict

class NaturalLanguageQueriesEfi(TextVideoDataset):
    def _load_metadata(self):

        self.metadata = pd.read_csv(self.meta_dir)

        done_vids = os.listdir('/checkpoint/afourast/data/ego4d/nlq/egoclip_features_train_val_2_per_sec/embeddings_768d')
        done_vids = set([vid.replace('.pt','') for vid in done_vids])

        self.metadata = self.metadata[~self.metadata.video_uids.isin(done_vids)]

        # import ipdb; ipdb.set_trace(context=20)

        self.transforms = init_video_transform_dict()['test']

    def _get_video_path(self, sample):
        rel_video_fp = sample[0]
        full_video_fp = os.path.join(self.data_dir, rel_video_fp + '.mp4')
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        caption = sample['query']
        return caption

    def __getitem__(self, item):
        sample = self.metadata.iloc [item]
        video_fp, rel_fp = self._get_video_path(sample)

        feats_per_sec = 2
        fps = 30 / (feats_per_sec * self.video_params['num_frames']) # e.g. 30 / (2*4) = 3.75 
        try:
            # imgs, idxs = self.video_reader(video_fp, sample[2]*30, sample[3]*30,
            #                                    (sample[3]-sample[2]) * fps * self.video_params['num_frames'])
            imgs, idxs = self.video_reader(video_fp, fps = fps )
        except:
            print(f"Warning: missing video file {video_fp}.")

        if self.transforms is not None:
            imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
            imgs = self.transforms(imgs)
            imgs = imgs.transpose(0, 1)  # recover

        meta_arr = {'video_uid': sample[0],'dataset': self.dataset_name}
        data = {'video': imgs, 'meta' : meta_arr}
        return data

if __name__ == "__main__":
    split = 'val'
    kwargs = dict(
        dataset_name="Ego4d_NLQ",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
        },
        data_dir="/apdcephfs/private_qinghonglin/video_dataset/ego4d_256/data",
        meta_dir="/apdcephfs/private_qinghonglin/video_dataset/ego4d/benchmark_splits/nlq/",
        tsfms=init_video_transform_dict()['test'],
        # reader='decord_start_end',
        reader='decord',
        subsample='text',
        split=split,
    )
    dataset = NaturalLanguageQueries(**kwargs)
    print(len(dataset))
    # for i in range(1000):
    #     item = dataset[i]
    #     # print(item.keys())
    #     print(item)