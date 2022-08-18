import os
import sys
import json
import pandas as pd
# sys.path.append('/apdcephfs/private_qinghonglin/video_codebase/EgoVLP/')

from base.base_dataset import TextVideoDataset
from data_loader.transforms import init_transform_dict, init_video_transform_dict

class EgoExoFeaturize(TextVideoDataset):
    def _load_metadata(self):

        ann_file = self.meta_dir

        self.metadata = pd.read_csv(ann_file)

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

        dur = 2
        start = sample[2] - dur/2 
        start = max(start, 0)
        end = start + dur 

        # fps = 1.87
        try:
            # imgs, idxs = self.video_reader(video_fp, start*30, end*30,
                                            #    (end-start) * fps * self.video_params['num_frames'])
            imgs, idxs = self.video_reader(video_fp, start*30, end*30, self.video_params['num_frames'] - 1)
        except:
            print(f"Warning: missing video file {video_fp}.")

        if self.transforms is not None:
            imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
            imgs = self.transforms(imgs)
            imgs = imgs.transpose(0, 1)  # recover

        meta_arr = {'video_uid': sample[0], 'dataset': self.dataset_name, 'idx': item}
        data = {'video': imgs, 'meta' : meta_arr}
        return data

if __name__ == "__main__":
    split = 'train'
    kwargs = dict(
        dataset_name="Ego4d_MQ",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
        },
        data_dir="dataset/ego4d_256",
        meta_dir="/datasets01/ego4d_track2/v1/annotations",
        tsfms=init_video_transform_dict()['test'],
        reader='decord_start_end',
        split=split,
    )
    dataset = MomentQueries(**kwargs)
    print(len(dataset))
    # for i in range(1000):
    #     item = dataset[i]
    #     # print(item.keys())
    #     print(item)