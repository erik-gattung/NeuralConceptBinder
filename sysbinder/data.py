from collections import defaultdict
from copy import deepcopy
import glob
from torchvision import transforms
from torch.utils.data import Dataset, Sampler
import random
import numpy as np

from PIL import Image


class GlobDataset(Dataset):
    def __init__(self, root, phase, img_height, img_width, recursive=False):
        self.root = root
        # self.img_size = img_size
        self.img_height = img_height
        self.img_width = img_width
        self.total_imgs = sorted(glob.glob(root, recursive=recursive))

        if phase == 'train':
            self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * 0.7)]
        elif phase == 'val':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.7):int(len(self.total_imgs) * 0.85)]
        elif phase == 'test':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.85):]
        elif isinstance(phase, int):
            self.total_imgs = np.array(self.total_imgs)[np.random.choice(len(self.total_imgs), size=phase, replace=False)].tolist()
        else:
            pass

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        # image = image.resize((self.img_size, self.img_size))
        image = image.resize((self.img_width, self.img_height))
        tensor_image = self.transform(image)

        # get the basename of the file without the file ending
        filename = img_loc.split("/")[-1].split(".")[0]

        return tensor_image, filename
    

class ContrastiveBatchSampler(Sampler):
    def __init__(self, data_source: GlobDataset, batch_size):
        """
        Args:
            data_source (Dataset): The dataset.
            batch_size (int): The desired batch size.
        """
        self.data_source = data_source
        self.batch_size = batch_size

        self.video_ids = np.unique([data_source.total_imgs[i].split("/")[-1].split("_")[1] for i in range(len(data_source))]).tolist()
        # Create lists of indices for frames of each video
        self.video_indices = defaultdict(list)
        for i, filename in enumerate(self.data_source.total_imgs):
            vid = filename.split("/")[-1].split("_")[1]
            self.video_indices[vid].append(i)

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):     
        available_videos = deepcopy(self.video_ids)
        available_frames = deepcopy(self.video_indices)

        # We will generate batches until no videos are left
        while len(available_videos) >= 2:
            vid1, vid2 = random.sample(available_videos, 2)  # Select two different video ids

            # Get available indices for these two videos
            available_vid1 = available_frames[vid1]
            available_vid2 = available_frames[vid2]

            half_batch = self.batch_size // 2

            selected_indices1 = random.sample(range(len(available_vid1)), min(half_batch, len(available_vid1)))
            selected_indices2 = random.sample(range(len(available_vid2)), min(half_batch, len(available_vid2)))
            selected_vid1 = [available_vid1.pop(idx) for idx in sorted(selected_indices1, reverse=True)]
            selected_vid2 = [available_vid2.pop(idx) for idx in sorted(selected_indices2, reverse=True)]

            # If one of the videos doesn't have enough remaining images, we fill up from other videos
            if len(selected_vid1) < half_batch or len(selected_vid2) < half_batch:
                while len(selected_vid1) < half_batch and len(available_videos) >= 3:
                    other_vids = available_videos.copy()
                    other_vids.remove(vid1)
                    other_vids.remove(vid2)
                    other_vid = random.choice(other_vids)
                    other_frame_index = random.choice(range(len(available_frames[other_vid])))
                    selected_vid1.append(available_frames[other_vid].pop(other_frame_index))
                    if len(available_frames[other_vid]) == 0:
                        available_videos.remove(other_vid)
                while len(selected_vid2) < half_batch and len(available_videos) >= 3:
                    other_vids = available_videos.copy()
                    other_vids.remove(vid1)
                    other_vids.remove(vid2)
                    other_vid = random.choice(other_vids)
                    other_frame_index = random.choice(range(len(available_frames[other_vid])))
                    selected_vid2.append(available_frames[other_vid].pop(other_frame_index))
                    if len(available_frames[other_vid]) == 0:
                        available_videos.remove(other_vid)

            # Add the selected indices to the batch
            batch_indices = selected_vid1 + selected_vid2
            for idx in batch_indices:
                yield idx   # yield indices one by one and let DataLoader do the batching lol

            # If any video has no more images left, remove it from available_videos
            if len(available_frames[vid1]) == 0:
                available_videos.remove(vid1)
            if len(available_frames[vid2]) == 0:
                available_videos.remove(vid2)
