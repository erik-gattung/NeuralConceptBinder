import pickle
import numpy as np


def cosine_similarity(arr1: np.ndarray, arr2: np.ndarray):
    arr1 = arr1.flatten()
    arr2 = arr2.flatten()
    return np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))

def get_frame_segment_index(video_frame_segment_indices, image_index):
    image_index = int(image_index)
    for i, segment in enumerate(video_frame_segment_indices):
        if image_index in segment:
            return i
    return len(video_frame_segment_indices) # should cause an Index Error later

def main(hard_encoding_filepath, soft_encoding_filepath):

    with open(hard_encoding_filepath, "rb") as f:
        hard_encodings = pickle.load(f)
    with open(soft_encoding_filepath, "rb") as f:
        soft_encodings = pickle.load(f)

    video_shift_frame_index = [0, 16, 24, 32, 48]
    video_frame_segment_indices = [range(video_shift_frame_index[i], video_shift_frame_index[i+1]) for i in range(len(video_shift_frame_index) - 1)]
    
    # compute sim to mean representation
    mean_sim_dict = dict()
    for video_id in soft_encodings.keys():
        mean_sim_dict[video_id] = dict()
        # mean_video_encoding = np.mean([soft_code.flatten() for soft_code in soft_encodings[video_id].values()], axis=0)
        mean_video_encodings = [np.mean([soft_code.flatten() for frame_id, soft_code in soft_encodings[video_id].items() if int(frame_id) in segment_indices], axis=0) for segment_indices in video_frame_segment_indices]

        for frame_id in soft_encodings[video_id].keys():
            # segment_index = get_frame_segment_index(video_frame_segment_indices, frame_id)
            # sim = cosine_similarity(soft_encodings[video_id][frame_id], mean_video_encodings[segment_index])
            sim = [cosine_similarity(soft_encodings[video_id][frame_id], mve) for mve in mean_video_encodings]
            mean_sim_dict[video_id][frame_id] = sim

    # save dict
    with open(f"/app/ncb/encoding_comparison_scripts/results/{checkpoint_name}/mean_segment_similarities_{min_video}-{max_video}.pkl", "wb") as f:
    # with open(f"/app/ncb/encoding_comparison_scripts/results/default/similarities_{min_video}-{max_video}.pkl", "wb") as f:
        pickle.dump(mean_sim_dict, f)


if __name__ == "__main__":
    # min_video, max_video = 10000, 10099
    # min_video, max_video = 10000, 10000
    min_video, max_video = 0, 7
    # checkpoint_name = "finetuning_0.1_0.8"
    # checkpoint_name = "SAM"
    checkpoint_name = "memory_v2_custom_complex_CLEVR"
    hard_encoding_filepath = f"/app/ncb/encoding_comparison_scripts/results/{checkpoint_name}/val_hard_encodings_{min_video}-{max_video}.pkl"
    soft_encoding_filepath = f"/app/ncb/encoding_comparison_scripts/results/{checkpoint_name}/val_soft_encodings_{min_video}-{max_video}.pkl"
    main(hard_encoding_filepath, soft_encoding_filepath)
