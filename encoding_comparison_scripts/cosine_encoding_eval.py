import pickle
import numpy as np


def cosine_similarity(arr1: np.ndarray, arr2: np.ndarray):
    arr1 = arr1.flatten()
    arr2 = arr2.flatten()
    return np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))


def main(hard_encoding_filepath, soft_encoding_filepath):

    with open(hard_encoding_filepath, "rb") as f:
        hard_encodings = pickle.load(f)
    with open(soft_encoding_filepath, "rb") as f:
        soft_encodings = pickle.load(f)

    sim_dict = dict()

    # compute overall slot sim
    for video_id in soft_encodings.keys():
        sim_dict[video_id] = dict()
        for frame_id in soft_encodings[video_id].keys():
            next_frame_id_key = str(int(frame_id) + 1).zfill(len(frame_id))
            if next_frame_id_key in soft_encodings[video_id].keys():
                sim = cosine_similarity(soft_encodings[video_id][frame_id], soft_encodings[video_id][next_frame_id_key])
                sim_dict[video_id][frame_id] = sim

    # save dict
    with open(f"/app/ncb/encoding_comparison_scripts/results/{checkpoint_name}/similarities_{min_video}-{max_video}.pkl", "wb") as f:
    # with open(f"/app/ncb/encoding_comparison_scripts/results/default/similarities_{min_video}-{max_video}.pkl", "wb") as f:
        pickle.dump(sim_dict, f)

    
    # compute slot wise sim
    slot_sim_dict = dict()
    for video_id in soft_encodings.keys():
        slot_sim_dict[video_id] = dict()
        for frame_id in soft_encodings[video_id].keys():
            next_frame_id_key = str(int(frame_id) + 1).zfill(len(frame_id))
            if next_frame_id_key in soft_encodings[video_id].keys():
                slot_sim_dict[video_id][frame_id] = dict()
                se = soft_encodings[video_id][frame_id]
                next_se = soft_encodings[video_id][next_frame_id_key]
                for i in range(se.shape[0]):
                    sim = cosine_similarity(se[i], next_se[i])
                    slot_sim_dict[video_id][frame_id][i] = sim

    # save dict
    with open(f"/app/ncb/encoding_comparison_scripts/results/{checkpoint_name}/slot_similarities_{min_video}-{max_video}.pkl", "wb") as f:
    # with open(f"/app/ncb/encoding_comparison_scripts/results/default/similarities_{min_video}-{max_video}.pkl", "wb") as f:
        pickle.dump(slot_sim_dict, f)


    # compute sim to mean representation
    mean_sim_dict = dict()
    for video_id in soft_encodings.keys():
        mean_sim_dict[video_id] = dict()
        mean_video_encoding = np.mean([soft_code.flatten() for soft_code in soft_encodings[video_id].values()], axis=0)

        for frame_id in soft_encodings[video_id].keys():
            sim = cosine_similarity(soft_encodings[video_id][frame_id], mean_video_encoding)
            mean_sim_dict[video_id][frame_id] = sim

    # save dict
    with open(f"/app/ncb/encoding_comparison_scripts/results/{checkpoint_name}/mean_similarities_{min_video}-{max_video}.pkl", "wb") as f:
    # with open(f"/app/ncb/encoding_comparison_scripts/results/default/similarities_{min_video}-{max_video}.pkl", "wb") as f:
        pickle.dump(mean_sim_dict, f)



if __name__ == "__main__":
    min_video, max_video = 10000, 10099
    checkpoint_name = "finetuning_0.1_0.8"
    hard_encoding_filepath = f"/app/ncb/encoding_comparison_scripts/results/{checkpoint_name}/val_hard_encodings_{min_video}-{max_video}.pkl"
    soft_encoding_filepath = f"/app/ncb/encoding_comparison_scripts/results/{checkpoint_name}/val_soft_encodings_{min_video}-{max_video}.pkl"
    main(hard_encoding_filepath, soft_encoding_filepath)
