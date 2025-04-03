import pickle
import numpy as np


def jaccard_similarity(arr1: np.ndarray, arr2: np.ndarray):
    arr1 = arr1.flatten()
    arr2 = arr2.flatten()
    return np.sum(arr1 == arr2) / arr1.size   # approximate union by len of array


def main(hard_encoding_filepath, soft_encoding_filepath):

    with open(hard_encoding_filepath, "rb") as f:
        hard_encodings = pickle.load(f)
    with open(soft_encoding_filepath, "rb") as f:
        soft_encodings = pickle.load(f)

    sim_dict = dict()

    # compute overall slot sim
    for video_id in hard_encodings.keys():
        sim_dict[video_id] = dict()
        for frame_id in hard_encodings[video_id].keys():
            next_frame_id_key = str(int(frame_id) + 1).zfill(len(frame_id))
            if next_frame_id_key in hard_encodings[video_id].keys():
                sim = jaccard_similarity(hard_encodings[video_id][frame_id], hard_encodings[video_id][next_frame_id_key])
                sim_dict[video_id][frame_id] = sim

    # save dict
    with open(f"/app/ncb/encoding_comparison_scripts/results/{checkpoint_name}/jac_similarities_{min_video}-{max_video}.pkl", "wb") as f:
    # with open(f"/app/ncb/encoding_comparison_scripts/results/default/similarities_{min_video}-{max_video}.pkl", "wb") as f:
        pickle.dump(sim_dict, f)

    
    # compute slot wise sim
    slot_sim_dict = dict()
    for video_id in hard_encodings.keys():
        slot_sim_dict[video_id] = dict()
        for frame_id in hard_encodings[video_id].keys():
            next_frame_id_key = str(int(frame_id) + 1).zfill(len(frame_id))
            if next_frame_id_key in hard_encodings[video_id].keys():
                slot_sim_dict[video_id][frame_id] = dict()
                se = hard_encodings[video_id][frame_id]
                next_se = hard_encodings[video_id][next_frame_id_key]
                for i in range(se.shape[0]):
                    sim = jaccard_similarity(se[i], next_se[i])
                    slot_sim_dict[video_id][frame_id][i] = sim

    # save dict
    with open(f"/app/ncb/encoding_comparison_scripts/results/{checkpoint_name}/slot_jac_similarities_{min_video}-{max_video}.pkl", "wb") as f:
    # with open(f"/app/ncb/encoding_comparison_scripts/results/default/similarities_{min_video}-{max_video}.pkl", "wb") as f:
        pickle.dump(slot_sim_dict, f)


    # compute sim to mean representation
    mean_sim_dict = dict()
    for video_id in hard_encodings.keys():
        mean_sim_dict[video_id] = dict()
        mean_video_encoding = np.rint(np.mean([hard_code.flatten() for hard_code in hard_encodings[video_id].values()], axis=0)).astype(int)

        for frame_id in hard_encodings[video_id].keys():
            sim = jaccard_similarity(hard_encodings[video_id][frame_id], mean_video_encoding)
            mean_sim_dict[video_id][frame_id] = sim

    # save dict
    with open(f"/app/ncb/encoding_comparison_scripts/results/{checkpoint_name}/mean_jac_similarities_{min_video}-{max_video}.pkl", "wb") as f:
    # with open(f"/app/ncb/encoding_comparison_scripts/results/default/similarities_{min_video}-{max_video}.pkl", "wb") as f:
        pickle.dump(mean_sim_dict, f)


if __name__ == "__main__":
    # min_video, max_video = 10000, 10099
    # checkpoint_name = "finetuning_0.1_0.8"
    min_video, max_video = 0, 7
    checkpoint_name = "memory_v2_custom_complex_CLEVR"
    hard_encoding_filepath = f"/app/ncb/encoding_comparison_scripts/results/{checkpoint_name}/val_hard_encodings_{min_video}-{max_video}.pkl"
    soft_encoding_filepath = f"/app/ncb/encoding_comparison_scripts/results/{checkpoint_name}/val_soft_encodings_{min_video}-{max_video}.pkl"
    main(hard_encoding_filepath, soft_encoding_filepath)
