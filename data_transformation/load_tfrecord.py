import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pickle
from tqdm import tqdm
import os
import dlimp as dl
from traj_transforms import STANDARDIZATION_TRANSFORMS
from np_processes import STANDARDIZATION_PROCESSES


dataset_dir = "../raw_data"
subdataset = "libero/libero_object_no_noops/1.0.0"
dataset_path = os.path.join(dataset_dir, subdataset)
print("dataset_path", dataset_path)
builder = tfds.builder_from_directory(dataset_path)

dataset = dl.dataset.DLataset.from_rlds(builder, shuffle=False)


print(builder.info)

target_dir = "../data_transformation/dataset"
subdataset_list = subdataset.split("/")
subdataset_name = subdataset_list[1]
subdataset_list[0] = subdataset_list[0] + "_pkl"
subdataset = "/".join(subdataset_list)
target_path = os.path.join(target_dir, subdataset)
print("target_path", target_path)
os.makedirs(target_path, exist_ok=True)

# use enumerate to get the index of the sample
for i, traj in tqdm(enumerate(dataset), total=len(dataset)):
    # print(sample)
    transformed_traj = STANDARDIZATION_TRANSFORMS[subdataset_name](traj)
    proprio_state, thrid_view_img, wrist_img, lang_instruction, action = STANDARDIZATION_PROCESSES[subdataset_name](transformed_traj)
    # create a dict to store the data
    data_dict = {
        "proprio_state": proprio_state,
        "thrid_view_img": thrid_view_img,
        "wrist_img": wrist_img,
        "lang_instruction": lang_instruction,
        "action": action,
    }

    # save the data
    with open(os.path.join(target_path, f"traj{i}.pkl"), "wb") as f:
        pickle.dump(data_dict, f)