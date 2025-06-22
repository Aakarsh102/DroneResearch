from datasets import load_dataset, DatasetDict, load_from_disk
# from huggingface_hub import snapshot_download

# local_dir = snapshot_download(
#     repo_id="bairuqi/StarCraft-Motion",
#     repo_type="dataset",         # <-- critical!
#     local_dir="StarCraft-Motion",
#     local_dir_use_symlinks=False
# )

# print("Downloaded to:", local_dir)

ds = load_dataset(
    path="StarCraft-Motion",
    data_dir=".",
    split="train"
)
print(ds[0])
print(ds[0].keys())  # Print the keys of the first item in the dataset
print(ds[1])  # Print the shape of the video tensor


# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("bairuqi/StarCraft-Motion")

# s1 = ds["train"].train_test_split(test_size=0.2)  # Split the training set into train and test sets
# s2 = s1["test"].train_test_split(test_size=0.5)  # Further split the test set into validation and test sets
# final_ds = DatasetDict({
#     "train":      s1["train"],
#     "validation": s2["train"],
#     "test":       s2["test"]
# })
# final_ds.save_to_disk("StarCraft-Motion-split")
# final_ds = load_from_disk("StarCraft-Motion-split")


# train_split = final_ds["train"]
# # print(train_split.features)  # Print the available splits in the dataset
# print(train_split[0])  # Print the first item in the training set
# print(type(train_split))
# print("*****")
# print(train_split[0].keys())  # Print the length of the training set


# print(type(ds))

# # train_split = ds["train"]
# # validation_split = ds["validation"]
# # test_split = ds["test"]
# print(ds.keys())  # Print the available splits in the dataset
# print(ds["train"][0])  # Print the first item in the training set
# print(ds["train"].features)  # Print the features of the training set
# ds.train_test_split(test_size=0.1)  # Split the training set into train and test sets


# print(type(train_split))
# print(len(test_split))
# print(test_split[0])
