import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class dataTool:
    def __init__(self, dataset, task_name, mode, MULTI_TASKS_list=None):
        self.text_emb = []
        self.crops = []
        self.obj_names = []
        if "multi" not in task_name:
            if "processed" in task_name:
                # if real table exp
                dataset = dataset.cache
            else:
                data_path = dataset._path
                if mode == "val":
                    data_path = data_path.replace("val", "train")
                npy_path = os.path.join(data_path, "crop_database", "crop_database.npy")
                dataset = np.load(npy_path, allow_pickle=True)
            dataset_list = [dataset]
        else:
            if "processed" in task_name:
                # if real table exp
                dataset = dataset.cache
                dataset_list = [dataset]
            else:
                data_path = dataset.root_path
                task_list = MULTI_TASKS_list[task_name]["train"]
                dataset_list = []
                for subtask in task_list:
                    npy_path = os.path.join(
                        data_path,
                        subtask + "-train",
                        "crop_database",
                        "crop_database.npy",
                    )
                    dataset = np.load(npy_path, allow_pickle=True)
                    dataset_list.append(dataset)
        for dataset in dataset_list:
            for episode in dataset:
                for step in episode:
                    self.text_emb.append(step["pick_text_emb"])
                    self.crops.append(step["pick_crop"])
                    self.text_emb.append(step["place_text_emb"])
                    self.crops.append(step["place_crop"])
        #             self.obj_names.append(step['pick_obj_name'])
        #             self.obj_names.append(step['place_obj_name'])
        # self.obj_names = np.stack(self.obj_names).squeeze()
        self.text_embs = np.stack(self.text_emb).squeeze()
        self.crops = np.stack(self.crops)

    def query_crop(self, text_emb_query):
        text_emb_query = np.array(text_emb_query).reshape(1, -1)
        scores = cosine_similarity(self.text_embs, text_emb_query)
        scores = scores[:, 0]
        all_max_choices = np.argwhere(scores == np.amax(scores))
        selection = np.random.choice(all_max_choices.reshape(-1))
        # selection = np.argmax(scores)

        score = scores[selection]
        crop = self.crops[selection]
        # name = self.obj_names[selection]

        if score > 0.965:
            return crop, score
        else:
            return None, score


if __name__ == "__main__":
    tool = dataTool("data/pick-part-in-box-real.npy")
    text_emb_query_example = tool.text_embs[0]
    crop, score = tool.query_crop(text_emb_query_example)
