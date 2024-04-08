from collections import defaultdict
from utils.generator import russe_generator
from src.wsi.clustering import cluster_inst_ids_representatives, kmeans_clustering
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score


class WSI_Inductor:
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def run(self, path):
        gen = russe_generator(path + '/train.csv')
        ds_by_target = defaultdict(dict)
        for pre, target, post, inst_id in gen:
            lemma_pos = inst_id.rsplit('.', 1)[0]
            ds_by_target[lemma_pos][inst_id] = (pre, target, post)
        
        df = pd.read_csv(path + '/train.csv', sep='\t')

        agglomerative_score, kmeans_score = [], []
        for key, target in tqdm(ds_by_target.items()):
            obj = self.model.predict_substitutes(target)
            ans = cluster_inst_ids_representatives(obj, self.config.max_number_senses, self.config.min_sense_instances)
            collection = sorted(ans.items(), key=lambda x: int(x[0].rsplit('.', 1)[1]))
            labels = [int(list(item[1].keys())[0].rsplit('.', 1)[1]) + 1 for item in collection]
            agglomerative_score.append(adjusted_rand_score(df[df['word'] == key]['gold_sense_id'].values, labels))

            ans = self.model.get_sentence_embedding(target)
            labels = kmeans_clustering(ans)
            kmeans_score.append(adjusted_rand_score(df[df['word'] == key]['gold_sense_id'].values, labels))
        return np.mean(agglomerative_score), np.mean(kmeans_score)
