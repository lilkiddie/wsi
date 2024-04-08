import logging
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, cdist


def cluster_inst_ids_representatives(inst_ids_to_representatives: Dict[str, List[Dict[str, int]]],
                                     max_number_senses: float,min_sense_instances:int) -> Tuple[Dict[str, Dict[str, int]], List]:
    inst_ids_ordered = list(inst_ids_to_representatives.keys())
    lemma = inst_ids_ordered[0].rsplit('.', 1)[0]
    representatives = [y for x in inst_ids_ordered for y in inst_ids_to_representatives[x]]
    n_represent = len(representatives) // len(inst_ids_ordered)
    dict_vectorizer = DictVectorizer(sparse=False)
    rep_mat = dict_vectorizer.fit_transform(representatives)
    transformed = TfidfTransformer(norm=None).fit_transform(rep_mat).todense()


    metric = 'cosine'
    method = 'average'
    dists = pdist(transformed, metric=metric)
    Z = linkage(dists, method=method, metric=metric)

    distance_crit = Z[-max_number_senses, 2]

    labels = fcluster(Z, distance_crit,
                      'distance') - 1

    n_senses = np.max(labels) + 1

    senses_n_domminates = Counter()
    instance_senses = {}
    for i, inst_id in enumerate(inst_ids_ordered):
        inst_id_clusters = Counter(labels[i * n_represent:
                                          (i + 1) * n_represent])
        instance_senses[inst_id] = inst_id_clusters
        senses_n_domminates[inst_id_clusters.most_common()[0][0]] += 1

    big_senses = [x for x in senses_n_domminates if senses_n_domminates[x] >= min_sense_instances]

    sense_means = np.zeros((n_senses, transformed.shape[1]))
    for sense_idx in range(n_senses):
        idxs_this_sense = np.where(labels == sense_idx)
        cluster_center = np.mean(np.array(transformed)[idxs_this_sense], 0)
        sense_means[sense_idx] = cluster_center

    sense_remapping = {}
    if min_sense_instances > 0:
        dists = cdist(sense_means, sense_means, metric='cosine')
        closest_senses = np.argsort(dists, )[:, ]

        for sense_idx in range(n_senses):
            for closest_sense in closest_senses[sense_idx]:
                if closest_sense in big_senses:
                    sense_remapping[sense_idx] = closest_sense
                    break
        new_order_of_senses = list(set(sense_remapping.values()))
        sense_remapping = dict((k, new_order_of_senses.index(v)) for k, v in sense_remapping.items())

        labels = np.array([sense_remapping[x] for x in labels])


    best_instance_for_sense = {}
    senses = {}
    for inst_id, inst_id_clusters in instance_senses.items():
        senses_inst = {}
        for sense_idx, count in inst_id_clusters.most_common():
            if sense_remapping:
                sense_idx = sense_remapping[sense_idx]
            senses_inst[f'{lemma}.sense.{sense_idx}'] = count
            if sense_idx not in best_instance_for_sense:
                best_instance_for_sense[sense_idx] = (count, inst_id)
            else:
                current_count, current_best_inst = best_instance_for_sense[sense_idx]
                if current_count < count:
                    best_instance_for_sense[sense_idx] = (count, inst_id)

        senses[inst_id] = senses_inst

    return senses

def kmeans_clustering(inst_ids_to_representatives):
    collection = sorted(inst_ids_to_representatives.items(), key=lambda x: int(x[0].rsplit('.', 1)[1]))
    kmeans = KMeans(n_clusters=3, n_init=5)
    kmeans.fit([x[1] for x in collection])
    return kmeans.labels_
