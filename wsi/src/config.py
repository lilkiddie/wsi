from collections import namedtuple

WSISettings = namedtuple('WSISettings', ['n_represents', 'n_samples_per_rep', 'cuda_device',
                                         'pattern', 'min_sense_instances', 'model',
                                         'max_batch_size', 'prediction_cutoff', 'max_number_senses',
                                         ])

DEFAULT_PARAMS = WSISettings(
    n_represents=15,
    n_samples_per_rep=20,
    cuda_device=0,
    pattern = '{pre} {mask_predict} (а также {target}) {post}',
    max_number_senses=7,
    min_sense_instances=2,
    max_batch_size=10,
    prediction_cutoff=200,
    # model='ai-forever/ruRoberta-large'
    # model='cointegrated/rubert-tiny2',
    model='FacebookAI/xlm-roberta-large'
)
