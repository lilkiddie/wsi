from src.models.bert import Bert
from src.config import DEFAULT_PARAMS
from src.wsi.wsi_inductor import WSI_Inductor

if __name__ == '__main__':
    config = DEFAULT_PARAMS
    model = Bert(config)
    wsi_inductor = WSI_Inductor(model, config)

    ans = wsi_inductor.run('../resources/russe-wsi-kit-fixed_datasets/data/main/bts-rnc')
    print(f'ARI score: {ans}')
