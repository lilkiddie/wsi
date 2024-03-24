import pandas as pd
import pymorphy3
from razdel import sentenize


def russe_generator(path):
    df = pd.read_csv(path, sep='\t')
    prev = None
    idx = 0
    for index in df.index:
        word = df['word'][index]
        if word != prev:
            idx = 1
        pos = df['positions'][index]
        start, end = map(int, pos.split(',')[0].split('-'))
        for sent in sentenize(df['context'][index]):
            start_, end_, text = sent
            if not start_ <= start <= end_:
                continue
            start -= start_
            end -= start_
            pre, target, post = text[:start], text[start:end], text[end:]
            yield (pre, target, post, df['word'][index] + f'.{idx}')
            prev = word
            idx += 1
