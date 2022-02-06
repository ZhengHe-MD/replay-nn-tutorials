import logging
import os.path


def load_data(source_lang: str, target_lang: str):
    fp = f'./dataset/{source_lang}-{target_lang}.txt'
    is_reversed = False
    if not os.path.exists(fp):
        fp = f'./dataset/{target_lang}-{source_lang}.txt'
        is_reversed = True
    if not os.path.exists(fp):
        raise FileNotFoundError(f'dataset file not found for {source_lang} and {target_lang}')

    pairs = []
    n_discarded = 0

    with open(fp) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            processed = line[:line.index('CC-BY 2.0')].strip()
            parts = processed.split('\t')
            if len(parts) < 2:
                logging.info(f'invalid data line: {line} file: {fp}')
                n_discarded += 1
                continue
            pair = (parts[1], parts[0]) if is_reversed else (parts[0], parts[1])
            pairs.append(pair)
    logging.info(f'{n_discarded} lines discarded')
    return pairs