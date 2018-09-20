from fastai.text import *
from jianfan2 import ftoj

import html
import fire


BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

re1 = re.compile(r'  +')


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    src_text = re1.sub(' ', html.unescape(x))
    # 繁简转化
    src_text = ftoj(src_text)
    return src_text


def get_texts(df, n_lbls, lang='en'):
    if len(df.columns) == 1:
        # 1列时就是文本列，没有labels
        labels = []
        texts = f'\n{BOS} {FLD} 1 ' + df[0].astype(str)
    else:
        # 多列时，n_lbls 是文本列
        labels = df.iloc[:,1].values.astype(np.int64)
        texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
        # TODO: 新语料结构应该是0列是lable，后面的都作为文本
        # for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls+1} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)
    # 分词处理
    tok = Tokenizer(lang=lang).proc_all_mp(partition_by_cores(texts), lang=lang)
    return tok, list(labels)


def get_all(df, n_lbls, lang='en', flag=''):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls, lang=lang)
        if flag:
            np.save(tmp_path / f'tok_{flag}_{i}.npy', tok_)
            np.save(tmp_path / f'lbl_{flag}_{i}.npy', labels_)
        tok += tok_
        labels += labels_
    return tok, labels


def create_toks(dir_path, chunksize=1000, n_lbls=1, lang='en'):
    print(f'dir_path {dir_path} chunksize {chunksize} n_lbls {n_lbls} lang {lang}')
    try:
        spacy.load(lang)
    except OSError:
        # TODO handle tokenization of Chinese, Japanese, Korean
        print(f'spacy tokenization model is not installed for {lang}.')
        lang = lang if lang in ['en', 'de', 'es', 'pt', 'fr', 'it', 'nl'] else 'xx'
        print(f'Command: python -m spacy download {lang}')
        sys.exit(1)
    dir_path = Path(dir_path)
    assert dir_path.exists(), f'Error: {dir_path} does not exist.'
    df_trn = pd.read_csv(dir_path / 'train.csv', header=None, chunksize=chunksize)
    df_val = pd.read_csv(dir_path / 'val.csv', header=None, chunksize=chunksize)

    global tmp_path = dir_path / 'tmp'
    tmp_path.mkdir(exist_ok=True)
    tok_trn, trn_labels = get_all(df_trn, n_lbls, lang=lang, flag='trn')
    tok_val, val_labels = get_all(df_val, n_lbls, lang=lang, flag='val')

    np.save(tmp_path / 'tok_trn.npy', tok_trn)
    np.save(tmp_path / 'tok_val.npy', tok_val)
    np.save(tmp_path / 'lbl_trn.npy', trn_labels)
    np.save(tmp_path / 'lbl_val.npy', val_labels)

    trn_joined = [' '.join(o) for o in tok_trn]
    open(tmp_path / 'joined.txt', 'w', encoding='utf-8').writelines(trn_joined)


if __name__ == '__main__': 
    fire.Fire(create_toks)
