# Dataset
Starting from version V2, we support both `Kaldi-form` and `Dynamic-form` manifests use cases.

## `Dynamic-dataset`: Metafile, Parser and Sampler

Overall data pipelines:

             Parser            Smpler
    Metafile -------> Dataset --------> Dataloader

A metafile looks like:
    
    uttid, spkid, gender, path, length, sample rate, channels
    116-288045-0000, 116, M, dev-other/116/288045/116-288045-0000.flac, 170400, 16000, 1
    116-288045-0001, 116, M, dev-other/116/288045/116-288045-0001.flac, 138160, 16000, 1


Parsing with speaker as key first

    train_dataset.meta[vox2_id04778].keys()
    // dict_keys(['gender', 'channels', 'utts', 'corpus_id'])

    train_dataset.meta['vox2_id04778']['utts']['vox2_jDbFankRpiU_00007']
    // {'path': 'voxceleb2/id04778/jDbFankRpiU/00007.wav', 'length': '142336', 'channels': '1', 'sr': '16000'}

