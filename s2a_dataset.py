from torch.utils.data import Dataset
import pandas as pd
import torch, torchaudio
import os
from typing import List, Dict, Tuple, Optional, Union
import torch.nn.functional as F
from h5py import File
import numpy as np


class Semantic2AcousticDataset(Dataset):
    """dataset class for text tokens to semantic model training."""

    def __init__(self, metadata_path: Union[str, List[str]],
                 semantic_token_path: Union[str, List[str]],
                 max_sample=None) -> None:
        super().__init__()
        if isinstance(metadata_path, str):
            self.single_dataset = True
            self.dataframe: pd.DataFrame = pd.read_csv(os.path.join(metadata_path, 'metadata.csv'), sep='|')
            self.semantic_tokens: List[List[int]] = self._read_semantic_tokens(semantic_token_path)
            self.acoustic_h5 = File(os.path.join(metadata_path, "audio_features", 'audio_features.hdf5'), "r")
        else:
            # read data from lits of path and concat
            self.single_dataset = False
            dataframes: List[pd.DataFrame] = []
            self.semantic_tokens: List[List[int]] = []
            self.acoustic_h5: Dict = {}
            for meta_path, sem_path in zip(metadata_path, semantic_token_path):
                print(f'loading dataset: {meta_path}')
                df = pd.read_csv(os.path.join(meta_path, 'metadata.csv'), sep='|')
                df['Datasetkey'] = meta_path
                dataframes.append(df)
                self.semantic_tokens += self._read_semantic_tokens(sem_path)
                self.acoustic_h5[meta_path] = \
                    File(os.path.join(meta_path, "audio_features", 'audio_features.hdf5'), "r")

            self.dataframe = pd.concat(dataframes, ignore_index=True)
            self.dataframe.reset_index(drop=True, inplace=True)

        assert len(self.dataframe) == len(self.semantic_tokens)

        if max_sample:
            self.dataframe = self.dataframe[:max_sample]
            self.semantic_tokens = self.semantic_tokens[:max_sample]

        self.n_codebook = 8
        self.max_frame_len: int = 1536  # 20 seconds
        self.PAD: int = 1024

    def __len__(self)-> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict:
        row = self.dataframe.iloc[idx]

        # semantic tokens input
        semantic_ids = self.semantic_tokens[idx]
        semantic_ids = torch.tensor(semantic_ids, dtype=torch.long)
        semantic_ids_len = len(semantic_ids)

        # acoustic token target
        audio_frame_path = row['AudioFramesPath']
        if self.single_dataset:
            audio_frame = self.acoustic_h5['feature'][audio_frame_path]
        else:
            datakey = row['Datasetkey']
            audio_frame = self.acoustic_h5[datakey]['feature'][audio_frame_path]
        audio_frame = np.array(audio_frame)
        audio_frame = torch.tensor(audio_frame)
        #audio_frame = torch.load(audio_frame_path)[:self.max_frame_len]
        audio_frame_len = audio_frame.size(-1)

        return {
            'id': idx,
            'semantic_ids': semantic_ids,
            'semantic_ids_len': semantic_ids_len,
            'acoustic_ids': audio_frame,
            'acoustic_ids_len': audio_frame_len
        }

    def get_sample_length(self, idx: int):
        semantic_ids = self.semantic_tokens[idx]
        sec = 1.0 * len(semantic_ids) / 50
        return sec


    def collate(self, features: List[Dict]) -> Dict:
        sample_index: List[int] = []
        semantic_ids: List[torch.LongTensor] = []
        semantic_ids_lens: List[int] = []
        acoustic_ids: List[torch.LongTensor] = []
        acoustic_ids_lens: List[int] = []

        for feature in features:
            sample_index.append(feature["id"])
            semantic_ids.append(feature["semantic_ids"])
            semantic_ids_lens.append(feature["semantic_ids_len"])
            acoustic_ids.append(feature["acoustic_ids"])
            acoustic_ids_lens.append(feature["acoustic_ids_len"])

        batch_size: int = len(sample_index)
        max_semantic_ids_len: int = max(semantic_ids_lens)
        max_acoustic_ids_len: int = max(acoustic_ids_lens)

        # collate semantic frames
        semantic_ids_t: torch.Tensor = torch.zeros((batch_size, max_semantic_ids_len),
                                                   dtype=torch.long) + self.PAD
        for i, frame_seq in enumerate(semantic_ids):
            semantic_ids_t[i, :frame_seq.size(0)] = frame_seq
        semantic_ids_lens_t: torch.Tensor = torch.tensor(semantic_ids_lens, dtype=torch.long)

        # collate acoustic frames
        acoustic_ids_t: torch.Tensor = torch.zeros((batch_size, self.n_codebook, max_acoustic_ids_len),
                                                   dtype=torch.long) + self.PAD
        for i, frame_seq in enumerate(acoustic_ids):
            acoustic_ids_t[i, :, :frame_seq.size(1)] = frame_seq
        acoustic_ids_lens_t: torch.Tensor = torch.tensor(acoustic_ids_lens, dtype=torch.long)
        return {
            "ids": sample_index,                                # List[int]
            "semantic_ids": semantic_ids_t,                     # bs * max_semantic_ids_length
            "semantic_ids_len": semantic_ids_lens_t,            # bs
            "acoustic_ids": acoustic_ids_t.transpose(1, 2),     # bs * max_semantic_ids_length
            "acoustic_ids_len": acoustic_ids_lens_t             # bs
        }
    
    def _read_semantic_tokens(self, semantic_token_path: str) ->List[List[int]]:
        semantic_tokens: List[List[int]] = []
        with open(semantic_token_path, 'r') as f:
            for line in f:
                semantic_tokens.append([int(x) for x in line.split(' ')])
        return semantic_tokens


if __name__ == '__main__':
    # dataset = Semantic2AcousticDataset(metadata_path='data/ljspeech_prod',
    #                                    semantic_token_path='data/ljspeech_prod/km_semtok/ljspeech_0_1.km')

    dataset = Semantic2AcousticDataset(metadata_path=['data/libritts_train_clean_100', 'data/libritts_train_clean_360', 'data/libritts_train_other_500'],
                                       semantic_token_path=['data/libritts_train_clean_100/km_semtok/libritts_train_clean_100.km',
                                                            'data/libritts_train_clean_360/km_semtok/libritts_train_clean_360.km',
                                                            'data/libritts_train_other_500/km_semtok/libritts_train_other_500.km'])
    sample1 = dataset.__getitem__(0)
    sample2 = dataset.__getitem__(6)
    sample = dataset.collate([sample1, sample2])

    print(sample['semantic_ids'].size())
    print(sample['semantic_ids_len'])
    print(sample['acoustic_ids'].size())
    print(sample['acoustic_ids_len'])
    print('found', len(dataset), 'samples')