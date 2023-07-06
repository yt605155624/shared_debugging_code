import pandas as pd
import os
import torch
import torchaudio
import h5py

import regex as re
from tqdm import tqdm

from audio_processing.audio_tokenizer import AudioTokenizer
from text_processing.phonemizer import GruutPhonemizer

MAX_DURATION_SEC = 20
MIN_PHONEMES_PER_SEC = 8
MAX_PHONEMES_PER_SEC = 30


def _adjust_begin(audio: torch.Tensor, sr=24000, win_len=200, wait_time=0.05):
    assert(audio.size(0) == 1)
    audio_len = audio.size(1)
    max_audio = audio.abs().max()
    cut = 0
    for i in range(audio_len // win_len - 1):
        wavclip = audio[:, (i * win_len): (i * win_len + win_len)]
        if wavclip.max().item() > 0.05 * max_audio:
            cut = i * win_len
            break

    cut = int(cut - wait_time * sr)
    if cut < 0:
        cut = 0
    return audio[:, cut:]


def _trim_audio(waveform: torch.Tensor, sr: int =24000, win_len: int = 200, wait_time=0.05):
    """
    Trim low amplitude portions at beginning and end of file (based on max amplitude)
    """
    waveform = torch.flip(waveform, dims=[-1])
    waveform = _adjust_begin(waveform, sr=sr, win_len=win_len, wait_time=wait_time)
    waveform = torch.flip(waveform, dims=[-1])
    waveform = _adjust_begin(waveform, sr=sr, win_len=win_len, wait_time=wait_time)
    return waveform


def preprocess(data_path: str, output_path: str):

    df = pd.read_csv(f'{data_path}/metadata.csv', sep='|')

    audio_tokenizer = AudioTokenizer()
    phonemizer = GruutPhonemizer(language='en-us')
    # make sure paths are ready
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, 'wavs'))
        os.mkdir(os.path.join(output_path, 'audio_features'))
    h5f = h5py.File(os.path.join(output_path, 'audio_features', "audio_features.hdf5"), "w")
    grp = h5f.create_group('feature')



    audio_file_paths = []
    phonemes_transcriptions = []
    audio_frame_keys = []
    # build dataset
    total_duration = 0
    for file, text in tqdm(zip(df.iloc[:, 0], df.iloc[:, 1]), total=len(df)):

        speaker = 'LJ'
        file = file + '.wav'
        # check digits in text using regex
        #if re.search(r'\d', text):
        #    continue
        ## check # and $ in text using one regex
        #if re.search(r'[$]', text):
        #    continue

        basename = os.path.basename(file)
        read_path = os.path.join(data_path, 'wavs', file)
        save_path_audio = os.path.join('wavs', speaker.lower() + '_' + basename)
        save_path_frame = speaker.lower() + '_' + basename.replace('.wav', '')


        # check duration and speed
        try:
            wav, sr = torchaudio.load(read_path)
        except:
            print(f'error reading {read_path}, skip')
            continue

        wav = audio_tokenizer.convert_audio(wav, sr, audio_tokenizer.sample_rate, audio_tokenizer.channels)
        #wav = _trim_audio(wav)
        wav_len_in_sec = wav.size(-1) / audio_tokenizer.sample_rate

        # skip if audio too long or too fast / slow

        #if wav_len_in_sec > MAX_DURATION_SEC:
        #    print(f'{basename} of length {wav_len_in_sec} exceeds {MAX_DURATION_SEC} seconds, skip ...')
        #   continue
        phonemes = phonemizer.phonemize(text, espeak=False)
        #phoneme_per_sec = len(phonemes) / wav_len_in_sec
        #if phoneme_per_sec < MIN_PHONEMES_PER_SEC or phoneme_per_sec > MAX_PHONEMES_PER_SEC:
        #    print(f'{basename} of speed {phoneme_per_sec} skipped ...')
        #    continue

        torchaudio.save(os.path.join(output_path, save_path_audio), wav, audio_tokenizer.sample_rate)
        audio_file_paths.append(save_path_audio)

        # save phonemes and text
        total_duration += wav_len_in_sec
        phonemes_transcriptions.append(phonemes)

        # compute codes and save
        wav = wav.unsqueeze(0)
        with torch.no_grad():
            codes = audio_tokenizer.encode(wav)[0][0][0]  # n_codebooks, T
        codes = codes.cpu().numpy()
        grp.create_dataset(name=save_path_frame, shape=codes.shape, data=codes)
        audio_frame_keys.append(save_path_frame)

    with open(os.path.join(output_path, 'ljspeech.tsv'), 'w') as f:
        f.write(output_path + '\n')
        for line in audio_file_paths:
            f.write(line + '\n')

    data_dict = {
        'PathToFile': audio_file_paths,
        'Phonemes': phonemes_transcriptions,
        'AudioFramesPath': audio_frame_keys
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(f'{output_path}/metadata.csv', sep='|', index=False)
    print(f'processed dataset of {total_duration / 3600} hours')
    h5f.flush()
    h5f.close()


if __name__ == '__main__':
    preprocess('/home/yufei/icefall/valle/examples/ljspeech/download/LJSpeech-1.1/backup', 'data/ljspeech')

