# SpearTTS + Soundstorm Reproduced
Draft version of SpearTTS + SoundStorm Pipeline that will be opensourced by Inworld AI (Will be removed after the Inworld version is released)

# environment
torch / torchaudio / pytorch_lightning / fairseq (only for preprocess) / encodec / h5py

# preprocess
run preprare_data_xxx.py to manage audio files and encodec feature,
then follow the code in hubert_kmeans/hubert_feature_xxx.py --> learn_kmeans_xxx.py --> dump_kmeans_label.py
(please resolve the path and naming if you have errors preproceesing)

# text to semantic
python train_t2s.py --path-to-config configuration/xxxx_t2s.yaml

# semantic to acoustic 
python train_s2a.py --path-to-config configuration/xxxx_s2a.yaml

# inference
(resolve path for data and saved checkpoints)
python inference.py

# listen to audio clips
eval/xxxx_x.wav
 (first 3 sec is ground truth, followed by the generated audio)

# Credits to
https://github.com/rishikksh20/SoundStorm-pytorch

https://github.com/lifeiteng/vall-e

https://github.com/lucidrains/soundstorm-pytorch
