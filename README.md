# environment
torch / torchaudio / pytorch_lightning / fairseq (only for preprocess) / encodec / h5py

# preprocess
run preprare_data_xxx.py
follow hubert_kmeans/hubert_feature_xxx.py --> learn_kmeans_xxx.py --> dump_kmeans_label.py
(please resolve the path and naming if you have errors preproceesing)

# text to semantic
python train_t2s.py --configuration/xxxx_t2s.yaml

# semantic to acoustic 
python train_s2a.py --configuration/xxxx_s2a.yaml

# inference
(resolve path for data and saved checkpoints)
python inference.py

# listen
eval/xxxx_x.wav
