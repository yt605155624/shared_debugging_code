import torch
import yaml
from model.s2a_lightning_module import Semantic2AcousticLightningModule
from model.t2s_lightning_module import Text2SemanticLightningModule
from s2a_dataset import Semantic2AcousticDataset
from t2s_dataset import Text2SemanticDataset
from model.utils import code_to_wav
from time import time

def generate_semantic(idx):
    with open('configuration/ljspeech_t2s.yaml', "r") as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    # test_dataset = Text2SemanticDataset(metadata_path='data/demo',
    #                                     semantic_token_path='data/demo/km_semtok/audio_files_0_1.km', max_sample=8)

    test_dataset = Text2SemanticDataset(metadata_path='data/libritts_dev_clean',
                                        semantic_token_path='data/libritts_dev_clean/km_semtok/libritts_dev_clean.km', max_sample=8)

    t2s_model = Text2SemanticLightningModule.load_from_checkpoint(checkpoint_path='checkpoints/t2s_libritts.ckpt',
                                                                  config=configuration)
    t2s_model.eval()

    batch = test_dataset.collate([test_dataset.__getitem__(idx)])
    semantic_len = batch['semantic_ids'].size(1)
    prompt_len = min(int(semantic_len * 0.5), 150)
    prompt = batch['semantic_ids'][:, :prompt_len]

    st = time()
    with torch.no_grad():
        pred_semantic = t2s_model.model.infer(batch['phoneme_ids'].cuda(),
                                              batch['phoneme_ids_len'].cuda(), prompt.cuda(),
                                              top_k=configuration['inference']['top_k'])
    print(f'{time() - st} sec used in T2S')
    torch.save(pred_semantic.detach().cpu(), f'eval/semantic_toks_{idx}.pt')

def generate_audio(idx):
    with open('configuration/ljspeech_s2a.yaml', "r") as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
        configuration['model']["hubert_kmean_path"] = None

    s2a_model = Semantic2AcousticLightningModule.load_from_checkpoint(checkpoint_path='checkpoints/s2a_libritts_70.ckpt', config=configuration)
    # test_dataset = Semantic2AcousticDataset(metadata_path='data/demo',
    #                                         semantic_token_path='data/demo/km_semtok/audio_files_0_1.km', max_sample=8)
    test_dataset = Semantic2AcousticDataset(metadata_path='data/libritts_dev_clean',
                                            semantic_token_path='data/libritts_dev_clean/km_semtok/libritts_dev_clean.km', max_sample=8)
    s2a_model.eval()


    semantic_ids = torch.load(f'eval/semantic_toks_{idx}.pt')
    acoustic_ids = test_dataset.__getitem__(idx)['acoustic_ids'].transpose(0, 1).unsqueeze(0)
    ref_ids = test_dataset.__getitem__(idx)['semantic_ids'].unsqueeze(0)

    print('ground truth length of semantic tokens: \t', ref_ids.size(1))
    print('predicted length of acoustic tokens: \t', int(semantic_ids.size(1) * 1.5))
    print('ground truth length of acoustic tokens: \t', acoustic_ids.size(1))
    st = time()
    with torch.no_grad():
        # note that acoustic ids will be crop-ed to first 3 sec as prompt
        y_codes = s2a_model.model.generate(semantic_ids.cuda(), acoustic_ids.cuda()).clamp(0, 1023)
    print(f'{time() - st} sec used in S2A')

    st = time()
    #y_codes = s2a_model.model.generate(ref_ids.cuda(), acoustic_ids.cuda()).clamp(0, 1023)
    with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
        code_to_wav(s2a_model.encodec_model, y_codes.transpose(1, 2), f'eval/inference_{idx}.wav')
    print(f'{time() - st} sec used in Decode')

if __name__ == '__main__':
    idx = 0
    generate_semantic(idx)
    generate_audio(idx)