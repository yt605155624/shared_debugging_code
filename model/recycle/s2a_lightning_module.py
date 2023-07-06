from typing import Dict
from encodec import EncodecModel
import pytorch_lightning
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from model.soundstorm2 import ConformerWrapper, SoundStorm
from model.utils import code_to_wav
import torch
from modules.optim import ScaledAdam


class Semantic2AcousticLightningModule(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()

        self.encodec_model = EncodecModel.encodec_model_24khz()
        self.encodec_model.set_target_bandwidth(6.0)

        conformer = ConformerWrapper(
            codebook_size=1025,
            num_quantizers=8,
            conformer=dict(
                dim=256,
                depth=6
            ),
        )

        self.model = SoundStorm(
            conformer,
            steps=8,  # 18 steps, as in original maskgit paper
            schedule='cosine',  # currently the best schedule is cosine
            encodec=self.encodec_model,
            hubert_kmean_path='hubert_checkpoint/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin'
        )

        self.automatic_optimization = False
        self.save_hyperparameters()


    def training_step(self, batch: Dict, batch_idx: int):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        loss, acc, _ = self.model.forward(batch['semantic_ids'], batch['acoustic_ids'])
        self.manual_backward(loss)


        if batch_idx > 0 and batch_idx % 4 == 0:
            self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt.step()
            opt.zero_grad()
            scheduler.step()

        self.log("total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("lr", scheduler.get_last_lr()[0], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("acc_s2a_top10", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)


    def validation_step(self, batch: Dict, batch_idx: int):

        y_codes = self.model.generate(sem_cond=batch['semantic_ids'],
                                      prompt=batch['acoustic_ids'],
                                      batch_size=1).clamp(0, 1023)
        with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
            code_to_wav(self.encodec_model, y_codes.transpose(1, 2), f'eval/test_ljembn_{batch_idx}.wav')
        if batch_idx == 0:
            print('')


    def configure_optimizers(self):

        lm_opt = AdamW(
            self.model.parameters(),
            0.0002,
            betas=(0.8, 0.99),
            eps=0.000000001
        )

        return {
                "optimizer": lm_opt,
                "lr_scheduler": {
                    "scheduler": ExponentialLR(lm_opt, gamma=0.999875)
                }
            }
