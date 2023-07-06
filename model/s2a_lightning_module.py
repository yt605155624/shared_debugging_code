from typing import Dict
from encodec import EncodecModel
import pytorch_lightning
from torch.optim import AdamW
from model.soundstorm import SoundStorm
from model.utils import code_to_wav
import torch
from model.lr_schedulers import WarmupCosineLRSchedule


class Semantic2AcousticLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.encodec_model = EncodecModel.encodec_model_24khz()
        self.encodec_model.set_target_bandwidth(6.0)

        self.model = SoundStorm(
            config,
            encodec=self.encodec_model,
            hubert_kmean_path=config["model"]["hubert_kmean_path"]
        )

        self.gradient_accumulation = config['train']['gradient_accumulation']
        self.clip_grad = config['train']['gradient_clip']
        self.automatic_optimization = False
        self.save_hyperparameters()

    def training_step(self, batch: Dict, batch_idx: int):


        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        loss, acc = self.model.forward(batch['semantic_ids'], batch['acoustic_ids'])
        loss_original = loss.item()

        # optional: divide loss by number of accumulation
        # loss = loss / accum
        self.manual_backward(loss)

        if batch_idx > 0 and batch_idx % self.gradient_accumulation == 0:
            self.clip_gradients(opt, gradient_clip_val=self.clip_grad, gradient_clip_algorithm="norm")
            opt.step()
            opt.zero_grad()
            scheduler.step()

        self.log("total_loss", loss_original, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("lr", scheduler.get_last_lr()[0], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("acc_s2a_top10", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch: Dict, batch_idx: int):

        y_codes = self.model.generate(conds=batch['semantic_ids'],
                                      codes=batch['acoustic_ids']).clamp(0, 1023)
        with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
            code_to_wav(self.encodec_model, y_codes.transpose(1, 2), f'eval/test_libri20_{batch_idx}.wav')
        if batch_idx == 0:
            print('')

    def configure_optimizers(self):

        lm_opt = AdamW(
            self.model.parameters(),
            self.config['optimizer']['lr'],
            betas=(0.8, 0.99),
            eps=0.000000001,
            # weight_decay=4.5e-2
        )
        
        return {
                 "optimizer": lm_opt,
                 "lr_scheduler": {
                     "scheduler": WarmupCosineLRSchedule(lm_opt,
                                                         init_lr=self.config['optimizer']['lr_init'],
                                                         peak_lr=self.config['optimizer']['lr'],
                                                         end_lr=self.config['optimizer']['lr_end'],
                                                         warmup_steps=self.config['optimizer']['warmup_steps'],
                                                         total_steps=self.config['optimizer']['decay_steps'])
                 }
             }

