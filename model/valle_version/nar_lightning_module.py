from typing import Dict
from encodec import EncodecModel
import pytorch_lightning
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from model.utils import code_to_wav
import torch
from model.nar_decoder import NARDecoder
from modules.optim import ScaledAdam


class Semantic2AcousticLightningModule(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()

        self.encodec_model = EncodecModel.encodec_model_24khz()
        self.encodec_model.set_target_bandwidth(6.0)

        self.config = {
            "embedding_dim": 1024,
            "hidden_dim": 1024,
            "num_head": 16,
            "num_layers": 12,
            "num_codebook": 8,
            "p_dropout": 0,
            "vocab_size": 1024 + 1,
            "EOS": 1024
        }
        self.model = NARDecoder(self.config, norm_first=True)

        self.automatic_optimization = False
        self.save_hyperparameters()


    def training_step(self, batch: Dict, batch_idx: int):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        loss, acc, _ = self.model.forward(batch['semantic_ids'], batch['semantic_ids_len'],
                                          batch['acoustic_ids'], batch['acoustic_ids_len'])
        self.manual_backward(loss)


        if batch_idx > 0 and batch_idx % 4 == 0:
            self.clip_gradients(opt, gradient_clip_val=2.0, gradient_clip_algorithm="norm")
            opt.step()
            opt.zero_grad()
            scheduler.step()

        self.log("total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("lr", scheduler.get_last_lr()[0], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("acc_s2a_top10", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)


    def validation_step(self, batch: Dict, batch_idx: int):

        y_codes = self.model.infer(batch['semantic_ids'], batch['semantic_ids_len'],
                                      batch['acoustic_ids']).clamp(0, 1023)
        with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
            code_to_wav(self.encodec_model, y_codes.transpose(1, 2), f'eval/test_ljvallelrg2_{batch_idx}.wav')
        if batch_idx == 0:
            print('')


    def configure_optimizers(self):

        model_parameters = self.model.parameters()
        parameters_names = []
        parameters_names.append(
            [
                name_param_pair[0]
                for name_param_pair in self.model.named_parameters()
            ]
        )
        lm_opt = ScaledAdam(
            model_parameters,
            lr=0.05,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )

        return {
            "optimizer": lm_opt,
            "lr_scheduler": {
                "scheduler": ExponentialLR(lm_opt, gamma=0.9999125)
            }
        }
