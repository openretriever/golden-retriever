from cliport.utils import utils
from lepp.lepp_lang_goal import LEPPAgent
from lepp.pick_transporter_net import Attention as AttentionTrans
from lepp.place_net_noFourier import TransportWoFourier


class TransporterAgent(LEPPAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)
        # super().__init__(name, cfg, train_ds, test_ds)

        self._build_model()

    def _build_model(self):
        self.attention = AttentionTrans(
            cfg=self.cfg,
            device=self.device_type,
            preprocess=utils.preprocess,
            init=self.init,
            model_name=self.model_name,
            kernel_name=self.pick_kernel_name,
            vlm_name=self.vlm_name,
            lan_kernel=True,
            dist=self.dist,
            vlm_model=self.vlm_model,
            lan_emb_dim=512,
        )
        self.transport = TransportWoFourier(
            cfg=self.cfg,
            device=self.device_type,
            preprocess=utils.preprocess,
            init=self.init,
            model_name=self.model_name,
            kernel_name=self.place_kernel_name,
            vlm_name=self.vlm_name,
            crop_size=self.crop_size,
            lan_kernel=True,
            dist=self.dist,
            vlm_model=self.vlm_model,
            lan_emb_dim=512,
        )
