
import logging
import time
from dataclasses import dataclass
import numpy as np

from retriever.flow import Flow, io, Rate
from retriever.flow import Flow, io, Rate
from retriever.lib.hf import TransformerInput, from_hf
from transformers import pipeline




def create_pipeline():
    """Factory for deferred pipeline creation (picklable)."""
    return pipeline(
        task="feature-extraction", 
        model="prajjwal1/bert-tiny"
    )

logger = logging.getLogger(__name__)

# We use a standard IO for the Perception module's output
@io
@dataclass
class GoalEmbedding:
    vector: np.ndarray
    timestamp: float

class PerceptionFlow(Flow[TransformerInput, GoalEmbedding]):
    """
    Simulates a "Slow" Perception module (e.g. VLA or Vision Encoder).
    Runs at low frequency ( ~1 Hz ).
    
    It wraps a Transformer to encode text commands into goal vectors.
    """
    def __init__(self):
        # Use flexible factory from_hf with a top-level function for pickling
        self.encoder = from_hf(create_pipeline)
        
    def init_config(self) -> dict:
        return {}

    def init(self):
        logger.info("[Perception] Initializing heavy model...")
        self.encoder.init()
        logger.info("[Perception] Ready.")

    def run(self, input_data: TransformerInput) -> GoalEmbedding:
        start_t = time.time()
        
        # 1. heavy compute (inference)
        # We input the command text (simulating "Pick up the apple")
        output = self.encoder.run(input_data)
        
        # 2. simulated extra latency (processing overhead)
        time.sleep(0.1) 
        
        # 3. extract embedding (mocking VLA goal)
        # output.result is usually a list of lists/tensors for feature-extraction
        if output.result:
            # Take the first token embedding (CLS) or mean
            feats = np.array(output.result[0][0])
            # Normalize size for our 'control' policy
            if len(feats) > 10:
                feats = feats[:10] 
        else:
            feats = np.zeros(10)

        logger.info(f"[Perception] Generated goal vector (shape={feats.shape})")
        
        return GoalEmbedding(
            vector=feats,
            timestamp=start_t
        )
