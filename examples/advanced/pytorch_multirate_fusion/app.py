
import argparse
import time
import os
import sys
import logging
from dataclasses import dataclass
from typing import Optional, Any
import torch
import torch.nn as nn

# Project root setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
src_root = os.path.join(project_root, "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

import retriever
from retriever.flow import flow_io, Flow, Rate
from retriever.flow.adapter import Latest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FusionUnified")

# ============================================================================
# DATA TYPES (FLOW IO)
# ============================================================================

@flow_io
@dataclass
class RobotState:
    """Fast proprioceptive state (100Hz)"""
    joint_pos: Any # torch.Tensor (7,)
    joint_vel: Any # torch.Tensor (7,)
    timestamp: float

@flow_io
@dataclass
class CameraImage:
    """Raw camera image (10Hz)"""
    data: Any # torch.Tensor (3, 224, 224)
    timestamp: float

@flow_io
@dataclass
class VisualContext:
    """Heavy visual features (10Hz)"""
    features: Any # torch.Tensor (2048,)
    timestamp: float

@flow_io
@dataclass
class RobotAction:
    """Control output (100Hz)"""
    joint_torques: Any # torch.Tensor (7,)
    timestamp: float
    
@flow_io
@dataclass
class PolicyInput:
    joint_pos: Any 
    joint_vel: Any
    visual_features: Any

# ============================================================================
# MOCK MODELS
# ============================================================================

class MockResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2) 
        self.fc = nn.Linear(16 * 111 * 111, 2048)
        
    def forward(self, x):
        # Simulate heavy compute
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

class MockPolicyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 14 (proprio) + 2048 (vision)
        self.net = nn.Sequential(
            nn.Linear(14 + 2048, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )
        
    def forward(self, state, visual_ctx):
        x = torch.cat([state, visual_ctx], dim=1)
        return self.net(x)

# ============================================================================
# FLOWS
# ============================================================================

class RobotSimFlow(Flow[None, RobotState]):
    def init(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RobotSim] Running on {self.device}")
        
    def run(self, _):
        q = torch.randn(7, device=self.device)
        dq = torch.randn(7, device=self.device)
        return RobotState(joint_pos=q, joint_vel=dq, timestamp=time.time())

class CameraSimFlow(Flow[None, CameraImage]):
    def init(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CameraSim] Running on {self.device}")
        
    def run(self, _):
        img = torch.randn(3, 224, 224, device=self.device)
        return CameraImage(data=img, timestamp=time.time())

class VisionBackboneFlow(Flow[CameraImage, VisualContext]):
    """10Hz Heavy Compute"""
    def init(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MockResNet().to(self.device).eval()
        print(f"[VisionBackbone] Initialized on {self.device}")

    def run(self, inp: CameraImage) -> Optional[VisualContext]:
        if inp.data is None: return None
        with torch.no_grad():
            img = inp.data.to(self.device).unsqueeze(0)
            time.sleep(0.08) # Simulate latency
            feats = self.model(img)
        return VisualContext(features=feats.squeeze(0), timestamp=inp.timestamp)

class FusionPolicyFlow(Flow[PolicyInput, RobotAction]):
    """100Hz Fast Fusion"""
    def init(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = MockPolicyMLP().to(self.device).eval()
        self.last_vision: Optional[torch.Tensor] = None
        self.last_vision_ts: float = 0.0
        self.count = 0
        print(f"[FusionPolicy] Initialized on {self.device}")
        
    def run(self, inp: PolicyInput) -> Optional[RobotAction]:
        if inp.joint_pos is None: return None
        
        # 1. Handle Vision (Stateful Async Memory)
        if inp.visual_features is not None:
            self.last_vision = inp.visual_features.to(self.device)
            self.last_vision_ts = inp.visual_features.timestamp
            
        vision_feat = self.last_vision if self.last_vision is not None else torch.zeros(2048, device=self.device)
            
        # 2. Prepare Inputs
        proprio = torch.cat([inp.joint_pos.to(self.device), inp.joint_vel.to(self.device)], dim=0).unsqueeze(0)
        vision = vision_feat.unsqueeze(0)
        
        # 3. Fast Inference
        self.count += 1
        with torch.no_grad():
            action = self.policy(proprio, vision)
            
        if self.count % 20 == 0:
             print(f"[Policy] Step {self.count:04d} | Vision TS: {self.last_vision_ts:.3f} | Action: {action[0,0]:.3f}")

        return RobotAction(joint_torques=action.squeeze(0), timestamp=time.time())

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser("PyTorch Multi-Rate Async Fusion (Unified)")
    parser.add_argument("--backend", default="dora", choices=["dora", "multiprocessing"])
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    
    print("Creating flows...")
    # Sources
    robot = RobotSimFlow() @ Rate(hz=100)
    camera = CameraSimFlow() @ Rate(hz=10)
    
    # Compute
    vision = VisionBackboneFlow() @ Rate(hz=10) 
    policy = FusionPolicyFlow() @ Rate(hz=100)
    
    print("Connecting graph (Global DSL)...")
    # Camera -> Vision
    retriever.connect(camera, vision)
    
    # Fusion: Robot -> Policy
    retriever.connect(robot, policy)
    
    # Fusion: Vision -> Policy (Latest strategy on edge)
    # Allows Policy to skip old frames if buffer fills, but Logic handles "holding"
    retriever.connect(vision, policy, map={"features": "visual_features"}, sync=Latest())
    
    print("Starting Multi-Rate Async Fusion...")
    try:
        retriever.run(backend=args.backend, duration=args.duration)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
