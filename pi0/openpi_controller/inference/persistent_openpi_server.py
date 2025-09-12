#!/usr/bin/env python3
"""
Persistent OpenPI Server

A long-running server that loads the OpenPI model once and serves inference requests.
This avoids the 50+ second model loading time on each request.
"""

import json
import sys
import os
import argparse
import pathlib
import numpy as np
import socket
import threading
import time
from typing import Dict, Any

class PersistentOpenPIServer:
    def __init__(self, checkpoint_dir: str, config_name: str, port: int = 9999, default_prompt: str = None):
        self.checkpoint_dir = checkpoint_dir
        self.config_name = config_name
        self.port = port
        self.default_prompt = default_prompt
        self.policy = None
        self.is_loaded = False
        
    def load_policy(self):
        """Load OpenPI policy once"""
        try:
            # Add OpenPI to path
            sys.path.insert(0, 'src')
            
            # Import OpenPI modules
            from openpi.policies import policy_config
            from openpi.training import config as _config
            
            print(f"🤖 Loading OpenPI model from {self.checkpoint_dir}")
            train_config = _config.get_config(self.config_name)
            
            self.policy = policy_config.create_trained_policy(
                train_config=train_config,
                checkpoint_dir=self.checkpoint_dir,
                default_prompt=self.default_prompt
            )
            
            self.is_loaded = True
            print(f"✅ OpenPI policy loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load OpenPI policy: {e}")
            raise
    
    def run_inference(self, obs_dict: dict) -> dict:
        """Run inference on loaded policy"""
        if not self.is_loaded:
            return {"success": False, "error": "Policy not loaded"}
        
        try:
            # Convert lists back to numpy arrays for images
            for key in obs_dict:
                if key.endswith('_image') or key.endswith('/image'):
                    obs_dict[key] = np.array(obs_dict[key], dtype=np.uint8)
                elif key.endswith('state') or key.endswith('/state'):
                    obs_dict[key] = np.array(obs_dict[key], dtype=np.float32)
            
            result = self.policy.infer(obs_dict)
            return {
                "success": True,
                "actions": result["actions"].tolist(),  # Convert numpy to list for JSON
                "state": result.get("state", None),
                "timing": result.get("policy_timing", {})
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def handle_client(self, conn, addr):
        """Handle a single client connection"""
        print(f"🔗 Client connected: {addr}")
        
        try:
            while True:
                # Receive data
                data = conn.recv(4096).decode('utf-8')
                if not data:
                    break
                
                # Parse request
                try:
                    request = json.loads(data)
                    if request.get("action") == "infer":
                        response = self.run_inference(request.get("obs", {}))
                    else:
                        response = {"success": False, "error": "Unknown action"}
                except json.JSONDecodeError:
                    response = {"success": False, "error": "Invalid JSON"}
                
                # Send response
                response_data = json.dumps(response) + "\n"
                conn.sendall(response_data.encode('utf-8'))
                
        except Exception as e:
            print(f"❌ Error handling client {addr}: {e}")
        finally:
            conn.close()
            print(f"🔌 Client disconnected: {addr}")
    
    def start_server(self):
        """Start the persistent server"""
        print(f"🚀 Starting OpenPI server on port {self.port}")
        
        # Load policy first
        self.load_policy()
        
        # Create socket server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', self.port))
        server_socket.listen(5)
        
        print(f"✅ OpenPI server listening on port {self.port}")
        print("Press Ctrl+C to stop the server")
        
        try:
            while True:
                conn, addr = server_socket.accept()
                # Handle each client in a separate thread
                client_thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
        finally:
            server_socket.close()

def main():
    parser = argparse.ArgumentParser(description="Persistent OpenPI Server")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory")
    parser.add_argument("--config-name", required=True, help="Config name")
    parser.add_argument("--port", type=int, default=9999, help="Server port")
    parser.add_argument("--default-prompt", help="Default prompt")
    
    args = parser.parse_args()
    
    server = PersistentOpenPIServer(
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config_name,
        port=args.port,
        default_prompt=args.default_prompt
    )
    
    server.start_server()

if __name__ == "__main__":
    main()
