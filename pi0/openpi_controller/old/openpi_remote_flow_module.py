"""
OpenPI Remote (WebSocket) Inference Flow Module for Retriever

This script provides a function to instantiate an OpenPI websocket client for remote inference and wraps it as a callable for Retriever's Flow system.
"""

def make_openpi_remote_client(host="localhost", port=8000, api_key=None):
    """
    Instantiate an OpenPI websocket client for remote inference.
    Returns a callable: obs_dict -> action_dict
    """
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy(host=host, port=port, api_key=api_key)
    return client.infer

# Example usage with Retriever Flow:
if __name__ == "__main__":
    from retriever.core.flow import Flow
    openpi_client = make_openpi_remote_client()
    openpi_flow = Flow.from_module(openpi_client)
    # Example observation (replace with real data)
    obs = {"prompt": "pick up the fork"}
    action = openpi_flow(obs)
    print("Action:", action) 