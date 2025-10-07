"""
OpenPI Local Inference Flow Module for Retriever

This script provides a function to instantiate an OpenPI policy for local inference and wraps it as a callable for Retriever's Flow system.
"""

def make_openpi_local_policy(model_name="pi0_fast_droid"):
    """
    Instantiate an OpenPI policy for local inference.
    Returns a callable: obs_dict -> action_dict
    """
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.shared import download

    cfg = config.get_config(model_name)
    checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{model_name}")
    policy = policy_config.create_trained_policy(cfg, checkpoint_dir)
    return policy.infer

# Example usage with Retriever Flow:
if __name__ == "__main__":
    from retriever.core.flow import Flow
    openpi_policy = make_openpi_local_policy()
    openpi_flow = Flow.from_module(openpi_policy)
    # Example observation (replace with real data)
    obs = {"prompt": "pick up the fork"}
    action = openpi_flow(obs)
    print("Action:", action) 