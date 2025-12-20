import argparse
import random
import warnings

# from transformers import AutoTokenizer, AutoImageProcessor
# from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
# import transformers
import numpy as np
import torch

# from vlm_history import VLMHistory, GPTHistory
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# from llava.model import LlavaLlamaForCausalLM
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from tqdm import tqdm

# import openai
from utils import AlfEnv, agent_factory, load_config_file

warnings.filterwarnings("ignore")
# openai.api_key = os.getenv("OPENAI_API_KEY")


PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


def main():
    parser = argparse.ArgumentParser(description="VLM4ALFWorld")
    parser.add_argument("--alf-config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--use-wandb", default=False, action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ALFWorld")
    parser.add_argument("--wandb-run", type=str, default="test")
    parser.add_argument(
        "--env-name", default="Alfred-Thor", help="environment to train on"
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=134,
        help="number of episodes to evaluate the agent (default: 134)",
    )
    # parser.add_argument("--vlm-model-path", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["pure_vlm", "unaware_vlm_llm", "aware_vlm_llm"],
        default="pure_vlm",
        help="Type of agent to use: 'pure_vlm', 'unaware_vlm_llm', or 'aware_vlm_llm'.",
    )
    parser.add_argument("--vlm-model", type=str, default="gpt-4o")
    parser.add_argument("--llm-model", type=str, default="gpt-4o")

    args = parser.parse_args()
    agent = agent_factory(
        agent_type=args.agent_type,
        llm_model=args.llm_model,
        vlm_model=args.vlm_model,
        num_eval=args.num_eval,
    )

    ###### set random seed #########
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    ####### init env ########################

    assert args.alf_config is not None, "Alfworld environment requires a config file"
    config = load_config_file(args.alf_config)
    assert "AlfredThorEnv" in config["env"]["type"], "Only AlfredThorEnv is supported"
    envs = AlfEnv(args.alf_config, train_eval="eval_out_of_distribution")
    obs, infos = envs.reset(seed=args.seed)

    horizon = 30
    if args.use_wandb:
        import wandb

        run_name = args.wandb_run + "-" + args.env_name + "-" + agent.name
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=run_name,
            config=args,
            notes=" ",
            save_code=True,
        )

    for i in tqdm(range(args.num_eval)):
        # ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = "/".join(infos["extra.gamefile"][0].split("/")[-2:])
        # if not name.startswith('pick_heat_then_place') and not name.startswith('pick_and_place') and not name.startswith('look_at_obj'):
        if not name.startswith("pick_and_place"):
            obs, infos = envs.reset()
            continue

        start_info = "\n".join(infos["observation_text"][0].split("\n\n")[1:])
        agent.logger.info(f"start info: {start_info}")
        agent.get_task_info(task_name=name, task_info=start_info)

        if args.use_wandb:
            ############### init logger ##################
            traj_logger = wandb.Table(
                columns=["obs_img", "obs_text", "object_binding", "vlm_feedback"]
            )
            traj_logger.add_data(
                wandb.Image(obs["image"].cpu().numpy()),
                infos["observation_text"][0],
                " ",
                " ",
            )

        for step in range(horizon):
            ############# Inference of vlm ######################
            # start_time = time.time()
            response = agent.predict()
            # end_time = time.time()
            # print(f"Inference time: {end_time - start_time} seconds")
            action = agent.process_action(response)
            agent.add_to_memory("response", response)

            cur_obs, reward, done, infos = envs.step([action])
            text_obs = infos["observation_text"][0]
            agent.logger.info(f"{response}\n{text_obs}\n")
            # print("VLM response: "+vlm_response)

            agent.observe(cur_obs)
            ############# log trajectory ########################
            if args.use_wandb:
                traj_logger.add_data(
                    wandb.Image(cur_obs["image"].cpu().numpy()),
                    text_obs,
                    cur_obs["object_bindings"],
                    response,
                )

            if (
                step == horizon - 1
                or infos["won"][0]
                or envs._is_reset
                or agent.is_exhausted
            ):
                agent.track_metrics(infos)
                obs, infos = envs.reset()
                agent.reset()

                agent.logger.info(
                    "Experiment {}, success rate {:.2f}\n".format(
                        i + 1, np.mean(agent.metrics["episode_success_rate"])
                    )
                )

                if args.use_wandb:
                    summary = agent.get_metrics_summary(iteration=i)
                    wandb.log(summary)
                    wandb.log({f"trajectory_{i}": traj_logger})

                break


if __name__ == "__main__":
    main()
