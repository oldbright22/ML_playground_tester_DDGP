import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Training policy agent")

    parser.add_argument("--run_name", type=str, default="train",
                        help="Name of the run")
    parser.add_argument("--algorithm", type=str, default="A2C",
                        choices=["A2C", "PPO"],
                        help="Type of algorithm to use for training")
    parser.add_argument("--env_id", type=str, default="Ant-v4",
                        help="Id of the environment to train on")
    parser.add_argument("--perform_testing", action="store_true",
                        help="Whether to perform testing after training")
    parser.add_argument("--log_video", action="store_true",
                        help="Whether to log video of agent's performance")

    
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum number of steps to train for")
    parser.add_argument("--steps_per_epoch", type=int, default=100,
                        help="Number of steps to train for per epoch")
    parser.add_argument("--num_envs", type=int, default=8,
                        help="Number of environments to train on")
    parser.add_argument("--num_rollout_steps", type=int, default=5,
                        help="Number of steps to rollout policy for")
    
    parser.add_argument("--optimizer", type=str, default="RMSprop",
                        choices=["Adam", "RMSprop", "SGD"])
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for training")
    parser.add_argument("--lr_decay", type=float, default=1.0,
                        help="Learning rate decay for training")
    parser.add_argument("--weight_decay", type=float, default=0.,
                        help="Weight decay (L2 regularization) for training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=1.,
                        help="Lambda parameter for GAE")
    parser.add_argument("--value_coef", type=float, default=0.5,
                        help="Coefficient for value loss")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Coefficient for entropy loss")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--init_std", type=float, default=0.2,
                        help="Initial standard deviation for policy")
    
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size for policy")
    parser.add_argument("--shared_extractor", action="store_true",
                        help="Whether to use a shared feature extractor for policy")

    
    parser.add_argument("--ppo_batch_size", type=int, default=None,
                        help="Batch size for PPO")
    parser.add_argument("--ppo_epochs", type=int, default=None,
                        help="Number of epochs to train PPO for")
    parser.add_argument("--ppo_clip_ratio", type=float, default=None,
                        help="Clip ratio for PPO")
    parser.add_argument("--ppo_clip_anneal", action="store_true",
                        help="Whether to anneal the clip ratio for PPO")
    

    args = parser.parse_args()
    return args
