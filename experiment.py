import torch
from initializations import Distribution, initialize, train_step
from env import Env, Normalizer
from torch.utils.tensorboard import SummaryWriter
import math
from argparse import ArgumentParser
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--num_steps", required=False, type=int)
parser.add_argument("--kind", required=False, type=str,
                    help='distribution uniform|normal')
parser.add_argument("--vanilla", type=int, required=True,
                    help='whether to use Glorot or Learned Initialization 0|1')
parser.add_argument("--dataset", required=False, type=str,
                    help='mnist|fashion|cifar10')
args = parser.parse_args()

max_steps_env = 500
num_steps = args.num_steps
kind = args.kind

writer = SummaryWriter(
    f"./logs/{kind}_{args.dataset}_{num_steps}" if args.vanilla == 0 else f"./logs/glorot_{args.dataset}_{num_steps}")

if args.vanilla == 0:
    dist = Distribution(kind)
    optim_dist = torch.optim.RMSprop(dist.parameters(), lr=1e-3)

e = Env(training_steps=num_steps, dataset_name=args.dataset)

step = 0
for _ in tqdm(range(max_steps_env)):
    target_model = e.reset()
    Distribution.init_params(target_model)

    if args.vanilla == 0:
        log_probs = initialize(target_model, dist)
    reward = e.step(target_model)
    assert(math.isfinite(reward))
    if args.vanilla == 0:
        loss = train_step(dist, reward,
                          log_probs, optim_dist)
        writer.add_scalar("train/loss", loss, step)

    writer.add_scalar("train/average_reward", reward, step)
    step += 1

    # if (step+1) % 10 == 0:
    #     print("Step:", step+1, "Avg Reward:", e.reward_normalizer.mean())
