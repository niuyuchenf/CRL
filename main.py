from utils import str2bool, Action_adapter, Action_adapter_reverse
from SAC import SAC_countinuous
import os
import argparse
import torch
from state_predict import next_state_predict,StateNetwork
from reward_predict import reward_predict, RewardNetwork
from train_environment import train_LanechangingRunEnv
from tqdm import tqdm

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0)

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(2e4), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(10e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2.5e3), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')
parser.add_argument('--buffer_save_interval', type=int, default=int(1e5), help='buffer interal, in stpes')

parser.add_argument('--update_causal', type=int, default=int(2.5e3), help='causal interal, in stpes')
parser.add_argument('--state_size', type=int, default=500, help='state sample')
parser.add_argument('--state_eopchs', type=int, default=200, help='state epochs')
parser.add_argument('--state_temperature', type=float, default=0.5, help='state temperature')
parser.add_argument('--state_lambda_rege', type=float, default=1e-11, help='state lambda rege')
parser.add_argument('--reward_size', type=int, default=20, help='reward sample')
parser.add_argument('--reward_eopchs', type=int, default=500, help='reward epochs')
parser.add_argument('--reward_temperature', type=float, default=0.6, help='reward temperature')
parser.add_argument('--reward_lambda_rege', type=float, default=1e-11, help='reward lambda rege')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
opt = parser.parse_args(args=[])
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

for number in [11]:
    print(f"Training Run: {number}")
    env = train_LanechangingRunEnv()
    opt.state_dim = 14
    opt.action_dim = 1
    opt.max_action = 5
    opt.max_e_steps = 200

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_network = RewardNetwork((opt.state_dim, 1),(opt.action_dim, 1),opt.reward_temperature,1,
                                   opt.reward_lambda_rege,0.01,False)
    reward_network.to(device)  
    reward_network.save_parameters(r'.\reward\reward_{}.pth'.format(number))
    state_network = StateNetwork((opt.state_dim, opt.state_dim), (opt.action_dim, opt.state_dim),
                                 opt.state_temperature, opt.state_dim, opt.state_lambda_rege,0.01,False)
    state_network.to(device)  
    state_network.save_parameters(r'.\next_state\state_{}.pth'.format(number))

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = SAC_countinuous(**vars(opt))
    total_steps = 0

    with tqdm(total=opt.Max_train_steps, desc="Training Progress") as pbar:
        while total_steps < opt.Max_train_steps:
            s, info = env.reset()  
            done = False

            '''Interact & train'''
            while not done:
                if total_steps < (5 * opt.max_e_steps):
                    act = env.action_space.sample()  
                    a = Action_adapter_reverse(act, opt.max_action)  
                else:
                    a = agent.select_action(s, deterministic=False)  
                    act = Action_adapter(a, opt.max_action)  
                s_next, r, dw, tr, info = env.step(act)  
                done = (dw or tr)

                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next
                total_steps += 1

                pbar.update(1)

                if ((total_steps >= opt.max_e_steps) and (total_steps % opt.update_causal == 0)) or (total_steps == 300):
                    state_features, action_features, rewards_features, next_state_features, _ = agent.replay_buffer.sample(opt.state_size)
                    out_states, out_actions, out_rewards, out_next_state_features = agent.replay_buffer.modify_state(
                        state_features, action_features, rewards_features, next_state_features, 50)
                    next_state_predict(opt.state_dim, opt.action_dim, opt.state_temperature,
                                       opt.state_lambda_rege, out_states, out_actions,
                                       out_next_state_features, 16, opt.state_eopchs,number)

                    trajectories = agent.replay_buffer.reward_sample(opt.reward_size)
                    reward_predict(trajectories, opt.state_dim, 1, opt.reward_temperature,
                               opt.action_dim, opt.reward_eopchs, opt.reward_lambda_rege,number)

                '''train if it's time'''
                if (total_steps >= 2 * opt.max_e_steps) and (total_steps % opt.update_every == 0):
                    for j in range(opt.update_every):
                        agent.train(number)

                '''save model'''
                if total_steps % opt.save_interval == 0 or total_steps == opt.Max_train_steps:
                    agent.save('LC0', int(total_steps / 1000), number)
