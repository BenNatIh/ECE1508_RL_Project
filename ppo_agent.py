import gymnasium as gym
import gymnasium_robotics
import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import pickle

class Agent():
    def __init__(self, env, state_dim, action_dim, gamma, actor_model, critic_model, actor_optimizer, critic_optimizer, device='cpu', entropy_coef=0.0):
        self.gamma = gamma
        self.clip_ratio = None  # set externally if needed
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.env = env
        self.device = device
        self.entropy_coef = entropy_coef
        self._last_print_mult = 0

    def get_action(self, state):
        action_mu = self.actor_model(state)
        action_std = torch.exp(self.actor_model.log_std).unsqueeze(0).to(self.device)
        dist = torch.distributions.Independent(torch.distributions.Normal(action_mu, action_std), 1)
        z = dist.sample()
        a = torch.tanh(z)
        logp = (dist.log_prob(z) - torch.sum(torch.log(1 - a**2 + 1e-6), dim=-1)).item()
        action_sent = torch.clamp(a, -1.0, 1.0)
        return action_sent.cpu().numpy().squeeze(0), logp
    
    def get_value(self, states, actions):
        action_mu = self.actor_model(states)
        action_std = torch.exp(self.actor_model.log_std).unsqueeze(0).expand_as(action_mu).to(self.device)
        dist = torch.distributions.Independent(torch.distributions.Normal(action_mu, action_std), 1)
        a = torch.tanh(actions)
        log_probs = dist.log_prob(actions) - torch.sum(torch.log(1 - a**2 + 1e-6), dim=-1)
        state_values = self.critic_model(states)
        entropy = dist.entropy().mean()
        return state_values.squeeze(-1), log_probs, entropy

    def evaluate_policy(self, n_eval_episodes=5, render=False):
        rewards = []
        dists = []
        for _ in range(n_eval_episodes):
            state_dict, _ = self.env.reset()
            state = np.concatenate([state_dict['observation'], state_dict['desired_goal']])
            done = False
            total_r = 0.0
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_mu = self.actor_model(state_tensor)
                action = torch.tanh(action_mu).cpu().numpy().squeeze(0)
                action = np.clip(action, -1.0, 1.0)
                next_state_dict, reward, terminated, truncated, _ = self.env.step(action)
                next_state = np.concatenate([next_state_dict['observation'], next_state_dict['desired_goal']])
                done = terminated or truncated
                total_r += reward
                state = next_state
            rewards.append(total_r)
            achieved = next_state_dict.get('achieved_goal', None)
            goal = next_state_dict.get('desired_goal')
            if achieved is not None:
                dists.append(np.linalg.norm(achieved - goal))
        return np.mean(rewards), (np.mean(dists) if len(dists)>0 else None)


    def compute_rtg(self, rewards):
        R = 0
        rtg = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            rtg.insert(0, R)
        return torch.FloatTensor(rtg).to(self.device)

    def compute_gae(self, rewards, values, last_value, lam=0.95):
        advantages = []
        gae = 0
        values = values + [last_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] - values[step]
            gae = delta + self.gamma * lam * gae
            advantages.insert(0, gae)
        return torch.FloatTensor(advantages).to(self.device)
    
    def compute_advantages(self, rewards, values):
        rtg = self.compute_rtg(rewards)
        advantages = rtg - values
        ad_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        return ad_norm

    def ppo_loss(self, old_probs, new_probs, advantages):
        ratio = torch.exp(new_probs - old_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        return loss

    def train(self, n_episodes, batch_size, epochs, plot=True):
        episode_rewards = []
        episode_distances = []
        lam = 0.95
        episode = 0
        while episode < n_episodes:
            # Collect a batch of data (concatenate full episodes) until we have >= batch_size timesteps
            batch_states = []
            batch_actions = []
            batch_old_log_probs = []
            batch_returns = []
            batch_advantages = []
            batch_values_old = []
            timesteps_collected = 0

            while timesteps_collected < batch_size and episode < n_episodes:
                state_dict, _ = self.env.reset()
                state = np.concatenate([state_dict['observation'], state_dict['desired_goal']])
                done = False
                states_ep = []
                actions_ep = []
                rewards_ep = []
                logp_ep = []
                values_ep = []

                while not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action_mu = self.actor_model(state_tensor)
                        value = self.critic_model(state_tensor)
                    action_std = torch.exp(self.actor_model.log_std).unsqueeze(0).to(self.device)
                    dist = torch.distributions.Independent(torch.distributions.Normal(action_mu, action_std), 1)
                    z = dist.sample()
                    a = torch.tanh(z)
                    logp = (dist.log_prob(z) - torch.sum(torch.log(1 - a**2 + 1e-6), dim=-1)).item()
                    action_raw = z.cpu().numpy().squeeze(0)
                    action_sent = np.clip(a.cpu().numpy().squeeze(0), -1.0, 1.0)

                    next_state_dict, reward, terminated, truncated, _ = self.env.step(action_sent)
                    next_state = np.concatenate([next_state_dict['observation'], next_state_dict['desired_goal']])
                    done = terminated or truncated

                    states_ep.append(state)
                    actions_ep.append(action_raw)
                    rewards_ep.append(reward)
                    logp_ep.append(logp)
                    values_ep.append(float(value.cpu().item()))

                    state = next_state

                # episode finished, compute GAE advantages for this episode
                last_value = 0.0
                advantages_ep = self.compute_gae(rewards_ep, values_ep, last_value, lam=lam)
                returns_ep = advantages_ep + torch.FloatTensor(values_ep).to(self.device)

                # append episode data to batch
                batch_states.extend(states_ep)
                batch_actions.extend(actions_ep)
                batch_old_log_probs.extend(logp_ep)
                batch_returns.extend(returns_ep.cpu().numpy().tolist())
                batch_advantages.extend(advantages_ep.cpu().numpy().tolist())
                batch_values_old.extend(values_ep)

                timesteps_collected += len(rewards_ep)
                total_reward = sum(rewards_ep)
                episode_rewards.append(total_reward)
                achieved = next_state_dict['achieved_goal'] if 'achieved_goal' in next_state_dict else None
                goal = next_state_dict['desired_goal']
                if achieved is not None:
                    dist_to_goal = np.linalg.norm(achieved - goal)
                    episode_distances.append(dist_to_goal)
                else:
                    episode_distances.append(None)

                episode += 1

            # convert batch data to tensors
            states_tensor = torch.FloatTensor(np.array(batch_states)).to(self.device)
            actions_tensor = torch.FloatTensor(np.array(batch_actions)).to(self.device)
            old_log_probs_tensor = torch.tensor(batch_old_log_probs, dtype=torch.float32, device=self.device)
            returns_tensor = torch.FloatTensor(np.array(batch_returns)).to(self.device)
            advantages_tensor = torch.FloatTensor(np.array(batch_advantages)).to(self.device)
            # normalize advantages
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-10)

            # values_old used for value clipping (detach)
            values_old_tensor = torch.FloatTensor(np.array(batch_values_old)).to(self.device).detach()

            # perform PPO updates using minibatches
            total_timesteps = states_tensor.shape[0]
            minibatch_size = min(64, total_timesteps)
            indices = np.arange(total_timesteps)

            for _ in range(epochs):
                np.random.shuffle(indices)
                for start in range(0, total_timesteps, minibatch_size):
                    mb_idx = indices[start:start+minibatch_size]
                    mb_states = states_tensor[mb_idx]
                    mb_actions = actions_tensor[mb_idx]
                    mb_old_log_probs = old_log_probs_tensor[mb_idx]
                    mb_returns = returns_tensor[mb_idx]
                    mb_advantages = advantages_tensor[mb_idx]
                    mb_values_old = values_old_tensor[mb_idx]

                    # compute current policy values and log probs
                    action_mu = self.actor_model(mb_states)
                    action_std = torch.exp(self.actor_model.log_std).unsqueeze(0).expand_as(action_mu).to(self.device)
                    dist = torch.distributions.Independent(torch.distributions.Normal(action_mu, action_std), 1)
                    a = torch.tanh(mb_actions)
                    new_log_probs = dist.log_prob(mb_actions) - torch.sum(torch.log(1 - a**2 + 1e-6), dim=-1)
                    entropy = dist.entropy().mean()

                    actor_loss = self.ppo_loss(mb_old_log_probs, new_log_probs, mb_advantages)

                    values = self.critic_model(mb_states)
                    # Ensure values and returns are both [batch_size, 1]
                    if values.dim() == 1:
                        values = values.unsqueeze(-1)
                    mb_returns_reshaped = mb_returns.unsqueeze(-1) if mb_returns.dim() == 1 else mb_returns
                    mb_values_old_reshaped = mb_values_old.unsqueeze(-1) if mb_values_old.dim() == 1 else mb_values_old
                    values_clipped = mb_values_old_reshaped + torch.clamp(values - mb_values_old_reshaped, -self.clip_ratio, self.clip_ratio)
                    critic_loss_unclipped = torch.nn.MSELoss()(values, mb_returns_reshaped)
                    critic_loss_clipped = torch.nn.MSELoss()(values_clipped, mb_returns_reshaped)
                    critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped)

                    actor_loss_total = actor_loss - self.entropy_coef * entropy
                    critic_loss_total = critic_loss

                    # accumulate per-update stats
                    try:
                        actor_val = actor_loss.item()
                    except Exception:
                        actor_val = float(actor_loss)
                    try:
                        critic_val = critic_loss.item()
                    except Exception:
                        critic_val = float(critic_loss)
                    try:
                        ent_val = entropy.item()
                    except Exception:
                        ent_val = float(entropy)

                    if not (torch.isnan(actor_loss_total) or torch.isinf(actor_loss_total)):
                        if 'update_actor_sum' not in locals():
                            update_actor_sum = 0.0
                            update_critic_sum = 0.0
                            update_entropy_sum = 0.0
                            update_count = 0
                        update_actor_sum += actor_val
                        update_critic_sum += critic_val
                        update_entropy_sum += ent_val
                        update_count += 1

                        self.actor_optimizer.zero_grad()
                        actor_loss_total.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_norm=0.5)
                        self.actor_optimizer.step()

                        self.critic_optimizer.zero_grad()
                        critic_loss_total.backward()
                        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_norm=0.5)
                        self.critic_optimizer.step()

            # Print diagnostics every 50 episodes (account for batches that cross multiples)
            total_eps = len(episode_rewards)
            cur_mult = total_eps // 50
            if cur_mult > self._last_print_mult:
                # update last printed multiple
                self._last_print_mult = cur_mult
                recent_rewards = episode_rewards[-50:]
                recent_dists = [d for d in episode_distances[-50:] if d is not None]
                print(f'\n=== Episode {total_eps}/{n_episodes} ===')
                print(f'Avg reward (last 50): {np.mean(recent_rewards):.3f}')
                if len(recent_dists) > 0:
                    print(f'Avg distance (last 50): {np.mean(recent_dists):.4f}')
                # report average per-update stats if available
                if 'update_count' in locals() and update_count > 0:
                    print(f"Update stats (per-update avg): actor_loss={update_actor_sum/update_count:.4f}, critic_loss={update_critic_sum/update_count:.4f}, entropy={update_entropy_sum/update_count:.4f}")
                # report log_std
                try:
                    log_std_mean = float(self.actor_model.log_std.mean().cpu().item())
                    print(f"Policy log_std mean: {log_std_mean:.4f}")
                except Exception:
                    pass
                print(f'Value estimates: mean={values_old_tensor.mean().item():.3f}, std={values_old_tensor.std().item():.3f}')
                print(f'Advantages: mean={advantages_tensor.mean().item():.3f}, std={advantages_tensor.std().item():.3f}')

                # run deterministic evaluation and print
                eval_r, eval_d = self.evaluate_policy(n_eval_episodes=10)
                print(f'Eval (deterministic) over 10 episodes: avg_reward={eval_r:.3f}, avg_dist={eval_d}')
                # clear running update stats so next print shows recent stats
                if 'update_count' in locals():
                    del update_actor_sum, update_critic_sum, update_entropy_sum, update_count
        self.env.close()
        if plot:
            plt.figure(figsize=(12,5))
            plt.subplot(1,2,1)
            plt.plot(episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.subplot(1,2,2)
            plt.plot(episode_distances)
            plt.title('Distance to Goal')
            plt.xlabel('Episode')
            plt.ylabel('Distance')
            plt.tight_layout()
            plt.show()
