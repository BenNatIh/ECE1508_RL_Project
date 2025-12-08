import torch
import numpy as np
import matplotlib.pyplot as plt

class Agent():
    def __init__(self, env, state_dim, action_dim, gamma, actor_model, critic_model, actor_optimizer, critic_optimizer, device='cpu', entropy_coef=0.0, her_k=4, use_her=True):
        self.gamma = gamma  # Use 1.0 for sparse binary rewards
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
        self.her_k = her_k
        self.use_her = use_her
        self._last_print_mult = 0
        # history for diagnostics (only eval metrics kept to save memory)
        self.history = {
            'eval_rewards': [],  # track eval rewards (or avg steps) during training
            'eval_episodes': [],  # track episode numbers when evals occur
            'eval_success_rate': [],  # track success rate during evals
            'train_success_flags': []  # track per-episode success during training
        }

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

    def evaluate_policy(self, n_eval_episodes=5, render=False, env_override=None):
        rewards = []
        steps_to_success = []
        env_to_use = env_override if env_override is not None else self.env

        step_counts = []
        for _ in range(n_eval_episodes):
            state_dict, _ = env_to_use.reset()
            # use non-noisy obs for eval
            obs = state_dict.get('observation', None)
            goal = state_dict.get('desired_goal', None)
            if 'observation_clean' in state_dict:
                obs = state_dict['observation_clean']
            state = np.concatenate([obs, goal])
            done = False
            total_r = 0.0
            step_count = 0
            success_step = None
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_mu = self.actor_model(state_tensor)
                action = torch.tanh(action_mu).cpu().numpy().squeeze(0)
                action = np.clip(action, -1.0, 1.0)
                next_state_dict, reward, terminated, truncated, info = env_to_use.step(action)
                next_goal = next_state_dict.get('desired_goal', None)
                #use non noisy obs for eval
                if 'observation_clean' in next_state_dict:
                    next_obs = next_state_dict['observation_clean']
                else:
                    next_obs = next_state_dict.get('observation', None)
                next_state = np.concatenate([next_obs, next_goal])
                done = terminated or truncated
                total_r += reward
                # end episode early if success detected
                step_count += 1
                if info is not None and isinstance(info, dict) and info.get('is_success', False):
                    done = True
                    terminated = True
                    if success_step is None:
                        success_step = step_count
                    # stop the episode immediately so length reflects success
                    break
                state = next_state

            rewards.append(total_r)
            # record steps-to-success (None if never succeeded) and episode length
            steps_to_success.append(success_step)
            step_counts.append(step_count)

        avg_reward = np.mean(rewards) if len(rewards) > 0 else 0.0
        # average episode length (always use episode length; termination may indicate success)
        avg_episode_length = float(np.mean(step_counts)) if len(step_counts) > 0 else 0.0
        # success rate (fraction of eval episodes that reported success)
        succ_count = len([s for s in steps_to_success if s is not None])
        success_rate = float(succ_count) / float(n_eval_episodes) if n_eval_episodes > 0 else 0.0
        return avg_reward, avg_episode_length, success_rate


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

    def her(self, achieved_ep, obs_only_ep, actions_ep, terminated, lam, max_relabels,
                   batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages, batch_values_old,
                   batch_relabeled_count):
        available_idxs = [i for i, a in enumerate(achieved_ep) if a is not None]
        if not available_idxs:
            return batch_relabeled_count

        k = min(self.her_k, len(available_idxs))
        sampled_idxs = np.random.choice(available_idxs, size=k, replace=False)

        for idx in sampled_idxs:
            if batch_relabeled_count >= max_relabels:
                break

            new_goal = achieved_ep[idx]
            T = min(idx + 1, len(obs_only_ep), len(actions_ep))
            if T < 2:
                continue

            rel_states = [np.concatenate([obs_only_ep[i], new_goal]) for i in range(T)]

            rel_rewards = []
            for t in range(T):
                ach_t = achieved_ep[t]
                if ach_t is None:
                    rel_rewards.append(0.0)
                else:
                    rel_rewards.append(self.env.unwrapped.compute_reward(ach_t, new_goal, {}))

            rel_states_t = torch.FloatTensor(np.array(rel_states)).to(self.device)
            rel_actions_t = torch.FloatTensor(np.array(actions_ep[:T])).to(self.device)

            with torch.no_grad():
                rel_mu = self.actor_model(rel_states_t)
                rel_std = torch.exp(self.actor_model.log_std).unsqueeze(0).expand_as(rel_mu).to(self.device)
                rel_dist = torch.distributions.Independent(torch.distributions.Normal(rel_mu, rel_std), 1)
                rel_actions_tanh = torch.tanh(rel_actions_t)
                rel_logp_tensor = rel_dist.log_prob(rel_actions_t) - torch.sum(torch.log(1 - rel_actions_tanh**2 + 1e-6), dim=-1)
                rel_vals_tensor = self.critic_model(rel_states_t).squeeze(-1)

            rel_logp = rel_logp_tensor.detach().cpu().numpy().tolist()
            rel_vals = rel_vals_tensor.detach().cpu().numpy().tolist()

            rel_last_val = 0.0 if terminated else (rel_vals[-1] if len(rel_vals) > 0 else 0.0)

            rel_adv = self.compute_gae(rel_rewards, rel_vals, rel_last_val, lam=lam)
            rel_ret = self.compute_rtg(rel_rewards)

            n_add = min(T, max_relabels - batch_relabeled_count)
            batch_states.extend(rel_states[:n_add])
            batch_actions.extend(actions_ep[:n_add])
            batch_old_log_probs.extend(rel_logp[:n_add])
            batch_returns.extend(rel_ret.cpu().numpy().tolist()[:n_add])
            batch_advantages.extend(rel_adv.cpu().numpy().tolist()[:n_add])
            batch_values_old.extend(rel_vals[:n_add])
            batch_relabeled_count += n_add

        return batch_relabeled_count
    
    def ppo_loss(self, old_probs, new_probs, advantages):
        ratio = torch.exp(new_probs - old_probs)
        if self.clip_ratio is None:
            return -(ratio * advantages).mean()
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
            # diagnostic: cap for relabeled transitions per batch (avoid overwhelming on-policy data)
            batch_relabeled_count = 0
            max_relabels = batch_size // 2

            while timesteps_collected < batch_size and episode < n_episodes:
                state_dict, _ = self.env.reset()
                state = np.concatenate([state_dict['observation'], state_dict['desired_goal']])
                done = False
                success = False
                # per-episode storage
                states_ep = []
                obs_only_ep = []
                achieved_ep = []
                actions_ep = []
                rewards_ep = []
                logp_ep = []
                values_ep = []
                episode_goal = state_dict.get('desired_goal', None)

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

                    next_state_dict, reward, terminated, truncated, info = self.env.step(action_sent)
                    next_state = np.concatenate([next_state_dict['observation'], next_state_dict['desired_goal']])
                    done = terminated or truncated

                    # Detect success and optionally end early for sparse tasks (HER)
                    info_success = bool(info is not None and isinstance(info, dict) and info.get('is_success', False))

                    if info_success:
                        success = True
                        # Early termination on success - ONLY for sparse tasks
                        # Dense tasks need full trajectories to learn precise placement
                        if self.use_her:
                            done = True
                            terminated = True

                    # store episode data
                    states_ep.append(state)
                    obs_only_ep.append(next_state_dict['observation'].copy())
                    achieved = next_state_dict.get('achieved_goal')
                    achieved_ep.append(achieved.copy() if achieved is not None else None)
                    
                    actions_ep.append(action_raw)
                    rewards_ep.append(reward)
                    logp_ep.append(logp)
                    values_ep.append(float(value.cpu().item()))

                    state = next_state

                # episode finished, compute GAE advantages for this episode
                # bootstrap last value from critic when episode was truncated (not terminal)
                if terminated:
                    last_value = 0.0
                else:
                    state_tensor_last = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        last_value = float(self.critic_model(state_tensor_last).cpu().item())

                advantages_ep = self.compute_gae(rewards_ep, values_ep, last_value, lam=lam)
                # Compute proper returns from rewards, not from advantages
                returns_ep = self.compute_rtg(rewards_ep)

                # append episode data to batch
                returns_list = returns_ep.cpu().numpy().tolist()
                adv_list = advantages_ep.cpu().numpy().tolist()
                batch_states.extend(states_ep)
                batch_actions.extend(actions_ep)
                batch_old_log_probs.extend(logp_ep)
                batch_returns.extend(returns_list)
                batch_advantages.extend(adv_list)
                batch_values_old.extend(values_ep)

                # Optional success duplication removed; keeping batch as collected

                # Only attempt HER if episode did NOT succeed and goal is available
                if self.use_her and not success and episode_goal is not None and any(a is not None for a in achieved_ep):
                    batch_relabeled_count = self.her(
                        achieved_ep,
                        obs_only_ep,
                        actions_ep,
                        terminated,
                        lam,
                        max_relabels,
                        batch_states,
                        batch_actions,
                        batch_old_log_probs,
                        batch_returns,
                        batch_advantages,
                        batch_values_old,
                        batch_relabeled_count
                    )

                timesteps_collected += len(rewards_ep)
                total_reward = sum(rewards_ep)
                episode_rewards.append(total_reward)
                
                # Track training success rate
                self.history['train_success_flags'].append(success)
                
                # use clean achieved_goal for accurate distance metrics
                achieved = next_state_dict.get('achieved_goal_clean', next_state_dict.get('achieved_goal', None))
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
            # keep raw (unnormalized) advantages for diagnostics
            advantages_tensor_raw = torch.FloatTensor(np.array(batch_advantages)).to(self.device)
            raw_adv_mean = advantages_tensor_raw.mean()
            raw_adv_std = advantages_tensor_raw.std()
            # normalize advantages for training (keep numerical stability)
            advantages_tensor = (advantages_tensor_raw - raw_adv_mean) / (raw_adv_std + 1e-10)
            
            # values_old used for value clipping (detach)
            values_old_tensor = torch.FloatTensor(np.array(batch_values_old)).to(self.device).detach()

            # perform PPO updates using minibatches
            total_timesteps = states_tensor.shape[0]
            minibatch_size = min(64, total_timesteps)
            indices = np.arange(total_timesteps)

            # initialize per-batch update accumulators for diagnostics
            update_actor_sum = 0.0
            update_critic_sum = 0.0
            update_entropy_sum = 0.0
            update_count = 0
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

                    # compute current policy values, log probs, and entropy (reuse helper)
                    values, new_log_probs, entropy = self.get_value(mb_states, mb_actions)

                    # basic NaN/Inf guard
                    if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any() or \
                       torch.isnan(values).any() or torch.isinf(values).any():
                        print("\nWARNING: NaN/Inf detected in value/log_prob. Skipping this minibatch.")
                        continue

                    actor_loss = self.ppo_loss(mb_old_log_probs, new_log_probs, mb_advantages)

                    values = self.critic_model(mb_states)
                    # Ensure values and returns are both [batch_size, 1]
                    if values.dim() == 1:
                        values = values.unsqueeze(-1)
                    mb_returns_reshaped = mb_returns.unsqueeze(-1) if mb_returns.dim() == 1 else mb_returns
                    mb_values_old_reshaped = mb_values_old.unsqueeze(-1) if mb_values_old.dim() == 1 else mb_values_old
                    critic_loss_unclipped = torch.nn.MSELoss()(values, mb_returns_reshaped)
                    if self.clip_ratio is not None:
                        values_clipped = mb_values_old_reshaped + torch.clamp(values - mb_values_old_reshaped, -self.clip_ratio, self.clip_ratio)
                        critic_loss_clipped = torch.nn.MSELoss()(values_clipped, mb_returns_reshaped)
                        critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped)
                    else:
                        critic_loss = critic_loss_unclipped

                    actor_loss_total = actor_loss - self.entropy_coef * entropy
                    critic_loss_total = critic_loss
                    
                    # Check for NaN in losses before backprop
                    if torch.isnan(actor_loss_total) or torch.isinf(actor_loss_total) or torch.isnan(critic_loss_total) or torch.isinf(critic_loss_total):
                        print(f"\nWARNING: NaN/Inf in loss. Skipping update. actor_loss={actor_loss_total}, critic_loss={critic_loss_total}")
                        continue

                    # accumulate per-update stats
                    actor_val = float(actor_loss.item())
                    critic_val = float(critic_loss.item())
                    ent_val = float(entropy.item())

                    update_actor_sum += actor_val
                    update_critic_sum += critic_val
                    update_entropy_sum += ent_val
                    update_count += 1

                    # (No per-update history stored to reduce memory/IO)

                    self.actor_optimizer.zero_grad()
                    actor_loss_total.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_norm=0.5)
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    critic_loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_norm=0.5)
                    self.critic_optimizer.step()

            # Print diagnostics and run evaluation every 50 episodes
            total_eps = len(episode_rewards)
            cur_mult = total_eps // 50
            if cur_mult > self._last_print_mult:
                # update last printed multiple
                self._last_print_mult = cur_mult
                recent_rewards = episode_rewards[-50:]
                recent_dists = [d for d in episode_distances[-50:] if d is not None]
                # Calculate batch composition
                on_policy_count = timesteps_collected
                her_count = batch_relabeled_count if 'batch_relabeled_count' in locals() else 0
                total_batch = len(batch_states) if 'batch_states' in locals() else 0
                print(f'\n=== Episode {total_eps}/{n_episodes} ===')
                print(f'Batch composition: {on_policy_count} on-policy, {her_count} HER, {total_batch} total')
                print(f'Avg reward (last 50): {np.mean(recent_rewards):.3f}')
                # Calculate training success rate over last 50 episodes
                if len(self.history['train_success_flags']) >= 50:
                    recent_train_success = self.history['train_success_flags'][-50:]
                    train_success_rate = sum(recent_train_success) / len(recent_train_success)
                    print(f'Training success rate (last 50): {train_success_rate:.2f}')
                if len(recent_dists) > 0:
                    print(f'Avg distance (last 50): {np.mean(recent_dists):.4f}')
                # report average per-update stats if available
                if 'update_count' in locals() and update_count > 0:
                    avg_actor = update_actor_sum/update_count
                    avg_critic = update_critic_sum/update_count
                    avg_entropy = update_entropy_sum/update_count
                    print(f"Update stats: actor_loss={avg_actor:.4f}, critic_loss={avg_critic:.4f}, entropy={avg_entropy:.4f}")
                # report log_std
                log_std_mean = float(self.actor_model.log_std.mean().cpu().item())
                print(f"Policy log_std mean: {log_std_mean:.4f}")
                print(f'Value estimates: mean={values_old_tensor.mean().item():.3f}, std={values_old_tensor.std().item():.3f}')
                # show raw (unnormalized) advantages for diagnostics
                print(f'Advantages (raw): mean={raw_adv_mean.item():.6f}, std={raw_adv_std.item():.6f}')
                # report how many relabeled transitions were added in this batch (if tracked)
                if 'batch_relabeled_count' in locals():
                    print(f'Relabeled transitions added in batch: {batch_relabeled_count}')

                # run deterministic evaluation and print (returns reward, avg_episode_length, success_rate)
                eval_r, eval_len, eval_success = self.evaluate_policy(n_eval_episodes=10)
                # determine whether this environment is Dense (measure episode length) or Sparse (measure reward)
                env_id = self.env.unwrapped.spec.id

                if 'Dense' in env_id:
                    # Dense env: use average episode length (shorter means earlier termination)
                    print(f'Eval (deterministic) over 10 episodes: avg_episode_length={eval_len:.2f}, avg_reward={eval_r:.3f}, success_rate={eval_success:.2f}')
                    metric_val = float(eval_len)
                else:
                    # Sparse env: use average reward
                    print(f'Eval (deterministic) over 10 episodes: avg_reward={eval_r:.3f}, avg_episode_length={eval_len:.2f}, success_rate={eval_success:.2f}')
                    metric_val = float(eval_r)

                # track eval metric for plotting (avg reward for sparse, avg steps for dense)
                self.history['eval_rewards'].append(metric_val)
                self.history['eval_episodes'].append(total_eps)
                self.history['eval_success_rate'].append(float(eval_success))
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
