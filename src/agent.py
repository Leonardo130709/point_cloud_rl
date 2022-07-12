import torch
from torch.nn.utils import clip_grad_norm_
from pytorch3d.loss import chamfer_distance
from . import models, utils

td = torch.distributions
nn = torch.nn
F = nn.functional


class SAC(nn.Module):
    def __init__(self, env, config, callback):
        super().__init__()
        self.observation_spec = env.observation_spec()
        self.action_spec = env.action_spec()
        self.callback = callback
        self._c = config
        self._step = 0
        self._build()

    @torch.no_grad()
    def policy(self, observation, training):
        state = self.encoder(observation)
        dist = self.actor(state)

        if training:
            action = dist.sample()
        else:
            action = torch.tanh(dist.base_dist.mean)

        return action

    def step(self, observations, actions, rewards, dones, next_observations):
        states = self.encoder(observations)
        next_states = self.encoder(observations)
        target_next_states = self._target_encoder(next_observations)

        alpha = torch.maximum(self._log_alpha, torch.full_like(self._log_alpha, -18.))
        alpha = F.softplus(alpha) + 1e-8

        policy = self.actor(next_states.detach())
        next_actions = policy.rsample()
        next_log_probs = policy.log_prob(next_actions)

        critic_loss = self._policy_learning(states,
                                            actions,
                                            rewards,
                                            dones,
                                            target_next_states,
                                            next_actions,
                                            next_log_probs,
                                            alpha)

        auxiliary_loss = self._auxiliary_loss(observations, next_states) # TODO: update method

        actor_loss, dual_loss = self._policy_improvement(next_states.detach(),
                                                         next_actions,
                                                         next_log_probs,
                                                         alpha)

        loss = critic_loss + auxiliary_loss + actor_loss + dual_loss

        self.optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.actor.parameters(), self._c.max_grad)
        clip_grad_norm_(self.critic.parameters(), self._c.max_grad)
        clip_grad_norm_(self.encoder.parameters(), self._c.max_grad)
        self.optim.step()

        if self._c.debug:
            self.callback.add_scalar('train/actor_loss', actor_loss, self._step)
            self.callback.add_scalar('train/auxiliary_loss', auxiliary_loss, self._step)
            self.callback.add_scalar('train/critic_loss', critic_loss, self._step)
            self.callback.add_scalar('train/actor_grads', utils.grads_sum(self.actor), self._step)
            self.callback.add_scalar('train/critic_grads', utils.grads_sum(self.critic), self._step)
            self.callback.add_scalar(
                'train/encoder_grads', utils.grads_sum(self.encoder), self._step)
        self._update_targets()
        self._step += 1

        return loss.item()

    def _policy_learning(
            self,
            states,
            actions,
            rewards,
            dones,
            next_states,
            next_actions,
            next_log_probs,
            alpha
    ):
        del dones  # not used for continuous control tasks
        with torch.no_grad():

            q_values = self._target_critic(
                next_states,
                next_actions
            ).min(-1, keepdim=True).values

            soft_values = q_values - alpha * next_log_probs.unsqueeze(-1)
            target_q_values = rewards + self._c.discount*soft_values

        q_values = self.critic(states, actions)
        loss = .5 * (q_values - target_q_values).pow(2)

        if self._c.debug:
            self.callback.add_scalar('train/mean_reward',
                                     rewards.detach().mean() / self._c.action_repeat, self._step)
            self.callback.add_scalar('train/mean_value', q_values.detach().mean(), self._step)

        return loss.mean()

    def _policy_improvement(
            self,
            states,
            actions,
            log_probs,
            alpha
    ):
        self.critic.requires_grad_(False)
        q_values = self.critic(
            states,
            actions
        ).min(-1).values
        self.critic.requires_grad_(True)

        actor_loss = alpha.detach() * log_probs - q_values
        dual_loss = - alpha * (log_probs.detach() + self._target_entropy)

        if self._c.debug:
            ent = -log_probs.detach().mean()
            self.callback.add_scalar('train/actor_entropy', ent, self._step)
            self.callback.add_scalar('train/alpha', alpha.detach().mean(), self._step)

        return actor_loss.mean(), dual_loss.mean()

    def _auxiliary_loss(self, obs, states_emb):
        # todo check l2 reg; introduce lagrange multipliers
        if self._c.aux_loss == 'None':
            return torch.tensor(0.)
        elif self._c.aux_loss == 'reconstruction':
            obs_pred = self.decoder(states_emb)
            loss = chamfer_distance(obs.flatten(0, 2), obs_pred.flatten(0, 2))[0]
            return self._c.reconstruction_coef * loss.mean()
        else:
            raise NotImplementedError

    def _build(self):
        emb = self._c.obs_emb_dim
        act_dim = self.action_spec.shape[0]
        self.device = torch.device(self._c.device if torch.cuda.is_available() else 'cpu')

        # RL
        self.actor = models.Actor(emb, act_dim, self._c.actor_layers)

        self.critic = models.Critic(emb + act_dim, self._c.critic_layers)

        # Encoder+decoder
        frames_stack, pn_number, in_channels = self.observation_spec.shape
        self.encoder = models.PointCloudEncoder(
            in_channels,
            num_frames=frames_stack,
            out_features=emb,
            layers=self._c.pn_layers,
            features_from_layers=self._c.features_from_layers
        )
        self.decoder = models.PointCloudDecoder(
            emb,
            layers=self._c.pn_layers,
            pn_number=self._c.pn_number,
            num_frames=frames_stack,
            out_channels=in_channels
        )

        init_log_alpha = torch.log(torch.tensor(self._c.init_temperature).exp() - 1.)
        self._log_alpha = nn.Parameter(init_log_alpha)

        self._target_encoder, self._target_critic = \
            utils.make_targets(self.encoder, self.critic)

        self.optim = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': self._c.actor_lr},
            {'params': self.critic.parameters(), 'lr': self._c.critic_lr},
            {
                'params': list(self.encoder.parameters()) + list(self.decoder.parameters()),
                'lr': self._c.ae_lr,
                'weight_decay': self._c.weight_decay
             },
            {'params': [self._log_alpha], 'lr': self._c.dual_lr, 'betas': (0.5, .999)}
        ])

        self._target_entropy = self._c.target_ent_per_dim * act_dim

        def weight_init(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)

        self.apply(weight_init)

        self.to(self.device)

    @torch.no_grad()
    def _update_targets(self):
        utils.soft_update(self._target_encoder, self.encoder, self._c.encoder_tau)
        utils.soft_update(self._target_critic, self.critic, self._c.critic_tau)
