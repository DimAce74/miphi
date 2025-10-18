# PPO (Proximal Policy Optimization) implementation from CleanRL
# Docs: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
# Based on "The 37 Implementation Details of PPO" — includes all core tricks for stable training

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    # 🧪 Экспериментальные настройки
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """Имя эксперимента — по умолчанию имя файла"""
    seed: int = 1
    """Фиксированный seed для воспроизводимости"""
    torch_deterministic: bool = True
    """Включает детерминированное поведение PyTorch (медленнее, но воспроизводимо)"""
    cuda: bool = True
    """Использовать GPU, если доступен"""
    track: bool = False
    """Логировать в Weights & Biases"""
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    """Записывать видео эпизодов агента"""

    # 🎮 Настройки среды и алгоритма
    env_id: str = "CartPole-v1"
    """Среда из Gymnasium"""
    total_timesteps: int = 500000
    """Общее число шагов взаимодействия со средой"""
    learning_rate: float = 2.5e-4
    """Начальный learning rate для Adam"""
    num_envs: int = 4
    """Количество параллельных сред — ускоряет сбор данных"""
    num_steps: int = 128
    """Длина rollout'а в каждой среде (T)"""
    anneal_lr: bool = True
    """Постепенно уменьшать learning rate до нуля к концу обучения"""
    gamma: float = 0.99
    """Коэффициент дисконтирования γ"""
    gae_lambda: float = 0.95
    """λ для Generalized Advantage Estimation (GAE) — компромисс между bias и variance"""
    num_minibatches: int = 4
    """Количество минибатчей на один rollout"""
    update_epochs: int = 4
    """Сколько раз переиспользовать один и тот же буфер для обучения (K эпох)"""
    norm_adv: bool = True
    """Нормализовать advantage: (A - mean) / (std + ε) — стабилизирует обучение"""
    clip_coef: float = 0.2
    """Коэффициент clipping'а ε в PPO — обычно 0.1–0.3"""
    clip_vloss: bool = True
    """Clipping и для value loss — как в оригинальной статье PPO"""
    ent_coef: float = 0.01
    """Коэффициент энтропийного бонуса — поощряет exploration"""
    vf_coef: float = 0.5
    """Вес лосса критика в общей функции потерь"""
    max_grad_norm: float = 0.5
    """Максимальная норма градиента — предотвращает exploding gradients"""
    target_kl: float = None
    """Порог KL-дивергенции для early stopping (None = отключено)"""

    # 📏 Вычисляются автоматически при запуске
    batch_size: int = 0
    """Общий размер буфера: num_envs * num_steps"""
    minibatch_size: int = 0
    """Размер одного минибатча"""
    num_iterations: int = 0
    """Число итераций обучения: total_timesteps / batch_size"""


def make_env(env_id, idx, capture_video, run_name):
    """Фабрика для создания среды. Используется для векторизации."""
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        # Обёртка для автоматической записи статистики эпизодов
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Ортогональная инициализация весов — улучшает градиентный поток в начале обучения."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Агент с отдельными сетями для актора (политики) и критика (ценности)."""
    def __init__(self, envs):
        super().__init__()
        # Критик: оценивает V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # std=1.0 для выхода ценности
        )
        # Актор: выдаёт logits для дискретных действий
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),  # std=0.01 — слабая инициализация
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """Возвращает действие, logprob, энтропию и ценность."""
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()  # для сбора данных
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class RPOAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else: # new to RPO
            # sample again to add stochasticity, for the policy update
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)


if __name__ == "__main__":
    args = tyro.cli(Args)
    # 📐 Вычисляем размеры буферов
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # 🔒 Фиксируем seed для воспроизводимости
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 🌐 Создаём векторизованные среды (параллельные экземпляры)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Только дискретные действия"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)  # eps=1e-5 — как в OpenAI Baselines

    # 🗃️ Буферы для хранения данных rollout'а
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)  # ← это π_old!
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # 🚀 Запуск обучения
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # 📉 Уменьшение learning rate (если включено)
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # 📥 Сбор данных (rollout)
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Получаем действие и лог-вероятность БЕЗ градиентов (on-policy!)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob  # ← сохраняем π_old(a|s) для PPO loss!

            # Выполняем шаг в среде
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Логируем завершённые эпизоды
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # 🔁 Вычисление advantage с помощью GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # TD-ошибка
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                # Рекурсивное вычисление GAE
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values  # Return = Advantage + Value

        # 🧱 Преобразуем [T, N, ...] → [T*N, ...] для обучения
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)      # ← π_old (фиксировано!)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 🔁 Обучение: K эпох на одном буфере
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)  # перемешиваем для минибатчей
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Получаем НОВЫЕ logprob и value для тех же (s, a)
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]  # log(π_θ / π_old)
                ratio = logratio.exp()  # r_t(θ) = π_θ(a|s) / π_old(a|s)

                # 📊 Метрики для отладки
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()  # k1 estimator KL
                    approx_kl = ((ratio - 1) - logratio).mean()  # k3 estimator — точнее
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 📉 PPO Clipped Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # ← берём худший случай (conservative update)

                # 📉 Value Loss (с clipping'ом, как в статье)
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                # 💡 Общая функция потерь: policy + value + энтропия
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)  # ← предотвращает большие шаги
                optimizer.step()

            # 🛑 Early stopping по KL (если задан порог)
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # 📈 Метрики качества критика
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # 📊 Логирование в TensorBoard
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)  # доля clipped обновлений
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()