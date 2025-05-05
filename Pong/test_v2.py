import gymnasium as gym
import time
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Verificar se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Parte 1: Treinamento
print("Criando ambiente de treinamento...")
env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

# Configurações para usar GPU
print("Criando modelo DQN...")
model = DQN(
    "CnnPolicy", 
    env, 
    verbose=1, 
    learning_rate=1e-4, 
    buffer_size=10000,
    device=device  # Especifica o dispositivo (GPU se disponível)
)

# Treinar o modelo
print(f"Treinando o modelo usando {device}...")
model.learn(total_timesteps=500000)

# Salvar o modelo
model.save("dqn_pong")
print("Modelo salvo com sucesso!")

# Fechar o ambiente de treinamento
env.close()

# Parte 2: Visualização
print("Preparando ambiente para visualização...")

# Recriamos o mesmo ambiente para teste
env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

# Carregar o modelo treinado
print("Carregando modelo treinado...")
model = DQN.load("dqn_pong", device=device)

# Criar o ambiente de renderização
try:
    render_env = gym.make("ALE/Pong-v5", render_mode="human")
except TypeError:
    # Fallback para versões mais antigas do Gymnasium/Gym
    try:
        render_env = gym.make("ALE/Pong-v5", render=True)
    except TypeError:
        render_env = gym.make("ALE/Pong-v5")
        render_env.render_mode = "human"

# Loop de teste
print("Iniciando visualização...")
episodes = 5
for ep in range(episodes):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    # Reset do ambiente de renderização
    render_state = render_env.reset()
    if isinstance(render_state, tuple):
        render_state = render_state[0]
    
    done = False
    total_reward = 0
    
    while not done:
        # Predição com o modelo
        action, _ = model.predict(obs, deterministic=True)
        
        # Executar no ambiente de teste (ambiente vetorizado)
        try:
            # Para ambientes vetorizados, step retorna diferentes formatos
            obs, rewards, dones, infos = env.step(action)
            
            # Em ambientes vetorizados rewards e dones são arrays
            # Como temos apenas 1 ambiente (n_envs=1), pegamos o primeiro elemento
            reward = rewards[0]
            done = dones[0]
            total_reward += reward
            
        except Exception as e:
            print(f"Erro ao executar step no ambiente de teste: {e}")
            break
        
        # Executar no ambiente de renderização (ambiente normal)
        try:
            # Precisamos converter de array para int para o ambiente de renderização
            render_action = int(action[0])
            
            # Para o ambiente de renderização, usamos a API do Gymnasium
            result = render_env.step(render_action)
            
            # Verificar se estamos na API nova ou antiga
            if len(result) == 5:  # API nova: obs, reward, terminated, truncated, info
                _, _, terminated, truncated, _ = result
                render_done = terminated or truncated
            else:  # API antiga: obs, reward, done, info
                _, _, render_done, _ = result
            
            # Se o ambiente de renderização suportar render(), use-o
            try:
                render_env.render()
            except Exception:
                pass  # Renderização já está sendo feita pelo render_mode="human"
                
        except Exception as e:
            print(f"Erro ao renderizar: {e}")
            # Continue mesmo que a renderização falhe
        
        time.sleep(0.01)  # Pequeno atraso
    
    print(f"Episódio {ep+1}/{episodes} - Recompensa total: {total_reward}")

print("Visualização concluída.")
env.close()
render_env.close()