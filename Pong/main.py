import gymnasium as gym
import numpy as np
import time
import cv2
from stable_baselines3 import DQN
from collections import deque

# Carregar o modelo DQN treinado
print("Carregando modelo treinado...")
model = DQN.load("dqn_pong.zip")

# Criar ambiente para renderização
print("Iniciando ambiente para visualização...")
env = gym.make("ALE/Pong-v5", render_mode="human")
obs, _ = env.reset()

# Função para processar o frame: RGB -> Cinza -> 84x84
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized

# Rodar alguns episódios
episodes = 5
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    # Inicializar pilha de frames
    frame_stack = deque(maxlen=4)
    initial_frame = preprocess_frame(obs)
    for _ in range(4):
        frame_stack.append(initial_frame)

    while not done:
        # Empilhar frames na ordem (shape: 4 x 84 x 84)
        stacked_frames = np.stack(frame_stack, axis=0)
        stacked_frames = stacked_frames.reshape((1, 4, 84, 84))
        stacked_frames = stacked_frames.astype(np.uint8)

        # Predizer ação com o modelo
        action, _ = model.predict(stacked_frames, deterministic=True)
        action = int(action)  # Corrige erro de tipo

        # Executar ação no ambiente
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Atualizar a pilha de frames
        processed = preprocess_frame(obs)
        frame_stack.append(processed)

        # Pequena pausa para visualização
        time.sleep(0.01)

    print(f"Episódio {ep + 1} finalizado. Recompensa total: {total_reward}")

env.close()
print("Visualização finalizada.")
