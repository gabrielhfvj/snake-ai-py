import numpy as np
from snake_game import SnakeGame
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt # type: ignore
from sklearn.linear_model import LinearRegression
import time

def train_agent():
    env = SnakeGame()
    state = env.reset()
    state_size = np.prod(state.shape)  # Calcula o tamanho do estado a partir da forma do estado
    action_size = 4  # UP, DOWN, LEFT, RIGHT
    agent = DQNAgent(state_size, action_size)
    rewards = []
    times = []

    plt.ion()  # Ativa o modo interativo para plotagem em tempo real
    fig, ax = plt.subplots()
    line, = ax.plot(times, rewards, label='Total Reward')
    trend_line, = ax.plot([], [], 'r--', label='Linha de Tendência')
    ax.set_xlabel('Tempo (minutos)')
    ax.set_ylabel('Recompensa Total')
    ax.set_title('Progresso do Treinamento')
    ax.legend()

    max_score_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    
    start_time = time.time()
    max_duration = 300  # 1 hora em segundos
    episode_counter = 0
    max_score = float('-inf')  # Inicializa com infinito negativo

    while True:
        episode_start_time = time.time()
        state = env.reset().flatten()  # Achata o estado para a rede neural
        total_reward = 0

        while True:
            action = agent.get_action(state, agent.epsilon)  # Usa o método get_action do DQNAgent
            next_state, reward, done = env.step(action)
            next_state = next_state.flatten()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break
        
        agent.replay(agent.batch_size)
        rewards.append(total_reward)
        max_score = max(max_score, total_reward)  # Atualiza a maior pontuação

        # Calcula o tempo decorrido
        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time / 60  # Converte segundos para minutos
        times.append(elapsed_minutes)

        # Imprime informações de depuração
        print(f"Tentativa: {episode_counter}, Pontuacao: {total_reward}, Tempo corrido: {elapsed_minutes:.2f} minutos")

        # Para o treinamento se 1 hora tiver passado
        if elapsed_time >= max_duration:
            print(f"Tempo máximo atingido. Finalizando o treinamento.")
            break
        
        episode_counter += 1

        # Atualiza o gráfico
        line.set_xdata(times)
        line.set_ydata(rewards)
        ax.relim()
        ax.autoscale_view()
        
        # Calcula e plota a linha de tendência
        if len(times) > 1:
            X = np.array(times).reshape(-1, 1)  # Redefine para LinearRegression
            y = np.array(rewards)
            model = LinearRegression().fit(X, y)
            trend_line.set_xdata(X)
            trend_line.set_ydata(model.predict(X))
        
        # Atualiza o texto da maior pontuação no gráfico
        max_score_text.set_text(f'Maior Pontuação: {max_score}')
        
        plt.draw()
        plt.pause(0.1)  # Pausa para atualizar o gráfico

    plt.ioff()  # Desativa o modo interativo
    plt.show()

    # Imprime a maior pontuação alcançada
    print(f"Maior Pontuação Alcançada: {max_score}")

if __name__ == "__main__":
    train_agent()
