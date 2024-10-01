# Snake AI using Deep Q-Learning
Este projeto treina uma inteligência artificial (IA) para jogar o clássico jogo Snake usando Deep Q-Learning (DQN). O objetivo da IA é maximizar a pontuação ao aprender com suas próprias ações durante os episódios de treinamento.

## Estrutura do Projeto
### Arquivos Principais:
snake_game.py: Implementa o ambiente do jogo Snake, onde a IA interage.
dqn_agent.py: Define o agente que utiliza o modelo DQN para decidir as ações durante o jogo.
dqn_model.py: Contém a arquitetura da rede neural que será treinada para o agente tomar decisões.
train.py: Script de treinamento que executa o ciclo de treinamento da IA, salva o modelo e plota gráficos dos resultados.

### Dependências
O projeto requer as seguintes bibliotecas:

numpy
pygame
torch
matplotlib

Instale todas as dependências com:

bash
pip install -r requirements.txt

### Como Rodar o Projeto
Clone o repositório para o seu ambiente local.
Certifique-se de ativar seu ambiente virtual e instalar as dependências.
Para iniciar o treinamento, execute:

bash
python train.py

### Treinamento da IA
O treinamento da IA é realizado com o algoritmo Deep Q-Learning. A IA aprende jogando múltiplos episódios e ajustando suas ações para maximizar o total de recompensas (pontuação no jogo).

Estados: A posição da cabeça da cobra, a localização da maçã, e a direção da cobra.
Ações: A cobra pode mover-se em quatro direções (cima, baixo, esquerda, direita).
Recompensa: A IA recebe uma recompensa positiva por comer uma maçã e negativa ao bater nas paredes ou em si mesma.

### Resultados do Treinamento
Os gráficos de desempenho são gerados automaticamente durante o treinamento. Eles mostram:

A pontuação total de cada episódio.
O máximo score que a IA conseguiu durante o processo.
Uma linha de tendência para indicar a evolução do aprendizado da IA.
Adicione aqui alguns prints de exemplo mostrando os resultados do treinamento:

Exemplo de Print:

### Maiores Pontuações Alcançadas:
Ao final do treinamento, o programa também exibe a maior pontuação que a IA conseguiu alcançar durante os episódios.

Melhor pontuação: ### (substituir com o valor gerado ao final)
Adicione aqui um print mostrando a maior pontuação alcançada:


### Parâmetros de Treinamento
O treinamento padrão está configurado para rodar por no máximo 1 hora. O eixo X no gráfico corresponde ao tempo de execução (em minutos).

Você pode ajustar os seguintes parâmetros no código:

Número máximo de jogos (max_games)
Exploração inicial (epsilon inicial) e seu decaimento
Tamanho do batch de replay (batch_size)
Tamanho da memória de replay (memory_size)

### Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests para melhorar o projeto.

### Licença
Este projeto é licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

