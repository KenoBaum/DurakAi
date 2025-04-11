import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from collections import deque
import copy
import time


MEMORY_SIZE = 10000
BATCH_SIZE = 32  
GAMMA = 0.95  
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.0005
UPDATE_TARGET_EVERY = 10  
MIN_REPLAY_MEMORY_SIZE = 500  
DEBUG = False

class Player:
    rank_order = {"6": 1, "7": 2, "8": 3, "9": 4, "10": 5, "J": 6, "Q": 7, "K": 8, "A": 9}
    
    def __init__(self, player_id, name, is_ai=False):
        self.player_id = player_id
        self.name = name
        self.hand = []
        self.is_ai = is_ai
        
    def show_hand(self):
        return ", ".join(self.hand)
    
    def remove_card(self, card):
        if card in self.hand:
            self.hand.remove(card)
            return True
        else:
            if DEBUG:
                print(f"FEHLER: Karte {card} nicht in Hand von {self.name}: {self.hand}")
            return False
    
    def add_cards(self, cards):
        self.hand.extend(cards)
    
    def get_valid_attack_cards(self, field):
        if not field:
            return self.hand
        
        possible_ranks = [card[:-1] for card in field if card != "PASS"]
        return [card for card in self.hand if card[:-1] in possible_ranks]
    
    def get_valid_defense_cards(self, attack_card, trumpf_suit):
        if attack_card == "PASS":
            return []
            
        attack_rank = self.rank_order[attack_card[:-1]]
        attack_suit = attack_card[-1]
        
        valid_cards = []
        for card in self.hand:
            if card == "PASS":
                continue
                
            defense_rank = self.rank_order[card[:-1]]
            defense_suit = card[-1]

            if defense_suit == attack_suit and defense_rank > attack_rank:
                valid_cards.append(card)
            elif defense_suit == trumpf_suit and attack_suit != trumpf_suit:
                valid_cards.append(card)
                
        return valid_cards

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_tmodel()
        
    def build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mse')
        return model
    
    def update_tmodel(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions):
        if not valid_actions:
            if DEBUG:
                print("keine gültige Aktion vorhanden")
            return None

        if np.random.rand() <= self.epsilon:
            chosen_action = random.choice(valid_actions)
            return chosen_action

        try:
            state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            act_values = self.model(state_tensor, training=False).numpy()[0]
            
            masked_values = np.ones(self.action_size) * -1000.0 
            for action in valid_actions:
                masked_values[action] = act_values[action]
                
            chosen_action = np.argmax(masked_values)
            return chosen_action
        except Exception as e:
            print(f"Error in act: {e}")
            chosen_action = random.choice(valid_actions)
            print(f"Fallback zu zufälliger Aktion: {chosen_action}")
            return chosen_action
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        try:
            minibatch = random.sample(self.memory, batch_size)
            states = np.zeros((batch_size, self.state_size))
            targets = np.zeros((batch_size, self.action_size))
            
            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                states[i] = state
                target = self.model.predict(np.array([state]), verbose=0)[0]
                
                if done:
                    target[action] = reward
                else:
                    next_values = self.target_model.predict(np.array([next_state]), verbose=0)[0]
                    target[action] = reward + GAMMA * np.max(next_values)
                
                targets[i] = target
            
            history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
            loss = history.history['loss'][0]
            if DEBUG and i % 10 == 0:  
                print(f"Training-Loss: {loss:.4f}")
            
        except Exception as e:
            print(f"Error in replay: {e}")

        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

class DurakGame:
    def __init__(self, num_players=4):
        self.players = []
        self.deck = []
        self.played_cards = []
        self.field = []
        self.trumpf_suit = None
        self.current_index = 0
        self.num_players = num_players
        self.game_over = False
        self.winner = None

        self.card_ids = {}
        suits = ['♣', '♠', '♥', '♦']
        ranks = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        card_id = 0
        for rank in ranks:
            for suit in suits:
                card = rank + suit
                self.card_ids[card] = card_id
                card_id += 1

        self.id_to_card = {v: k for k, v in self.card_ids.items()}

        self.state_size = 36 + 36 + 36 + 4 + 4  
        self.action_size = 36 + 1  
        
    def initialize_game(self, ai_players=4):
        self.players = []
        for i in range(self.num_players):
            is_ai = i < ai_players
            name = f"AI-{i}" if is_ai else f"Human-{i}"
            self.players.append(Player(i, name, is_ai))
            
        self.deck = self.initialize_deck()
        random.shuffle(self.deck)
        self.played_cards = []
        self.deal_cards(self.players)
        
        trumpf_card = self.determine_trumpf()
        self.trumpf_suit = trumpf_card[-1]
        print(f"Trumpf: {trumpf_card}")
        
        self.current_index = 0
        self.field = []
        self.game_over = False
        self.winner = None
        
    def initialize_deck(self):
        suits = ['♣', '♠', '♥', '♦']
        ranks = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        return [rank + suit for suit in suits for rank in ranks]
    
    def deal_cards(self, players, num_cards=6):
        for player in players:
            while len(player.hand) < num_cards and self.deck:
                card = self.deck.pop(0)
                player.add_cards([card])
                
    def determine_trumpf(self):
        return self.deck.pop()
    
    def get_state_for_player(self, player_index):

        state = np.zeros(self.state_size)
        
        current_player = self.players[player_index]
        for card in current_player.hand:
            if card != "PASS":
                card_id = self.card_ids.get(card)
                if card_id is not None:
                    state[card_id] = 1
        
        for card in self.played_cards:
            if card != "PASS":
                card_id = self.card_ids.get(card)
                if card_id is not None:
                    state[36 + card_id] = 1
        
        for card in self.field:
            if card != "PASS":
                card_id = self.card_ids.get(card)
                if card_id is not None:
                    state[72 + card_id] = 1
        
        suits = ['♣', '♠', '♥', '♦']
        if self.trumpf_suit in suits:
            trumpf_id = suits.index(self.trumpf_suit)
            state[108 + trumpf_id] = 1
        
        for i, player in enumerate(self.players):
            relative_position = (i - player_index) % len(self.players)
            state[112 + relative_position] = len(player.hand) / 12
        
        return state
    
    def card_id_to_card(self, action_id):
        if action_id == self.action_size - 1:
            return "PASS"
        return self.id_to_card.get(action_id)
    
    def card_to_action_id(self, card):
        if card == "PASS":
            return self.action_size - 1
        return self.card_ids.get(card)
    
    def get_valid_actions(self, player, is_attacker):
        valid_action_ids = []
        
        if is_attacker:
            valid_cards = player.get_valid_attack_cards(self.field)
            for card in valid_cards:
                valid_action_ids.append(self.card_to_action_id(card))
            valid_action_ids.append(self.action_size - 1)
        else:
            attack_card = self.field[-1] if self.field else None
            if attack_card:
                valid_cards = player.get_valid_defense_cards(attack_card, self.trumpf_suit)
                for card in valid_cards:
                    card_id = self.card_to_action_id(card)
                    if card_id is not None: 
                        valid_action_ids.append(card_id)
            valid_action_ids.append(self.action_size - 1)
                            
        return valid_action_ids
    
    def step(self, action_id, is_attacker):
        reward = 0
        card = self.card_id_to_card(action_id)
        
        current_player = self.players[self.current_index]
        next_player_index = (self.current_index + 1) % len(self.players)
        next_player = self.players[next_player_index]

        if is_attacker:
            if card == "PASS":
                self.played_cards.extend(self.field)
                self.field = []
                self.pick_up_card(next_player, current_player)
                reward = 0
                if DEBUG:
                    print(f"{current_player.name} endet Angriff")
                self.current_index = next_player_index
            else:
                removed = current_player.remove_card(card)
                if not removed:
                    print(f"Karte {card} nicht entfernt")
                    reward = -5
                else:
                    self.field.append(card)
                    reward = 5
        else:
            attack_card = self.field[-1] if self.field else None
            
            if card == "PASS":
                next_player.add_cards(self.field)
                self.field = []
                reward = -1
                self.pick_up_card(next_player, current_player)
                self.current_index = (next_player_index + 1) % len(self.players)
            else:
                removed = next_player.remove_card(card)
                if not removed:
                    print(f"Karte {card}nicht entfernt")
                    reward = -5
                else:
                    self.field.append(card)
                    reward = 10 

        if not self.deck:
            for player in self.players:
                if not player.hand:
                    self.game_over = True
                    self.winner = player
                    if player == current_player and is_attacker:
                        reward += 120
                    elif player == next_player and not is_attacker:
                        reward += 120
        
        return reward, self.game_over

    def pick_up_card(self, defender, attacker):
        missing_att = 6 - len(attacker.hand)
        for i in range(missing_att):
            if self.deck:
                card = self.deck.pop()
                attacker.hand.append(card)
                if DEBUG:
                    print(f"{attacker.name} nimmt Karte {card} auf")
                
        missing_def = 6 - len(defender.hand)
        for i in range(missing_def):
            if self.deck:
                card = self.deck.pop()
                defender.hand.append(card)
                if DEBUG:
                    print(f"{defender.name} nimmt Karte {card} auf")
                
    def next_turn(self):
        self.current_index = (self.current_index + 1) % len(self.players)
        return self.current_index
    
    def print_game_state(self):
        print(f"\nRundenInfo:")
        print(f"Trumpf: {self.trumpf_suit}")
        print(f"Karten im Deck: {len(self.deck)}")
        print(f"Feld: {', '.join(self.field) if self.field else 'leer'}")
        
        for player in self.players:
            print(f"{player.name}: {player.show_hand()} ({len(player.hand)} Karten)")
            
        print(f"Aktueller Spieler: {self.players[self.current_index].name}")
        
def train_ai_agents(episodes=1):
    game = DurakGame(num_players=4)
    
    agents = [DQNAgent(game.state_size, game.action_size) for _ in range(4)]
    
    for episode in range(episodes):
        print(f"\nEpisode {episode+1}/{episodes}")
        game.initialize_game(ai_players=4)
        game.print_game_state()
        
        turn_count = 0
        max_turns = 200 
        
        start_time = time.time()
        
        experiences_collected = 0
        train_every_n_experiences = 20 
        
        while not game.game_over and turn_count < max_turns:
            if time.time() - start_time > 120: 
                print(f"Timeout:{turn_count} Züge")
                break
                
            current_player_idx = game.current_index
            next_player_idx = (current_player_idx + 1) % len(game.players)
            
            current_agent = agents[current_player_idx]
            next_agent = agents[next_player_idx]
            
            print(f"\nZug {turn_count + 1}")
            print(f"Spieler {game.players[current_player_idx].name} ist am Zug")
            
            attack_completed = False
            attack_continues = True
            
            while attack_continues and not game.game_over:
                state = game.get_state_for_player(current_player_idx)
                valid_actions = game.get_valid_actions(game.players[current_player_idx], True)
                
                if not valid_actions:
                    print(f"Keine gültige Verteidigung")
                    attack_continues = False
                    break
                    
                action = current_agent.act(state, valid_actions)
                
                if action is None:
                    print(f"Keine Aktion gewählt")
                    attack_continues = False
                    break
                    
                card = game.card_id_to_card(action)
                print(f"{game.players[current_player_idx].name} spielt: {card}")
                
                if card == "PASS":
                    reward, done = game.step(action, True)
                    experiences_collected += 1
                    
                    next_state = game.get_state_for_player(current_player_idx)
                    current_agent.remember(state, action, reward, next_state, done)
                    
                    attack_continues = False
                    attack_completed = True
                    
                    if done:
                        print(f"Spiel beendet nach {game.players[current_player_idx].name}")
                        break
                else:

                    reward, done = game.step(action, True)
                    experiences_collected += 1
                    
                    next_state = game.get_state_for_player(current_player_idx)
                    current_agent.remember(state, action, reward, next_state, done)
                    
                    if done:
                        print(f"Spiel beendet nach: {game.players[current_player_idx].name}")
                        break

                    print(f"Spieler {game.players[next_player_idx].name} verteidigt")
                    
                    defense_state = game.get_state_for_player(next_player_idx)
                    valid_defense_actions = game.get_valid_actions(game.players[next_player_idx], False)
                    
                    if valid_defense_actions:
                        defense_action = next_agent.act(defense_state, valid_defense_actions)
                        
                        if defense_action is not None:
                            defense_card = game.card_id_to_card(defense_action)
                            print(f"{game.players[next_player_idx].name} verteidigt mit: {defense_card}")
                            
                            defense_reward, done = game.step(defense_action, False)
                            experiences_collected += 1
                            
                            defense_next_state = game.get_state_for_player(next_player_idx)
                            next_agent.remember(defense_state, defense_action, defense_reward, 
                                               defense_next_state, done)
                            
                            if done:
                                print(f"Spiel beendet nach: {game.players[next_player_idx].name}")
                                break

                            if defense_card == "PASS":
                                attack_continues = False
                                attack_completed = True
                                break

                            if not game.players[current_player_idx].hand or len(game.players[current_player_idx].hand) == 0:
                                attack_continues = False
                                attack_completed = True
                                print(f"{game.players[current_player_idx].name} hat keine Karten zum Angreifen")
                                break
                        else:
                            attack_continues = False
                            break
                    else:
                        attack_continues = False
                        break

            if not attack_completed:
                game.next_turn()

            if experiences_collected >= train_every_n_experiences:
                print("Training start")
                for agent_idx, agent in enumerate(agents):
                    if len(agent.memory) >= BATCH_SIZE:
                        agent.replay(BATCH_SIZE)
                    else:
                        print("Überspringe Training")
                        
                experiences_collected = 0
                
            if turn_count % UPDATE_TARGET_EVERY == 0:
                for agent_idx, agent in enumerate(agents):
                    agent.update_tmodel()
                    
            game.print_game_state()
            turn_count += 1

        print(f"Episode {episode+1}/{episodes} abgeschlossen, Züge: {turn_count}")
        if game.winner:
            print(f"Gewinner: {game.winner.name}")
        else:
            print("Kein Gewinner")
            
        epsilons = [agent.epsilon for agent in agents]
        print(f"Epsilon-Werte: {epsilons}")

    for i, agent in enumerate(agents):
        try:
            agent.model.save(f'durak_agent_{i}.h5')
            print(f"Modell {i} gespeichert")
        except Exception as e:
            print(f"Fehler beim Speichern {i}: {e}")
    
    print("Training fertig")
    return agents

def play_game_with_trained_agents(agents=None):
    #Spiel mit trainierten Agenten spielen
    print("\n\nSPIELMODUS")
    game = DurakGame(num_players=4)
    
    if agents is None:
        agents = []
        for i in range(4):
            agent = DQNAgent(game.state_size, game.action_size)
            try:
                agent.model = tf.keras.models.load_model(f'durak_agent_{i}.h5')
                agent.target_model = tf.keras.models.load_model(f'durak_agent_{i}.h5')
                agent.epsilon = EPSILON_END
                agents.append(agent)
                print(f"Modell für Agent {i} erfolgreich geladen")
            except Exception as e:
                print(f"Konnte Modell für Agent {i} nicht laden: {e}")
                return
    else:
        for agent in agents:
            agent.epsilon = EPSILON_END
    
    game.initialize_game(ai_players=4)
    game.print_game_state()
    
    turn_count = 0
    max_turns = 100

    start_time = time.time()
    
    while not game.game_over and turn_count < max_turns:
        if time.time() - start_time > 60: 
            print(f"Timeout: {turn_count} Züge")
            break
            
        current_player_idx = game.current_index
        next_player_idx = (current_player_idx + 1) % len(game.players)
        
        current_agent = agents[current_player_idx]
        next_agent = agents[next_player_idx]
        
        print(f"Spieler {game.players[current_player_idx].name} ist am Zug (Angreifer)")
        
        attack_continues = True
        
        while attack_continues and not game.game_over:
            state = game.get_state_for_player(current_player_idx)
            valid_actions = game.get_valid_actions(game.players[current_player_idx], True)
            
            if not valid_actions:
                print(f"{game.players[current_player_idx].name} keine Aktionen")
                attack_continues = False
                break
                
            action = current_agent.act(state, valid_actions)
            
            if action is None:
                print(f"{game.players[current_player_idx].name} keine Aktion.")
                attack_continues = False
                break
                
            card = game.card_id_to_card(action)
            print(f"{game.players[current_player_idx].name} spielt: {card}")
            
            if card == "PASS":
                reward, done = game.step(action, True)
                attack_continues = False
                
                if done:
                    print(f"Spiel beendet nach Angriff: {game.players[current_player_idx].name}")
                    break
            else:
                reward, done = game.step(action, True)
                
                if done:
                    print(f"Spiel beendet nach Angriff: {game.players[current_player_idx].name}")
                    break
                
                print(f"Spieler {game.players[next_player_idx].name} verteidigt")
                
                defense_state = game.get_state_for_player(next_player_idx)
                valid_defense_actions = game.get_valid_actions(game.players[next_player_idx], False)
                
                if valid_defense_actions:
                    defense_action = next_agent.act(defense_state, valid_defense_actions)
                    
                    if defense_action is not None:
                        defense_card = game.card_id_to_card(defense_action)
                        print(f"{game.players[next_player_idx].name} verteidigt mit: {defense_card}")
                        
                        defense_reward, done = game.step(defense_action, False)
                        
                        if done:
                            print(f"Spiel beendet nach Verteidigung: {game.players[next_player_idx].name}")
                            break

                        if defense_card == "PASS":
                            attack_continues = False
                            break
                        
                        if not game.players[current_player_idx].hand or len(game.players[current_player_idx].hand) == 0:
                            attack_continues = False
                            print(f"{game.players[current_player_idx].name} keine Karten zum Angreifen")
                            break
                    else:
                        attack_continues = False
                        break
                else:
                    attack_continues = False
                    break
        
        game.print_game_state()
        turn_count += 1
    
    if game.winner:
        print(f"\nGewinner: {game.winner.name}")
    else:
        print("\nKein Gewinner")
        
    return game.winner

if __name__ == "__main__":
    program_start_time = time.time()
    
    print("Starte Training")
    trained_agents = train_ai_agents(episodes=1)
    
    print("\nSpielphase")
    play_game_with_trained_agents(trained_agents)

    play_more = "n"########input("noch ein Spiel? (j/n):")
    while play_more.lower() == 'j':
        play_game_with_trained_agents(trained_agents)
        play_more = input("noch ein Spiel? (j/n):")
    
    total_execution_time = time.time() - program_start_time
    hours, remainder = divmod(total_execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("ENDE.")
    print(f"Gesamtlaufzeit: {int(hours):02}:{int(minutes):02}:{seconds:.2f}")