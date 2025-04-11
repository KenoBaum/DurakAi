import random
import tensorflow as tf
import time
import os

from durak_AI import Player, DQNAgent, DurakGame

AI_THINKING_TIME = 1.5
PAUSE = 1.5
CARD_PLAY_PAUSE = 2.0     
TURN_PAUSE = 1.2

class HumanPlayer(Player):
    def __init__(self, player_id, name):
        super().__init__(player_id, name, is_ai=False)
    
    def choose_card(self, valid_cards, is_attacker, field):
        if not valid_cards:
            print("No valid cards")
            return "PASS"
        
        action_type = "attack" if is_attacker else "defend"
        
        print(f"\nYour hand: {', '.join(self.hand)}")
        if field:
            print(f"Field: {', '.join(field)}")
        print(f"Valid cards to {action_type}: {', '.join(valid_cards)}")
        
        while True:
            choice = input(f"Choose card to {action_type} or PASS: ").strip()
            
            if choice == "PASS" or choice in valid_cards:
                return choice
            else:
                print("Invalid")

class DurakGame(DurakGame):
    
    def __init__(self, num_players=4, human_position=0):
        super().__init__(num_players)
        self.human_position = human_position
        
    def initialize_game(self, ai_players=3):
        self.players = []
        
        for i in range(self.num_players):
            if i == self.human_position:
                self.players.append(HumanPlayer(i, "You"))
            else:
                self.players.append(Player(i, f"AI-{i}", is_ai=True))
        
        self.deck = self.initialize_deck()
        random.shuffle(self.deck)
        self.played_cards = []
        self.deal_cards(self.players)
        
        trumpf_card = self.determine_trumpf()
        self.trumpf_suit = trumpf_card[-1]
        print(f"Trump card: {trumpf_card}")
        time.sleep(CARD_PLAY_PAUSE)
        
        self.current_index = 0
        self.field = []
        self.game_over = False
        self.winner = None
    
    def print_game_state(self):
        print("\n" + "="*50)
        print(f"Trump suit: {self.trumpf_suit}")
        print(f"Cards in deck: {len(self.deck)}")
        print(f"Field: {', '.join(self.field) if self.field else 'empty'}")
        
        for player in self.players:
            if player.name == "You":
                print(f"{player.name}: {player.show_hand()} ({len(player.hand)} cards)")
            else:
                print(f"{player.name}: {len(player.hand)} cards")
        
        current_player = self.players[self.current_index]
        print(f"Current player: {current_player.name}")
        print("="*50)
        time.sleep(PAUSE)

def load_ai_agents(difficulty):
    game = DurakGame(num_players=4)
    agents = []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    difficulty_files = {
        "easy": os.path.join(script_dir, "durak_ai_easy.h5"),
        "hard": os.path.join(script_dir, "durak_ai_hard.h5"),
        "master": os.path.join(script_dir, "durak_ai_master.h5")
    }
    
    model_file = difficulty_files.get(difficulty.lower())
    if not model_file:
        print(f"Invalid, default: 'hard'")
        model_file = os.path.join(script_dir, "durak_ai_hard.h5")

    for i in range(4):
        agent = DQNAgent(game.state_size, game.action_size)
        
        if difficulty.lower() == "easy":
            agent.epsilon = 0.3  
        elif difficulty.lower() == "hard":
            agent.epsilon = 0.1  
        elif difficulty.lower() == "master":
            agent.epsilon = 0.01  
        
        try:
            if os.path.exists(model_file):
                agent.model = tf.keras.models.load_model(model_file)
                agent.target_model = tf.keras.models.load_model(model_file)
                print(f"Successfully loaded {difficulty} model")
            else:
                basic_model = os.path.join(script_dir, f'durak_agent_{i}.h5')
                if os.path.exists(basic_model):
                    agent.model = tf.keras.models.load_model(basic_model)
                    agent.target_model = tf.keras.models.load_model(basic_model)
        except Exception as e:
            print(f"FEHLER: AI-{i}: {e}")
        
        agents.append(agent)
        time.sleep(0.2)
    
    return agents

def simulate_ai_thinking():
    thinking_time = AI_THINKING_TIME * (0.8 + 0.4 * random.random())
    print("AI thinking", end="", flush=True)
    steps = int(thinking_time / 0.2)
    for _ in range(steps):
        time.sleep(0.2)
        print(".", end="", flush=True)
    print()

def play_against_ai():
    print("\n" + "="*50)
    print("DURAK VS AI")
    print("="*50)
    time.sleep(0.5)

    while True:
        difficulty = input("Choose difficulty (easy/hard/master): ").strip().lower()
        if difficulty in ["easy", "hard", "master"]:
            break
        print("Invalid choice")
    
    while True:
        try:
            position = int(input("Choose position (0-3): "))
            if 0 <= position <= 3:
                break
            print("Invalid choice")
        except ValueError:
            print("Invalid choice")
    
    print("Loading agents")
    agents = load_ai_agents(difficulty)
    
    print("\nStarting game")
    time.sleep(1)

    game = DurakGame(num_players=4, human_position=position)
    game.initialize_game(ai_players=3)
    game.print_game_state()

    turn_count = 0
    max_turns = 200
    
    while not game.game_over and turn_count < max_turns:
        current_player_idx = game.current_index
        next_player_idx = (current_player_idx + 1) % len(game.players)
        
        current_player = game.players[current_player_idx]
        next_player = game.players[next_player_idx]
        
        print(f"\nTurn {turn_count + 1}")
        print(f"Player {current_player.name} is attacking")
        time.sleep(TURN_PAUSE)
        
        attack_continues = True
        attacker_passed = False
        
        while attack_continues and not game.game_over:
            if current_player.is_ai:
                current_agent = agents[current_player_idx]
                state = game.get_state_for_player(current_player_idx)
                valid_actions = game.get_valid_actions(current_player, True)
                
                if not valid_actions:
                    print(f"{current_player.name} has no actions.")
                    attack_continues = False
                    attacker_passed = True
                    break
                
                simulate_ai_thinking()
                
                action = current_agent.act(state, valid_actions)
                if action is None:
                    print(f"{current_player.name} has no actions.")
                    attack_continues = False
                    attacker_passed = True
                    break
                
                card = game.card_id_to_card(action)
                print(f"{current_player.name} plays: {card}")
                time.sleep(CARD_PLAY_PAUSE)
                
            else:
                valid_cards = current_player.get_valid_attack_cards(game.field)
                card = current_player.choose_card(valid_cards, True, game.field)
                action = game.card_to_action_id(card)

            if card == "PASS":
                reward, done = game.step(action, True)
                attack_continues = False
                attacker_passed = True
                print(f"{current_player.name} passes")
                time.sleep(CARD_PLAY_PAUSE)
                
                if done:
                    print(f"Game over after Attack: {current_player.name}")
                    break
            else:
                reward, done = game.step(action, True)
                
                if done:
                    print(f"Game over after attack: {current_player.name}")
                    break
                
                print(f"Player {next_player.name} defends")
                time.sleep(CARD_PLAY_PAUSE)
                
                if next_player.is_ai:
                    next_agent = agents[next_player_idx]
                    defense_state = game.get_state_for_player(next_player_idx)
                    valid_defense_actions = game.get_valid_actions(next_player, False)
                    
                    if valid_defense_actions:
                        simulate_ai_thinking()
                        
                        defense_action = next_agent.act(defense_state, valid_defense_actions)
                        
                        if defense_action is not None:
                            defense_card = game.card_id_to_card(defense_action)
                            print(f"{next_player.name} defends with: {defense_card}")
                            time.sleep(CARD_PLAY_PAUSE)
                            
                            defense_reward, done = game.step(defense_action, False)
                            
                            if done:
                                print(f"Game over after defence: {next_player.name}")
                                break

                            if defense_card == "PASS":
                                print(f"{next_player.name} takes cards")
                                time.sleep(CARD_PLAY_PAUSE)
                                attack_continues = False
                                break
                            
                            if not current_player.hand:
                                attack_continues = False
                                print(f"{current_player.name} has no cards to attack")
                                time.sleep(CARD_PLAY_PAUSE)
                                break
                        else:
                            attack_continues = False
                            break
                    else:
                        attack_continues = False
                        break
                else:
                    valid_defense_cards = next_player.get_valid_defense_cards(card, game.trumpf_suit)
                    defense_card = next_player.choose_card(valid_defense_cards, False, game.field)
                    defense_action = game.card_to_action_id(defense_card)
                    
                    done = game.step(defense_action, False)
                    
                    if done:
                        print(f"Game over after defence: {next_player.name}")
                        break
                        
                    if defense_card == "PASS":
                        print("You take the cards")
                        time.sleep(CARD_PLAY_PAUSE)
                        attack_continues = False
                        break
                        
                    if not current_player.hand:
                        attack_continues = False
                        print(f"{current_player.name} has no cards to attack")
                        time.sleep(CARD_PLAY_PAUSE)
                        break
        
        game.print_game_state()
        
        if attacker_passed:
            game.current_index = next_player_idx
            time.sleep(TURN_PAUSE)
        else:
            game.next_turn()
            
        turn_count += 1
    
    if game.winner:
        print(f"\n Winner: {game.winner.name} ")
        if game.winner.name == "You":
            print("🏆🏆🏆")
        else:
            print("Du Lappen")
    else:
        print("\nNo winner")

if __name__ == "__main__":
    play_against_ai()