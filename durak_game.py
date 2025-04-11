from random import shuffle

class Player:
    rank_order = {"6": 1, "7": 2, "8": 3, "9": 4, "10": 5, "J": 6, "Q": 7, "K": 8, "A": 9}

    def __init__(self, player_id, name):
        self.player_id = player_id
        self.name = name
        self.hand = []

    def show_hand(self):
        return ", ".join(self.hand)

    def remove_card(self, card):
        self.hand.remove(card)

    def add_cards(self, cards):
        self.hand.extend(cards)
        

def initialize_deck():
    suits = ['♣', '♠', '♥', '♦']
    ranks = ['6', '7', '8', '9', '10','J','Q', 'K', 'A']
    return [rank + suit for suit in suits for rank in ranks]

def deal_cards(deck, players, num_cards=6):
    for player in players:
        while len(player.hand) < num_cards and deck:
            card = deck.pop(0)
            player.add_cards([card])

def determine_trumpf(deck):
    return deck.pop()

def attack(player, field):
    while True:
        print(f"{player.name}, deine Hand: {player.show_hand()}")
        attack_card = input("Wähle eine Karte zum Angreifen (oder 'skip', zum überspringen): ").strip().upper()
        if attack_card in player.hand:
            possible_numb = []
            for card in field:
                possible_numb.append(card[:-1])
            if attack_card[:-1] in possible_numb or field == []:       
                return attack_card
        if attack_card.lower() == 'skip':
             return attack_card
        print("ungültiger Angriff")


def defend(player, attack_card, trumpf_suit):
    print(f"Angriff mit {attack_card}")
    while True:
        print(f"{player.name}, deine Hand: {player.show_hand()}")
        defense_card = input("Wähle eine Karte zur Verteidigung (oder 'give up', um aufzugeben): ").strip().upper()
        if defense_card == 'GIVE UP':
            return None
        if defense_card in player.hand:
            attack_rank = Player.rank_order[attack_card[:-1]]
            defense_rank = Player.rank_order[defense_card[:-1]]
            if defense_card[-1] == attack_card[-1] and defense_rank > attack_rank:
                return defense_card
            if defense_card[-1] == trumpf_suit and attack_card[-1] != trumpf_suit:
                return defense_card
            print("Ungültige Verteidigung!")

def pick_up_card(defender, attacker, deck):
    missingatt = 6 - len(attacker.hand)
    for i in range(missingatt):
        if deck:
            attacker.hand.append(deck.pop())
    missingdef = 6 - len(defender.hand)
    for i in range(missingdef):
        if deck:
            defender.hand.append(deck.pop())

def next_turn(current_index, players):
    return (current_index + 1) % (len(players))

def void_cards(field):
    field.clear()        
    

def end_game(winner):
    print(f"{winner.name} gewinnt!")
    exit()

def main():
    field = []
    deck = initialize_deck()
    shuffle(deck)
    players = []
    while True:
        num_players = int(input("Anzahl der Spieler: "))
        if num_players <= 4:
            break
        
    for i in range(num_players):
        name = input(f"Name von Spieler {i + 1}: ")
        players.append(Player(i, name))

    deal_cards(deck, players)
    trumpf_card = determine_trumpf(deck)
    trumpf_suit = trumpf_card[-1]
    print(f"Trumpf: {trumpf_card}")

    current_index = 0
    while True:
        attacker = players[current_index]
        defender = players[(current_index + 1 )% (len(players))]
        print(f"rest Deck: {len(deck)}")
        while True:
            attack_card = attack(attacker, field)
            if attack_card.lower() == 'skip' or not deck:
                current_index = next_turn(current_index, players)
                void_cards(field)
                pick_up_card(defender, attacker, deck)
                break
            attacker.remove_card(attack_card)
            field.extend(attack_card)
            defense_card = defend(defender, attack_card, trumpf_suit)
            
            if defense_card:
                defender.remove_card(defense_card)
                field.extend(defense_card)
                print(f"{defender.name} hat mit {defense_card} verteidigt!")
            else:
                defender.add_cards(field)
                void_cards(field)
                print(f"{defender.name} konnte nicht verteidigen und hat {attack_card} erhalten.")
                pick_up_card(defender, attacker, deck)
            if not deck:             
                if not attacker.hand:
                    end_game(attacker)
                if not defender.hand:
                    end_game(defender)      
            

main()
