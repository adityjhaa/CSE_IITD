import time
import math
import random
import numpy as np
from helper import *

from collections import defaultdict

# Global dictionaries for LGR
last_good_reply = defaultdict(dict)
# Global dictionaries for N-grams
ngram_counts = defaultdict(lambda: defaultdict(int))
ngram_wins = defaultdict(lambda: defaultdict(int))

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.move = move
        self.terminal_node = False
        self.rave_visits = {}
        self.rave_value = {}
        self.last_good_reply = {}

    def is_fully_expanded(self) -> bool:
        """Checks if all possible actions from the current state have been expanded."""
        return len(self.children) == len(get_valid_actions(self.state))

    def best_child(self, c=1.41, beta_func=None) -> 'Node':
        """Select the child node with the highest combined UCT-RAVE score."""
        def uct_rave_value(child):
            if child.visits == 0:
                return float('inf')

            # UCT value (exploitation + exploration)
            exploitation = child.value / child.visits
            exploration = c * math.sqrt(math.log(self.visits) / child.visits)
            uct_value = exploitation + exploration

            # RAVE value
            rave_visits = self.rave_visits.get(child.move, 0)
            rave_value = self.rave_value.get(child.move, 0) / rave_visits if rave_visits > 0 else 0

            # Weight the combination of UCT and RAVE values
            beta = beta_func(child) if beta_func else rave_visits / (child.visits + rave_visits + 1)
            return (1 - beta) * uct_value + beta * rave_value

        return max(self.children, key=uct_rave_value)

    def update_rave(self, move: Tuple[int, int], outcome: float) -> None:
        """Update RAVE statistics for an action."""
        if move not in self.rave_visits:
            self.rave_visits[move] = 0
            self.rave_value[move] = 0
        self.rave_visits[move] += 1
        self.rave_value[move] += outcome

    def add_child(self, move: Tuple[int, int], state: np.array) -> 'Node':
        """Add a child node for a given move and state."""
        child_node = Node(state=state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def update_lgrf1(self, opponent_move: Tuple[int, int], agent_move: Tuple[int, int]):
        """Update last good reply for the opponent's move."""
        self.last_good_reply[opponent_move] = agent_move

    def forget_lgrf1(self, opponent_move: Tuple[int, int]):
        """Forget the last good reply if it is no longer effective."""
        if opponent_move in self.last_good_reply:
            del self.last_good_reply[opponent_move]

def beta_func(child: Node, k=500) -> float:
    """RAVE weight function based on the number of visits to a child node."""
    return k / (k + child.visits)

def mcts(state: np.array, timer_per_move: float, player_number: int, target_depth=3, num_rollouts=10) -> Tuple[int, int]:
    """Monte Carlo Tree Search with RAVE, including one-step win and block moves."""
    
    opponent = 3 - player_number
    valid_moves = get_valid_actions(state)
    
    # Step 1: Check for an immediate winning move
    for move in valid_moves:
        new_state = state.copy()
        new_state[move] = player_number
        if check_win(new_state, move, player_number)[0]:
            return move

    # Step 2: Check if the opponent is one step away from winning and block
    for move in valid_moves:
        new_state = state.copy()
        new_state[move] = opponent
        if check_win(new_state, move, opponent)[0]:
            return move  # Block the opponent's winning move

    # Step 3: MCTS loop
    root = Node(state=state)
    start_time = time.time()
    max_depth_reached = False

    while time.time() - start_time < timer_per_move and not max_depth_reached:
        leaf_node, depth = tree_policy(root, player_number)
        if leaf_node is None:
            continue

        outcome = rollout(leaf_node, player_number, num_rollouts)
        backpropagate(leaf_node, outcome)

        # Update RAVE statistics
        node = leaf_node
        while node.parent:
            node.parent.update_rave(node.move, outcome)
            node = node.parent

        # Update LGRF-1 statistics
        node = leaf_node
        while node.parent:
            node.parent.update_rave(node.move, outcome)
            if outcome > 0.5:  # If the outcome is positive for the player, store last good reply
                node.parent.update_lgrf1(node.move, leaf_node.move)
            else:
                node.parent.forget_lgrf1(node.move)  # Forget the reply if it led to a poor outcome
            node = node.parent

        if depth >= target_depth:
            max_depth_reached = True

    return root.best_child(c=0.9, beta_func=beta_func).move

def tree_policy(node: Node, player_number: int, current_depth=0, max_depth=3) -> Tuple[Node, int]:
    """Select a leaf node for exploration using UCB1 and track depth."""
    while not node.terminal_node and current_depth < max_depth:
        if not node.is_fully_expanded():
            return expand(node, player_number), current_depth + 1

        # Use LGRF-1: check if there's a remembered good reply to the opponent's move
        opponent_move = node.move
        if opponent_move in node.last_good_reply:
            next_move = node.last_good_reply[opponent_move]
            node = [child for child in node.children if child.move == next_move][0]
        else:
            node = node.best_child(c=0.9, beta_func=beta_func)
        
        current_depth += 1
    return node, current_depth

def expand(node: Node, player_number: int) -> Node:
    """Expand a node by creating one of its child nodes."""
    valid_moves = get_valid_actions(node.state)
    tried_moves = [child.move for child in node.children]

    for move in valid_moves:
        if move not in tried_moves:
            new_state = node.state.copy()
            new_state[move] = player_number
            return node.add_child(move, new_state)
    return None

def rollout(node: Node, player_number: int, num_rollouts: int = 10) -> float:
    """Simulate multiple random games from the current node and return the average outcome."""
    total_outcome = 0.0
    
    for _ in range(num_rollouts):
        current_state = node.state.copy()
        current_player = player_number
        moves_sequence = []
        outcome = None
        previous_move = None

        while True:
            moves = get_valid_actions(current_state)
            if not moves:
                outcome = 0.5
                break

            # Use LGR if available
            lgr_move = None
            if previous_move and previous_move in last_good_reply[current_player]:
                lgr_move = last_good_reply[current_player][previous_move]
                if lgr_move in moves:
                    move = lgr_move
                else:
                    move = select_ngram_move(previous_move, moves)
            else:
                move = select_ngram_move(previous_move, moves)

            current_state[move] = current_player
            moves_sequence.append(move)
            previous_move = move

            # Check for terminal state
            if check_win(current_state, move, current_player)[0]:
                outcome = 1 if current_player == player_number else 0
                break

            current_player = 3 - current_player  # Switch player

        # Update Last-Good-Reply policy
        update_lgr(moves_sequence, outcome, player_number)
        # Update N-gram statistics
        update_ngram_stats(moves_sequence, outcome, player_number)

        total_outcome += outcome if outcome is not None else 0.5  # Handle draws


    # Return the average outcome
    return total_outcome / num_rollouts


def select_ngram_move(previous_move: Tuple[int, int], moves: List[Tuple[int, int]], epsilon: float = 0.1) -> Tuple[int, int]:
    """Select a move using N-gram statistics with epsilon-greedy strategy."""
    if random.random() < epsilon or not previous_move:
        return random.choice(moves)

    move_stats = []
    total_counts = sum(ngram_counts[previous_move][move] for move in moves)

    if total_counts == 0:
        return random.choice(moves)

    for move in moves:
        count = ngram_counts[previous_move][move]
        win = ngram_wins[previous_move][move]
        win_rate = win / count if count > 0 else 0
        move_stats.append((win_rate, move))

    # Select the move with the highest win rate
    _, best_move = max(move_stats)
    return best_move

def update_lgr(moves_sequence: List[Tuple[int, int]], outcome: float, player_number: int) -> None:
    """Update the Last-Good-Reply tables based on the outcome of a simulation."""
    winner = player_number if outcome == 1 else 3 - player_number if outcome == 0 else None

    if winner is not None:
        for i in range(len(moves_sequence) - 2, -1, -2):  # Iterate over player's moves
            player = (player_number if (len(moves_sequence) - i) % 2 == 1 else 3 - player_number)
            opponent_move = moves_sequence[i]
            reply_move = moves_sequence[i + 1]

            # Update LGR for the winner
            if player == winner:
                last_good_reply[player][opponent_move] = reply_move
            else:
                # For LGRF-1, forget the last good reply if the player lost
                if opponent_move in last_good_reply[player]:
                    del last_good_reply[player][opponent_move]

def update_ngram_stats(moves_sequence: List[Tuple[int, int]], outcome: float, player_number: int) -> None:
    """Update N-gram statistics based on the moves sequence and outcome."""
    winner = player_number if outcome == 1 else 3 - player_number if outcome == 0 else None
    for i in range(len(moves_sequence) - 1):
        prev_move = moves_sequence[i]
        curr_move = moves_sequence[i + 1]
        ngram_counts[prev_move][curr_move] += 1
        if winner == player_number:
            ngram_wins[prev_move][curr_move] += 1

def backpropagate(node: Node, outcome: float) -> None:
    """Propagate the result of the simulation back up the tree."""
    while node is not None:
        node.visits += 1
        node.value += outcome
        node = node.parent
        outcome = 1 - outcome

def is_terminal(state: np.array, move: Tuple[int, int]) -> bool:
    """Check if the current state is terminal (win or draw)."""
    return check_win(state, move, 1)[0] or check_win(state, move, 2)[0]

class AIPlayer:

    def __init__(self, player_number: int, timer):
        """
        Intitialize the AIPlayer Agent

        # Parameters
        `player_number (int)`: Current player number, num==1 starts the game
        
        `timer: Timer`
            - a Timer object that can be used to fetch the remaining time for any player
            - Run `fetch_remaining_time(timer, player_number)` to fetch remaining time of a player
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer

    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Given the current state of the board, return the next move

        # Parameters
        `state: Tuple[np.array]`
            - a numpy array containing the state of the board using the following encoding:
            - the board maintains its same two dimensions
            - spaces that are unoccupied are marked as 0
            - spaces that are blocked are marked as 3
            - spaces that are occupied by player 1 have a 1 in them
            - spaces that are occupied by player 2 have a 2 in them

        # Returns
        Tuple[int, int]: action (coordinates of a board cell)
        """

        per_move_time = fetch_remaining_time(self.timer, 1) / (state.shape[0]*10)
        move = mcts(state, timer_per_move=per_move_time, player_number=self.player_number, target_depth=2**32-1, num_rollouts=10)
        return (int(move[0]), int(move[1]))

