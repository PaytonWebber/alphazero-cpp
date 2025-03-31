#include "nn.hpp"
#include "mcts.hpp"
#include "tictactoe.hpp"

int main() {
  TicTacToeState test_state = TicTacToeState();
  TicTacToeNet net = TicTacToeNet();
  MCTS mcts = MCTS(net, 1.414, 1000);

  while (!test_state.is_terminal()) {
    test_state.render();
    auto [best_move, probs] = mcts.search(test_state);
    test_state = test_state.step(best_move);
    
    for (auto i: probs) {
     std::cout << i << " ";
    }
    std::cout << std::endl;
  }
  test_state.render();
}
