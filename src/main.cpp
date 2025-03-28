#include "nn.hpp"
#include "mcts.hpp"
#include "tictactoe.hpp"

int main() {
  TicTacToeState test_state = TicTacToeState();
  TicTacToeNet net = TicTacToeNet();
  MCTS mcts = MCTS(net);

  std::vector<float> probs = mcts.search(test_state);
  for (auto i: probs) {
   std::cout << i << " ";
  }
  std::cout << std::endl;
}
