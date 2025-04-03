#include <iostream>
#include <string>
#include <torch/torch.h>

#include "othello.hpp"  
#include "az_mcts.hpp"       
#include "reference_mcts.hpp"       
#include "az_net.hpp"         


int main(int argc, char* argv[]) {
    AZNet net = AZNet(2, 64, 65, 5);
    if (argc > 1) {
        const std::string checkpoint_path = argv[1];
        std::cout << "Loading model from: " << checkpoint_path << std::endl;
        torch::load(net, checkpoint_path);
    }

    net->eval();

    const int eval_games = 10;

    MCTS az_mcts(net, 1.414, 100, false);  
    RefMCTS ref_mcts(1.414, 100);  

    int az_wins = 0;
    int ref_wins = 0;
    int draws = 0;

    Player trained_player = Player::Black;
    for (int game_num = 0; game_num < eval_games; ++game_num) {
        std::cout << "Starting game " << game_num + 1 << std::endl;
        std::cout << "Trained Player: " << (trained_player == Player::Black ? "Black" : "White") << std::endl;
        OthelloState state;
        while (!state.is_terminal()) {
            // state.render();
            if (state.current_player == trained_player) {
                auto [best_move, _] = az_mcts.search(state);
                state = state.step(best_move);
            } else {
                auto [best_move, _] = ref_mcts.search(state);
                state = state.step(best_move);
            }
        }
        int reward = state.reward(trained_player);
        if (reward == 1) {
            az_wins++;
        } else if (reward == -1) {
            ref_wins++;
        } else {
            draws++;
        }
        std::cout << "Final board:" << std::endl;
        state.render();
        trained_player = (trained_player == Player::Black ? Player::White : Player::Black);
    }

    std::cout << "Stats:\n"
              << "AZ MCTS wins - " << az_wins
              << "; Ref MCTS wins - " << ref_wins
              << "; Draws - " << draws
              << std::endl;
    return 0;
}

