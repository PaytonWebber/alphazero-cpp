#include <iostream>
#include <string>
#include <torch/torch.h>

#include "othello.hpp"  
#include "mcts.hpp"       
#include "az_net.hpp"         


int main(int argc, char* argv[]) {
    AZNet net = AZNet(2, 64, 64, 5);

    if (argc > 1) {
        const std::string checkpoint_path = argv[1];
        std::cout << "Loading model from: " << checkpoint_path << std::endl;
        torch::load(net, checkpoint_path);
    }

    net->eval();

    const int eval_games = 1;

    MCTS mcts(net, 1.414, 10);  
    for (int game_num = 0; game_num < eval_games; ++game_num) {
        std::cout << "Starting game " << game_num + 1 << std::endl;
        OthelloState state;

        while (!state.is_terminal()) {
            auto [best_move, _] = mcts.search(state);
            state.render();
            state = state.step(best_move);
        }
        std::cout << "Final board:" << std::endl;
        state.render();

    }
    return 0;
}

