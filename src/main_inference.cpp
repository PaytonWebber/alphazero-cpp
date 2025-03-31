#include <iostream>
#include <string>
#include <torch/torch.h>

#include "tictactoe.hpp"  
#include "mcts.hpp"       
#include "nn.hpp"         


int main(int argc, char* argv[]) {
    std::string checkpoint_path = "models/model_iter_50.pt";
    if (argc > 1) {
        checkpoint_path = argv[1];
    }
    std::cout << "Loading model from: " << checkpoint_path << std::endl;

    TicTacToeNet net = TicTacToeNet();
    torch::load(net, checkpoint_path);
    net->eval();

    const int eval_games = 10;
    int wins_x = 0, wins_o = 0, draws = 0;

    for (int game_num = 0; game_num < eval_games; ++game_num) {
        std::cout << "Starting game " << game_num + 1 << std::endl;
        TicTacToeState state;

        while (!state.is_terminal()) {
            MCTS mcts(net, 1.414, 10);  
            auto [best_move, _] = mcts.search(state);

            std::cout << "Player " << (state.current_player == X ? "X" : "O")
                      << " chooses move: " << best_move << std::endl;
            state.render();

            state = state.step(best_move);
        }

        std::cout << "Final board:" << std::endl;
        state.render();

        int outcome = state.reward(X);
        if (outcome == 1) {
            std::cout << "Player X wins!" << std::endl;
            wins_x++;
        } else if (outcome == -1) {
            std::cout << "Player O wins!" << std::endl;
            wins_o++;
        } else {
            std::cout << "Game is a draw!" << std::endl;
            draws++;
        }
        std::cout << "-------------------------------------" << std::endl;
    }

    std::cout << "Evaluation results over " << eval_games << " games:" << std::endl;
    std::cout << "Player X wins: " << wins_x << std::endl;
    std::cout << "Player O wins: " << wins_o << std::endl;
    std::cout << "Draws: " << draws << std::endl;

    return 0;
}

