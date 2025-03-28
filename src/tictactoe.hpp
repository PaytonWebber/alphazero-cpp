#pragma once

#include <vector>
#include <array>
#include <iostream>

enum Player { EMPTY = 0, X = 1, O = -1 };

class TicTacToeState {
public:
    std::array<int, 9> board;
    std::vector<int> legal_actions;
    Player current_player;

    TicTacToeState() : board{0,0,0,0,0,0,0,0,0}, current_player(X) {
        for (int i=0; i<9; i++) {
            legal_actions.push_back(i);
        }
    }

    TicTacToeState(std::array<int, 9> board, std::vector<int> legal_actions, Player player)
    : board(board), legal_actions(legal_actions), current_player(player) {}

    std::vector<int> available_moves() const {
        std::vector<int> moves;
        for (int i = 0; i < 9; ++i) {
            if (board[i] == EMPTY) moves.push_back(i);
        }
        return moves;
    }

    bool is_winner(Player player) const {
        static int win_lines[8][3] = {
            {0,1,2}, {3,4,5}, {6,7,8},  // rows
            {0,3,6}, {1,4,7}, {2,5,8},  // cols
            {0,4,8}, {2,4,6}            // diagonals
        };
        for (auto& line : win_lines) {
            if (board[line[0]] == player && board[line[1]] == player && board[line[2]] == player) {
                return true;
            }
        }
        return false;
    }

    bool is_terminal() const {
        return is_winner(X) || is_winner(O) || legal_actions.empty();
    }

    TicTacToeState step(int action) const {
        std::array<int, 9> next_board = board;
        next_board[action] = current_player;
        std::vector<int> next_legal_actions = available_moves();
        Player next_player = (current_player == X ? O : X);

        return TicTacToeState {
            next_board,
            next_legal_actions,
            next_player
        };
    }

    int reward(Player to_play) const {
        Player other = (to_play == X ? O : X);
        if (is_winner(to_play)) return +1;
        if(is_winner(other)) return -1;
        return 0; // draw or ongoing
    }

    void render() const {
        for (int i = 0; i < 9; ++i) {
            char symbol;
            switch (board[i]) {
                case X:     symbol = 'X'; break;
                case O:     symbol = 'O'; break;
                case EMPTY: symbol = '.'; break;
                default:    symbol = '?'; break;
            }

            std::cout << symbol;

            if ((i + 1) % 3 == 0)
                std::cout << '\n';
            else
                std::cout << ' ';
        }
        std::cout << '\n';
    }
};

