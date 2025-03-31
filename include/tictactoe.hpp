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

    TicTacToeState();
    TicTacToeState(const std::array<int, 9>& board, const std::vector<int>& legal_actions, Player player);

    std::vector<int> available_moves(const std::array<int, 9>& board) const;
    bool is_winner(Player player) const;
    bool is_terminal() const;
    TicTacToeState step(int action) const;
    int reward(Player to_play) const;
    void render() const;
};

