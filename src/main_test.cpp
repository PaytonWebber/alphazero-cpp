#include "othello.hpp"
#include <vector>
#include <iostream>

int main() {
    OthelloState state;
    std::cout << "Initial Board:\n";
    state.render();
    state = state.step(state.legal_actions()[0]);
    state.render();
    state = state.step(state.legal_actions()[0]);
    state.render();
    std::vector<float> board = state.board();

    return 0;
}
