#pragma once

#include <cmath>
#include <limits>
#include <sys/types.h>
#include <utility>
#include <memory>
#include <vector>
#include "torch/torch.h"
#include "nn.hpp"

template <typename State>
struct Node {
  State state;
  int action;
  float P;       // prior probability from nn
  float W;       // total reward
  float Q;       // average reward (W/N)
  int N;         // visit count
  std::vector<std::shared_ptr<Node<State>>> children;
  Node<State>* parent;

  Node(State s, int action, float prior=0.0, Node* parent=nullptr)
  : state(std::move(s)), action(action), P(prior), W(0), Q(0), N(0), parent(parent)
  {}
  
  bool is_leaf() const {
    return children.empty();
  }

  float puct(float C) const {
    if (N == 0) {
      return std::numeric_limits<float>::max();
    }
    int parent_n = std::max(1, parent->N);
    return Q + C * P * std::sqrt(parent_n) / (1 + N);
  }
};

class MCTS {
public:
  TicTacToeNet net;
  float C;
  int simulations;

  MCTS(TicTacToeNet network, float C=1.414, int sims=100)
    : net(network), C(C), simulations(sims) {}

  template <typename State>
  std::pair<int, std::vector<float>> search(State root_state) {
    std::shared_ptr<Node<State>> root = std::make_shared<Node<State>>(root_state, -1, 0.0, nullptr); 
    for (int s=0; s<simulations; ++s) {
      std::shared_ptr<Node<State>> leaf = select(root);
      float value;
      if (!leaf->state.is_terminal()) {
        value = expand_and_evaluate(leaf);
      } else {
        value = leaf->state.reward(leaf->state.current_player);
      }
      backpropagate(leaf, value);
    }
    std::vector<float> action_probs(9, 0.0);
    float sumN = 0;
    for (auto& child : root->children) { sumN += child->N; }
    int best_move = 0;
    float best_prob = 0.0; 
    for (auto& child : root->children) {
      float prob = child->N / sumN;
      action_probs[child->action] = prob;
      if (prob > best_prob) {
        best_move = child->action;
        best_prob = prob;
      }
    }
    return {best_move, action_probs};
  }

  template <typename State>
  std::shared_ptr<Node<State>> select(std::shared_ptr<Node<State>> node) {
    while(!node->children.empty()) {
      std::shared_ptr<Node<State>> best_child = nullptr;
      float best_value = -std::numeric_limits<float>::infinity();
      for (auto& child : node->children) {
       float U = child->puct(C);
        if (U > best_value) {
          best_value = U;
          best_child = child;
        }
      }
      node = best_child;
    }
    return node;
  }

  template <typename State>
  float expand_and_evaluate(std::shared_ptr<Node<State>> leaf) {
    State state = leaf->state;

    torch::Tensor input = torch::tensor(std::vector<float>(state.board.begin(), state.board.end())).reshape({1,9});
    input = input.to(torch::kF32);
    auto [policy_logits, value_tensor] = net->forward(input);
    auto policy_probs = torch::softmax(policy_logits, /*dim=*/1).flatten();
    float value = value_tensor.item<float>();

    std::vector<int> actions = state.legal_actions;
    for (int action : actions) {
      State next_state = state.step(action);
      float prior = policy_probs[action].item<float>();
      auto child_node = std::make_shared<Node<State>>(next_state, action, prior, leaf.get());
      leaf->children.push_back(child_node);
    }
    return value;
  }

  template<typename State>
  void backpropagate(std::shared_ptr<Node<State>> node, int value) {
    float v = value;
    Node<State>* cur = node.get();
    while(cur != nullptr) {
      cur->N += 1;
      cur->W += v;
      cur->Q = cur->W / cur->N;
      v = -v;
      cur = cur->parent;
    }
  }
};
