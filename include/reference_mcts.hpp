#pragma once

#include <cmath>
#include <random>
#include <limits>
#include <sys/types.h>
#include <utility>
#include <memory>
#include <vector>

template <typename State>
struct RefNode {
  State state;
  int action;
  float W;       // total reward
  float Q;       // average reward (W/N)
  int N;         // visit count
  std::vector<std::shared_ptr<RefNode<State>>> children;
  RefNode<State>* parent;

  RefNode(State s, int action, RefNode* parent)
  : state(std::move(s)), action(action), parent(parent), W(0), Q(0), N(0)
  {}
  
  bool is_leaf() const {
    return children.empty();
  }

  float ucb(float C) const {
    if (N == 0) {
      return std::numeric_limits<float>::max();
    }
    int parent_n = std::max(1, parent->N);
    return Q + C * std::sqrt((std::log(parent_n) / (N+1)));
  }
};

class RefMCTS {
public:
  float C;
  int simulations;
  std::mt19937 gen;

  RefMCTS(float C=1.414, int sims=100)
    : C(C), simulations(sims), gen(std::random_device{}()) {}

  template <typename State>
  std::pair<int, std::vector<float>> search(State root_state) {
    std::shared_ptr<RefNode<State>> root = std::make_shared<RefNode<State>>(root_state, -1, nullptr); 
    for (int s=0; s<simulations; ++s) {
      std::shared_ptr<RefNode<State>> leaf = select(root);
      float value;
      if (!leaf->state.is_terminal()) {
        expand(leaf);
        std::shared_ptr<RefNode<State>> child = leaf->children[0];
        value = simulate(child);
        leaf = child;
      } else {
        value = leaf->state.reward(leaf->state.current_player);
      }
      backpropagate(leaf, -value);
    }
    std::vector<float> action_probs(65, 0.0);
    float sumN = 0;
    for (auto& child : root->children) { sumN += child->N; }
    int best_move = 64;
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
  std::shared_ptr<RefNode<State>> select(std::shared_ptr<RefNode<State>> node) {
    while(!node->children.empty()) {
      std::shared_ptr<RefNode<State>> best_child = nullptr;
      float best_value = -std::numeric_limits<float>::infinity();
      for (auto& child : node->children) {
       float U = child->ucb(C);
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
  void expand(std::shared_ptr<RefNode<State>> leaf) {
    State state = leaf->state;
    std::vector<int> actions = state.legal_actions();
    for (int action : actions) {
      State next_state = state.step(action);
      auto child_node = std::make_shared<RefNode<State>>(next_state, action, leaf.get());
      leaf->children.push_back(child_node);
    }
  }

  template<typename State>
  int simulate(std::shared_ptr<RefNode<State>> node) {
    State state = node->state;
    while(!state.is_terminal()) {
      std::vector<int> actions = state.legal_actions();
      std::uniform_int_distribution<> dist(0, actions.size() - 1);
      int action = actions[dist(gen)];
      state = state.step(action);
    }
    return state.reward(node->state.current_player);
  }

  template<typename State>
  void backpropagate(std::shared_ptr<RefNode<State>> node, float value) {
    float v = value;
    RefNode<State>* cur = node.get();
    while(cur != nullptr) {
      cur->N += 1;
      cur->W += v;
      cur->Q = cur->W / cur->N;
      v = -v;
      cur = cur->parent;
    }
  }
};
