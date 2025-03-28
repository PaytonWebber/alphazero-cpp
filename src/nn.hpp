#pragma once

#include <torch/torch.h>

struct TicTacToeNetImpl : torch::nn::Module {
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  torch::nn::Linear policy_head{nullptr}, value_head{nullptr};

  TicTacToeNetImpl() {
    fc1 = register_module("fc1", torch::nn::Linear(9, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 64));
    policy_head = register_module("policy_head", torch::nn::Linear(64, 9));
    value_head  = register_module("value_head",  torch::nn::Linear(64, 1));
  }

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
    x = torch::relu(fc1(x));
    x = torch::relu(fc2(x));
    torch::Tensor policy_logits = policy_head(x);      // raw scores for each move
    torch::Tensor value         = torch::tanh(value_head(x));  // [-1, 1] scalar
    return {policy_logits, value};
  }
};
TORCH_MODULE(TicTacToeNet);
