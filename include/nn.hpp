#pragma once

#include <torch/torch.h>

struct OthelloNetImpl : torch::nn::Module {
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
  torch::nn::Linear policy_head{nullptr}, value_head{nullptr};

  OthelloNetImpl() {
    fc1 = register_module("fc1", torch::nn::Linear(64, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 128));
    policy_head = register_module("policy_head", torch::nn::Linear(128, 64));
    value_head  = register_module("value_head",  torch::nn::Linear(128, 1));
  }

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
    x = torch::relu(fc1(x));
    x = torch::relu(fc2(x));
    torch::Tensor policy_logits = policy_head(x);      // raw scores for each move
    torch::Tensor value         = torch::tanh(value_head(x));  // [-1, 1] scalar
    return {policy_logits, value};
  }
};
TORCH_MODULE(OthelloNet);
