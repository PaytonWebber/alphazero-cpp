#include <algorithm>
#include <array>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mcts.hpp"
#include "nn.hpp"
#include "tictactoe.hpp"
#include <torch/torch.h>

struct TrainingSample {
  std::array<int, 9> state;
  std::vector<float> policy;
  float outcome;
};

int sample_from_policy(const std::vector<float> &policy) {
  static std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
  std::discrete_distribution<> dist(policy.begin(), policy.end());
  return dist(gen);
}

void train_network(TicTacToeNet &net, const std::vector<TrainingSample> &buffer,
                   int epochs, int batch_size) {
  net->train();
  torch::optim::Adam optimizer(net->parameters(),
                               torch::optim::AdamOptions(1e-3));

  size_t num_samples = buffer.size();
  if (num_samples == 0)
    return;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    float epoch_value_loss = 0.0f;
    float epoch_policy_loss = 0.0f;
    float epoch_total_loss = 0.0f;
    int num_batches = 0;

    std::vector<size_t> indices(num_samples);
    for (size_t i = 0; i < num_samples; ++i)
      indices[i] = i;
    std::shuffle(indices.begin(), indices.end(),
                 std::mt19937{std::random_device{}()});

    for (size_t i = 0; i < num_samples; i += batch_size) {
      size_t current_batch_size =
          std::min(static_cast<size_t>(batch_size), num_samples - i);

      std::vector<torch::Tensor> state_tensors;
      std::vector<torch::Tensor> policy_tensors;
      std::vector<float> outcomes;

      for (size_t j = i; j < i + current_batch_size; ++j) {
        size_t idx = indices[j];

        std::vector<float> state_vec(buffer[idx].state.begin(),
                                     buffer[idx].state.end());
        auto state_tensor = torch::tensor(state_vec).reshape({9});
        state_tensors.push_back(state_tensor);

        auto policy_tensor = torch::tensor(buffer[idx].policy);
        policy_tensors.push_back(policy_tensor);
        outcomes.push_back(buffer[idx].outcome);
      }

      auto state_batch = torch::stack(state_tensors).to(torch::kF32);
      auto policy_batch = torch::stack(policy_tensors).to(torch::kF32);
      auto outcome_batch = torch::tensor(outcomes).to(torch::kF32).unsqueeze(1);

      auto outputs = net->forward(state_batch);
      auto policy_logits = std::get<0>(outputs);
      auto value_pred = std::get<1>(outputs);

      auto value_loss = torch::mse_loss(value_pred, outcome_batch);
      auto log_probs = torch::log_softmax(policy_logits, /*dim=*/1);
      auto policy_loss = -(policy_batch * log_probs).sum(1).mean();
      auto loss = value_loss + policy_loss;

      epoch_value_loss += value_loss.item<float>();
      epoch_policy_loss += policy_loss.item<float>();
      epoch_total_loss += loss.item<float>();
      ++num_batches;

      std::cout << std::fixed << std::setprecision(4) << "Epoch ["
                << std::setw(3) << epoch + 1 << "/" << epochs << "]"
                << " | Batch [" << std::setw(3) << (i / batch_size) + 1 << "/"
                << (num_samples + batch_size - 1) / batch_size << "]"
                << " | Value Loss: " << std::setw(8) << value_loss.item<float>()
                << " | Policy Loss: " << std::setw(8)
                << policy_loss.item<float>()
                << " | Total Loss: " << std::setw(8) << loss.item<float>()
                << std::endl;

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    }

    std::cout << std::fixed << std::setprecision(4) << ">> Epoch [" << epoch + 1
              << "] Average - "
              << "Value Loss: " << epoch_value_loss / num_batches
              << " | Policy Loss: " << epoch_policy_loss / num_batches
              << " | Total Loss: " << epoch_total_loss / num_batches << "\n"
              << std::endl;
  }

  std::cout << "Training epoch complete. Buffer size: " << num_samples << "\n";
}

int main(int argc, char* argv[]) {
  const int num_iterations = 1000;    // Total training iterations.
  const int games_per_iteration = 20; // Self-play games per iteration.
  const int training_epochs = 5;      // How many epochs per training iteration.
  const int batch_size = 32;
  const int checkpoint_interval = 50; // Save model every N iterations.
  const std::string checkpoint_dir = "models/";

  std::filesystem::create_directories(checkpoint_dir);

  TicTacToeNet net = TicTacToeNet();

  if (argc > 1) {
    const std::string checkpoint_path = argv[1];
      std::cout << "Loading model from: " << checkpoint_path << std::endl;
      torch::load(net, checkpoint_path);
  }
  std::vector<TrainingSample> replay_buffer;

  for (int iter = 1; iter <= num_iterations; ++iter) {
    std::cout << "Training Iteration " << iter << std::endl;

    for (int game = 0; game < games_per_iteration; ++game) {
      std::vector<TrainingSample> game_samples;
      std::vector<Player> sample_players;
      TicTacToeState state;

      while (!state.is_terminal()) {
        MCTS mcts(net, 1.414, 100);
        auto [_, policy] = mcts.search(state);

        int action = sample_from_policy(policy);

        TrainingSample sample;
        sample.state = state.board;
        sample.policy = policy;
        sample.outcome = 0; // Outcome will be determined after the game.
        game_samples.push_back(sample);
        sample_players.push_back(state.current_player);

        state = state.step(action);
      }

      int outcome = state.reward(X);

      for (size_t i = 0; i < game_samples.size(); ++i) {
        game_samples[i].outcome = (sample_players[i] == X) ? outcome : -outcome;
        replay_buffer.push_back(game_samples[i]);
      }
    }

    train_network(net, replay_buffer, training_epochs, batch_size);

    if (iter % checkpoint_interval == 0) {
      std::string checkpoint_path =
          checkpoint_dir + "model_iter_" + std::to_string(iter) + ".pt";
      torch::save(net, checkpoint_path);
      std::cout << "Saved checkpoint: " << checkpoint_path << std::endl;
    }
  }

  return 0;
}
