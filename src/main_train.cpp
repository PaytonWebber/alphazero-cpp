#include <algorithm>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mcts.hpp"
#include "az_net.hpp"
#include "othello.hpp"
#include <torch/torch.h>

struct TrainingSample {
  std::vector<float> state;
  std::vector<float> policy;
  float outcome;
};

std::vector<float> flip_horizontal_channel(const std::vector<float>& channel) {
    std::vector<float> flipped(64, 0.0f);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            flipped[i * 8 + j] = channel[i * 8 + (7 - j)];
        }
    }
    return flipped;
}

std::vector<float> flip_horizontal_state(const std::vector<float>& state) {
    std::vector<float> flipped;
    for (int c = 0; c < 2; ++c) {
        std::vector<float> channel(state.begin() + c * 64, state.begin() + (c + 1) * 64);
        std::vector<float> flipped_channel = flip_horizontal_channel(channel);
        flipped.insert(flipped.end(), flipped_channel.begin(), flipped_channel.end());
    }
    return flipped;
}

std::vector<float> flip_horizontal_policy(const std::vector<float>& policy) {
    std::vector<float> flipped(65, 0.0f);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            flipped[i * 8 + j] = policy[i * 8 + (7 - j)];
        }
    }
    flipped[64] = policy[64];
    return flipped;
}

std::vector<float> rotate_90_channel(const std::vector<float>& channel) {
    std::vector<float> rotated(64, 0.0);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            rotated[j * 8 + (7 - i)] = channel[i * 8 + j];
        }
    }
    return rotated;
}

std::vector<float> rotate_90_state(const std::vector<float>& state) {
    std::vector<float> rotated_state;
    for (int c = 0; c < 2; ++c) {
        std::vector<float> channel(state.begin() + c * 64, state.begin() + (c + 1) * 64);
        std::vector<float> rotated_channel = rotate_90_channel(channel);
        rotated_state.insert(rotated_state.end(), rotated_channel.begin(), rotated_channel.end());
    }
    return rotated_state;
}

std::vector<float> rotate_90_policy(const std::vector<float>& policy) {
  std::vector<float> rotated_policy(65, 0.0f);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      rotated_policy[j * 8 + (7 - i)] = policy[i * 8 + j];
    }
  }
  rotated_policy[64] = policy[64];
  return rotated_policy;
}

std::vector<TrainingSample> augment_sample(const TrainingSample &sample) {
    std::vector<TrainingSample> augmented_samples;
    augmented_samples.push_back(sample);

    TrainingSample rotated_sample = sample;
    for (int i = 0; i < 3; ++i) {
        rotated_sample.state = rotate_90_state(rotated_sample.state);
        rotated_sample.policy = rotate_90_policy(rotated_sample.policy);
        augmented_samples.push_back(rotated_sample);
    }

    TrainingSample flipped_sample = sample;
    flipped_sample.state = flip_horizontal_state(sample.state);
    flipped_sample.policy = flip_horizontal_policy(sample.policy);
    augmented_samples.push_back(flipped_sample);

    TrainingSample rotated_flipped = flipped_sample;
    for (int i = 0; i < 3; ++i) {
        rotated_flipped.state = rotate_90_state(rotated_flipped.state);
        rotated_flipped.policy = rotate_90_policy(rotated_flipped.policy);
        augmented_samples.push_back(rotated_flipped);
    }

    return augmented_samples;
}

int sample_from_policy(const std::vector<float> &policy) {
  static std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)));
  std::discrete_distribution<> dist(policy.begin(), policy.end());
  return dist(gen);
}

void train_network(AZNet &net, const std::vector<TrainingSample> &buffer,
                   int epochs, int batch_size) {
  net->train();
  torch::optim::Adam optimizer(net->parameters(),
                               torch::optim::AdamOptions(1e-3));

  size_t num_samples = buffer.size();
  if (num_samples == 0) { return; }

  for (int epoch = 0; epoch < epochs; ++epoch) {
    float epoch_value_loss = 0.0f;
    float epoch_policy_loss = 0.0f;
    float epoch_total_loss = 0.0f;
    int num_batches = 0;

    std::vector<size_t> indices(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
      indices[i] = i;
    }
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
        auto state_tensor = torch::tensor(state_vec).reshape({2, 8, 8});
        state_tensors.push_back(state_tensor);

        auto policy_tensor = torch::tensor(buffer[idx].policy);
        policy_tensors.push_back(policy_tensor);
        outcomes.push_back(buffer[idx].outcome);
      }

      auto state_batch = torch::stack(state_tensors).to(torch::kF32);
      auto policy_batch = torch::stack(policy_tensors).to(torch::kF32);
      auto outcome_batch = torch::tensor(outcomes).to(torch::kF32).unsqueeze(1);

      NetOutputs outputs = net->forward(state_batch);
      auto policy_logits = outputs.pi;
      auto value_pred = outputs.v;

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
  const int num_iterations = 1000;    
  const int games_per_iteration = 20; 
  const int training_epochs = 5;      
  const int batch_size = 64;
  const int checkpoint_interval = 10; 
  const std::string checkpoint_dir = "models/";

  std::filesystem::create_directories(checkpoint_dir);

  AZNet net = AZNet(2, 64, 65, 5);

  if (argc > 1) {
    const std::string checkpoint_path = argv[1];
      std::cout << "Loading model from: " << checkpoint_path << std::endl;
      torch::load(net, checkpoint_path);
  }
  std::vector<TrainingSample> replay_buffer;

  MCTS mcts(net, 1.414, 100, true);
  for (int iter = 1; iter <= num_iterations; ++iter) {
    std::cout << "Training Iteration " << iter << std::endl;

    for (int game = 0; game < games_per_iteration; ++game) {
      std::vector<TrainingSample> game_samples;
      std::vector<Player> sample_players;
      OthelloState state;

      while (!state.is_terminal()) {
        auto [_, policy] = mcts.search(state);

        int action = sample_from_policy(policy);

        TrainingSample sample;
        sample.state = state.board();
        sample.policy = policy;
        sample.outcome = 0; // this will be updated once the game is finished
        game_samples.push_back(sample);
        sample_players.push_back(state.current_player);

        state = state.step(action);
      }

      int outcome = state.reward(Player::Black);
      for (size_t i = 0; i < game_samples.size(); ++i) {
        game_samples[i].outcome = (sample_players[i] == Player::Black) ? outcome : -outcome;
        std::vector<TrainingSample> augmented_samples = augment_sample(game_samples[i]);
        replay_buffer.insert(replay_buffer.end(), augmented_samples.begin(), augmented_samples.end());
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
