#include "Network.h"

Network::Network(const std::vector<int>& sizes)
	:m_LayerSizes(sizes)
{
	m_Weigths.resize(sizes.size() - 1); // There are N-1 "connections layers"
	m_Biases.resize(sizes.size() - 1);  // Maybe the last layer doesn't have the bias?
	m_Activations.resize(sizes.size());

	// RANDOM INTIALISATION SHOULD BE RE-MADE (there are nuances that I don't know yet)

	// Initialize the weights and the biases for each layer
	for (size_t i = 0; i < sizes.size() - 1; i++) // Fix loop bounds
	{
		m_Weigths[i] = Eigen::MatrixXf::Random(sizes[i + 1], sizes[i]);
		m_Biases[i] = Eigen::VectorXf::Zero(sizes[i + 1]);
	}

	// Initialize activations with zeros
	for (size_t i = 0; i < sizes.size(); i++)
	{
		m_Activations[i] = Eigen::VectorXf::Zero(sizes[i]);
	}
}

Network::~Network()
{
}

Eigen::VectorXf Network::forward(const Eigen::VectorXf& input)
{
	m_Activations[0] = input;

	// Propagate through the layers
	for (size_t i = 0; i < m_Weigths.size(); i++) {
		m_Activations[i + 1] = m_Weigths[i] * m_Activations[i] + m_Biases[i];
		m_Activations[i + 1] = ActivationFunction(m_Activations[i + 1]);
	}

	// Return the output layer activation (FOR NOW this hasn't a different activation function)
	return m_Activations.back();
}

// FOR THE MOMENT ONLY SIGMOID 
Eigen::VectorXf Network::ActivationFunction(const Eigen::VectorXf& x)
{
	return 1.0f / (1.0f + (-x).array().exp());
}





