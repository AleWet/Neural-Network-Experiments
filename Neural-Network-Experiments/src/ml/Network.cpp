#include "Network.h"

Network::Network(const std::vector<int>& sizes)
	:m_LayerSizes(sizes)
{
	m_Weights.resize(sizes.size() - 1);          // N-1 connection layers
	m_Biases.resize(sizes.size() - 1);           // N-1 bias vectors
	m_Activations.resize(sizes.size());          // N activation vectors
	m_PreActivations.resize(sizes.size());       // N pre-activation vectors
	m_WeightGradients.resize(sizes.size() - 1);  // N-1 weight gradient matrices
	m_BiasGradients.resize(sizes.size() - 1);    // N-1 bias gradient vectors
	m_Deltas.resize(sizes.size());               // N delta vectors

	// RANDOM INTIALISATION SHOULD BE RE-MADE (there are nuances that I don't know yet)

	// Initialize the weights and the biases for each layer
	for (size_t i = 0; i < sizes.size() - 1; i++) // Fix loop bounds
	{
		m_Weights[i] = Eigen::MatrixXf::Random(sizes[i + 1], sizes[i]);
		m_Biases[i] = Eigen::VectorXf::Zero(sizes[i + 1]);
	}

	// Initialize Zs and As with zeros
	for (size_t i = 0; i < sizes.size(); i++)
	{
		m_Activations[i] = Eigen::VectorXf::Zero(sizes[i]);
		m_PreActivations[i] = Eigen::VectorXf::Zero(sizes[i]);
		m_Deltas[i] = Eigen::VectorXf::Zero(sizes[i]);
	}
}

Network::~Network()
{
}

Eigen::VectorXf Network::forward(const Eigen::VectorXf& input)
{
	m_Activations[0] = input;
	m_PreActivations[0] = input;

	// Propagate through the layers
	for (size_t i = 0; i < m_Weights.size(); i++) {
		m_PreActivations[i + 1] = m_Weights[i] * m_Activations[i] + m_Biases[i];
		m_Activations[i + 1] = ActivationFunction(m_PreActivations[i + 1]);
	}

	// Return the output layer activation (FOR NOW this hasn't a different activation function)
	return m_Activations.back();
}

void Network::BackPropagation(const Eigen::VectorXf& input, const Eigen::VectorXf& target, float learningRate)
{
	// Assume you have already called Forward() AND that you have already calculated the deltas for the current batch

	int numLayers = m_LayerSizes.size();
	int outputLayerIndex = numLayers - 1;

	// TBD
}

// FOR THE MOMENT ONLY SIGMOID 
Eigen::VectorXf Network::ActivationFunction(const Eigen::VectorXf& x)
{
	return 1.0f / (1.0f + (-x).array().exp());
}

// Sigmoid derivative : sigmoid(x)* (1 - sigmoid(x))
Eigen::VectorXf Network::ActivationFunctionDerivative(const Eigen::VectorXf& x)
{
	Eigen::VectorXf sigmoidX = ActivationFunction(x);
	return sigmoidX.cwiseProduct(Eigen::VectorXf::Ones(sigmoidX.size()) - sigmoidX);
}

float Network::LossFunction(const Eigen::VectorXf& output, const Eigen::VectorXf& target)
{
	Eigen::VectorXf diff = output - target;
	return 0.5f * diff.dot(diff);				// sum of squared differences (I don't know why online it's * 0.5 yet)
}

void Network::TrainBatch(const std::vector<DataSample>& batch, float learningRate) 
{
	// TBD
}



