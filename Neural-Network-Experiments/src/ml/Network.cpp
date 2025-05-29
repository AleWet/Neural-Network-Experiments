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

Eigen::VectorXf Network::Forward(const Eigen::VectorXf& input)
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
	// TRANSCRIBE WRITTEN NOTES ONTO OBSIDIAN

	Forward(input);
	int numLayers = m_LayerSizes.size();
	int outputLayerIndex = numLayers - 1;

	// Initialize gradients to 0
	for (size_t i = 0; i < m_WeightGradients.size(); i++) 
	{
		m_WeightGradients[i] = Eigen::MatrixXf::Zero(m_Weights[i].rows(), m_Weights[i].cols());
		m_BiasGradients[i] = Eigen::VectorXf::Zero(m_Biases[i].size());
	}

	// Error = ∂C(network) / ∂z

	// Calculate output layer error : C'(a) ⊙ σ'(z)
	Eigen::VectorXf outputError = m_Activations[outputLayerIndex] - target;
	m_Deltas[outputLayerIndex] = outputError.cwiseProduct(
		ActivationFunctionDerivative(m_PreActivations[outputLayerIndex])
	);

	// Propagate the error backwords 
	for (int layer = outputLayerIndex - 1; layer >= 1; layer--) {
		// error for any layer : (W^T * delta_next) ⊙ σ'(z)
		m_Deltas[layer] = (m_Weights[layer].transpose() * m_Deltas[layer + 1]).cwiseProduct(
			ActivationFunctionDerivative(m_PreActivations[layer])
		);
	}

	// Calculate gradients for weigths and biases (you needed ∂C(network) / ∂z)
	for (int layer = 0; layer < numLayers - 1; layer++) {
		// W gradient = error * activation^(L-1) -> you also need to transpose for dimension reasons 
		m_WeightGradients[layer] = m_Deltas[layer + 1] * m_Activations[layer].transpose();

		// B gradient = error
		m_BiasGradients[layer] = m_Deltas[layer + 1];
	}

	// Update network with the components of the gradient of the Cost() 
	for (size_t layer = 0; layer < m_Weights.size(); layer++) 
	{
		m_Weights[layer] -= learningRate * m_WeightGradients[layer];
		m_Biases[layer] -= learningRate * m_BiasGradients[layer];
	}

}

void Network::TrainBatch(const std::vector<DataSample>& batch, float learningRate)
{
	if (batch.empty()) return;

	// Initialize cumulative gradients	
	std::vector<Eigen::MatrixXf> WeightGradients(m_Weights.size());
	std::vector<Eigen::VectorXf> BiasGradients(m_Biases.size());
	for (size_t i = 0; i < m_Weights.size(); i++) 
	{
		WeightGradients[i] = Eigen::MatrixXf::Zero(m_Weights[i].rows(), m_Weights[i].cols());
		BiasGradients[i] = Eigen::VectorXf::Zero(m_Biases[i].size());
	}

	for (const auto& sample : batch) {
		// Calculate deltas (same as in BackPropagation)
		int numLayers = m_LayerSizes.size();
		int outputLayerIndex = numLayers - 1;

		Eigen::VectorXf outputError = m_Activations[outputLayerIndex] - sample.target;
		m_Deltas[outputLayerIndex] = outputError.cwiseProduct(
			ActivationFunctionDerivative(m_PreActivations[outputLayerIndex])
		);

		for (int layer = outputLayerIndex - 1; layer >= 1; layer--) 
		{
			m_Deltas[layer] = (m_Weights[layer].transpose() * m_Deltas[layer + 1]).cwiseProduct(
				ActivationFunctionDerivative(m_PreActivations[layer])
			);
		}

		// Accumulate gradients instead of updating the weigths and biases immediately
		for (int layer = 0; layer < numLayers - 1; layer++) 
		{
			WeightGradients[layer] += m_Deltas[layer + 1] * m_Activations[layer].transpose();
			BiasGradients[layer] += m_Deltas[layer + 1];
		}
	}

	// Update parameters
	float batchSize = static_cast<float>(batch.size());
	for (size_t layer = 0; layer < m_Weights.size(); layer++) 
	{
		m_Weights[layer] -= learningRate * WeightGradients[layer] / batchSize;
		m_Biases[layer] -= learningRate * BiasGradients[layer] / batchSize;
	}
}

// Not the most optimal implementation but it's easy and it works
float Network::CalculateAccuracy(const std::vector<DataSample>& testBatch) 
{
	if (testBatch.empty()) return 0.0f;

	int correct = 0;
	for (const auto& sample : testBatch) 
	{
		Eigen::VectorXf output = Forward(sample.input);

		// Find predicted class (highest output)
		int predicted = 0;
		float maxOutput = output[0];
		for (int i = 1; i < output.size(); i++) 
		{
			if (output[i] > maxOutput) 
			{
				maxOutput = output[i];
				predicted = i;
			}
		}

		if (predicted == sample.label) 
			correct++;
	}

	return static_cast<float>(correct) / static_cast<float>(testBatch.size());
}

// Not the most optimal implementation but it's easy and it works
float Network::CalculateAverageLoss(const std::vector<DataSample>& testBatch) 
{
	if (testBatch.empty()) return -1.0f;

	float totalLoss = 0.0f;
	for (const auto& sample : testBatch) 
	{
		Eigen::VectorXf output = Forward(sample.input);
		totalLoss += LossFunction(output, sample.target);
	}

	return totalLoss / static_cast<float>(testBatch.size());
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




