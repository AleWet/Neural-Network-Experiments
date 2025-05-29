#pragma once
#include "Dataset.h"


class Network
{
private:
	std::vector<int> m_LayerSizes;
	std::vector<Eigen::MatrixXf> m_Weights;
	std::vector<Eigen::VectorXf> m_Biases;
	std::vector<Eigen::VectorXf> m_Activations;	   // a = actFun(Z)
	std::vector<Eigen::VectorXf> m_PreActivations; // z

	// For backpropagation
	std::vector<Eigen::MatrixXf> m_WeightGradients;
	std::vector<Eigen::VectorXf> m_BiasGradients;
	std::vector<Eigen::VectorXf> m_Deltas;

	Eigen::VectorXf ActivationFunction(const Eigen::VectorXf&);
	Eigen::VectorXf ActivationFunctionDerivative(const Eigen::VectorXf& x);

	// Returns the loss for a single target and single input
	// Theoretical loss/cost function : 
	//		INPUT = output del network su N samples (one batch), BIASES and WEIGHTS of the network
	//		OUTPUT = (somma di tutti (output - expected)^2) * 1/N  
	float LossFunction(const Eigen::VectorXf& output, const Eigen::VectorXf& target);

public:
	Network(const std::vector<int>& sizes);
	~Network();

	// Advance the input in the simulation
	Eigen::VectorXf Forward(const Eigen::VectorXf& input);

	void BackPropagation(const Eigen::VectorXf& input, const Eigen::VectorXf& target, float learningRate);

	// Setters
	void setWeights(int layerIndex, const Eigen::MatrixXf& newWeights)
	{
		if (layerIndex >= 0 && layerIndex <= m_Weights.size())
			m_Weights[layerIndex] = newWeights;

	}
	void setBiases(int layerIndex, const Eigen::VectorXf& newBiases)
	{
		if (layerIndex >= 0 && layerIndex <= m_Biases.size())
			m_Biases[layerIndex] = newBiases;

	}

	// Getters
	int getLayerCount() const { return m_LayerSizes.size(); }
	int getLayerSize(int layerIndex) const { return m_LayerSizes[layerIndex]; }
	const Eigen::MatrixXf& getWeights(int layerIndex) const { return m_Weights[layerIndex]; }
	const Eigen::VectorXf& getBiases(int layerIndex) const { return m_Biases[layerIndex]; }

	// Get activation levels output
	Eigen::VectorXf getLayerOutput(int layerIndex) const
	{
		if (layerIndex >= 0 && layerIndex < m_Activations.size()) {
			return m_Activations[layerIndex];
		}
		return Eigen::VectorXf();
	}

	// Same as backProp but with a batch of input data to approximate Cost()
	void TrainBatch(const std::vector<DataSample>& batch, float learningRate);

	float CalculateAccuracy(const std::vector<DataSample>& testBatch);
	float CalculateAverageLoss(const std::vector<DataSample>& testBatch);
};