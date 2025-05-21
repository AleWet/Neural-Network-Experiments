#pragma once
#include <vector>
#include <string>
#include "Eigen/Dense"

class Network
{
private:
	std::vector<int> m_LayerSizes;
	std::vector<Eigen::MatrixXf> m_Weigths;
	std::vector<Eigen::MatrixXf> m_Biases;
	std::vector<Eigen::VectorXf> m_Activations; // OUTPUT OF THE SINGLE LAYER

	// RETURNS THE UPDATED VALUES
	Eigen::VectorXf ActivationFunction(const Eigen::VectorXf&);

public:
	Network(const std::vector<int>& sizes);
	~Network();

	// Advance the input in the simulation
	Eigen::VectorXf forward(const Eigen::VectorXf& input);

	// Setters
	void setWeights(int layerIndex, const Eigen::MatrixXf& newWeights)
	{
		if (layerIndex > 0 && layerIndex < m_Weigths.size())
			m_Weigths[layerIndex] = newWeights;
		
	}
	void setBiases(int layerIndex, const Eigen::VectorXf& newBiases)
	{
		if (layerIndex > 0 && layerIndex < m_Biases.size()) 
			m_Biases[layerIndex] = newBiases;
		
	}

	// Getters
	int getLayerCount() const { return m_LayerSizes.size(); }
	int getLayerSize(int layerIndex) const { return m_LayerSizes[layerIndex]; }
	const Eigen::MatrixXf& getWeights(int layerIndex) const { return m_Weigths[layerIndex]; }
	const Eigen::VectorXf& getBiases(int layerIndex) const { return m_Biases[layerIndex]; }

	// Get activation levels output
	Eigen::VectorXf getLayerOutput(int layerIndex) const 
	{
		if (layerIndex >= 0 && layerIndex < m_Activations.size()) {
			return m_Activations[layerIndex];
		}
		return Eigen::VectorXf();
	}
};
