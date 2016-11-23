package iprocess;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

import Jama.Matrix;

public class ANN {
	ArrayList<Matrix> weights;
	ArrayList<Matrix> activations;
	ArrayList<Matrix> biasWeights;

	Matrix inputX;
	Matrix inputY;

	int activationNodes;
	int hiddenLayers;
	int inputNodes;
	int classNodes;
	double learningRate;

	public ANN(String filePath) throws Exception {
		FileReader fileReader = new FileReader(filePath);
		BufferedReader bufferedReader = new BufferedReader(fileReader);

		inputNodes = Integer.parseInt(bufferedReader.readLine());
		activationNodes = Integer.parseInt(bufferedReader.readLine());
		hiddenLayers = Integer.parseInt(bufferedReader.readLine());
		classNodes = Integer.parseInt(bufferedReader.readLine());
		learningRate = Double.parseDouble(bufferedReader.readLine());
		int numWeights = Integer.parseInt(bufferedReader.readLine());

		weights = new ArrayList<Matrix>();
		activations = new ArrayList<Matrix>(hiddenLayers + 2);
		biasWeights = new ArrayList<Matrix>();

		// initialize weights including bias
		for (int i = 0; i <= hiddenLayers; i++) {

			Matrix aWeight;
			Matrix aBiasWeight;
			if (i == 0) {
				aWeight = new Matrix(new double[inputNodes][activationNodes]);
				aBiasWeight = new Matrix(new double[1][activationNodes]);
			} else if (i == hiddenLayers) {
				aWeight = new Matrix(new double[activationNodes][classNodes]);
				aBiasWeight = new Matrix(new double[1][classNodes]);
			} else {
				aWeight = new Matrix(new double[activationNodes][activationNodes]);
				aBiasWeight = new Matrix(new double[1][activationNodes]);
			}

			// fill weights
			fillWeights(bufferedReader, aWeight);
			fillWeights(bufferedReader, aBiasWeight);

			weights.add(aWeight);
			biasWeights.add(aBiasWeight);

			activations.add(null);
		}

		activations.add(null);

		bufferedReader.close();
	}

	public ANN(int aInputNodes, int aActivationNodes, int aClassNodes, int aHiddenLayers, double aLearningRate) {
		activationNodes = aActivationNodes;
		hiddenLayers = aHiddenLayers;
		learningRate = aLearningRate;
		inputNodes = aInputNodes;
		classNodes = aClassNodes;

		weights = new ArrayList<Matrix>();
		activations = new ArrayList<Matrix>(hiddenLayers + 2);
		biasWeights = new ArrayList<Matrix>();

		// initialize weights including bias
		for (int i = 0; i <= aHiddenLayers; i++) {

			Matrix aWeight;
			Matrix aBiasWeight;
			if (i == 0) {
				aWeight = new Matrix(new double[inputNodes][activationNodes]);
				aBiasWeight = new Matrix(new double[1][activationNodes]);
			} else if (i == aHiddenLayers) {
				aWeight = new Matrix(new double[activationNodes][classNodes]);
				aBiasWeight = new Matrix(new double[1][classNodes]);
			} else {
				aWeight = new Matrix(new double[activationNodes][activationNodes]);
				aBiasWeight = new Matrix(new double[1][activationNodes]);
			}

			initializeWeights(aWeight);
			initializeWeights(aBiasWeight);

			weights.add(aWeight);
			biasWeights.add(aBiasWeight);

			activations.add(null);
		}

		activations.add(null);
	}

	public void fillWeights(BufferedReader reader, Matrix a) throws Exception {
		for (int i = 0; i < a.getRowDimension(); i++) {
			for (int j = 0; j < a.getColumnDimension(); j++) {
				a.set(i, j, Double.parseDouble(reader.readLine()));
			}
		}
	}

	public void train(double[] aInputX, double[] aInputY) {
		inputX = new Matrix(aInputX, 1); // inputNodes x 1
		inputY = new Matrix(aInputY, 1);

		activations.set(0, inputX);

		forwardPropragation();
		// activationMatrix(activations.get(0));
		backwardPropragation();
	}

	public void test(double[] aInputX) {
		inputX = new Matrix(aInputX, 1);
		activations.set(0, inputX);
		forwardPropragation();
	}

	public void initializeWeights(Matrix weights) {
		for (int i = 0; i < weights.getRowDimension(); i++)
			for (int j = 0; j < weights.getColumnDimension(); j++)
				weights.set(i, j, 2 * Math.random() - 1);
	}

	public static void printDimensions(Matrix a) {
		System.out.println(a.getColumnDimension() + "  " + a.getRowDimension());
	}

	public static void printMatrix(Matrix a) {
		System.out.println(Arrays.deepToString(a.getArray()));
	}

	public double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	public void activationMatrix(Matrix a) {
		for (int i = 0; i < a.getColumnDimension(); i++) {
			double actVal = sigmoid(a.get(0, i));
			a.set(0, i, actVal);
		}
	}

	public Matrix activationMatrixPrime(Matrix a) {
		for (int i = 0; i < a.getColumnDimension(); i++) {
			double num = a.get(0, i);
			double actVal = num * (1 - num);
			a.set(0, i, actVal);
		}

		return a;
	}

	public void update(int index, Matrix left) {
		Matrix aMat = left.times(weights.get(index));
		// zActivations.set(index, aMat.plusEquals(biasWeights.get(index))); //
		// z
		aMat = aMat.plus(biasWeights.get(index));
		activationMatrix(aMat); // a
		activations.set(index + 1, aMat);
	}

	public void forwardPropragation() {
		for (int i = 0; i <= hiddenLayers; i++)
			update(i, activations.get(i));
	}

	public void setWeightChange(Matrix delta, int index) {
		// set outer layer weights
		Matrix weightChange = (activations.get(index).transpose().times(delta)).times(learningRate);
		weights.get(index).plusEquals(weightChange);
		biasWeights.get(index).plusEquals(delta.times(learningRate));
	}

	public void backwardPropragation() {
		Matrix deltaOut;
		Matrix deltaIn;
		Matrix tempWeight;

		// output layer
		deltaOut = inputY.minus(activations.get(hiddenLayers + 1));
		Matrix actPrime = activationMatrixPrime(activations.get(hiddenLayers + 1));
		deltaOut.arrayTimesEquals(actPrime);
		tempWeight = weights.get(hiddenLayers).copy();
		setWeightChange(deltaOut, hiddenLayers);

		for (int i = hiddenLayers; i > 0; i--) {
			Matrix aActPrime = activationMatrixPrime(activations.get(i));
			deltaIn = deltaOut.times(tempWeight.transpose()).arrayTimesEquals(aActPrime);
			// update weights
			tempWeight = weights.get(i - 1).copy();
			setWeightChange(deltaIn, i - 1);
			deltaOut = deltaIn;
		}
	}

	public Matrix getClasses() {
		return activations.get(hiddenLayers + 1);
	}

	public ArrayList<Matrix> getWeights() {
		return weights;
	}

	public ArrayList<Matrix> getBiasWeights() {
		return biasWeights;
	}

	public void saveWeights(String weightsPath) throws Exception {
		PrintWriter weightsWriter = new PrintWriter(weightsPath);

		weightsWriter.println(inputNodes);
		weightsWriter.println(activationNodes);
		weightsWriter.println(hiddenLayers);
		weightsWriter.println(classNodes);
		weightsWriter.println(learningRate);
		weightsWriter.println(weights.size());

		for (int i = 0; i < weights.size(); i++) {
			double[][] arr = weights.get(i).getArray();
			double[][] arr2 = biasWeights.get(i).getArray();

			for (int j = 0; j < arr.length; j++) {
				for (int k = 0; k < arr[0].length; k++) {
					weightsWriter.println(arr[j][k]);
				}
			}

			for (int a = 0; a < arr2.length; a++) {
				for (int b = 0; b < arr2[0].length; b++) {
					weightsWriter.println(arr2[a][b]);
				}
			}
		}
		weightsWriter.close();
	}

}