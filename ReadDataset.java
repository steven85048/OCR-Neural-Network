package iprocess;

import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Arrays;

import javax.imageio.ImageIO;

import Jama.Matrix;

public class ReadDataset {
	static int LABEL_MAGIC = 2049;
	static int IMAGE_MAGIC = 2051;

	static int NUM_TRAIN = 60000;
	static int NUM_TEST = 10000;

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		ANN neuralNet = new ANN(28 * 28, 150, 10, 1, .4);

		String imagesPath = "C://Users//Steven//Desktop//train-images.idx3-ubyte";
		String labelsPath = "C://Users//Steven//Desktop//train-labels.idx1-ubyte";
		// saveImage(imagesPath);
		trainTest(imagesPath, labelsPath, 0, neuralNet);

		neuralNet.saveWeights("C:/Users/Steven/Desktop/weights.txt");
		
//		ANN neuralNet = new ANN("C:/Users/Steven/Desktop/weights.txt");
		
		String testImagesPath = "C://Users//Steven//Desktop//t10k-images.idx3-ubyte";
		String testLabelsPath = "C://Users//Steven//Desktop//t10k-labels.idx1-ubyte";
		trainTest(testImagesPath, testLabelsPath, 1, neuralNet);
		
//		saveImage(testImagesPath);

	}

	public static void saveImage(String imgPath) throws Exception {
		DataInputStream images = new DataInputStream(new FileInputStream(
				imgPath));
		images.readInt();
		images.readInt();
		images.readInt();

		String newDir = "C:/Users/Steven/Desktop/SHITSHIT/";

		for (int i = 0; i < 100; i++) {
			BufferedImage finalImage = new BufferedImage(28, 28,
					BufferedImage.TYPE_INT_RGB);
			for (int j = 0; j < 28; j++) {
				for (int k = 0; k < 28; k++) {
					int rgb = images.readUnsignedByte();
					finalImage.setRGB(k, j, rgb);
				}
			}
			String aDir = newDir + i + ".png";
			File f = new File(aDir);
			ImageIO.write(finalImage, "png", f);
		}
	}

	// 0 to train; 1 to test
	public static void trainTest(String imgPath, String lblPath, int test,
			ANN neuralNet) throws Exception {
		DataInputStream images = new DataInputStream(new FileInputStream(
				imgPath));
		DataInputStream labels = new DataInputStream(new FileInputStream(
				lblPath));

		int lblMag = labels.readInt();
		int imgMag = images.readInt();

		int numSet = labels.readInt();
		images.readInt();

		int rowLen = images.readInt();
		int colLen = images.readInt();

		if (lblMag != LABEL_MAGIC || imgMag != IMAGE_MAGIC)
			System.out.println("MAGIC NUMBER NOT CORRECT");

		if (test == 0) {
			for (int i = 0; i < NUM_TRAIN; i++) {
				System.out.println(i);
				double[] inputX;
				double[] inputY;

				inputX = getX(images, rowLen, colLen);
				inputY = getY(labels);

				neuralNet.train(inputX, inputY);
			}
		} else {
			int success = 0;
			for (int i = 0; i < NUM_TEST; i++) {
				double[] inputX;
				inputX = getX(images, rowLen, colLen);
				int check = labels.readByte();
				neuralNet.test(inputX);
				double[][] a = neuralNet.getClasses().getArray();
				int val = findMaxIndex(a[0]);
				if (val == check) {
					System.out.println(check + Arrays.deepToString(a));
					success++;
				}
			}
			System.out.println(success);
		}

		System.out.println("DONE");

	}

	public static int findMaxIndex(double[] arr) {
		double a = Double.MIN_VALUE;
		int index = 0;
		for (int i = 0; i < arr.length; i++) {
			if (a < arr[i]) {
				index = i;
				a = arr[i];
			}
		}

		return index;
	}

	public static void printList(ArrayList<Matrix> arr) {
		for (int i = 0; i < arr.size(); i++) {
			System.out.println(Arrays.deepToString(arr.get(i).getArray()));
		}
	}

	public static double[] getX(DataInputStream images, int rowLen, int colLen)
			throws Exception {
		double[] arr = new double[rowLen * colLen];
		for (int i = 0; i < (rowLen * colLen); i++) {
			int curr = images.readUnsignedByte();
			arr[i] = curr / 256.0;
		}

		return arr;
	}

	public static double[] getY(DataInputStream labels) throws Exception {
		int curr = labels.readUnsignedByte();
		double[] arr = new double[10];
		arr[curr] = 1;
		return arr;
	}

}