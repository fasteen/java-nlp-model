import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

class LinearLayer {

    double[][] W; // [output][input]
    double[] b;
    double learningRate = 0.01;

    public LinearLayer(int inputSize, int outputSize) {
        W = new double[outputSize][inputSize];
        b = new double[outputSize];

        Random rand = new Random();

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                W[i][j] = rand.nextDouble() * 0.1;
            }
            b[i] = 0;
        }
    }

    // Forward: y = Wx + b
    public double[] forward(double[] x) {
        double[] y = new double[W.length];

        for (int i = 0; i < W.length; i++) {
            y[i] = b[i];
            for (int j = 0; j < x.length; j++) {
                y[i] += W[i][j] * x[j];
            }
        }
        return y;
    }

    // Backward + Update
    public void backward(double[] x, double[] yPred, double[] yTrue) {
        for (int i = 0; i < W.length; i++) {
            double error = yPred[i] - yTrue[i];

            for (int j = 0; j < x.length; j++) {
                W[i][j] -= learningRate * error * x[j];
            }

            b[i] -= learningRate * error;
        }
    }
}

public class Main {

    public static void main(String[] args) {

        String filePath = "fuel_data.csv";

        List<double[]> X_list = new ArrayList<>();
        List<double[]> Y_list = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {

            String line;
            br.readLine();

            while ((line = br.readLine()) != null) {
                String[] v = line.split(",");

                double speed = Double.parseDouble(v[0]) / 100.0;
                double engine = Double.parseDouble(v[1]) / 3000.0;
                double mileage = Double.parseDouble(v[2]);

                X_list.add(new double[]{speed, engine});
                Y_list.add(new double[]{mileage});
            }

        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }

        int n = X_list.size();

        LinearLayer model = new LinearLayer(2, 1);

        int epochs = 5000;

        for (int epoch = 0; epoch < epochs; epoch++) {

            double loss = 0;

            for (int i = 0; i < n; i++) {

                double[] x = X_list.get(i);
                double[] yTrue = Y_list.get(i);

                double[] yPred = model.forward(x);

                double error = yPred[0] - yTrue[0];
                loss += error * error;

                model.backward(x, yPred, yTrue);
            }

            if (epoch % 500 == 0) {
                System.out.println("Epoch " + epoch + " Loss: " + loss / n);
            }
        }

    
        double[] test = {60.0 / 100.0, 1500.0 / 3000.0};
        double[] pred = model.forward(test);

        System.out.println("\nTest Input: Speed=60, Engine=1500");
        System.out.println("Predicted Mileage: " + pred[0]);
    }
}