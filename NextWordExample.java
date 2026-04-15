import java.util.*;

class NextWordModel {

    double[][] W; // [vocab][vocab]
    double[] b;
    double lr = 0.1;

    public NextWordModel(int vocabSize) {
        W = new double[vocabSize][vocabSize];
        b = new double[vocabSize];
        Random r = new Random();

        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < vocabSize; j++) {
                W[i][j] = r.nextDouble() * 0.1;
            }
        }
    }

    public double[] forward(double[] x) {
        double[] y = new double[W.length];

        for (int i = 0; i < W.length; i++) {
            y[i] = b[i];
            for (int j = 0; j < x.length; j++) {
                y[i] += W[i][j] * x[j];
            }
        }

        return softmax(y);
    }

    private double[] softmax(double[] z) {
        double sum = 0;
        double[] exp = new double[z.length];

        for (int i = 0; i < z.length; i++) {
            exp[i] = Math.exp(z[i]);
            sum += exp[i];
        }

        for (int i = 0; i < z.length; i++) {
            exp[i] /= sum;
        }

        return exp;
    }

    public void backward(double[] x, double[] yPred, int targetIndex) {

        for (int i = 0; i < W.length; i++) {

            double error = yPred[i] - (i == targetIndex ? 1 : 0);

            for (int j = 0; j < x.length; j++) {
                W[i][j] -= lr * error * x[j];
            }

            b[i] -= lr * error;
        }
    }
}

public class NextWordExample {

    static Map<String, Integer> vocab = new HashMap<>();
    static List<String> indexToWord = new ArrayList<>();



    static double[] vectorize(String sentence) {
        double[] vec = new double[vocab.size()];
        String[] words = sentence.split(" ");

        for (String w : words) {
            if (vocab.containsKey(w)) {
                vec[vocab.get(w)] = 1;
            }
        }
        return vec;
    }

    public static void main(String[] args) {

        // Training pairs
        String[][] data = {
            {"i love this", "car"},
            {"i hate this", "bike"},
            {"this is good", "ground"},
            {"this is bad", "thing"}
        };  

        // Build vocab
        int index = 0;
        for (String[] pair : data) {
            for (String w : (pair[0] + " " + pair[1]).split(" ")) {
                if (!vocab.containsKey(w)) {
                    vocab.put(w, index++);
                    indexToWord.add(w);
                }
            }
        }

        NextWordModel model = new NextWordModel(vocab.size());

        // Training
        for (int epoch = 0; epoch < 1000; epoch++) {

            double loss = 0;

            for (String[] pair : data) {

                double[] x = vectorize(pair[0]);
                int target = vocab.get(pair[1]);

                double[] yPred = model.forward(x);

                loss += -Math.log(yPred[target] + 1e-9);

                model.backward(x, yPred, target);
            }

            if (epoch % 200 == 0) {
                System.out.println("Epoch " + epoch + " Loss: " + loss);
            }
        }

        // Test
        Scanner scan = new Scanner(System.in);

        System.out.println("Enter the input");
        String input = scan.nextLine();
        double[] test = vectorize(input);
        double[] pred = model.forward(test);

        int best = 0;
        for (int i = 1; i < pred.length; i++) {
            if (pred[i] > pred[best]) best = i;
        }

        System.out.println("Prediction: " + input +" "+indexToWord.get(best));
    }
}