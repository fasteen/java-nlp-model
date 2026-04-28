import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;
import java.util.*;


 class Model  implements java.io.Serializable{
    private static final long serialVersionUID = 1L;

    double[][] w1;
    double[][] w2;
    double[] b1;
    double[] b2;
    double learningRate = 0.0005;


    int inputLayer , hiddenLayer, outputLayer;

    Model(int inputLayer , int hiddenLayer , int outputLayer){
        this.inputLayer = inputLayer;
        this.hiddenLayer = hiddenLayer;
        this.outputLayer = outputLayer;

        Random rng = new Random();
        w1 = new double[hiddenLayer][inputLayer];
        w2 = new double[outputLayer][hiddenLayer];
        b1 = new double[hiddenLayer];
        b2 = new double[outputLayer];

        for(int i=0; i<hiddenLayer; i++){
            for(int j=0; j<inputLayer; j++){

                w1[i][j] = rng.nextGaussian() * 0.01;

            }
        }

        for(int i=0; i<outputLayer; i++){
            for(int j=0; j<hiddenLayer; j++){
                w2[i][j] = rng.nextGaussian() * 0.01;
            }
        }
    }

     double[] h;

   public double[] forward(double[] x){
    h = new double[hiddenLayer];

    for(int i=0; i<hiddenLayer; i++){
        for(int j=0; j<inputLayer; j++){
            h[i] += w1[i][j] * x[j];
        }
        h[i]  += b1[i];
        h[i] = Math.max(0, h[i]); 
    }

    double[] out = new double[outputLayer];

    for(int i=0; i<outputLayer; i++){
        for(int j=0; j<hiddenLayer; j++){
            out[i] += w2[i][j] * h[j];
        }
        out[i] += b2[i]; 
    }

   return softmax(out);
   }

   private double[] softmax(double[] z){
    double max = z[0];
    for(double v:z){
        if(v>max) max = v;
    }
    double sum = 0;
    double[] exp = new double[z.length];

    for(int i=0; i<z.length; i++){
        exp[i] = Math.exp(z[i] - max);
        sum += exp[i];
    }

    for(int i=0; i<z.length; i++){
        exp[i] /= sum;
    }

    return exp;

   }

   public void backward(double[] x, double[] yPred, int target, String sentence) {

   
    double[] errorOut = new double[outputLayer];
    for (int i = 0; i < outputLayer; i++) {
        errorOut[i] = yPred[i] - (i == target ? 1 : 0);
    }
    for (int i = 0; i < outputLayer; i++) {
        for (int j = 0; j < hiddenLayer; j++) {
            w2[i][j] -= learningRate * errorOut[i] * h[j];
        }
        b2[i] -= learningRate * errorOut[i];
    }
    double[] errorHidden = new double[hiddenLayer];
    for (int i = 0; i < hiddenLayer; i++) {
        for (int j = 0; j < outputLayer; j++) {
            errorHidden[i] += w2[j][i] * errorOut[j];
        }
        if (h[i] <= 0) errorHidden[i] = 0;
    }
    double[] gradInput = new double[inputLayer];

    for (int j = 0; j < inputLayer; j++) {
        for (int i = 0; i < hiddenLayer; i++) {
            gradInput[j] += errorHidden[i] * w1[i][j];
        }
    }
    for (int i = 0; i < hiddenLayer; i++) {
        for (int j = 0; j < inputLayer; j++) {
            w1[i][j] -= learningRate * errorHidden[i] * x[j];
        }
        b1[i] -= learningRate * errorHidden[i];
    }
    String[] words = sentence.split(" ");
    int n = words.length;

    for (String w : words) {
        int idx = practice.vacob.get(w);

        for (int j = 0; j < inputLayer; j++) {
            practice.embedding[idx][j] -= learningRate * gradInput[j] / n;
        }
    }
}
    
    
}

public class practice {

    static Map<String , Integer> vacob = new HashMap<>();
    static List<String> nextWordId = new ArrayList<>(); 
    static int EMBED_DIM = 64;
    static double [][]embedding; 

   static double[] vectorize(String sentence){
    double[] vec = new double[EMBED_DIM];
    String[] word = sentence.toLowerCase().trim().split(" ");

    int count = 0;

    for(String w:word){

        if(!vacob.containsKey(w)) continue; 

        int idx = vacob.get(w);

        for(int i=0; i<EMBED_DIM; i++){
            vec[i] += embedding[idx][i];
        }

        count++;
    }

    if(count == 0) count = 1;

    for(int i=0; i<EMBED_DIM; i++){
        vec[i] /= count;
    }

    return vec;
}
 public static void saveModel(Model model) {
    try {
        ObjectOutputStream out = new ObjectOutputStream(
                new FileOutputStream("model1.bin"));

        out.writeObject(model);                
        out.writeObject(embedding);             
        out.writeObject(vacob);                 
        out.writeObject(nextWordId);            

        out.close();
        System.out.println(" Model saved");

    } catch (Exception e) {
        e.printStackTrace();
    }
}

    public static void main(String[] args) {
        

      List<String> data = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader("testModel.csv"));
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if(line.isEmpty()) continue;
                data.add(line);
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }


        int index = 0;
        for(String sentence : data){
     for(String w : sentence.split(" ")){
        if(!vacob.containsKey(w)){
            vacob.put(w, index++);
            nextWordId.add(w);
        }
    }
}

        Random rng = new Random();
        embedding = new double[vacob.size()][EMBED_DIM];

        for(int i=0; i<vacob.size(); i++){
            for(int j=0; j<EMBED_DIM; j++){
                embedding[i][j] = rng.nextGaussian() * 0.01;
            }
        }

        Model model = new Model(EMBED_DIM, 32, vacob.size());

        for(int epoch = 0; epoch<3000; epoch++){
            int step = 0;
            double loss = 0;
            
            
            for(String sentence : data){
                
                String[] words = sentence.split(" ");
                
                for(int i = 0; i < words.length - 1; i++){
                    
                   
                    StringBuilder inputBuilder = new StringBuilder();
                    for(int j = 0; j <= i; j++){
                        inputBuilder.append(words[j]).append(" ");
                    }
                    
                    String input = inputBuilder.toString().trim();
                    String targetWord = words[i + 1];
                    
                    double[] x = vectorize(input);
                    int target = vacob.get(targetWord);
                    
                    double[] yPred = model.forward(x);
                    
                    loss += -Math.log(yPred[target] + 1e-9);
                    step++;
                    
                    model.backward(x, yPred, target, input);
                }
            }
            loss /= step;

             if(epoch % 100 == 0){
                System.out.println("epoch: "+epoch+" loss: "+ loss);

                double[] test = vectorize("this is good");
                double[] pred = model.forward(test);

                

                int best = 0;
                for(int i=1;i<pred.length;i++){
                    if(pred[i] > pred[best]) best = i;
                }

                System.out.println("Test Prediction: " + nextWordId.get(best));
            }
        }

          saveModel(model);
        


         Scanner scan = new Scanner(System.in);

        while(true){
            String input = scan.nextLine();

            double[] test = vectorize(input);
            double[] pred = model.forward(test);

            int best = 0;
            for(int i=1;i<pred.length;i++){
                if(pred[i] > pred[best]) best = i;
            }

            System.out.println("Prediction: " + input + " " + nextWordId.get(best));
        }



    }
}
