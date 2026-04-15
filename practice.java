import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

 class NextWordModel {

  double[][] w;
  double[] b;
  double lenearRate = 0.01;

  public NextWordModel(int vocabSize ){
    w = new double[vocabSize][vocabSize];
    b = new double[vocabSize];

    Random rng = new Random();

    for(int i=0; i<vocabSize; i++){
      for(int j=0; j<vocabSize; j++){
        w[i][j] = rng.nextGaussian() * 0.01;
      }
    }
  }

  public double[] forward(double[] x){
    double[] y = new double[w.length];

    for(int i=0; i<w.length; i++){
      y[i] = b[i];
      for(int j=0; j<x.length; j++){
        y[i] += w[i][j] * x[j];
      }

    }
    return softmax(y);
  }

  private double[] softmax(double[] z){
    double sum = 0;
    double[] exp = new double[z.length];

    for(int i=0; i<z.length; i++){
      exp[i] = Math.exp(z[i]);
      sum += exp[i];
    }

    for(int i=0; i<z.length; i++){
      sum /= exp[i];
    }
    return exp;
  }

 
  public void backward(double[] x  , double[] yPred , int targetIndex){

    for(int i=0; i<w.length; i++){
      double error = yPred[i] - (i==targetIndex ?1 : 0);
      for(int j=0; j<x.length; j++){
        w[i][j] -= lenearRate * error * x[j]; 
      }

      b[i] -= lenearRate * error; 
    }
  }


  
}


public class practice {

 static Map<String , Integer> vacob = new HashMap<>();
 static List<String> nextWordId = new ArrayList<>(); 

 static double[] vectorize(String sentence){
  double[] vec = new double[vacob.size()];
  String[] word = sentence.split(" ");

  for(String w:word){
    if(vacob.containsKey(w)){
      vec[vacob.get(w)] = 1;
    }
  }
  return vec;
 }
   public static void main(String[] args) {



     String[][] data = {
            {"i love this", "car"},
            {"i hate this", "bike"},
            {"this is good", "ground"},
            {"this is bad", "thing"}
        };

        int index = 0;

        for(String[] words:data){

          for(String w:(words[0]+" "+words[1]).split(" ")){
            if(!vacob.containsKey(w)){
              vacob.put(w, index++);
              nextWordId.add(w);
            }
          }
        }

        NextWordModel model = new NextWordModel(vacob.size());

        for(int epoch = 0; epoch<1000; epoch++){
          double loss = 0;
          for(String[] pair:data){

            double[] x = vectorize(pair[0]);
            int target = vacob.get(pair[1]);

            double[] yPred = model.forward(x);

            loss += -Math.log(yPred[target] + 1e-9);

            model.backward(x, yPred, target);
          }

          if(epoch % 100 == 0){
            System.out.println("epoch: "+epoch+" loss: "+ loss);
          }
        }

        Scanner scan = new Scanner(System.in);
        String input = scan.nextLine();
        double[] test = vectorize(input);
        double[] pred = model.forward(test);
        
         int best = 0;
        for (int i = 1; i < pred.length; i++) {
            if (pred[i] > pred[best]) best = i;
        }

        System.out.println("Prediction: " + input +" "+nextWordId.get(best));
    
  }
}