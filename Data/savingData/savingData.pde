
import processing.serial.*;
import grafica.*;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.BufferedWriter;
import java.io.FileWriter;


Serial myPort;  // Create object from Serial class

//Visluzation Related Variables
int locationPlotIMU_X = 100;// The position of the plot in the window
int locationPlotIMU_Y = 100;
int widthPlotIMU = 800;// size of the plot 
int heigthPlotIMU = 600;
int absRangeofY = 15;// The value range of the Y Axis in the plot
int numAxis = 6; 
int winSize = 100;// How many data points are saved in the data array and draw on the screen
int counter = 0;


//define thresholds
float upDownThreshold = 3;
float leftRightThreshold = 70;
float rotation_Threshold = 70;

GPlot plotIMU[] = new GPlot[numAxis];
long plotIMUIndex[] = new long[numAxis]; // Save the index for GPoints in GPlot 
String myString = null;
long lastTimeTriggered = 0;

//Weka ML related Variables
static ArffSaver saver;
static public FastVector atts; // name and type of features
public static FastVector attsResult; //put classlabels here
BufferedWriter trainingfileWriter = null;
static Instances mInstances;  // Save the training instances
String[] classLabels= {
"straight",
"forward",
"straight_to_forward",
"forward_to_straight"
}; // The names of the class 
int numfeatures = 24;
double[] featurelist = new double[numfeatures+1];// The last one is for the lables
int numofTrainingSamples = 40;
int samplecounter = 0;
String savingpath = "/Users/valentin/text-neck-recognition/Data/";


//Save the data of the current window in multiple axis
ArrayList<ArrayList> IMUDataArray = new ArrayList();
ArrayList<ArrayList> IMUDataFilter = new ArrayList();


void setup() {
  size(1200, 900);
  background(255);
  printArray(Serial.list());
  String portName = Serial.list()[3];
  myPort = new Serial(this, portName, 115200);// 

  // Initialize Plot Setting 
  plotInitialization();

  for (int i=0; i<numAxis; ++i) {
    IMUDataArray.add(new ArrayList());
    IMUDataFilter.add(new ArrayList());
  }

  setupARFF(savingpath, classLabels);
}

void draw() {
  //background(red, green, blue);
  updataSerial();
  //draw_plot();
}

void updataSerial() {
  while (myPort.available() > 0) {
    myString = myPort.readStringUntil(10);    // '\n'(ASCII=10) every number end flag
    //print(myString);
    if (myString!=null) {
      analysisData(myString);
    }
  }
}


void analysisData(String myString) {
  String[] list = split(myString.substring(0, myString.length()-2), ',');
  if (list.length == 6) {
    float[] imuValue = new float[numAxis]; // imuValue 0-6 : acclx, y, z, gyro x, y, z;
    for (int i = 0; i<numAxis; i++) {
      imuValue[i] = Float.parseFloat(list[i]);
    }

    // If the size is more than the windowsize, remove a data before adding new values in
    while (IMUDataArray.get(0).size() >= winSize ) {
      for (int i= 0; i < numAxis; ++i) {
        // Maintain the lenght of the array, If the size of array is larger than winSize, remove the oldest data.
        IMUDataArray.get(i).remove(0);
        plotIMU[i].removePoint(0);
      }
    }

    // Add data into dataArray and PlotPoints at the same time
    for (int i= 0; i < numAxis; ++i) {
      //System.out.println(IMUDataArray.get(0).size());
      plotIMU[i].addPoint(new GPoint(plotIMUIndex[i]++, imuValue[i]));
      IMUDataArray.get(i).add(imuValue[i]);
      IMUDataFilter.get(i).add(imuValue[i]);
    }

    counter++;

    if (IMUDataFilter.get(0).size() > 10) {
      for (int i= 0; i < numAxis; ++i) {
        IMUDataFilter.get(i).remove(0);
      }
    }

    // filters
    for (int i= 0; i < numAxis; ++i) {
      setAvgMovingFilter10(IMUDataFilter.get(i), IMUDataArray.get(i));
    }

    counter = counter % 51;


    // Write your data processing algorithm here , miminum 1000 ms between two gestures
    long currenttime = millis();
    // process the windows with 50% overlap
    if (counter == 25) {

      //If the gyroscope x [3] is larger than 2, then save the feature vector generated from the curent window
      if ( (getABSMax(IMUDataArray.get(0))> rotation_Threshold || getABSMax(IMUDataArray.get(1))> leftRightThreshold || getABSMax(IMUDataArray.get(2))> upDownThreshold)  && (currenttime-lastTimeTriggered) >2000) {
        
        System.out.println("3" + IMUDataArray.get(3));
        System.out.println("2" + IMUDataArray.get(2));
        System.out.println("1" + IMUDataArray.get(1));
        
        lastTimeTriggered = currenttime;
        featurelist = new double[numfeatures+1];
        featurelist[0] = getMean(IMUDataArray.get(3)); 
        featurelist[1] = getMax(IMUDataArray.get(3)); 
        featurelist[2] = getMin(IMUDataArray.get(3));
        featurelist[3] = getRMS(IMUDataArray.get(3));
        
        featurelist[4] = getMean(IMUDataArray.get(1)); // left right
        featurelist[5] = getMax(IMUDataArray.get(1));
        featurelist[6] = getMin(IMUDataArray.get(1));
        featurelist[7] = getRMS(IMUDataArray.get(1));
        
        featurelist[8] = getMean(IMUDataArray.get(2)); // up down
        featurelist[9] = getMax(IMUDataArray.get(2));
        featurelist[10] = getMin(IMUDataArray.get(2));
        featurelist[11] = getRMS(IMUDataArray.get(2));
        
        featurelist[12] = getMean(IMUDataArray.get(0));
        featurelist[13] = getMax(IMUDataArray.get(0));
        featurelist[14] = getMin(IMUDataArray.get(0));
        featurelist[15] = getRMS(IMUDataArray.get(0));
        
        featurelist[16] = getMean(IMUDataArray.get(4));
        featurelist[17] = getMax(IMUDataArray.get(4));
        featurelist[18] = getMin(IMUDataArray.get(4));
        featurelist[19] = getRMS(IMUDataArray.get(4));
        
        featurelist[20] = getMean(IMUDataArray.get(5));
        featurelist[21] = getMax(IMUDataArray.get(5));
        featurelist[22] = getMin(IMUDataArray.get(5));
        featurelist[23] = getRMS(IMUDataArray.get(5));
        
        
        samplecounter++;
        if (samplecounter<= 20) {
          featurelist[24] = (double) attsResult.indexOf("straight"); 
          System.out.println("Sample Counter:" + samplecounter +"  straight");
          plotIMU[0].setTitleText("IMU Data straight: " + samplecounter);
        } else if (samplecounter<= 40){
          featurelist[24] = (double) attsResult.indexOf("forward"); 
          System.out.println("Sample Counter:" + samplecounter +"  forward");
          plotIMU[0].setTitleText("IMU Data forward: " + samplecounter);
        }
        //else if (samplecounter<= 60){
        //  featurelist[24] = (double) attsResult.indexOf("straight_to_forward"); 
        //  System.out.println("Sample Counter:" + samplecounter +"  straight_to_forward");
        //  plotIMU[0].setTitleText("IMU Data Left: " + samplecounter);
        //} else if (samplecounter<= 80){
        //  featurelist[24] = (double) attsResult.indexOf("forward_to_straight"); 
        //  System.out.println("Sample Counter:" + samplecounter +"  forward_to_straight");
        //  plotIMU[0].setTitleText("IMU Data forward_to_straight: " + samplecounter);
        //}
        }

        //System.out.println(featurelist[0] +","+featurelist[1] +","+featurelist[2] +","+featurelist[3] +","+featurelist[4] +","+featurelist[5] +","+featurelist[6] +","+featurelist[7] +","+featurelist[8]);
        DenseInstance addinstance = new DenseInstance(1.0, featurelist);
        mInstances.add(addinstance);

        try {
          // If we have collected enough samples. output the features into the file and exit the program
          if (samplecounter == numofTrainingSamples) {
            //System.out.println(mInstances.toString()); // Output the feature vectors in console
            trainingfileWriter.write(mInstances.toString());
            trainingfileWriter.flush();
            trainingfileWriter.close();
            exit();
          }
        } 
        catch (IOException e) {
          // TODO Auto-generated catch block
          e.printStackTrace();
        }
      }
    }
  }

// Initialization data display
void plotInitialization() {
  // initialization of plotIMU 
  for (int i= 0; i < numAxis; ++i) {
    plotIMU[i]= new GPlot(this);
    plotIMU[i].setPos(locationPlotIMU_X, locationPlotIMU_Y);
    plotIMU[i].setDim(widthPlotIMU, heigthPlotIMU);
  }
  plotIMU[0].setTitleText("IMU Data");
  plotIMU[0].getXAxis().setAxisLabelText("Time (t)");
  plotIMU[0].getYAxis().setAxisLabelText("y axis");
}

void draw_plot() {
  // cover the interface
  background(color(150));
  // Display plotIMU 
  // add points
  ArrayList<ArrayList> copy = new ArrayList();
  copy.addAll(IMUDataArray);


  for (int i= 0; i < numAxis; ++i) {
    for (int j=0; j < copy.get(i).size(); j++) {
      plotIMU[i].addPoint(new GPoint(plotIMUIndex[i]++, (Float)copy.get(i).get(j)));
      //System.out.println("size2:" + IMUDataArray.get(0).size());
    }
  }
  copy.clear();


  // plot background and axis 
  plotIMU[0].beginDraw();
  plotIMU[0].drawBackground();
  plotIMU[0].drawXAxis();
  plotIMU[0].drawYAxis();
  plotIMU[0].drawTitle();
  plotIMU[0].endDraw();

  // plot lines
  for (int i= 0; i < numAxis; ++i) {   
    plotIMU[i].beginDraw();
    plotIMU[i].setXLim(plotIMUIndex[i]-winSize, plotIMUIndex[i]);
    plotIMU[i].setYLim(-absRangeofY, absRangeofY);
    plotIMU[i].drawLines();
    plotIMU[i].endDraw();
  }
}

/**
 * Set up Arff files for later retrieving data out from here
 * 
 * @param folder
 */
private void setupARFF(String folder, String[] mylabels) {
  atts = new FastVector(); // Save the feature namse
  attsResult = new FastVector(); // Save the label names

  //Set up the folder , in case the folder dose not exist
  File writeFolder = new File(folder);
  if (!writeFolder.exists()) {
    writeFolder.mkdirs();
  }

  for (int i=0; i<mylabels.length; ++i) {
    attsResult.addElement(mylabels[i]);
  }

  atts.add(new Attribute("Mean_GyroX"));
  atts.add(new Attribute("Max_GyroX"));
  atts.add(new Attribute("Min_Gyrox"));
  atts.add(new Attribute("RMS_Gyrox"));
  
  atts.add(new Attribute("Mean_AcclY"));
  atts.add(new Attribute("Max_AcclY"));
  atts.add(new Attribute("Min_AcclY"));
  atts.add(new Attribute("RMS_AcclY"));
  
  atts.add(new Attribute("Mean_AcclZ"));
  atts.add(new Attribute("Max_AcclZ"));
  atts.add(new Attribute("Min_AcclZ"));
  atts.add(new Attribute("RMS_AcclZ"));
  
  atts.add(new Attribute("Mean_AcclX")); 
  atts.add(new Attribute("Max_AcclX"));
  atts.add(new Attribute("Min_AcclX"));
  atts.add(new Attribute("RMS_AcclX"));
  
  atts.add(new Attribute("Mean_GyroY"));
  atts.add(new Attribute("Max_GyroY"));
  atts.add(new Attribute("Min_GyroY"));
  atts.add(new Attribute("RMS_GyroY"));
  
  atts.add(new Attribute("Mean_GyroZ"));
  atts.add(new Attribute("Max_GyroZ"));
  atts.add(new Attribute("Min_GyroZ"));
  atts.add(new Attribute("RMS_GyroZ"));
  
  atts.add(new Attribute("result", attsResult));
  mInstances = new Instances("Gestures", atts, 0);

  try {
    //Set up the training file
    trainingfileWriter = new BufferedWriter(new FileWriter(folder+getCurrentTime()+"_Training.arff"));
  } 
  catch (IOException e) {
    // TODO Auto-generated catch block
    e.printStackTrace();
  }
}


float getABSMax(ArrayList data) {
  float max = Math.abs((float)data.get(0));
  for (int i=1; i<data.size(); ++i) {
    if (max< Math.abs((float)data.get(i))) max = Math.abs((float)data.get(i));
  }
  return max;
}

float getMax(ArrayList data) {
  float max = (float)data.get(0);
  for (int i=1; i<data.size(); ++i) {
    if (max< (float)data.get(i)) max = (float)data.get(i);
  }
  return max;
}

float getMin(ArrayList data) {
  float min = (float)data.get(0);
  for (int i=1; i<data.size(); ++i) {
    if (min > (float)data.get(i)) min = (float)data.get(i);
  }
  return min;
}

float getMean(ArrayList data) {
  float total = (float)data.get(0);
  for (int i=1; i<data.size(); ++i) {
    total += (float) data.get(i);
  }
  float mean = (float) total/((float)(data.size()));
  return mean;
}

float getRMS(ArrayList data) {
  float total = (float)data.get(0) * (float)data.get(0);
  for (int i=1; i<data.size(); ++i) {
    total += (float) data.get(i) * (float) data.get(i);
  }
  float mean = (float) total/((float)(data.size()));
  mean = (float) Math.sqrt(mean);
  return mean;
}

// Output the current date in String
String getCurrentTime() {

  //add year month day to the file name
  String fname= "";
  fname = fname + year() + "_";
  if (month() < 10) fname=fname+"0";
  fname = fname + month() + "_";
  if (day() < 10) fname = fname + "0";
  fname = fname + day();
  //add hour minute sec to the file name
  fname = fname + "_";
  if (hour() < 10) fname = fname + "0";
  fname = fname + hour() + "_";
  if (minute() < 10) fname = fname + "0";
  fname = fname + minute() + "_";
  if (second() < 10) fname = fname + "0";
  fname = fname + second();

  return fname;
}

void setAvgMovingFilter10(ArrayList<Float> last10values, ArrayList<Float> dataList) {
  float sum = 0;     
  for (int i = 0; i < last10values.size(); i++) {
    sum += last10values.get(i);
  }
  float mean = sum / last10values.size();
  //System.out.println(last10values.size());
  dataList.set(dataList.size() - 1, mean);
  return;
}
