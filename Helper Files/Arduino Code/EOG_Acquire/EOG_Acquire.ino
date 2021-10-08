// Global Variables
unsigned long totalTime;
unsigned long startTime;

void setup() {
   Serial.begin(115200);         //Use 115200 baud rate for serial communication
   unsigned long startTime = millis();
}

void loop() {

  // Read in Hardware-Filtered BioElectric Data
  int Channel1 = analogRead(A0);    //Read the voltage value of A0 port (Channel1)
  int Channel2 = analogRead(A1);    //Read the voltage value of A1 port (Channel2)
  // Record Time
  totalTime = millis() - startTime;
  
  // Print EMG Data
  Serial.println(String(totalTime) + ',' + String(Channel1) + ',' + String(Channel2));   //Total Time
}
