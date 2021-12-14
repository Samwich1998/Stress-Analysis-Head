

// Global Variables
unsigned long totalTime;
unsigned long startTime;

// Setup Arduino; Runs Once
void setup() {
   Serial.begin(115200);   //Use 115200 baud rate for serial communication
   startTime = millis();   // Start the Timer
}

// Arduino Loop; Runs Until Arduino Closes
void loop() {

  // Read in Hardware-Filtered BioElectric Data
  int Channel1 = analogRead(A0);    // Read the voltage value of A0 port (EOG Channel1)
  int Channel2 = analogRead(A1);    // Read the voltage value of A1 port (EOG Channel2)
  // Record Time
  totalTime = millis() - startTime; // Calculate RunTime
  
  // Print EOG Data for Python to Read
  Serial.println(String(totalTime) + ',' + String(Channel1) + ',' + String(Channel2)); 
}
