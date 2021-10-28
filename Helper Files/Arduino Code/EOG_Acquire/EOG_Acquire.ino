// Global Variables
unsigned long totalTime;
unsigned long startTime;
// Channel Variables
int Channel1;
int Channel2;

void setup() {
   Serial.begin(115200);         //Use 115200 baud rate for serial communication
   unsigned long startTime = millis();
}

void loop() {

  // Read in Hardware-Filtered BioElectric Data
  Channel1 = analogRead(A0);    //Read the voltage value of A0 port (Channel1)
  Channel2 = analogRead(A1);    //Read the voltage value of A1 port (Channel2)
  // Get Current Time
  totalTime = millis() - startTime;
  
  // Print Out the Information
  Serial.println(String(totalTime) + ',' + String(Channel1) + ',' + String(Channel2));
}
