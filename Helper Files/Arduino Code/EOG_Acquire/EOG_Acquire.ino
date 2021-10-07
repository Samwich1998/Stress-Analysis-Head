void setup() {
   Serial.begin(115200);         //Use 115200 baud rate for serial communication
}

void loop() {

  // Read in Hardware-Filtered EMG Data
  int Channel1 = analogRead(A0);    //Read the voltage value of A0 port (Channel1)
  int Channel2 = analogRead(A1);    //Read the voltage value of A1 port (Channel2)

  // Print EMG Data
  Serial.print(Channel1);           //Output Channel1 data
  Serial.println(Channel2);           //Output Channel2 data
}
