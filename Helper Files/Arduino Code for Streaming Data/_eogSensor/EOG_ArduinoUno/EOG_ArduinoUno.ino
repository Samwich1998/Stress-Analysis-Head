
// Fast Analog Read Library
#include "avdweb_AnalogReadFast.h"

// ******************************** Initialize Variables ******************************** //

// Time Variables
const unsigned long oneSecMicro = pow(10,6);
int currentSecond;
unsigned long startTimerMicros;
unsigned long currentMicros;
// Analog Pins
const byte ADC0 = A0;
const byte ADC1 = A1;
const byte ADC2 = A2;
const byte ADC3 = A3;
// Streaming Variables
int galvanicSkinResponse;
int foreheadTemp;
int Channel2;
int Channel1;
// Buffer for Serial Printing
char buffer[33];

// ************************************************************************************** //
// ********************************* Relevant Functions ********************************* //

String padZeros(unsigned long number, int totalLength) {
    String finalNumber = String(number);
    int numZeros = totalLength - finalNumber.length();
    for (int i = 0; i < numZeros; i++) {
      finalNumber = "0" + finalNumber;
    }
    return finalNumber;
}

// ************************************************************************************** //
// *********************************** Arduino Setup ************************************ //

// Setup Arduino; Runs Once
void setup() {
    // Initialize Streaming
    Serial.begin(115200);     // Use 115200 baud rate for serial communication
    analogReadResolution(12); // Initialize ADC Resolution (Arduino Nano 33 IoT Max = 12)
    while (!Serial)           // Wait for Serial to Connect
    Serial.flush();
    
    // Start the Timer
    startTimerMicros = micros();
    currentSecond = 0;
}

// ************************************************************************************** //
// ************************************ Arduino Loop ************************************ //

// Arduino Loop; Runs Until Arduino Closes
void loop() {  
  
    // Read in Hardware-Filtered BioElectric Data
    Channel1 = analogReadFast(ADC0);             // Read the voltage value of A0 port (EOG Channel1)
    Channel2 = analogReadFast(ADC1);             // Read the voltage value of A1 port (EOG Channel2)
    foreheadTemp = analogReadFast(ADC2);         // Read the voltage value of A2 port (Temperature)
    galvanicSkinResponse = analogReadFast(ADC3); // Read the voltage value of A3 port (GSR)

    // Record the Time the Signals Were Collected
    currentMicros = micros() - startTimerMicros;
    // Keep Track of Seconds
    if (currentMicros >= oneSecMicro) {
        currentSecond += 1;

        // Reset Micros
        startTimerMicros += oneSecMicro; currentMicros -= oneSecMicro;
    }

    // Print EOG Data for Python to Read
    sprintf(buffer, "%i.%s,%i,%i,%i,%i", currentSecond, padZeros(currentMicros, 6).c_str(), Channel1, Channel2, foreheadTemp, galvanicSkinResponse);
    Serial.println(buffer);
    Serial.flush();
}

// ************************************************************************************** //
