
// ********************************** Import Libraries ********************************** //

// Fast Analog Read Library
#include "avdweb_AnalogReadFast.h"
// PPG Libraries
#include <SparkFun_Bio_Sensor_Hub_Library.h>
#include <Wire.h>

#ifdef ARDUINO_AVR_NANO_EVERY
    // Libraries for the Clock
    #include <RTCZero.h>
    // Libraries for WiFi
    #include <WiFiNINA.h> 
    #include <WiFiUdp.h>
#endif

// ******************************** Initialize Variables ******************************** //

// Time Variables
const unsigned long oneSecMicro = pow(10,6);
unsigned long startTimerMicros;
unsigned long currentMicros;
unsigned long currentSecond;
int currentMinute;
int currentHour;

// PPG Variables.
int redValue;
int irValue;
// Set Sample Width and Sampling Frequency for PPG. Note: There is a Dependance Between Them
int width = 215;    // Possible widths: 69, 118, 215, 411us
int samples = 200;  // Possible samples: 50, 100, 200, 400, 800, 1000, 1600, 3200 samples/second
int takeEvery_PPG = 4;
int generalCounter = takeEvery_PPG;
// Set the Reset and MFIO pin for PPG
int resPin = 4;
int mfioPin = 5;
// Takes address, reset pin, and MFIO pin for PPG.
SparkFun_Bio_Sensor_Hub bioHub(resPin, mfioPin); 
bioData body; 

// Buffer for Serial Printing
char buffer[40];

// Object for Real Time Clock
#ifdef ARDUINO_AVR_NANO_EVERY
    RTCZero rtc;
    // Time zone constant - change as required for your location
    const int GMT = -8; // West Coast = -8; East Coast = -5 
#endif

// ************************************************************************************** //
// ********************************** Helper Functions ********************************** //

String padZeros(unsigned long number, int totalLength) {
    String finalNumber = String(number);
    int numZeros = totalLength - finalNumber.length();
    for (int i = 0; i < numZeros; i++) {
      finalNumber = "0" + finalNumber;
    }
    return finalNumber;
}

// ************************************************************************************** //
// ********************************** Connect to Clock ********************************** //

#ifdef ARDUINO_AVR_NANO_EVERY
    void connectToClock() {
      // Start Real Time Clock
      RTCZero rtc;
      rtc.begin();
      
      // Variable to represent epoch
      unsigned long epoch;
     
      // Variable for number of tries to NTP service
      int numberOfTries = 0, maxTries = 6;
     
      // Get epoch
      do {
        epoch = WiFi.getTime();
        numberOfTries++;
      }
     
      while ((epoch == 0) && (numberOfTries < maxTries));
     
        if (numberOfTries == maxTries) {
        Serial.print("NTP unreachable!!");
        while (1);
        }
     
        else {
        Serial.print("Epoch received: ");
        Serial.println(epoch);
        rtc.setEpoch(epoch);
        Serial.println();
        }
    }
#endif

void startTimer() {
    // If Using Nano, Calculate the Current Time
    #ifdef ARDUINO_AVR_NANO_EVERY
        // Connect to Clock
        connectToClock();
        // Align the MicroSecond Counter with Seconds (as Best as You Can)
        currentSecond = rtc.getSeconds();
        startTimerMicros = micros();
        while (rtc.getSeconds() != currentSecond && currentSecond > 40) {
            currentSecond = rtc.getSeconds();
            startTimerMicros = micros();
        }
        // Initiate the Full Time
        currentHour = rtc.getHours() + GMT;
        if (currentHour < 0) {currentHour += 24;}
        currentMinute = rtc.getMinutes();

        // Merge Seconds, Hours, and Minutes
        currentSecond = currentSecond + currentMinute*60 + currentHour*60*60;
    #else
        // Start the Micros Timer
        startTimerMicros = micros();
        currentSecond = 0;
    #endif
}

// ************************************************************************************** //
// *********************************** Arduino Setup ************************************ //

// Setup Arduino; Runs Once
void setup() {
    // Initialize Streaming
    Serial.begin(115200);     // Use 115200 baud rate for serial communication
    
    // Initialize ADC Resolution
    #ifdef ARDUINO_AVR_NANO_EVERY
      analogReadResolution(12); // Arduino Nano 33 IoT Max = 12
    #endif
    
    // Wait for Serial to Connect and Flush Port
    while (!Serial)           
    Serial.flush();
    
    // Set up PPG Board/Sensor
    Wire.begin();
    bioHub.begin();
    //bioHub.configSensorBpm(MODE_ONE);
    bioHub.configSensor();
    bioHub.setPulseWidth(width);
    bioHub.setSampleRate(samples);
    delay(4000);

    // Start the Timer
    startTimer();
}

// ************************************************************************************** //
// ************************************ Arduino Loop ************************************ //

// Arduino Loop; Runs Until Arduino Closes
void loop() {  
    // Only Take Every Other PPG Value
    if (generalCounter == takeEvery_PPG) {
        generalCounter = 0;
        
        // Information from the readSensor function will be saved to our "body" variable.
        body = bioHub.readSensor();
        // Read in PPG Data
        redValue = body.redLed;
        irValue = body.irLed;
    } else {
      redValue = 0;
      irValue = 0;
      generalCounter += 1;
    }
    
    // Record the Time the Signals Were Collected
    currentMicros = micros() - startTimerMicros;
    // Keep Track of Seconds
    if (currentMicros >= oneSecMicro) {
        currentSecond += 1;

        // Reset Micros
        startTimerMicros += oneSecMicro; currentMicros -= oneSecMicro;
    }

    // Print Data for Python to Read
    sprintf(buffer, "%s.%s,%i,%i", String(currentSecond).c_str(), padZeros(currentMicros, 6).c_str(), redValue, irValue);
    Serial.println(buffer);
    Serial.flush();
}

// ************************************************************************************** //
