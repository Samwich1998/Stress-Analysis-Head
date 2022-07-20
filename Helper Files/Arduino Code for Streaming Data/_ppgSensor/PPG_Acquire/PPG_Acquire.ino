
// ********************************** Import Libraries ********************************** //

// Fast Analog Read Library
#include "avdweb_AnalogReadFast.h"
// PPG Libraries
#include "max30102.h"

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

// PPG Variables
uint16_t aun_ir_buffer[1];   //infrared LED sensor data
uint16_t aun_red_buffer[1];  //red LED sensor data
uint8_t uch_dummy;

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
    
    //pinMode(10, INPUT);
    // Set up PPG Board/Sensor
    maxim_max30102_reset(); //resets the MAX30102
    maxim_max30102_read_reg(REG_INTR_STATUS_1,&uch_dummy);  //Reads/clears the interrupt status register
    maxim_max30102_init();  //initialize the MAX30102

    // Start the Timer
    startTimer();
}

// ************************************************************************************** //
// ************************************ Arduino Loop ************************************ //

// Arduino Loop; Runs Until Arduino Closes
void loop() {  

    // Read in PPG Data
    while(digitalRead(10)==1);
    maxim_max30102_read_fifo(aun_red_buffer, aun_ir_buffer);

    // Record the Time the Signals Were Collected
    currentMicros = micros() - startTimerMicros;
    // Keep Track of Seconds
    if (currentMicros >= oneSecMicro) {
        currentSecond += 1;

        // Reset Micros
        startTimerMicros += oneSecMicro; currentMicros -= oneSecMicro;
    }

    // Print Data for Python to Read
    sprintf(buffer, "%s.%s,%s,%s", String(currentSecond).c_str(), padZeros(currentMicros, 6).c_str(), String(aun_red_buffer[0]).c_str(), String(aun_ir_buffer[0]).c_str());
    Serial.println(buffer);
    Serial.flush();
}

// ************************************************************************************** //
