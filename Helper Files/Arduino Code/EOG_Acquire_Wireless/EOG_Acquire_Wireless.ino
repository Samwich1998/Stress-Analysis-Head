
// Include required libraries
#include <SPI.h>
#include <WiFiNINA.h> 
#include <WiFiUdp.h>
#include <RTCZero.h>
#include "avdweb_AnalogReadFast.h"

// ******************************** Initialize Variables ******************************** //

// Global Variables
unsigned long totalTime;
unsigned long startTime;
const byte ADC0 = A0;
const byte ADC1 = A1;

// WiFi Credentials (edit as required)
char ssid[] = "87 WiFi";            // Wifi SSID
char pass[] = "noahisthebomb.com";  // Wifi password
int keyIndex = 0;                   // Network key Index number (needed only for WEP)

// Object for Real Time Clock
RTCZero rtc;
int status = WL_IDLE_STATUS;
// Time zone constant - change as required for your location
const int GMT = -8; // Los Angeles = -8; Maryland = -5 

// ************************************************************************************** //
// ********************************** Print Functions *********************************** //

void printTime() {
  // Print Hour
  print2digits(rtc.getHours() + GMT); Serial.print(":");
  // Print Minutes
  print2digits(rtc.getMinutes()); Serial.print(":");
  // Print Seconds
  print2digits(rtc.getSeconds()); Serial.println();
}

void printWiFiStatus() {
  // Print the network SSID
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());
  
  // Print the IP address
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);
  
  // Print the received signal strength
  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
}
 
void print2digits(int number) {
  if (number < 0) {number += 24;}
  else if (number < 10) {Serial.print("0");}
  Serial.print(number);
}

// ************************************************************************************** //
// ******************************** Important Functions ********************************* //

void connectToWiFi() {
  // Check if the WiFi module works
  if (WiFi.status() == WL_NO_SHIELD) {
    // Wait until WiFi ready
    Serial.println("WiFi adapter not ready");
    while (true);
  }
    
  // Establish a WiFi connection
  while ( status != WL_CONNECTED) {
 
    Serial.println("Attempting to connect to SSID: " + String(ssid));
    status = WiFi.begin(ssid, pass);
 
    // Wait 10 seconds for connection:
    delay(10000);
  }

  // Print connection status
  printWiFiStatus();
}

void connectToClock() {
  // Start Real Time Clock
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

// ************************************************************************************** //
// *********************************** Arduino Setup ************************************ //

// Setup Arduino; Runs Once
void setup() {
    // Initialize Streaming
    Serial.begin(115200);     // Use 115200 baud rate for serial communication
    analogReadResolution(12); // Initialize ADC Resolution (Arduino Nano 33 IoT Max = 12)
    
    // Connect to WiFi and Clock
    connectToWiFi();
    connectToClock();

    // Start the Timer
    startTime = micros();   // Start the Timer
}

// ************************************************************************************** //
// ************************************ Arduino Loop ************************************ //

// Arduino Loop; Runs Until Arduino Closes
void loop() {

  // Read in Hardware-Filtered BioElectric Data
  int Channel1 = analogReadFast(ADC0);     // Read the voltage value of A0 port (EOG Channel1)
  int Channel2 = analogReadFast(ADC1);    // Read the voltage value of A1 port (EOG Channel2)
  // Record Time
  totalTime = micros() - startTime; // Calculate RunTime
  
  // Print EOG Data for Python to Read
  Serial.println(String(totalTime) + ',' + String(Channel1) + ',' + String(Channel2)); 
}

// ************************************************************************************** //
