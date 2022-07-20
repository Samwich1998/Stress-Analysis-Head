
// ********************************** Import Libraries ********************************** //

// Wireless (BLE + Internet) Libraries
#include <ArduinoMqttClient.h>
#include <ArduinoBLE.h>
#include <WiFiNINA.h> 
#include <WiFiUdp.h>
#include <SPI.h>
// Libraries for the Clock
#include <RTCZero.h>
// Fast Analog Read Library
#include "avdweb_AnalogReadFast.h"

// ******************************** Initialize Variables ******************************** //

// Time Variables
const unsigned long oneSecMicro = pow(10,6);
unsigned long startTimerMicros;
unsigned long currentMicros;
unsigned long currentSecond;
int currentMinute;
int currentHour;
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

// WiFi Credentials (edit as required)
char ssid[] = "Caltech Visitor";     // Wifi SSID
char pass[] = "";    // Wifi password
int keyIndex = 0;                   // Network key Index number (needed only for WEP)
// Initialize the Wifi client
WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);

// Object for Real Time Clock
RTCZero rtc;
int status = WL_IDLE_STATUS;
// Time zone constant - change as required for your location
const int GMT = -5; // Los Angeles = -8; Maryland = -5 

// ************************************************************************************** //
// ********************************** Helper Functions ********************************** //

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

String padZeros(unsigned long number, int totalLength) {
    String finalNumber = String(number);
    int numZeros = totalLength - finalNumber.length();
    for (int i = 0; i < numZeros; i++) {
      finalNumber = "0" + finalNumber;
    }
    return finalNumber;
}

// ************************************************************************************** //
// *************************** Connect to Clock/WiFi Functions ************************** //

void connectToWiFi() {
  // Check if the WiFi module works
  if (WiFi.status() == WL_NO_SHIELD) {
    // Wait until WiFi ready
    Serial.println("WiFi adapter not ready");
    while (true);
  }

  // Check Firmware
  String fv = WiFi.firmwareVersion();
  if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
    Serial.println("Please upgrade the firmware");
  }
    
  // Establish a WiFi connection
  while (status != WL_CONNECTED) {
 
    Serial.println("Attempting to connect to SSID: " + String(ssid));
    status = WiFi.begin(ssid, pass);
 
    // Wait 10 seconds for connection:
    delay(10000);
  }

  // Print connection status
  printWiFiStatus();
}

void connectToMQTT() {

  const char broker[]    = "test.mosquitto.org";
  int        port        = 1883;
  const char willTopic[] = "arduino/will";
  String topic = "";

  if (!mqttClient.connect(broker, port)) {
    Serial.print("MQTT connection failed! Error code = ");
    Serial.println(mqttClient.connectError());
    while (1);
  }

  Serial.println("You're connected to the MQTT broker!");
  Serial.println();

  // set the message receive callback
  //mqttClient.onMessage(onMqttMessage);

  Serial.print("Subscribing to topic: ");
  Serial.println(topic);
  Serial.println();

  // subscribe to a topic
  mqttClient.subscribe(topic);
   
  Serial.print("Topic: ");
  Serial.println(topic);  
}

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

// ************************************************************************************** //
// *********************************** Arduino Setup ************************************ //

// Setup Arduino; Runs Once
void setup() {
    // Initialize Streaming
    Serial.begin(115200);     // Use 115200 baud rate for serial communication
    analogReadResolution(12); // Initialize ADC Resolution (Arduino Nano 33 IoT Max = 12)
    while (!Serial)           // Wait for Serial to Connect
    Serial.flush();
    
    // Connect to WiFi and Clock
    //connectToWiFi();
    //connectToMQTT();
    //connectToClock();

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
    //currentSecond = currentSecond + currentMinute*60 + currentHour*60*60;
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
