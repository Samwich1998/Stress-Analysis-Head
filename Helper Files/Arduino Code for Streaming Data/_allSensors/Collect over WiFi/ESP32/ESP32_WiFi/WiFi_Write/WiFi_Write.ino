// ********************************** Import Libraries ********************************** //

// WiFi Libraries
#include <AsyncUDP.h>
#include <WiFi.h>
#include <SPI.h>
// Secret WiFi Passwords
//#include "./arduino_secrets.h"
// Fast Analog Read Library
//#include "avdweb_AnalogReadFast.h"


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
int galvanicSkinResponse = 1234;
int foreheadTemp = 1234;
int Channel2 = 1234;
int Channel1 = 1234;
// Buffer for Serial Printing
const int maxLengthSending = 45;
char buffer[maxLengthSending];
char bufferSmall[25];

// WiFi Credentials (edit in arduino_secrets.h)
//char ssid[] = SECRET_SSID;    // your network SSID (name)
//char password[] = SECRET_PASS;    // your network password (use for WPA, or use as key for WEP)
const char * ssid = "87 Marion WiFi";
const char * password = "coldbrew";
// Specify WiFi Topic 
AsyncUDP udp;
const int port = 1234;

// Send Multiple Values Together
int publishInterval = 0;
const int publishingEvery = 1;
// Create Structure to Hold a List of Sensor Data
char sendingBuffer[publishingEvery*maxLengthSending];

// ************************************************************************************** //
// ********************************** Helper Functions ********************************** //

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
  
  Serial.print("MAC Address:");
  Serial.println(WiFi.macAddress());
}

String padZeros(unsigned long number, int totalLength) {
    String finalNumber = String(number);
    int numZeros = totalLength - finalNumber.length();
    for (int i = 0; i < numZeros; i++) {
      finalNumber = "0" + finalNumber;
    }
    return finalNumber;
}

void setupDataTransmission() {
    if(udp.connect(IPAddress(192,168,1,100), port)) {
        Serial.println("UDP connected");
        // Callback Protocol Detailing What to Do with Incoming Data
        udp.onPacket([](AsyncUDPPacket packet) {
            Serial.write(packet.data(), packet.length());
        });
        //Send unicast
        udp.print("Hello Server!");
    }
}


// ************************************************************************************** //
// ********************************* Connection Functions ******************************* //

void connectToWiFi() {
    // Establish a WiFi connection
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    if (WiFi.waitForConnectResult() != WL_CONNECTED) {
        Serial.println("WiFi Failed");
        while(1) {
            delay(1000);
        }
    }

    // Print connection status
    printWiFiStatus();
}

// ************************************************************************************** //
// *********************************** Arduino Setup ************************************ //

// Setup Arduino; Runs Once
void setup() {
  //Initialize serial and wait for port to open:
    Serial.begin(115200);     // Use 115200 baud rate for serial communication
    //analogReadResolution(12); // Initialize ADC Resolution (Arduino Nano 33 IoT Max = 12)
    //Serial.flush();           // Flush anything left in the Serial port
    
    // Connect to WiFi and Clock
    connectToWiFi();
    //connectToClock();
    setupDataTransmission();

    currentSecond = 0;
    startTimerMicros = micros();
}

// ************************************************************************************** //
// ************************************ Arduino Loop ************************************ //

void loop() {
    // Read in Hardware-Filtered BioElectric Data
    /**
    Channel1 = analogRead(ADC0);             // Read the voltage value of A0 port (EOG Channel1)
    Channel2 = analogRead(ADC1);             // Read the voltage value of A1 port (EOG Channel2)
    foreheadTemp = analogRead(ADC2);         // Read the voltage value of A2 port (Temperature)
    galvanicSkinResponse = analogRead(ADC3); // Read the voltage value of A3 port (GSR)
    **/
    // Record the Time the Signals Were Collected
    currentMicros = micros() - startTimerMicros;
    // Keep Track of Seconds
    if (currentMicros >= oneSecMicro) {
        currentSecond += 1;
  
        // Reset Micros
        startTimerMicros += oneSecMicro; currentMicros -= oneSecMicro;
    }

    /**
    // Compile the Data Collected and Add onto Growing String
    if (sendingBuffer[0] == 0) {
        sprintf(sendingBuffer, "%i.%s,%i,%i,%i,%i,%i,%i\n", currentSecond, padZeros(currentMicros, 6).c_str(), Channel1, Channel2, foreheadTemp, galvanicSkinResponse, galvanicSkinResponse, galvanicSkinResponse);
  
    } else {
        sprintf(sendingBuffer, "%s%i.%s,%i,%i,%i,%i,%i,%i\n", sendingBuffer, currentSecond, padZeros(currentMicros, 6).c_str(), Channel1, Channel2, foreheadTemp, galvanicSkinResponse, galvanicSkinResponse, galvanicSkinResponse);
    }
    
    publishInterval += 1;
    // After Collecting Some Data, Publish the Collected Data
    if (publishInterval == publishingEvery) {
        udp.broadcastTo(sendingBuffer, port);
        memset(&sendingBuffer[0], 0, sizeof(sendingBuffer));
            
        // Reset the Counter
        publishInterval = 0;
    }

    delay(3);
    **/

  
    
    // Compile Sensor Data to Send
    sprintf(buffer, "%i.%s,%i,%i,%i,%i,%i,%i\n", currentSecond, padZeros(currentMicros, 6).c_str(), Channel1, Channel2, foreheadTemp, galvanicSkinResponse, galvanicSkinResponse, galvanicSkinResponse);
    //sprintf(bufferSmall, "%i.%s,%i,%i,%i,%i,%i,%i\n", currentSecond, padZeros(currentMicros, 6).c_str(), 1, 1, 1, 1, 1, 1);
    // Send Sensor Data Over WiFi using Port 1234
    udp.broadcastTo(buffer, port);

    delay(5);
}
