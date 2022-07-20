
// ********************************** Import Libraries ********************************** //

// WiFi Libraries
#include <WiFiNINA.h>
#include <SPI.h>
#include <MQTT.h>
// Secret WiFi Passwords
//#include "./arduino_secrets.h"

// ******************************** Initialize Variables ******************************** //

// Buffer for Serial Printing
char buffer[40];

// WiFi Credentials (edit in arduino_secrets.h)
//char ssid[] = SECRET_SSID;    // your network SSID (name)
//char pass[] = SECRET_PASS;    // your network password (use for WPA, or use as key for WEP)
const char ssid[] = "87 Marion WiFi";
const char pass[] = "coldbrew";

// Initialize the Wifi client
int status = WL_IDLE_STATUS;
WiFiClient wifiClient;
// Initialize the WiFi Communication Protocol
const String topic = "/sensorData";
MQTTClient mqttClient(1024);

int timeout = 12000;
int keepAlive = 12000;
bool cleanSession = true;

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
}
 
void printMacAddress() {
    byte mac[6];  // the MAC address of your Wifi shield                  
    WiFi.macAddress(mac);
    Serial.print("MAC: ");
    Serial.print(mac[5],HEX);
    Serial.print(":");
    Serial.print(mac[4],HEX);
    Serial.print(":");
    Serial.print(mac[3],HEX);
    Serial.print(":");
    Serial.print(mac[2],HEX);
    Serial.print(":");
    Serial.print(mac[1],HEX);
    Serial.print(":");
    Serial.println(mac[0],HEX);
}

// ************************************************************************************** //
// ********************************* Connection Functions ******************************* //

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
 
    // Wait for Connection:
    delay(5000);
  }

  // Print connection status
  printMacAddress();
  printWiFiStatus();
}

void setupMQTT() {
    // Set MQTT Options: Keep Alive, Clearn Session, Timeout
    mqttClient.setOptions(keepAlive, cleanSession, timeout);
      
  // Note: Local domain names (e.g. "Computer.local" on OSX) are not supported
  // by Arduino. You need to set the IP address directly.
  mqttClient.begin("public.cloud.shiftr.io", wifiClient);
  mqttClient.onMessage(messageReceived);
    
    Serial.println("Attempting to connect to MQTT");
    connectToMQTT();
  
    Serial.println("You're connected to the MQTT broker!");
}

void connectToMQTT() {
    // Connect to MQTT
    while (!mqttClient.connect("arduino", "public", "public")) {
      delay(1);
    }
    // Suscribe to Topic
    mqttClient.subscribe(topic);
}


// ************************************************************************************** //
// *********************************** Arduino Setup ************************************ //

// Setup Arduino; Runs Once
void setup() {
  //Initialize serial and wait for port to open:
    Serial.begin(115200);     // Use 115200 baud rate for serial communication
    analogReadResolution(12); // Initialize ADC Resolution (Arduino Nano 33 IoT Max = 12)
    Serial.flush();           // Flush anything left in the Serial port

    // Connect to WiFi and Clock
    connectToWiFi();
    setupMQTT();
}

// ************************************************************************************** //
// ************************************ Arduino Loop ************************************ //

void messageReceived(String &topic, String &payload) {
  Serial.println(payload);

  // Note: Do not use the client in the callback to publish, subscribe or
  // unsubscribe as it may cause deadlocks when other things arrive while
  // sending and receiving acknowledgments. Instead, change a global variable,
  // or push to a queue and handle it in the loop after calling `client.loop()`.
}

void loop() {
  mqttClient.loop();
  if (!mqttClient.connected()) {
      connectToMQTT();
  }
}
