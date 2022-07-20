
// ********************************** Import Libraries ********************************** //

#include <ArduinoMqttClient.h>
#include <WiFiNINA.h>
#include <SPI.h>
#include "./arduino_secrets.h"

// ******************************** Initialize Variables ******************************** //

// WiFi Credentials (edit in arduino_secrets.h)
char ssid[] = SECRET_SSID;    // your network SSID (name)
char pass[] = SECRET_PASS;    // your network password (use for WPA, or use as key for WEP)

// Initialize the Wifi client
WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);
int status = WL_IDLE_STATUS;

// Information for MQQT server
const char broker[]  = "test.mosquitto.org";
int        port      = 1883;
const char topic[]   = "sensorData";

// Buffer for Serial Printing
char buffer[40];

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
    Serial.println("\nPlease upgrade the firmware");
  }

  // Establish a WiFi connection
  while (status != WL_CONNECTED) {
    Serial.println("\nAttempting to connect to SSID: " + String(ssid));
    status = WiFi.begin(ssid, pass);
    
    // Wait 10 seconds for connection:
    delay(5000);
  }

  // Print connection status
  printMacAddress();
  printWiFiStatus();
  Serial.println("You're connected to the network");
}

void connectToMQTT() {
  Serial.print("\nAttempting to connect to the MQTT broker: ");
  Serial.println(broker);

  if (!mqttClient.connect(broker, port)) {
    Serial.print("MQTT connection failed! Error code = ");
    Serial.println(mqttClient.connectError());
    while (1);
  }

  /**
   * Keep Alive:
   * Setting the value to zero deactivates keep alive functionality.
   * The maximum keep alive interval that can be specified is 18 hours, 12 minutes, and 15 seconds.
  **/
  mqttClient.setKeepAliveInterval(90);

  Serial.println("You're connected to the MQTT broker!\n");
}

void suscribeToBroker() {
  // set the message receive callback
  mqttClient.onMessage(onMqttMessage);

  // subscribe to a topic
  mqttClient.subscribe(topic);
  Serial.print("Subscribed to topic: " + String(topic) + "\n");
  // topics can be unsubscribed using:
  // mqttClient.unsubscribe(topic);
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
  connectToMQTT();
  suscribeToBroker();
}

void loop() {
  // call poll() regularly to allow the library to receive MQTT messages and send MQTT keep alive which avoids being disconnected by the broker
  mqttClient.poll();
}

void onMqttMessage(int messageSize) {
  if (false) {
    // we received a message, print out the topic and contents
    Serial.println("Received a message with topic '");
    Serial.print(mqttClient.messageTopic());
    Serial.print("', length ");
    Serial.print(messageSize);
    Serial.println(" bytes:");
  }
  
  // use the Stream interface to print the contents
  while (mqttClient.available()) {
    //char a = (char)mqttClient.read();
    //Serial.print(a);
    //Serial.println("");
    Serial.print((char)mqttClient.read());

    // Print Data for Python to Read
    //sprintf(buffer, "%s", String((char)mqttClient.read()).c_str());
    //Serial.print(buffer);
    //Serial.flush();
  }
  Serial.println("");
}
