// ********************************** Import Libraries ********************************** //

// WiFi Libraries
#include <esp_wifi.h> // ESP32 WiFi Libraries
#include <esp_now.h>  // ESPNOW WiFi communication
#include <WiFi.h>     // Arduino WiFi Libraries
// ESP32 ADC Library
#include "esp_adc_cal.h" // Calibrate ADC for ESP32

// **************************** Initialize General Variables **************************** //

// Create structure to hold incoming data
char decompressedMessage[18];

// ************************************************************************************** //
// ******************************* Recieve and Print Data ******************************* //

// callback function that will be executed when data is received
void OnDataRecv(const uint8_t *mac, const uint8_t *incomingData, int len) {
    Serial.write(incomingData, len - 4);
    Serial.println();
    //compileAndSendData(incomingData);
}

// ************************************************************************************** //
// ********************************** Helper Functions ********************************** //

String decompressByte(char compressedChar) {
    // Convert to Bytes
    byte compressedByte = byte(compressedChar);
    // Split into Two Bytes
    uint8_t leftInt = 0x0F >> 4 | compressedByte >> 4;
    uint8_t rightInt = 0x0F >> 4 | (0x0F & compressedByte);
    // Return the Final Characters
    return String(leftInt) + String(rightInt);
}

void compileAndSendData(const uint8_t *incomingData) {
    sprintf(decompressedMessage, "%s.%s%s%s,%s%s,%s%s,%s%s,%s%s", 
        decompressByte(incomingData[0]),
        decompressByte(incomingData[1]), decompressByte(incomingData[2]), decompressByte(incomingData[3]),
        decompressByte(incomingData[4]), decompressByte(incomingData[5]),
        decompressByte(incomingData[6]), decompressByte(incomingData[7]),
        decompressByte(incomingData[8]), decompressByte(incomingData[9]),
        decompressByte(incomingData[10]), decompressByte(incomingData[11]), decompressByte(incomingData[12])
    );
    Serial.println(decompressedMessage);
}

void connectToPeer() {
    // Establish a WiFi 
    WiFi.mode(WIFI_AP_STA);
    // Print the MAC Address of the Device
    Serial.print("MAC Address:");
    Serial.println(WiFi.macAddress());

    // Init ESP-NOW
    if (esp_now_init() != ESP_OK) {
      Serial.println("Error initializing ESP-NOW");
      return;
    }/Users/samuelsolomon/Desktop/Gao Group/Projects/Stress Response - Forehead/Code/Stress-Analysis-Head/helperFiles/_hardwareCode/Patch Control/Data Streaming/_allSensors/ESP32_NOW_WiFI/WiFi_SendData/WiFi_SendData.ino
    
    // Once ESPNow is successfully Init, we will register for recv CB to get recv packer info
    esp_now_register_recv_cb(OnDataRecv);
    
    // Setup WiFi parameters
    esp_wifi_start();
    // ESP-Now General Setup
    esp_wifi_config_espnow_rate(WIFI_IF_STA, WIFI_PHY_RATE_1M_L);  // Set Data Transfer Rate
    esp_wifi_set_storage(WIFI_STORAGE_FLASH);  // Store Data in Flash and Memory
    esp_event_loop_create_default();
    esp_netif_init();
}

// ************************************************************************************** //
// *********************************** Arduino Setup ************************************ //

void setup() {
    //Initialize serial and wait for port to open:
    Serial.begin(115200);     // Use 115200 baud rate for serial communication
    Serial.flush();           // Flush anything left in the Serial port
    
    // Setup ESP32
    connectToPeer();  // Initialize WiFi
}

// ************************************************************************************** //
// ************************************ Arduino Loop ************************************ //

void loop() {
}
