#include <esp_wifi.h>
#include <esp_now.h>
#include <WiFi.h>

char finalMessage[15];

String decompressByte(char compressedChar) {
    // Convert to Bytes
    byte compressedByte = byte(compressedChar);
    // Split into Two Bytes
    uint8_t leftInt = 0x0F >> 4 | compressedByte >> 4;
    uint8_t rightInt = 0x0F >> 4 | (0x0F & compressedByte);
    // Return the Final Characters
    return String(leftInt) + String(rightInt);
}

// callback function that will be executed when data is received
void OnDataRecv(const uint8_t *mac, const uint8_t *incomingData, int len) {
    Serial.write(incomingData, len);
    Serial.println();
    //compileAndSendData(incomingData);
}

void compileAndSendData(const uint8_t *incomingData) {
    sprintf(finalMessage, "%s.%s%s%s,%s%s,%s%s%s,%s%s", 
        decompressByte(incomingData[0]),
        decompressByte(incomingData[1]), decompressByte(incomingData[2]), decompressByte(incomingData[3]),
        decompressByte(incomingData[4]), decompressByte(incomingData[5]),
        decompressByte(incomingData[6]), decompressByte(incomingData[7]), decompressByte(incomingData[8]),
        decompressByte(incomingData[9]), decompressByte(incomingData[10])
    );
    Serial.println(finalMessage);
}
 
void setup() {
    // Initialize Serial Monitor
    Serial.begin(115200);
    
    // Set device as a Wi-Fi Station
    WiFi.mode(WIFI_STA);
    esp_wifi_start();
  
    // Init ESP-NOW
    if (esp_now_init() != ESP_OK) {
      Serial.println("Error initializing ESP-NOW");
      return;
    }
    
    // Once ESPNow is successfully Init, we will register for recv CB to
    // get recv packer info
    esp_now_register_recv_cb(OnDataRecv);
  
    // ESP-Now General Setup
    esp_wifi_config_espnow_rate(WIFI_IF_STA, WIFI_PHY_RATE_1M_L);  // Set Data Transfer Rate
    esp_wifi_set_storage(WIFI_STORAGE_FLASH);  // Store Data in Flash and Memory
    esp_event_loop_create_default();
    esp_netif_init();

    // wifi_bandwidth_t, WIFI_FAST_SCAN
}
 
void loop() {
}
