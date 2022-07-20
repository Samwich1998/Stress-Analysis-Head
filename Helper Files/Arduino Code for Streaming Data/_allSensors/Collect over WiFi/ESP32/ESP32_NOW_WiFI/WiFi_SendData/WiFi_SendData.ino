// ********************************** Import Libraries ********************************** //

// WiFi Libraries
#include <esp_wifi.h>
#include <esp_now.h>
#include <WiFi.h>
#include <SPI.h>
// ESP32 ADC Library
#include "esp_adc_cal.h"

// **************************** Initialize General Variables **************************** //

// Time Variables
const unsigned long oneSecMicro = pow(10,6);
unsigned long beginSamplingTime;
unsigned long endSamplingTime;
unsigned long previousMicros;
unsigned long currentMicros;
unsigned long currentSecond;
unsigned long lastMicros;
int currentMinute;
int currentHour;
// String-Form Time Variables
String currentSecond_String;
String currentMicros_String;

// Analog Pins
const byte ADC0 = A0;
const byte ADC3 = A5;
const byte ADC4 = A4;
// ESP32 Pins
adc1_channel_t channelEOG = ADC1_CHANNEL_0;
adc1_channel_t channelPPG = ADC1_CHANNEL_4;
adc1_channel_t channelGSR = ADC1_CHANNEL_5;
// Specify ADC Parameters
int numberOfReadsADC = 10;
float calibratedVolts;

// Streaming Variables
int ppgChannel_ADC;
int eogChannelVertical_ADC;
int galvanicSkinResponse_ADC;
// String-Form Variables
String ppgChannel_String;
String eogChannelVertical_String;
String galvanicSkinResponse_String;

// Buffer for Serial Printing
const int maxLengthSending = 11;
char sendingMessage[maxLengthSending];

// ****************************** Initialize WiFi Variables ***************************** //

// Broadcasting Variables
uint8_t broadcastAddress[] = {0x7C, 0xDF, 0xA1, 0xF3, 0xCB, 0xBC};
esp_now_peer_info_t peerInfo;
// Assuming No Data MisHandled
esp_err_t recieverReply = ESP_OK;

// ************************************************************************************** //
// ********************************** Helper Functions ********************************** //

byte compressBytes(uint8_t leftInt, uint8_t rightInt) {
    // Convert to Bytes
    byte leftByte = byte(leftInt);
    byte rightByte = byte(rightInt);
    // Compress to One Byte
    byte compressedByte = leftByte << 4 | (0x0F & rightByte);

    return compressedByte;
}

int calibrateADC(float adcRead) {
    // Convert to Volts: THIS IS SPECIFIC TO THIS ESP32!!!
    // Edge Effect: Value too low
    if (adcRead < 500) {
        calibratedVolts = -3.07543411e-13*pow(adcRead, 4) + 4.61714400e-10*pow(adcRead, 3) + -2.34442411e-07*pow(adcRead, 2) + 8.66338357e-04*adcRead + 2.86563719e-02;
    } else {
        calibratedVolts = 1.74592785e-21*pow(adcRead, 6) + -2.36105943e-17*pow(adcRead, 5) + 1.16407137e-13*pow(adcRead, 4) + -2.64411520e-10*pow(adcRead, 3) + 2.74206734e-07*pow(adcRead, 2) + 6.95916329e-04*adcRead + 5.09256786e-02;
    }
    
    // Convert Back to 12-Bits
    return round(calibratedVolts*(4096/3.3));
}

String padZeros(unsigned long number, int totalLength) {
    String finalNumber = String(number);
    int numZeros = totalLength - finalNumber.length();
    for (int i = 0; i < numZeros; i++) {
      finalNumber = "0" + finalNumber;
    }
    return finalNumber;
}

void printBytes(byte inputByte) {
    for (int i = 7; i >= 0; i--) {
        bool bitVal = bitRead(inputByte, i);
        Serial.print(bitVal);
    }
    Serial.println();
}

// ************************************************************************************** //
// *********************************** Setup Functions ********************************** //

void setupADC() {
    // Attach ADC Pins
    adcAttachPin(ADC0);
    adc1_config_channel_atten(channelPPG, ADC_ATTEN_DB_11);  
    adcAttachPin(ADC3);
    adc1_config_channel_atten(channelEOG, ADC_ATTEN_DB_11);  
    adcAttachPin(ADC4);
    adc1_config_channel_atten(channelGSR, ADC_ATTEN_DB_11); 
     
    // ADC Calibration
    adc_set_clk_div(2);
    analogReadResolution(12);  // Initialize ADC Resolution (Arduino Nano 33 IoT Max = 12)
    adc1_config_width(ADC_WIDTH_12Bit);
    adc_set_data_width(ADC_UNIT_1, ADC_WIDTH_BIT_12);
    // Calibrate ADC  
    esp_adc_cal_characteristics_t adc_chars;
    esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 1000, &adc_chars);
}

void connectToPeer() {
    // Establish a WiFi Station
    WiFi.mode(WIFI_STA);
    esp_wifi_start();
    // Print the MAC Address of the Device
    Serial.print("MAC Address:");
    Serial.println(WiFi.macAddress());

    // Init ESP-NOW
    if (esp_now_init() != ESP_OK) {
      Serial.println("Error initializing ESP-NOW");
      return;
    }
    
    // Register peer
    memcpy(peerInfo.peer_addr, broadcastAddress, 6);
    peerInfo.channel = 0;  
    peerInfo.encrypt = false;
    
    // Add peer        
    if (esp_now_add_peer(&peerInfo) != ESP_OK){
      Serial.println("Failed to add peer");
      return;
    }

    // ESP-Now General Setup
    esp_wifi_config_espnow_rate(WIFI_IF_STA, WIFI_PHY_RATE_1M_L);  // Set Data Transfer Rate
    esp_wifi_set_storage(WIFI_STORAGE_FLASH);  // Store Data in Flash and Memory
    esp_event_loop_create_default();
    esp_netif_init();
}

// ************************************************************************************** //
// *********************************** Arduino Setup ************************************ //

// Setup Arduino; Runs Once
void setup() {
    //Initialize serial and wait for port to open:
    Serial.begin(115200);     // Use 115200 baud rate for serial communication
    Serial.flush();           // Flush anything left in the Serial port
    
    // Setup ESP32
    setupADC(); // ADC Calibration
    connectToPeer();  // Initialize WiFi
    //connectToClock();

    currentSecond = 0;
    previousMicros = micros();
}

// ************************************************************************************** //
// ************************************ Arduino Loop ************************************ //

void loop() {

    beginSamplingTime = micros() - previousMicros;
    // Reset Variables
    ppgChannel_ADC = 0;
    eogChannelVertical_ADC = 0;
    galvanicSkinResponse_ADC = 0;
    // Multisampling Analog Read
    for (int i = 0; i < numberOfReadsADC; i++) {
        // Stream in the Data from the Board
        ppgChannel_ADC += adc1_get_raw(channelPPG);
        eogChannelVertical_ADC += adc1_get_raw(channelEOG);
        galvanicSkinResponse_ADC += adc1_get_raw(channelGSR);
    }
    ppgChannel_ADC = calibrateADC(ppgChannel_ADC/numberOfReadsADC);
    eogChannelVertical_ADC = calibrateADC(eogChannelVertical_ADC/numberOfReadsADC);
    galvanicSkinResponse_ADC = calibrateADC(galvanicSkinResponse_ADC/numberOfReadsADC);
    // Record Final Time
    endSamplingTime = micros() - previousMicros;

    // Record the Time the Signals Were Collected (from Previous Point)
    currentMicros = (beginSamplingTime + endSamplingTime)/2;
    while (currentMicros >= oneSecMicro) {
        currentSecond += 1;
        currentMicros -= oneSecMicro;
    }
    
    // Convert Data into String
    ppgChannel_String = padZeros(ppgChannel_ADC, 6);
    eogChannelVertical_String = padZeros(eogChannelVertical_ADC, 4);
    galvanicSkinResponse_String = padZeros(galvanicSkinResponse_ADC, 4);
    // Convert Times into String
    currentSecond_String = padZeros(currentSecond, 2);
    currentMicros_String = padZeros(currentMicros, 6);

    // Compile Sensor Data to Send
    sprintf(sendingMessage, "%c%c%c%c%c%c%c%c%c%c%c", 
        compressBytes(currentSecond_String[0], currentSecond_String[1]),
        compressBytes(currentMicros_String[0], currentMicros_String[1]), compressBytes(currentMicros_String[2], currentMicros_String[3]), compressBytes(currentMicros_String[4], currentMicros_String[5]),
        compressBytes(eogChannelVertical_String[0], eogChannelVertical_String[1]), compressBytes(eogChannelVertical_String[2], eogChannelVertical_String[3]),
        compressBytes(ppgChannel_String[0], ppgChannel_String[1]), compressBytes(ppgChannel_String[2], ppgChannel_String[3]), compressBytes(ppgChannel_String[4], ppgChannel_String[5]),
        compressBytes(galvanicSkinResponse_String[0], galvanicSkinResponse_String[1]), compressBytes(galvanicSkinResponse_String[2], galvanicSkinResponse_String[3])
    );
    // Send Sensor Data Using ESP-NOW
    esp_err_t recieverReply = esp_now_send(broadcastAddress, (uint8_t *) &sendingMessage, sizeof(sendingMessage));
    
    // If Data Sent
    if (recieverReply == ESP_OK) {
      // Keep Track of Time Gap Between Points
      previousMicros = previousMicros + currentMicros + oneSecMicro*currentSecond;
    }
    
    // Reset Parameters
    currentSecond = 0;
    memset(&sendingMessage[0], 0, sizeof(sendingMessage));
    
    // Add Delay for WiFi to Send Data
    delayMicroseconds(1300);
}
