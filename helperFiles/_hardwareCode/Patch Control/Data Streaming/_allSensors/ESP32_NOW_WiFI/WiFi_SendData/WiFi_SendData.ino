// ********************************** Import Libraries ********************************** //

// WiFi Libraries
#include <esp_wifi.h> // ESP32 WiFi Libraries
#include <esp_now.h>  // ESPNOW WiFi communication
#include <WiFi.h>     // Arduino WiFi Libraries
// Digital Write Library
#include <digitalWriteFast.h>
// ESP32 ADC Library
#include "esp_adc_cal.h" // Calibrate ADC for ESP32

// **************************** Initialize General Variables **************************** //

// Time Variables
const unsigned int oneSecMicro = pow(10,6);
unsigned long beginSamplingTime;
unsigned long endSamplingTime;
unsigned int previousMicros;
unsigned int currentMicros;
unsigned int currentSecond;
// String-Form Time Variables
String currentSecond_String;
String currentMicros_String;
// Keep track of loop time
unsigned long totalLoopTime;
unsigned long startLoopTime;

// Analog read variables
float readingADC = 0;
int numberOfReadsADC = 30;
const byte allAdcChannels[] = {A4}; // Connect common (Z) to A3 (analog input)
adc1_channel_t analogChannel = ADC1_CHANNEL_3; // ADC1_CHANNEL_3 for board, ADC1_CHANNEL_4 for mine
// ADC storage variables
const int numChannels = 4;
float adcReadings[numChannels] = {0.000, 0.000, 0.000, 0.000};
const int padLength[numChannels] = {4, 4, 8, 6};
String adcReadings_String[numChannels] = {"", "", "", ""};
// Pinout variables
const gpio_num_t multiplexPins[3] = {GPIO_NUM_40, GPIO_NUM_41, GPIO_NUM_42}; // {S0, S1, S2}

// Heating pad variables
const gpio_num_t heatingPadPins[1] = {GPIO_NUM_48};
boolean connectToPython = false;

// Calibration variables
float calibratedVolts;
float calibratedResistance;

// ****************************** Initialize WiFi Variables ***************************** //

// Buffer for Serial Printing
const int maxLengthSending = 22;
char sendingMessage[maxLengthSending];
char decompressedMessage[maxLengthSending];

// Broadcasting Variables
uint8_t broadcastAddress[] = {0x7C, 0xDF, 0xA1, 0xF3, 0xCC, 0x54}; // of recieving board!
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

String decompressByte(char compressedChar) {
    // Convert to Bytes
    byte compressedByte = byte(compressedChar);
    // Split into Two Bytes
    uint8_t leftInt = 0x0F >> 4 | compressedByte >> 4;
    uint8_t rightInt = 0x0F >> 4 | (0x0F & compressedByte);
    // Return the Final Characters
    return String(leftInt) + String(rightInt);
}

void compileAndSendData(const uint8_t * incomingData) {
    sprintf(decompressedMessage, "%s.%s%s%s,%s%s,%s%s,%s%s,%s%s", 
        decompressByte(incomingData[0]),
        decompressByte(incomingData[1]), decompressByte(incomingData[2]), decompressByte(incomingData[3]),
        decompressByte(incomingData[4]), decompressByte(incomingData[5]),
        decompressByte(incomingData[6]), decompressByte(incomingData[7]),
        decompressByte(incomingData[8]), decompressByte(incomingData[9]),
        decompressByte(incomingData[10]), decompressByte(incomingData[11])
    );
    Serial.println(decompressedMessage);
    Serial.println("This function may not work");
}

// The selectMuxPin function sets the S0, S1, and S2 pins accordingly, given a pin from 0-7.
void selectMuxPin(byte pin) {
  for (int i=0; i<3; i++) {
    if (pin & (1<<i))
      digitalWriteFast(multiplexPins[i], HIGH);
    else
      digitalWriteFast(multiplexPins[i], LOW);
  }
}

void setHeatingPad(int pwmValue) {
    // Ensure the received value is within the valid PWM range (0 to 255)
    if (pwmValue > 0 && pwmValue <= 255) {

      for (int pinInd = 0; pinInd < (int) sizeof(heatingPadPins)/sizeof(heatingPadPins[0]); pinInd++) {
          analogWrite(heatingPadPins[pinInd], pwmValue); // Set PWM duty cycle
          Serial.println(pwmValue);
          Serial.println(pinInd);
      }
    } else {
      Serial.println("Invalid PWM value. Please enter a value between 0 and 255.");
    }
}

// Read and average ADC values. Throw out initial reads.
float readADC(adc1_channel_t channel) {
    // Throw out the first result
    adc1_get_raw(channel);
    
    readingADC = 0.00000000;
    // Multisampling Analog Read
    for (int i = 0; i < numberOfReadsADC; i++) {
        // Stream in the Data from the Board
        readingADC += adc1_get_raw(channel);
    }
    // Calibrate the ADC value - BOARD SPECIFIC!
    readingADC = calibrateADC(readingADC/numberOfReadsADC);

    return readingADC;
}

// Convert to Volts: THIS IS SPECIFIC TO THIS ESP32!!!
float calibrateADC(float adcRead) {
    // Edge Effect: Value too low
    if (adcRead < 1000) { // 0.0 - 0.8
      float x1_polynomials [] = {1.75280262e-10, -2.95228712e-07,  9.63684730e-04,  5.52278394e-03};
      calibratedVolts = (x1_polynomials[0] * pow(adcRead, 3)) + (x1_polynomials[1] * pow(adcRead, 2)) + (x1_polynomials[2] * adcRead) + x1_polynomials[3];
    }
    else if (adcRead < 3200) { // 0.9 - 2.5
      float x2_polynomials [] = {-6.16258006e-09,  8.07852499e-04,  4.16612572e-02};
      calibratedVolts = ((x2_polynomials[0] * pow(adcRead, 2)) + (x2_polynomials[1] * adcRead) + x2_polynomials[2]);
    }
    else if (adcRead < 4095){ // 2.6 - 3.3
      float x3_polynomials [] = {1.10884628e-09, -1.20158482e-05,  4.39214648e-02, -5.13083813e+01};
      calibratedVolts = ((x3_polynomials[0]  * pow(adcRead, 3)) + (x3_polynomials[1] * pow(adcRead, 2)) + (x3_polynomials[2] * adcRead) + x3_polynomials[3]);
    }
    else {
        calibratedVolts =  3.2;
    }

    // Convert Back to 12-Bits
    return calibratedVolts;
}

int calibrateGSR(float adcVolts) {
    // For low values, its exponential. kOhms
    if (adcVolts < 1.2) { // 0.0 - 1.2
      calibratedResistance = exp(1398.8809330070371*pow(adcVolts,8) + -11146.017834608085*pow(adcVolts,7) + 38670.73709503147*pow(adcVolts,6) + -76316.02640066305*pow(adcVolts,5) + 93726.3007636591*pow(adcVolts,4) + -73390.19405540789*pow(adcVolts,3) + 35811.97025934075*pow(adcVolts,2) + -9974.21177281656*pow(adcVolts,1) + 1225.519190873052*pow(adcVolts,0));
    }
    // For high values its near linear. kOhms
    else { // 1.2 - 4.7
      calibratedResistance = exp(-0.002034474066178099*pow(adcVolts,8) + 0.03402571786784948*pow(adcVolts,7) + -0.2253949394278506*pow(adcVolts,6) + 0.6988972820867152*pow(adcVolts,5) + -0.6450154560251511*pow(adcVolts,4) + -2.243456006311656*pow(adcVolts,3) + 8.112278915546295*pow(adcVolts,2) + -11.595426973626145*pow(adcVolts,1) + 12.81436857642018*pow(adcVolts,0));
    }

    // Return the final resistance
    return (int) (calibratedResistance*1000);
}

int calibrateGSR_Original(float adcVolts) {
    // For low values, its exponential. kOhms
    if (adcVolts < 1.2) { // 0.0 - 1.2
      calibratedResistance = exp(-5.702526020482179*pow(adcVolts,8) + 21.680488058047903*pow(adcVolts,7) + -22.25516726208301*pow(adcVolts,6) + -19.345739611631604*pow(adcVolts,5) + 67.0414659544719*pow(adcVolts,4) + -69.86737865451816*pow(adcVolts,3) + 39.465200722925275*pow(adcVolts,2) + -14.558594467786026*pow(adcVolts,1) + 9.870159927878047*pow(adcVolts,0));
    }
    // For high values its near linear. kOhms
    else { // 1.2 - 4.7
      calibratedResistance = exp(-0.0020689204531101715*pow(adcVolts,8) + 0.04242617856240086*pow(adcVolts,7) + -0.3622468043185021*pow(adcVolts,6) + 1.6728866202661836*pow(adcVolts,5) + -4.526179476386782*pow(adcVolts,4) + 7.1859341955229805*pow(adcVolts,3) + -6.089451318720299*pow(adcVolts,2) + 1.0696618084689502*pow(adcVolts,1) + 7.3370135013102304*pow(adcVolts,0));
    }

    // Return the final resistance
    return (int) (calibratedResistance*1000);
}

int calibrateTemp_TMP36(float tempVolts) {
    // Convert mV to temperature.
    float calibratedTemp = (tempVolts*1000.0000 - 500.0000) / 10.0000;
    //Serial.println(calibratedTemp);

    // Return the 12-Bit value
    return (int) (calibratedTemp*4095.0000/3.3000);
}

String padZeros(int number, int totalLength) {
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

void heatingPadInterface() {
    if (Serial.available() > 0) {
      int pwmValue = Serial.parseInt(); // Read the integer value from Serial Monitor
      setHeatingPad(pwmValue);
    }
}

// ************************************************************************************** //
// *********************************** Setup Functions ********************************** //

void setupMultiplexer() {
    // Set up the select pins as outputs:
    for (int pinInd = 0; pinInd < (int) sizeof(multiplexPins)/sizeof(multiplexPins[0]); pinInd++) {
        pinMode(multiplexPins[pinInd], OUTPUT);
        digitalWriteFast(multiplexPins[pinInd], HIGH);
    }
}

void setupHeatingPad() {
    for (int pinInd = 0; pinInd < (int) sizeof(heatingPadPins)/sizeof(heatingPadPins[0]); pinInd++) {
        pinMode(heatingPadPins[pinInd], OUTPUT);
    }
}

void setupADC() {
    // Attach ADC Pins
    for (int adcChannelInd = 0; adcChannelInd < (int) sizeof(allAdcChannels)/sizeof(allAdcChannels[0]); adcChannelInd++) {
        adcAttachPin(allAdcChannels[adcChannelInd]);
    }
    // Configure ADC Pins
    adc1_config_channel_atten(analogChannel, ADC_ATTEN_DB_11); 

    // Set ADC Calibration
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
    // Print the MAC Address of the Device
    Serial.print("MAC Address:");
    Serial.println(WiFi.macAddress());

    // Init ESP-NOW
    if (esp_now_init() != ESP_OK) {
      Serial.println("Error initializing ESP-NOW");
      return;
    }
    
    // Specify the peer structure
    peerInfo.channel = 0; peerInfo.encrypt = false;
    memcpy(peerInfo.peer_addr, broadcastAddress, 6);
    // Registar the peer  
    if (esp_now_add_peer(&peerInfo) != ESP_OK){
      Serial.println("Failed to add peer");
      return;
    }
    
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

// Setup Arduino; Runs Once
void setup() {
    //Initialize serial and wait for port to open:
    Serial.begin(115200);     // Use 115200 baud rate for serial communication
    Serial.flush();           // Flush anything left in the Serial port
    
    // Setup ESP32
    setupADC();         // ADC calibration
    //connectToPeer();    // Initialize WiFi
    setupMultiplexer(); // Setup output pins
    setupHeatingPad();

    // Start the Timer at Zero
    currentSecond = 0;
    previousMicros = micros();
}

// ************************************************************************************** //
// ************************************ Arduino Loop ************************************ //

void loop() {
    startLoopTime = micros();
    beginSamplingTime = micros() - previousMicros;

    // Multisampling with multiplex analog read
    for (byte multiplexPin = 0; multiplexPin < numChannels; multiplexPin++) {
      selectMuxPin(multiplexPin); // Select one pin to connect to Z
      adcReadings[multiplexPin] = readADC(analogChannel); // Reads in Z from multiplexer
    }
    // Calibrate signals
    adcReadings[0] = adcReadings[0]*4095.0/3.3;
    adcReadings[1] = adcReadings[1]*4095.0/3.3;
    adcReadings[2] = calibrateGSR(adcReadings[2]);
    adcReadings[3] = calibrateTemp_TMP36(adcReadings[3]);

    // Record Final Time
    endSamplingTime = micros() - previousMicros;
    // Record the Time the Signals Were Collected (from Previous Point)
    currentMicros = (beginSamplingTime + endSamplingTime)/2;
    while (currentMicros >= oneSecMicro) {
        currentSecond += 1;
        currentMicros -= oneSecMicro;
    }
    
    // Convert Data into String
    currentSecond_String = padZeros(currentSecond, 2);
    currentMicros_String = padZeros(currentMicros, 6);
    for (byte multiplexPin = 0; multiplexPin < numChannels; multiplexPin++) {
        adcReadings_String[multiplexPin] = padZeros((int) adcReadings[multiplexPin], padLength[multiplexPin]);
    }
    
    // Compile Sensor Data to Send
    sprintf(sendingMessage, "%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c", 
        compressBytes(currentSecond_String[0], currentSecond_String[1]),
        compressBytes(currentMicros_String[0], currentMicros_String[1]), compressBytes(currentMicros_String[2], currentMicros_String[3]), compressBytes(currentMicros_String[4], currentMicros_String[5]),
        compressBytes(adcReadings_String[1][0], adcReadings_String[1][1]), compressBytes(adcReadings_String[1][2], adcReadings_String[1][3]),
        compressBytes(adcReadings_String[0][0], adcReadings_String[0][1]), compressBytes(adcReadings_String[0][2], adcReadings_String[0][3]),
        compressBytes(adcReadings_String[2][0], adcReadings_String[2][1]), compressBytes(adcReadings_String[2][2], adcReadings_String[2][3]), compressBytes(adcReadings_String[2][4], adcReadings_String[2][5]), compressBytes(adcReadings_String[2][6], adcReadings_String[2][7]),
        compressBytes(adcReadings_String[3][0], adcReadings_String[3][1]), compressBytes(adcReadings_String[3][2], adcReadings_String[3][3]), compressBytes(adcReadings_String[3][4], adcReadings_String[3][5])
    );
    // Send Sensor Data Using ESP-NOW
    //recieverReply = esp_now_send(broadcastAddress, (uint8_t *) &sendingMessage, sizeof(sendingMessage));
    //char buff[13];
    //compileAndSendData(sendingMessage.toCharArray(buf, 16);
    //Serial.write(sendingMessage, 16);
    //Serial.println();

    // If Data Sent
    if (recieverReply == ESP_OK) {
        // Keep Track of Time Gap Between Points
        previousMicros = previousMicros + currentMicros + oneSecMicro*currentSecond;
    }

    // Feedback loop.
    heatingPadInterface();

    // Reset Parameters
    currentSecond = 0;
    memset(&sendingMessage[0], 0, sizeof(sendingMessage));

    // Add Delay for WiFi to Send Data
    totalLoopTime = micros() - startLoopTime;
    if (3000 - 100 > totalLoopTime) {delayMicroseconds(3000 - totalLoopTime);}
    //delay(1000);
}

