#include "WiFi.h"
#include "AsyncUDP.h"

const char * ssid = "87 Marion WiFi";
const char * password = "coldbrew";

AsyncUDP udp;

void setup()
{
    Serial.begin(115200);
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    if (WiFi.waitForConnectResult() != WL_CONNECTED) {
        Serial.println("WiFi Failed");
        while(1) {
            delay(1000);
        }
    }
    Serial.print("MAC Address:");
    Serial.println(WiFi.macAddress());
    
    if(udp.listen(1234)) {
        Serial.print("UDP Listening on IP: ");
        Serial.println(WiFi.localIP());
        udp.onPacket([](AsyncUDPPacket packet) {
            Serial.write(packet.data(), packet.length());
        });
    }
}

void loop() {
    //Send broadcast
    //udp.broadcast("Anyone here?");
}
