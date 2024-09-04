#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <EEPROM.h>
#include <Wire.h>
#include <Adafruit_SH1106.h>
#include <ArduinoEigen.h>

// Definicje pinów dla tranzystorów 
const int mosfetPin = D4;      // Pin dla MOSFET IRFP460
const int bjtPin1 = D5;         // Pin dla BJT MJE13009
const int bjtPin2 = D6;         // Pin dla BJT 2SC5200
const int bjtPin3 = D7;         // Pin dla BJT 2SA1943

// Stałe konfiguracyjne
const float LOAD_THRESHOLD = 0.5;
const float COMPENSATION_FACTOR = 0.1;
const int MAX_EXCITATION_CURRENT = 255;

// Definicje pinów
const int muxSelectPinA = D2;
const int muxSelectPinB = D3;
const int muxInputPin = A0;
const int PIN_EXCITATION_COIL_1 = D0;
const int PIN_EXCITATION_COIL_2 = D1;
const int NUM_SENSORS = 4;
const float VOLTAGE_REFERENCE = 3.3;
const int ADC_MAX_VALUE = 1023;
float VOLTAGE_SETPOINT = 230.0;
const float VOLTAGE_REGULATION_HYSTERESIS = 0.1;

// Parametry PID
float Kp = 2.0, Ki = 0.5, Kd = 1.0;
float previousError = 0;
float integral = 0;

// Parametry adaptacji Kp
float Kp_max = 5.0;
float Kp_min = 1.0;
float Kp_change_rate = 0.01;
float Kp_change_threshold = 0.5;

// Zmienne globalne
float voltageIn[2] = {0};
float currentIn[2] = {0};
float bestKp = Kp;
float bestEfficiency = 0.0;
ESP8266WebServer server(80);

// RLS Variables
float theta[3] = {Kp, Ki, Kd};
float P[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
float lambda = 0.99;

// Historia danych
const int HISTORY_SIZE = 100;
float voltageHistory[HISTORY_SIZE];
float currentHistory[HISTORY_SIZE];
int historyIndex = 0;

// Q-learning
const int NUM_STATES = 10;
const int NUM_ACTIONS = 5;
float Q[NUM_STATES][NUM_ACTIONS] = {0};
float alpha = 0.1;
float gamma = 0.9;
float epsilon = 0.1;
int lastAction = 0;
float epsilonDecay = 0.99;
int episodeCount = 0;

// Tryb pracy
int mode = 0;

// Inicjalizacja wyświetlacza OLED
Adafruit_SH1106 display(128, 64, &Wire, -1);

// Piny dla zewnętrznych czujników napięcia i prądu
const int PIN_EXTERNAL_VOLTAGE_SENSOR = A1;
const int PIN_EXTERNAL_CURRENT_SENSOR = A2;

// Piny dla czujników
const int muxInputPins[NUM_SENSORS] = {A0, A1, A2, A3};

// Zmienne dla metody Zieglera-Nicholsa
bool isTuning = false;          // Flaga wskazująca, czy trwa dostrajanie
float Ku = 0.0;                 // Wzmocnienie krytyczne
float Tu = 0.0;                 // Okres oscylacji
unsigned long tuningStartTime = 0; // Czas rozpoczęcia dostrajania
const unsigned long TUNING_TIMEOUT = 30000; // Maksymalny czas dostrajania (30 sekund)
bool oscillationsDetected = false; // Flaga wskazująca, czy wykryto oscylacje

void setup() {
    Serial.begin(115200);

    // Ustawienie pinów jako wyjścia dla multipleksera
    pinMode(muxSelectPinA, OUTPUT);
    pinMode(muxSelectPinB, OUTPUT);

    // Ustawienie pinów PWM
    pinMode(PIN_EXCITATION_COIL_1, OUTPUT);
    pinMode(PIN_EXCITATION_COIL_2, OUTPUT);

    // Ustawienie pinów tranzystorów jako wyjścia
    pinMode(mosfetPin, OUTPUT);
    pinMode(bjtPin1, OUTPUT);
    pinMode(bjtPin2, OUTPUT);
    pinMode(bjtPin3, OUTPUT);

    // Inicjalizacja serwera i wyświetlacza
    server.begin();
    display.begin();
    calibrateSensors();

    // Opcjonalnie: Rozpocznij dostrajanie Zieglera-Nicholsa po uruchomieniu
    // isTuning = true; 
    // tuningStartTime = millis();
}

void loop() {
    server.handleClient();

    // Odczyt wartości z czujników za pomocą multipleksera
    readSensors();

    // Odczyt napięcia i prądu z dodatkowych zewnętrznych czujników
    float externalVoltage = analogRead(PIN_EXTERNAL_VOLTAGE_SENSOR) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
    float externalCurrent = analogRead(PIN_EXTERNAL_CURRENT_SENSOR) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);

    // Logowanie danych
    logData();

    // Sprawdzenie alarmów
    checkAlarm();

    // Automatyczna kalibracja
    autoCalibrate();

    // Zarządzanie energią
    energyManagement();

    // Obliczanie wydajności
    float efficiency = calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);

    // Monitorowanie wydajności i aktualizacja Kp
    if (efficiency > bestEfficiency) {
        bestEfficiency = efficiency;
        bestKp = Kp;
    } else {
        Kp = bestKp;
    }

    if (isTuning) {
        // Wykonaj procedurę dostrajania Zieglera-Nicholsa
        performZieglerNicholsTuning();
    } else {
        // Normalna praca stabilizatora z obliczonymi parametrami PID
        // Adaptacja Kp (algorytm adaptacyjny PID)
        float error = VOLTAGE_SETPOINT - voltageIn[0];
        if (abs(error) > Kp_change_threshold) {
            Kp += Kp_change_rate * error;
            Kp = constrain(Kp, Kp_min, Kp_max);
        }

        // Q-learning
        int state = (int)(abs(VOLTAGE_SETPOINT - voltageIn[0]) / 2);
        state = constrain(state, 0, NUM_STATES - 1);
        int action = chooseAction(state);
        Kp = action * 0.5 + 1.0;
        float pidOutput = calculatePID(VOLTAGE_SETPOINT, voltageIn[0]);
        controlExcitationCoils(pidOutput);
        float reward = calculateReward(VOLTAGE_SETPOINT - voltageIn[0]);
        updateQ(state, lastAction, reward, state);

        // Aktualizacja algorytmu uczenia maszynowego
        updateLearningAlgorithm(VOLTAGE_SETPOINT - voltageIn[0]);
    }

    // Sterowanie tranzystorami
    controlTransistors(voltageIn[0]); 

    // Opóźnienie 100ms
    delay(100);

    // Wyświetlanie danych na OLED
    display.clearDisplay();
    display.setCursor(0, 0);
    display.print("Napięcie: ");
    display.print(voltageIn[0]);
    display.println(" V");
    display.print("Prąd: ");
    display.print(currentIn[0]);
    display.println(" A");
    display.display();
}

// Funkcje pomocnicze
// readSensors(), logData(), checkAlarm(), autoCalibrate(), energyManagement(), calibrateSensors(), calculatePID
// Funkcje pomocnicze
void readSensors() {
    for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
        digitalWrite(muxSelectPinA, sensor & 0x01);
        digitalWrite(muxSelectPinB, (sensor >> 1) & 0x01);
        if (sensor < 2) {
            float Vcc = 5.0;
            float Sensitivity = 0.066;
            float Vout = analogRead(muxInputPins[sensor]) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
            currentIn[sensor] = (Vout - (Vcc / 2.0)) / Sensitivity;
        } else {
            float multiplier = 100.0;
            voltageIn[sensor - 2] = analogRead(muxInputPins[sensor]) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE) * multiplier;
        }
    }
}

void logData() {
    voltageHistory[historyIndex] = voltageIn[0];
    currentHistory[historyIndex] = currentIn[0];
    historyIndex = (historyIndex + 1) % HISTORY_SIZE;
    Serial.print("Logged Voltage: ");
    Serial.print(voltageIn[0]);
    Serial.print(" V, Current: ");
    Serial.print(currentIn[0]);
    Serial.println(" A");
}

void checkAlarm() {
    if (voltageIn[0] < 220 || voltageIn[0] > 240) {
        Serial.println("Alarm: Voltage out of range!");
    }
    if (currentIn[0] > 5.0) {
        Serial.println("Alarm: Current too high!");
    }
}

void autoCalibrate() {
    static unsigned long lastCalibrationTime = 0;
    if (millis() - lastCalibrationTime > 60000) {
        calibrateSensors();
        lastCalibrationTime = millis();
        Serial.println("Auto-calibration completed.");
    }
}

void energyManagement() {
    float totalPower = voltageIn[0] * currentIn[0];
    if (totalPower > 1000) {
        Serial.println("Energy Management: High power consumption detected!");
    }
}

void calibrateSensors() {
    Serial.println("Kalibracja czujników...");
    for (int i = 0; i < NUM_SENSORS; i++) {
        float voltageSum = 0;
        float currentSum = 0;
        for (int j = 0; j < 10; j++) {
            digitalWrite(muxSelectPinA, i & 0x01);
            digitalWrite(muxSelectPinB, (i >> 1) & 0x01);
            voltageSum += analogRead(muxInputPin);
            currentSum += analogRead(muxInputPin);
            delay(100);
        }
        float voltageOffset = (voltageSum / 10) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
        float currentOffset = (currentSum / 10) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
        Serial.print("Offset napięcia dla czujnika ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(voltageOffset);
        Serial.print("Offset prądu dla czujnika ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(currentOffset);
        voltageIn[i] -= voltageOffset;
        currentIn[i] -= currentOffset;
    }
}

float calculatePID(float setpoint, float measuredValue) {
    float error = setpoint - measuredValue;
    integral += error;
    float derivative = error - previousError;
    previousError = error;
    float controlSignal = Kp * error + Ki * integral + Kd * derivative;
    return controlSignal;
}

float calculateEfficiency(float voltageIn, float currentIn, float externalVoltage, float externalCurrent) {
    float outputPower = externalVoltage * externalCurrent;
    float inputPower = voltageIn[0] * currentIn[0] + voltageIn[1] * currentIn[1];
    return outputPower / inputPower;
}

void updateRLS(float voltageError) {
    float z[3] = {voltageError, voltageError * voltageError, voltageError * voltageError * voltageError};
    float y = voltageError;
    float prediction = theta[0] * voltageError + theta[1] * voltageError * voltageError + theta[2] * voltageError * voltageError * voltageError;
    float error = y - prediction;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            P[i][j] = (1 / lambda) * (P[i][j] - (P[i][j] * z[i] * z[j] * P[i][j]) / (1 + z[j] * P[j][j]));
        }
        theta[i] += P[i][0] * error;
    }
}

void updateQ(int state, int action, float reward, int nextState) {
    float maxQNext = Q[nextState][0];
    for (int i = 1; i < NUM_ACTIONS; i++) {
        if (Q[nextState][i] > maxQNext) {
            maxQNext = Q[nextState][i];
        }
    }
    Q[state][action] += alpha * (reward + gamma * maxQNext - Q[state][action]);
}

float calculateReward(float voltageError) {
    if (abs(voltageError) < 1.0) {
        return 1.0;
    } else if (abs(voltageError) < 2.0) {
        return 0.5;
    } else {
        return -1.0;
    }
}

int chooseAction(int state) {
    float maxQ = Q[state][0];
    int bestAction = 0;
    if (random(100) < epsilon * 100) {
        bestAction = random(NUM_ACTIONS);
    } else {
        for (int i = 1; i < NUM_ACTIONS; i++) {
            if (Q[state][i] > maxQ) {
                maxQ = Q[state][i];
                bestAction = i;
            }
        }
    }
    return bestAction;
}

void controlExcitationCoils(float controlSignal) {
    int pwmValue = constrain(controlSignal, 0, 255);
    analogWrite(PIN_EXCITATION_COIL_1, pwmValue);
    analogWrite(PIN_EXCITATION_COIL_2, pwmValue);
}

void updateLearningAlgorithm(float voltageError) {
    int state = (int)(abs(voltageError) / 2);
    state = constrain(state, 0, NUM_STATES - 1);
    float reward = calculateReward(voltageError);
    updateQ(state, lastAction, reward, state);
}

void testingAndOptimization() {
    Serial.println("Rozpoczęcie testów i optymalizacji...");
    for (int i = 0; i < 10; i++) {
        float testSetpoint = random(200, 250);
        VOLTAGE_SETPOINT = testSetpoint;
        Serial.print("Testowane napięcie: ");
        Serial.println(testSetpoint);
        delay(1000);
        voltageIn[0] = analogRead(muxInputPin) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
        currentIn[0] = analogRead(muxInputPin) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
        float externalVoltage = analogRead(PIN_EXTERNAL_VOLTAGE_SENSOR) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
        float externalCurrent = analogRead(PIN_EXTERNAL_CURRENT_SENSOR) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
        float efficiency = calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);
        Serial.print("Wydajność: ");
        Serial.println(efficiency);
        int currentState = (int)(abs(VOLTAGE_SETPOINT - voltageIn[0]) / 2);
        currentState = constrain(currentState, 0, NUM_STATES - 1);
        int action = lastAction;
        float reward = calculateReward(VOLTAGE_SETPOINT - voltageIn[0]);
        int nextState = currentState;
        updateQ(currentState, action, reward, nextState);
    }
    Serial.println("Testy i optymalizacja zakończone.");
 #include <cmath> // Do ewentualnych obliczeń matematycznych

// Stałe i zmienne globalne (dostosuj do swojego systemu)
const float MAX_VOLTAGE = 5.0; // Maksymalne napięcie sterujące
const float MIN_VOLTAGE = 0.0; // Minimalne napięcie sterujące
bool isTuning = false; // Flaga wskazująca, czy trwa dostrajanie
const int MAX_TUNING_TIME = 10000; // Maksymalny czas dostrajania (w ms)

// Funkcja sterowania tranzystorami
void controlTransistors(float voltage) {
    // Ograniczenie napięcia do zakresu
    voltage = std::max(MIN_VOLTAGE, std::min(voltage, MAX_VOLTAGE));

    // Przykładowa implementacja dla MOSFETów (dostosuj do swoich potrzeb)
    float gateVoltage1 = voltage; // Napięcie bramki dla pierwszego tranzystora
    float gateVoltage2 = MAX_VOLTAGE - voltage; // Napięcie bramki dla drugiego tranzystora

    // Analogicznie dla pozostałych tranzystorów (lub innych typów)

    // Tutaj wyślij sygnały sterujące do tranzystorów (np. za pomocą PWM)
    // ...
}

// Funkcja dostrajania Zieglera-Nicholsa
void performZieglerNicholsTuning() {
    isTuning = true;

    // Znajdź okres oscylacji i wzmocnienie krytyczne
    // ... (implementacja zależy od systemu)

    // Oblicz parametry regulatora PID na podstawie okresu i wzmocnienia
    // ... (zgodnie z tabelami Zieglera-Nicholsa)

    // Ustaw parametry regulatora
    // ...

    isTuning = false;
} }