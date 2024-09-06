#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <EEPROM.h>
#include <Wire.h>
#include <Adafruit_SH1106.h>
#include <ArduinoEigen.h>
#include <BayesOptimizer.h>

// Definicje pinów dla tranzystorów
const int mosfetPin = D4;      
const int bjtPin1 = D5;        
const int bjtPin2 = D6;        
const int bjtPin3 = D7;        

// Definicje pinów dla dodatkowych tranzystorów sterujących cewkami wzbudzenia
const int excitationBJT1Pin = D8;
const int excitationBJT2Pin = D9;

// Stałe konfiguracyjne
const float LOAD_THRESHOLD = 0.5;
const float COMPENSATION_FACTOR = 0.1;
const int MAX_EXCITATION_CURRENT = 255;
const float MAX_VOLTAGE = 5.0;
const float MIN_VOLTAGE = 0.0;

// Dodane zmienne
const float Kp_max = 5.0;
const float Kp_min = 0.1;
float previousVoltage = 0.0;
unsigned long tuningStartTime = 0;
const unsigned long TUNING_TIMEOUT = 30000;
bool oscillationsDetected = false;
float excitationGain = 1.0;
float Ku = 0.0;
float Tu = 0.0;

// Parametry PID
float Kp = 2.0, Ki = 0.5, Kd = 1.0;
float previousError = 0;
float integral = 0;

// Parametry dyskretyzacji
const int NUM_STATE_BINS_ERROR = 10;
const int NUM_STATE_BINS_LOAD = 5;
const int NUM_STATE_BINS_KP = 5;
const int NUM_STATE_BINS_KI = 3;
const int NUM_STATE_BINS_KD = 3;
const int NUM_STATE_BINS_EXCITATION_CURRENT = 5;

// Całkowita liczba stanów
int total_states = (
    NUM_STATE_BINS_ERROR *
    NUM_STATE_BINS_LOAD *
    NUM_STATE_BINS_KP *
    NUM_STATE_BINS_KI *
    NUM_STATE_BINS_KD *
    NUM_STATE_BINS_EXCITATION_CURRENT *
    NUM_STATE_BINS_EXCITATION_CURRENT
);

const int NUM_ACTIONS = 6;
const float epsilon = 0.1;
const float learningRate = 0.1;
const float discountFactor = 0.9;

// Role elementów w stabilizacji napięcia:
// * MOSFET: główny tranzystor przełączający, kontroluje przepływ prądu do obciążenia
// * BJT 1-3: tranzystory sterujące MOSFETem i cewkami wzbudzenia, wzmacniają sygnały sterujące
// * Cewki wzbudzenia: regulują natężenie pola magnetycznego, wpływając na napięcie generowane

// Parametry automatycznej optymalizacji
const unsigned long OPTIMIZATION_INTERVAL = 60000; 
const unsigned long TEST_DURATION = 10000; 
unsigned long lastOptimizationTime = 0;

// Zmienne dla optymalizacji bayesowskiej
BayesOptimizer optimizer;
float params[3] = {0.1, 0.9, 0.1}; 
float bounds[3][2] = {{0.01, 0.5}, {0.8, 0.99}, {0.01, 0.3}}; 
float bestEfficiency = 0.0;

// Stałe dane w pamięci flash (PROGMEM)
const char* welcomeMessage PROGMEM = "Witaj w systemie stabilizacji napięcia!";

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

// Zmienne globalne
float voltageIn[2] = {0};
float currentIn[2] = {0};
ESP8266WebServer server(80);
Adafruit_SH1106 display(128, 64, &Wire, -1);
int lastAction = 0;
float excitationCurrent = 0; 
float generatorLoad = 0;     

// Tablica Q-learning
float qTable[total_states][NUM_ACTIONS][3]; 

// Funkcja dyskretyzacji stanu
int discretizeState(float error, float generatorLoad, float Kp, float Ki, float Kd, float excitationCurrent1, float excitationCurrent2) {
    int errorBin = constrain((int)(abs(error) / (VOLTAGE_SETPOINT / NUM_STATE_BINS_ERROR)), 0, NUM_STATE_BINS_ERROR - 1);
    int loadBin = constrain((int)(generatorLoad / (LOAD_THRESHOLD / NUM_STATE_BINS_LOAD)), 0, NUM_STATE_BINS_LOAD - 1);
    int kpBin = constrain((int)(Kp / (5.0 / NUM_STATE_BINS_KP)), 0, NUM_STATE_BINS_KP - 1);
    int kiBin = constrain((int)(Ki / (1.0 / NUM_STATE_BINS_KI)), 0, NUM_STATE_BINS_KI - 1);
    int kdBin = constrain((int)(Kd / (5.0 / NUM_STATE_BINS_KD)), 0, NUM_STATE_BINS_KD - 1);
    int excitationCurrent1Bin = constrain((int)(excitationCurrent1 / (1.0 / NUM_STATE_BINS_EXCITATION_CURRENT)), 0, NUM_STATE_BINS_EXCITATION_CURRENT - 1);
    int excitationCurrent2Bin = constrain((int)(excitationCurrent2 / (1.0 / NUM_STATE_BINS_EXCITATION_CURRENT)), 0, NUM_STATE_BINS_EXCITATION_CURRENT - 1);

    return errorBin * NUM_STATE_BINS_LOAD * NUM_STATE_BINS_KP * NUM_STATE_BINS_KI * NUM_STATE_BINS_KD * NUM_STATE_BINS_EXCITATION_CURRENT * NUM_STATE_BINS_EXCITATION_CURRENT +
           loadBin * NUM_STATE_BINS_KP * NUM_STATE_BINS_KI * NUM_STATE_BINS_KD * NUM_STATE_BINS_EXCITATION_CURRENT * NUM_STATE_BINS_EXCITATION_CURRENT +
           kpBin * NUM_STATE_BINS_KI * NUM_STATE_BINS_KD * NUM_STATE_BINS_EXCITATION_CURRENT * NUM_STATE_BINS_EXCITATION_CURRENT +
           kiBin * NUM_STATE_BINS_KD * NUM_STATE_BINS_EXCITATION_CURRENT * NUM_STATE_BINS_EXCITATION_CURRENT +
           kdBin * NUM_STATE_BINS_EXCITATION_CURRENT * NUM_STATE_BINS_EXCITATION_CURRENT +
           excitationCurrent1Bin * NUM_STATE_BINS_EXCITATION_CURRENT +
           excitationCurrent2Bin;
}

// Funkcja chooseAction
int chooseAction(int state) {
    int start_index = state * NUM_ACTIONS;
    int end_index = start_index + NUM_ACTIONS;
    float maxQ = qTable[state][0][0];
    int bestAction = 0;
    if (random(100) < epsilon * 100) {
        bestAction = random(NUM_ACTIONS);
    } else {
        for (int i = 1; i < NUM_ACTIONS; i++) {
            if (qTable[state][i][0] > maxQ) {
                maxQ = qTable[state][i][0];
                bestAction = i;
            }
        }
    }
    return bestAction;
}

// Funkcja updateQ
void updateQ(int state, int action, float reward, int nextState) {
    float maxFutureQ = 0;
    for (int nextAction = 0; nextAction < NUM_ACTIONS; nextAction++) {
        maxFutureQ = max(maxFutureQ, qTable[nextState][nextAction][0]);
    }
    qTable[state][action][0] += learningRate * (reward + discountFactor * maxFutureQ - qTable[state][action][0]);
}

void loop() {
    server.handleClient();

    readSensors();

    float externalVoltage = analogRead(PIN_EXTERNAL_VOLTAGE_SENSOR_1) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
    float externalCurrent = analogRead(PIN_EXTERNAL_CURRENT_SENSOR_1) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);

    logData();
    checkAlarm();
    autoCalibrate();
    energyManagement();

    float efficiency = calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);

    int state = discretizeState(VOLTAGE_SETPOINT - voltageIn[0], currentIn[0], Kp, Ki, Kd, readExcitationCoilCurrent(excitationBJT1Pin), readExcitationCoilCurrent(excitationBJT2Pin));
    int action = chooseAction(state);
    executeAction(action);

    float reward = calculateReward(VOLTAGE_SETPOINT - voltageIn[0]);
    updateQ(state, lastAction, reward, state);
    lastAction = action;

    if (millis() - lastOptimizationTime > OPTIMIZATION_INTERVAL) {
        lastOptimizationTime = millis();

        float newParams[3];
        optimizer.suggestNextParameters(newParams);
        params[0] = newParams[0];
        params[1] = newParams[1];
        params[2] = newParams[2];

        float totalEfficiency = 0;
        unsigned long startTime = millis();
        while (millis() - startTime < TEST_DURATION) {
            totalEfficiency += calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);
        }
        float averageEfficiency = totalEfficiency / (TEST_DURATION / 100);

        optimizer.update(newParams, averageEfficiency);

        if (averageEfficiency > bestEfficiency) {
            bestEfficiency = averageEfficiency;
            memcpy(params, newParams, sizeof(params));
        }
    }

    delay(100);

    displayData();

    Serial.print("Wolna pamięć: ");
    Serial.println(freeMemory());
}

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
    Serial.print("Napięcie: ");
    Serial.print(voltageIn[0]);
    Serial.print(" V, Prąd: ");
    Serial.print(currentIn[0]);
    Serial.println(" A");
}

void checkAlarm() {
    if (voltageIn[0] < 220 || voltageIn[0] > 240) {
        Serial.println("Alarm: Napięcie poza zakresem!");
    }
    if (currentIn[0] > 5.0) {
        Serial.println("Alarm: Prąd za wysoki!");
    }
}

void autoCalibrate() {
    static unsigned long lastCalibrationTime = 0;
    if (millis() - lastCalibrationTime > 60000) {
        calibrateSensors();
        lastCalibrationTime = millis();
        Serial.println("Auto-kalibracja zakończona.");
    }
}

void energyManagement() {
    float totalPower = voltageIn[0] * currentIn[0];
    if (totalPower > 1000) {
        Serial.println("Zarządzanie energią: Wykryto wysokie zużycie energii!");
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
    float inputPower = voltageIn * currentIn;
    return outputPower / inputPower;
}

void displayData() {
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

void controlTransistors(float voltage, float excitationCurrent, float generatorLoad) {
    voltage = constrain(voltage, MIN_VOLTAGE, MAX_VOLTAGE);

    digitalWrite(mosfetPin, voltage > 0 ? HIGH : LOW);

    float baseCurrent1 = excitationCurrent * 0.3;
    float baseCurrent2 = excitationCurrent * 0.3;
    float baseCurrent3 = excitationCurrent * 0.4;

    int pwmValueBJT1 = map(baseCurrent1, 0, 0.1, 0, 255);
    int pwmValueBJT2 = map(baseCurrent2, 0, 0.1, 0, 255);
    int pwmValueBJT3 = map(baseCurrent3, 0, 0.1, 0, 255);

    analogWrite(bjtPin1, pwmValueBJT1);
    analogWrite(bjtPin2, pwmValueBJT2);
    analogWrite(bjtPin3, pwmValueBJT3);
}

void setup() {
    Serial.begin(115200);

    pinMode(muxSelectPinA, OUTPUT);
    pinMode(muxSelectPinB, OUTPUT);
    pinMode(PIN_EXCITATION_COIL_1, OUTPUT);
    pinMode(PIN_EXCITATION_COIL_2, OUTPUT);
    pinMode(mosfetPin, OUTPUT);
    pinMode(bjtPin1, OUTPUT);
    pinMode(bjtPin2, OUTPUT);
    pinMode(bjtPin3, OUTPUT);

    server.begin();
    display.begin();
    display.display();

    optimizer.initialize(3, bounds, 50, 10);

    char buffer[64];
    strcpy_P(buffer, welcomeMessage);
    display.println(buffer);
    display.display();
}
}

def updateLearningAlgorithm(voltageError):
    # Obliczanie bieżącego stanu
    currentState = discretizeState(
        VOLTAGE_SETPOINT - voltageIn[0], 
        currentIn[0], 
        Kp, Ki, Kd,
        readExcitationCoilCurrent(excitationBJT1Pin), 
        readExcitationCoilCurrent(excitationBJT2Pin)
    )

    # Obliczanie nagrody - uwzględniając wpływ akcji na hamowanie i wydajność
    loadFactor = currentIn[0] / LOAD_THRESHOLD
    brakingFactor = 1.0 - loadFactor  # Im większe obciążenie, tym mniejszy brakingFactor
    efficiency = calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent) 
    reward = calculateReward(voltageError) * brakingFactor * efficiency

    # Wykonanie akcji (już zaimplementowane w funkcji executeAction)

    # Obliczanie następnego stanu
    nextState = discretizeState(
        VOLTAGE_SETPOINT - voltageIn[0], 
        currentIn[0], 
        Kp, Ki, Kd,
        readExcitationCoilCurrent(excitationBJT1Pin), 
        readExcitationCoilCurrent(excitationBJT2Pin)
    )

    # Aktualizacja tablicy Q
    updateQ(currentState, lastAction, reward, nextState)                         

const int NUM_STATE_BINS_EXCITATION_CURRENT = 5;

int total_states = (
     NUM_STATE_BINS_ERROR *
     NUM_STATE_BINS_LOAD *
     NUM_STATE_BINS_KP *
     NUM_STATE_BINS_KI *
     NUM_STATE_BINS_KD *
     NUM_STATE_BINS_EXCITATION_CURRENT *
     NUM_STATE_BINS_EXCITATION_CURRENT
 );


float qTable[total_states * NUM_ACTIONS][3];

int chooseAction(int state) {
    int start_index = state * NUM_ACTIONS;
    int end_index = start_index + NUM_ACTIONS;
    float maxQ = qTable[start_index][0];
    int bestAction = 0;
    if (random(100) < epsilon * 100) {
        bestAction = random(NUM_ACTIONS);
    } else {
        for (int i = start_index + 1; i < end_index; i++) {
            if (qTable[i][0] > maxQ) {
                maxQ = qTable[i][0];
                bestAction = i - start_index;  // Adjust action index
            }
        }
    }
    return bestAction;
}

void updateQ(int state, int action, float reward, int nextState) {
    int stateActionIndex = state * NUM_ACTIONS + action;
    float maxFutureQ = 0;
    for (int nextAction = 0; nextAction < NUM_ACTIONS; nextAction++) {
        int nextStateActionIndex = nextState * NUM_ACTIONS + nextAction;
        maxFutureQ = max(maxFutureQ, qTable[nextStateActionIndex][0]);
    }
    qTable[stateActionIndex][0] += learningRate * (reward + discountFactor * maxFutureQ - qTable[stateActionIndex][0]);
}

int discretizeState(float error, float generatorLoad, float Kp, float Ki, float Kd, float excitationCurrent1, float excitationCurrent2) {
    // ... (implementacja - zakładam, że jest już poprawna, uwzględniając prądy cewek wzbudzenia)
}

// Ograniczenie parametrów PID do sensownych zakresów
 Kp = constrain(Kp, Kp_min, Kp_max); // Upewnij się, że Kp_min jest zdefiniowane
 Ki = constrain(Ki, 0, 1.0);
 Kd = constrain(Kd, 0, 5.0);

 float pidOutput = calculatePID(VOLTAGE_SETPOINT, voltageIn[0]);

 // Inteligentne sterowanie cewką wzbudzenia
 controlTransistors(pidOutput, excitationCurrent, generatorLoad); // Upewnij się, że excitationCurrent i generatorLoad są zdefiniowane

 // Obserwacja nowego stanu i nagrody
 delay(100);
 float newError = VOLTAGE_SETPOINT - voltageIn[0];
 int newState = discretizeState(newError, generatorLoad, Kp, Ki, Kd, 
                                readExcitationCoilCurrent(excitationBJT1Pin), 
                                readExcitationCoilCurrent(excitationBJT2Pin));
 float reward = calculateReward(newError);

 // Aktualizacja tablicy Q
 updateQ(state, action, reward, newState);

 // Automatyczna optymalizacja parametrów
 if (millis() - lastOptimizationTime > OPTIMIZATION_INTERVAL) {
     lastOptimizationTime = millis();

     // Testowanie nowych parametrów
     float newParams[3];
     optimizer.suggestNextParameters(newParams);
     alpha = newParams[0];
     gamma = newParams[1];
     epsilon = newParams[2];

     // Zbieranie danych o wydajności przez TEST_DURATION
     float totalEfficiency = 0;
     unsigned long startTime = millis();
     while (millis() - startTime < TEST_DURATION) {
         totalEfficiency += calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);
     }
     float averageEfficiency = totalEfficiency / (TEST_DURATION / 100);

     // Przekazanie wyniku do optymalizatora
     optimizer.update(newParams, averageEfficiency);

     // Jeśli nowe parametry są lepsze, zachowaj je
     if (averageEfficiency > bestEfficiency) {
         bestEfficiency = averageEfficiency;
         memcpy(params, newParams, sizeof(params));
         alpha = params[0];
         gamma = params[1];
         epsilon = params[2];
     } else {
         alpha = params[0];
         gamma = params[1];
         epsilon = params[2];
     }
 }

 // Opóźnienie
 delay(100);

 // Wyświetlanie danych na OLED
 displayData();

 // Monitorowanie zużycia pamięci
 Serial.print("Wolna pamięć: ");
 Serial.println(freeMemory());

