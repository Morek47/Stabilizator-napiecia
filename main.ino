#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <EEPROM.h>
#include <Wire.h>
#include <Adafruit_SH1106.h>
#include <ArduinoEigen.h>
#include <BayesOptimizer.h>

// Definicje pinów dla tranzystorów
const int mosfetPin = D4;      // Pin dla MOSFET IRFP460
const int bjtPin1 = D5;        // Pin dla BJT MJE13009
const int bjtPin2 = D6;        // Pin dla BJT 2SC5200
const int bjtPin3 = D7;        // Pin dla BJT 2SA1943

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

const int NUM_ACTIONS = 6;
const float epsilon = 0.1;
const float learningRate = 0.1;
const float discountFactor = 0.9;

// Role elementów w stabilizacji napięcia:
// * MOSFET: główny tranzystor przełączający, kontroluje przepływ prądu do obciążenia
// * BJT 1-3: tranzystory sterujące MOSFETem i cewkami wzbudzenia, wzmacniają sygnały sterujące
// * Cewki wzbudzenia: regulują natężenie pola magnetycznego, wpływając na napięcie generowane

// Parametry automatycznej optymalizacji
const unsigned long OPTIMIZATION_INTERVAL = 60000; // Optymalizacja co minutę
const unsigned long TEST_DURATION = 10000; // Test nowych parametrów przez 10 sekund
unsigned long lastOptimizationTime = 0;

// Zmienne dla optymalizacji bayesowskiej
BayesOptimizer optimizer;
float params[3] = {0.1, 0.9, 0.1}; // Parametry do optymalizacji
float bounds[3][2] = {{0.01, 0.5}, {0.8, 0.99}, {0.01, 0.3}}; // Zakresy wartości parametrów
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

// Tablica Q-learning
float qTable[NUM_STATE_BINS_ERROR * NUM_STATE_BINS_LOAD * NUM_STATE_BINS_KP * NUM_STATE_BINS_KI * NUM_STATE_BINS_KD][NUM_ACTIONS][3]; // 3 wyjścia dla prądów bazowych

// Funkcja dyskretyzacji stanu
int discretizeState(float error, float generatorLoad, float Kp, float Ki, float Kd) {
    int errorBin = constrain((int)(abs(error) / (VOLTAGE_SETPOINT / NUM_STATE_BINS_ERROR)), 0, NUM_STATE_BINS_ERROR - 1);
    int loadBin = constrain((int)(generatorLoad / (LOAD_THRESHOLD / NUM_STATE_BINS_LOAD)), 0, NUM_STATE_BINS_LOAD - 1);
    int kpBin = constrain((int)(Kp / (5.0 / NUM_STATE_BINS_KP)), 0, NUM_STATE_BINS_KP - 1);
    int kiBin = constrain((int)(Ki / (1.0 / NUM_STATE_BINS_KI)), 0, NUM_STATE_BINS_KI - 1);
    int kdBin = constrain((int)(Kd / (5.0 / NUM_STATE_BINS_KD)), 0, NUM_STATE_BINS_KD - 1);

    return errorBin + NUM_STATE_BINS_ERROR * (loadBin + NUM_STATE_BINS_LOAD * (kpBin + NUM_STATE_BINS_KP * (kiBin + NUM_STATE_BINS_KI * kdBin)));
}

// Funkcja wyboru akcji (epsilon-greedy)
int chooseAction(int state) {
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

// Funkcja obliczania nagrody
float calculateReward(float error) {
    return 1.0 / (1.0 + abs(error));
}

// Funkcja aktualizacji Q
void updateQ(int state, int action, float reward, int nextState) {
    float maxFutureQ = 0;
    for (int nextAction = 0; nextAction < NUM_ACTIONS; nextAction++) {
        maxFutureQ = max(maxFutureQ, qTable[nextState][nextAction][0]);
    }
    qTable[state][action][0] += learningRate * (reward + discountFactor * maxFutureQ - qTable[state][action][0]);
}

// Funkcja odczytu prądu cewek wzbudzenia - Zakładamy pomiar pośredni przez tranzystory sterujące
float readExcitationCoilCurrent(int bjtPin) {
    // Odczytujemy wartość PWM z pinu tranzystora BJT
    int pwmValue = analogRead(bjtPin);

    // Zakładamy liniową zależność między PWM a prądem bazy
    float current = pwmValue * (5.0 / 255.0); // Przykład przeliczania - dostosuj do swoich potrzeb
    return current;
}

// Funkcja dostrajania Zieglera-Nicholsa
void performZieglerNicholsTuning() {
    bool isTuning = true;
    unsigned long startTime = millis();
    int oscillationCount = 0;
    unsigned long lastPeakTime = 0;
    bool isIncreasing = true;

    // Reset parametrów regulatora PID
    Kp = 0;
    Ki = 0;
    Kd = 0;

    // Zwiększaj Kp, aż do wystąpienia stabilnych oscylacji
    while (oscillationCount < 5 && millis() - startTime < 30000) {
        Kp += 0.1;
        float pidOutput = calculatePID(VOLTAGE_SETPOINT, voltageIn[0]);
        controlExcitationCoils(pidOutput);
        delay(100);

        // Wykrywanie oscylacji
        if (abs(voltageIn[0] - previousError) > VOLTAGE_REGULATION_HYSTERESIS) {
            if ((voltageIn[0] > previousError && !isIncreasing) ||
                (voltageIn[0] < previousError && isIncreasing)) {
                if (millis() - lastPeakTime > 500) {
                    oscillationCount++;
                    lastPeakTime = millis();
                    if (oscillationCount == 1) {
                        startTime = millis();
                    } else if (oscillationCount == 5) {
                        float Tu = (millis() - startTime) / 4.0 / 1000.0;
                        Kp = 0.45 * Kp;
                        Ki = 1.2 * Kp / Tu;
                        Kd = 0;
                    }
                }
                isIncreasing = !isIncreasing;
            }
        }
        previousError = voltageIn[0];
    }

    if (oscillationCount < 5) {
        Serial.println("Błąd: Nie wykryto oscylacji podczas dostrajania Zieglera-Nicholsa.");
    }

    isTuning = false;
    Serial.print("Kp = ");
    Serial.print(Kp);
    Serial.print(", Ki = ");
    Serial.print(Ki);
    Serial.print(", Kd = ");
    Serial.println(Kd);
}

// Funkcja sterowania tranzystorami
void controlTransistors(float voltage, float excitationCurrent) {
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

void loop() {
    server.handleClient();

    // Odczyt wartości z czujników
    readSensors();

    float externalVoltage = analogRead(PIN_EXTERNAL_VOLTAGE_SENSOR_1) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
    float externalCurrent = analogRead(PIN_EXTERNAL_CURRENT_SENSOR_1) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);

    logData();

    checkAlarm();

    autoCalibrate();

    energyManagement();

    float efficiency = calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);

    int state = discretizeState(VOLTAGE_SETPOINT - voltageIn[0], currentIn[0], Kp, Ki, Kd);
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