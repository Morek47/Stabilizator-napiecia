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
const int excitationBJT1Pin = D8;
const int excitationBJT2Pin = D9;

// Stałe konfiguracyjne
float LOAD_THRESHOLD = 0.5;
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
const int NUM_STATE_BINS_ERROR = 5;
const int NUM_STATE_BINS_LOAD = 3;
const int NUM_STATE_BINS_KP = 5;
const int NUM_STATE_BINS_KI = 3;
const int NUM_STATE_BINS_KD = 3;

const int NUM_ACTIONS = 4;
const float epsilon = 0.1;
const float learningRate = 0.1;
const float discountFactor = 0.9;

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
float qTable[NUM_STATE_BINS_ERROR * NUM_STATE_BINS_LOAD * NUM_STATE_BINS_KP * NUM_STATE_BINS_KI * NUM_STATE_BINS_KD][NUM_ACTIONS][3];

// Stałe dla minimalnych i maksymalnych wartości zmiennych stanu
const float MIN_ERROR = -230.0;
const float MAX_ERROR = 230.0;
const float MIN_LOAD = 0.0;
const float MAX_LOAD = LOAD_THRESHOLD * 2;
const float MIN_KP = 0.0;
const float MAX_KP = 5.0;
const float MIN_KI = 0.0;
const float MAX_KI = 1.0;
const float MIN_KD = 0.0;
const float MAX_KD = 5.0;

// Funkcja dyskretyzacji stanu
int discretizeState(float error, float generatorLoad, float Kp, float Ki, float Kd) {
    float normalizedError = (error - MIN_ERROR) / (MAX_ERROR - MIN_ERROR);
    float normalizedLoad = (generatorLoad - MIN_LOAD) / (MAX_LOAD - MIN_LOAD);
    float normalizedKp = (Kp - MIN_KP) / (MAX_KP - MIN_KP);
    float normalizedKi = (Ki - MIN_KI) / (MAX_KI - MIN_KI);
    float normalizedKd = (Kd - MIN_KD) / (MAX_KD - MIN_KD);

    int errorBin = constrain((int)(normalizedError * NUM_STATE_BINS_ERROR), 0, NUM_STATE_BINS_ERROR - 1);
    int loadBin = constrain((int)(normalizedLoad * NUM_STATE_BINS_LOAD), 0, NUM_STATE_BINS_LOAD - 1);
    int kpBin = constrain((int)(normalizedKp * NUM_STATE_BINS_KP), 0, NUM_STATE_BINS_KP - 1);
    int kiBin = constrain((int)(normalizedKi * NUM_STATE_BINS_KI), 0, NUM_STATE_BINS_KI - 1);
    int kdBin = constrain((int)(normalizedKd * NUM_STATE_BINS_KD), 0, NUM_STATE_BINS_KD - 1);

    return errorBin + NUM_STATE_BINS_ERROR * (loadBin + NUM_STATE_BINS_LOAD * (kpBin + NUM_STATE_BINS_KP * (kiBin + NUM_STATE_BINS_KI * kdBin)));
}

// Funkcja wybierająca akcję na podstawie stanu
int chooseAction(int state) {
    if (random(0, 100) < epsilon * 100) {
        return random(0, NUM_ACTIONS);
    } else {
        int bestAction = 0;
        float bestQValue = qTable[state][0][0];
        for (int a = 1; a < NUM_ACTIONS; a++) {
            if (qTable[state][a][0] > bestQValue) {
                bestAction = a;
                bestQValue = qTable[state][a][0];
            }
        }
        return bestAction;
    }
}

// Funkcja wykonująca akcję
void executeAction(int action) {
    switch (action) {
        case 0:
            // Akcja 1
            break;
        case 1:
            // Akcja 2
            break;
        case 2:
            if (currentIn[0] > LOAD_THRESHOLD) {
                digitalWrite(mosfetPin, HIGH);
            } else {
                digitalWrite(mosfetPin, LOW);
            }
            break;
        case 3:
            float excitationAdjustment = map(currentIn[0], 0, MAX_EXCITATION_CURRENT, MIN_VOLTAGE, MAX_VOLTAGE);
            analogWrite(excitationBJT1Pin, excitationAdjustment);
            analogWrite(excitationBJT2Pin, excitationAdjustment);
            break;
        default:
            break;
    }
}

// Funkcja obliczająca nagrodę
float calculateReward(float error, float efficiency, float voltageDrop) {
    float reward = 1.0 / abs(error);
    reward -= voltageDrop * 0.01;
    return reward;
}

// Funkcja obliczająca wydajność
float calculateEfficiency(float voltageIn, float currentIn, float externalVoltage, float externalCurrent) {
    float inputPower = voltageIn * currentIn;
    float outputPower = externalVoltage * externalCurrent;

    if (inputPower == 0) {
        return 0;
    }

    return outputPower / inputPower;
}

// Zaktualizowana funkcja updateQ
void updateQ(int state, int action, float reward, int nextState) {
    float maxQNextState = qTable[nextState][0][0];
    for (int a = 1; a < NUM_ACTIONS; a++) {
        if (qTable[nextState][a][0] > maxQNextState) {
            maxQNextState = qTable[nextState][a][0];
        }
    }

    qTable[state][action][0] += learningRate * (reward + discountFactor * maxQNextState - qTable[state][action][0]);
}

// Nowa funkcja monitorująca wydajność i dostosowująca sterowanie
void monitorPerformanceAndAdjust() {
    static unsigned long lastAdjustmentTime = 0;
    if (millis() - lastAdjustmentTime > 1000) {
        float efficiency = calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);
        float efficiencyPercent = efficiency * 100.0;

        advancedLogData(efficiencyPercent);

        if (efficiencyPercent < 90.0) {
            excitationGain += 0.1;
        } else if (efficiencyPercent > 95.0) {
            excitationGain -= 0.1;
        }

        excitationGain = constrain(excitationGain, 0.0, 1.0);

        lastAdjustmentTime = millis();
    }
}

// Funkcja sterowania tranzystorami
void controlTransistors(float voltage, float excitationCurrent) {
    voltage = constrain(voltage, MIN_VOLTAGE, MAX_VOLTAGE);

    if (excitationCurrent > LOAD_THRESHOLD) {
        digitalWrite(mosfetPin, HIGH);
    } else {
        digitalWrite(mosfetPin, LOW);
    }

    float baseCurrent1 = excitationCurrent * 0.3;
    float baseCurrent2 = excitationCurrent * 0.3;
    float baseCurrent3 = excitationCurrent * 0.4;

    int pwmValueBJT1 = map(baseCurrent1, 0, MAX_EXCITATION_CURRENT, 0, 255);
    int pwmValueBJT2 = map(baseCurrent2, 0, MAX_EXCITATION_CURRENT, 0, 255);
    int pwmValueBJT3 = map(baseCurrent3, 0, MAX_EXCITATION_CURRENT, 0, 255);

    analogWrite(bjtPin1, pwmValueBJT1);
    analogWrite(bjtPin2, pwmValueBJT2);
    analogWrite(bjtPin3, pwmValueBJT3);
}

// Funkcja monitorująca tranzystory
void monitorTransistors() {
    const int tranzystorPins[3] = {bjtPin1, bjtPin2, bjtPin3};
    for (int i = 0; i < 3; i++) {
        int state = digitalRead(tranzystorPins[i]);
        if (state == LOW) {
            Serial.print("Tranzystor ");
            Serial.print(i);
            Serial.println(" jest wyłączony.");
        } else {
            Serial.print("Tranzystor ");
            Serial.print(i);
            Serial.println(" jest włączony.");
        }
    }
}

// Funkcja regulująca częstotliwość sterowania
int controlFrequency = 0;
const int HIGH_FREQUENCY = 1000;
const int LOW_FREQUENCY = 100;

void adjustControlFrequency() {
    static unsigned long lastAdjustmentTime = 0;
    if (millis() - lastAdjustmentTime > 1000) {
        if (currentIn[0] > LOAD_THRESHOLD) {
            controlFrequency = HIGH_FREQUENCY;
        } else {
            controlFrequency = LOW_FREQUENCY;
        }
        lastAdjustmentTime = millis();
    }
}

// Funkcja logująca zaawansowane dane
void advancedLogData(float efficiencyPercent) {
    Serial.print("Napięcie: ");
    Serial.print(voltageIn[0]);
    Serial.print(" V, Prąd: ");
    Serial.print(currentIn[0]);
    Serial.print(" A, Wydajność: ");
    Serial.print(efficiencyPercent);
    Serial.println(" %");
    
    float power = voltageIn[0] * currentIn[0];
    Serial.print("Moc: ");
    Serial.print(power);
    Serial.println(" W");
}

// Funkcja wspinania się na wzniesienie do optymalizacji progu
float hillClimbing(float currentThreshold, float stepSize, float(*evaluate)(float)) {
    float currentScore = evaluate(currentThreshold);
    float bestThreshold = currentThreshold;
    float bestScore = currentScore;
    
    float newThreshold = currentThreshold + stepSize;
    float newScore = evaluate(newThreshold);
    if (newScore > bestScore) {
        bestThreshold = newThreshold;
        bestScore = newScore;
    } else {
        newThreshold = currentThreshold - stepSize;
        newScore = evaluate(newThreshold);
        if (newScore > bestScore) {
            bestThreshold = newThreshold;
            bestScore = newScore;
        }
    }

    return bestThreshold;
}

// Funkcja oceniająca próg
float evaluateThreshold(float threshold) {
    LOAD_THRESHOLD = threshold;
    
    float totalEfficiency = 0;
    unsigned long startTime = millis();
    while (millis() - startTime < TEST_DURATION) {
        totalEfficiency += calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);
    }
    return totalEfficiency / (TEST_DURATION / 100);
}

// Funkcja setup
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

// Funkcja odczytu sensorów
void readSensors() {
    voltageIn[0] = analogRead(muxInputPin) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
    // Dodaj więcej kodu do odczytu innych sensorów, jeśli jest to wymagane
}

// Funkcja sprawdzania alarmów
void checkAlarm() {
    if (voltageIn[0] > VOLTAGE_SETPOINT + VOLTAGE_REGULATION_HYSTERESIS) {
        Serial.println("Alarm: Napięcie przekroczyło górny próg!");
    } else if (voltageIn[0] < VOLTAGE_SETPOINT - VOLTAGE_REGULATION_HYSTERESIS) {
        Serial.println("Alarm: Napięcie spadło poniżej dolnego progu!");
    }
}

// Funkcja automatycznej kalibracji
void autoCalibrate() {
    previousError = 0;
    integral = 0;
    Serial.println("Automatyczna kalibracja zakończona");
}

// Funkcja zarządzania energią
void energyManagement() {
    if (currentIn[0] > LOAD_THRESHOLD) {
        analogWrite(excitationBJT1Pin, 255);
        analogWrite(excitationBJT2Pin, 255);
    } else {
        analogWrite(excitationBJT1Pin, 0);
        analogWrite(excitationBJT2Pin, 0);
    }
}

// Funkcja wyświetlania danych na ekranie
void displayData(float efficiencyPercent) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.print("Napięcie: ");
    display.print(voltageIn[0]);
    display.println(" V");
    display.print("Prąd: ");
    display.print(currentIn[0]);
    display.println(" A");
    display.print("Wydajność: ");
    display.print(efficiencyPercent);
    display.println(" %");
    display.display();
}

void loop() {
  // Odczyt czujników
  readSensors();

  // Optymalizacja bayesowska (co określony interwał czasu)
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

      // Zapis do EEPROM tylko jeśli zmieniła się najlepsza wydajność
      if (averageEfficiency > bestEfficiency) {
          EEPROM.put(0, lastOptimizationTime);

          int qTableSize = sizeof(qTable); 
          for (int i = 0; i < qTableSize; i++) {
              EEPROM.put(i + sizeof(lastOptimizationTime), ((byte*)qTable)[i]);
          }
          EEPROM.commit(); 
          Serial.println("Zapisano tablicę Q-learning i lastOptimizationTime do EEPROM.");
      }

      // Wywołanie Hill Climbing do optymalizacji progu przełączania faz wzbudzenia
      LOAD_THRESHOLD = hillClimbing(LOAD_THRESHOLD, 0.01, evaluateThreshold);
      Serial.print("Zaktualizowano próg przełączania faz wzbudzenia: ");
      Serial.println(LOAD_THRESHOLD);
  }

  // Opóźnienie
  delay(100);

  // Wyświetlanie danych na ekranie
  displayData(efficiencyPercent); 

  // Dostosowanie częstotliwości sterowania
  adjustControlFrequency(); 

  // Monitorowanie tranzystorów
  monitorTransistors(); 

  // Monitorowanie wydajności i dostosowanie sterowania
  monitorPerformanceAndAdjust(); 

  // Wyświetlanie informacji o wolnej pamięci
  Serial.print("Wolna pamięć: ");
  Serial.println(freeMemory()); 

  // Komunikacja z komputerem
  Serial.print(voltageIn[0]);
  Serial.print(",");
  Serial.print(currentIn[0]);
  Serial.print(",");
  Serial.print(ex