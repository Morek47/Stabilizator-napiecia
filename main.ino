<Arduino.h>        najnowsze dzisaj 
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <EEPROM.h>
#include <Wire.h>
#include <Adafruit_SH1106.h>
#include <ArduinoEigen.h>
// Deklaracje funkcji
void setup();
void loop();
void readSensors();
void logData();
void checkAlarm();
void autoCalibrate();
void energyManagement();
void calibrateSensors();
float calculatePID(float setpoint, float measuredValue);
float calculateEfficiency(float voltageIn, float currentIn, float externalVoltage, float externalCurrent);
void updateRLS(float voltageError);
void updateQ(int state, int action, float reward, int nextState);
float calculateReward(float voltageError);
int chooseAction(int state);
void controlExcitationCoils(float controlSignal);
void updateLearningAlgorithm(float voltageError);

// Główne funkcje
void setup() {
    // Ustawienie pinów jako wyjścia dla multipleksera
    pinMode(muxSelectPinA, OUTPUT);
    pinMode(muxSelectPinB, OUTPUT);

    // Ustawienie pinów PWM
    pinMode(PIN_EXCITATION_COIL_1, OUTPUT);
    pinMode(PIN_EXCITATION_COIL_2, OUTPUT);

    server.begin();
    display.begin();
    calibrateSensors();
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
        Kp = bestKp; // Przywrócenie do najlepszego Kp
    }

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
    // Implementacja kalibracji czujników
}

float calculatePID(float setpoint, float measuredValue) {
    // Implementacja obliczeń PID
}

float calculateEfficiency(float voltageIn, float currentIn, float externalVoltage, float externalCurrent) {
    // Implementacja obliczeń wydajności
}

void updateRLS(float voltageError) {
    // Implementacja RLS
}

void updateQ(int state, int action, float reward, int nextState) {
    // Implementacja aktualizacji Q-learning
}

float calculateReward(float voltageError) {
    // Implementacja obliczeń nagrody
}

int chooseAction(int state) {
    // Implementacja wyboru akcji
}

void controlExcitationCoils(float controlSignal) {
    // Implementacja kontroli cewek wzbudzenia
}

void updateLearningAlgorithm(float voltageError) {
    // Implementacja aktualizacji algorytmu uczenia
}
#include <Arduino.h>  // Ensure required header is included
#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <EEPROM.h>
#include <Wire.h>
#include <Adafruit_SH1106.h>
#include <ArduinoEigen.h>

// Definicje pinów dla tranzystorów
const int transistorPin1 = D4;
const int transistorPin2 = D5;
const int transistorPin3 = D6;
const int transistorPin4 = D7;

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

// Parametry adaptacji Kp
float Kp_max = 5.0;
float Kp_min = 1.0;
float Kp_change_rate = 0.01;
float Kp_change_threshold = 0.5;

// Piny dla czujników
const int muxInputPins[NUM_SENSORS] = {A0, A1, A2, A3};

void setup() {
    Serial.begin(115200);

    // Ustawienie pinów jako wyjścia dla multipleksera
    pinMode(muxSelectPinA, OUTPUT);
    pinMode(muxSelectPinB, OUTPUT);

    // Ustawienie pinów PWM
    pinMode(PIN_EXCITATION_COIL_1, OUTPUT);
    pinMode(PIN_EXCITATION_COIL_2, OUTPUT);

    // Inicjalizacja serwera i wyświetlacza
    server.begin();
    display.begin();
    calibrateSensors();
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

    // Opóźnienie 100ms
    delay(100);

    // Aktualizacja algorytmu uczenia maszynowego
    updateLearningAlgorithm(VOLTAGE_SETPOINT - voltageIn[0]);

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
// Definicje pinów dla tranzystorów
const int transistorPin1 = D4;
const int transistorPin2 = D5;
const int transistorPin3 = D6;
const int transistorPin4 = D7;

// Stałe konfiguracyjne
const float LOAD_THRESHOLD = 0.5; // Example threshold, adjust as needed
const float COMPENSATION_FACTOR = 0.1; // Example factor, adjust as needed
const int MAX_EXCITATION_CURRENT = 255; // Example max current, adjust as needed

void loop() {
    // ... (pozostała część kodu)

    // Odczyt wartości z czujników
    readSensors();
    float generatorLoad = analogRead(A3); // Example sensor read, adjust as needed

    // Inteligentne sterowanie cewką wzbudzenia
    // ... (implementacja algorytmu uczenia maszynowego)

    // Uwzględnij obciążenie prądnicy w algorytmie
    float excitationCurrent = 0; // Initialize excitation current
    if (generatorLoad > LOAD_THRESHOLD) {
        excitationCurrent = constrain(excitationCurrent + COMPENSATION_FACTOR * generatorLoad, 0, MAX_EXCITATION_CURRENT);
    }

    // Sterowanie 4 tranzystorami za pomocą PWM
    int pwmValue = map(excitationCurrent, 0, MAX_EXCITATION_CURRENT, 0, 255);

    // Ustaw wartość PWM na wszystkich pinach tranzystorów
    analogWrite(transistorPin1, pwmValue);
    analogWrite(transistorPin2, pwmValue);
    analogWrite(transistorPin3, pwmValue);
    analogWrite(transistorPin4, pwmValue);

    // ... (pozostała część kodu)
}

// Definicje pinów
const int muxSelectPinA = D2; // Pin A do wyboru multipleksera
const int muxSelectPinB = D3; // Pin B do wyboru multipleksera
const int muxInputPin = A0;    // Pin dla odczytu z multipleksera
const int PIN_EXCITATION_COIL_1 = D0; // Pin PWM dla cewki wzbudzenia 1 
const int PIN_EXCITATION_COIL_2 = D1; // Pin PWM dla cewki wzbudzenia 2



// Piny do tranzystorów
const int transistorPins[4] = {D4, D5, D6, D7}; // Piny dla tranzystorów



// Liczba czujników
const int NUM_SENSORS = 4; // Liczba czujników



// Stałe
const float VOLTAGE_REFERENCE = 3.3; // Napięcie referencyjne dla przetwornika ADC (w woltach)
const int ADC_MAX_VALUE = 1023; // Maksymalna wartość odczytu z przetwornika ADC
float VOLTAGE_SETPOINT = 230.0; // Ustawione napięcie docelowe na 230V
const float VOLTAGE_REGULATION_HYSTERESIS = 0.1; // Histereza regulacji napięcia (w woltach)



// Parametry PID
float Kp = 2.0, Ki = 0.5, Kd = 1.0; // Ustawiony Kp na 2.0
float previousError = 0;
float integral = 0;



// Zmienne globalne
float voltageIn[2] = {0}; // Zmienne dla czujników napięcia
float currentIn[2] = {0}; // Zmienne dla czujników prądu



// Dodanie zmiennych globalnych
float bestKp = Kp; // Najlepsza wartość Kp
float bestEfficiency = 0.0; // Najlepsza wydajność



ESP8266WebServer server(80);



// RLS Variables
float theta[3] = {Kp, Ki, Kd}; // Współczynniki RLS
float P[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}; // Macierz kowariancji
float lambda = 0.99; // Współczynnik zapomnienia



// Historia danych
const int HISTORY_SIZE = 100;
float voltageHistory[HISTORY_SIZE];
float currentHistory[HISTORY_SIZE];
int historyIndex = 0;



// Q-learning
const int NUM_STATES = 10; // Liczba stanów (przedziały błędu napięcia)
const int NUM_ACTIONS = 5; // Liczba akcji (wartości Kp)
float Q[NUM_STATES][NUM_ACTIONS] = {0}; // Tablica Q
float alpha = 0.1; // Współczynnik uczenia
float gamma = 0.9; // Współczynnik dyskontowy
float epsilon = 0.1; // Prawdopodobieństwo eksploracji
int lastAction = 0; // Ostatnia akcja
float epsilonDecay = 0.99; // Współczynnik zmniejszania epsilon
int episodeCount = 0; // Liczba epizodów



// Tryb pracy (0 = PID, 1 = RLS)
int mode = 0;



// Inicjalizacja wyświetlacza OLED
Adafruit_SH1106 display(128, 64, &Wire, -1); // Inicjalizacja, zmień -1 na pin resetu, jeśli potrzebny



// Piny dla zewnętrznych czujników napięcia i prądu
const int PIN_EXTERNAL_VOLTAGE_SENSOR = A1;
const int PIN_EXTERNAL_CURRENT_SENSOR = A2;



// Parametry adaptacji Kp
float Kp_max = 5.0; // Maksymalna wartość Kp
float Kp_min = 1.0; // Minimalna wartość Kp
float Kp_change_rate = 0.01; // Szybkość zmiany Kp
float Kp_change_threshold = 0.5; // Próg dla zmiany Kp



// Piny dla czujników
const int muxInputPins[NUM_SENSORS] = {A0, A1, A2, A3}; // Piny analogowe dla czujników



// Funkcja wyboru akcji (Q-learning)
int chooseAction(int state) {
  float maxQ = Q[state][0];
  int bestAction = 0;
  if (random(100) < epsilon * 100) { // Eksploracja
    bestAction = random(NUM_ACTIONS);
  } else { // Eksploatacja
    for (int i = 1; i < NUM_ACTIONS; i++) {
      if (Q[state][i] > maxQ) {
        maxQ = Q[state][i];
        bestAction = i;
      }
    }
  }
  return bestAction;
}



// Funkcja kalibracji czujników
void calibrateSensors() {
    Serial.println("Kalibracja czujników...");
    for (int i = 0; i < NUM_SENSORS; i++) {
        float voltageSum = 0;
        float currentSum = 0;



        for (int j = 0; j < 10; j++) {
            // Ustawienie multipleksera dla odpowiedniego czujnika
            digitalWrite(muxSelectPinA, i & 0x01); // Ustawienie LSB
            digitalWrite(muxSelectPinB, (i >> 1) & 0x01); // Ustawienie MSB



            voltageSum += analogRead(muxInputPin);
            currentSum += analogRead(muxInputPin);
            delay(100);
        }



        // Ustalanie offsetu
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



        // Zapisanie offsetu do zmiennych globalnych
        voltageIn[i] -= voltageOffset;
        currentIn[i] -= currentOffset;
    }
}



// Funkcja obliczania sygnału kontrolnego PID
float calculatePID(float setpoint, float measuredValue) {
    float error = setpoint - measuredValue;
    integral += error;
    float derivative = error - previousError;
    previousError = error;



    // Obliczanie sygnału kontrolnego
    float controlSignal = Kp * error + Ki * integral + Kd * derivative;
    return controlSignal;
}



// Funkcja obliczania wydajności
float calculateEfficiency(float voltageIn, float currentIn,
                          float externalVoltage, float externalCurrent) {
    // Oblicz miarę wydajności na podstawie odczytów z czujników
    float outputPower = externalVoltage * externalCurrent;
    float inputPower = voltageIn[0] * currentIn[0] + voltageIn[1] * currentIn[1]; // Suma mocy z obu czujników prądowych
    return outputPower / inputPower;
}



// Funkcja aktualizacji parametrów RLS
void updateRLS(float voltageError) {
    // Przykładowe pomiary dla RLS
    float z[3] = {voltageError, voltageError * voltageError, voltageError * voltageError * voltageError}; // Wartości z
    float y = voltageError; // Wynik



    // Przewidywanie
    float prediction = theta[0] * voltageError + theta[1] * voltageError * voltageError + theta[2] * voltageError * voltageError * voltageError;



    // Obliczenie błędu
    float error = y - prediction;



    // Aktualizacja współczynników RLS
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            P[i][j] = (1/lambda) * (P[i][j] - (P[i][j] * z[i] * z[j] * P[i][j]) / (1 + z[j] * P[j][j]));
        }
        theta[i] += P[i][0] * error; // Aktualizacja theta
    }
}



// Funkcja Q-learning
void updateQ(int state, int action, float reward, int nextState) {
    float maxQNext = Q[nextState][0];
    for (int i = 1; i < NUM_ACTIONS; i++) {
        if (Q[nextState][i] > maxQNext) {
            maxQNext = Q[nextState][i];
        }
    }
    Q[state][action] += alpha * (reward + gamma * maxQNext - Q[state][action]);
}



// Funkcja obliczania nagrody
float calculateReward(float voltageError) {
    if (abs(voltageError) < 1.0) {
        return 1.0; // Stabilny system
    } else if (abs(voltageError) < 2.0) {
        return 0.5; // Prawie stabilny
    } else {
        return -1.0; // Niestabilny system
    }
}



// Przeprowadzenie testów i optymalizacji
void testingAndOptimization() {
    Serial.println("Rozpoczęcie testów i optymalizacji...");



    // Testowanie w różnych warunkach
    for (int i = 0; i < 10; i++) {
        float testSetpoint = random(200, 250); // Losowe napięcie docelowe
        VOLTAGE_SETPOINT = testSetpoint;
        Serial.print("Testowane napięcie: ");
        Serial.println(testSetpoint);



        // Stabilizacja systemu
        delay(1000);



        // Odczyt wartości z czujników
        voltageIn[0] = analogRead(muxInputPin) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
        currentIn[0] = analogRead(muxInputPin) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
        float externalVoltage = analogRead(PIN_EXTERNAL_VOLTAGE_SENSOR) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
        float externalCurrent = analogRead(PIN_EXTERNAL_CURRENT_SENSOR) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);



        // Obliczanie wydajności
        float efficiency = calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);
        Serial.print("Wydajność: ");
        Serial.println(efficiency);



        // Aktualizacja Q-learning na podstawie nagród
        int currentState = (int)(abs(VOLTAGE_SETPOINT - voltageIn[0]) / 2);
        currentState = constrain(currentState, 0, NUM_STATES - 1);
        int action = lastAction; // Zachowaj ostatnią akcję
        float reward = calculateReward(VOLTAGE_SETPOINT - voltageIn[0]);
        int nextState = currentState; // Prosta strategia: utrzymujemy ten sam stan
        updateQ(currentState, action, reward, nextState);
    }



    Serial.println("Testy i optymalizacja zakończone.");
}



void setup() {
    // Ustawienie pinów jako wyjścia dla multipleksera
    pinMode(muxSelectPinA, OUTPUT);
    pinMode(muxSelectPinB, OUTPUT);



    // Ustawienie pinów PWM
    pinMode(PIN_EXCITATION_COIL_1, OUTPUT); // Zmień na OUTPUT
    pinMode(PIN_EXCITATION_COIL_2, OUTPUT); // Zmień na OUTPUT



    server.begin();
    display.begin();
    calibrateSensors();
}



void readSensors() {
    for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
        // Ustawienie multipleksera dla odpowiedniego czujnika
        digitalWrite(muxSelectPinA, sensor & 0x01); // Ustawienie LSB
        digitalWrite(muxSelectPinB, (sensor >> 1) & 0x01); // Ustawienie MSB



        // Odczyt wartości z czujnika i konwersja
        if (sensor < 2) { // Czujniki prądu ACS712-5A
            float Vcc = 5.0; // Napięcie zasilania czujnika
            float Sensitivity = 0.066; // Czułość czujnika w V/A
            float Vout = analogRead(muxInputPins[sensor]) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE); 
            currentIn[sensor] = (Vout - (Vcc / 2.0)) / Sensitivity; 
        } else { // Czujniki napięcia z dzielnikiem 100:1
            float multiplier = 100.0; // Mnożnik wynikający z dzielnika napięcia
            voltageIn[sensor - 2] = analogRead(muxInputPins[sensor]) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE) * multiplier; 
        }
    }
}



void controlExcitationCoils(float controlSignal) {
    // Ograniczenie sygnału kontrolnego do zakresu 0-255
    int pwmValue = constrain(controlSignal, 0, 255);
    analogWrite(PIN_EXCITATION_COIL_1, pwmValue); // Użyj analogWrite zamiast analogWrite
    analogWrite(PIN_EXCITATION_COIL_2, pwmValue); // Użyj analogWrite zamiast analogWrite
}



void updateLearningAlgorithm(float voltageError) {
    // Obliczanie stanu na podstawie błędu napięcia
    int state = (int)(abs(voltageError) / 2);
    state = constrain(state, 0, NUM_STATES - 1);



    // Uaktualnienie Q-learning
    float reward = calculateReward(voltageError);
    updateQ(state, lastAction, reward, state);
}



void loop() {
    server.handleClient();



    // Odczyt wartości z czujników za pomocą multipleksera
    readSensors();



    // Odczyt napięcia i prądu z dodatkowych zewnętrznych czujników
    float externalVoltage = analogRead(PIN_EXTERNAL_VOLTAGE_SENSOR) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
    float externalCurrent = analogRead(PIN_EXTERNAL_CURRENT_SENSOR) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);



    // Logowanie danych
    logData(); // Funkcja logowania danych



    // Sprawdzenie alarmów
    checkAlarm(); // Funkcja sprawdzająca alarmy



    // Automatyczna kalibracja
    autoCalibrate(); // Funkcja automatycznej kalibracji



    // Zarządzanie energią
    energyManagement(); // Funkcja zarządzania energią



    // Obliczanie wydajności
    float efficiency = calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);



    // Monitorowanie wydajności i aktualizacja Kp
    if (efficiency > bestEfficiency) {
        bestEfficiency = efficiency;
        bestKp = Kp;
    } else {
        Kp = bestKp; // Przywrócenie do najlepszego Kp
    }



    // Adaptacja Kp (algorytm adaptacyjny PID)
    float error = VOLTAGE_SETPOINT - voltageIn[0];
    if (abs(error) > Kp_change_threshold) { // Zmień Kp, jeśli błąd jest duży
      Kp += Kp_change_rate * error;
      Kp = constrain(Kp, Kp_min, Kp_max); // Ogranicz Kp do dozwolonego zakresu
    }



    // Q-learning
    // 1. Obliczanie stanu
    int state = (int)(abs(VOLTAGE_SETPOINT - voltageIn[0]) / 2);
    state = constrain(state, 0, NUM_STATES - 1); 



    // 2. Wybór akcji (Kp)
    int action = chooseAction(state);
    Kp = action * 0.5 + 1.0; // Przypisanie wartości Kp na podstawie akcji



    // Uwzględnienie odczytów z czujników w obliczeniach PID
    float pidOutput = calculatePID(VOLTAGE_SETPOINT, voltageIn[0]);
    controlExcitationCoils(pidOutput); // Kontrola cewek wzbudzenia



    // 4. Obliczenie nagrody
    float reward = calculateReward(VOLTAGE_SETPOINT - voltageIn[0]);



    // 5. Aktualizacja Q-learning
    updateQ(state, lastAction, reward, state); 



    // Opóźnienie 100ms
    delay(100);



    // Aktualizacja algorytmu uczenia maszynowego
    updateLearningAlgorithm(VOLTAGE_SETPOINT - voltageIn[0]);


// Dodane funkcje
void logData() {
    // Logowanie danych do historii
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
    // Sprawdzenie alarmów
    if (voltageIn[0] < 220 || voltageIn[0] > 240) {
        Serial.println("Alarm: Voltage out of range!");
    }
    if (currentIn[0] > 5.0) {
        Serial.println("Alarm: Current too high!");
    }
}

void autoCalibrate() {
    // Automatyczna kalibracja
    static unsigned long lastCalibrationTime = 0;
    if (millis() - lastCalibrationTime > 60000) { // Co 60 sekund
        calibrateSensors();
        lastCalibrationTime = millis();
        Serial.println("Auto-calibration completed.");
    }
}

void energyManagement() {
    // Zarządzanie energią
    float totalPower = voltageIn[0] * currentIn[0];
    if (totalPower > 1000) {
        Serial.println("Energy Management: High power consumption detected!");
    }
}

void loop() {
    server.handleClient();

    // Odczyt wartości z czujników za pomocą multipleksera
    readSensors();

    // Odczyt napięcia i prądu z dodatkowych zewnętrznych czujników
    float externalVoltage = analogRead(PIN_EXTERNAL_VOLTAGE_SENSOR) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
    float externalCurrent = analogRead(PIN_EXTERNAL_CURRENT_SENSOR) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);

    // Logowanie danych
    logData(); // Funkcja logowania danych

    // Sprawdzenie alarmów
    checkAlarm(); // Funkcja sprawdzająca alarmy

    // Automatyczna kalibracja
    autoCalibrate(); // Funkcja automatycznej kalibracji

    // Zarządzanie energią
    energyManagement(); // Funkcja zarządzania energią

    // Obliczanie wydajności
    float efficiency = calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);

    // Monitorowanie wydajności i aktualizacja Kp
    if (efficiency > bestEfficiency) {
        bestEfficiency = efficiency;
        bestKp = Kp;
    } else {
        Kp = bestKp; // Przywrócenie do najlepszego Kp
    }

    // Adaptacja Kp (algorytm adaptacyjny PID)
    float error = VOLTAGE_SETPOINT - voltageIn[0];
    if (abs(error) > Kp_change_threshold) { // Zmień Kp, jeśli błąd jest duży
      Kp += Kp_change_rate * error;
      Kp = constrain(Kp, Kp_min, Kp_max); // Ogranicz Kp do dozwolonego zakresu
    }

    // Q-learning
    // 1. Obliczanie stanu
    int state = (int)(abs(VOLTAGE_SETPOINT - voltageIn[0]) / 2);
    state = constrain(state, 0, NUM_STATES - 1); 

    // 2. Wybór akcji (Kp)
    int action = chooseAction(state);
    Kp = action * 0.5 + 1.0; // Przypisanie wartości Kp na podstawie akcji

    // Uwzględnienie odczytów z czujników w obliczeniach PID
    float pidOutput = calculatePID(VOLTAGE_SETPOINT, voltageIn[0]);
    controlExcitationCoils(pidOutput); // Kontrola cewek wzbudzenia

    // 4. Obliczenie nagrody
    float reward = calculateReward(VOLTAGE_SETPOINT - voltageIn[0]);

    // 5. Aktualizacja Q-learning
    updateQ(state, lastAction, reward, state); 

    // Opóźnienie 100ms
    delay(100);

    // Aktualizacja algorytmu uczenia maszynowego
    updateLearningAlgorithm(VOLTAGE_SETPOINT - voltageIn[0]);

    // Wyświetlanie danych na OLED
    display.clearDisplay();
    display.setCursor(0, 0);
    display.print("Napięcie: ");
    display.print(voltageIn[0]);
    display.println(" V");
    display.print("Prąd: ");
    display.print(currentIn[0]);
    display.println(" A");
    display.display(); // Wyświetl dane
}
   



