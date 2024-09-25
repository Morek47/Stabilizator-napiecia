
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Arduino.h>
#include <EEPROM.h>
#include <Wire.h>
#include <Adafruit_SH1106.h>
#include <ArduinoEigen.h>
#include <BayesOptimizer.h>

// Definicje stałych i zmiennych
#define MAX_CURRENT 25 // Maksymalny prąd wzbudzenia w amperach
#define TOLERANCE 0.5 // Margines tolerancji
#define NUM_STATES_AGENT2 100 // Definicja liczby stanów dla Agenta 2
#define MAX_BRAKING_EFFECT 100.0 // Przykładowa maksymalna wartość efektu hamowania
#define VOLTAGE_SETPOINT 230.0 // Docelowe napięcie w woltach
#define PWM_INCREMENT 5 // Przykładowa wartość przyrostu P
#define OLED_RESET -1 // Jeśli nie używasz pinu resetu, ustaw na -1
Adafruit_SH1106 display(OLED_RESET);

// Funkcja zapisu do EEPROM
void writeEEPROM(int address, byte value) {
    if (EEPROM.read(address) != value) {
        EEPROM.write(address, value);
        EEPROM.commit();
    }

// Zmienne globalne dla parametrów PID
float Kp = 1.0; // Przykładowa wartość początkowa
float Ki = 0.1; // Przykładowa wartość początkowa
float Kd = 0.01; // Przykładowa wartość początkowa

// Definicje pinów dla tranzystorów
const int mosfetPin = D4;
const int excitationBJT1Pin = D8;
const int excitationBJT2Pin = D9;
const int bjtPin1 = D5;
const int bjtPin2 = D6;
const int bjtPin3 = D7;

// Definicje pinów dla multipleksera
const int muxSelectPinA = D0;
const int muxSelectPinB = D1;
const int muxSelectPinC = D2;
const int muxSelectPinD = D3; // Dodatkowy pin selekcji dla 16-kanałowego multipleksera
const int muxInputPin = A0; // Wejście multipleksera podłączone do jedynego pinu analogowego

// Definicje pinów dla nowych tranzystorów
const int newPin1 = D10;
const int newPin2 = D11;
const int newPin3 = D12;
const int bjtPin4 = D13;

// Definicje zmiennych dla testowania parametrów
float testEpsilon = 0.2;
float testLearningRate = 0.05;
float testDiscountFactor = 0.95;

// Definicje stałych i zmiennych
const float VOLTAGE_REFERENCE = 5.0;
const int ADC_MAX_VALUE = 1023;
const float VOLTAGE_REGULATION_HYSTERESIS = 0.1;
const int INITIAL_BRAKE_PWM = 128;
const float MIN_ERROR = -5.0;
const float MAX_ERROR = 5.0;
const float MIN_LOAD = 0.0;
const float MAX_LOAD = 1.0;
const float MIN_KP = 0.0;
const float MAX_KP = 5.0;
const float MIN_KI = 0.0;
const float MAX_KI = 1.0;
const float MIN_KD = 0.0;
const float MAX_KD = 2.0;
const int HIGH_FREQUENCY = 1000;
const int LOW_FREQUENCY = 100;
const int controlFrequency = 50;
bool stopCompute = false;
HardwareSerial mySerial(1);
float evaluateThreshold = 0.1;

// Zmienna globalna do przechowywania poprzedniej wartości sygnału
float previousFilteredValue = 0.0; // Inicjalizacja na 0.0

// Deklaracja semaforów
SemaphoreHandle_t xSemaphore;

// Prototypy funkcji
float simulateVoltageControl(float Kp, float Ki, float Kd);
float simulateExcitationControl();
float someFunctionOfOtherParameters(float rotationalSpeed, float torque, float frictionCoefficient);
void hillClimbing();
void discretizeStateAgent3(float state[2], int discreteState[2]);
float chooseActionAgent3(int discreteState[2], float epsilon);
void executeActionAgent3(float action);
float calculateRewardAgent3(float state[2], float action);
void updateQAgent3(float state[2], float action, float reward, float nextState[2]);
float objectiveFunction(float params[3]);
void optimize();
void getBestParams(float params[3]);
float getBestObjective();
void suggestNextParameters(float params[3]);
float readVoltage();
float readExcitationCurrent();
float readBrakingEffect();
float lowPassFilter(float currentValue, float previousValue, float alpha);
float calibrateVoltage(float rawVoltage);
float calibrateCurrent(float rawCurrent);
float calibrateBrakingEffect(float rawBrakingEffect);
float simulateBrakingEffect(float rotationalSpeed, float torque, float frictionCoefficient);
void autoTunePID(PID &pid, float setpoint, float measuredValue);
float calculateError(float setpoint, float measuredValue);
void updateControlParameters(float params[3]);
void handleSerialCommunication();
void updateDisplay();
void monitorErrors();
void generateAndImplementFix(String error);
void checkVoltage();
void checkCurrent();
void checkBrakingEffect();
void resetVoltageController();
void reduceExcitationCurrent();
void calibrateBrakingEffect();
void logError(String error);
void readErrorLog();
void detectAnomalies();
void handleAnomaly(String anomaly);
String predictError(float voltage, float current, float brakingEffect);
void addFixFunction(String error, void(*fixFunction)());
void defaultFixFunction();
void trainModel();
String predictWithModel(float voltage, float current, float brakingEffect);

// Funkcja agenta 1
void agent1Function(void *pvParameters) {
    for (;;) {
        if (xSemaphoreTake(xSemaphore, portMAX_DELAY) == pdTRUE) {
            // Kod agenta 1 - stabilizacja napięcia
            float voltage = readVoltage();
            Serial.println("Agent 1: Odczytane napięcie: " + String(voltage));

            xSemaphoreGive(xSemaphore);
        }
        vTaskDelay(100 / portTICK_PERIOD_MS); // Opóźnienie dla symulacji
    }
}

// Funkcja agenta 2
void agent2Function(void *pvParameters) {
    for (;;) {
        if (xSemaphoreTake(xSemaphore, portMAX_DELAY) == pdTRUE) {
            // Kod agenta 2 - wzbudzanie prądu
            float current = readExcitationCurrent();
            Serial.println("Agent 2: Odczytany prąd wzbudzenia: " + String(current));

            xSemaphoreGive(xSemaphore);
        }
        vTaskDelay(100 / portTICK_PERIOD_MS); // Opóźnienie dla symulacji
    }
}

// Funkcja agenta 3
void agent3Function(void *pvParameters) {
    for (;;) {
        if (xSemaphoreTake(xSemaphore, portMAX_DELAY) == pdTRUE) {
            // Kod agenta 3 - kontrola efektu hamowania
            float brakingEffect = readBrakingEffect();
            Serial.println("Agent 3: Odczytany efekt hamowania: " + String(brakingEffect));

            xSemaphoreGive(xSemaphore);
        }
        vTaskDelay(100 / portTICK_PERIOD_MS); // Opóźnienie dla symulacji
    }
}

// Deklaracja globalnych zmiennych dla optymalizatora
BayesOptimizer optimizer;
float bestParams[3] = {0.0, 0.0, 0.0};
float bestObjective = FLT_MAX;

// Funkcja celu dla optymalizatora
float objectiveFunction(float params[3]) {
    float Kp = params[0];
    float Ki = params[1];
    float Kd = params[2];
    float controlSignal = simulateVoltageControl(Kp, Ki, Kd);
    float error = calculateError(VOLTAGE_SETPOINT, controlSignal);
    return abs(error); // Minimalizujemy wartość bezwzględną błędu
}

// Implementacja funkcji optimizePID
void optimizePID() {
    // Ustawienia optymalizatora
    optimizer.setObjectiveFunction(objectiveFunction);
    optimizer.setParameterBounds(0, MIN_KP, MAX_KP);
    optimizer.setParameterBounds(1, MIN_KI, MAX_KI);
    optimizer.setParameterBounds(2, MIN_KD, MAX_KD);

    // Przeprowadzenie optymalizacji
    for (int i = 0; i < 100; i++) { // Przykładowa liczba iteracji
        float params[3];
        optimizer.suggestNextParameters(params);
        float objective = objectiveFunction(params);
        optimizer.update(params, objective);

        // Aktualizacja najlepszych parametrów
        if (objective < bestObjective) {
            bestObjective = objective;
            memcpy(bestParams, params, sizeof(bestParams));
        }
    }

    // Zastosowanie najlepszych parametrów
    updateControlParameters(bestParams);
}

// Mapa funkcji naprawczych
std::map<String, void(*)()> fixFunctions;

// Prosty model uczenia maszynowego (przykład)
float modelWeights[4] = {1.0, 1.0, 1.0, 1.0}; // Wagi modelu

// Implementacja funkcji discretizeStateAgent3
void discretizeStateAgent3(float state[2], int discreteState[2]) {
    // Zakładamy, że stany są w zakresie od 0 do 1
    discreteState[0] = (int)(state[0] * NUM_STATES_AGENT2);
    discreteState[1] = (int)(state[1] * NUM_STATES_AGENT2);
}



// Deklaracja globalnych zmiennych dla optymalizatora
BayesOptimizer optimizer;
float bestParams[3] = {0.0, 0.0, 0.0};
float bestObjective = FLT_MAX;

// Funkcja celu dla optymalizatora
float objectiveFunction(float params[3]) {
    float Kp = params[0];
    float Ki = params[1];
    float Kd = params[2];
    float controlSignal = simulateVoltageControl(Kp, Ki, Kd);
    float error = calculateError(VOLTAGE_SETPOINT, controlSignal);
    return abs(error); // Minimalizujemy wartość bezwzględną błędu
}

// Implementacja funkcji optimizePID
void optimizePID() {
    // Ustawienia optymalizatora
    optimizer.setObjectiveFunction(objectiveFunction);
    optimizer.setParameterBounds(0, MIN_KP, MAX_KP);
    optimizer.setParameterBounds(1, MIN_KI, MAX_KI);
    optimizer.setParameterBounds(2, MIN_KD, MAX_KD);

    // Przeprowadzenie optymalizacji
    for (int i = 0; i < 100; i++) { // Przykładowa liczba iteracji
        float params[3];
        optimizer.suggestNextParameters(params);
        float objective = objectiveFunction(params);
        optimizer.update(params, objective);

        // Aktualizacja najlepszych parametrów
        if (objective < bestObjective) {
            bestObjective = objective;
            memcpy(bestParams, params, sizeof(bestParams));
        }
    }

    // Zastosowanie najlepszych parametrów
    updateControlParameters(bestParams);
}


// Mapa funkcji naprawczych
std::map<String, void(*)()> fixFunctions;

// Prosty model uczenia maszynowego (przykład)
float modelWeights[4] = {1.0, 1.0, 1.0, 1.0}; // Wagi modelu


// Implementacja funkcji discretizeStateAgent3
void discretizeStateAgent3(float state[2], int discreteState[2]) {
    // Zakładamy, że stany są w zakresie od 0 do 1
    discreteState[0] = (int)(state[0] * (NUM_STATES_AGENT3 - 1));
    discreteState[1] = (int)(state[1] * (NUM_STATES_AGENT3 - 1));
}

// Implementacja funkcji simulateVoltageControl
float simulateVoltageControl(float Kp, float Ki, float Kd) {
    // Przykładowa implementacja symulacji kontroli napięcia
    float setpoint = VOLTAGE_SETPOINT;
    float measuredValue = readVoltage();
    float error = calculateError(setpoint, measuredValue);
    float controlSignal = Kp * error + Ki * (error * controlFrequency) + Kd * (error / controlFrequency);
    return controlSignal;
}

// Implementacja funkcji simulateExcitationControl
float simulateExcitationControl() {
    // Przykładowa implementacja symulacji kontroli wzbudzenia
    float excitationCurrent = readExcitationCurrent();
    return excitationCurrent;
}

// Implementacja funkcji simulateBrakingEffect
float simulateBrakingEffect(float rotationalSpeed, float torque, float frictionCoefficient) {
    // Przykładowa implementacja symulacji efektu hamowania
    return (rotationalSpeed * torque) / (frictionCoefficient * 10.0);
}

// Implementacja funkcji objectiveFunction
float objectiveFunction(float params[3]) {
    // Przykładowa implementacja funkcji celu
    float Kp = params[0];
    float Ki = params[1];
    float Kd = params[2];
    float controlSignal = simulateVoltageControl(Kp, Ki, Kd);
    return controlSignal;
}

// Implementacja funkcji calibrateVoltage
float calibrateVoltage(float rawVoltage) {
    // Przykładowa implementacja kalibracji napięcia
    return rawVoltage * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
}

// Implementacja funkcji calibrateCurrent
float calibrateCurrent(float rawCurrent) {
    // Przykładowa implementacja kalibracji prądu
    return rawCurrent * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
}

// Implementacja funkcji autoTunePID
void autoTunePID(PID &pid, float setpoint, float measuredValue) {
    // Przykładowa implementacja automatycznego strojenia PID
    float error = calculateError(setpoint, measuredValue);
    pid.autoTune(error);
}


// Implementacja funkcji chooseActionAgent3
float chooseActionAgent3(int discreteState[2], float epsilon) {
    // Przykładowa implementacja wyboru akcji dla Agenta 3
    if (random(0, 100) < epsilon * 100) {
        return random(0, NUM_ACTIONS_AGENT3);
    } else {
        int bestAction = 0;
        float bestValue = qTableAgent3[discreteState[0]][0];
        for (int i = 1; i < NUM_ACTIONS_AGENT3; i++) {
            if (qTableAgent3[discreteState[0]][i] > bestValue) {
                bestValue = qTableAgent3[discreteState[0]][i];
                bestAction = i;
            }
        }
        return bestAction;
    }
}

// Implementacja funkcji executeActionAgent3
void executeActionAgent3(float action) {
    // Przykładowa implementacja wykonania akcji dla Agenta 3
    analogWrite(mosfetPin, action * PWM_INCREMENT);
}

// Implementacja funkcji calculateRewardAgent3
float calculateRewardAgent3(float state[2], float action) {
    // Przykładowa implementacja obliczania nagrody dla Agenta 3
    float voltage = readVoltage();
    float error = calculateError(VOLTAGE_SETPOINT, voltage);
    return -abs(error);
}

// Implementacja funkcji updateQAgent3
void updateQAgent3(float state[2], float action, float reward, float nextState[2]) {
    // Dyskretyzacja stanów
    int discreteState[2];
    int discreteNextState[2];
    discretizeStateAgent3(state, discreteState);
    discretizeStateAgent3(nextState, discreteNextState);

    // Znalezienie najlepszej wartości Q dla następnego stanu
    float bestNextActionValue = qTableAgent3[discreteNextState[0]][0];
    for (int i = 1; i < NUM_ACTIONS_AGENT3; i++) {
        if (qTableAgent3[discreteNextState[0]][i] > bestNextActionValue) {
            bestNextActionValue = qTableAgent3[discreteNextState[0]][i];
        }
    }

    // Aktualizacja wartości Q dla bieżącego stanu i akcji
    qTableAgent3[discreteState[0]][(int)action] += learningRate * (reward + discountFactor * bestNextActionValue - qTableAgent3[discreteState[0]][(int)action]);
}


// Implementacja funkcji readTemperature
float readTemperature() {
    // Przykładowa implementacja odczytu temperatury
    return analogRead(A3) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
}

// Implementacja funkcji readBrakeWear
float readBrakeWear() {
    // Przykładowa implementacja odczytu zużycia hamulców
    return analogRead(A4) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
}

// Implementacja funkcji readRotationalSpeed
float readRotationalSpeed() {
    // Przykładowa implementacja odczytu prędkości obrotowej
    return analogRead(A5) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
}

// Definicje stałych i zmiennych globalnych

// Parametry dla Agenta 3
const int NUM_STATES_AGENT3 = 100;
const int NUM_ACTIONS_AGENT3 = 10;
float qTableAgent1[NUM_STATES_AGENT3][NUM_ACTIONS_AGENT3] = {0}; // Inicjalizacja tablicy Q dla Agenta 1
float qTableAgent3[NUM_STATES_AGENT3][NUM_ACTIONS_AGENT3] = {0}; // Inicjalizacja tablicy Q dla Agenta 3
float learningRate = 0.1; // Przykładowa wartość współczynnika uczenia
float discountFactor = 0.9; // Przykładowa wartość współczynnika

// Zmienne do zarządzania czasem
unsigned long previousMillisDisplay = 0;
unsigned long previousMillisSerial = 0;
const long intervalDisplay = 1000; // Aktualizacja wyświetlacza co 1 sekundę
const long intervalSerial = 500; // Sprawdzanie komunikacji szeregowej co 0.5 sekundy



// Implementacja funkcji

float someFunctionOfOtherParameters(float rotationalSpeed, float torque, float frictionCoefficient) {
    // Przykładowa implementacja funkcji, która oblicza efekt hamowania na podstawie prędkości obrotowej, momentu obrotowego i współczynnika tarcia
    return (rotationalSpeed * torque) / (frictionCoefficient * 10.0);
}

void hillClimbing() {
    float currentParams[3] = {1.0, 1.0, 1.0}; // Początkowe parametry
    float bestParams[3] = {1.0, 1.0, 1.0}; // Inicjalizacja na te same wartości
    float bestObjective = objectiveFunction(currentParams);

    for (int i = 0; i < 100; i++) {
        float newParams[3];
        for (int j = 0; j < 3; j++) {
            newParams[j] = currentParams[j] + random(-1, 2) * 0.1;
        }
        float newObjective = objectiveFunction(newParams);
        if (newObjective > bestObjective) {
            memcpy(bestParams, newParams, sizeof(bestParams)); // Kopiowanie newParams do bestParams
            bestObjective = newObjective;
        }
    }
    memcpy(currentParams, bestParams, sizeof(currentParams)); // Kopiowanie bestParams do currentParams

    float rotationalSpeed = 3000.0;
    float torque = 50.0;
    float frictionCoefficient = 0.8;
    float brakingEffect = someFunctionOfOtherParameters(rotationalSpeed, torque, frictionCoefficient);
    Serial.println(brakingEffect);
}

// Dodajemy brakujące definicje funkcji tutaj



void getBestParams(float params[3]) {
    // Przykładowa implementacja funkcji zwracającej najlepsze parametry
    params[0] = 1.0;
    params[1] = 1.0;
    params[2] = 1.0;
}


float getBestObjective() {
    // Przykładowa implementacja funkcji zwracającej najlepszy wynik funkcji celu
    // Możesz dostosować tę funkcję do swoich potrzeb
    return 0.0;
}

void suggestNextParameters(float params[3]) {
    // Przykładowa implementacja funkcji sugerującej kolejne parametry
    // Możesz dostosować tę funkcję do swoich potrzeb
}

float readVoltage() {
    // Przykładowa implementacja funkcji odczytu napięcia
    return analogRead(A0) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
}

float readExcitationCurrent() {
    // Przykładowa implementacja funkcji odczytu prądu wzbudzenia
    return analogRead(A1) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
}

float readBrakingEffect() {
    // Przykładowa implementacja funkcji odczytu efektu hamowania
    return analogRead(A2) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
}

float lowPassFilter(float currentValue, float previousValue, float alpha) {
    return alpha * currentValue + (1 - alpha) * previousValue;
}

void autoTunePID(PID &pid, float setpoint, float measuredValue) {
    // Przykładowa implementacja funkcji automatycznego strojenia PID
    // Możesz dostosować tę funkcję do swoich potrzeb
}

float calculateError(float setpoint, float measuredValue) {
    return setpoint - measuredValue;
}

void updateControlParameters(float params[3]) {
    // Przykładowa implementacja aktualizacji parametrów sterowania
    float Kp = params[0];
    float Ki = params[1];
    float Kd = params[2];

    // Aktualizacja parametrów PID
    // Zakładamy, że mamy obiekt PID o nazwie pidController
    pidController.setTunings(Kp, Ki, Kd);
}

void handleSerialCommunication() {
    // Przykładowa implementacja obsługi komunikacji szeregowej
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        
        if (command.startsWith("SET_PARAMS")) {
            float params[3];
            sscanf(command.c_str(), "SET_PARAMS %f %f %f", &params[0], &params[1], &params[2]);
            updateControlParameters(params);
            Serial.println("Parameters updated");
        } else if (command.startsWith("GET_STATUS")) {
            // Przykładowa odpowiedź statusu
            Serial.println("System is running");
        } else {
            Serial.println("Unknown command");
        }
    }
}


float chooseActionAgent3(int discreteState[2], float epsilon) {
    int stateIndex = discreteState[0] * 10 + discreteState[1];
    if (random(0, 100) < epsilon * 100) {
        // Eksploracja: wybierz losową akcję
        return (float)random(0, NUM_ACTIONS_AGENT3) / (NUM_ACTIONS_AGENT3 - 1);
    } else {
        // Eksploatacja: wybierz najlepszą akcję
        float maxQValue = qTableAgent1[stateIndex][0];
        int bestActionIndex = 0;
        for (int i = 1; i < NUM_ACTIONS_AGENT3; i++) {
            if (qTableAgent1[stateIndex][i] > maxQValue) {
                maxQValue = qTableAgent1[stateIndex][i];
                bestActionIndex = i;
            }
        }
        return (float)bestActionIndex / (NUM_ACTIONS_AGENT3 - 1);
    }
}


// Definicje pinów dla tranzystorów
const int mosfetPin = D4;

// Funkcja przekształcająca akcję na wartość PWM
void executeActionAgent3(float action) {
    // Przekształcenie akcji na wartość PWM (zakres 0-255)
    int pwmValue = (int)(action * 255.0);

    // Zabezpieczenie przed przekroczeniem dozwolonego zakresu
    if (pwmValue < 0) {
        pwmValue = 0;
    } else if (pwmValue > 255) {
        pwmValue = 255;
    }

    // Ustawienie wartości PWM na pinie MOSFET
    analogWrite(mosfetPin, pwmValue);
}

// Przykładowe funkcje korzystające z executeActionAgent3
void someFunction() {
    float action = 1.2; // Przykładowa wartość akcji
    executeActionAgent3(action);
}

void anotherFunction() {
    float action = -0.5; // Przykładowa wartość akcji
    executeActionAgent3(action);
}




float calculateRewardAgent3(float state[2], float action) {
    // Walidacja zakresu wartości
    if (state[0] < 0.0 || state[0] > 1.0 || state[1] < 0.0 || state[1] > 1.0) {
        Serial.println("Error: State values out of range");
        return -FLT_MAX; // Zwracamy dużą wartość ujemną jako karę
    }
    if (action < 0.0 || action > 1.0) {
        Serial.println("Error: Action value out of range");
        return -FLT_MAX; // Zwracamy dużą wartość ujemną jako karę
    }

    float target = 1.0; // Docelowy stan
    float error = target - (state[0] + state[1]) / 2.0;
    float reward = -abs(error); // Nagroda to negatywna wartość błędu

    // Dodatkowa kara za zbyt dużą akcję
    if (action > 0.8) {
        reward -= 0.1 * (action - 0.8);
    }

    return reward;
}

void updateQAgent3(float state[2], float action, float reward, float nextState[2]) {
    // Walidacja zakresu wartości
    if (state[0] < 0.0 || state[0] > 1.0 || state[1] < 0.0 || state[1] > 1.0) {
        Serial.println("Error: State values out of range");
        return;
    }
    if (nextState[0] < 0.0 || nextState[0] > 1.0 || nextState[1] < 0.0 || nextState[1] > 1.0) {
        Serial.println("Error: Next state values out of range");
        return;
    }
    if (action < 0.0 || action > 1.0) {
        Serial.println("Error: Action value out of range");
        return;
    }
    if (reward < -FLT_MAX || reward > FLT_MAX) {
        Serial.println("Error: Reward value out of range");
        return;
    }

    // Dyskretyzacja stanów
    int discreteState[2];
    int discreteNextState[2];
    discretizeStateAgent3(state, discreteState);
    discretizeStateAgent3(nextState, discreteNextState);

    // Obliczenie indeksów stanów i akcji
    int stateIndex = discreteState[0] * NUM_STATES_AGENT3 + discreteState[1];
    int nextStateIndex = discreteNextState[0] * NUM_STATES_AGENT3 + discreteNextState[1];
    int actionIndex = (int)(action * (NUM_ACTIONS_AGENT3 - 1));

    // Aktualna wartość Q 
    float currentQ = qTableAgent3[stateIndex][actionIndex


    // Maksymalna wartość Q dla następnego stanu
    float maxNextQ = qTableAgent3[nextStateIndex][0];
    for (int i = 1; i < NUM_ACTIONS_AGENT3; i++) {
        if (qTableAgent3[nextStateIndex][i] > maxNextQ) {
            maxNextQ = qTableAgent3[nextStateIndex][i];
        }
    }


    // Aktualizacja wartości Q
    qTableAgent3[stateIndex][actionIndex] = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);
}


    
float objectiveFunction(float params[3]) {
    float Kp = params[0];
    float Ki = params[1];
    float Kd = params[2];

    // Walidacja parametrów
    if (Kp < MIN_KP || Kp > MAX_KP || Ki < MIN_KI || Ki > MAX_KI || Kd < MIN_KD || Kd > MAX_KD) {
        Serial.println("Error: Invalid parameters");
        return -1.0; // Zwróć wartość błędu
    }

    // Przykładowe wartości zadane i zmierzone (w rzeczywistości powinny pochodzić z systemu)
    float setpoint = VOLTAGE_SETPOINT; // Docelowe napięcie
    float measuredValues[10] = {228.0, 229.5, 230.5, 231.0, 229.0, 230.0, 230.2, 229.8, 230.1, 230.0};

    // Inicjalizacja kontrolera PID
    PID pidController(Kp, Ki, Kd);

    // Symulacja działania kontrolera PID
    float sse = 0.0; // Suma kwadratów błędów
    for (int i = 0; i < 10; i++) {
        float measuredValue = measuredValues[i];
        float error = setpoint - measuredValue;
        float controlSignal = pidController.compute(setpoint, measuredValue);
        
        // Aktualizacja sumy kwadratów błędów
        sse += error * error;
    }

    // Uwzględnienie minimalnego hamowania przy wzbudzeniu do 25 amperów
    float excitationCurrent = 25.0; // Przykładowa wartość wzbudzenia
    float brakingEffect = simulateBrakingEffect(3000.0, 50.0, 0.8); // Przykładowe wartości prędkości obrotowej, momentu obrotowego i współczynnika tarcia
    if (excitationCurrent >= MAX_CURRENT) {
        sse += (MAX_BRAKING_EFFECT - brakingEffect) * (MAX_BRAKING_EFFECT - brakingEffect);
    }

    // Zwróć odwrotność SSE, ponieważ chcemy maksymalizować funkcję celu
    return 1.0 / sse;
}




    // Symulacja systemu z użyciem parametrów PID
    float voltageError = simulateVoltageControl(Kp, Ki, Kd);
    float excitationCurrentError = simulateExcitationControl();
    float brakingEffect = simulateBrakingEffect();

    // Funkcja celu: minimalizacja błędu napięcia, minimalizacja błędu prądu wzbudzenia i minimalizacja hamowania
    float objective = -voltageError - excitationCurrentError - brakingEffect;

    return objective;
}

float simulateVoltageControl() {
    float error = 0.0;
    float setpoint = 230.0; // Docelowe napięcie 230V
    float measuredValue = readVoltage();
    float previousError = 0.0;
    float integral = 0.0;
    unsigned long previousMillis = millis();
    const long interval = 10; // Interwał w milisekundach

    for (int i = 0; i < 100; i++) {
        unsigned long currentMillis = millis();
        if (currentMillis - previousMillis >= interval) {
            previousMillis = currentMillis;

            float currentError = setpoint - measuredValue;
            integral += currentError;
            float derivative = currentError - previousError;
            float output = Kp * currentError + Ki * integral + Kd * derivative;

            // Wykonanie akcji na podstawie wyjścia PID
            analogWrite(mosfetPin, constrain(output, 0, 255));

            // Aktualizacja wartości zmierzonej
            measuredValue = readVoltage();
            error += abs(currentError);
            previousError = currentError;
        }
    }

    return error;
}

float simulateVoltageControl() {
    float error = 0.0;
    float setpoint = 230.0; // Docelowe napięcie 230V
    float measuredValue = readVoltage();
    float previousError = 0.0;
    float integral = 0.0;
    unsigned long previousMillis = millis();
    const long interval = 10; // Interwał w milisekundach

    // Obliczenia PID
    error = setpoint - measuredValue;
    integral += error * interval;
    float derivative = (error - previousError) / interval;
    float controlSignal = Kp * error + Ki * integral + Kd * derivative;
    previousError = error;

    return controlSignal;
}



float simulateExcitationControl() {
    float setpoint = 25.0; // Docelowy prąd wzbudzenia 25A
    float measuredValue = 0.0;
    float error = 0.0;
    float previousError = 0.0;
    float integral = 0.0;
    float Kp = 1.0; // Przykładowa wartość Kp
    float Ki = 0.1; // Przykładowa wartość Ki
    float Kd = 0.01; // Przykładowa wartość Kd
    unsigned long previousMillis = millis();
    const long interval = 10; // Interwał w milisekundach

    for (int i = 0; i < 100; i++) {
        unsigned long currentMillis = millis();
        if (currentMillis - previousMillis >= interval) {
            previousMillis = currentMillis;

            measuredValue = readExcitationCurrent();
            float currentError = setpoint - measuredValue;
            integral += currentError;
            float derivative = currentError - previousError;
            float output = Kp * currentError + Ki * integral + Kd * derivative;

            // Wykonanie akcji na podstawie wyjścia PID
            analogWrite(excitationBJT1Pin, constrain(output, 0, 255));
            analogWrite(excitationBJT2Pin, constrain(output, 0, 255));

            error += abs(currentError);
            previousError = currentError;
        }
    }

    return error;
}

float simulateBrakingEffect(float rotationalSpeed, float torque, float frictionCoefficient) {
    // Model matematyczny: efekt hamowania jest proporcjonalny do prędkości obrotowej i momentu obrotowego,
    // a odwrotnie proporcjonalny do współczynnika tarcia
    float simulatedBrakingEffect = (rotationalSpeed * torque) / (frictionCoefficient * 10.0);
    unsigned long previousMillis = millis();
    const long interval = 10; // Interwał w milisekundach

    // Symulacja dynamicznego efektu hamowania
    for (int i = 0; i < 100; i++) {
        unsigned long currentMillis = millis();
        if (currentMillis - previousMillis >= interval) {
            previousMillis = currentMillis;

            // Aktualizacja prędkości obrotowej na podstawie efektu hamowania
            rotationalSpeed -= simulatedBrakingEffect * 0.01;

            // Aktualizacja momentu obrotowego na podstawie prędkości obrotowej
            torque = torque * (1 - 0.01 * rotationalSpeed);

            // Aktualizacja współczynnika tarcia na podstawie prędkości obrotowej i momentu obrotowego
            frictionCoefficient = frictionCoefficient * (1 + 0.01 * torque);
        }
    }

    return simulatedBrakingEffect;
}



// Przykładowa funkcja symulująca efekt hamowania bez czujników
float readBrakingEffect(float load) {
    // Przykładowa prędkość obrotowa w RPM przy braku obciążenia
    float baseRotationalSpeed = 500.0;
    
    // Przykładowy moment obrotowy w Nm
    float torque = 50.0;
    
    // Przykładowy współczynnik tarcia
    float frictionCoefficient = 0.8;
    
    // Dostosowanie prędkości obrotowej w zależności od obciążenia
    float rotationalSpeed = baseRotationalSpeed * (1.0 - load);
    return someFunctionOfOtherParameters(rotationalSpeed, torque, frictionCoefficient);
}
    
    // Zakładamy, że mniejsza wartość efektu hamowania jest lepsza
    // Zastosowanie funkcji penalizującej większe wartości
    float penalizedBrakingEffect = 1.0 / (1.0 + brakingEffect);

    return penalizedBrakingEffect;
}

// Kalibracja napięcia
float calibrateVoltage(float rawVoltage) {
    // Przykładowa kalibracja z bardziej precyzyjną korektą
    const float calibrationFactor = 1.05; // Współczynnik kalibracji
    float calibratedVoltage = rawVoltage * calibrationFactor;

    // Dodatkowa korekta na podstawie pomiarów
    if (calibratedVoltage > 5.0) {
        calibratedVoltage = 5.0; // Ograniczenie maksymalnego napięcia
    } else if (calibratedVoltage < 0.0) {
        calibratedVoltage = 0.0; // Ograniczenie minimalnego napięcia
    }

    return calibratedVoltage;
}


// Funkcja kalibracji prądu wzbudzenia
float calibrateCurrent(float rawCurrent) {
    // Współczynnik kalibracji
    const float calibrationFactor = 0.95; // Korekta o -5%
    float calibratedCurrent = rawCurrent * calibrationFactor;

    // Dodatkowa korekta na podstawie pomiarów
    if (calibratedCurrent > MAX_CURRENT) {
        calibratedCurrent = MAX_CURRENT; // Ograniczenie maksymalnego prądu
    } else if (calibratedCurrent < 0.0) {
        calibratedCurrent = 0.0; // Ograniczenie minimalnego prądu
    }

    return calibratedCurrent;
}

// Definicje stałych
#define TEMPERATURE_COEFFICIENT 0.98 // Współczynnik kalibracji dla temperatury
#define WEAR_COEFFICIENT 0.95 // Współczynnik kalibracji dla zużycia hamulców
#define SPEED_COEFFICIENT 1.01 // Współczynnik kalibracji dla prędkości obrotowej

// Funkcja odczytu temperatury (przykładowa implementacja)
float readTemperature() {
    // Przykładowa wartość temperatury
    return 25.0; // 25 stopni Celsjusza
}

// Funkcja odczytu zużycia hamulców (przykładowa implementacja)
float readBrakeWear() {
    // Symulacja zmieniającego się zużycia hamulców
    static float brakeWear = 0.8; // Początkowa wartość zużycia hamulców (80%)
    brakeWear += random(-5, 6) / 100.0; // Losowa zmiana w zakresie -0.05 do 0.05
    brakeWear = constrain(brakeWear, 0.0, 1.0); // Ograniczenie wartości do zakresu 0-1 (0-100%)
    return brakeWear;
}


// Funkcja odczytu prędkości obrotowej (przykładowa implementacja)
float readRotationalSpeed() {
    // Przykładowa wartość prędkości obrotowej
    return 3000.0; // 3000 RPM
}

// Kalibracja efektu hamowania
float calibrateBrakingEffect(float rawBrakingEffect) {
    // Odczyt dodatkowych parametrów
    float temperature = readTemperature();
    float brakeWear = readBrakeWear();
    float rotationalSpeed = readRotationalSpeed();

    // Kalibracja na podstawie temperatury
    float temperatureFactor = 1.0;
    if (temperature > 30.0) {
        temperatureFactor = TEMPERATURE_COEFFICIENT;
    }

    // Kalibracja na podstawie zużycia hamulców
    float wearFactor = 1.0;
    if (brakeWear > 0.7) {
        wearFactor = WEAR_COEFFICIENT;
    }

    // Kalibracja na podstawie prędkości obrotowej
    float speedFactor = 1.0;
    if (rotationalSpeed > 2500.0) {
        speedFactor = SPEED_COEFFICIENT;
    }

    // Obliczenie skalibrowanego efektu hamowania
    float calibratedBrakingEffect = rawBrakingEffect * temperatureFactor * wearFactor * speedFactor;

    // Dodatkowa korekta o 2%
    calibratedBrakingEffect *= 1.02;

    return calibratedBrakingEffect;
}

// Filtrowanie sygnału (przykładowy filtr dolnoprzepustowy)
float lowPassFilter(float currentValue, float previousValue, float alpha) {
    return alpha * currentValue + (1 - alpha) * previousValue;
}



float readVoltage() {
    // Przykładowa implementacja odczytu napięcia z pinu analogowego
    int rawValue = analogRead(muxInputPin);
    float voltage = (rawValue / (float)ADC_MAX_VALUE) * VOLTAGE_REFERENCE;

    // Użycie filtra dolnoprzepustowego do wygładzenia odczytu napięcia
    static float previousVoltage = 0.0;
    float alpha = 0.1; // Przykładowa wartość alpha
    float filteredVoltage = lowPassFilter(voltage, previousVoltage, alpha);
    previousVoltage = filteredVoltage;

    return filteredVoltage;
}




// Funkcja odczytu prądu wzbudzenia
float readExcitationCurrent() {
    float rawCurrent = analogRead(A1) * (MAX_CURRENT / ADC_MAX_VALUE);
    float calibratedCurrent = calibrateCurrent(rawCurrent);
    static float filteredCurrent = 0.0; // Inicjalizacja na 0.0
    filteredCurrent = lowPassFilter(calibratedCurrent, filteredCurrent, 0.1);
    return filteredCurrent;
}





// Funkcja odczytu efektu hamowania
float readBrakingEffect() {
    float rawBrakingEffect = analogRead(A2) * (MAX_BRAKING_EFFECT / ADC_MAX_VALUE);
    float calibratedBrakingEffect = calibrateBrakingEffect(rawBrakingEffect);
    static float filteredBrakingEffect = 0.0; // Inicjalizacja na 0.0
    filteredBrakingEffect = lowPassFilter(calibratedBrakingEffect, filteredBrakingEffect, 0.1);
    return filteredBrakingEffect;
}

    class PID {
    public:
        PID(float kp, float ki, float kd) : Kp(kp), Ki(ki), Kd(kd), prevError(0), integral(0) {}

        float compute(float setpoint, float measuredValue) {
            float error = setpoint - measuredValue;
            integral += error;
            float derivative = error - prevError;
            float output = Kp * error + Ki * integral + Kd * derivative;
            prevError = error;
            return output;
        }

        void setTunings(float kp, float ki, float kd) {
            Kp = kp;
            Ki = ki;
            Kd = kd;
        }

    private:
        float Kp, Ki, Kd;
        float prevError;
        float integral;
    };
    

// Funkcja celu dla optymalizatora
float objectiveFunction(float params[3]) {
    float Kp = params[0];
    float Ki = params[1];
    float Kd = params[2];
    float controlSignal = simulateVoltageControl(Kp, Ki, Kd);
    float error = calculateError(VOLTAGE_SETPOINT, controlSignal);
    return abs(error); // Minimalizujemy wartość bezwzględną błędu
}


// Przykładowa funkcja sugerująca nowe parametry z użyciem optymalizacji Bayesowskiej
void suggestNextParameters(float params[3]) {
    // Inicjalizacja optymalizatora Bayesowskiego
    BayesOptimizer optimizer;

    // Definicja zakresów dla parametrów
    optimizer.setBounds(0, 0.0, 5.0); // Zakres dla parametru 0 (Kp)
    optimizer.setBounds(1, 0.0, 1.0); // Zakres dla parametru 1 (Ki)
    optimizer.setBounds(2, 0.0, 2.0); // Zakres dla parametru 2 (Kd)

    // Dodanie punktów startowych
    optimizer.addObservation({params[0], params[1], params[2]}, objectiveFunction(params));

    // Optymalizacja w celu znalezienia najlepszych parametrów
    std::vector<float> newParams = optimizer.optimize();

    // Aktualizacja parametrów
    for (int i = 0; i < 3; i++) {
        params[i] = newParams[i];
    }
}

// Przykładowa funkcja ustawiająca najlepsze parametry
void getBestParams(float params[3]) {
    // Tutaj można ustawić najlepsze parametry w odpowiednich zmiennych globalnych lub strukturach
    Serial.print("Best Params: ");
    Serial.print(params[0]);
    Serial.print(", ");
    Serial.print(params[1]);
    Serial.print(", ");
    Serial.println(params[2]);
}

float getBestObjective() {
    return 1.0; // Przykładowa najlepsza wartość funkcji celu
}

void autoTunePID(PID &pid, float setpoint, float measuredValue) {
    // Przykładowa implementacja auto-strojenia
    // Można użyć różnych metod, np. Ziegler-Nichols, Hill Climbing, itp.
    float bestKp = 0, bestKi = 0, bestKd = 0;
    float bestError = FLT_MAX;

    for (float kp = 0; kp < 10; kp += 0.1) {
        for (float ki = 0; ki < 1; ki += 0.1) {
            for (float kd = 0; kd < 2; kd += 0.1) {
                pid.setTunings(kp, ki, kd);
                float error = simulateSystem(pid, setpoint, measuredValue);
                if (error < bestError) {
                    bestError = error;
                    bestKp = kp;
                    bestKi = ki;
                    bestKd = kd;
                }
            }
        }
    }
float simulateSystem(PID &pid, float setpoint, float measuredValue) {
    float error = 0.0;
    float previousError = 0.0;
    float integral = 0.0;
    unsigned long previousMillis = millis();
    const long interval = 10; // Interwał w milisekundach

    for (int i = 0; i < 100; i++) {
        unsigned long currentMillis = millis();
        if (currentMillis - previousMillis >= interval) {
            previousMillis = currentMillis;

            float currentError = setpoint - measuredValue;
            integral += currentError;
            float derivative = currentError - previousError;
            float output = pid.compute(setpoint, measuredValue);

            // Wykonanie akcji na podstawie wyjścia PID
            analogWrite(mosfetPin, constrain(output, 0, 255));

            // Aktualizacja wartości zmierzonej
            measuredValue = readVoltage();
            error += abs(currentError);
            previousError = currentError;
        }
    }

    return error;
}
    pid.setTunings(bestKp, bestKi, bestKd);
}

// Dodaj funkcję autoTunePID
void autoTunePID() {
    // Przykładowe wartości początkowe
    float bestKp = Kp;
    float bestKi = Ki;
    float bestKd = Kd;
    float bestError = FLT_MAX;

    // Zakresy przeszukiwania
    float kpRange[] = {0.0, 5.0};
    float kiRange[] = {0.0, 1.0};
    float kdRange[] = {0.0, 2.0};

    // Liczba iteracji optymalizacji
    int iterations = 100;

    for (int i = 0; i < iterations; i++) {
        // Generowanie losowych wartości w zakresie
        float newKp = kpRange[0] + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (kpRange[1] - kpRange[0])));
        float newKi = kiRange[0] + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (kiRange[1] - kiRange[0])));
        float newKd = kdRange[0] + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (kdRange[1] - kdRange[0])));

        // Ustawienie nowych wartości PID
        Kp = newKp;
        Ki = newKi;
        Kd = newKd;

        // Symulacja lub rzeczywiste testowanie z nowymi wartościami PID
        float currentError = simulatePIDControl();

        // Sprawdzenie, czy nowe wartości są lepsze
        if (currentError < bestError) {
            bestError = currentError;
            bestKp = newKp;
            bestKi = newKi;
            bestKd = newKd;
        }
    }

    // Ustawienie najlepszych znalezionych wartości PID
    Kp = bestKp;
    Ki = bestKi;
    Kd = bestKd;
}

// Funkcja symulująca lub testująca kontrolę PID
float simulatePIDControl() {
    float error = 0.0;
    float setpoint = VOLTAGE_SETPOINT;
    float measuredValue = readVoltage();
    float previousError = 0.0;
    float integral = 0.0;

    for (int i = 0; i < 100; i++) {
        float currentError = setpoint - measuredValue;
        integral += currentError;
        float derivative = currentError - previousError;
        float output = Kp * currentError + Ki * integral + Kd * derivative;

        // Wykonanie akcji na podstawie wyjścia PID
        analogWrite(mosfetPin, constrain(output, 0, 255));

        // Aktualizacja wartości zmierzonej
        measuredValue = readVoltage();
        error += abs(currentError);
        previousError = currentError;

        // Krótkie opóźnienie, aby symulować czas rzeczywisty
        delay(10);
    }

    return error;
}

// Definicje zmiennych globalnych
float voltageIn[2] = {0};
float currentIn[2] = {0};
float externalVoltage = 0.0;
float externalCurrent = 0.0;
float efficiency = 0.0;
float efficiencyPercent = 0.0;
float voltageDrop = 0.0;
ESP8266WebServer server(80);
Adafruit_SH1106 display(128, 64, &Wire, -1);
int lastAction = 0;

// Definicje zmiennych globalnych dla PID
float Kp = 2.0, Ki = 0.5, Kd = 1.0;
const float Kp_max = 5.0;

float VOLTAGE_SETPOINT = 230.0; // Docelowe napięcie 230 V
float currentVoltage = 0.0; // Bieżące napięcie, inicjalizowane na 0.0
float currentCurrent = 0.0; // Bieżący prąd, inicjalizowany na 0.0
const float maxCurrent = 25.0; // Maksymalny prąd w amperach

// Deklaracja zmiennej globalnej
float LOAD_THRESHOLD = 0.5;

#define NUM_STATE_BINS_ERROR 10
#define NUM_STATE_BINS_LOAD 10
#define NUM_STATE_BINS_KP 10
#define NUM_STATE_BINS_KI 10
#define NUM_STATE_BINS_KD 10
#define NUM_ACTIONS 5
#define NUM_STATES_AGENT3 100
#define NUM_ACTIONS_AGENT3 5

// Dodatkowe zmienne globalne
float previousError = 0;
float integral = 0;
const float COMPENSATION_FACTOR = 0.1;
const int PWM_INCREMENT = 10;
float epsilon = 0.3;


// Definicje zmiennych globalnych dla agenta 1
float qTableAgent1[NUM_STATE_BINS_ERROR * NUM_STATE_BINS_LOAD][NUM_ACTIONS];
float stateAgent1[2] = {0.0, 0.0}; // Stan agenta 1: [błąd, obciążenie]
float actionAgent1 = 0.0; // Akcja agenta 1

// Funkcja aktualizująca tablicę Q-learning dla agenta 1
void updateQTableAgent1(float state[2], float action, float reward, float nextState[2]) {
    int stateIndex = (int)(state[0] * NUM_STATE_BINS_ERROR + state[1] * NUM_STATE_BINS_LOAD);
    int nextStateIndex = (int)(nextState[0] * NUM_STATE_BINS_ERROR + nextState[1] * NUM_STATE_BINS_LOAD);
    int actionIndex = (int)action;

    // Oblicz wartość Q dla obecnego stanu i akcji
    float currentQ = qTableAgent1[stateIndex][actionIndex];

    // Znajdź maksymalną wartość Q dla następnego stanu
    float maxNextQ = qTableAgent1[nextStateIndex][0];
    for (int i = 1; i < NUM_ACTIONS; i++) {
        if (qTableAgent1[nextStateIndex][i] > maxNextQ) {
            maxNextQ = qTableAgent1[nextStateIndex][i];
        }
    }

    // Zaktualizuj wartość Q
    qTableAgent1[stateIndex][actionIndex] = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);
}

// Funkcja wybierająca akcję dla agenta 1
float selectActionAgent1(float state[2]) {
    int stateIndex = (int)(state[0] * NUM_STATE_BINS_ERROR) * NUM_STATE_BINS_LOAD + (int)(state[1] * NUM_STATE_BINS_LOAD);

    if (random(0, 100) < epsilon * 100) {
        return random(0, NUM_ACTIONS);
    } else {
        float bestAction = 0;
        float maxQ = qTableAgent1[stateIndex][0];
        for (int i = 1; i < NUM_ACTIONS; i++) {
            if (qTableAgent1[stateIndex][i] > maxQ) {
                maxQ = qTableAgent1[stateIndex][i];
                bestAction = i;
            }
        }
        return bestAction;
    }
}

// Funkcja ucząca agenta 1
void trainAgent1() {
    // Oblicz nagrodę na podstawie wydajności
    float reward = calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);

    // Oblicz następny stan
    float nextState[2] = {VOLTAGE_SETPOINT - currentVoltage, currentCurrent};

    // Zaktualizuj tablicę Q-learning
    updateQTableAgent1(stateAgent1, actionAgent1, reward, nextState);

    // Zaktualizuj stan agenta
    stateAgent1[0] = nextState[0];
    stateAgent1[1] = nextState[1];

    // Wybierz nową akcję na podstawie zaktualizowanego stanu
    actionAgent1 = selectActionAgent1(stateAgent1);

    // Wykonaj wybraną akcję
    performActionAgent1(actionAgent1);
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


// Zmienne globalne dla komunikacji między agentami
bool agent2IncreasedExcitation = false;
bool agent3MinimizedBraking = false;
float agent2ExcitationCurrent = 0.0;

// Funkcja komunikacji między agentami
void communicateBetweenAgents() {
    if (agent2IncreasedExcitation) {
        Serial.println("Agent 2 zwiększył prąd wzbudzenia");
        // Przykład: agent 1 reaguje na zwiększenie prądu wzbudzenia przez agenta 2
        autoTunePID(); // Wywołanie funkcji autoTunePID
        agent2IncreasedExcitation = false; // Resetowanie flagi
    }
    if (agent3MinimizedBraking) {
        Serial.println("Agent 3 zminimalizował hamowanie");
        // Przykład: agent 1 reaguje na minimalizowanie hamowania przez agenta 3
        autoTunePID(); // Wywołanie funkcji autoTunePID
        agent3MinimizedBraking = false; // Resetowanie flagi
    }
}

// Funkcja wykonująca akcję agenta 3
void performActionAgent3(float action) {
    switch ((int)action) {
        case 0:
            // Przykładowa akcja zmniejszająca hamowanie
            analogWrite(mosfetPin, analogRead(mosfetPin) - PWM_INCREMENT);
            agent3MinimizedBraking = true; // Ustawienie flagi
            break;
        case 1:
            // Inna akcja
            analogWrite(mosfetPin, analogRead(mosfetPin) + PWM_INCREMENT);
            break;
        // Dodaj inne akcje
        default:
            // Kara za nieznaną akcję
            Serial.println("Nieznana akcja, kara za zwiększenie hamowania");
            break;
    }
}

// Funkcja wykonująca akcję agenta 1
void performActionAgent1(float action) {
    // Implementacja akcji agenta
    switch ((int)action) {
        case 0:
            Kp += 0.1;
            break;
        case 1:
            Kp -= 0.1;
            break;
        case 2:
            Ki += 0.1;
            break;
        case 3:
            Ki -= 0.1;
            break;
        case 4:
            Kd += 0.1;
            break;
        case 5:
            Kd -= 0.1;
            break;
        default:
            // Nieznana akcja
            break;
    }

    // Upewnij się, że wartości PID są w odpowiednich zakresach
    Kp = constrain(Kp, 0.0, 5.0); // Zakres dla Kp: 0.0 - 5.0
    Ki = constrain(Ki, 0.0, 1.0); // Zakres dla Ki: 0.0 - 1.0
    Kd = constrain(Kd, 0.0, 2.0); // Zakres dla Kd: 0.0 - 2.0
}

// Funkcja wybierająca akcję dla agenta 3
float selectActionAgent3(float state[2]) {
    int stateIndex = (int)(state[0] * NUM_STATE_BINS_ERROR) * NUM_STATE_BINS_LOAD + (int)(state[1] * NUM_STATE_BINS_LOAD);

    if (random(0, 100) < epsilon * 100) {
        return random(0, NUM_ACTIONS);
    } else {
        float bestAction = 0;
        float maxQ = qTableAgent3[stateIndex][0];
        for (int i = 1; i < NUM_ACTIONS; i++) {
            if (qTableAgent3[stateIndex][i] > maxQ) {
                maxQ = qTableAgent3[stateIndex][i];
                bestAction = i;
            }
        }
        return bestAction;
    }
}

// Funkcja aktualizująca tablicę Q-learning dla agenta 3
void updateQTableAgent3(float state[2], float action, float reward, float nextState[2]) {
    int stateIndex = (int)(state[0] * NUM_STATE_BINS_ERROR + state[1] * NUM_STATE_BINS_LOAD);
    int nextStateIndex = (int)(nextState[0] * NUM_STATE_BINS_ERROR + nextState[1] * NUM_STATE_BINS_LOAD);
    int actionIndex = (int)action;

    // Oblicz wartość Q dla obecnego stanu i akcji
    float currentQ = qTableAgent3[stateIndex][actionIndex];

    // Znajdź maksymalną wartość Q dla następnego stanu
    float maxNextQ = qTableAgent3[nextStateIndex][0];
    for (int i = 1; i < NUM_ACTIONS; i++) {
        if (qTableAgent3[nextStateIndex][i] > maxNextQ) {
            maxNextQ = qTableAgent3[nextStateIndex][i];
        }
    }

    // Zaktualizuj wartość Q
    qTableAgent3[stateIndex][actionIndex] = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);
}

// Funkcja ucząca agenta 3
void trainAgent3() {
    // Oblicz karę na podstawie zwiększonego hamowania
    float reward = -analogRead(mosfetPin); // Przykładowa kara za zwiększone hamowanie

    // Uwzględnij informację od Agenta 2
    if (agent2IncreasedExcitation) {
        reward -= 10; // Dodatkowa kara za zwiększenie prądu wzbudzenia przez Agenta 2
    }

    // Oblicz następny stan
    float nextState[2] = {VOLTAGE_SETPOINT - currentVoltage, currentCurrent};

    // Zaktualizuj tablicę Q-learning
    updateQTableAgent3(stateAgent3, actionAgent3, reward, nextState);

    // Zaktualizuj stan agenta
    stateAgent3[0] = nextState[0];
    stateAgent3[1] = nextState[1];

    // Wybierz nową akcję na podstawie zaktualizowanego stanu
    actionAgent3 = selectActionAgent3(stateAgent3);

    // Wykonaj wybraną akcję
    performActionAgent3(actionAgent3);
}

// Stałe konfiguracyjne
const int MAX_EXCITATION_CURRENT = 255;
const float MAX_VOLTAGE = 230.0;
const float MIN_VOLTAGE = 0.0;

// Funkcja symulująca działanie systemu stabilizatora napięcia
float simulateSystem(float Kp, float Ki, float Kd) {
    float setpoint = 230.0; // Docelowe napięcie
    currentVoltage = 0.0; // Użycie globalnej zmiennej
    currentError = 0.0; // Użycie globalnej zmiennej
    float previousError = 0.0;
    float integral = 0.0;
    float derivative = 0.0;
    float controlSignal = 0.0;
    float simulatedEfficiency = 0.0;
    int simulationSteps = 100; // Liczba kroków symulacji
    float timeStep = 0.1; // Krok czasowy symulacji

    // Implementacja symulacji
    for (int i = 0; i < simulationSteps; i++) {
        currentError = setpoint - currentVoltage;
        integral += currentError * timeStep;
        derivative = (currentError - previousError) / timeStep;
        controlSignal = Kp * currentError + Ki * integral + Kd * derivative;
        currentVoltage += controlSignal * timeStep; // Przykładowa aktualizacja napięcia
        previousError = currentError;
        simulatedEfficiency += abs(currentError); // Przykładowa metryka wydajności
    }

    return simulatedEfficiency / simulationSteps; // Zwracanie średniej wydajności
}


    // Parametry modelu stabilizatora
    float systemGain = 1.0;
    float systemTimeConstant = 1.0;
    float systemDelay = 0.1;

    for (int i = 0; i < simulationSteps; i++) {
        currentError = setpoint - currentVoltage;
        integral += currentError * timeStep;
        derivative = (currentError - previousError) / timeStep;
        controlSignal = Kp * currentError + Ki * integral + Kd * derivative;

        // Modelowanie dynamiki systemu z opóźnieniem
        float delayedControlSignal = controlSignal * exp(-systemDelay / systemTimeConstant);
        currentVoltage += (systemGain * delayedControlSignal - currentVoltage) * (timeStep / systemTimeConstant);

        previousError = currentError;
        simulatedEfficiency = 1.0 - abs(currentError / setpoint);
    }

    return simulatedEfficiency;
}

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

const int NUM_ACTIONS = 6;
float epsilon = 0.3;

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

// Definicje stałych
#define MAX_EXCITATION_CURRENT 25 // Maksymalny prąd wzbudzenia w amperach

// Funkcja kontrolująca MOSFET i trzy tranzystory stabilizatora napięcia
void controlVoltageRegulator(float voltage, float excitationCurrent) {
    // Stabilizacja napięcia na poziomie 230V
    voltage = constrain(voltage, 0.0, 230.0);

    // Kontrola tranzystora MOSFET w zależności od prądu wzbudzenia
    if (excitationCurrent > LOAD_THRESHOLD) {
        digitalWrite(mosfetPin, HIGH);
    } else {
        digitalWrite(mosfetPin, LOW);
    }

    // Kontrola trzech tranzystorów stabilizatora napięcia
    analogWrite(newPin1, map(voltage, 0, 230, 0, 255));
    analogWrite(newPin2, map(voltage, 0, 230, 0, 255));
    analogWrite(newPin3, map(voltage, 0, 230, 0, 255));
}

// Funkcja kontrolująca tranzystory dla czterofazowej prądnicy
void controlExcitationTransistors(float excitationCurrent) {
    // Sprawdzenie, czy prąd wzbudzenia mieści się w oczekiwanym zakresie
    if (excitationCurrent < 0 || excitationCurrent > MAX_CURRENT) {
        Serial.println("Prąd wzbudzenia poza zakresem!");
        return;
    }

    // Rozdzielenie prądu wzbudzenia na cztery tranzystory BJT
    float baseCurrentPerTransistor = excitationCurrent / 4.0; // Dzielimy prąd na 4 tranzystory

    // Mapowanie prądu wzbudzenia na wartości PWM dla każdego tranzystora
    int pwmValueBJT1 = map(baseCurrentPerTransistor, 0, MAX_CURRENT / 4.0, 0, 255);
    int pwmValueBJT2 = map(baseCurrentPerTransistor, 0, MAX_CURRENT / 4.0, 0, 255);
    int pwmValueBJT3 = map(baseCurrentPerTransistor, 0, MAX_CURRENT / 4.0, 0, 255);
    int pwmValueBJT4 = map(baseCurrentPerTransistor, 0, MAX_CURRENT / 4.0, 0, 255);

    // Upewnienie się, że wartości PWM są w zakresie 0-255
    pwmValueBJT1 = constrain(pwmValueBJT1, 0, 255);
    pwmValueBJT2 = constrain(pwmValueBJT2, 0, 255);
    pwmValueBJT3 = constrain(pwmValueBJT3, 0, 255);
    pwmValueBJT4 = constrain(pwmValueBJT4, 0, 255);

    // Sterowanie czterema tranzystorami BJT za pomocą sygnałów PWM
    analogWrite(excitationBJT1Pin, pwmValueBJT1);
    analogWrite(excitationBJT2Pin, pwmValueBJT2);
    analogWrite(bjtPin3, pwmValueBJT3); 
    analogWrite(bjtPin4, pwmValueBJT4);
}


// Funkcja automatycznej optymalizacji PID
void handleSerialCommands() {
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        command.trim(); // Usuwa białe znaki na początku i końcu

        if (command == "START") {
            // Przykład: obsługa komendy START
            Serial.println("Komenda START otrzymana");
            // Dodaj tutaj kod do uruchomienia odpowiedniej funkcji
        } else if (command == "STOP") {
            // Przykład: obsługa komendy STOP
            Serial.println("Komenda STOP otrzymana");
            // Dodaj tutaj kod do zatrzymania odpowiedniej funkcji
        } else if (command == "OPTIMIZE") {
            // Przykład: obsługa komendy OPTIMIZE
            Serial.println("Komenda OPTIMIZE otrzymana");
            optimizePID();
        } else {
            Serial.println("Nieznana komenda: " + command);
        }
    }
}

// Funkcja automatycznego dostrajania PID
void autoTunePID() {
    float Ku = 0.0; // Krytyczny współczynnik wzmocnienia
    float Tu = 0.0; // Okres oscylacji
    bool oscillationsDetected = false;
    unsigned long startTime = millis();
    unsigned long currentTime;
    float previousError = 0.0;
    float currentError;
    float maxError = 0.0;
    float minError = 0.0;

    Kp = 1.0;
    Ki = 0.0;
    Kd = 0.0;

    while (!oscillationsDetected && (millis() - startTime) < TUNING_TIMEOUT) {
        currentTime = millis();
        currentError = VOLTAGE_SETPOINT - currentVoltage;

        if (currentError > maxError) {
            maxError = currentError;
        }
        if (currentError < minError) {
            minError = currentError;
        }

        if ((maxError - minError) > 0.1) {
            oscillationsDetected = true;
            Tu = (currentTime - startTime) / 1000.0; // Okres oscylacji w sekundach
            Ku = 4.0 * (VOLTAGE_SETPOINT / (maxError - minError));
        }

        float controlSignal = Kp * currentError;
        analogWrite(mosfetPin, constrain(controlSignal, 0, 255));
        delay(100); // Opóźnienie dla stabilizacji
    }
}


   // Funkcja testująca ustawienia PID
void testPIDSettings() {
    float efficiency = simulateSystem(Kp, Ki, Kd);
    Serial.print("Efektywność dla Kp="); Serial.print(Kp);
    Serial.print(", Ki="); Serial.print(Ki);
    Serial.print(", Kd="); Serial.println(Kd);
    Serial.print("Efektywność: "); Serial.println(efficiency);
}

// Funkcja zapisywania lub odczytywania parametrów PID w pamięci EEPROM
void handlePIDParams(float &Kp, float &Ki, float &Kd, bool save) {
    if (save) {
        EEPROM.put(0, Kp); // Zapisz wartość Kp na początku pamięci EEPROM
        EEPROM.put(sizeof(float), Ki); // Zapisz wartość Ki po Kp
        EEPROM.put(2 * sizeof(float), Kd); // Zapisz wartość Kd po Ki
        EEPROM.commit(); // Zatwierdź zmiany w pamięci EEPROM
    } else {
        EEPROM.get(0, Kp); // Odczytaj wartość Kp z początku pamięci EEPROM
        EEPROM.get(sizeof(float), Ki); // Odczytaj wartość Ki po Kp
        EEPROM.get(2 * sizeof(float), Kd); // Odczytaj wartość Kd po Ki
    }
}

// Funkcja optymalizacji PID z zapisywaniem wyników
void optimizePID() {
    // Ustawienia początkowe
    float Ku = 0.0; // Krytyczny współczynnik wzmocnienia
    float Tu = 0.0; // Okres oscylacji
    bool oscillationsDetected = false;
    unsigned long startTime = millis();
    unsigned long currentTime;
    float previousError = 0.0;
    float currentError;
    float maxError = 0.0;
    float minError = 0.0;

    // Wstępne ustawienia PID
    Kp = 1.0;
    Ki = 0.0;
    Kd = 0.0;

    // Wprowadzenie oscylacji
    while (!oscillationsDetected && (millis() - startTime) < TUNING_TIMEOUT) {
        currentTime = millis();
        currentError = VOLTAGE_SETPOINT - currentVoltage;

        // Sprawdzenie oscylacji
        if (currentError > maxError) {
            maxError = currentError;
        }
        if (currentError < minError) {
            minError = currentError;
        }

        // Wykrywanie oscylacji
        if ((maxError - minError) > 0.1) {
            oscillationsDetected = true;
            Tu = (currentTime - startTime) / 1000.0; // Okres oscylacji w sekundach
            Ku = 4.0 * (VOLTAGE_SETPOINT / (maxError - minError));
        }

        // Aktualizacja sterowania
        float controlSignal = Kp * currentError;
        analogWrite(mosfetPin, constrain(controlSignal, 0, 255));

        delay(100); // Opóźnienie dla stabilizacji
    }

    // Ustawienia PID na podstawie metody Zieglera-Nicholsa
    if (oscillationsDetected) {
        Kp = 0.6 * Ku;
        Ki = 2 * Kp / Tu;
        Kd = Kp * Tu / 8;
        Serial.println("Optymalizacja PID zakończona:");
        Serial.print("Kp: "); Serial.println(Kp);
        Serial.print("Ki: "); Serial.println(Ki);
        Serial.print("Kd: "); Serial.println(Kd);

        // Zapisz najlepsze parametry w pamięci EEPROM
        handlePIDParams(Kp, Ki, Kd, true);

        // Wyślij najlepsze parametry do komputera
        Serial.print("Zapisane parametry PID: Kp="); Serial.print(Kp);
        Serial.print(", Ki="); Serial.print(Ki);
        Serial.print(", Kd="); Serial.println(Kd);
    } else {
        Serial.println("Nie udało się wykryć oscylacji w czasie optymalizacji PID.");
    }
}


// Wczytaj parametry PID z pamięci EEPROM podczas uruchamiania

int analogRead(int pin) {
    // Implementacja odczytu analogowego
    // W przypadku ESP8266, używamy funkcji analogRead z biblioteki Arduino
    return ::analogRead(pin);
}

// Inicjalizacja tablicy Q-learning
float qTable[NUM_STATE_BINS_ERROR * NUM_STATE_BINS_LOAD * NUM_STATE_BINS_KP * NUM_STATE_BINS_KI * NUM_STATE_BINS_KD][NUM_ACTIONS][3] = {0};

// Funkcja konwertująca stan na indeks tablicy Q
int getStateIndex(float error, float load, float kp, float ki, float kd) {
    int errorBin = constrain(map(error, MIN_ERROR, MAX_ERROR, 0, NUM_STATE_BINS_ERROR - 1), 0, NUM_STATE_BINS_ERROR - 1);
    int loadBin = constrain(map(load, MIN_LOAD, MAX_LOAD, 0, NUM_STATE_BINS_LOAD - 1), 0, NUM_STATE_BINS_LOAD - 1);
    int kpBin = constrain(map(kp, MIN_KP, MAX_KP, 0, NUM_STATE_BINS_KP - 1), 0, NUM_STATE_BINS_KP - 1);
    int kiBin = constrain(map(ki, MIN_KI, MAX_KI, 0, NUM_STATE_BINS_KI - 1), 0, NUM_STATE_BINS_KI - 1);
    int kdBin = constrain(map(kd, MIN_KD, MAX_KD, 0, NUM_STATE_BINS_KD - 1), 0, NUM_STATE_BINS_KD - 1);

    return errorBin + NUM_STATE_BINS_ERROR * (loadBin + NUM_STATE_BINS_LOAD * (kpBin + NUM_STATE_BINS_KP * (kiBin + NUM_STATE_BINS_KI * kdBin)));
}

// Funkcja wybierająca akcję na podstawie strategii epsilon-greedy
int chooseAction(int stateIndex) {
    if (random(0, 100) < epsilon * 100) {
        return random(0, NUM_ACTIONS); // Wybór losowej akcji
    } else {
        int bestAction = 0;
        float bestValue = qTable[stateIndex][0][0];
        for (int i = 1; i < NUM_ACTIONS; i++) {
            if (qTable[stateIndex][i][0] > bestValue) {
                bestValue = qTable[stateIndex][i][0];
                bestAction = i;
            }
        }
        return bestAction;
    }
}

// Funkcja aktualizująca tablicę Q
void updateQTable(int stateIndex, int action, float reward, int nextStateIndex) {
    float bestNextValue = qTable[nextStateIndex][0][0];
    for (int i = 1; i < NUM_ACTIONS; i++) {
        if (qTable[nextStateIndex][i][0] > bestNextValue) {
            bestNextValue = qTable[nextStateIndex][i][0];
        }
    }
    qTable[stateIndex][action][0] = (1 - learningRate) * qTable[stateIndex][action][0] + learningRate * (reward + discountFactor * bestNextValue);
}

// Główna pętla uczenia
void qLearningAgent3() {
    // Inicjalizacja zmiennych stanu
    float error = VOLTAGE_SETPOINT - currentVoltage;
    float load = currentIn[0]; // Przykładowe obciążenie
    int stateIndex = getStateIndex(error, load, Kp, Ki, Kd);
    int action = chooseAction(stateIndex, qTableAgent3, epsilon);

    // Wykonanie akcji (przykładowa implementacja)
    switch (action) {
        case 0: Kp += 0.1; break;
        case 1: Kp -= 0.1; break;
        case 2: Ki += 0.1; break;
        case 3: Ki -= 0.1; break;
        case 4: Kd += 0.1; break;
        case 5: Kd -= 0.1; break;
    }
}


    // Symulacja systemu z nowymi parametrami PID
    float newError = VOLTAGE_SETPOINT - simulateSystem(Kp, Ki, Kd);
    float reward = -abs(newError); // Nagroda jest ujemną wartością błędu, aby minimalizować błąd

    int nextStateIndex = getStateIndex(newError, load, Kp, Ki, Kd);
    updateQTable(stateIndex, action, reward, nextStateIndex);
}

unsigned long lastUpdateAgent1 = 0;
unsigned long lastUpdateAgent2 = 0;
unsigned long lastUpdateAgent3 = 0;
const unsigned long updateInterval = 1;

// Funkcja wybierająca akcję na podstawie stanu
int chooseAction(int stateIndex, float qTable[][NUM_ACTIONS], float epsilon) {
    if (random(0, 100) < epsilon * 100) {
        // Eksploracja: wybierz losową akcję
        return random(0, NUM_ACTIONS);
    } else {
        // Eksploatacja: wybierz najlepszą akcję na podstawie tablicy Q
        float maxQ = -INFINITY;
        int bestAction = 0;
        for (int a = 0; a < NUM_ACTIONS; a++) {
            float qValue = qTable[stateIndex][a];
            if (qValue > maxQ) {
                maxQ = qValue;
                bestAction = a;
            }
        }
        return bestAction;
    }
}

// Funkcja aktualizująca tablicę Q
void updateQTable(int stateIndex, int action, float reward, int nextStateIndex, float qTable[][NUM_ACTIONS], float learningRate, float discountFactor) {
    float maxQNext = -INFINITY;
    for (int a = 0; a < NUM_ACTIONS; a++) {
        float qValue = qTable[nextStateIndex][a];
        if (qValue > maxQNext) {
            maxQNext = qValue;
        }
    }
    qTable[stateIndex][action] += learningRate * (reward + discountFactor * maxQNext - qTable[stateIndex][action]);
}

// Funkcja mapująca stan na indeks tablicy Q
int getStateIndex(float error, float load, float Kp, float Ki, float Kd, float MIN_ERROR, float MAX_ERROR, float MIN_LOAD, float MAX_LOAD, float MIN_KP, float MAX_KP, float MIN_KI, float MAX_KI, float MIN_KD, float MAX_KD) {
    int stateError = map(error, MIN_ERROR, MAX_ERROR, 0, NUM_STATE_BINS_ERROR - 1);
    int stateLoad = map(load, MIN_LOAD, MAX_LOAD, 0, NUM_STATE_BINS_LOAD - 1);
    int stateKp = map(Kp, MIN_KP, MAX_KP, 0, NUM_STATE_BINS_KP - 1);
    int stateKi = map(Ki, MIN_KI, MAX_KI, 0, NUM_STATE_BINS_KI - 1);
    int stateKd = map(Kd, MIN_KD, MAX_KD, 0, NUM_STATE_BINS_KD - 1);
    return stateError * NUM_STATE_BINS_LOAD * NUM_STATE_BINS_KP * NUM_STATE_BINS_KI * NUM_STATE_BINS_KD +
           stateLoad * NUM_STATE_BINS_KP * NUM_STATE_BINS_KI * NUM_STATE_BINS_KD +
           stateKp * NUM_STATE_BINS_KI * NUM_STATE_BINS_KD +
           stateKi * NUM_STATE_BINS_KD +
           stateKd;
}


# Cel projektu:

System stabilizacji napięcia z wykorzystaniem trzech agentów uczących się:

* **Agent1:** Odpowiedzialny za stabilizację napięcia wyjściowego 230 volt.
* **Agent2:** Steruje 24 cewkami wzbudzenia w prądnicy, dążąc do utrzymania prądu wzbudzenia na poziomie 25A.
* **Agent3:** Minimalizuje hamowanie, nie wpływając na działanie Agent2 (sterowanie cewkami wzbudzenia).

# Współpraca agentów:

* Agent1 i Agent2 przekazują informacje zwrotne do Agent3, informując go o wpływie swoich akcji na hamowanie.
* Agent3 wykorzystuje te informacje zwrotne do podejmowania decyzji, które minimalizują hamowanie, jednocześnie wspierając cele innych agentów.

# Ostateczny cel:

Osiągnięcie minimalnego hamowania przy jednoczesnym utrzymaniu wysokiego prądu wzbudzenia w cewkach i stabilizacji napięcia na poziomie 230V.

class Agent1 {
public:
    void stabilizeVoltage(float currentVoltage) {
        // Implementacja stabilizacji napięcia
        if (currentVoltage < VOLTAGE_SETPOINT) {
            // Zwiększ napięcie
            analogWrite(mosfetPin, constrain(analogRead(mosfetPin) + PWM_INCREMENT, 0, 255));
        }
        // Usunięto logikę zmniejszania napięcia
    }

    void odbierz_informacje_od_agenta3(float feedback) {
        // Przetwarzanie informacji zwrotnej od Agent3
    }

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

    int chooseAction(int state) {
        if (rand() % 100 < epsilon * 100) {
            return rand() % NUM_ACTIONS;
        } else {
            auto bestAction = std::max_element(qTable[state], qTable[state] + NUM_ACTIONS);
            return std::distance(qTable[state], bestAction);
        }
    }

    void executeAction(int action) {
        const int pins[] = {bjtPin1, bjtPin2, bjtPin3};
        const int increments[] = {PWM_INCREMENT, -PWM_INCREMENT, PWM_INCREMENT, -PWM_INCREMENT, PWM_INCREMENT, -PWM_INCREMENT};

        analogWrite(pins[action % 3], constrain(analogRead(pins[action % 3]) + increments[action], 0, 255));
    }

    float calculateReward(float error, float efficiency, float voltageDrop) {
        float reward = 1.0 / (abs(error) + 1); // Dodanie 1, aby uniknąć dzielenia przez 0
        reward -= voltageDrop * 0.01;
        return reward;
    }

    float calculateAdvancedReward(float next_observation) {
        float napiecie = next_observation;
        float spadek_napiecia = voltageDrop;
        float excitation_current = currentIn[0]; // Przykładowa wartość, dostosuj według potrzeb
        float excitation_current_threshold = 20.0; // Przykładowa wartość, dostosuj według potrzeb
        float nagroda = 0;

        nagroda -= abs(napiecie - VOLTAGE_SETPOINT);
        nagroda -= spadek_napiecia;

        // Zwiększona kara za spadek napięcia
        nagroda -= COMPENSATION_FACTOR * spadek_napiecia; 

        // Kara jeśli akcja powoduje spadek prądu wzbudzenia poniżej progu
        if (excitation_current < excitation_current_threshold) {
            nagroda -= COMPENSATION_FACTOR; 
        }

        // Niewielka nagroda jeśli akcja pomaga zwiększyć prąd wzbudzenia w pożądanym zakresie
        if (excitation_current > excitation_current_threshold && excitation_current <= 23) { 
            nagroda += COMPENSATION_FACTOR; 
        }

        // Dodatkowa nagroda za stabilizację napięcia na poziomie 230V
        if (abs(napiecie - VOLTAGE_SETPOINT) < 1.0) {
            nagroda += 10.0; // Przykładowa wartość nagrody, dostosuj według potrzeb
        }

        return nagroda;
    }

    void updateQ(int state, int action, float reward, int nextState) {
        float bestNextQ = *std::max_element(qTable[nextState], qTable[nextState] + NUM_ACTIONS);
        qTable[state][action] += learningRate * (reward + discountFactor * bestNextQ - qTable[state][action]);
    }

    void monitorPerformanceAndAdjust() {
        static unsigned long lastAdjustmentTime = 0;
        if (millis() - lastAdjustmentTime > 1000) {
            float efficiency = calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);
            float efficiencyPercent = efficiency * 100.0;

            // advancedLogData(efficiencyPercent); // Upewnij się, że ta funkcja jest zaimplementowana

            if (efficiencyPercent < 90.0) {
                excitationGain += 0.1;
            } else if (efficiencyPercent > 95.0) {
                excitationGain -= 0.1;
            }

            excitationGain = constrain(excitationGain, 0.0, 1.0);

            lastAdjustmentTime = millis();
        }
    }

private:
    float qTable[NUM_STATE_BINS_ERROR * NUM_STATE_BINS_LOAD * NUM_STATE_BINS_KP * NUM_STATE_BINS_KI * NUM_STATE_BINS_KD][NUM_ACTIONS];




UWAGA:
 * Agent 2 jest zaprojektowany wyłącznie do STEROWANIA (ZWIĘKSZANIA) prądu wzbudzenia w cewkach.
 * Nie należy dodawać funkcjonalności zmniejszania prądu wzbudzenia ani do Agenta 2, ani do Agenta 3.
 */


 class Agent2 {
public:
    float qTable[NUM_STATES_AGENT2][NUM_ACTIONS_AGENT2];
    int state;
    float feedbackFromAgent3 = 0.0;

    Agent2() {
        for (int i = 0; i < NUM_STATES_AGENT2; i++) {
            for (int j = 0; j < NUM_ACTIONS_AGENT2; j++) {
                qTable[i][j] = 0.0;
            }
        }
        state = 0;
    }

    int chooseAction() {
        if (random(0, 100) < epsilon * 100) {
            return random(0, NUM_ACTIONS_AGENT2);
        } else {
            int bestAction = 0;
            float bestValue = qTable[state][0];
            for (int i = 1; i < NUM_ACTIONS_AGENT2; i++) {
                if (qTable[state][i] > bestValue) {
                    bestValue = qTable[state][i];
                    bestAction = i;
                }
            }
            return bestAction;
        }
    }

    void executeAction(int action) {
        int current = analogRead(muxInputPin);

        switch (action) {
            case 0:
                if (current + PWM_INCREMENT <= MAX_CURRENT) {
                    analogWrite(excitationBJT1Pin, current + PWM_INCREMENT);
                }
                break;
            case 1:
                if (current + PWM_INCREMENT <= MAX_CURRENT) {
                    analogWrite(excitationBJT2Pin, current + PWM_INCREMENT);
                }
                break;
            case 2:
                // Utrzymaj prąd wzbudzenia (bez zmian)
                if (current >= MAX_CURRENT - TOLERANCE) {
                    analogWrite(excitationBJT1Pin, MAX_CURRENT);
                    analogWrite(excitationBJT2Pin, MAX_CURRENT);
                }
                break;
            case 3:
                // Zwiększ prąd wzbudzenia w cewkach 1, 3, 5, 7, 9, 11
                if (current + PWM_INCREMENT <= MAX_CURRENT) {
                    for (int i = 0; i < 24; i += 2) {
                        analogWrite(excitationBJT1Pin, current + PWM_INCREMENT);
                    }
                }
                break;
            default:
                // Nieznana akcja
                break;
        }
    }

    void updateQ(int nextState, float reward, int action) {
        float maxNextQ = *std::max_element(qTable[nextState], qTable[nextState] + NUM_ACTIONS_AGENT2);
        qTable[state][action] += learningRate * (reward + discountFactor * maxNextQ - qTable[state][action]);
    }

    int discretizeState(float error, float load) {
        int errorBin = (int)(error * NUM_STATE_BINS_ERROR / (VOLTAGE_SETPOINT * 2));
        int loadBin = (int)(load * NUM_STATE_BINS_LOAD);
        return errorBin * NUM_STATE_BINS_LOAD + loadBin;
    }

    float reward(float next_observation, float feedbackFromAgent3) {
        const int TARGET_CURRENT = 25;
        float prad_wzbudzenia = next_observation;
        float nagroda = 0.0;

        nagroda -= 2 * abs(TARGET_CURRENT - prad_wzbudzenia);

        if (prad_wzbudzenia >= TARGET_CURRENT - TOLERANCE && prad_wzbudzenia <= TARGET_CURRENT + TOLERANCE) {
            nagroda += 50.0;
        }

        nagroda += 0.5 * feedbackFromAgent3;

        return nagroda;
    }

    void odbierz_informacje_od_agenta3(float feedback) {
        feedbackFromAgent3 = feedback;
    }

    void learn(float error, float load) {
        // Dyskretyzacja stanu
        int currentState = discretizeState(error, load);

        // Wybór akcji
        int action = chooseAction();

        // Wykonanie akcji
        executeAction(action);

        // Odczyt nowego stanu
        float newError = VOLTAGE_SETPOINT - currentVoltage;
        float newLoad = currentCurrent / maxCurrent; // Normalizacja obciążenia

        int nextState = discretizeState(newError, newLoad);

        // Obliczenie nagrody
        float nextObservation = analogRead(muxInputPin);
        float rewardValue = reward(nextObservation, feedbackFromAgent3);

        // Aktualizacja tablicy Q
        updateQ(nextState, rewardValue, action);

        // Aktualizacja stanu
        state = nextState;
    }
};


/*
 * UWAGA:
 * Klasa Agent3 jest zaprojektowana tak, aby zawsze zmniejszać obciążenie prądnicy.
 * Proszę nie zmieniać tej funkcjonalności, aby uniknąć nieporozumień i błędów w działaniu systemu.
 */

class Agent3 {
public:
    int currentBrakePWM = INITIAL_BRAKE_PWM; // Początkowa wartość PWM hamowania
    const int MIN_BRAKE_PWM = 0; // Minimalna wartość PWM hamowania (może być różna od 0)
    const int MAX_BRAKE_PWM = 255; // Maksymalna wartość PWM hamowania

    // Dyskretyzacja stanu (tylko na podstawie PWM hamowania)
    int discretizeStateAgent3() {
        return map(currentBrakePWM, MIN_BRAKE_PWM, MAX_BRAKE_PWM, 0, NUM_STATES_AGENT3 - 1);
    }

    // Funkcja wykonująca akcję agenta 3
    void performAction(float action) {
        // Agent 3 zawsze zmniejsza hamowanie
        currentBrakePWM = max(currentBrakePWM - PWM_INCREMENT, MIN_BRAKE_PWM);
        analogWrite(mosfetPin, currentBrakePWM);
        agent3MinimizedBraking = true; // Ustawienie flagi
    }
};


    // Wybór akcji na podstawie epsilon-greedy policy
    int chooseActionAgent3(int state) {
        if (random(0, 100) < epsilon * 100) {
            return random(0, NUM_ACTIONS_AGENT3); 
        } else {
            int bestAction = 0;
            float bestValue = qTable[state][0];
            for (int i = 1; i < NUM_ACTIONS_AGENT3; i++) {
                if (qTable[state][i] > bestValue) {
                    bestValue = qTable[state][i];
                    bestAction = i;
                }
            }
            return bestAction;
        }
    }

    // Wykonanie akcji
    void executeActionAgent3(int action) {
        const int brakePWMPin = D3; // Przykładowy pin PWM dla hamowania - dostosuj do swojego systemu

        switch (action) {
            case 0:
                // Zmniejsz hamowanie
                currentBrakePWM = constrain(currentBrakePWM - PWM_INCREMENT, MIN_BRAKE_PWM, MAX_BRAKE_PWM);
                break;
            case 1:
                // Zwiększ hamowanie
                currentBrakePWM = constrain(currentBrakePWM + PWM_INCREMENT, MIN_BRAKE_PWM, MAX_BRAKE_PWM);
                break;
        }

        analogWrite(brakePWMPin, currentBrakePWM);
    }

    // Obliczanie nagrody (uwzględniające informacje od innych agentów)
    float calculateRewardAgent3(float feedbackFromAgent1, float feedbackFromAgent2) {
        float reward = -currentBrakePWM; // Podstawowa nagroda za zmniejszanie hamowania

        // Dodatkowa nagroda za współpracę z Agent1 i Agent2
        reward += feedbackFromAgent1 + feedbackFromAgent2;

        return reward;
    }

    void odbierz_informacje_od_agentow(int akcja1, int akcja2) {
        // Tutaj możesz przetworzyć akcje Agent1 i Agent2, 
        // aby dostosować zachowanie Agent3, jeśli to konieczne.
    }

    void updateQAgent3(int state, int action, float reward, int nextState) {
        float bestNextQ = *std::max_element(qTable[nextState], qTable[nextState] + NUM_ACTIONS_AGENT3);
        qTable[state][action] += learningRate * (reward + discountFactor * bestNextQ - qTable[state][action]);
    }

    void odbierz_informacje_hamowania(float hamowanie) {
        // Ta funkcja może pozostać pusta, ponieważ nie ma czujników hamowania
    }

    void wyslij_informacje_do_agenta2(class Agent2& agent2, float feedback) {
        // Ta funkcja może pozostać pusta, ponieważ Agent3 nie wysyła informacji do Agent2
    }

    void wyslij_informacje_do_agenta1(class Agent1& agent1, float feedback) {
        // Ta funkcja może pozostać pusta, ponieważ Agent3 nie wysyła informacji do Agent1
    }

void hillClimbing() {
    // Implementacja algorytmu wspinaczki górskiej
    float currentParams[3] = {Kp, Ki, Kd};
    float bestParams[3];
    getBestParams(bestParams);
    float bestObjective = getBestObjective();

    for (int i = 0; i < 100; i++) {
        suggestNextParameters(currentParams);
        float objective = objectiveFunction(currentParams);
        if (objective > bestObjective) {
            bestObjective = objective;
            for (int j = 0; j < 3; j++) {
                bestParams[j] = currentParams[j];
            }
        }
    }

    for (int j = 0; j < 3; j++) {
        currentParams[j] = bestParams[j];
    }
}

   void discretizeStateAgent3(float state[2], int discreteState[2]) {
    // Normalizacja stanów
    state[0] = constrain(state[0], 0.0, 1.0);
    state[1] = constrain(state[1], 0.0, 1.0);

    // Dyskretyzacja stanów
    discreteState[0] = (int)(state[0] * (NUM_STATES_AGENT3 - 1));
    discreteState[1] = (int)(state[1] * (NUM_STATES_AGENT3 - 1));
}

    float chooseActionAgent3(int discreteState[2]) {
        // Implementacja wyboru akcji dla agenta 3
        int stateIndex = discreteState[0] * NUM_STATE_BINS_LOAD + discreteState[1];
        float bestAction = 0;
        float maxQ = qTableAgent1[stateIndex][0];
        for (int i = 1; i < NUM_ACTIONS_AGENT3; i++) {
            if (qTableAgent1[stateIndex][i] > maxQ) {
                maxQ = qTableAgent1[stateIndex][i];
                bestAction = i;
            }
        }
        return bestAction;
    }

    void executeActionAgent3(float action) {
        // Implementacja wykonania akcji dla agenta 3
        performActionAgent3(action);
    }

private:
    float qTable[NUM_STATES_AGENT3][NUM_ACTIONS_AGENT3] = {0}; // Tablica Q-wartości
};

int main() {
    Agent1 agent1;
    Agent2 agent2;
    Agent3 agent3;

    // Przykładowa logika główna
    float voltage = 230.0;
    float current = 25.0;

    agent3.discretizeStateAgent3(voltage, current);
    int action = agent3.chooseActionAgent3();
    agent3.executeActionAgent3(action);
    float reward = agent3.calculateRewardAgent3(voltage, current);
    int nextState = agent3.currentState; // Zakładając, że stan jest aktualizowany gdzieś indziej
    agent3.updateQ(agent3.currentState, action, reward, nextState, learningRate, discountFactor);

    return 0;
}

      // Przykładowe dane wejściowe
    int states[] = {0, 1, 2};
    int actions[] = {0, 1, 2};
    float rewards[] = {1.0, 0.5, -1.0};
    int nextStates[] = {1, 2, 0};

    while (true) {
        // Aktualizacja Q-wartości dla Agent1 i Agent3
        agent1.updateQ(states[0], actions[0], rewards[0], nextStates[0], learningRate, discountFactor);
        agent3.updateQ(states[2], actions[2], rewards[2], nextStates[2], learningRate, discountFactor);

        // Agent2 działa niezależnie
        agent2.performIndependentActions();

        // Dodaj inne operacje programu tutaj

        // Przykładowe opóźnienie, aby nie wykonywać pętli zbyt często
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}



// Przypisz testowe wartości do używanych zmiennych raz na starcie
static bool initialized = false;
if (!initialized) {
    epsilon = testEpsilon;
    learningRate = testLearningRate;
    discountFactor = testDiscountFactor;
    initialized = true;
}
using System;

class Program
{
    static void Main()
    {
        int totalEpochs = 100; // Całkowita liczba epok
        int completedEpochs = 0; // Ukończone epoki

        // Symulacja procesu nauki
        for (int epoch = 1; epoch <= totalEpochs; epoch++) // Poprawiono błąd składniowy
        {
            // Symulacja treningu AI
            TrainAI(epoch);

            // Aktualizacja ukończonych epok
            completedEpochs = epoch;

            // Obliczenie postępu w procentach
            double progress = (double)completedEpochs / totalEpochs * 100;

            // Wyświetlenie postępu
            Console.WriteLine($"Postęp nauki AI: {progress:F2}%");
        }
    }

    static void TrainAI(int epoch)
    {
        // Symulacja treningu AI (zastąp rzeczywistym kodem treningu)
        System.Threading.Thread.Sleep(50); // Symulacja czasu treningu
    }
}

void advancedLogData() {
    // Przykładowe logowanie danych
    Serial.print("Napięcie wejściowe: ");
    Serial.println(voltageIn[0]);
    Serial.print("Prąd wejściowy: ");
    Serial.println(currentIn[0]);
    Serial.print("Napięcie zewnętrzne: ");
    Serial.println(externalVoltage);
    Serial.print("Prąd zewnętrzny: ");
    Serial.println(externalCurrent);
    Serial.print("Spadek napięcia: ");
    Serial.println(voltageDrop);
    Serial.print("Wydajność: ");
    Serial.println(efficiency);
    Serial.print("Wydajność procentowa: ");
    Serial.println(efficiencyPercent);
    Serial.print("Ostatnia akcja: ");
    Serial.println(lastAction);
}

void detectComputerConnection() {
    static bool isConnected = false;
    if (Serial) {
        if (!isConnected) {
            Serial.println("Połączenie z komputerem nawiązane.");
            isConnected = true;
        }
    } else {
        if (isConnected) {
            Serial.println("Połączenie z komputerem utracone.");
            isConnected = false;
        }
    }
}

void updateOptimizer() {
    unsigned long currentTime = millis();
    if (currentTime - lastOptimizationTime >= OPTIMIZATION_INTERVAL) {
        lastOptimizationTime = currentTime;

        // Pobierz nowe parametry z optymalizatora
        optimizer.getNextParams(params);

        // Oblicz wydajność dla nowych parametrów
        float averageEfficiency = objectiveFunction(params);

        // Zaktualizuj optymalizator z nowymi parametrami i uzyskaną wydajnością
        optimizer.update(params, averageEfficiency);

        // Sprawdź, czy uzyskana wydajność jest najlepsza
        if (averageEfficiency > bestEfficiency) {
            bestEfficiency = averageEfficiency;
            // Zapisz najlepsze parametry
            // Możesz dodać kod do zapisu najlepszych parametrów
        }
    }
}



    // Dodatkowa inicjalizacja
    EEPROM.begin(512); // Inicjalizacja pamięci EEPROM
    handlePIDParams(Kp, Ki, Kd, false); // Wczytaj parametry PID z pamięci EEPROM
    optimizePID(); // Optymalizacja PID na starcie
}

// Inicjalizacja portu szeregowego
void initializeSerial() {
    Serial.begin(115200);
    while (!Serial) {
        ; // Czekaj na połączenie z komputerem
    }
    detectComputerConnection(); // Wykryj połączenie z komputerem i rozpocznij przenoszenie mocy obliczeniowej
}

// Inicjalizacja EEPROM
void initializeEEPROM() {
    EEPROM.begin(512);
}


// Inicjalizacja pinów i innych komponentów
void initializeComponents() {
    initializePins();
    initializeServer();
    initializeDisplay();
    initializeOptimizer();
}

// Wyświetlenie wiadomości powitalnej
void displayWelcome() {
    displayWelcomeMessage();
}

// Inicjalizacja zmiennych globalnych
void initializeGlobals() {
    initializeGlobalVariables();
}

// Inicjalizacja WiFi
void initializeWiFiConnection() {
    initializeWiFi();
}

// Wywołanie funkcji autoCalibrate
void autoCalibrateSystem() {
    autoCalibrate();
}

// Inicjalizacja funkcji setup
void setup() {
    // Inicjalizacja komunikacji szeregowej
    Serial.begin(9600);
    while (!Serial) {
        ; // Czekaj na otwarcie portu szeregowego
    }

    // Inicjalizacja dodatkowej komunikacji szeregowej
    mySerial.begin(9600);

    // Inicjalizacja EEPROM
    EEPROM.begin(512);

    // Przykładowe dane do zapisu w EEPROM
    int address = 0;
    byte value = 42;
    writeEEPROM(address, value);

    // Inicjalizacja mapy funkcji naprawczych
    fixFunctions["Voltage out of range"] = resetVoltageController;
    fixFunctions["Current exceeds maximum limit"] = reduceExcitationCurrent;
    fixFunctions["Braking effect out of range"] = calibrateBrakingEffect;

    // Trenuj model uczenia maszynowego
    trainModel();

    // Inicjalizacja komunikacji I2C
    Wire.begin();

    // Inicjalizacja wyświetlacza
    if (!display.begin(SH1106_I2C_ADDRESS, OLED_RESET)) {
        Serial.println(F("SH1106 allocation failed"));
        for (;;); // Zatrzymanie programu w przypadku niepowodzenia
    }
    display.display();
    delay(2000);
    display.clearDisplay();

    // Inicjalizacja semafora
    xSemaphore = xSemaphoreCreateMutex();

    if (xSemaphore != NULL) {
        // Tworzenie zadań dla agentów
        xTaskCreate(agent1Function, "Agent 1", 1000, NULL, 1, NULL);
        xTaskCreate(agent2Function, "Agent 2", 1000, NULL, 1, NULL);
        xTaskCreate(agent3Function, "Agent 3", 1000, NULL, 1, NULL);
    }
}


    // Inicjalizacja komponentów
    initializePins();
    initializeDisplay();
    initializeServer();
    initializeOptimizer();
    displayWelcomeMessage();
    initializeGlobalVariables();
    initializeWiFi();
    initializeSerial();
    initializeEEPROM();
    initializeComponents();
    displayWelcome();
    initializeGlobals();
    autoCalibrateSystem();
    
    // Inicjalizacja PID
    PID pid(1.0, 0.1, 0.01);
    float setpoint = 230.0; // Docelowe napięcie
    float measuredValue = readVoltage(); // Funkcja do odczytu napięcia

    autoTunePID(pid, setpoint, measuredValue);

    // Teraz możesz używać pid do kontroli
    float output = pid.compute(setpoint, measuredValue);
    analogWrite(mosfetPin, constrain(output, 0, 255));
}

void initializePins() {
    int pins[] = {muxSelectPinA, muxSelectPinB, muxSelectPinC, muxSelectPinD, mosfetPin, bjtPin1, bjtPin2, bjtPin3, excitationBJT1Pin, excitationBJT2Pin, newPin1, newPin2, newPin3, bjtPin4};
    for (int pin : pins) {
        pinMode(pin, OUTPUT);
    }
    pinMode(muxInputPin, INPUT);
}

void initializeServer() {
    server.begin();
}

void initializeDisplay() {
    // Inicjalizacja wyświetlacza
    if (!display.begin(SH1106_I2C_ADDRESS, OLED_RESET)) {
        Serial.println(F("SH1106 allocation failed"));
        for (;;); // Zatrzymanie programu w przypadku niepowodzenia
    }
    display.display();
    delay(2000);
    display.clearDisplay();
}

void initializeOptimizer() {
    optimizer.initialize(3, bounds, 50, 10);
}

void displayWelcomeMessage() {
    char buffer[64];
    strcpy_P(buffer, welcomeMessage);
    display.println(buffer);
    display.display();
}

void initializeGlobalVariables() {
    externalVoltage = 0.0;
    externalCurrent = 0.0;
    efficiency = 0.0;
    efficiencyPercent = 0.0;
    voltageDrop = 0.0;
    currentVoltage = 0.0;
    currentCurrent = 0.0;
}

void initializeWiFi() {
    WiFi.begin("admin", "admin");
    unsigned long startAttemptTime = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < 10000) {
        delay(1000);
        Serial.println("Łączenie z WiFi...");
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("Połączono z WiFi");
        Serial.print("Adres IP: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("Nie udało się połączyć z WiFi");
    }
}

void initializeSerial() {
    // Inicjalizacja dodatkowej komunikacji szeregowej
    mySerial.begin(9600);
}

void initializeEEPROM() {
    // Inicjalizacja EEPROM
    EEPROM.begin(512);
}

void initializeComponents() {
    // Inicjalizacja innych komponentów
    // Dodaj tutaj kod inicjalizujący inne komponenty
}

void displayWelcome() {
    // Wyświetlenie wiadomości powitalnej
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(WHITE);
    display.setCursor(0, 0);
    display.println("Witamy!");
    display.display();
}

void initializeGlobals() {
    // Inicjalizacja zmiennych globalnych
    // Dodaj tutaj kod inicjalizujący zmienne globalne
}

void autoCalibrateSystem() {
    // Automatyczna kalibracja systemu
    // Dodaj tutaj kod kalibracji systemu
}



void initializePIDParams() {
    handlePIDParams(Kp, Ki, Kd, false);
    previousError = 0;
    integral = 0;
}

void initializeQTable() {
    for (int i = 0; i < NUM_STATE_BINS_ERROR * NUM_STATE_BINS_LOAD; i++) {
        for (int j = 0; j < NUM_ACTIONS; j++) {
            qTableAgent1[i][j] = 0.0;
        }
    }
}

void initializeServerRoutes() {
    server.on("/", []() {
        server.send(200, "text/plain", "Witaj w systemie stabilizacji napięcia!");
    });
    server.begin();
}

void initializeRandomSeed() {
    randomSeed(analogRead(0));
}

void displayWelcomeMessageSerial() {
    Serial.println(FPSTR(welcomeMessage));
}

void initializeAgent1State() {
    stateAgent1[0] = 0.0;
    stateAgent1[1] = 0.0;
    actionAgent1 = 0.0;
}

void initializeBayesianOptimization() {
    bestEfficiency = 0.0;
    lastOptimizationTime = millis();
}


    stateAgent1[0] = 0.0;
    stateAgent1[1] = 0.0;
    actionAgent1 = 0.0;

    // Inicjalizacja zmiennych globalnych dla optymalizacji bayesowskiej
    bestEfficiency = 0.0;
    lastOptimizationTime = millis();
}


float objectiveFunction(const float* params) {
    // Przypisanie parametrów PID
    float Kp = params[0];
    float Ki = params[1];
    float Kd = params[2];

    // Symulacja lub rzeczywiste testowanie wydajności systemu
    // Tutaj zakładamy, że mamy funkcję `simulateSystem` która zwraca wydajność
    float efficiency = simulateSystem(Kp, Ki, Kd);

    return efficiency;
}

float evaluate(float threshold) {
    // Implementacja funkcji oceniającej
    // Zwraca wartość oceny dla danego progu
    float voltage = readVoltage();
    if (voltage > threshold) {
        return 1.0; // Próg przekroczony
    } else {
        return 0.0; // Próg nieprzekroczony
    }
}

void selectMuxChannel(int channel) {
    digitalWrite(muxSelectPinA, channel & 1);
    digitalWrite(muxSelectPinB, (channel >> 1) & 1);
    digitalWrite(muxSelectPinC, (channel >> 2) & 1);
    digitalWrite(muxSelectPinD, (channel >> 3) & 1);
}




// Funkcja odczytu sensorów
void readSensors() {
    // Odczyt z czujników napięcia (kanały 0 i 1)
    for (int i = 0; i < 2; i++) {
        selectMuxChannel(i);
        voltageIn[i] = analogRead(muxInputPin) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
    }

    // Odczyt z czujników prądu (kanały 2 i 3)
    for (int i = 0; i < 2; i++) {
        selectMuxChannel(i + 2);

        int raw_current_adc = analogRead(muxInputPin);

        // Obliczanie napięcia wyjściowego czujnika
        float sensorVoltage = raw_current_adc * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);

        // Odejmowanie napięcia spoczynkowego (zakładamy Vcc/2)
        float voltageOffset = VOLTAGE_REFERENCE / 2;
        sensorVoltage -= voltageOffset;

        // Przeliczanie napięcia na prąd (używając czułości 185 mV/A)
        const float sensitivity = 0.185; // Dostosuj, jeśli czułość Twoich czujników jest inna
        currentIn[i] = sensorVoltage / sensitivity;
    }
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
// Dodaj funkcję autoCalibrate
void autoCalibrate() {
    // Przykładowa funkcja automatycznej kalibracji
    Serial.println("Rozpoczynanie automatycznej kalibracji...");

    // Krok 1: Kalibracja napięcia
    float measuredVoltage = readVoltage();
    Serial.print("Zmierzono napięcie: ");
    Serial.println(measuredVoltage);

    // Krok 2: Kalibracja prądu
    float measuredCurrent = readCurrent();
    Serial.print("Zmierzono prąd: ");
    Serial.println(measuredCurrent);

    // Krok 3: Ustawienie parametrów kalibracji
    // Przykładowe ustawienie parametrów kalibracji na podstawie zmierzonych wartości
    float voltageCalibrationFactor = VOLTAGE_REFERENCE / measuredVoltage;
    float currentCalibrationFactor = MAX_CURRENT / measuredCurrent;

    Serial.print("Współczynnik kalibracji napięcia: ");
    Serial.println(voltageCalibrationFactor);
    Serial.print("Współczynnik kalibracji prądu: ");
    Serial.println(currentCalibrationFactor);

    // Zapisanie współczynników kalibracji do pamięci EEPROM
    EEPROM.begin(512);
    EEPROM.put(0, voltageCalibrationFactor);
    EEPROM.put(sizeof(voltageCalibrationFactor), currentCalibrationFactor);
    EEPROM.commit();

    Serial.println("Kalibracja zakończona.");
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

void updateDisplay(float efficiencyPercent) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(WHITE, BLACK); // Ustawienie koloru tekstu na biały na czarnym tle
    display.setCursor(0, 0);
    display.println("System Status:");
    display.print("Napięcie: ");
    display.print(readVoltage());
    display.println(" V");
    display.print("Prąd: ");
    display.print(readExcitationCurrent());
    display.println(" A");
    display.print("Wydajność: ");
    display.print(efficiencyPercent);
    display.println(" %");
    display.print("Braking Effect: ");
    display.print(readBrakingEffect());
    display.println(" %");
    display.display();
}




  void handleSerialCommands() {
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        command.trim(); // Usuwa białe znaki na początku i końcu

        if (command == "START") {
            // Przykład: obsługa komendy START
            Serial.println("Komenda START otrzymana");
            // Dodaj tutaj kod do uruchomienia odpowiedniej funkcji
        } else if (command == "STOP") {
            // Przykład: obsługa komendy STOP
            Serial.println("Komenda STOP otrzymana");
            // Dodaj tutaj kod do zatrzymania odpowiedniej funkcji
        } else if (command == "OPTIMIZE") {
            // Przykład: obsługa komendy OPTIMIZE
            Serial.println("Komenda OPTIMIZE otrzymana");
            optimizePID();
        } else {
            Serial.println("Nieznana komenda: " + command);
        }
    }
}
 


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

// Definicja funkcji obliczającej wydajność
float calculateEfficiency(float voltageIn, float currentIn, float externalVoltage, float externalCurrent) {
    float inputPower = voltageIn * currentIn;
    float outputPower = externalVoltage * externalCurrent;

    if (inputPower == 0) {
        return 0;
    }

    return outputPower / inputPower;
}


void adjustControlFrequency() {
    static unsigned long lastAdjustmentTime = 0;
    if (millis() - lastAdjustmentTime > 1000) {

        if (stopCompute) { // Sprawdź flagę stopCompute
            return; // Jeśli obliczenia są zatrzymane, wyjdź z funkcji
        }

        if (currentIn[0] > LOAD_THRESHOLD) {
            controlFrequency = HIGH_FREQUENCY;
        } else {
            controlFrequency = LOW_FREQUENCY;
        }
        lastAdjustmentTime = millis();
    }
}

void loop() {
    unsigned long currentMillis = millis();

    // Aktualizacja wyświetlacza
    if (currentMillis - previousMillisDisplay >= intervalDisplay) {
        previousMillisDisplay = currentMillis;
        updateDisplay();
    }

    // Obsługa komunikacji szeregowej
    if (currentMillis - previousMillisSerial >= intervalSerial) {
        previousMillisSerial = currentMillis;
        handleSerialCommunication();
    }

    // Obsługa serwera
    server.handleClient();

    // Przykładowe wywołanie funkcji optymalizacji
    optimize();

    // Przykładowe wywołanie funkcji auto-strojenia PID
    PID pid(Kp, Ki, Kd);
    autoTunePID(pid, VOLTAGE_SETPOINT, readVoltage());

    // Przykładowe wywołanie funkcji hill climbing
    hillClimbing();

    // Przykładowe wywołanie funkcji agenta 3
    float state[2] = {readVoltage(), readExcitationCurrent()};
    int discreteState[2];
    discretizeStateAgent3(state, discreteState);
    float action = chooseActionAgent3(discreteState);
    executeActionAgent3(action);
    float reward = calculateRewardAgent3(state, action);
    float nextState[2] = {readVoltage(), readExcitationCurrent()};
    updateQAgent3(state, action, reward, nextState);

    // Krótkie opóźnienie, aby symulować czas rzeczywisty
    delay(100);

    // Aktualizacja bieżących wartości napięcia i prądu
    updateCurrentValues();

    unsigned long currentTime = millis();

    // Regularne wywoływanie optymalizacji PID
    if (currentTime - lastOptimizationTime > OPTIMIZATION_INTERVAL) {
        lastOptimizationTime = currentTime;

        // Wykonanie kroku optymalizacji
        optimizer.optimize();

        // Pobranie najlepszych parametrów
        optimizer.getBestParams(params);
        bestEfficiency = -optimizer.getBestObjective();

        // Aktualizacja parametrów PID
        Kp = params[0];
        Ki = params[1];
        Kd = params[2];

        Serial.println("Optymalizacja zakończona:");
        Serial.print("Kp: "); Serial.println(Kp);
        Serial.print("Ki: "); Serial.println(Ki);
        Serial.print("Kd: "); Serial.println(Kd);
        Serial.print("Wydajność: "); Serial.println(bestEfficiency);
    }

    // Wysyłanie danych do komputera
    if (Serial) {
        float dataToSend = analogRead(A0); // Przykładowe dane do wysłania
        Serial.println(dataToSend); // Wysyłanie danych do komputera
        delay(1000); // Opóźnienie dla przykładu
    }
}

// Funkcja do aktualizacji wartości napięcia i prądu
void updateCurrentValues() {
    currentVoltage = readVoltage();
    currentCurrent = readExcitationCurrent();
}

const float VOLTAGE_CONVERSION_FACTOR = 5.0 / 1023.0;

float readVoltage() {
    // Przykładowa implementacja funkcji odczytu napięcia
    int rawValue = analogRead(A0);
    float voltage = rawValue * VOLTAGE_CONVERSION_FACTOR;
    return voltage;
}

float readExcitationCurrent() {
    // Przykładowa implementacja funkcji odczytu prądu wzbudzenia
    int rawValue = analogRead(A1);
    float current = rawValue * VOLTAGE_CONVERSION_FACTOR;
    return current;
}

float readBrakingEffect() {
    // Przykładowa implementacja funkcji odczytu efektu hamowania
    int rawValue = analogRead(A2);
    float brakingEffect = rawValue * VOLTAGE_CONVERSION_FACTOR;
    return brakingEffect;
}



    // Przykładowa logika sterowania
    if (currentCurrent > LOAD_THRESHOLD) {
        analogWrite(excitationBJT1Pin, 255);
    } else {
        analogWrite(excitationBJT1Pin, 0);
    }

    // Monitorowanie błędów i wykrywanie anomalii
    monitorErrors();
    detectAnomalies();
    delay(1000); // Sprawdzaj błędy co 1 sekundę
}

void monitorErrors() {
    checkVoltage();
    checkCurrent();
    checkBrakingEffect();
}

void checkVoltage() {
    float voltage = readVoltage();
    if (voltage < 220.0 || voltage > 240.0) {
        generateAndImplementFix("Voltage out of range");
    }
}

void checkCurrent() {
    float current = readExcitationCurrent();
    if (current > MAX_CURRENT) {
        generateAndImplementFix("Current exceeds maximum limit");
    }
}

void checkBrakingEffect() {
    float brakingEffect = readBrakingEffect();
    if (brakingEffect < 0.0 || brakingEffect > 100.0) {
        generateAndImplementFix("Braking effect out of range");
    }
}

void generateAndImplementFix(String error) {
    Serial.println("Error detected: " + error);
    logError(error);
    if (fixFunctions.find(error) != fixFunctions.end()) {
        fixFunctions[error](); // Wywołanie odpowiedniej funkcji naprawczej
    } else {
        defaultFixFunction(); // Wywołanie domyślnej funkcji naprawczej
    }
}

void resetVoltageController() {
    Serial.println("Resetting voltage controller...");
    // Kod resetowania kontrolera
}

void reduceExcitationCurrent() {
    Serial.println("Reducing excitation current...");
    // Kod zmniejszenia prądu wzbudzenia
}

void calibrateBrakingEffect() {
    Serial.println("Calibrating braking effect...");
    // Kod kalibracji efektu hamowania
}


void logError(String error) {
    int addr = 0;
    while (addr < 512) {
        if (EEPROM.read(addr) == 0) {
            if (addr + error.length() + 1 < 512) { // Sprawdzenie, czy jest wystarczająco dużo miejsca
                for (int i = 0; i < error.length(); i++) {
                    EEPROM.write(addr + i, error[i]);
                }
                EEPROM.write(addr + error.length(), 0); // Zakończenie stringa
                EEPROM.commit();
            }
            break;
        }
        addr += error.length() + 1;
    }
}

void readErrorLog() {
    int addr = 0;
    while (addr < 512) {
        char c = EEPROM.read(addr);
        if (c == 0) break;
        String error = "";
        while (c != 0) {
            error += c;
            addr++;
            c = EEPROM.read(addr);
        }
        Serial.println("Logged error: " + error);
        addr++;
    }
}

void detectAnomalies() {
    float voltage = readVoltage();
    float current = readExcitationCurrent();
    float brakingEffect = readBrakingEffect();

    // Wykorzystanie modelu uczenia maszynowego do przewidywania błędów
    String predictedError = predictWithModel(voltage, current, brakingEffect);
    if (predictedError != "") {
        handleAnomaly(predictedError);
    }
}

void handleAnomaly(String anomaly) {
    Serial.println("Anomaly detected: " + anomaly);
    logError(anomaly);
    generateAndImplementFix(anomaly); // Wywołanie funkcji naprawczej dla anomalii
}

String predictError(float voltage, float current, float brakingEffect) {
    // Prosta logika oparta na regułach
    if (voltage < 220.0 || voltage > 240.0) {
        return "Voltage out of range";
    }
    if (current > MAX_CURRENT) {
        return "Current exceeds maximum limit";
    }
    if (brakingEffect < 0.0 || brakingEffect > 100.0) {
        return "Braking effect out of range";
    }
    return "";
}

void addFixFunction(String error, void(*fixFunction)()) {
    fixFunctions[error] = fixFunction;
}

void defaultFixFunction() {
    Serial.println("No specific fix available. Executing default fix...");
    // Kod domyślnej naprawy
}

void trainModel() {
    // Przykładowe trenowanie modelu (w rzeczywistości powinno być bardziej zaawansowane)
    modelWeights[0] = 1.2; // Waga dla napięcia
    modelWeights[1] = 1.5; // Waga dla prądu
    modelWeights[2] = 0.8; // Waga dla efektu hamowania
    modelWeights[3] = 2.0; // Dodatkowa waga dla bardziej złożonego modelu
}

String predictWithModel(float voltage, float current, float brakingEffect) {
    // Prosty model liniowy do przewidywania błędów
    float score = modelWeights[0] * voltage + modelWeights[1] * current + modelWeights[2] * brakingEffect + modelWeights[3] * (voltage * current);
    if (score > 1500) { // Zwiększony próg dla wykrywania większych anomalii
        return "Anomaly detected by model";
    }
    return "";
}

void updateDisplay() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(WHITE);
    display.setCursor(0, 0);
    display.println("System Status:");
    display.println("Voltage: " + String(readVoltage()) + " V");
    display.println("Current: " + String(readExcitationCurrent()) + " A");
    display.println("Braking Effect: " + String(readBrakingEffect()));
    display.display();
}



    // Inne operacje w pętli głównej
    handleSerialCommands();
    detectComputerConnection();
    trainAgent1();
    trainAgent2();
    communicateBetweenAgents();
    controlTransistors(currentVoltage, currentCurrent);

    // Testowanie różnych wartości epsilon, learningRate i discountFactor
    static const float testEpsilon = 0.3;
    static const float testLearningRate = 0.01;
    static const float testDiscountFactor = 0.95;

    // Przypisz testowe wartości do używanych zmiennych raz na starcie
    static bool initialized = false;
    if (!initialized) {
        epsilon = testEpsilon;
        learningRate = testLearningRate;
        discountFactor = testDiscountFactor;
        initialized = true;
    }

    // Przykładowe wywołanie funkcji hillClimbing
    hillClimbing();
    
    // Dodanie opóźnienia, aby nie przeciążać procesora
    delay(1000); // Opóźnienie 1 sekundy

    // Przykładowe wywołanie funkcji simulateVoltageControl
    float Kp = 1.0, Ki = 0.5, Kd = 0.1;
    float voltageError = simulateVoltageControl(Kp, Ki, Kd);
    Serial.println(voltageError);

    // Dodanie opóźnienia, aby nie przeciążać procesora
    delay(1000); // Opóźnienie 1 sekundy

    // Przykładowe wywołanie funkcji simulateExcitationControl
    float excitationError = simulateExcitationControl();
    Serial.println(excitationError);

    // Dodanie opóźnienia, aby nie przeciążać procesora
    delay(1000); // Opóźnienie 1 sekundy

    // Przykładowe wywołanie funkcji someFunctionOfOtherParameters
    float rotationalSpeed = 3000.0;
    float torque = 50.0;
    float frictionCoefficient = 0.8;
    float brakingEffect = someFunctionOfOtherParameters(rotationalSpeed, torque, frictionCoefficient);
    Serial.println(brakingEffect);

    // Dodanie opóźnienia, aby nie przeciążać procesora
    delay(1000); // Opóźnienie 1 sekundy

    // Sprawdzenie, czy są dostępne dane do odczytu
    if (mySerial.available()) {
        String command = mySerial.readStringUntil('\n');
        if (command == "START_COMPUTE") {
            Serial.println("Rozpoczęcie obliczeń");
            stopCompute = false; // Reset flagi przed rozpoczęciem obliczeń

            while (true) {
                // Wysłanie danych do komputera
                mySerial.println("DANE_DO_PRZETWORZENIA");

                // Oczekiwanie na potwierdzenie
                while (!mySerial.available()) {
                    // Czekanie na potwierdzenie
                    if (stopCompute) {
                        break;
                    }
                }

                if (stopCompute) {
                    break;
                }

                String ack = mySerial.readStringUntil('\n');
                if (ack == "ACK") {
                    Serial.println("Potwierdzenie odebrane");
                }

                // Warunek wyjścia z pętli
                if (stopCompute) {
                    break;
                }
            }
        }
    }

    // Przykładowe zmienne
    static unsigned long previousMillis = 0;
    const long interval = 1000; // Interwał w milisekundach
    unsigned long currentMillis = millis();

    // Warunek sprawdzający, czy upłynął określony czas
    if (currentMillis - previousMillis >= interval) {
        previousMillis = currentMillis;

        // Operacje, które mają być wykonywane co określony interwał czasu
        float voltage = readVoltage();
        float current = readExcitationCurrent();
        float brakingEffect = readBrakingEffect();

        // Przykładowe obliczenia
        float error = calculateError(voltage, current, brakingEffect);

        // Aktualizacja zmiennych tylko wtedy, gdy jest to konieczne
        if (error > TOLERANCE) {
            // Wykonaj operacje tylko wtedy, gdy błąd przekracza margines tolerancji
            updateControlParameters(error);
        }
    }

    // Inne operacje, które mogą być wykonywane w pętli loop
    handleSerialCommunication();
    updateDisplay();
}

// Przykładowa funkcja obliczająca błąd
float calculateError(float voltage, float current, float brakingEffect) {
    // Przykładowe obliczenia błędu
    return abs(voltage - current - brakingEffect);
}

// Przykładowa funkcja aktualizująca parametry sterowania
void updateControlParameters(float error) {
    // Przykładowa aktualizacja parametrów sterowania
    float newKp = Kp + error * 0.1;
    float newKi = Ki + error * 0.01;
    float newKd = Kd + error * 0.001;
    pid.setTunings(newKp, newKi, newKd);
}

// Przykładowa funkcja obsługująca komunikację szeregową
void handleSerialCommunication() {
    if (Serial.available() > 0) {
        String input = Serial.readString();
        Serial.println("Received: " + input);
    }
}



    // Wywołanie funkcji hillClimbing
    float currentThreshold = 0.5; // Przykładowa wartość początkowa
    float stepSize = 0.1; // Przykładowy rozmiar kroku
    float optimizedThreshold = hillClimbing(currentThreshold, stepSize);
}


    // Obliczanie efektywności i innych parametrów na podstawie aktualnych danych z sensorów
    efficiency = calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);
    efficiencyPercent = efficiency * 100.0;
    voltageDrop = voltageIn[1] - voltageIn[0];

    // Bayesian optimization (at a certain interval)
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

            // Stała dla adresu początku tablicy Q
            const int Q_TABLE_START_ADDRESS = sizeof(lastOptimizationTime);

            // Zapisujemy czas ostatniej optymalizacji
            EEPROM.put(0, lastOptimizationTime);

            // Zapisujemy tablicę Q-learning, sprawdzając każdy bajt
            int qTableSize = sizeof(qTable);
            for (int i = 0; i < qTableSize; i++) {
                EEPROM.update(i + Q_TABLE_START_ADDRESS, ((byte*)qTable)[i]);
            }

            // Zatwierdzamy zmiany w EEPROM
            if (!EEPROM.commit()) {
                Serial.println("Error committing changes to EEPROM!");
                return; // Przerywamy dalsze wykonywanie w przypadku błędu
            }

            Serial.println("Saved Q-learning table and lastOptimizationTime to EEPROM.");
        }
    }
}



    // Aktualizacja agentów
if (currentTime - lastUpdateAgent1 >= updateInterval) {
    float error = VOLTAGE_SETPOINT - currentVoltage;
    float load = currentCurrent;
    qLearningAgent1(error, load, Kp, Ki, Kd);
    lastUpdateAgent1 = currentTime;
}

if (currentTime - lastUpdateAgent2 >= updateInterval) {
    float error = VOLTAGE_SETPOINT - currentVoltage;
    float load = currentCurrent;
    qLearningAgent2(error, load, Kp, Ki, Kd);
    lastUpdateAgent2 = currentTime;
}

if (currentTime - lastUpdateAgent3 >= updateInterval) {
    float error = VOLTAGE_SETPOINT - currentVoltage;
    float load = currentCurrent;
    qLearningAgent3(error, load, Kp, Ki, Kd);
    lastUpdateAgent3 = currentTime;
}

// Hill Climbing optimization of excitation phase switching threshold
LOAD_THRESHOLD = hillClimbing(LOAD_THRESHOLD, 0.01, evaluateThreshold);
Serial.print("Updated excitation phase switching threshold: ");
Serial.println(LOAD_THRESHOLD);

// Display data on the screen
displayData(efficiencyPercent);

// Adjust control frequency
adjustControlFrequency();

// Monitor transistors
monitorTransistors();

// Monitor performance and adjust control
monitorPerformanceAndAdjust();

// Communication with the computer
if (Serial.available() > 0) {
    efficiency = Serial.parseFloat();
    efficiencyPercent = efficiency * 100.0;
    voltageDrop = Serial.parseFloat();
}

advancedLogData(efficiencyPercent); // Wywołanie funkcji advancedLogData
delay(1000); // Przykładowe opóźnienie, aby nie logować zbyt często

checkAlarm();
autoCalibrate();
energyManagement();

// Q-learning 1 (voltage stabilizer)
int state1 = discretizeState(VOLTAGE_SETPOINT - voltageIn[0], currentIn[0], Kp, Ki, Kd);
int action1 = chooseAction(state1);
executeAction(action1);
float reward1 = calculateReward(VOLTAGE_SETPOINT - voltageIn[0], efficiency, voltageDrop);
int nextState1 = discretizeState(VOLTAGE_SETPOINT - voltageIn[0], currentIn[0], Kp, Ki, Kd);
updateQ(state1, action1, reward1, nextState1);

// Q-learning 2 (excitation coils)
int state2 = discretizeState(VOLTAGE_SETPOINT - voltageIn[0], currentIn[0], Kp, Ki, Kd);
int action2 = chooseAction(state2);
executeAction(action2);
float reward2 = calculateReward(VOLTAGE_SETPOINT - voltageIn[0], efficiency, voltageDrop);
int nextState2 = discretizeState(VOLTAGE_SETPOINT - voltageIn[0], currentIn[0], Kp, Ki, Kd);
updateQ(state2, action2, reward2, nextState2);

// Q-learning 3 (generator braking)
float power_output = externalVoltage * externalCurrent; // Obliczamy moc wyjściową generatora
int state3 = discretizeStateAgent3(VOLTAGE_SETPOINT - voltageIn[0], currentIn[0]);
int action3 = chooseActionAgent3(state3);
executeActionAgent3(action3);

// Przekazujemy wszystkie 4 argumenty do funkcji calculateRewardAgent3
float reward3 = calculateRewardAgent3(efficiency, voltageIn[0], voltageDrop, power_output);

int nextState3 = discretizeStateAgent3(VOLTAGE_SETPOINT - voltageIn[0], currentIn[0]);
updateQAgent3(state3, action3, reward3, nextState3);

// Dostosuj minimalny próg mocy wejściowej - dodane tutaj
float inputPower = voltageIn[0] * currentIn[0];
adjustMinInputPower(inputPower);

// Obsługa serwera
    server.handleClient();

    // Obsługa komend szeregowych
    handleSerialCommands();

    // Trening agenta 1
    trainAgent1();

    // Aktualizacja agenta 3
    qLearningAgent3();

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

// Globalna zmienna dla minimalnej mocy wejściowej
float minInputPower = 1e-7; // Początkowa wartość, dostosowana w funkcji

// Funkcja do automatycznego dostosowywania progu (umieszczona poza pętlą loop)
void adjustMinInputPower(float inputPower) {
    static float minObservedPower = 1e-6;
    static float maxObservedPower = 1e-3;

 // Aktualizuj minimalną i maksymalną obserwowaną moc
if (inputPower > 0 && inputPower < minObservedPower) {
    minObservedPower = inputPower;
}

    if (inputPower > maxObservedPower) {
        maxObservedPower = inputPower;
    }

    // Dostosuj próg na podstawie obserwowanych wartości
    minInputPower = minObservedPower * 0.1; // Możesz dostosować współczynnik 0.1
}

void detectComputerConnection() {
    if (Serial.available()) {
        Serial.println("Komputer podłączony. Przenoszenie mocy obliczeniowej...");
        // Wyślij komendę do komputera, aby rozpocząć przenoszenie mocy obliczeniowej
        Serial.println("START_COMPUTE");
    }
}
