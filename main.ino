#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Arduino.h>
#include <EEPROM.h>
#include <Wire.h>
#include <Adafruit_SH1106.h>
#include <ArduinoEigen.h>
#include <BayesOptimizer.h>

// Definicje stałych
#define MAX_CURRENT 25 // Maksymalny prąd wzbudzenia w amperach
#define TOLERANCE 0.5 // Margines tolerancji
#define NUM_STATES_AGENT2 100 // Definicja liczby stanów dla Agenta 2


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

// Deklaracje funkcji
void hillClimbing();
void discretizeStateAgent3(float state[2], int discreteState[2]);
float chooseActionAgent3(int discreteState[2]);
void executeActionAgent3(float action);
float calculateRewardAgent3(float state[2], float action);
void updateQAgent3(float state[2], float action, float reward, float nextState[2]);
float objectiveFunction(float params[3]);
void optimize();
void getBestParams(float params[3]);
float getBestObjective();
void suggestNextParameters(float params[3]);

// Definicje stałych
#define MAX_CURRENT 25
#define TOLERANCE 0.5
#define NUM_STATES_AGENT2 100

// Definicje pinów
const int mosfetPin = D4;
const int excitationBJT1Pin = D8;
const int excitationBJT2Pin = D9;
const int bjtPin1 = D5;
const int bjtPin2 = D6;
const int bjtPin3 = D7;
const int muxSelectPinA = D0;
const int muxSelectPinB = D1;
const int muxSelectPinC = D2;
const int muxSelectPinD = D3;
const int muxInputPin = A0;
const int newPin1 = D10;
const int newPin2 = D11;
const int newPin3 = D12;
const int bjtPin4 = D13;

// Definicje zmiennych
float testEpsilon = 0.2;
float testLearningRate = 0.05;
float testDiscountFactor = 0.95;
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

// Deklaracje funkcji
void hillClimbing();
void discretizeStateAgent3(float state[2], int discreteState[2]);
float chooseActionAgent3(int discreteState[2]);
void executeActionAgent3(float action);
float calculateRewardAgent3(float state[2], float action);
void updateQAgent3(float state[2], float action, float reward, float nextState[2]);
float objectiveFunction(float params[3]);
float simulateVoltageControl(float Kp, float Ki, float Kd);
float simulateExcitationControl();
float simulateBrakingEffect();
float readVoltage();
float readExcitationCurrent();
float readBrakingEffect();
float someFunctionOfOtherParameters(float rotationalSpeed, float torque, float frictionCoefficient);

// Implementacja funkcji
void hillClimbing() {
    float currentParams[3] = {1.0, 1.0, 1.0}; // Początkowe parametry
    float bestParams[3];
    memcpy(bestParams, currentParams, sizeof(currentParams));
    float bestObjective = objectiveFunction(currentParams);

    for (int i = 0; i < 100; i++) { // Liczba iteracji
        float newParams[3];
        for (int j = 0; j < 3; j++) {
            newParams[j] = currentParams[j] + random(-1, 2) * 0.1; // Mała zmiana parametru
        }
        float newObjective = objectiveFunction(newParams);
        if (newObjective > bestObjective) {
            memcpy(bestParams, newParams, sizeof(newParams));
            bestObjective = newObjective;
        }
    }
    memcpy(currentParams, bestParams, sizeof(bestParams));
}

void discretizeStateAgent3(float state[2], int discreteState[2]) {
    discreteState[0] = (int)(state[0] * 10); // Dyskretyzacja stanu 1
    discreteState[1] = (int)(state[1] * 10); // Dyskretyzacja stanu 2
}

float chooseActionAgent3(int discreteState[2]) {
    return (float)(discreteState[0] + discreteState[1]) / 2.0; // Przykładowa logika wyboru akcji
}

void executeActionAgent3(float action) {
    analogWrite(mosfetPin, (int)(action * 255.0)); // Wykonanie akcji przez ustawienie PWM
}

float calculateRewardAgent3(float state[2], float action) {
    float target = 1.0; // Docelowy stan
    float error = target - (state[0] + state[1]) / 2.0;
    return -abs(error); // Nagroda to negatywna wartość błędu
}

void updateQAgent3(float state[2], float action, float reward, float nextState[2]) {
    int stateIndex = (int)(state[0] * 10 + state[1] * 10);
    int nextStateIndex = (int)(nextState[0] * 10 + nextState[1] * 10);
    int actionIndex = (int)(action * 10);

    float currentQ = qTableAgent1[stateIndex][actionIndex];
    float maxNextQ = qTableAgent1[nextStateIndex][0];
    for (int i = 1; i < NUM_ACTIONS_AGENT3; i++) {
        if (qTableAgent1[nextStateIndex][i] > maxNextQ) {
            maxNextQ = qTableAgent1[nextStateIndex][i];
        }
    }
    qTableAgent1[stateIndex][actionIndex] = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);
}

// Przykładowa funkcja celu
float objectiveFunction(float params[3]) {
    float Kp = params[0];
    float Ki = params[1];
    float Kd = params[2];

    // Symulacja systemu z użyciem parametrów PID
    float voltageError = simulateVoltageControl(Kp, Ki, Kd);
    float excitationCurrentError = simulateExcitationControl();
    float brakingEffect = simulateBrakingEffect();

    // Funkcja celu: minimalizacja błędu napięcia, minimalizacja błędu prądu wzbudzenia i minimalizacja hamowania
    float objective = -voltageError - excitationCurrentError - brakingEffect;

    return objective;
}

// Przykładowa funkcja symulująca kontrolę napięcia
float simulateVoltageControl(float Kp, float Ki, float Kd) {
    float error = 0.0;
    float setpoint = 230.0; // Docelowe napięcie 230V
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

// Przykładowa funkcja symulująca kontrolę prądu wzbudzenia
float simulateExcitationControl() {
    float setpoint = 25.0; // Docelowy prąd wzbudzenia 25A
    float measuredValue = readExcitationCurrent();
    float error = abs(setpoint - measuredValue);

    return error;
}

// Przykładowa funkcja modelująca efekt hamowania na podstawie parametrów prądnicy
float someFunctionOfOtherParameters(float rotationalSpeed, float torque, float frictionCoefficient) {
    // Model matematyczny: efekt hamowania jest proporcjonalny do prędkości obrotowej i momentu obrotowego,
    // a odwrotnie proporcjonalny do współczynnika tarcia
    float simulatedBrakingEffect = (rotationalSpeed * torque) / (frictionCoefficient * 10.0);
    return simulatedBrakingEffect;
}

// Przykładowa funkcja symulująca efekt hamowania bez czujników
float readBrakingEffect() {
    // Zamiast odczytu z czujnika, używamy zaawansowanego modelu matematycznego
    float rotationalSpeed = 3000.0; // Przykładowa prędkość obrotowa w RPM
    float torque = 50.0; // Przykładowy moment obrotowy w Nm
    float frictionCoefficient = 0.8; // Przykładowy współczynnik tarcia

    float simulatedBrakingEffect = someFunctionOfOtherParameters(rotationalSpeed, torque, frictionCoefficient);

    return simulatedBrakingEffect;
}

// Przykładowa funkcja symulująca efekt hamowania
float simulateBrakingEffect() {
    float brakingEffect = readBrakingEffect();
    
    // Zakładamy, że mniejsza wartość efektu hamowania jest lepsza
    // Zastosowanie funkcji penalizującej większe wartości
    float penalizedBrakingEffect = 1.0 / (1.0 + brakingEffect);

    return penalizedBrakingEffect;
}

// Przykładowe funkcje odczytu wartości
float readVoltage() {
    // Implementacja odczytu napięcia
    return analogRead(A0) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
}

float readExcitationCurrent() {
    // Implementacja odczytu prądu wzbudzenia
    return analogRead(A1) * (MAX_CURRENT / ADC_MAX_VALUE);
}




// Przykładowe funkcje odczytu wartości
float readVoltage() {
    // Implementacja odczytu napięcia
    return analogRead(A0) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
}

float readExcitationCurrent() {
    // Implementacja odczytu prądu wzbudzenia
    return analogRead(A1) * (MAX_CURRENT / ADC_MAX_VALUE);
}

float readBrakingEffect() {
    // Implementacja odczytu efektu hamowania
    return analogRead(A2) * (MAX_BRAKING_EFFECT / ADC_MAX_VALUE);
}


void optimize() {
    float params[3] = {1.0, 1.0, 1.0};
    float bestParams[3];
    memcpy(bestParams, params, sizeof(params));
    float bestObjective = objectiveFunction(params);

    for (int i = 0; i < 100; i++) {
        suggestNextParameters(params);
        float objective = objectiveFunction(params);
        if (objective > bestObjective) {
            memcpy(bestParams, params, sizeof(params));
            bestObjective = objective;
        }
    }

    // Ustawienie najlepszych znalezionych parametrów
    getBestParams(bestParams);
}

// Przykładowa funkcja celu
float objectiveFunction(float params[3]) {
    return params[0] * params[1] - params[2]; // Przykładowa funkcja celu
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


void getBestParams(float params[3]) {
    params[0] = 1.0; // Przykładowe najlepsze parametry
    params[1] = 1.0;
    params[2] = 1.0;
}

float getBestObjective() {
    return 1.0; // Przykładowa najlepsza wartość funkcji celu
}

void suggestNextParameters(float params[3]) {
    for (int i = 0; i < 3; i++) {
        params[i] += random(-1, 2) * 0.1; // Sugerowanie nowych parametrów
    }
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
float learningRate = 0.1;
float discountFactor = 0.9;

// Funkcje do odczytu napięcia i prądu
float readVoltage() {
    // Kod do odczytu napięcia z czujnika
    return analogRead(A0) * (5.0 / 1023.0); // Przykład dla czujnika analogowego
}

float readCurrent() {
    // Kod do odczytu prądu z czujnika
    return analogRead(A1) * (5.0 / 1023.0); // Przykład dla czujnika analogowego
}

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
    float currentVoltage = 0.0;
    float currentError = 0.0;
    float previousError = 0.0;
    float integral = 0.0;
    float derivative = 0.0;
    float controlSignal = 0.0;
    float simulatedEfficiency = 0.0;
    int simulationSteps = 100; // Liczba kroków symulacji
    float timeStep = 0.1; // Krok czasowy symulacji

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
float learningRate = 0.1;
float discountFactor = 0.9;

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



// Funkcja obliczająca wydajność
float calculateEfficiency(float voltageIn, float currentIn, float externalVoltage, float externalCurrent) {
    float inputPower = voltageIn * currentIn;
    float outputPower = externalVoltage * externalCurrent;

    if (inputPower == 0) {
        return 0;
    }

    return outputPower / inputPower;
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

    if (oscillationsDetected) {
        Kp = 0.6 * Ku;
        Ki = 2 * Kp / Tu;
        Kd = Kp * Tu / 8;
        Serial.println("Optymalizacja PID zakończona:");
        Serial.print("Kp: "); Serial.println(Kp);
        Serial.print("Ki: "); Serial.println(Ki);
        Serial.print("Kd: "); Serial.println(Kd);
        savePIDParams(Kp, Ki, Kd);
        Serial.print("Zapisane parametry PID: Kp="); Serial.print(Kp);
        Serial.print(", Ki="); Serial.print(Ki);
        Serial.print(", Kd="); Serial.println(Kd);
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
    // Parametry Q-learning
    const float learningRate = 0.1;
    const float discountFactor = 0.9;
    const float epsilon = 0.3; // Współczynnik eksploracji

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

    float calculateEfficiency(float voltageIn, float currentIn, float externalVoltage, float externalCurrent) {
        // Przykładowa implementacja funkcji calculateEfficiency
        return (voltageIn * currentIn) / (externalVoltage * externalCurrent);
    }
};



#define NUM_STATES_AGENT2 100
#define NUM_ACTIONS_AGENT2 5

UWAGA:
 * Agent 2 jest zaprojektowany wyłącznie do STEROWANIA (ZWIĘKSZANIA) prądu wzbudzenia w cewkach.
 * Nie należy dodawać funkcjonalności zmniejszania prądu wzbudzenia ani do Agenta 2, ani do Agenta 3.
 */

  #define MAX_CURRENT 25 // Maksymalny prąd wzbudzenia w amperach
#define TOLERANCE 0.5 // Margines tolerancji
#define NUM_STATES_AGENT2 100 // Definicja liczby stanów dla Agenta 2

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
        // Ogranicz wartości stanu do zakresu
        state[0] = max(MIN_ERROR, min(MAX_ERROR, state[0]));
        state[1] = max(MIN_LOAD, min(MAX_LOAD, state[1]));

        // Dyskretyzacja stanu agenta 3
        discreteState[0] = (int)((state[0] - MIN_ERROR) * NUM_STATE_BINS_ERROR / (MAX_ERROR - MIN_ERROR));
        discreteState[1] = (int)((state[1] - MIN_LOAD) * NUM_STATE_BINS_LOAD / (MAX_LOAD - MIN_LOAD));
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

    float learningRate = 0.1;
    float discountFactor = 0.9;

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

void setup() {
    // Inicjalizacja portu szeregowego
    Serial.begin(115200);
    while (!Serial) {
        ; // Czekaj na połączenie z komputerem
    }
    detectComputerConnection(); // Wykryj połączenie z komputerem i rozpocznij przenoszenie mocy obliczeniowej

    // Inicjalizacja EEPROM
    EEPROM.begin(512);

    // Inicjalizacja optymalizatora
    optimizer.setBounds(bounds);
    optimizer.setObjectiveFunction(objectiveFunction);
    optimizer.setInitialParams(params);

    // Inicjalizacja pinów i innych komponentów
    initializePins();
    initializeServer();
    initializeDisplay();
    initializeOptimizer();

    // Wyświetlenie wiadomości powitalnej
    displayWelcomeMessage();

    // Inicjalizacja zmiennych globalnych
    initializeGlobalVariables();

    // Inicjalizacja WiFi
    initializeWiFi();

    // Wywołanie funkcji autoCalibrate
    autoCalibrate();
}


void initializePins() {
    pinMode(muxSelectPinA, OUTPUT);
    pinMode(muxSelectPinB, OUTPUT);
    pinMode(muxSelectPinC, OUTPUT);
    pinMode(muxSelectPinD, OUTPUT);
    pinMode(muxInputPin, INPUT);
    pinMode(mosfetPin, OUTPUT);
    pinMode(bjtPin1, OUTPUT);
    pinMode(bjtPin2, OUTPUT);
    pinMode(bjtPin3, OUTPUT);
    pinMode(excitationBJT1Pin, OUTPUT);
    pinMode(excitationBJT2Pin, OUTPUT);
    pinMode(newPin1, OUTPUT);
    pinMode(newPin2, OUTPUT);
    pinMode(newPin3, OUTPUT);
    pinMode(bjtPin4, OUTPUT);
}

void initializeServer() {
    server.begin();
}

void initializeDisplay() {
    display.begin(SH1106_SWITCHCAPVCC, 0x3C);
    display.clearDisplay();
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



    // Dodatkowa inicjalizacja
    EEPROM.begin(512); // Inicjalizacja pamięci EEPROM
    handlePIDParams(Kp, Ki, Kd, false); // Wczytaj parametry PID z pamięci EEPROM
    optimizePID(); // Optymalizacja PID na starcie
}


    // Wczytaj parametry PID z pamięci EEPROM
    handlePIDParams(Kp, Ki, Kd, false);

    // Inicjalizacja zmiennych globalnych dla PID
    previousError = 0;
    integral = 0;

    // Inicjalizacja zmiennych globalnych dla agenta 1
    for (int i = 0; i < NUM_STATE_BINS_ERROR * NUM_STATE_BINS_LOAD; i++) {
        for (int j = 0; j < NUM_ACTIONS; j++) {
            qTableAgent1[i][j] = 0.0;
        }
    }

    // Inicjalizacja serwera
    server.on("/", []() {
        server.send(200, "text/plain", "Witaj w systemie stabilizacji napięcia!");
    });
    server.begin();

    // Inicjalizacja optymalizatora bayesowskiego
    optimizer.init(params, bounds, 3);

    // Inicjalizacja losowości
    randomSeed(analogRead(0));

    // Wyświetlenie komunikatu powitalnego
    Serial.println(FPSTR(welcomeMessage));
}

    stateAgent1[0] = 0.0;
    stateAgent1[1] = 0.0;
    actionAgent1 = 0.0;

    // Inicjalizacja zmiennych globalnych dla optymalizacji bayesowskiej
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
    return threshold; // Przykładowa implementacja
}

void selectMuxChannel(int channel) {
    digitalWrite(muxSelectPinA, channel & 1);
    digitalWrite(muxSelectPinB, (channel >> 1) & 1);
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

// Funkcja wyświetlania danych na ekranie w kolorze
void displayData(float efficiencyPercent) {
    display.clearDisplay();
    display.setTextColor(WHITE, BLACK); // Ustawienie koloru tekstu na biały na czarnym tle
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
    // Aktualizacja bieżących wartości napięcia i prądu
    currentVoltage = readVoltage();
    currentCurrent = readCurrent();

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

    // Wywołanie funkcji learn dla agenta 2
    float error = VOLTAGE_SETPOINT - currentVoltage;
    float load = currentCurrent / maxCurrent; // Normalizacja obciążenia
    agent2.learn(error, load);

    // Aktualizuj wyświetlacz
    display.display();

    // Odczyt wartości z multipleksera
    int muxValue = analogRead(muxInputPin);

    // Aktualizacja stanu systemu
    voltageIn[0] = muxValue * (MAX_VOLTAGE / 1023.0);
    currentIn[0] = muxValue * (maxCurrent / 1023.0);

    // Kontrola stabilizatora napięcia
    controlVoltageRegulator(voltageIn[0], currentIn[0]);

    // Aktualizacja wyświetlacza
    display.clearDisplay();
    display.setCursor(0, 0);
    display.print("Voltage: ");
    display.print(voltageIn[0]);
    display.print(" V");
    display.setCursor(0, 10);
    display.print("Current: ");
    display.print(currentIn[0]);
    display.print(" A");
    display.display();

    // Obsługa serwera
    server.handleClient();
}


    readSensors();

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

    // Wywołanie funkcji hillClimbing
    float currentThreshold = 0.5; // Przykładowa wartość początkowa
    float stepSize = 0.1; // Przykładowy rozmiar kroku
    float optimizedThreshold = hillClimbing(currentThreshold, stepSize);

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
