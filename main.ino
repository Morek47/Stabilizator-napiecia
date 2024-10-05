#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Arduino.h>
#include <EEPROM.h>
#include <Wire.h>
#include <Adafruit_SH1106.h>
#include <ArduinoEigen.h>
#include <BayesOptimizer.h>
#include <map>
#include <queue>
#include <cfloat>
#include <FreeRTOS.h>

// Definicje stałych i zmiennych
#define MAX_CURRENT 25
#define TOLERANCE 0.5
#define NUM_STATES_AGENT2 100
#define MAX_BRAKING_EFFECT 100.0
#define VOLTAGE_SETPOINT 230.0
#define PWM_INCREMENT 5
#define OLED_RESET -1

#define NUM_STATE_BINS_ERROR 10
#define NUM_STATE_BINS_LOAD 10
#define NUM_STATE_BINS_KP 10
#define NUM_STATE_BINS_KI 10
#define NUM_STATE_BINS_KD 10
#define NUM_ACTIONS 5
#define NUM_STATES_AGENT3 100
#define NUM_ACTIONS_AGENT3 5
#define NUM_STATE_BINS_AGENT1 10
#define NUM_STATE_BINS_AGENT2 10
#define NUM_ACTIONS_AGENT1 5
#define NUM_ACTIONS_AGENT2 5

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
const unsigned long timeSlice = 1000;
const unsigned long maxWaitTime = 5000;

float Kp = 1.0;
float Ki = 0.1;
float Kd = 0.01;
float currentVoltage = 0.0;
float currentCurrent = 0.0;
float currentError = 0.0;
float testEpsilon = 0.2;
float testLearningRate = 0.05;
float testDiscountFactor = 0.95;
bool stopCompute = false;
float evaluateThreshold = 0.1;
float previousFilteredValue = 0.0;
float qTableAgent1[NUM_STATE_BINS_AGENT1][NUM_ACTIONS_AGENT1];
float qTableAgent2[NUM_STATE_BINS_AGENT2][NUM_ACTIONS_AGENT2];
float qTableAgent3[NUM_STATES_AGENT3][NUM_ACTIONS_AGENT3];
float bestParams[3] = {0.0, 0.0, 0.0};
float bestObjective = FLT_MAX;
float learningRate = 0.1;
float discountFactor = 0.9;
float Kp, Ki, Kd;
float voltageError;
float excitationError;
float brakingEffect;
unsigned long currentMillis, previousMillis;
unsigned long interval;
float voltage, current;
float error;

unsigned long previousMillisDisplay = 0;
unsigned long previousMillisSerial = 0;
const long intervalDisplay = 1000;
const long intervalSerial = 500;

struct Agent {
    int id;
    int priority;
    int waitCount;
    unsigned long waitTime;
    unsigned long lastAccessTime;
    float targetVoltage;
};

Agent agents[] = {
    {1, 0, 0, 0, 0},
    {2, 0, 0, 0, 0},
    {3, 0, 0, 0, 0}
};

SemaphoreHandle_t resourceSemaphore;
SemaphoreHandle_t eepromSemaphore;

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

Adafruit_SH1106 display(OLED_RESET);

float simulateVoltageControl(float Kp, float Ki, float Kd);
float simulateExcitationControl();
float someFunctionOfOtherParameters(float rotationalSpeed, float torque, float frictionCoefficient);
void hillClimbing();
void discretizeStateAgent3(float state[2], int discreteState[2]);
float chooseActionAgent3(int discreteState[2], float epsilon);
void executeActionAgent3(float action);
float calculateRewardAgent3(float state[2], float action);
void updateQAgent3(float state[2], float action, float reward, float nextState[2]);
float objectiveFunction(const float* params);
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
void autoTunePID(float &Kp, float &Ki, float &Kd, float setpoint, float measuredValue);
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
void discretizeStateAgent2(float state, int &discreteState);

void adjustMinInputPower(float inputPower) {
    static float minObservedPower = 1e-6;
    static float maxObservedPower = 1e-3;

    if (inputPower < minObservedPower) {
        minObservedPower = inputPower;
    }
    if (inputPower > maxObservedPower) {
        maxObservedPower = inputPower;
    }

    float threshold = (minObservedPower + maxObservedPower) / 2;

    Serial.print("Min Observed Power: ");
    Serial.println(minObservedPower);
    Serial.print("Max Observed Power: ");
    Serial.println(maxObservedPower);
    Serial.print("Current Threshold: ");
    Serial.println(threshold);
}

void decreasePriority(Agent &agent) {
    if (agent.priority > 0) {
        agent.priority--;
    }
}

void increasePriority(Agent &agent) {
    if (millis() - agent.lastAccessTime > maxWaitTime) {
        agent.priority++;
    }
}

void allocateResources() {
    std::priority_queue<std::pair<int, int>> pq;

    for (int i = 0; i < sizeof(agents) / sizeof(agents[0]); i++) {
        pq.push({agents[i].priority, agents[i].id});
        Serial.print("Agent ");
        Serial.print(agents[i].id);
        Serial.print(" z priorytetem ");
        Serial.println(agents[i].priority);
    }

    while (!pq.empty()) {
        int agentId = pq.top().second;
        pq.pop();

        if (xSemaphoreTake(resourceSemaphore, portMAX_DELAY) == pdTRUE) {
            performAgentOperation(agentId);
            xSemaphoreGive(resourceSemaphore);
        }
    }
}

void performAgentOperation(int agentId) {
    // Przykładowa operacja: odczyt napięcia
    float voltage = readVoltage();
    Serial.print("Agent ");
    Serial.print(agentId);
    Serial.print(": Odczytane napięcie: ");
    Serial.println(voltage);

    // Zmniejszenie priorytetu po udanej operacji
    decreasePriority(agents[agentId - 1]);

    // Debugowanie: wyświetlenie nowego priorytetu agenta
    Serial.print("Nowy priorytet agenta ");
    Serial.print(agentId);
    Serial.print(": ");
    Serial.println(agents[agentId - 1].priority);
}

float readVoltage() {
    float calibratedVoltage = 0.0;
    if (resourceSemaphore == NULL) {
        Serial.println("Error: resourceSemaphore is not initialized");
        return calibratedVoltage;
    }

    if (xSemaphoreTake(resourceSemaphore, portMAX_DELAY) == pdTRUE) {
        int rawValue = analogRead(A0);
        if (rawValue < 0 || rawValue > ADC_MAX_VALUE) {
            Serial.println("Błąd: Nieprawidłowa wartość odczytu ADC");
        } else {
            calibratedVoltage = calibrateVoltage(rawValue);
        }
        xSemaphoreGive(resourceSemaphore);
    } else {
        Serial.println("Error: Failed to take resourceSemaphore");
    }
    return calibratedVoltage;
}

 // Przykładowa kalibracja: przelicz surowe napięcie na rzeczywiste napięcie
float calibrateVoltage(float rawVoltage) {
    return rawVoltage * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
}

// Implementacja funkcji simulateVoltageControl
float simulateVoltageControl(float Kp, float Ki, float Kd) {
    float setpoint = 230.0; // Docelowe napięcie w woltach
    float measuredValue = readVoltage();
    float error = setpoint - measuredValue;
    float controlSignal = Kp * error + Ki * (error * 50) + Kd * (error / 50);
    return controlSignal;
}

// Implementacja funkcji simulateExcitationControl
float simulateExcitationControl() {
    int rawValue = analogRead(A1);
    
    // Sprawdzenie, czy wartość mieści się w oczekiwanym zakresie
    if (rawValue < 0 || rawValue > ADC_MAX_VALUE) {
        Serial.println("Błąd: Odczytana wartość spoza zakresu!");
        return 0.0; // Zwróć 0.0 w przypadku błędu
    }

    // Debugowanie: wyświetlenie odczytanej wartości
    Serial.print("Odczytana wartość z A1: ");
    Serial.println(rawValue);

    // Przeliczenie surowej wartości na prąd wzbudzenia
    float excitationCurrent = calibrateVoltage(rawValue);
    return excitationCurrent;
}

// Implementacja funkcji centralController
void centralController() {
    float averageTargetVoltage = 0.0;
    int numAgents = sizeof(agents) / sizeof(agents[0]);

    for (int i = 0; i < numAgents; i++) {
        averageTargetVoltage += agents[i].targetVoltage;
    }
    averageTargetVoltage /= numAgents;

    setTargetVoltageForAllAgents(averageTargetVoltage);
}

// Funkcja ustawiająca docelowe napięcie dla wszystkich agentów na średnią wartość
void setTargetVoltageForAllAgents(float averageTargetVoltage) {
    int numAgents = sizeof(agents) / sizeof(agents[0]);
    for (int i = 0; i < numAgents; i++) {
        agents[i].targetVoltage = averageTargetVoltage;
    }
}

// Implementacja funkcji simulateBrakingEffect
float simulateBrakingEffect(float rotationalSpeed, float torque, float frictionCoefficient, float brakeTemperature, float brakeWear) {
    // Model matematyczny: efekt hamowania jest proporcjonalny do prędkości obrotowej i momentu obrotowego,
    // a odwrotnie proporcjonalny do współczynnika tarcia
    float simulatedBrakingEffect = (rotationalSpeed * torque) / frictionCoefficient;

    // Uwzględnienie temperatury hamulców: wyższa temperatura zmniejsza efekt hamowania
    float temperatureEffect = 1.0 - (brakeTemperature / 100.0);
    simulatedBrakingEffect *= temperatureEffect;

    // Uwzględnienie zużycia hamulców: większe zużycie zmniejsza efekt hamowania
    float wearEffect = 1.0 - (brakeWear / 100.0);
    simulatedBrakingEffect *= wearEffect;

    // Upewnij się, że efekt hamowania nie przekracza maksymalnej wartości
    if (simulatedBrakingEffect > MAX_BRAKING_EFFECT) {
        simulatedBrakingEffect = MAX_BRAKING_EFFECT;
    }

    // Upewnij się, że efekt hamowania nie jest mniejszy niż minimalna wartość
    const float MIN_BRAKING_EFFECT = 0.0;
    if (simulatedBrakingEffect < MIN_BRAKING_EFFECT) {
        simulatedBrakingEffect = MIN_BRAKING_EFFECT;
    }

    return simulatedBrakingEffect;
}

// Implementacja funkcji lowPassFilter
float lowPassFilter(float currentValue, float previousValue, float alpha) {
    return alpha * currentValue + (1 - alpha) * previousValue;
}

// Definicja klasy SemaphoreGuard
class SemaphoreGuard {
public:
    SemaphoreGuard(SemaphoreHandle_t sem) : sem(sem) {
        xSemaphoreTake(sem, portMAX_DELAY);
    }
    ~SemaphoreGuard() {
        xSemaphoreGive(sem);
    }
private:
    SemaphoreHandle_t sem;
};

// Przykład użycia
void someFunction() {
    SemaphoreGuard guard(resourceSemaphore);
    // Wykonaj operację
    // Semafor zostanie automatycznie zwolniony po zakończeniu funkcji
}

// Funkcja agenta 1
void agent1Function(void *pvParameters) {
    for (;;) {
        {
            SemaphoreGuard guard(resourceSemaphore);
            // Wykonaj operację
            float voltage = readVoltage();
            Serial.println("Agent 1: Odczytane napięcie: " + String(voltage));

            // Ustawienie PWM dla agenta 1
            analogWrite(mosfetPin, map(voltage, 0, 5, 0, 255));

            // Zmniejszenie priorytetu po udanej operacji
            decreasePriority(agents[0]);
        }
        vTaskDelay(100 / portTICK_PERIOD_MS);
    }
}

void discretizeStateAgent2(float state, int &discreteState) {
    float binSize = MAX_CURRENT / NUM_STATE_BINS_AGENT2;
    discreteState = min(NUM_STATE_BINS_AGENT2 - 1, max(0, int(state / binSize)));
}

// Funkcja do wyboru akcji dla Agenta 2
int chooseActionAgent2(int discreteState, float epsilon) {
    if (random(0, 100) < epsilon * 100) {
        // Eksploracja: wybierz losową akcję
        return random(0, NUM_ACTIONS_AGENT2);
    } else {
        // Eksploatacja: wybierz najlepszą akcję na podstawie tabeli Q
        int bestAction = 0;
        float bestValue = qTableAgent2[discreteState][0];
        for (int i = 1; i < NUM_ACTIONS_AGENT2; i++) {
            if (qTableAgent2[discreteState][i] > bestValue) {
                bestValue = qTableAgent2[discreteState][i];
                bestAction = i;
            }
        }
        return bestAction;
    }
}



float calculateRewardAgent2(float current) {
    float error = abs(current - MAX_CURRENT);
    return -error; // Nagroda jest ujemna, im większy błąd, tym mniejsza nagroda
}

void updateQAgent2(int discreteState, int action, float reward, int nextDiscreteState) {
    float bestNextActionValue = qTableAgent2[nextDiscreteState][0];
    for (int i = 1; i < NUM_ACTIONS_AGENT2; i++) {
        if (qTableAgent2[nextDiscreteState][i] > bestNextActionValue) {
            bestNextActionValue = qTableAgent2[nextDiscreteState][i];
        }
    }
    qTableAgent2[discreteState][action] = qTableAgent2[discreteState][action] + 
        learningRate * (reward + discountFactor * bestNextActionValue - qTableAgent2[discreteState][action]);
}

// Funkcja agenta 2
void agent2Function(void *pvParameters) {
    for (;;) {
        {
            SemaphoreGuard guard(resourceSemaphore);
            float current = readExcitationCurrent();
            Serial.println("Agent 2: Odczytany prąd wzbudzenia: " + String(current));

            int discreteState;
            discretizeStateAgent2(current, discreteState);

            int action = chooseActionAgent2(discreteState, testEpsilon);
            analogWrite(excitationBJT1Pin, action * (255 / (NUM_ACTIONS_AGENT2 - 1)));

            float reward = calculateRewardAgent2(current);
            float nextCurrent = readExcitationCurrent();
            int nextDiscreteState;
            discretizeStateAgent2(nextCurrent, nextDiscreteState);

            updateQAgent2(discreteState, action, reward, nextDiscreteState);

            decreasePriority(agents[1]);
        }
        vTaskDelay(100 / portTICK_PERIOD_MS);
    }
}

// Funkcja agenta 3
void agent3Function(void *pvParameters) {
    for (;;) {
        {
            SemaphoreGuard guard(resourceSemaphore);
            float brakingEffect = readBrakingEffect();
            Serial.println("Agent 3: Odczytany efekt hamowania: " + String(brakingEffect));

            // Ustawienie PWM dla agenta 3
            analogWrite(bjtPin1, map(brakingEffect, 0, 5, 0, 255));

            // Zmniejszenie priorytetu po udanej operacji
            decreasePriority(agents[2]);
        }
        vTaskDelay(100 / portTICK_PERIOD_MS);
    }
}

// Funkcja celu dla optymalizatora
float objectiveFunction(const float* params) {
    float Kp = params[0];
    float Ki = params[1];
    float Kd = params[2];
    float controlSignal = simulateVoltageControl(Kp, Ki, Kd);
    float error = calculateError(VOLTAGE_SETPOINT, controlSignal);
    return abs(error); // Minimalizujemy wartość bezwzględną błędu
}

// Mapa funkcji naprawczych
std::map<String, void(*)()> fixFunctions;

// Implementacja funkcji defaultFixFunction
void defaultFixFunction() {
    // Przykładowa implementacja domyślnej funkcji naprawczej
    Serial.println("Wykonano domyślną funkcję naprawczą.");

    // Resetowanie kontrolera napięcia
    resetVoltageController();
    Serial.println("Kontroler napięcia został zresetowany.");

    // Zmniejszenie prądu wzbudzenia
    reduceExcitationCurrent();
    Serial.println("Prąd wzbudzenia został zmniejszony.");

    // Kalibracja efektu hamowania
    calibrateBrakingEffect();
    Serial.println("Efekt hamowania został skalibrowany.");

    // Kalibracja napięcia
    float rawVoltage = analogRead(A0);
    float calibratedVoltage = calibrateVoltage(rawVoltage);
    Serial.println("Napięcie zostało skalibrowane: " + String(calibratedVoltage));

    // Kalibracja prądu
    float rawCurrent = analogRead(A1);
    float calibratedCurrent = calibrateCurrent(rawCurrent);
    Serial.println("Prąd został skalibrowany: " + String(calibratedCurrent));

    // Automatyczne dostrajanie parametrów PID
    autoTunePID(Kp, Ki, Kd, VOLTAGE_SETPOINT, calibratedVoltage);
    Serial.println("Parametry PID zostały dostrojone.");

    // Logowanie błędu
    logError("Wykonano domyślną funkcję naprawczą.");
    Serial.println("Błąd został zalogowany.");
}

// Implementacja funkcji resetVoltageController
void resetVoltageController() {
    // Przykładowa implementacja resetowania kontrolera napięcia
    currentVoltage = 0.0;
    currentError = 0.0;
    Serial.println("Kontroler napięcia został zresetowany.");
}

// Implementacja funkcji reduceExcitationCurrent
void reduceExcitationCurrent() {
    // Przykładowa implementacja zmniejszenia prądu wzbudzenia
    analogWrite(excitationBJT1Pin, 0);
    analogWrite(excitationBJT2Pin, 0);
    Serial.println("Prąd wzbudzenia został zmniejszony.");
}

// Implementacja funkcji calibrateBrakingEffect
void calibrateBrakingEffect() {
    // Przykładowa implementacja kalibracji efektu hamowania
    float rawBrakingEffect = analogRead(A2);
    float calibratedBrakingEffect = rawBrakingEffect * (5.0 / 1023.0);
    Serial.println("Efekt hamowania został skalibrowany: " + String(calibratedBrakingEffect));
}


// Funkcja kalibracji prądu wzbudzenia
float calibrateCurrent(float rawCurrent) {
    // Współczynnik kalibracji
    const float calibrationFactor = 0.95; // Korekta o -5%

    // Kalibracja surowego prądu
    float calibratedCurrent = rawCurrent * calibrationFactor;

    // Sprawdzenie, czy skalibrowany prąd mieści się w dopuszczalnym zakresie
    if (calibratedCurrent < 0) {
        Serial.println("Błąd: Skalibrowany prąd jest mniejszy niż 0. Ustawiam na 0.");
        calibratedCurrent = 0;
    } else if (calibratedCurrent > MAX_CURRENT) {
        Serial.println("Błąd: Skalibrowany prąd przekracza maksymalny dopuszczalny prąd. Ustawiam na MAX_CURRENT.");
        calibratedCurrent = MAX_CURRENT;
    }

    // Wyświetlenie skalibrowanego prądu dla celów debugowania
    Serial.print("Surowy prąd: ");
    Serial.print(rawCurrent);
    Serial.print(" A, Skalibrowany prąd: ");
    Serial.print(calibratedCurrent);
    Serial.println(" A");

    return calibratedCurrent;
}

// Implementacja funkcji autoTunePID
void autoTunePID(float &Kp, float &Ki, float &Kd, float setpoint, float measuredValue) {
    float error = calculateError(setpoint, measuredValue);
    // Przykładowa implementacja automatycznego dostrajania parametrów PID
    Kp = 1.0; // Przykładowa wartość
    Ki = 0.1; // Przykładowa wartość
    Kd = 0.01; // Przykładowa wartość
    Serial.println("Automatyczne dostrajanie PID zakończone.");
}

// Implementacja funkcji chooseActionAgent3
float chooseActionAgent3(int discreteState[2], float epsilon) {
    // Przykładowa implementacja wyboru akcji dla Agenta 3
    if (random(0, 100) < epsilon * 100) {
        // Wybór losowej akcji z prawdopodobieństwem epsilon
        return random(0, NUM_ACTIONS_AGENT3);
    } else {
        // Wybór najlepszej akcji na podstawie wartości Q
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

// Implementacja funkcji calculateRewardAgent3
float calculateRewardAgent3(float state[2], float action) {
    // Normalizacja stanów
    float normalizedState[2];
    normalizedState[0] = state[0] / NUM_STATES_AGENT3;
    normalizedState[1] = state[1] / NUM_STATES_AGENT3;

    // Przykładowa implementacja obliczania nagrody dla Agenta 3
    float voltage = readVoltage();
    float error = calculateError(VOLTAGE_SETPOINT, voltage);

    // Obliczanie nagrody
    float reward = -abs(error);

    // Zapewnienie, że nagroda nie jest ujemna
    if (reward < 0) {
        reward = 0;
    }

    return reward;
}

// Implementacja funkcji updateQAgent3
void updateQAgent3(float state[2], float action, float reward, float nextState[2]) {
    // Dyskretyzacja stanów
    int discreteState[2];
    int discreteNextState[2];
    discretizeStateAgent3(state, discreteState);
    discretizeStateAgent3(nextState, discreteNextState);

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

// Inicjalizacja tablicy Q
void initializeQTable() {
    for (int i = 0; i < NUM_STATES_AGENT3; i++) {
        for (int j = 0; j < NUM_ACTIONS_AGENT3; j++) {
            qTableAgent3[i][j] = 0.0;
        }
    }
}

// Implementacja funkcji hillClimbing
void hillClimbing() {
    float currentParams[3] = {1.0, 1.0, 1.0}; // Początkowe parametry
    float bestParams[3] = {1.0, 1.0, 1.0}; // Inicjalizacja na te same wartości
    float bestObjective = objectiveFunction(currentParams);
    float stepSize = 0.1; // Rozmiar kroku dla optymalizacji

    for (int i = 0; i < 100; i++) { // Przykładowa liczba iteracji
        for (int j = 0; j < 3; j++) {
            float newParams[3] = {currentParams[0], currentParams[1], currentParams[2]};
            newParams[j] += stepSize; // Zwiększanie parametru
            float newObjective = objectiveFunction(newParams);

            if (newObjective < bestObjective) {
                bestObjective = newObjective;
                bestParams[j] = newParams[j];
            } else {
                newParams[j] -= 2 * stepSize; // Zmniejszanie parametru
                newObjective = objectiveFunction(newParams);

                if (newObjective < bestObjective) {
                    bestObjective = newObjective;
                    bestParams[j] = newParams[j];
                }
            }
        }
    }
}

   // Aktualizacja bieżących parametrów
void updateCurrentParams(float currentParams[3], const float bestParams[3]) {
    currentParams[0] = bestParams[0];
    currentParams[1] = bestParams[1];
    currentParams[2] = bestParams[2];
}

// Zaktualizuj globalne najlepsze parametry
void updateBestParams(float bestParams[3], const float currentParams[3]) {
    bestParams[0] = currentParams[0];
    bestParams[1] = currentParams[1];
    bestParams[2] = currentParams[2];
}

// Funkcja do symulacji efektu hamowania
void simulateBrakingEffect() {
    float currentParams[3];
    updateCurrentParams(currentParams, bestParams);
    updateBestParams(bestParams, currentParams);

    float rotationalSpeed = 3000.0;
    float torque = 50.0;
    float frictionCoefficient = 0.8;
    float brakingEffect = someFunctionOfOtherParameters(rotationalSpeed, torque, frictionCoefficient);
    Serial.println(brakingEffect);
}

// Implementacja funkcji printOptimizationResults
void printOptimizationResults() {
    // Ograniczenie wartości zmiennych do ustalonych zakresów
    Kp = constrain(Kp, MIN_KP, MAX_KP);
    Ki = constrain(Ki, MIN_KI, MAX_KI);
    Kd = constrain(Kd, MIN_KD, MAX_KD);
    float bestEfficiency = constrain(bestEfficiency, 0.0, 100.0); // Zakładam, że wydajność mieści się w zakresie 0-100

    Serial.println("Optymalizacja zakończona:");
    Serial.print("Kp: "); Serial.println(Kp);
    Serial.print("Ki: "); Serial.println(Ki);
    Serial.print("Kd: "); Serial.println(Kd);
    Serial.print("Wydajność: "); Serial.println(bestEfficiency);
}

// Przykładowa implementacja funkcji zwracającej najlepsze parametry
void getBestParams(float params[3]) {
    params[0] = 1.0;
    params[1] = 1.0;
    params[2] = 1.0;
}

// Przykładowa implementacja funkcji zwracającej najlepszy wynik funkcji celu
float getBestObjective() {
    return 0.0;
}

// Przykładowa implementacja funkcji sugerującej kolejne parametry
void suggestNextParameters(float params[3]) {
    // Możesz dostosować tę funkcję do swoich potrzeb
}

// Funkcja do odczytu napięcia
float readVoltage() {
    int rawValue = analogRead(A0);
    if (rawValue < 0 || rawValue > ADC_MAX_VALUE) {
        Serial.println("Błąd: Nieprawidłowa wartość odczytu z ADC");
        return -1.0; // Zwróć wartość błędu
    }
    return calibrateVoltage(rawValue * (VOLTAGE_REFERENCE / ADC_MAX_VALUE));
}

// Funkcja do odczytu prądu wzbudzenia
float readExcitationCurrent() {
    int rawValue = analogRead(A1);
    if (rawValue < 0 || rawValue > ADC_MAX_VALUE) {
        Serial.println("Błąd: Nieprawidłowa wartość odczytu z ADC");
        return -1.0; // Zwróć wartość błędu
    }
    return calibrateCurrent(rawValue * (VOLTAGE_REFERENCE / ADC_MAX_VALUE));
}

// Funkcja do odczytu efektu hamowania
float readBrakingEffect() {
    int rawValue = analogRead(A2);
    if (rawValue < 0 || rawValue > ADC_MAX_VALUE) {
        Serial.println("Błąd: Nieprawidłowa wartość odczytu z ADC");
        return -1.0; // Zwróć wartość błędu
    }
    return calibrateBrakingEffect(rawValue * (VOLTAGE_REFERENCE / ADC_MAX_VALUE));
}

// Przykładowa implementacja funkcji automatycznego strojenia PID
void autoTunePID(PID &pid, float setpoint, float measuredValue) {
    // Możesz dostosować tę funkcję do swoich potrzeb
}

// Funkcja do obliczania błędu
float calculateError(float setpoint, float measuredValue) {
    return setpoint - measuredValue;
}


// Aktualizacja parametrów PID
void updateControlParameters(float params[3]) {
    float Kp = params[0];
    float Ki = params[1];
    float Kd = params[2];
    pidController.setTunings(Kp, Ki, Kd);
}

// Obsługa komunikacji szeregowej
void handleSerialCommunication() {
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        if (command.startsWith("SET_PARAMS")) {
            float params[3];
            sscanf(command.c_str(), "SET_PARAMS %f %f %f", &params[0], &params[1], &params[2]);
            updateControlParameters(params);
            Serial.println("Parameters updated");
        } else if (command.startsWith("GET_STATUS")) {
            Serial.println("System is running");
        } else {
            Serial.println("Unknown command");
        }
    }
}

// Wybór akcji dla agenta 3
float chooseActionAgent3(int discreteState[2], float epsilon) {
    int stateIndex = discreteState[0] * 10 + discreteState[1];
    if (random(0, 100) < epsilon * 100) {
        return (float)random(0, NUM_ACTIONS_AGENT3) / (NUM_ACTIONS_AGENT3 - 1);
    } else {
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

// Wykonanie akcji dla agenta 3
void executeActionAgent3(float action) {
    int pwmValue = (int)(action * 255.0);
    pwmValue = constrain(pwmValue, 0, 255);
    analogWrite(mosfetPin, pwmValue);
}

// Obliczenie nagrody dla agenta 3
float calculateRewardAgent3(float state[2], float action) {
    if (state[0] < 0.0 || state[0] > 1.0 || state[1] < 0.0 || state[1] > 1.0 || action < 0.0 || action > 1.0) {
        Serial.println("Error: State or action values out of range");
        return -FLT_MAX;
    }
    float target = 1.0;
    float error = target - (state[0] + state[1]) / 2.0;
    float reward = -abs(error);
    if (action > 0.8) {
        reward -= 0.1 * (action - 0.8);
    }
    return reward;
}

// Aktualizacja wartości Q dla agenta 3
void updateQAgent3(float state[2], float action, float reward, float nextState[2]) {
    if (state[0] < 0.0 || state[0] > 1.0 || state[1] < 0.0 || state[1] > 1.0 || nextState[0] < 0.0 || nextState[0] > 1.0 || nextState[1] < 0.0 || nextState[1] > 1.0 || action < 0.0 || action > 1.0 || reward < -FLT_MAX || reward > FLT_MAX) {
        Serial.println("Error: State, action, or reward values out of range");
        return;
    }
    int stateIndex = (int)(state[0] * 10) * 10 + (int)(state[1] * 10);
    int nextStateIndex = (int)(nextState[0] * 10) * 10 + (int)(nextState[1] * 10);
    int actionIndex = (int)(action * (NUM_ACTIONS_AGENT3 - 1));
    float maxNextQValue = qTableAgent1[nextStateIndex][0];
    for (int i = 1; i < NUM_ACTIONS_AGENT3; i++) {
        if (qTableAgent1[nextStateIndex][i] > maxNextQValue) {
            maxNextQValue = qTableAgent1[nextStateIndex][i];
        }
    }
    qTableAgent1[stateIndex][actionIndex] += testLearningRate * (reward + testDiscountFactor * maxNextQValue - qTableAgent1[stateIndex][actionIndex]);
}


float chooseActionAgent3(int discreteState[2], float epsilon) {
    int stateIndex = discreteState[0] * 10 + discreteState[1];
    if (random(0, 100) < epsilon * 100) {
        // Eksploracja: wybierz losową akcję
        return (float)random(0, NUM_ACTIONS_AGENT3) / (NUM_ACTIONS_AGENT3 - 1);
    } else {
        // Eksploatacja: wybierz najlepszą akcję
        float maxQValue = qTableAgent3[stateIndex][0];
        int bestActionIndex = 0;
        for (int i = 1; i < NUM_ACTIONS_AGENT3; i++) {
            if (qTableAgent3[stateIndex][i] > maxQValue) {
                maxQValue = qTableAgent3[stateIndex][i];
                bestActionIndex = i;
            }
        }
        return (float)bestActionIndex / (NUM_ACTIONS_AGENT3 - 1);
    }
}

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

   

    // Obliczenie indeksów stanów i akcji
    int stateIndex = discreteState[0] * NUM_STATES_AGENT3 + discreteState[1];
    int nextStateIndex = discreteNextState[0] * NUM_STATES_AGENT3 + discreteNextState[1];
    int actionIndex = (int)(action * (NUM_ACTIONS_AGENT3 - 1));

    // Aktualna wartość Q 
    float currentQ = qTableAgent3[stateIndex][actionIndex];

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
float currentQ = qTableAgent3[stateIndex][actionIndex];

// Maksymalna wartość Q dla następnego stanu
float maxNextQ = qTableAgent3[nextStateIndex][0];
for (int i = 1; i < NUM_ACTIONS_AGENT3; i++) {
    if (qTableAgent3[nextStateIndex][i] > maxNextQ) {
        maxNextQ = qTableAgent3[nextStateIndex][i];
    }
}

// Aktualizacja wartości Q
qTableAgent3[stateIndex][actionIndex] = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);

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

// Symulacja systemu z użyciem parametrów PID
float voltageError = simulateVoltageControl(Kp, Ki, Kd);
float excitationCurrentError = simulateExcitationControl();
brakingEffect = simulateBrakingEffect(3000.0, 50.0, 0.8);

// Funkcja celu: minimalizacja błędu napięcia, minimalizacja błędu prądu wzbudzenia i minimalizacja hamowania
float calculateObjective(float voltageError, float excitationCurrentError, float brakingEffect) {
    float objective = -voltageError - excitationCurrentError - brakingEffect;
    return objective;
}

// Implementacja funkcji simulateVoltageControl
float simulateVoltageControl(float Kp, float Ki, float Kd) {
    float setpoint = 230.0; // Docelowe napięcie w woltach
    float measuredValue = readVoltage();
    float error = setpoint - measuredValue;
    static float integral = 0.0;
    static float previousError = 0.0;

    integral += error * (1.0 / controlFrequency);
    float derivative = (error - previousError) * controlFrequency;
    previousError = error;

    float controlSignal = Kp * error + Ki * integral + Kd * derivative;
    return controlSignal;
}

// Implementacja funkcji simulateExcitationControl
float simulateExcitationControl() {
    // Przykładowe wartości dla symulacji
    float excitationCurrent = 0.0;
    float targetCurrent = 10.0; // Docelowy prąd wzbudzenia
    float controlSignal = 0.0;
    float error = 0.0;
    float previousError = 0.0;
    float integral = 0.0;
    float derivative = 0.0;
    float dt = 0.1; // Przykładowy krok czasowy

    // Parametry PID
    float Kp = 1.0;
    float Ki = 0.1;
    float Kd = 0.01;

    // Ograniczenia prądu wzbudzenia
    float maxExcitationCurrent = MAX_CURRENT;
    float minExcitationCurrent = 0.0;

    // Pętla symulacyjna
    for (int i = 0; i < 100; i++) {
        // Oblicz błąd
        error = targetCurrent - excitationCurrent;

        // Oblicz całkę
        integral += error * dt;

        // Oblicz pochodną
        derivative = (error - previousError) / dt;

        // Oblicz sygnał sterujący (PID)
        controlSignal = Kp * error + Ki * integral + Kd * derivative;

        // Aktualizuj prąd wzbudzenia
        excitationCurrent += controlSignal * dt;

        // Ogranicz prąd wzbudzenia do dopuszczalnych wartości
        if (excitationCurrent > maxExcitationCurrent) {
            excitationCurrent = maxExcitationCurrent;
        } else if (excitationCurrent < minExcitationCurrent) {
            excitationCurrent = minExcitationCurrent;
        }

        // Aktualizuj poprzedni błąd
        previousError = error;

        // Debugowanie: wyświetlenie aktualnych wartości
        Serial.print("Czas: ");
        Serial.print(i * dt);
        Serial.print("s, Prąd wzbudzenia: ");
        Serial.print(excitationCurrent);
        Serial.print("A, Błąd: ");
        Serial.println(error);

        // Logowanie danych do EEPROM co 10 iteracji
        if (i % 10 == 0) {
            int address = i * sizeof(float);
            EEPROM.put(address, excitationCurrent);
        }

        // Dynamiczne dostosowywanie parametrów PID (przykładowa logika)
        if (i % 20 == 0) {
            Kp += 0.1;
            Ki += 0.01;
            Kd += 0.005;
        }
    }

    return excitationCurrent;
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
    float brakingEffect = someFunctionOfOtherParameters(rotationalSpeed, torque, frictionCoefficient);
    
    // Zakładamy, że mniejsza wartość efektu hamowania jest lepsza
    // Zastosowanie funkcji penalizującej większe wartości
    float penalizedBrakingEffect = 1.0 / (1.0 + brakingEffect);

    return penalizedBrakingEffect;
}

// Dodatkowa korekta na podstawie pomiarów
if (calibratedVoltage > 5.0) {
    calibratedVoltage = 5.0; // Ograniczenie maksymalnego napięcia
} else if (calibratedVoltage < 0.0) {
    calibratedVoltage = 0.0; // Ograniczenie minimalnego napięcia
}

return calibratedVoltage;

  // Definicje stałych
#define TEMPERATURE_COEFFICIENT 0.98 // Współczynnik kalibracji dla temperatury
#define WEAR_COEFFICIENT 0.95 // Współczynnik kalibracji dla zużycia hamulców
#define SPEED_COEFFICIENT 1.01 // Współczynnik kalibracji dla prędkości obrotowej
#define MAX_CURRENT 25
#define MAX_BRAKING_EFFECT 100.0
#define ADC_MAX_VALUE 1023
#define VOLTAGE_REFERENCE 5.0

// Funkcja odczytu temperatury (przykładowa implementacja)
float readTemperature() {
    return 25.0; // 25 stopni Celsjusza
}

// Funkcja odczytu zużycia hamulców (przykładowa implementacja)
float readBrakeWear() {
    static float brakeWear = 0.8; // Początkowa wartość zużycia hamulców (80%)
    brakeWear += random(-5, 6) / 100.0; // Losowa zmiana w zakresie -0.05 do 0.05
    brakeWear = constrain(brakeWear, 0.0, 1.0); // Ograniczenie wartości do zakresu 0-1 (0-100%)
    return brakeWear;
}

// Funkcja odczytu prędkości obrotowej (przykładowa implementacja)
float readRotationalSpeed() {
    return 3000.0; // 3000 RPM
}

// Funkcja kalibracji efektu hamowania
float calibrateBrakingEffect(float rawBrakingEffect) {
    const float calibrationFactor = 0.9; // Korekta o -10%
    float calibratedBrakingEffect = rawBrakingEffect * calibrationFactor;

    // Sprawdzenie, czy skalibrowany efekt hamowania mieści się w dopuszczalnym zakresie
    if (calibratedBrakingEffect < 0) {
        Serial.println("Błąd: Skalibrowany efekt hamowania jest mniejszy niż 0. Ustawiam na 0.");
        calibratedBrakingEffect = 0;
    } else if (calibratedBrakingEffect > MAX_BRAKING_EFFECT) {
        Serial.println("Błąd: Skalibrowany efekt hamowania przekracza maksymalny dopuszczalny efekt. Ustawiam na MAX_BRAKING_EFFECT.");
        calibratedBrakingEffect = MAX_BRAKING_EFFECT;
    }

    // Wyświetlenie skalibrowanego efektu hamowania dla celów debugowania
    Serial.print("Surowy efekt hamowania: ");
    Serial.print(rawBrakingEffect);
    Serial.print(" A, Skalibrowany efekt hamowania: ");
    Serial.print(calibratedBrakingEffect);
    Serial.println(" A");

    // Minimalizacja efektu hamowania
    float minimizedBrakingEffect = minimizeBrakingEffect(calibratedBrakingEffect);

    // Wyświetlenie zminimalizowanego efektu hamowania dla celów debugowania
    Serial.print("Zminimalizowany efekt hamowania: ");
    Serial.print(minimizedBrakingEffect);
    Serial.println(" A");

    return minimizedBrakingEffect;
}

// Funkcja minimalizacji efektu hamowania
float minimizeBrakingEffect(float calibratedBrakingEffect) {
    const float minimizationFactor = 0.8; // Korekta o -20%
    float minimizedBrakingEffect = calibratedBrakingEffect * minimizationFactor;

    // Sprawdzenie, czy zminimalizowany efekt hamowania mieści się w dopuszczalnym zakresie
    if (minimizedBrakingEffect < 0) {
        minimizedBrakingEffect = 0;
    } else if (minimizedBrakingEffect > MAX_BRAKING_EFFECT) {
        minimizedBrakingEffect = MAX_BRAKING_EFFECT;
    }

    return minimizedBrakingEffect;
}

// Funkcja kalibracji prądu
float calibrateCurrent(float rawCurrent) {
    float calibratedCurrent = rawCurrent;

    // Dodatkowa korekta na podstawie pomiarów
    if (calibratedCurrent > MAX_CURRENT) {
        calibratedCurrent = MAX_CURRENT; // Ograniczenie maksymalnego prądu
    } else if (calibratedCurrent < 0.0) {
        calibratedCurrent = 0.0; // Ograniczenie minimalnego prądu
    }

    return calibratedCurrent;
}

// Funkcja kalibracji na podstawie różnych czynników
float calibrateBrakingEffectWithFactors(float rawBrakingEffect, float temperature, float brakeWear, float rotationalSpeed) {
    float temperatureFactor = (temperature > 30.0) ? TEMPERATURE_COEFFICIENT : 1.0;
    float wearFactor = (brakeWear > 0.7) ? WEAR_COEFFICIENT : 1.0;
    float speedFactor = (rotationalSpeed > 2500.0) ? SPEED_COEFFICIENT : 1.0;

    float calibratedBrakingEffect = rawBrakingEffect * temperatureFactor * wearFactor * speedFactor;
    calibratedBrakingEffect *= 1.02; // Dodatkowa korekta o 2%

    return calibratedBrakingEffect;
}

// Przykładowa implementacja odczytu napięcia z pinu analogowego
float readVoltage() {
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

// Funkcja filtra dolnoprzepustowego
float lowPassFilter(float currentValue, float previousValue, float alpha) {
    return alpha * currentValue + (1 - alpha) * previousValue;
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
    pid.setTunings(bestKp, bestKi, bestKd);
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

// Funkcja symulująca lub testująca kontrolę PID
float simulatePIDControl() {
    PID pid(Kp, Ki, Kd);
    return simulateSystem(pid, VOLTAGE_SETPOINT, readVoltage());
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
const float Kp_max = 5.0;

// Deklaracja zmiennej globalnej
float LOAD_THRESHOLD = 0.5;

// Dodatkowe zmienne globalne
float previousError = 0;
float integral = 0;
const float COMPENSATION_FACTOR = 0.1;
float epsilon = 0.3;

// Definicje zmiennych globalnych dla agenta 1
float qTableAgent1[NUM_STATE_BINS_ERROR * NUM_STATE_BINS_LOAD][NUM_ACTIONS];
float stateAgent1[2] = {0.0, 0.0}; // Stan agenta 1: [błąd, obciążenie]
float actionAgent1 = 0.0; // Akcja agenta 1

// Funkcja aktualizująca tablicę Q-learning dla agenta 1
void updateQAgent1(float state[2], float action, float reward, float nextState[2]) {
    int discreteState[2];
    int discreteNextState[2];

    // Dyskretyzacja stanów
    discretizeStateAgent1(state, discreteState);
    discretizeStateAgent1(nextState, discreteNextState);

    // Oblicz indeksy stanów
    int stateIndex = discreteState[0] * NUM_STATE_BINS_LOAD + discreteState[1];
    int nextStateIndex = discreteNextState[0] * NUM_STATE_BINS_LOAD + discreteNextState[1];
    int actionIndex = (int)action;

    // Oblicz wartość Q dla obecnego stanu i akcji
    float currentQ = qTableAgent1[stateIndex][actionIndex];

    // Znajdź maksymalną wartość Q dla następnego stanu
    float maxNextQ = qTableAgent1[nextStateIndex][0];
    for (int i = 1; i < NUM_ACTIONS_AGENT1; i++) {
        if (qTableAgent1[nextStateIndex][i] > maxNextQ) {
            maxNextQ = qTableAgent1[nextStateIndex][i];
        }
    }

    // Zaktualizuj wartość Q
    qTableAgent1[stateIndex][actionIndex] = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);
}

// Funkcja dyskretyzująca stan agenta 1
void discretizeStateAgent1(float state[2], int discreteState[2]) {
    discreteState[0] = (int)((state[0] - MIN_ERROR) / (MAX_ERROR - MIN_ERROR) * NUM_STATE_BINS_ERROR);
    discreteState[1] = (int)((state[1] - MIN_LOAD) / (MAX_LOAD - MIN_LOAD) * NUM_STATE_BINS_LOAD);
}

// Funkcja wybierająca akcję dla agenta 1
float chooseActionAgent1(int discreteState[2], float epsilon) {
    if (random(0, 100) < epsilon * 100) {
        return random(0, NUM_ACTIONS_AGENT1);
    } else {
        int stateIndex = discreteState[0] * NUM_STATE_BINS_LOAD + discreteState[1];
        int bestAction = 0;
        float bestValue = qTableAgent1[stateIndex][0];
        for (int i = 1; i < NUM_ACTIONS_AGENT1; i++) {
            if (qTableAgent1[stateIndex][i] > bestValue) {
                bestValue = qTableAgent1[stateIndex][i];
                bestAction = i;
            }
        }
        return bestAction;
    }
}

// Funkcja wykonująca akcję dla agenta 1
void executeActionAgent1(float action) {
    int actionIndex = (int)action;

    switch (actionIndex) {
        case 0:
            // Przykładowa akcja: zwiększenie napięcia
            increaseVoltage();
            break;
        case 1:
            // Przykładowa akcja: zwiększenie prądu wzbudzenia
            // Kod usunięty
            break;
        case 2:
            // Przykładowa akcja: zmniejszenie prądu wzbudzenia
            // Kod usunięty
            break;
        case 3:
            // Przykładowa akcja: resetowanie kontrolera napięcia
            // Kod usunięty
            break;
        default:
            // Domyślna akcja: brak działania
            Serial.println("Nieznana akcja");
            break;
    }
}

// Funkcja ucząca agenta 1
void trainAgent1() {
    // Oblicz nagrodę na podstawie wydajności
    float reward = calculateEfficiency(voltageIn[0], currentIn[0], externalVoltage, externalCurrent);

    // Oblicz następny stan
    float nextState[2] = {VOLTAGE_SETPOINT - currentVoltage, currentCurrent};

    // Zaktualizuj tablicę Q-learning
    updateQAgent1(stateAgent1, actionAgent1, reward, nextState);

    // Zaktualizuj stan agenta
    stateAgent1[0] = nextState[0];
    stateAgent1[1] = nextState[1];

    // Wybierz nową akcję na podstawie zaktualizowanego stanu
    actionAgent1 = chooseActionAgent1(stateAgent1, epsilon);

    // Wykonaj wybraną akcję
    executeActionAgent1(actionAgent1);
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
        autoTunePID(Kp, Ki, Kd, VOLTAGE_SETPOINT, currentVoltage); // Wywołanie funkcji autoTunePID
        agent2IncreasedExcitation = false; // Resetowanie flagi
    }
    if (agent3MinimizedBraking) {
        Serial.println("Agent 3 zminimalizował hamowanie");
        // Przykład: agent 1 reaguje na minimalizowanie hamowania przez agenta 3
        autoTunePID(Kp, Ki, Kd, VOLTAGE_SETPOINT, currentVoltage); // Wywołanie funkcji autoTunePID
        agent3MinimizedBraking = false; // Resetowanie flagi
    }
}


// Funkcja wykonująca akcję agenta
void performAction(int agentId, float action) {
    switch (agentId) {
        case 1:
            // Akcje agenta 1
            switch ((int)action) {
                case 0: Kp += 0.1; break;
                case 1: Kp -= 0.1; break;
                case 2: Ki += 0.1; break;
                case 3: Ki -= 0.1; break;
                case 4: Kd += 0.1; break;
                case 5: Kd -= 0.1; break;
                default: break;
            }
            // Upewnij się, że wartości PID są w odpowiednich zakresach
            Kp = constrain(Kp, 0.0, 5.0);
            Ki = constrain(Ki, 0.0, 1.0);
            Kd = constrain(Kd, 0.0, 2.0);
            break;
        case 3:
            // Akcje agenta 3
            switch ((int)action) {
                case 0:
                    analogWrite(mosfetPin, analogRead(mosfetPin) - PWM_INCREMENT);
                    agent3MinimizedBraking = true;
                    break;
                case 1:
                    analogWrite(mosfetPin, analogRead(mosfetPin) + PWM_INCREMENT);
                    break;
                default:
                    Serial.println("Nieznana akcja, kara za zwiększenie hamowania");
                    break;
            }
            break;
        default:
            Serial.println("Nieznany agent");
            break;
    }
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
    performAction(3, actionAgent3);
}

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

// Funkcja symulująca kontrolę napięcia
float simulateVoltageControl(float Kp, float Ki, float Kd, float setpoint, float currentVoltage, int simulationSteps, float timeStep) {
    float currentError, integral = 0, derivative, previousError = 0, controlSignal, simulatedEfficiency;

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

// Funkcja obsługująca komendy z portu szeregowego
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
    float currentError;
    float maxError = 0.0;
    float minError = 0.0;

    // Wstępne ustawienia PID
    Kp = 1.0;
    Ki = 0.0;
    Kd = 0.0;

    // Pętla do wykrywania oscylacji
    while (!oscillationsDetected && (millis() - startTime) < TUNING_TIMEOUT) {
        currentTime = millis();
        currentError = calculateError(VOLTAGE_SETPOINT, readVoltage());

        // Aktualizacja maksymalnego i minimalnego błędu
        if (currentError > maxError) maxError = currentError;
        if (currentError < minError) minError = currentError;

        // Sprawdzenie, czy wystąpiły oscylacje
        if ((maxError - minError) > TOLERANCE) {
            oscillationsDetected = true;
            Tu = (currentTime - startTime) / 1000.0; // Okres oscylacji w sekundach
            Ku = 4.0 * (VOLTAGE_SETPOINT / (maxError - minError)); // Krytyczny współczynnik wzmocnienia
        }

        float controlSignal = Kp * currentError;
        analogWrite(mosfetPin, constrain(controlSignal, 0, 255));
        delay(100); // Opóźnienie dla stabilizacji
    }

    // Ustawienia PID na podstawie Ziegler-Nichols
    if (oscillationsDetected) {
        Kp = 0.6 * Ku;
        Ki = 2.0 * Kp / Tu;
        Kd = Kp * Tu / 8.0;
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

// Funkcja celu dla optymalizatora
float objectiveFunction(const float* params) {
    float Kp = params[0];
    float Ki = params[1];
    float Kd = params[2];
    float controlSignal = simulateVoltageControl(Kp, Ki, Kd);
    float error = calculateError(VOLTAGE_SETPOINT, controlSignal);
    return abs(error); // Minimalizujemy wartość bezwzględną błędu
}

// Funkcja aktualizacji sterowania
void updateControl() {
    float controlSignal = Kp * currentError;
    analogWrite(mosfetPin, constrain(controlSignal, 0, 255));
    delay(100); // Opóźnienie dla stabilizacji
}

int analogRead(int pin) {
    // Implementacja odczytu analogowego
    // W przypadku ESP8266, używamy funkcji analogRead z biblioteki Arduino
    return ::analogRead(pin);
}

// Inicjalizacja tablicy Q-learning
float qTable[NUM_STATE_BINS_ERROR * NUM_STATE_BINS_LOAD * NUM_STATE_BINS_KP * NUM_STATE_BINS_KI * NUM_STATE_BINS_KD][NUM_ACTIONS] = {0};

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
        float bestValue = qTable[stateIndex][0];
        for (int i = 1; i < NUM_ACTIONS; i++) {
            if (qTable[stateIndex][i] > bestValue) {
                bestValue = qTable[stateIndex][i];
                bestAction = i;
            }
        }
        return bestAction;
    }
}

// Funkcja aktualizująca tablicę Q
void updateQTable(int stateIndex, int action, float reward, int nextStateIndex) {
    float bestNextValue = qTable[nextStateIndex][0];
    for (int i = 1; i < NUM_ACTIONS; i++) {
        if (qTable[nextStateIndex][i] > bestNextValue) {
            bestNextValue = qTable[nextStateIndex][i];
        }
    }
    qTable[stateIndex][action] = (1 - learningRate) * qTable[stateIndex][action] + learningRate * (reward + discountFactor * bestNextValue);
}

// Główna pętla uczenia
void qLearningAgent3() {
    // Inicjalizacja zmiennych stanu
    float error = VOLTAGE_SETPOINT - currentVoltage;
    float load = currentCurrent; // Przykładowe obciążenie
    int stateIndex = getStateIndex(error, load, Kp, Ki, Kd);
    int action = chooseAction(stateIndex);

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
    float newError = VOLTAGE_SETPOINT - simulateVoltageControl(Kp, Ki, Kd);
    float reward = -abs(newError); // Nagroda jest ujemną wartością błędu, aby minimalizować błąd

    int nextStateIndex = getStateIndex(newError, load, Kp, Ki, Kd);
    updateQTable(stateIndex, action, reward, nextStateIndex);
}

unsigned long lastUpdateAgent1 = 0;
unsigned long lastUpdateAgent2 = 0;
unsigned long lastUpdateAgent3 = 0;
const unsigned long updateInterval = 1;

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


// Cel projektu:
// System stabilizacji napięcia z wykorzystaniem trzech agentów uczących się:
// * Agent1: Odpowiedzialny za stabilizację napięcia wyjściowego 230 volt.
// * Agent2: Steruje 24 cewkami wzbudzenia w prądnicy, dążąc do utrzymania prądu wzbudzenia na poziomie 25A.
// * Agent3: Minimalizuje hamowanie, nie wpływając na działanie Agent2 (sterowanie cewkami wzbudzenia).

// Współpraca agentów:
// * Agent1 i Agent2 przekazują informacje zwrotne do Agent3, informując go o wpływie swoich akcji na hamowanie.
// * Agent3 wykorzystuje te informacje zwrotne do podejmowania decyzji, które minimalizują hamowanie, jednocześnie wspierając cele innych agentów.

// Ostateczny cel:
// Osiągnięcie minimalnego hamowania przy jednoczesnym utrzymaniu wysokiego prądu wzbudzenia w cewkach i stabilizacji napięcia na poziomie 230V.

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
        return discretize(error, generatorLoad, Kp, Ki, Kd, NUM_STATE_BINS_ERROR, NUM_STATE_BINS_LOAD, NUM_STATE_BINS_KP, NUM_STATE_BINS_KI, NUM_STATE_BINS_KD);
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
        return calculateBasicReward(error, voltageDrop);
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

    int discretize(float error, float generatorLoad, float Kp, float Ki, float Kd, int numBinsError, int numBinsLoad, int numBinsKp, int numBinsKi, int numBinsKd) {
        float normalizedError = (error - MIN_ERROR) / (MAX_ERROR - MIN_ERROR);
        float normalizedLoad = (generatorLoad - MIN_LOAD) / (MAX_LOAD - MIN_LOAD);
        float normalizedKp = (Kp - MIN_KP) / (MAX_KP - MIN_KP);
        float normalizedKi = (Ki - MIN_KI) / (MAX_KI - MIN_KI);
        float normalizedKd = (Kd - MIN_KD) / (MAX_KD - MIN_KD);

        int errorBin = constrain((int)(normalizedError * numBinsError), 0, numBinsError - 1);
        int loadBin = constrain((int)(normalizedLoad * numBinsLoad), 0, numBinsLoad - 1);
        int kpBin = constrain((int)(normalizedKp * numBinsKp), 0, numBinsKp - 1);
        int kiBin = constrain((int)(normalizedKi * numBinsKi), 0, numBinsKi - 1);
        int kdBin = constrain((int)(normalizedKd * numBinsKd), 0, numBinsKd - 1);

        return errorBin + numBinsError * (loadBin + numBinsLoad * (kpBin + numBinsKp * (kiBin + numBinsKi * kdBin)));
    }

    float calculateBasicReward(float error, float voltageDrop) {
        float reward = 1.0 / (abs(error) + 1); // Dodanie 1, aby uniknąć dzielenia przez 0
        reward -= voltageDrop * 0.01;
        return reward;
    }
};


/* Agent 2 jest zaprojektowany wyłącznie do STEROWANIA (ZWIĘKSZANIA) prądu wzbudzenia w cewkach.
 * Nie należy dodawać funkcjonalności zmniejszania prądu wzbudzenia ani do Agenta 2, ani do Agenta 3.
 */

class Agent2 {
public:
    float qTable[NUM_STATES_AGENT2][NUM_ACTIONS_AGENT2];
    int state;
    float feedbackFromAgent3 = 0.0;

    Agent2() {
        // Inicjalizacja tablicy Q
        for (int i = 0; i < NUM_STATES_AGENT2; i++) {
            for (int j = 0; j < NUM_ACTIONS_AGENT2; j++) {
                qTable[i][j] = 0.0;
            }
        }
        state = 0;
    }

    // Wybór akcji na podstawie epsilon-greedy
    int chooseAction() {
        if (random(0, 100) < testEpsilon * 100) {
            return random(0, NUM_ACTIONS_AGENT2);
        } else {
            return getBestAction();
        }
    }

    // Wykonanie wybranej akcji
    void executeAction(int action) {
        int current = analogRead(muxInputPin);
        if (current + PWM_INCREMENT <= MAX_CURRENT) {
            switch (action) {
                case 0:
                    analogWrite(excitationBJT1Pin, current + PWM_INCREMENT);
                    break;
                case 1:
                    analogWrite(excitationBJT2Pin, current + PWM_INCREMENT);
                    break;
                case 3:
                    for (int i = 1; i <= 11; i += 2) {
                        analogWrite(excitationBJT1Pin, current + PWM_INCREMENT);
                    }
                    break;
                case 2:
                default:
                    // Utrzymaj prąd wzbudzenia (bez zmian)
                    if (current >= MAX_CURRENT - TOLERANCE) {
                        analogWrite(excitationBJT1Pin, MAX_CURRENT);
                        analogWrite(excitationBJT2Pin, MAX_CURRENT);
                    }
                    break;
            }
        }
    }

    // Aktualizacja tablicy Q
    void updateQ(int nextState, float reward, int action) {
        float maxNextQ = *std::max_element(qTable[nextState], qTable[nextState] + NUM_ACTIONS_AGENT2);
        qTable[state][action] += testLearningRate * (reward + testDiscountFactor * maxNextQ - qTable[state][action]);
    }

    // Dyskretyzacja stanu
    int discretizeState(float error, float load) {
        int errorBin = (int)((error + VOLTAGE_SETPOINT) * NUM_STATE_BINS_ERROR / (VOLTAGE_SETPOINT * 2));
        int loadBin = (int)(load * NUM_STATE_BINS_LOAD);
        return errorBin * NUM_STATE_BINS_LOAD + loadBin;
    }

    // Obliczenie nagrody
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

    // Odbieranie informacji od agenta 3
    void odbierz_informacje_od_agenta3(float feedback) {
        feedbackFromAgent3 = feedback;
    }

    // Proces uczenia agenta
    void learn(float error, float load) {
        int currentState = discretizeState(error, load);
        int action = chooseAction();
        executeAction(action);

        float newError = VOLTAGE_SETPOINT - currentVoltage;
        float newLoad = currentCurrent / MAX_CURRENT; // Normalizacja obciążenia
        int nextState = discretizeState(newError, newLoad);

        float nextObservation = analogRead(muxInputPin);
        float rewardValue = reward(nextObservation, feedbackFromAgent3);

        updateQ(nextState, rewardValue, action);
        state = nextState;
    }

private:
    // Znalezienie najlepszej akcji na podstawie tablicy Q
    int getBestAction() {
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
    void performAction() {
        // Agent 3 zawsze zmniejsza hamowanie
        currentBrakePWM = max(currentBrakePWM - PWM_INCREMENT, MIN_BRAKE_PWM);
        analogWrite(mosfetPin, currentBrakePWM);
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
        float maxQ = qTable[stateIndex][0];
        for (int i = 1; i < NUM_ACTIONS_AGENT3; i++) {
            if (qTable[stateIndex][i] > maxQ) {
                maxQ = qTable[stateIndex][i];
                bestAction = i;
            }
        }
        return bestAction;
    }

private:
    float qTable[NUM_STATES_AGENT3][NUM_ACTIONS_AGENT3] = {0}; // Tablica Q-wartości
};

int main() {
    Agent1 agent1;
    Agent2 agent2;
    Agent3 agent3;

    // Przykładowa logika główna
    float state[2] = {230.0, 25.0};
    int discreteState[2];

    agent3.discretizeStateAgent3(state, discreteState);
    int action = agent3.chooseActionAgent3(discreteState);
    agent3.executeActionAgent3(action);
    float reward = agent3.calculateRewardAgent3(state[0], state[1]);
    int nextState = agent3.discretizeStateAgent3(); // Zakładając, że stan jest aktualizowany gdzieś indziej
    agent3.updateQAgent3(discreteState[0], action, reward, nextState);

    return 0;
}

// Przykładowe dane wejściowe
int states[] = {0, 1, 2};
int actions[] = {0, 1, 2};
float rewards[] = {1.0, 0.5, -1.0};
int nextStates[] = {1, 2, 0};

// Przypisz testowe wartości do używanych zmiennych raz na starcie
void initializeTestValues() {
    static bool initialized = false;
    if (!initialized) {
        epsilon = testEpsilon;
        learningRate = testLearningRate;
        discountFactor = testDiscountFactor;
        initialized = true;
    }
}

void mainLoop() {
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

void checkComputerConnection() {
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

// Funkcja ustawiająca docelowe napięcie dla wszystkich agentów na średnią wartość
void setTargetVoltageForAllAgents(float averageTargetVoltage) {
    int numAgents = sizeof(agents) / sizeof(agents[0]);
    for (int i = 0; i < numAgents; i++) {
        agents[i].targetVoltage = averageTargetVoltage;
    }
}

// Globalne zmienne dla minimalnej i maksymalnej mocy wejściowej
float minInputPower = 1e-7; // Początkowa wartość, dostosowana w funkcji
float minObservedPower = 1e-6;
float maxObservedPower = 1e-3;

// Aktualizuj minimalną i maksymalną obserwowaną moc
void updateObservedPower(float inputPower) {
    if (inputPower > 0 && inputPower < minObservedPower) {
        minObservedPower = inputPower;
    }

    if (inputPower > maxObservedPower) {
        maxObservedPower = inputPower;
    }

    // Dostosuj próg na podstawie obserwowanych wartości
    minInputPower = minObservedPower * 0.1; // Możesz dostosować współczynnik 0.1
}

// Inicjalizacja funkcji setup
void setup() {
    // Inicjalizacja komunikacji szeregowej
    Serial.begin(115200); // Ustawienie prędkości transmisji na 115200 bps
    while (!Serial) {
        ; // Czekaj na otwarcie portu szeregowego
    }

    // Inicjalizacja wyświetlacza
    if (!display.begin(SH1106_SWITCHCAPVCC, 0x3C)) {
        Serial.println(F("SH1106 allocation failed"));
        for (;;); // Zatrzymanie programu w przypadku niepowodzenia
    }
    display.display();
    delay(2000);
    display.clearDisplay();

    // Inicjalizacja semaforów
    initializeSemaphores();

    // Inicjalizacja EEPROM
    EEPROM.begin(512);

    // Ustawienie docelowego napięcia dla wszystkich agentów
    setTargetVoltageForAllAgents(VOLTAGE_SETPOINT);

    // Wywołanie funkcji centralController
    centralController();

    // Inicjalizacja zmiennych globalnych
    initializeGlobalVariables();

    // Przykładowe dane do zapisu w EEPROM
    int address = 0;
    byte value = 42;
    writeEEPROM(address, value);

    // Inicjalizacja mapy funkcji naprawczych
    initializeFixFunctions();

    // Trenuj model uczenia maszynowego
    trainModel();

    // Inicjalizacja komunikacji I2C
    Wire.begin();

    // Inicjalizacja tablicy Q
    initializeQTable();

    // Inicjalizacja pinów
    initializePins();

    // Tworzenie zadań dla agentów
    createAgentTasks();

    // Inicjalizacja komponentów
    initializeComponents();

    Serial.println(F("Setup complete"));
}

// Inicjalizacja semaforów
void initializeSemaphores() {
    resourceSemaphore = xSemaphoreCreateMutex();
    if (resourceSemaphore == NULL) {
        Serial.println("Error: Failed to create resourceSemaphore");
    }

    eepromSemaphore = xSemaphoreCreateMutex();
    if (eepromSemaphore == NULL) {
        Serial.println("Error: Failed to create eepromSemaphore");
    }
}

// Inicjalizacja mapy funkcji naprawczych
void initializeFixFunctions() {
    fixFunctions["default"] = defaultFixFunction;
    fixFunctions["Voltage out of range"] = resetVoltageController;
    fixFunctions["Current exceeds maximum limit"] = reduceExcitationCurrent;
    fixFunctions["Braking effect out of range"] = calibrateBrakingEffect;
    // Dodaj inne funkcje naprawcze w razie potrzeby
}

// Inicjalizacja pinów
void initializePins() {
    int pins[] = {muxSelectPinA, muxSelectPinB, muxSelectPinC, muxSelectPinD, mosfetPin, bjtPin1, bjtPin2, bjtPin3, excitationBJT1Pin, excitationBJT2Pin, newPin1, newPin2, newPin3, bjtPin4};
    for (int pin : pins) {
        pinMode(pin, OUTPUT);
    }
    pinMode(muxInputPin, INPUT);
}

// Tworzenie zadań dla agentów
void createAgentTasks() {
    xTaskCreate(agent1Function, "Agent1", 2048, NULL, 1, NULL);
    xTaskCreate(agent2Function, "Agent2", 2048, NULL, 1, NULL);
    xTaskCreate(agent3Function, "Agent3", 2048, NULL, 1, NULL);
}

// Inicjalizacja komponentów
void initializeComponents() {
    initializeServer();
    initializeOptimizer();
    displayWelcomeMessage();
    initializeGlobalVariables();
    initializeWiFi();
}

// Inicjalizacja serwera
void initializeServer() {
    server.begin();
}

// Inicjalizacja PID
void initializePID() {
    PID pid(1.0, 0.1, 0.01);
    float setpoint = 230.0; // Docelowe napięcie
    float measuredValue = readVoltage(); // Funkcja do odczytu napięcia

    autoTunePID(pid, setpoint, measuredValue);

    // Teraz możesz używać pid do kontroli
    float output = pid.compute(setpoint, measuredValue);
    analogWrite(mosfetPin, constrain(output, 0, 255));
}



// Inicjalizacja optymalizatora
void initializeOptimizer() {
    optimizer.initialize(3, bounds, 50, 10);
}

// Wyświetlenie wiadomości powitalnej
void displayWelcomeMessage() {
    char buffer[64];
    strcpy_P(buffer, welcomeMessage);
    display.println(buffer);
    display.display();
}

// Inicjalizacja zmiennych globalnych
void initializeGlobalVariables() {
    externalVoltage = 0.0;
    externalCurrent = 0.0;
    efficiency = 0.0;
    efficiencyPercent = 0.0;
    voltageDrop = 0.0;
    currentVoltage = 0.0;
    currentCurrent = 0.0;
}

// Inicjalizacja WiFi
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

// Aktualizacja priorytetu agenta
void updateAgentPriority(int agentId) {
    for (int i = 0; i < sizeof(agents) / sizeof(agents[0]); i++) {
        if (agents[i].id == agentId) {
            agents[i].waitCount++; // Zwiększ licznik prób uzyskania dostępu
            agents[i].waitTime = millis(); // Ustaw czas oczekiwania
            // Ustal priorytet na podstawie liczby prób i czasu oczekiwania
            agents[i].priority = agents[i].waitCount + (millis() - agents[i].waitTime) / 1000; 
            agents[i].priority = constrain(agents[i].priority, 0, 10); // Ogranicz maksymalny priorytet
            break;
        }
    }
}

// Zwraca agenta o najwyższym priorytecie
int getHighestPriorityAgent() {
    int highestPriority = -1;
    int index = -1;
    for (int i = 0; i < sizeof(agents) / sizeof(agents[0]); i++) {
        if (agents[i].priority > highestPriority) {
            highestPriority = agents[i].priority;
            index = i;
        }
    }
    return index; // Zwraca indeks agenta o najwyższym priorytecie
}

// Ustawia PWM
bool setPWM(int pin, float action, int agentId) {
    int agentIndex = getHighestPriorityAgent();
    
    // Sprawdź, czy agent o najwyższym priorytecie to ten, który prosi o dostęp
    if (agentIndex != -1 && agents[agentIndex].id == agentId) {
        if (xSemaphoreTake(resourceSemaphore, pdMS_TO_TICKS(100)) == pdTRUE) {
            unsigned long currentTime = millis();
            if (currentTime - agents[agentIndex].lastAccessTime < timeSlice) {
                int pwmValue = (int)(action * 255.0);
                pwmValue = constrain(pwmValue, 0, 255);
                analogWrite(pin, pwmValue);
                agents[agentIndex].lastAccessTime = currentTime; // Ustaw czas ostatniego dostępu
                xSemaphoreGive(resourceSemaphore); // Zwolnienie semafora
                
                // Resetowanie priorytetu agenta po udanym dostępie
                agents[agentIndex].waitCount = 0; // Resetuj licznik prób
                agents[agentIndex].priority = 0;   // Resetuj priorytet
                return true; // Sukces
            } else {
                // Jeśli czas przydzielony minął, resetuj dostęp
                xSemaphoreGive(resourceSemaphore); 
                return false; // Niepowodzenie, czas przydzielony minął
            }
        } else {
            Serial.println("Timeout while waiting for semaphore.");
            updateAgentPriority(agentId); // Aktualizacja priorytetu, jeśli czas oczekiwania
            return false; // Niepowodzenie
        }
    }
    return false; // Agent nie ma dostępu
}

// Wykonuje akcję agenta
void executeActionAgent(int agentId, float action) {
    int pin;
    switch (agentId) {
        case 1:
            pin = 9; // Pin dla Agenta 1
            break;
        case 2:
            pin = 10; // Pin dla Agenta 2
            break;
        case 3:
            pin = 11; // Pin dla Agenta 3
            break;
        default:
            Serial.println("Invalid agent ID.");
            return;
    }
    if (!setPWM(pin, action, agentId)) {
        Serial.printf("Agent %d failed to set PWM.\n", agentId);
    }
}

// Inicjalizacja komponentów
void initializeComponents() {
    // Dodaj tutaj kod inicjalizujący inne komponenty
}

// Wyświetlenie wiadomości powitalnej
void displayWelcome() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(WHITE);
    display.setCursor(0, 0);
    display.println("Witamy!");
    display.display();
}

// Inicjalizacja zmiennych globalnych
void initializeGlobals() {
    // Dodaj tutaj kod inicjalizujący zmienne globalne
}

// Automatyczna kalibracja systemu
void autoCalibrateSystem() {
    // Dodaj tutaj kod kalibracji systemu
}

// Inicjalizacja parametrów PID
void initializePIDParams() {
    handlePIDParams(Kp, Ki, Kd, false);
    previousError = 0;
    integral = 0;
}

// Inicjalizacja tras serwera
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
    return (voltage > threshold) ? 1.0 : 0.0; // Próg przekroczony lub nie
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
        currentIn[i] = calculateCurrent(analogRead(muxInputPin));
    }
}

// Funkcja obliczania prądu
float calculateCurrent(int raw_current_adc) {
    float sensorVoltage = raw_current_adc * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);
    float voltageOffset = VOLTAGE_REFERENCE / 2;
    sensorVoltage -= voltageOffset;
    const float sensitivity = 0.185; // Dostosuj, jeśli czułość Twoich czujników jest inna
    return sensorVoltage / sensitivity;
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
    Serial.println("Rozpoczynanie automatycznej kalibracji...");

    // Kalibracja napięcia
    float measuredVoltage = readVoltage();
    Serial.print("Zmierzono napięcie: ");
    Serial.println(measuredVoltage);

    // Kalibracja prądu
    float measuredCurrent = readCurrent();
    Serial.print("Zmierzono prąd: ");
    Serial.println(measuredCurrent);

    // Ustawienie parametrów kalibracji
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
    int excitationValue = (currentIn[0] > LOAD_THRESHOLD) ? 255 : 0;
    analogWrite(excitationBJT1Pin, excitationValue);
    analogWrite(excitationBJT2Pin, excitationValue);
}

// Funkcja aktualizacji wyświetlacza
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

// Funkcja obsługująca komendy z portu szeregowego
void handleSerialCommands() {
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        command.trim(); // Usuwa białe znaki na początku i końcu

        if (command == "START") {
            Serial.println("Komenda START otrzymana");
            // Dodaj tutaj kod do uruchomienia odpowiedniej funkcji
        } else if (command == "STOP") {
            Serial.println("Komenda STOP otrzymana");
            // Dodaj tutaj kod do zatrzymania odpowiedniej funkcji
        } else if (command == "OPTIMIZE") {
            Serial.println("Komenda OPTIMIZE otrzymana");
            optimizePID();
        } else {
            Serial.println("Nieznana komenda: " + command);
        }
    }
}

// Funkcja dostosowująca częstotliwość sterowania
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

// Funkcja obliczająca wydajność
float calculateEfficiency(float voltageIn, float currentIn, float externalVoltage, float externalCurrent) {
    float inputPower = voltageIn * currentIn;
    float outputPower = externalVoltage * externalCurrent;

    if (inputPower == 0) {
        return 0;
    }

    return outputPower / inputPower;
}

// Funkcja logująca błędy do EEPROM
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

// Funkcja odczytująca logi błędów z EEPROM
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

// Funkcja wykrywająca anomalie
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

// Funkcja obsługująca wykryte anomalie
void handleAnomaly(String anomaly) {
    Serial.println("Anomaly detected: " + anomaly);
    logError(anomaly);
    generateAndImplementFix(anomaly); // Wywołanie funkcji naprawczej dla anomalii
}

// Prosta funkcja przewidująca błędy na podstawie reguł
String predictError(float voltage, float current, float brakingEffect) {
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

// Funkcja dodająca funkcję naprawczą dla danego błędu
void addFixFunction(String error, void(*fixFunction)()) {
    fixFunctions[error] = fixFunction;
}

// Domyślna funkcja naprawcza
void defaultFixFunction() {
    Serial.println("No specific fix available. Executing default fix...");
    // Kod domyślnej naprawy
}

// Funkcja trenująca model
void trainModel() {
    // Przykładowe trenowanie modelu (w rzeczywistości powinno być bardziej zaawansowane)
    modelWeights[0] = 1.2; // Waga dla napięcia
    modelWeights[1] = 1.5; // Waga dla prądu
    modelWeights[2] = 0.8; // Waga dla efektu hamowania
    modelWeights[3] = 2.0; // Dodatkowa waga dla bardziej złożonego modelu
}

// Funkcja przewidująca błędy za pomocą modelu
String predictWithModel(float voltage, float current, float brakingEffect) {
    // Prosty model liniowy do przewidywania błędów
    float score = modelWeights[0] * voltage + modelWeights[1] * current + modelWeights[2] * brakingEffect + modelWeights[3] * (voltage * current);
    if (score > 1500) { // Zwiększony próg dla wykrywania większych anomalii
        return "Anomaly detected by model";
    }
    return "";
}

// Funkcja aktualizująca wyświetlacz
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

void loop() {
    unsigned long currentMillis = millis();

    // Aktualizacja wyświetlacza
    if (currentMillis - previousMillisDisplay >= intervalDisplay) {
        previousMillisDisplay = currentMillis;
        updateDisplay();
    }

    // Obsługa poleceń szeregowych
    handleSerialCommands();

    // Wykrywanie połączenia z komputerem
    detectComputerConnection();

    // Trenowanie agentów
    trainAgents();

    // Komunikacja między agentami
    communicateBetweenAgents();

    // Kontrola tranzystorów
    controlTransistors(currentVoltage, currentCurrent);

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
    autoTunePID();

    // Przykładowe wywołanie funkcji hill climbing
    hillClimbing();

    // Przykładowe wywołanie funkcji agenta 3
    handleAgent3();

    // Krótkie opóźnienie, aby symulować czas rzeczywisty
    delay(100);

    // Aktualizacja bieżących wartości napięcia i prądu
    updateCurrentValues();

    // Regularne wywoływanie optymalizacji PID
    handlePIDOptimization();

    // Odczyt mocy wejściowej i dostosowanie minimalnej mocy wejściowej
    float inputPower = readInputPower(); // Odczytaj moc wejściową
    updateObservedPower(inputPower); // Aktualizuj obserwowaną moc

    // Dodanie nowej pętli z warunkiem wyjścia
    handleLoopWithExitCondition();

    // Symulacja systemu
    simulateSystemPerformance();

    // Przykładowe aktualizacje potrzeb agentów
    updateAgentNeeds();

    // Wysyłanie danych do komputera
    sendDataToComputer();

    // Aktualizacja priorytetów agentów
    updateAgentPriorities();

    // Przydzielanie zasobów agentom
    allocateResources();

    // Przykładowa logika sterowania
    controlLogic();

    // Monitorowanie błędów i wykrywanie anomalii
    monitorErrors();
    detectAnomalies();
    delay(1000); // Sprawdzaj błędy co 1 sekundę

    // Przypisz testowe wartości do używanych zmiennych raz na starcie
    static bool initialized = false;
    if (!initialized) {
        epsilon = testEpsilon;
        learningRate = testLearningRate;
        discountFactor = testDiscountFactor;
        initialized = true;
    }
}


    // Wywołania funkcji Q-learning dla agentów
    qLearning(1, voltageIn[0], currentIn[0], efficiency, voltageDrop, externalVoltage, externalCurrent);
    qLearning(2, voltageIn[0], currentIn[0], efficiency, voltageDrop, externalVoltage, externalCurrent);
    qLearning(3, voltageIn[0], currentIn[0], efficiency, voltageDrop, externalVoltage, externalCurrent);

    // Przykładowe wywołanie funkcji simulateVoltageControl
    float Kp = 1.0, Ki = 0.5, Kd = 0.1;
    float voltageError = simulateVoltageControl(Kp, Ki, Kd);
    Serial.println(voltageError);

    // Przykładowe wywołanie funkcji simulateExcitationControl
    float excitationError = simulateExcitationControl();
    Serial.println(excitationError);

    // Przykładowe wywołanie funkcji someFunctionOfOtherParameters
    float rotationalSpeed = 3000.0;
    float torque = 50.0;
    float frictionCoefficient = 0.8;
    float brakingEffect = someFunctionOfOtherParameters(rotationalSpeed, torque, frictionCoefficient);
    Serial.println(brakingEffect);

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
}

// Funkcja do trenowania agentów
void trainAgents() {
    trainAgent1();
    trainAgent2();
}

// Funkcja do auto-strojenia PID
void autoTunePID() {
    PID pid(Kp, Ki, Kd);
    autoTunePID(pid, VOLTAGE_SETPOINT, readVoltage());
}

// Funkcja do obsługi agenta 3
void handleAgent3() {
    float state[2] = {readVoltage(), readExcitationCurrent()};
    int discreteState[2];
    discretizeStateAgent3(state, discreteState);
    float action = chooseActionAgent3(discreteState);
    executeActionAgent3(action);
    float reward = calculateRewardAgent3(state, action);
    float nextState[2] = {readVoltage(), readExcitationCurrent()};
    updateQAgent3(state, action, reward, nextState);
}

// Funkcja do regularnej optymalizacji PID
void handlePIDOptimization() {
    unsigned long currentTime = millis();
    if (currentTime - lastOptimizationTime > OPTIMIZATION_INTERVAL) {
        lastOptimizationTime = currentTime;
        optimizer.optimize();
        optimizer.getBestParams(params);
        bestEfficiency = -optimizer.getBestObjective();
        Kp = params[0];
        Ki = params[1];
        Kd = params[2];
        printOptimizationResults();
    }
}

// Funkcja do obsługi pętli z warunkiem wyjścia
void handleLoopWithExitCondition() {
    unsigned long startTime = millis();
    while (true) {
        int sensorValue = analogRead(A0);
        Serial.println(sensorValue);
        if (sensorValue > 100) {
            break;
        }
        if (millis() - startTime > 1000) {
            Serial.println("Pętla trwa zbyt długo, przerywanie...");
            break;
        }
        delay(10);
    }
}

// Funkcja do symulacji wydajności systemu
void simulateSystemPerformance() {
    float efficiency = simulateSystem(Kp, Ki, Kd);
    Serial.println("Symulowana wydajność: " + String(efficiency));
    delay(1000);
}

// Funkcja do aktualizacji potrzeb agentów
void updateAgentNeeds() {
    executeActionAgent(1, 0.5);
    delay(300);
    executeActionAgent(2, 0.75);
    delay(300);
    executeActionAgent(3, 1.0);
    delay(300);
}

// Funkcja do wysyłania danych do komputera
void sendDataToComputer() {
    if (Serial) {
        float dataToSend = analogRead(A0);
        Serial.println(dataToSend);
        delay(1000);
    }
}

// Funkcja do aktualizacji priorytetów agentów
void updateAgentPriorities() {
    for (int i = 0; i < sizeof(agents) / sizeof(agents[0]); i++) {
        increasePriority(agents[i]);
    }
}

// Funkcja do przykładowej logiki sterowania
void controlLogic() {
    if (currentCurrent > LOAD_THRESHOLD) {
        analogWrite(excitationBJT1Pin, 255);
    } else {
        analogWrite(excitationBJT1Pin, 0);
    }
}

// Funkcja do aktualizacji wartości napięcia i prądu
void updateCurrentValues() {
    currentVoltage = readVoltage();
    currentCurrent = readExcitationCurrent();
}


// Stała konwersji napięcia
const float VOLTAGE_CONVERSION_FACTOR = 5.0 / 1023.0;

// Funkcja odczytu wartości z ADC i konwersji na odpowiednią jednostkę
float readSensorValue(int pin) {
    int rawValue = analogRead(pin);
    return rawValue * VOLTAGE_CONVERSION_FACTOR;
}

// Funkcje odczytu poszczególnych wartości
float readVoltage() {
    return readSensorValue(A0);
}

float readExcitationCurrent() {
    return readSensorValue(A1);
}

float readBrakingEffect() {
    return readSensorValue(A2);
}

// Funkcja monitorująca błędy
void monitorErrors() {
    checkSensorValue("Voltage", readVoltage(), 220.0, 240.0);
    checkSensorValue("Current", readExcitationCurrent(), 0.0, MAX_CURRENT);
    checkSensorValue("Braking effect", readBrakingEffect(), 0.0, 100.0);
}

// Funkcja sprawdzająca wartość sensora i generująca odpowiednią naprawę
void checkSensorValue(String sensorName, float value, float minValue, float maxValue) {
    if (value < minValue || value > maxValue) {
        generateAndImplementFix(sensorName + " out of range");
    }
}

// Funkcja generująca i implementująca naprawę
void generateAndImplementFix(String error) {
    Serial.println("Error detected: " + error);
    logError(error);
    if (fixFunctions.find(error) != fixFunctions.end()) {
        fixFunctions[error](); // Wywołanie odpowiedniej funkcji naprawczej
    } else {
        defaultFixFunction(); // Wywołanie domyślnej funkcji naprawczej
    }
}

// Funkcje naprawcze
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

// Funkcja logowania błędów do EEPROM
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

// Funkcja odczytu logów błędów z EEPROM
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

// Funkcja wykrywająca anomalie
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

// Funkcja obsługi anomalii
void handleAnomaly(String anomaly) {
    Serial.println("Anomaly detected: " + anomaly);
    logError(anomaly);
    generateAndImplementFix(anomaly); // Wywołanie funkcji naprawczej dla anomalii
}

// Prosta logika przewidywania błędów
String predictError(float voltage, float current, float brakingEffect) {
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

// Funkcja dodająca funkcję naprawczą do mapy
void addFixFunction(String error, void(*fixFunction)()) {
    fixFunctions[error] = fixFunction;
}

// Domyślna funkcja naprawcza
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

// Definicje funkcji zapisu i odczytu EEPROM
void writeEEPROM(int address, byte value) {
    if (xSemaphoreTake(eepromSemaphore, portMAX_DELAY) == pdTRUE) {
        if (EEPROM.read(address) != value) {
            EEPROM.write(address, value);
            EEPROM.commit();
        }
        xSemaphoreGive(eepromSemaphore);
    }
}

byte readEEPROM(int address) {
    byte value = 0;
    if (xSemaphoreTake(eepromSemaphore, portMAX_DELAY) == pdTRUE) {
        value = EEPROM.read(address);
        xSemaphoreGive(eepromSemaphore);
    }
    return value;
}

// Przykładowa funkcja użycia zapisu i odczytu EEPROM
void exampleUsage() {
    int address = 0; // Adres w EEPROM
    byte valueToWrite = 42; // Wartość do zapisu

    // Zapis wartości do EEPROM
    writeEEPROM(address, valueToWrite);

    // Odczyt wartości z EEPROM
    byte readValue = readEEPROM(address);
    Serial.print("Odczytana wartość: ");
    Serial.println(readValue);
}

// Funkcja do obsługi opóźnień
void delayWithMessage(unsigned long delayTime, const String &message) {
    Serial.println(message);
    delay(delayTime);
}

    // Przykładowe użycie funkcji zapisu i odczytu EEPROM
    exampleUsage();
    delayWithMessage(1000, "Opóźnienie dla celów demonstracyjnych");
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

// Aktualizacja agentów
void updateAgents(unsigned long currentTime, unsigned long &lastUpdate, void (*qLearningAgent)(float, float, float, float, float)) {
    if (currentTime - lastUpdate >= updateInterval) {
        float error = VOLTAGE_SETPOINT - currentVoltage;
        float load = currentCurrent;
        qLearningAgent(error, load, Kp, Ki, Kd);
        lastUpdate = currentTime;
    }
}

updateAgents(currentTime, lastUpdateAgent1, qLearningAgent1);
updateAgents(currentTime, lastUpdateAgent2, qLearningAgent2);
updateAgents(currentTime, lastUpdateAgent3, qLearningAgent3);

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

// Q-learning (voltage stabilizer, excitation coils, generator braking)
void qLearning(int agent, float voltage, float current, float efficiency, float voltageDrop, float externalVoltage, float externalCurrent) {
    int state, action, nextState;
    float reward, power_output;

    if (agent == 1 || agent == 2) {
        state = discretizeState(VOLTAGE_SETPOINT - voltage, current, Kp, Ki, Kd);
        action = chooseAction(state);
        executeAction(action);
        reward = calculateReward(VOLTAGE_SETPOINT - voltage, efficiency, voltageDrop);
        nextState = discretizeState(VOLTAGE_SETPOINT - voltage, current, Kp, Ki, Kd);
        updateQ(state, action, reward, nextState);
    } else if (agent == 3) {
        power_output = externalVoltage * externalCurrent; // Obliczamy moc wyjściową generatora
        state = discretizeStateAgent3(VOLTAGE_SETPOINT - voltage, current);
        action = chooseActionAgent3(state);
        executeActionAgent3(action);
        reward = calculateRewardAgent3(efficiency, voltage, voltageDrop, power_output);
        nextState = discretizeStateAgent3(VOLTAGE_SETPOINT - voltage, current);
        updateQAgent3(state, action, reward, nextState);
    }
}


// Implementacja funkcji centralController
void centralController() {
    float averageTargetVoltage = 0.0;
    int numAgents = sizeof(agents) / sizeof(agents[0]);

    for (int i = 0; i < numAgents; i++) {
        averageTargetVoltage += agents[i].targetVoltage;
    }
    averageTargetVoltage /= numAgents;

    setTargetVoltageForAllAgents(averageTargetVoltage);

    // Dostosuj minimalny próg mocy wejściowej - dodane tutaj
    float inputPower = voltageIn[0] * currentIn[0];
    adjustMinInputPower(inputPower);
}

// Obsługa serwera
server.handleClient();

// Obsługa komend szeregowych
handleSerialCommands();

// Trening agenta 1
trainAgent1();

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



void detectComputerConnection() {
    if (Serial.available()) {
        Serial.println("Komputer podłączony. Przenoszenie mocy obliczeniowej...");
        // Wyślij komendę do komputera, aby rozpocząć przenoszenie mocy obliczeniowej
        Serial.println("START_COMPUTE");
    }
}
