
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Arduino.h>
#include <EEPROM.h>
#include <Wire.h>
#include <Adafruit_SH1106.h>
#include <ArduinoEigen.h>
#include <BayesOptimizer.h>

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

float targetVoltage = 230.0; // Docelowe napięcie 230 V
float currentVoltage;
float currentCurrent;
const float maxCurrent = 25.0; // Maksymalny prąd w amperach

// Deklaracja zmiennej globalnej
float LOAD_THRESHOLD = 0.0;

#define NUM_STATE_BINS_ERROR 10
#define NUM_STATE_BINS_LOAD 10
#define NUM_STATE_BINS_KP 10
#define NUM_STATE_BINS_KI 10
#define NUM_STATE_BINS_KD 10
#define NUM_ACTIONS 5
#define NUM_STATES_AGENT3 100
#define NUM_ACTIONS_AGENT3 5

// Definicje pinów dla tranzystorów
const int mosfetPin = D4;
const int bjtPin1 = D5;
const int bjtPin2 = D6;
const int bjtPin3 = D7;
const int excitationBJT1Pin = D8;
const int excitationBJT2Pin = D9;
const int PIN_EXTERNAL_VOLTAGE_SENSOR_1 = A1;
const int PIN_EXTERNAL_CURRENT_SENSOR_1 = A2;
const int PWM_INCREMENT = 10;

// Nowa definicja pinów
const int newPin1 = D10;
const int newPin2 = D11;
const int newPin3 = D12;
const int newPin4 = D13;

// Stałe konfiguracyjne
float LOAD_THRESHOLD = 0.5;
const float COMPENSATION_FACTOR = 0.1;
const int MAX_EXCITATION_CURRENT = 255;
const float MAX_VOLTAGE = 230.0;
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

// Funkcja obliczająca wydajność
float calculateEfficiency(float voltageIn, float currentIn, float externalVoltage, float externalCurrent) {
    float inputPower = voltageIn * currentIn;
    float outputPower = externalVoltage * externalCurrent;

    if (inputPower == 0) {
        return 0;
    }

    return outputPower / inputPower;
}

// Funkcja oceniająca próg
float evaluateThreshold(float threshold) {
    LOAD_THRESHOLD = threshold;
    // Implementacja funkcji oceniającej
    // Zwraca wartość oceny dla danego progu
    return threshold; // Przykładowa implementacja
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

// Definicje zmiennych globalnych i stałych
const float VOLTAGE_SETPOINT = 230.0;
const float COMPENSATION_FACTOR = 0.1;
const int NUM_ACTIONS = 6;
const float epsilon = 0.3;
const int NUM_STATE_BINS_ERROR = 5;
const int NUM_STATE_BINS_LOAD = 3;
const int NUM_STATE_BINS_KP = 5;
const int NUM_STATE_BINS_KI = 3;
const int NUM_STATE_BINS_KD = 3;
const float MIN_ERROR = -230.0;
const float MAX_ERROR = 230.0;
const float MIN_LOAD = 0.0;
const float MAX_LOAD = 1.0;
const float MIN_KP = 0.0;
const float MAX_KP = 5.0;
const float MIN_KI = 0.0;
const float MAX_KI = 1.0;
const float MIN_KD = 0.0;
const float MAX_KD = 5.0;
const int PWM_INCREMENT = 10;
const int PIN_CURRENT_SENSOR = A2;
const int PIN_EXCITATION_COIL_1 = D0;
const int PIN_EXCITATION_COIL_2 = D1;
float qTable[NUM_STATE_BINS_ERROR * NUM_STATE_BINS_LOAD * NUM_STATE_BINS_KP * NUM_STATE_BINS_KI * NUM_STATE_BINS_KD][NUM_ACTIONS][3];
float voltageDrop = 0.0;
float currentIn[2] = {0.0, 0.0};


// Funkcja regulująca częstotliwość sterowania
int controlFrequency = 0;
const int HIGH_FREQUENCY = 1000;
const int LOW_FREQUENCY = 100;

int constrain(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}


int analogRead(int pin) {
    // Implementacja odczytu analogowego, dostosuj według potrzeb
    return 0;
}

void analogWrite(int pin, int value) {
    // Implementacja zapisu analogowego, dostosuj według potrzeb
}

class Agent1 {
public:
    int akcja;
    float feedbackToAgent2;

    void odbierz_informacje_hamowania(float hamowanie) {
        // Implementacja logiki używającej informacji o hamowaniu
    }

    void wyslij_informacje_do_agenta3(class Agent3& agent3, class Agent2& agent2) {
        agent3.odbierz_informacje_od_agentow(akcja, agent2.akcja);
    }

    void wyslij_informacje_do_agenta2(class Agent2& agent2, float feedback) {
        agent2.odbierz_informacje_od_agenta1(feedback);
    }

    int discretizeState(float error, float generatorLoad, float Kp, float Ki, float Kd) {
        float normalizedError = (error - MIN_ERROR) / (MAX_ERROR - MIN_ERROR);
        float normalizedLoad = (generatorLoad - MIN_LOAD) / (MAX_LOAD - MIN_LOAD);
        float normalizedKp = (Kp - MIN_KP) / (MAX_KP - MIN_KP);
        float normalizedKi = (Ki - MIN_KI) / (MAX_KI - MIN_KI);
        float normalizedKd = (Kd - MIN_KD) / (MAX_KD - MAX_KD);

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
            auto bestAction = std::max_element(qTable[state], qTable[state] + NUM_ACTIONS, 
                [](const float* a, const float* b) { return a[0] < b[0]; });
            return std::distance(qTable[state], bestAction);
        }
    }

    void executeAction(int action) {
        const int pins[] = {bjtPin1, bjtPin1, bjtPin2, bjtPin2, bjtPin3, bjtPin3};
        const int increments[] = {PWM_INCREMENT, -PWM_INCREMENT, PWM_INCREMENT, -PWM_INCREMENT, PWM_INCREMENT, -PWM_INCREMENT};

        analogWrite(pins[action], constrain(analogRead(pins[action]) + increments[action], 0, 255));
    }

    float calculateReward(float error, float efficiency, float voltageDrop) {
        float reward = 1.0 / abs(error);
        reward -= voltageDrop * 0.01;
        return reward;
    }

    float reward(float next_observation) {
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
        float bestNextQ = *std::max_element(qTable[nextState][0], qTable[nextState][NUM_ACTIONS]);
        qTable[state][action][0] += learningRate * (reward + discountFactor * bestNextQ - qTable[state][action][0]);
    }

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
};


class Agent2 {
public:
    int akcja;
    float feedbackFromAgent3;
    float feedbackFromAgent1; // Dodana zmienna

    void odbierz_informacje_hamowania(float hamowanie) {
        // Implementacja logiki używającej informacji o hamowaniu
    }

    void odbierz_informacje_od_agenta3(float feedback) {
        feedbackFromAgent3 = feedback;
    }

    void odbierz_informacje_od_agenta1(float feedback) {
        feedbackFromAgent1 = feedback; // Poprawione przypisanie
    }

    void wyslij_informacje_do_agenta3(class Agent3& agent3, class Agent1& agent1) {
        agent3.odbierz_informacje_od_agentow(agent1.akcja, akcja);
    }

    int discretizeState(float arg1, float arg2, float Kp, float Ki, float Kd) {
        // Implementacja logiki dyskretyzacji stanu dla Agenta 2
        float normalizedArg1 = (arg1 - MIN_ERROR) / (MAX_ERROR - MIN_ERROR);
        float normalizedArg2 = (arg2 - MIN_LOAD) / (MAX_LOAD - MIN_LOAD);
        float normalizedKp = (Kp - MIN_KP) / (MAX_KP - MIN_KP);
        float normalizedKi = (Ki - MIN_KI) / (MAX_KI - MIN_KI);
        float normalizedKd = (Kd - MIN_KD) / (MAX_KD - MIN_KD);

        int arg1Bin = constrain((int)(normalizedArg1 * NUM_STATE_BINS_ERROR), 0, NUM_STATE_BINS_ERROR - 1);
        int arg2Bin = constrain((int)(normalizedArg2 * NUM_STATE_BINS_LOAD), 0, NUM_STATE_BINS_LOAD - 1);
        int kpBin = constrain((int)(normalizedKp * NUM_STATE_BINS_KP), 0, NUM_STATE_BINS_KP - 1);
        int kiBin = constrain((int)(normalizedKi * NUM_STATE_BINS_KI), 0, NUM_STATE_BINS_KI - 1);
        int kdBin = constrain((int)(normalizedKd * NUM_STATE_BINS_KD), 0, NUM_STATE_BINS_KD - 1);

        return arg1Bin + NUM_STATE_BINS_ERROR * (arg2Bin + NUM_STATE_BINS_LOAD * (kpBin + NUM_STATE_BINS_KP * (kiBin + NUM_STATE_BINS_KI * kdBin)));
    }

    int chooseAction(int state) {
        // Implementacja logiki wyboru akcji dla Agenta 2
        if (rand() % 100 < epsilon * 100) {
            return rand() % NUM_ACTIONS;
        } else {
            int bestAction = 0;
            float bestQValue = qTable[state][0][1]; // Zakładamy, że Q-wartości dla Agenta 2 są w qTable[state][akcja][1]
            for (int a = 1; a < NUM_ACTIONS; a++) {
                if (qTable[state][a][1] > bestQValue) {
                    bestQValue = qTable[state][a][1];
                    bestAction = a;
                }
            }
            return bestAction;
        }
    }

    void executeAction(int action) {
        // Implementacja logiki wykonania akcji dla Agenta 2
        const int MAX_CURRENT = 25; // Maksymalny prąd wzbudzenia w amperach
        int current = analogRead(PIN_CURRENT_SENSOR); // Odczyt aktualnego prądu wzbudzenia z czujnika zewnętrznego

        switch (action) {
            case 0:
                if (current + PWM_INCREMENT <= MAX_CURRENT) {
                    analogWrite(PIN_EXCITATION_COIL_1, current + PWM_INCREMENT);
                }
                break;
            case 1:
                if (current + PWM_INCREMENT <= MAX_CURRENT) {
                    analogWrite(PIN_EXCITATION_COIL_2, current + PWM_INCREMENT);
                }
                break;
            // Dodaj inne przypadki akcji, jeśli są potrzebne
        }
    }

    float reward(float next_observation) {
        // Implementacja wspólnej funkcji nagrody
        const int TARGET_CURRENT = 25; // Docelowy prąd wzbudzenia w amperach
        float prad_wzbudzenia = next_observation;
        float nagroda = 0.0;

        // Nagroda za osiągnięcie docelowego prądu wzbudzenia
        if (prad_wzbudzenia >= TARGET_CURRENT) {
            nagroda = 100.0; // Duża nagroda za osiągnięcie celu
        } else {
            // Kara za mniejszy prąd wzbudzenia
            nagroda = -abs(TARGET_CURRENT - prad_wzbudzenia);
        }

        // Uwzględnij feedback od Agenta 3
        nagroda += feedbackFromAgent3;

        return nagroda;
    }
};



class Agent3 {
public:
    void odbierz_informacje_od_agentow(int akcja1, int akcja2) {
        // Implementacja logiki używającej akcji od Agentów 1 i 2
    }

    void updateQ(int state, int action, float reward, int nextState, float learningRate, float discountFactor) {
        // Przykładowa implementacja aktualizacji Q-wartości
        float bestNextQ = *std::max_element(qTable[nextState], qTable[nextState] + NUM_ACTIONS);
        qTable[state][action] = qTable[state][action] + learningRate * (reward + discountFactor * bestNextQ - qTable[state][action]);
    }

    void odbierz_informacje_hamowania(float hamowanie) {
        // Implementacja logiki używającej informacji o hamowaniu
    }

    void wyslij_informacje_do_agenta2(class Agent2& agent2, float feedback) {
        agent2.odbierz_informacje_od_agenta3(feedback);
    }

    void wyslij_informacje_do_agenta1(class Agent1& agent1, float feedback) {
        agent1.odbierz_informacje_od_agenta3(feedback);
    }

private:
    float qTable[NUM_STATES_AGENT3][NUM_ACTIONS_AGENT3] = {0}; // Tablica Q-wartości
};

int main() {
    Agent1 agent1;
    Agent2 agent2;
    Agent3 agent3;

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



using System;

class Program
{
    static void Main()
    {
        int totalEpochs = 100; // Całkowita liczba epok
        int completedEpochs = 0; // Ukończone epoki

        // Symulacja procesu nauki
        for (int epoch = 1; epoch <= totalEpochs; epoch++)
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

void setup() {
    Serial.begin(115200); // Inicjalizacja portu szeregowego

    // Inicjalizacja pinów i innych komponentów
    pinMode(muxSelectPinA, OUTPUT);
    pinMode(muxSelectPinB, OUTPUT);
    pinMode(PIN_EXCITATION_COIL_1, OUTPUT);
    pinMode(PIN_EXCITATION_COIL_2, OUTPUT);
    pinMode(mosfetPin, OUTPUT);
    pinMode(bjtPin1, OUTPUT);
    pinMode(bjtPin2, OUTPUT);
    pinMode(bjtPin3, OUTPUT);
    pinMode(excitationBJT1Pin, OUTPUT);
    pinMode(excitationBJT2Pin, OUTPUT);
    pinMode(PIN_CURRENT_SENSOR, INPUT);

    server.begin();
    display.begin(SH1106_SWITCHCAPVCC, 0x3C);
    display.clearDisplay();
    display.display();

    optimizer.initialize(3, bounds, 50, 10);

    char buffer[64];
    strcpy_P(buffer, welcomeMessage);
    display.println(buffer);
    display.display();

    // Inicjalizacja zmiennych globalnych
    externalVoltage = 0.0;
    externalCurrent = 0.0;
    efficiency = 0.0;
    efficiencyPercent = 0.0;
    voltageDrop = 0.0;

    // Inicjalizacja WiFi
    WiFi.begin("SSID", "PASSWORD");
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("Łączenie z WiFi...");
    }
    Serial.println("Połączono z WiFi");
    Serial.print("Adres IP: ");
    Serial.println(WiFi.localIP());
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
    display.setTextColor(WHITE, BLACK); // Ustawienie koloru tekstu na biały na czarnym tle
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
void loop() {
    handleSerialCommands();
    detectComputerConnection();

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

    // Odczyt danych z sensorów
    readSensors();

    // Wywołanie funkcji hillClimbing
    float currentThreshold = 0.5; // Przykładowa wartość początkowa
    float stepSize = 0.1; // Przykładowy rozmiar kroku
    float optimizedThreshold = hillClimbing(currentThreshold, stepSize, evaluate);

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

// Funkcja do automatycznego dostosowywania progu (umieszczona poza pętlą loop)
void adjustMinInputPower(float inputPower) {
    static float minObservedPower = 1e-6;
    static float maxObservedPower = 1e-3;

    // Aktualizuj minimalną i maksymalną obserwowaną moc
    if (inputPower > 0 && inputPower < minObservedPower) {
        minObservedPower = inputPower;
    }
    if (inputPower > maxObservedPower) {
       
    }

    // Dostosuj próg na podstawie obserwowanych wartości
    minInputPower = minObservedPower * 0.1; // Możesz dostosować współczynnik 0.1
}

void detectComputerConnection() {
    if (Serial) {
        Serial.println("Komputer podłączony. Przenoszenie mocy obliczeniowej...");
        // Wyślij komendę do komputera, aby rozpocząć przenoszenie mocy obliczeniowej
        Serial.println("START_COMPUTE");
    }
}
