#include <EEPROM.h>

// Constants used in the reward functions
const float VOLTAGE_SETPOINT = 230.0;  // Example setpoint, adjust as needed
const float MAX_POWER_OUTPUT = 1000.0;  // Example max power output, adjust as needed
const unsigned long OPTIMIZATION_INTERVAL = 10000;  // Example interval in milliseconds
const unsigned long TEST_DURATION = 5000;  // Example test duration in milliseconds

// Global variables
float epsilon, learningRate, discountFactor;
float efficiency, efficiencyPercent, voltageDrop, power_output;
unsigned long lastOptimizationTime = 0;
float params[3] = {0, 0, 0};
float bestEfficiency = 0;
float voltageIn[2], currentIn[2], externalVoltage, externalCurrent;
float LOAD_THRESHOLD;
byte qTable[256]; // Example size, adjust as needed
float Kp = 1.0, Ki = 1.0, Kd = 1.0;

class Agent1 {
public:
    int akcja;

    void odbierz_informacje_hamowania(float hamowanie) {
        // Implement logic using braking information
    }

    void wyslij_informacje_do_agenta3(class Agent3& agent3, class Agent2& agent2) {
        agent3.odbierz_informacje_od_agentow(akcja, agent2.akcja);
    }

    int discretizeState(float arg1, float arg2, float Kp, float Ki, float Kd) {
        // Implement state discretization logic for Agent 1
        return 0;
    }

    int chooseAction(int state) {
        // Implement action selection logic for Agent 1
        return 0;
    }

    void executeAction(int action) {
        // Implement action execution logic for Agent 1
    }

    float reward_agent1(float next_observation) {
        // Implement reward calculation for Agent 1
        float napiecie = next_observation;
        float spadek_napiecia = voltageDrop;

        float nagroda = 0;
        nagroda -= abs(napiecie - VOLTAGE_SETPOINT);
        nagroda -= spadek_napiecia;

        return nagroda;
    }
};

class Agent2 {
public:
    int akcja;

    void odbierz_informacje_hamowania(float hamowanie) {
        // Implement logic using braking information
    }

    void wyslij_informacje_do_agenta3(class Agent3& agent3, class Agent1& agent1) {
        agent3.odbierz_informacje_od_agentow(agent1.akcja, akcja);
    }

    int discretizeState(float arg1, float arg2, float Kp, float Ki, float Kd) {
        // Implement state discretization logic for Agent 2
        return 0;
    }

    int chooseAction(int state) {
        // Implement action selection logic for Agent 2
        return 0;
    }

    void executeAction(int action) {
        // Implement action execution logic for Agent 2
    }

    float reward_agent2(float next_observation) {
        // Implement reward calculation for Agent 2
        float prad_wzbudzenia = next_observation;

        if (prad_wzbudzenia > 25) {
            return -100;
        } else if (prad_wzbudzenia > 23) {
            return 100;
        } else {
            return prad_wzbudzenia;
        }
    }
};

class Agent3 {
public:
    void odbierz_informacje_od_agentow(int akcja1, int akcja2) {
        // Implement logic using actions from Agent 1 and Agent 2
    }

    int discretizeStateAgent3(float arg1, float arg2) {
        // Implement state discretization logic for Agent 3
        return 0;
    }

    int chooseActionAgent3(int state) {
        // Implement action selection logic for Agent 3
        return 0;
    }

    void executeActionAgent3(int action) {
        // Implement action execution logic for Agent 3
    }

    float calculateRewardAgent3(float efficiency, float voltageIn, float voltageDrop, float power_output) {
        // Implement reward calculation for Agent 3
        float nagroda = 0;
        float hamowanie = 0; // Placeholder, replace with actual value
        nagroda -= hamowanie * (1 + power_output / MAX_POWER_OUTPUT);
        nagroda += efficiency;

        return nagroda;
    }
};

Agent1 agent1;
Agent2 agent2;
Agent3 agent3;

void setup() {
    Serial.begin(9600);
    // Additional setup code here
}

void loop() {
    // Handle serial commands
    handleSerialCommands();

    // Testing different values of epsilon, learningRate and discountFactor
    static const float testEpsilon = 0.3;
    static const float testLearningRate = 0.01;
    static const float testDiscountFactor = 0.95;

    // Assign test values to the used variables once at startup
    static bool initialized = false;
    if (!initialized) {
        epsilon = testEpsilon;
        learningRate = testLearningRate;
        discountFactor = testDiscountFactor;
        initialized = true;
    }

    // Odczyt danych z sensorów
    readSensors();

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

    advancedLogData(efficiencyPercent);
    checkAlarm();
    autoCalibrate();
    energyManagement();

    // Q-learning 1 (voltage stabilizer)
    int state1 = agent1.discretizeState(VOLTAGE_SETPOINT - voltageIn[0], currentIn[0], Kp, Ki, Kd);
    int action1 = agent1.chooseAction(state1);
    agent1.executeAction(action1);
    float next_observation1 = ...; // Pobierz obserwację po wykonaniu akcji przez Agenta 1
    float reward1 = agent1.reward_agent1(next_observation1); 
    int nextState1 = agent1.discretizeState(VOLTAGE_SETPOINT - voltageIn[0], currentIn[0], Kp, Ki, Kd);
    updateQ(state1, action1, reward1, nextState1);

    // Q-learning 2 (excitation coils)
    int state2 = agent2.discretizeState(VOLTAGE_SETPOINT - voltageIn[0], currentIn[0], Kp, Ki, Kd);
    int action2 = agent2.chooseAction(state2);
    agent2.executeAction(action2);
    float next_observation2 = ...; // Pobierz obserwację po wykonaniu akcji przez Agenta 2
    float reward2 = agent2.reward_agent2(next_observation2);
    int nextState2 = agent2.discretizeState(VOLTAGE_SETPOINT - voltageIn[0], currentIn[0], Kp, Ki, Kd);
    updateQ(state2, action2, reward2, nextState2);

    // Q-learning 3 (generator braking)
    power_output = externalVoltage * externalCurrent;
    int state3 = agent3.discretizeStateAgent3(VOLTAGE_SETPOINT - voltageIn[0], currentIn[0]);
    int action3 = agent3.chooseActionAgent3(state3);
    agent3.executeActionAgent3(action3);

    float next_observation3 = ...; // Pobierz obserwację po wykonaniu akcji przez Agenta 3
    float hamowanie = next_observation3; // Pobierz wartość hamowania z obserwacji

    float reward3 = agent3.calculateRewardAgent3(efficiency, voltageIn[0], voltageDrop, power_output);
    int nextState3 = agent3.discretizeStateAgent3(VOLTAGE_SETPOINT - voltageIn[0], currentIn[0]);
    updateQAgent3(state3, action3, reward3, nextState3);

    // Communicate braking information from Agent 3 to Agent 1 and Agent 2
    agent1.odbierz_informacje_hamowania(hamowanie);
    agent2.odbierz_informacje_hamowania(hamowanie);

    // Adjust minimum input power threshold 
    float inputPower = voltageIn[0] * currentIn[0];
    adjustMinInputPower(inputPower);
}

// Placeholder definitions for required functions
void handleSerialCommands() {
    // Implement serial command handling logic
}

void readSensors() {
    // Implement sensor reading logic
}

float calculateEfficiency(float voltage, float current, float extVoltage, float extCurrent) {
    // Implement efficiency calculation logic
    return 0.0;
}

float hillClimbing(float threshold, float step, float (*evaluate)(float)) {
    // Implement hill climbing optimization logic
    return 0.0;
}

float evaluateThreshold(float threshold) {
    // Implement threshold evaluation logic
    return 0.0;
}

void displayData(float efficiencyPercent) {
    // Implement data display logic
}

void adjustControlFrequency() {
    // Implement control frequency adjustment logic
}

void monitorTransistors() {
    // Implement transistor monitoring logic
}

void monitorPerformanceAndAdjust() {
    // Implement performance monitoring and adjustment logic
}

void advancedLogData(float efficiencyPercent) {
    // Implement advanced logging logic
}

void checkAlarm() {
    // Implement alarm checking logic
}

void autoCalibrate() {
    // Implement auto-calibration logic
}

void energyManagement() {
    // Implement energy management logic
}

void updateQ(int state, int action, float reward, int nextState) {
    // Implement Q-learning update logic for Agent1 and Agent2
}

void updateQAgent3(int state, int action, float reward, int nextState) {
    // Implement Q-learning update logic for Agent3
}

void adjustMinInputPower(float inputPower) {
    // Implement minimum input power adjustment logic
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

    // Inicjalizacja zmiennych globalnych
    externalVoltage = 0.0;
    externalCurrent = 0.0;
    efficiency = 0.0;
    efficiencyPercent = 0.0;
    voltageDrop = 0.0;
}

void selectMuxChannel(int channel) {
  digitalWrite(muxSelectPinA, (channel >> 0) & 0x01);
  digitalWrite(muxSelectPinB, (channel >> 1) & 0x01);
  delay(1); // Opóźnienie dla ustabilizowania multipleksera
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

// Define constants for Agent 3
const int NUM_STATES_AGENT3 = 10; // Adjust as needed
const int NUM_ACTIONS_AGENT3 = 5; // Adjust as needed
const float VOLTAGE_TOLERANCE = 0.1; // Adjust as needed
const float MAX_GENERATOR_BRAKING = 1.0; // Adjust as needed

// Q-table for Agent 3
float qTableAgent3[NUM_STATES_AGENT3][NUM_ACTIONS_AGENT3];

// Function to discretize state for Agent 3
int discretizeStateAgent3(float error, float generatorLoad) {
    int errorBin = constrain((int)((error + MAX_ERROR) / (2 * MAX_ERROR) * NUM_STATES_AGENT3), 0, NUM_STATES_AGENT3 - 1);
    int loadBin = constrain((int)((generatorLoad / MAX_LOAD) * NUM_STATES_AGENT3), 0, NUM_STATES_AGENT3 - 1);
    return errorBin * NUM_STATES_AGENT3 + loadBin;
}

// Function to choose action for Agent 3
int chooseActionAgent3(int state) {
    if (random(0, 100) < epsilon * 100) {
        return random(0, NUM_ACTIONS_AGENT3);
    } else {
        int bestAction = 0;
        float bestQValue = qTableAgent3[state][0];
        for (int a = 1; a < NUM_ACTIONS_AGENT3; a++) {
            if (qTableAgent3[state][a] > bestQValue) {
                bestAction = a;
                bestQValue = qTableAgent3[state][a];
            }
        }
        return bestAction;
    }
}

// Function to execute action for Agent 3
void executeActionAgent3(int action) {
    switch (action) {
        case 0:
            analogWrite(excitationBJT1Pin, constrain(analogRead(excitationBJT1Pin) + PWM_INCREMENT, 0, 255));
            break;
        case 1:
            analogWrite(excitationBJT1Pin, constrain(analogRead(excitationBJT1Pin) - PWM_INCREMENT, 0, 255));
            break;
        case 2:
            analogWrite(excitationBJT2Pin, constrain(analogRead(excitationBJT2Pin) + PWM_INCREMENT, 0, 255));
            break;
        case 3:
            analogWrite(excitationBJT2Pin, constrain(analogRead(excitationBJT2Pin) - PWM_INCREMENT, 0, 255));
            break;
        case 4:
            // Add more actions if needed
            break;
        default:
            break;
    }
}

// Function to calculate reward for Agent 3
float calculateRewardAgent3(float efficiency, float voltage, float voltage_drop, float power_output) {
    const float MIN_EFFICIENCY = 0.8; // Minimalna akceptowalna wydajność
    const float MAX_VOLTAGE_DROP = 1.0; // Maksymalny akceptowalny spadek napięcia

    float reward = power_output; // Nagroda bazowa to moc wyjściowa

    // Kara za duże wahania napięcia
    float voltage_deviation = abs(voltage - VOLTAGE_SETPOINT);
    if (voltage_deviation > VOLTAGE_TOLERANCE) {
        reward -= voltage_deviation - VOLTAGE_TOLERANCE; 
    }

    // Kara za spadek wydajności poniżej minimalnego progu
    if (efficiency < MIN_EFFICIENCY) {
        reward -= (MIN_EFFICIENCY - efficiency) * 10; 
    }

    // Kara za duży spadek napięcia (hamowanie generatora)
    if (voltage_drop > MAX_VOLTAGE_DROP) {
        reward -= (voltage_drop - MAX_VOLTAGE_DROP) * 20;
    }

    return reward;
}

// Function to update Q-value for Agent 3
void updateQAgent3(int state, int action, float reward, int nextState) {
    float maxQNextState = qTableAgent3[nextState][0];
    for (int a = 1; a < NUM_ACTIONS_AGENT3; a++) {
        if (qTableAgent3[nextState][a] > maxQNextState) {
            maxQNextState = qTableAgent3[nextState][a];
        }
    }

    qTableAgent3[state][action] += learningRate * (reward + discountFactor * maxQNextState - qTableAgent3[state][action]);
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

void loop() {
    // Handle serial commands
    handleSerialCommands();

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

    // Display free memory info (zakomentowane, jeśli funkcja freeMemory() nie jest zdefiniowana)
    // Serial.print("Free memory: ");
    // Serial.println(freeMemory());

    // Communication with the computer
    if (Serial.available() > 0) {
        efficiency = Serial.parseFloat();
        efficiencyPercent = efficiency * 100.0;
        voltageDrop = Serial.parseFloat();
    }

    advancedLogData(efficiencyPercent);
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