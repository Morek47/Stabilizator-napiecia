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

// Funkcja odczytu sensorów
void readSensors() {
    voltageIn[0] = analogRead(muxInputPin) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);

    // Odczyt zewnętrznego napięcia
    externalVoltage = analogRead(PIN_EXTERNAL_VOLTAGE_SENSOR_1) * (VOLTAGE_REFERENCE / ADC_MAX_VALUE);

    // Odczyt zewnętrznego prądu
    int raw_current_adc = analogRead(PIN_EXTERNAL_CURRENT_SENSOR_1);

    // Definicja referencji prądowej (dostosuj do swojego sprzętu)
    const float CURRENT_REFERENCE = 5.0; // Przykład: maksymalny prąd 5A

    // Konwersja wartości ADC na rzeczywisty prąd
    externalCurrent = raw_current_adc * (CURRENT_REFERENCE / ADC_MAX_VALUE);
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

// Define constants for Agent 3
const int NUM_STATES_AGENT3 = 10; // Liczba możliwych stanów dla agenta 3 (możesz dostosować)
const int NUM_ACTIONS_AGENT3 = 8; // Zwiększono liczbę akcji na 8
const float VOLTAGE_TOLERANCE = 0.5; // Tolerancja odchylenia napięcia (możesz dostosować)
const float MAX_GENERATOR_BRAKING = 1.0; // Maksymalne dozwolone hamowanie generatora

// Q-table for Agent 3
float qTableAgent3[NUM_STATES_AGENT3][NUM_ACTIONS_AGENT3]; // Tablica przechowująca wartości Q dla każdej pary stan-akcja

// Function to discretize state for Agent 3 - funkcja "dyskretyzuje" ciągłe wartości stanu na skończoną liczbę "koszyków" (binów)
int discretizeStateAgent3(float error, float generatorLoad) {
    // Normalizacja błędu i obciążenia generatora do zakresu [0, 1]
    float normalizedError = (error + MAX_ERROR) / (2 * MAX_ERROR);
    float normalizedLoad = (generatorLoad / MAX_LOAD);

    // Przypisanie znormalizowanych wartości do odpowiednich koszyków (binów)
    int errorBin = constrain((int)(normalizedError * NUM_STATE_BINS_ERROR), 0, NUM_STATE_BINS_ERROR - 1);
    int loadBin = constrain((int)(normalizedLoad * NUM_STATES_AGENT3), 0, NUM_STATES_AGENT3 - 1);

    // Obliczenie indeksu stanu na podstawie koszyków (binów)
    return errorBin * NUM_STATES_AGENT3 + loadBin;
}

// Function to choose action for Agent 3 - funkcja wybiera akcję na podstawie stanu, stosując strategię epsilon-greedy
int chooseActionAgent3(int state) {
    if (random(0, 100) < epsilon * 100) { // z prawdopodobieństwem epsilon wybieramy losową akcję (eksploracja)
        return random(0, NUM_ACTIONS_AGENT3);
    } else { // w przeciwnym wypadku wybieramy akcję z największą wartością Q dla danego stanu (eksploatacja)
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

// Function to execute action for Agent 3 - funkcja wykonuje akcję na podstawie jej numeru
void executeActionAgent3(int action) {
    switch (action) {
        case 0: // Zwiększenie prądu wzbudzenia na pierwszym tranzystorze
            analogWrite(excitationBJT1Pin, constrain(analogRead(excitationBJT1Pin) + PWM_INCREMENT, 0, 255));
            break;
        case 1: // Zmniejszenie prądu wzbudzenia na pierwszym tranzystorze
            analogWrite(excitationBJT1Pin, constrain(analogRead(excitationBJT1Pin) - PWM_INCREMENT, 0, 255));
            break;
        case 2: // Zwiększenie prądu wzbudzenia na drugim tranzystorze
            analogWrite(excitationBJT2Pin, constrain(analogRead(excitationBJT2Pin) + PWM_INCREMENT, 0, 255));
            break;
        case 3: // Zmniejszenie prądu wzbudzenia na drugim tranzystorze
            analogWrite(excitationBJT2Pin, constrain(analogRead(excitationBJT2Pin) - PWM_INCREMENT, 0, 255));
            break;
        case 4: // Agresywniejsze zwiększenie prądu wzbudzenia na pierwszym tranzystorze
            analogWrite(excitationBJT1Pin, constrain(analogRead(excitationBJT1Pin) + 2 * PWM_INCREMENT, 0, 255));
            break;
        case 5: // Agresywniejsze zmniejszenie prądu wzbudzenia na pierwszym tranzystorze
            analogWrite(excitationBJT1Pin, constrain(analogRead(excitationBJT1Pin) - 2 * PWM_INCREMENT, 0, 255));
            break;
        case 6: // Agresywniejsze zwiększenie prądu wzbudzenia na drugim tranzystorze
            analogWrite(excitationBJT2Pin, constrain(analogRead(excitationBJT2Pin) + 2 * PWM_INCREMENT, 0, 255));
            break;
        case 7: // Agresywniejsze zmniejszenie prądu wzbudzenia na drugim tranzystorze
            analogWrite(excitationBJT2Pin, constrain(analogRead(excitationBJT2Pin) - 2 * PWM_INCREMENT, 0, 255));
            break;
        default:
            break;
    }
}

// Function to calculate reward for Agent 3 - funkcja oblicza nagrodę na podstawie wydajności, napięcia, hamowania generatora i mocy wyjściowej
float calculateRewardAgent3(float efficiency, float voltage, float voltage_drop, float power_output) { // Zmieniono nazwę argumentu na voltage_drop
    const float MIN_EFFICIENCY = 0.8; // Minimalna akceptowalna wydajność
    const float MAX_VOLTAGE_DROP = 1.0; // Maksymalny akceptowalny spadek napięcia (dostosuj według potrzeb)

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
        reward -= (voltage_drop - MAX_VOLTAGE_DROP) * 20; // Dostosuj wagę kary według potrzeb
    }

    return reward;
}

// Function to update Q-value for Agent 3 - funkcja aktualizuje tablicę Q na podstawie stanu, akcji, nagrody i następnego stanu
void updateQAgent3(int state, int action, float reward, int nextState) {
    float maxQNextState = qTableAgent3[nextState][0];
    for (int a = 1; a < NUM_ACTIONS_AGENT3; a++) {
        if (qTableAgent3[nextState][a] > maxQNextState) {
            maxQNextState = qTableAgent3[nextState][a];
        }
    }

    qTableAgent3[state][action] += learningRate * (reward + discountFactor * maxQNextState - qTableAgent3[state][action]);
}

void loop() {
    // Handle serial commands
    handleSerialCommands();

    // Testowanie różnych wartości epsilon, learningRate i discountFactor
    float testEpsilon = 0.3;
    float testLearningRate = 0.01;
    float testDiscountFactor = 0.95;

    // Przypisz testowe wartości do używanych zmiennych
    epsilon = testEpsilon;
    learningRate = testLearningRate;
    discountFactor = testDiscountFactor;

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