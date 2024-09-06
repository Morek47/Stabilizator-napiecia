();
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

