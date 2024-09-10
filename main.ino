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
    Serial.print(voltageIn[0]);
    Serial.print(",");
    Serial.print(currentIn[0]);
    Serial.print(",");
    Serial.print(externalVoltage);
    Serial.print(",");
    Serial.println(externalCurrent);

    // Wait for results from the computer
    while (Serial.available() == 0) {}

    // Read results from the computer
    efficiency = Serial.parseFloat();
    efficiencyPercent = efficiency * 100.0;
    voltageDrop = Serial.parseFloat();

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
}