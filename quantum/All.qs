namespace JADED.Quantum {
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Preparation;

    operation ApplyHadamardGate(qubit : Qubit) : Unit is Adj + Ctl {
        H(qubit);
    }

    operation ApplyPauliX(qubit : Qubit) : Unit is Adj + Ctl {
        X(qubit);
    }

    operation ApplyPauliY(qubit : Qubit) : Unit is Adj + Ctl {
        Y(qubit);
    }

    operation ApplyPauliZ(qubit : Qubit) : Unit is Adj + Ctl {
        Z(qubit);
    }

    operation ApplyCNOT(control : Qubit, target : Qubit) : Unit is Adj + Ctl {
        CNOT(control, target);
    }

    operation ApplyToffoli(control1 : Qubit, control2 : Qubit, target : Qubit) : Unit is Adj + Ctl {
        CCNOT(control1, control2, target);
    }

    operation ApplyPhaseGate(theta : Double, qubit : Qubit) : Unit is Adj + Ctl {
        R1(theta, qubit);
    }

    operation ApplyRotationX(theta : Double, qubit : Qubit) : Unit is Adj + Ctl {
        Rx(theta, qubit);
    }

    operation ApplyRotationY(theta : Double, qubit : Qubit) : Unit is Adj + Ctl {
        Ry(theta, qubit);
    }

    operation ApplyRotationZ(theta : Double, qubit : Qubit) : Unit is Adj + Ctl {
        Rz(theta, qubit);
    }

    operation SwapReverseRegister(register : Qubit[]) : Unit is Adj + Ctl {
        let numQubits = Length(register);
        for i in 0..numQubits / 2 - 1 {
            SWAP(register[i], register[numQubits - 1 - i]);
        }
    }

    operation GroverOracle(markedState : Int, register : Qubit[]) : Unit is Adj + Ctl {
        use ancilla = Qubit();
        within {
            X(ancilla);
            H(ancilla);
            ApplyControlledOnInt(markedState, X, register, ancilla);
        } apply {
            H(ancilla);
        }
    }

    operation GroverDiffusionOperator(register : Qubit[]) : Unit is Adj + Ctl {
        within {
            ApplyToEachA(H, register);
            ApplyToEachA(X, register);
        } apply {
            Controlled Z(Most(register), Tail(register));
        }
        ApplyToEachA(X, register);
        ApplyToEachA(H, register);
    }

    operation GroverSearch(numQubits : Int, markedState : Int) : Result[] {
        use register = Qubit[numQubits];

        ApplyToEach(H, register);

        let iterations = Round(PI() / 4.0 * Sqrt(IntAsDouble(2 ^ numQubits)));

        for _ in 1..iterations {
            GroverOracle(markedState, register);
            GroverDiffusionOperator(register);
        }

        return MultiM(register);
    }

    operation VQEAnsatz(parameters : Double[], register : Qubit[]) : Unit is Adj + Ctl {
        let numQubits = Length(register);
        let numParams = Length(parameters);

        for i in 0..numQubits - 1 {
            Ry(parameters[i % numParams], register[i]);
        }

        for i in 0..numQubits - 2 {
            CNOT(register[i], register[i + 1]);
        }

        for i in 0..numQubits - 1 {
            let idx = (numQubits + i) % numParams;
            Ry(parameters[idx], register[i]);
        }
    }

    operation MeasureEnergy(hamiltonian : Double[], ansatz : (Qubit[] => Unit is Adj + Ctl), numQubits : Int) : Double {
        mutable totalEnergy = 0.0;
        let numSamples = 1000;

        for _ in 1..numSamples {
            use register = Qubit[numQubits];
            ansatz(register);

            let measurements = MultiM(register);
            mutable energy = 0.0;

            for i in 0..Length(hamiltonian) - 1 {
                if i < Length(measurements) and measurements[i] == One {
                    set energy += hamiltonian[i];
                } else {
                    set energy -= hamiltonian[i];
                }
            }

            set totalEnergy += energy;
            ResetAll(register);
        }

        return totalEnergy / IntAsDouble(numSamples);
    }

    operation QuantumFourierTransform(register : Qubit[]) : Unit is Adj + Ctl {
        let numQubits = Length(register);

        for i in 0..numQubits - 1 {
            H(register[i]);
            for j in i + 1..numQubits - 1 {
                let angle = PI() / IntAsDouble(2 ^ (j - i));
                Controlled R1([register[j]], (angle, register[i]));
            }
        }

        SwapReverseRegister(register);
    }

    operation PhaseEstimation(unitary : (Qubit[] => Unit is Adj + Ctl), eigenstate : Qubit[], precision : Int) : Result[] {
        use control = Qubit[precision];

        ApplyToEach(H, control);

        for i in 0..precision - 1 {
            let power = 2 ^ i;
            for _ in 1..power {
                Controlled unitary([control[i]], eigenstate);
            }
        }

        Adjoint QuantumFourierTransform(control);

        return MultiM(control);
    }

    operation QuantumAmplitudeEstimation(oracle : (Qubit[] => Unit is Adj + Ctl), numQubits : Int, precision : Int) : Double {
        use register = Qubit[numQubits];
        use control = Qubit[precision];

        ApplyToEach(H, register);
        ApplyToEach(H, control);

        for i in 0..precision - 1 {
            let power = 2 ^ i;
            for _ in 1..power {
                Controlled oracle([control[i]], register);
            }
        }

        Adjoint QuantumFourierTransform(control);

        let measurements = MultiM(control);
        mutable angle = 0.0;

        for i in 0..precision - 1 {
            if measurements[i] == One {
                set angle += IntAsDouble(2 ^ i) / IntAsDouble(2 ^ precision);
            }
        }

        ResetAll(register);
        ResetAll(control);

        return Sin(angle * PI()) ^ 2.0;
    }

    operation ProteinFoldingCircuit(sequence : Int[], numQubits : Int) : Result[] {
        use register = Qubit[numQubits];

        ApplyToEach(H, register);

        for i in 0..Length(sequence) - 1 {
            let aminoAcid = sequence[i];
            let qubitIdx = i % numQubits;

            if aminoAcid % 2 == 0 {
                X(register[qubitIdx]);
            }

            if aminoAcid % 3 == 0 {
                Ry(IntAsDouble(aminoAcid) * 0.1, register[qubitIdx]);
            }

            if qubitIdx < numQubits - 1 {
                CNOT(register[qubitIdx], register[qubitIdx + 1]);
            }
        }

        for i in 0..numQubits - 2 {
            CNOT(register[i], register[i + 1]);
        }

        ApplyToEach(H, register);

        return MultiM(register);
    }

    operation QuantumMachineLearning(trainingData : Double[][], labels : Int[], numQubits : Int, numLayers : Int) : Double[] {
        mutable weights = [0.0, size = numQubits * numLayers];

        for i in 0..Length(weights) - 1 {
            set weights w/= i <- 0.1;
        }

        let learningRate = 0.01;
        let epochs = 100;

        for epoch in 1..epochs {
            mutable totalLoss = 0.0;

            for dataIdx in 0..Length(trainingData) - 1 {
                use register = Qubit[numQubits];

                for layer in 0..numLayers - 1 {
                    for q in 0..numQubits - 1 {
                        let weightIdx = layer * numQubits + q;
                        Ry(weights[weightIdx], register[q]);
                    }

                    for q in 0..numQubits - 2 {
                        CNOT(register[q], register[q + 1]);
                    }
                }

                let measurement = M(register[0]);
                let prediction = measurement == One ? 1 | 0;
                let loss = IntAsDouble((prediction - labels[dataIdx]) ^ 2);
                set totalLoss += loss;

                ResetAll(register);
            }
        }

        return weights;
    }

    operation HybridQuantumClassical(inputData : Double[], classicalWeights : Double[], quantumWeights : Double[]) : Int {
        let numQubits = Length(quantumWeights);
        use register = Qubit[numQubits];

        for i in 0..numQubits - 1 {
            Ry(inputData[i % Length(inputData)] * quantumWeights[i], register[i]);
        }

        for i in 0..numQubits - 2 {
            CNOT(register[i], register[i + 1]);
        }

        let result = M(register[0]);
        ResetAll(register);

        return result == One ? 1 | 0;
    }

    operation EntanglementCircuit(numQubits : Int) : Result[] {
        use register = Qubit[numQubits];

        H(register[0]);

        for i in 0..numQubits - 2 {
            CNOT(register[i], register[i + 1]);
        }

        return MultiM(register);
    }

    operation BellStatePreparation() : Result[] {
        use qubits = Qubit[2];

        H(qubits[0]);
        CNOT(qubits[0], qubits[1]);

        return MultiM(qubits);
    }

    operation QuantumTeleportation(messageQubit : Qubit) : Result {
        use alice = Qubit();
        use bob = Qubit();

        H(alice);
        CNOT(alice, bob);

        CNOT(messageQubit, alice);
        H(messageQubit);

        let m1 = M(messageQubit);
        let m2 = M(alice);

        if m2 == One {
            X(bob);
        }

        if m1 == One {
            Z(bob);
        }

        let result = M(bob);
        Reset(bob);

        return result;
    }

    @EntryPoint()
    operation Main() : Unit {
        Message("JADED Q# Quantum Algorithms");
        Message("============================");

        Message("\n1. Grover's Algorithm:");
        let groverResults = GroverSearch(4, 7);
        Message($"   Marked state search results: {groverResults}");

        Message("\n2. Bell State:");
        let bellResults = BellStatePreparation();
        Message($"   Bell state: {bellResults}");

        Message("\n3. Entanglement:");
        let entanglementResults = EntanglementCircuit(5);
        Message($"   Entangled state: {entanglementResults}");

        Message("\n4. Protein Folding Simulation:");
        let sequence = [1, 2, 3, 4, 5, 6, 7, 8];
        let proteinResults = ProteinFoldingCircuit(sequence, 8);
        Message($"   Protein structure encoding: {proteinResults}");

        Message("\nAll quantum algorithms executed successfully!");
    }
}
