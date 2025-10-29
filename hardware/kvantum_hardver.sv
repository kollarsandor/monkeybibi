module quantum_controller #(
    parameter NUM_QUBITS = 8,
    parameter PULSE_WIDTH = 20,
    parameter DAC_BITS = 16
)(
    input logic clk,
    input logic rst_n,
    input logic [31:0] gate_command,
    input logic gate_valid,
    output logic gate_ready,
    output logic [DAC_BITS-1:0] i_dac [NUM_QUBITS-1:0],
    output logic [DAC_BITS-1:0] q_dac [NUM_QUBITS-1:0],
    output logic [NUM_QUBITS-1:0] measurement_result,
    output logic measurement_valid
);

    typedef enum logic [3:0] {
        GATE_IDLE,
        GATE_HADAMARD,
        GATE_PAULI_X,
        GATE_PAULI_Y,
        GATE_PAULI_Z,
        GATE_CNOT,
        GATE_PHASE,
        GATE_RX,
        GATE_RY,
        GATE_RZ,
        GATE_MEASURE
    } gate_type_t;

    typedef struct packed {
        logic [3:0] gate_type;
        logic [7:0] qubit_target;
        logic [7:0] qubit_control;
        logic [15:0] angle;
    } gate_instruction_t;

    gate_instruction_t current_gate;
    logic [7:0] pulse_counter;
    logic [7:0] gate_delay_counter;

    logic [15:0] nco_phase [NUM_QUBITS-1:0];
    logic [15:0] nco_freq [NUM_QUBITS-1:0];
    logic [15:0] amplitude [NUM_QUBITS-1:0];

    logic [NUM_QUBITS-1:0] qubit_state_i;
    logic [NUM_QUBITS-1:0] qubit_state_q;

    typedef enum logic [2:0] {
        FSM_IDLE,
        FSM_LOAD_GATE,
        FSM_GENERATE_PULSE,
        FSM_WAIT_DELAY,
        FSM_READOUT,
        FSM_COMPLETE
    } fsm_state_t;

    fsm_state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= FSM_IDLE;
        end else begin
            state <= next_state;
        end
    end

    always_comb begin
        next_state = state;
        case (state)
            FSM_IDLE: begin
                if (gate_valid) next_state = FSM_LOAD_GATE;
            end
            FSM_LOAD_GATE: begin
                next_state = FSM_GENERATE_PULSE;
            end
            FSM_GENERATE_PULSE: begin
                if (pulse_counter >= PULSE_WIDTH) begin
                    if (current_gate.gate_type == GATE_MEASURE) begin
                        next_state = FSM_READOUT;
                    end else begin
                        next_state = FSM_WAIT_DELAY;
                    end
                end
            end
            FSM_WAIT_DELAY: begin
                if (gate_delay_counter >= 100) next_state = FSM_COMPLETE;
            end
            FSM_READOUT: begin
                if (gate_delay_counter >= 200) next_state = FSM_COMPLETE;
            end
            FSM_COMPLETE: begin
                next_state = FSM_IDLE;
            end
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_gate <= '0;
            pulse_counter <= '0;
            gate_delay_counter <= '0;
            gate_ready <= 1'b1;
        end else begin
            case (state)
                FSM_IDLE: begin
                    gate_ready <= 1'b1;
                    pulse_counter <= '0;
                    gate_delay_counter <= '0;
                end

                FSM_LOAD_GATE: begin
                    current_gate <= gate_command;
                    gate_ready <= 1'b0;
                end

                FSM_GENERATE_PULSE: begin
                    pulse_counter <= pulse_counter + 1;
                end

                FSM_WAIT_DELAY: begin
                    gate_delay_counter <= gate_delay_counter + 1;
                end

                FSM_READOUT: begin
                    gate_delay_counter <= gate_delay_counter + 1;
                end

                FSM_COMPLETE: begin
                    gate_ready <= 1'b1;
                end
            endcase
        end
    end

    genvar i;
    generate
        for (i = 0; i < NUM_QUBITS; i++) begin : nco_gen
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    nco_phase[i] <= '0;
                    nco_freq[i] <= 16'd5000 + (i * 16'd100);
                    amplitude[i] <= '0;
                end else begin
                    nco_phase[i] <= nco_phase[i] + nco_freq[i];

                    if (state == FSM_GENERATE_PULSE && current_gate.qubit_target == i) begin
                        case (current_gate.gate_type)
                            GATE_HADAMARD: amplitude[i] <= 16'h7FFF;
                            GATE_PAULI_X: amplitude[i] <= 16'hFFFF;
                            GATE_PAULI_Y: amplitude[i] <= 16'hFFFF;
                            GATE_RX: amplitude[i] <= current_gate.angle;
                            GATE_RY: amplitude[i] <= current_gate.angle;
                            default: amplitude[i] <= '0;
                        endcase
                    end else begin
                        amplitude[i] <= '0;
                    end
                end
            end
        end
    endgenerate

    generate
        for (i = 0; i < NUM_QUBITS; i++) begin : dac_gen
            logic signed [31:0] sin_val, cos_val;

            cordic_sin_cos #(
                .WIDTH(16)
            ) cordic_inst (
                .clk(clk),
                .phase(nco_phase[i]),
                .sin_out(sin_val),
                .cos_out(cos_val)
            );

            always_ff @(posedge clk) begin
                i_dac[i] <= (cos_val * amplitude[i]) >>> 16;
                q_dac[i] <= (sin_val * amplitude[i]) >>> 16;
            end
        end
    endgenerate

    logic [15:0] readout_integrator_i [NUM_QUBITS-1:0];
    logic [15:0] readout_integrator_q [NUM_QUBITS-1:0];
    logic [7:0] readout_counter;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int j = 0; j < NUM_QUBITS; j++) begin
                readout_integrator_i[j] <= '0;
                readout_integrator_q[j] <= '0;
            end
            measurement_result <= '0;
            measurement_valid <= 1'b0;
            readout_counter <= '0;
        end else begin
            if (current_gate.gate_type == GATE_MEASURE && state == FSM_READOUT) begin
                if (readout_counter < 200) begin
                    for (int j = 0; j < NUM_QUBITS; j++) begin
                        readout_integrator_i[j] <= readout_integrator_i[j] + i_dac[j];
                        readout_integrator_q[j] <= readout_integrator_q[j] + q_dac[j];
                    end
                    readout_counter <= readout_counter + 1;
                    measurement_valid <= 1'b0;
                end else begin
                    for (int j = 0; j < NUM_QUBITS; j++) begin
                        logic [31:0] magnitude;
                        magnitude = (readout_integrator_i[j] * readout_integrator_i[j]) +
                                  (readout_integrator_q[j] * readout_integrator_q[j]);
                        measurement_result[j] <= (magnitude > 32'h10000000) ? 1'b1 : 1'b0;
                    end
                    measurement_valid <= 1'b1;
                end
            end else begin
                for (int j = 0; j < NUM_QUBITS; j++) begin
                    readout_integrator_i[j] <= '0;
                    readout_integrator_q[j] <= '0;
                end
                readout_counter <= '0;
                measurement_valid <= 1'b0;
            end
        end
    end

    logic [NUM_QUBITS-1:0] error_syndrome;
    logic error_detected;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            error_syndrome <= '0;
            error_detected <= 1'b0;
        end else begin
            for (int j = 0; j < NUM_QUBITS-2; j++) begin
                error_syndrome[j] <= measurement_result[j] ^ measurement_result[j+1] ^ measurement_result[j+2];
            end
            error_detected <= |error_syndrome;
        end
    end

endmodule

module cordic_sin_cos #(
    parameter WIDTH = 16
)(
    input logic clk,
    input logic [WIDTH-1:0] phase,
    output logic signed [WIDTH-1:0] sin_out,
    output logic signed [WIDTH-1:0] cos_out
);

    logic signed [WIDTH-1:0] x[15:0];
    logic signed [WIDTH-1:0] y[15:0];
    logic signed [WIDTH-1:0] z[15:0];

    logic [WIDTH-1:0] atan_lut[15:0];

    initial begin
        atan_lut[0] = 16'h2000;
        atan_lut[1] = 16'h12E4;
        atan_lut[2] = 16'h09FB;
        atan_lut[3] = 16'h0511;
        atan_lut[4] = 16'h028B;
        atan_lut[5] = 16'h0146;
        atan_lut[6] = 16'h00A3;
        atan_lut[7] = 16'h0051;
        atan_lut[8] = 16'h0029;
        atan_lut[9] = 16'h0014;
        atan_lut[10] = 16'h000A;
        atan_lut[11] = 16'h0005;
        atan_lut[12] = 16'h0003;
        atan_lut[13] = 16'h0001;
        atan_lut[14] = 16'h0001;
        atan_lut[15] = 16'h0000;
    end

    always_ff @(posedge clk) begin
        x[0] <= 16'h4DBA;
        y[0] <= 16'h0000;
        z[0] <= phase;

        for (int i = 0; i < 15; i++) begin
            if (z[i][WIDTH-1]) begin
                x[i+1] <= x[i] + (y[i] >>> i);
                y[i+1] <= y[i] - (x[i] >>> i);
                z[i+1] <= z[i] + atan_lut[i];
            end else begin
                x[i+1] <= x[i] - (y[i] >>> i);
                y[i+1] <= y[i] + (x[i] >>> i);
                z[i+1] <= z[i] - atan_lut[i];
            end
        end

        cos_out <= x[15];
        sin_out <= y[15];
    end

endmodule

module quantum_top #(
    parameter NUM_QUBITS = 8
)(
    input logic clk_100mhz,
    input logic rst_n,

    input logic [31:0] axi_awaddr,
    input logic axi_awvalid,
    output logic axi_awready,

    input logic [31:0] axi_wdata,
    input logic axi_wvalid,
    output logic axi_wready,

    output logic [15:0] dac_i [NUM_QUBITS-1:0],
    output logic [15:0] dac_q [NUM_QUBITS-1:0],

    input logic [15:0] adc_i [NUM_QUBITS-1:0],
    input logic [15:0] adc_q [NUM_QUBITS-1:0]
);

    logic [31:0] gate_command;
    logic gate_valid;
    logic gate_ready;
    logic [NUM_QUBITS-1:0] measurement_result;
    logic measurement_valid;

    quantum_controller #(
        .NUM_QUBITS(NUM_QUBITS)
    ) qc (
        .clk(clk_100mhz),
        .rst_n(rst_n),
        .gate_command(gate_command),
        .gate_valid(gate_valid),
        .gate_ready(gate_ready),
        .i_dac(dac_i),
        .q_dac(dac_q),
        .measurement_result(measurement_result),
        .measurement_valid(measurement_valid)
    );

    always_ff @(posedge clk_100mhz or negedge rst_n) begin
        if (!rst_n) begin
            axi_awready <= 1'b0;
            axi_wready <= 1'b0;
            gate_command <= '0;
            gate_valid <= 1'b0;
        end else begin
            if (axi_awvalid && axi_wvalid && gate_ready) begin
                gate_command <= axi_wdata;
                gate_valid <= 1'b1;
                axi_awready <= 1'b1;
                axi_wready <= 1'b1;
            end else begin
                gate_valid <= 1'b0;
                axi_awready <= 1'b0;
                axi_wready <= 1'b0;
            end
        end
    end

endmodule
