module fpga_quantum_testbench;

reg clk_250mhz;
reg clk_50mhz;
reg rst_n;

reg [31:0] input_data [0:15];
reg start_compute;

wire [13:0] qpu_adc_i [0:7];
wire [13:0] qpu_adc_q [0:7];
wire [15:0] qpu_dac_i [0:7];
wire [15:0] qpu_dac_q [0:7];

wire [31:0] output_data [0:15];
wire compute_complete;

HybridComputePlatform dut (
    .clk_250mhz(clk_250mhz),
    .clk_50mhz(clk_50mhz),
    .rst_n(rst_n),
    .input_data(input_data),
    .start_compute(start_compute),
    .qpu_adc_i(qpu_adc_i),
    .qpu_adc_q(qpu_adc_q),
    .qpu_dac_i(qpu_dac_i),
    .qpu_dac_q(qpu_dac_q),
    .output_data(output_data),
    .compute_complete(compute_complete)
);

initial begin
    clk_250mhz = 0;
    forever #2 clk_250mhz = ~clk_250mhz;
end

initial begin
    clk_50mhz = 0;
    forever #10 clk_50mhz = ~clk_50mhz;
end

integer i;
initial begin
    $dumpfile("fpga_quantum_sim.vcd");
    $dumpvars(0, fpga_quantum_testbench);
    
    rst_n = 0;
    start_compute = 0;
    
    for (i = 0; i < 16; i = i + 1) begin
        input_data[i] = 32'd1000 + i * 32'd100;
    end
    
    #50 rst_n = 1;
    
    #100 start_compute = 1;
    #20 start_compute = 0;
    
    wait(compute_complete);
    
    $display("=== FPGA-Quantum Platform Test Results ===");
    for (i = 0; i < 16; i = i + 1) begin
        $display("Output[%0d] = %0d", i, output_data[i]);
    end
    
    #1000 $finish;
end

genvar g;
generate
    for (g = 0; g < 8; g = g + 1) begin : adc_sim
        assign qpu_adc_i[g] = (qpu_dac_i[g] >> 2) + 14'd4096 + $random % 100;
        assign qpu_adc_q[g] = (qpu_dac_q[g] >> 2) + 14'd4096 + $random % 100;
    end
endgenerate

endmodule
