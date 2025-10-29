import spinal.core._
import spinal.lib._
import spinal.lib.fsm._

case class QuantumGateInstruction() extends Bundle {
  val gateType = UInt(4 bits)
  val qubitTarget = UInt(8 bits)
  val qubitControl = UInt(8 bits)
  val angle = UInt(16 bits)
}

case class QuantumControllerConfig(
  numQubits: Int = 8,
  pulseWidth: Int = 20,
  dacBits: Int = 16
)

class QuantumController(config: QuantumControllerConfig) extends Component {
  val io = new Bundle {
    val gateCommand = in(QuantumGateInstruction())
    val gateValid = in Bool()
    val gateReady = out Bool()
    
    val iDac = out Vec(SInt(config.dacBits bits), config.numQubits)
    val qDac = out Vec(SInt(config.dacBits bits), config.numQubits)
    
    val measurementResult = out Bits(config.numQubits bits)
    val measurementValid = out Bool()
  }

  val currentGate = Reg(QuantumGateInstruction())
  val pulseCounter = Reg(UInt(8 bits)) init(0)
  val gateDelayCounter = Reg(UInt(8 bits)) init(0)
  
  val ncoPhase = Vec(Reg(UInt(16 bits)) init(0), config.numQubits)
  val ncoFreq = Vec.fill(config.numQubits)(Reg(UInt(16 bits)))
  val amplitude = Vec(Reg(UInt(16 bits)) init(0), config.numQubits)

  for(i <- 0 until config.numQubits) {
    ncoFreq(i) init(5000 + i * 100)
  }

  val fsm = new StateMachine {
    val IDLE = new State with EntryPoint
    val LOAD_GATE = new State
    val GENERATE_PULSE = new State
    val WAIT_DELAY = new State
    val READOUT = new State
    val COMPLETE = new State

    IDLE.whenIsActive {
      io.gateReady := True
      when(io.gateValid) {
        goto(LOAD_GATE)
      }
    }

    LOAD_GATE.whenIsActive {
      currentGate := io.gateCommand
      io.gateReady := False
      pulseCounter := 0
      gateDelayCounter := 0
      goto(GENERATE_PULSE)
    }

    GENERATE_PULSE.whenIsActive {
      pulseCounter := pulseCounter + 1
      when(pulseCounter >= config.pulseWidth) {
        goto(WAIT_DELAY)
      }
    }

    WAIT_DELAY.whenIsActive {
      gateDelayCounter := gateDelayCounter + 1
      when(gateDelayCounter >= 100) {
        when(currentGate.gateType === 10) {
          goto(READOUT)
        } otherwise {
          goto(COMPLETE)
        }
      }
    }

    READOUT.whenIsActive {
      goto(COMPLETE)
    }

    COMPLETE.whenIsActive {
      goto(IDLE)
    }
  }

  for(i <- 0 until config.numQubits) {
    ncoPhase(i) := ncoPhase(i) + ncoFreq(i)
    
    when(fsm.isActive(fsm.GENERATE_PULSE) && currentGate.qubitTarget === i) {
      switch(currentGate.gateType) {
        is(1) { amplitude(i) := 0x7FFF }
        is(2) { amplitude(i) := 0xFFFF }
        is(3) { amplitude(i) := 0xFFFF }
        is(7) { amplitude(i) := currentGate.angle }
        is(8) { amplitude(i) := currentGate.angle }
        default { amplitude(i) := 0 }
      }
    } otherwise {
      amplitude(i) := 0
    }
  }

  for(i <- 0 until config.numQubits) {
    val cordic = new CordicSinCos(16)
    cordic.io.phase := ncoPhase(i)
    
    io.iDac(i) := (cordic.io.cos.asSInt * amplitude(i).asSInt) >> 16
    io.qDac(i) := (cordic.io.sin.asSInt * amplitude(i).asSInt) >> 16
  }

  val readoutIntegratorI = Vec(Reg(SInt(32 bits)) init(0), config.numQubits)
  val readoutIntegratorQ = Vec(Reg(SInt(32 bits)) init(0), config.numQubits)
  val readoutCounter = Reg(UInt(8 bits)) init(0)

  when(fsm.isActive(fsm.READOUT)) {
    when(readoutCounter < 200) {
      for(i <- 0 until config.numQubits) {
        readoutIntegratorI(i) := readoutIntegratorI(i) + io.iDac(i).resize(32 bits)
        readoutIntegratorQ(i) := readoutIntegratorQ(i) + io.qDac(i).resize(32 bits)
      }
      readoutCounter := readoutCounter + 1
      io.measurementValid := False
    } otherwise {
      for(i <- 0 until config.numQubits) {
        val magnitude = (readoutIntegratorI(i) * readoutIntegratorI(i)) + 
                       (readoutIntegratorQ(i) * readoutIntegratorQ(i))
        io.measurementResult(i) := magnitude > S(0x10000000, 64 bits)
      }
      io.measurementValid := True
    }
  } otherwise {
    for(i <- 0 until config.numQubits) {
      readoutIntegratorI(i) := 0
      readoutIntegratorQ(i) := 0
    }
    readoutCounter := 0
    io.measurementValid := False
  }

  io.gateReady := fsm.isActive(fsm.IDLE)
}

class CordicSinCos(width: Int) extends Component {
  val io = new Bundle {
    val phase = in UInt(width bits)
    val sin = out SInt(width bits)
    val cos = out SInt(width bits)
  }

  val atanLut = Vec(
    S(0x2000, width bits),
    S(0x12E4, width bits),
    S(0x09FB, width bits),
    S(0x0511, width bits),
    S(0x028B, width bits),
    S(0x0146, width bits),
    S(0x00A3, width bits),
    S(0x0051, width bits),
    S(0x0029, width bits),
    S(0x0014, width bits),
    S(0x000A, width bits),
    S(0x0005, width bits),
    S(0x0003, width bits),
    S(0x0001, width bits),
    S(0x0001, width bits),
    S(0x0000, width bits)
  )

  val x = Vec(Reg(SInt(width bits)), 16)
  val y = Vec(Reg(SInt(width bits)), 16)
  val z = Vec(Reg(SInt(width bits)), 16)

  x(0) := S(0x4DBA, width bits)
  y(0) := S(0, width bits)
  z(0) := io.phase.asSInt

  for(i <- 0 until 15) {
    when(z(i).msb) {
      x(i + 1) := x(i) + (y(i) >> i)
      y(i + 1) := y(i) - (x(i) >> i)
      z(i + 1) := z(i) + atanLut(i)
    } otherwise {
      x(i + 1) := x(i) - (y(i) >> i)
      y(i + 1) := y(i) + (x(i) >> i)
      z(i + 1) := z(i) - atanLut(i)
    }
  }

  io.cos := x(15)
  io.sin := y(15)
}

class QuantumNeuralAccelerator(numNeurons: Int = 256, numLayers: Int = 8) extends Component {
  val io = new Bundle {
    val inputData = in Vec(SInt(16 bits), numNeurons)
    val weights = in Vec(Vec(SInt(16 bits), numNeurons), numNeurons)
    val biases = in Vec(SInt(16 bits), numNeurons)
    
    val outputData = out Vec(SInt(16 bits), numNeurons)
    val valid = out Bool()
    val start = in Bool()
  }

  val neuronOutputs = Vec(Reg(SInt(32 bits)), numNeurons)
  val computeCounter = Reg(UInt(16 bits)) init(0)
  val layerCounter = Reg(UInt(4 bits)) init(0)

  val computing = Reg(Bool()) init(False)

  when(io.start && !computing) {
    computing := True
    computeCounter := 0
    layerCounter := 0
  }

  when(computing) {
    when(computeCounter < numNeurons) {
      for(i <- 0 until numNeurons) {
        val sum = SInt(32 bits)
        sum := neuronOutputs(i) + (io.inputData(computeCounter.resized) * io.weights(i)(computeCounter.resized))
        neuronOutputs(i) := sum
      }
      computeCounter := computeCounter + 1
    } otherwise {
      for(i <- 0 until numNeurons) {
        val activated = neuronOutputs(i) + io.biases(i).resize(32 bits)
        val relu = activated.asSInt
        neuronOutputs(i) := Mux(relu > 0, relu, S(0, 32 bits))
      }
      
      layerCounter := layerCounter + 1
      computeCounter := 0
      
      when(layerCounter >= numLayers - 1) {
        computing := False
      }
    }
  }

  for(i <- 0 until numNeurons) {
    io.outputData(i) := neuronOutputs(i).resize(16 bits)
  }

  io.valid := !computing
}

object QuantumHardwareVerilog {
  def main(args: Array[String]): Unit = {
    val config = QuantumControllerConfig(
      numQubits = 8,
      pulseWidth = 20,
      dacBits = 16
    )
    
    SpinalVerilog(new QuantumController(config))
    SpinalVerilog(new QuantumNeuralAccelerator(256, 8))
    
    println("JADED Quantum Hardware modules generated successfully!")
  }
}
