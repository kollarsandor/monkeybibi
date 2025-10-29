class APIClient {
  constructor(baseURL = '') {
    this.baseURL = baseURL;
  }

  async request(method, endpoint, data = null) {
    const options = {
      method: method,
      headers: {
        'Content-Type': 'application/json'
      }
    };

    if (data && (method === 'POST' || method === 'PUT')) {
      options.body = JSON.stringify(data);
    }

    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, options);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed [${method} ${endpoint}]:`, error);
      throw error;
    }
  }

  async predict(sequenceData) {
    const payload = {
      protein_sequence: sequenceData.proteinSequence,
      dna_sequence: sequenceData.dnaSequence || null,
      rna_sequence: sequenceData.rnaSequence || null,
      num_recycles: sequenceData.numRecycles || 10,
      num_diffusion_samples: sequenceData.numDiffusionSamples || 5,
      seed: sequenceData.seed || null,
      max_template_date: sequenceData.maxTemplateDate || null,
      quantum_backend: sequenceData.quantumBackend || 'simulator',
      quantum_optimization_level: sequenceData.quantumOptLevel || 3,
      quantum_shots: sequenceData.quantumShots || 1024,
      quantum_error_mitigation: sequenceData.quantumErrorMitigation !== false,
      conformer_samples: sequenceData.conformerSamples || 2000,
      energy_threshold: sequenceData.energyThreshold || 10.0,
      ligands: sequenceData.ligands || []
    };

    return await this.request('POST', '/api/predict', payload);
  }

  async batchPredict(predictions) {
    return await this.request('POST', '/api/predict/batch', { predictions });
  }

  async getJobStatus(jobId) {
    return await this.request('GET', `/api/predict/${jobId}`);
  }

  async validateSequence(sequence, sequenceType = 'protein') {
    return await this.request('POST', '/api/sequences/validate', {
      sequence: sequence,
      type: sequenceType
    });
  }

  async analyzeSequence(sequence, sequenceType = 'protein') {
    return await this.request('POST', '/api/sequences/analyze', {
      sequence: sequence,
      type: sequenceType
    });
  }

  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${this.baseURL}/api/upload`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Upload failed! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('File upload failed:', error);
      throw error;
    }
  }

  async getHealth() {
    return await this.request('GET', '/api/health');
  }

  async getMetrics() {
    return await this.request('GET', '/api/metrics');
  }
}

if (typeof window !== 'undefined') {
  window.APIClient = APIClient;
}
