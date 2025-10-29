class AlphaFold3App {
  constructor() {
    this.api = new APIClient();
    this.ws = new WebSocketClient();
    this.currentJobId = null;
    this.currentPage = 'input';
    this.uploadedFiles = [];
    this.charts = {};
    this.viewer3D = null;

    this.initializeDOM();
    this.attachEventListeners();
    this.connectWebSocket();
  }

  initializeDOM() {
    this.elements = {
      sidebar: document.getElementById('sidebar'),
      sidebarToggle: document.getElementById('sidebarToggle'),
      navItems: document.querySelectorAll('.nav-item'),
      pageContents: document.querySelectorAll('.page-content'),
      pageTitle: document.getElementById('pageTitle'),
      breadcrumbCurrent: document.getElementById('breadcrumbCurrent'),

      proteinSequence: document.getElementById('proteinSequence'),
      dnaSequence: document.getElementById('dnaSequence'),
      rnaSequence: document.getElementById('rnaSequence'),
      proteinLength: document.getElementById('proteinLength'),
      proteinChains: document.getElementById('proteinChains'),
      proteinValid: document.getElementById('proteinValid'),
      proteinError: document.getElementById('proteinError'),
      dnaLength: document.getElementById('dnaLength'),
      gcContent: document.getElementById('gcContent'),
      rnaLength: document.getElementById('rnaLength'),
      auContent: document.getElementById('auContent'),

      ligandUploadZone: document.getElementById('ligandUploadZone'),
      ligandFileInput: document.getElementById('ligandFileInput'),
      ligandFileList: document.getElementById('ligandFileList'),

      validateSequencesBtn: document.getElementById('validateSequencesBtn'),
      proceedToParamsBtn: document.getElementById('proceedToParamsBtn'),
      backToInputBtn: document.getElementById('backToInputBtn'),
      startComputationBtn: document.getElementById('startComputationBtn'),
      abortComputationBtn: document.getElementById('abortComputationBtn'),
      loadExampleBtn: document.getElementById('loadExampleBtn'),
      clearSequencesBtn: document.getElementById('clearSequencesBtn'),

      numRecycles: document.getElementById('numRecycles'),
      numRecyclesValue: document.getElementById('numRecyclesValue'),
      numDiffusionSamples: document.getElementById('numDiffusionSamples'),
      numDiffusionSamplesValue: document.getElementById('numDiffusionSamplesValue'),
      quantumBackend: document.getElementById('quantumBackend'),
      quantumOptLevel: document.getElementById('quantumOptLevel'),
      quantumOptLevelValue: document.getElementById('quantumOptLevelValue'),
      quantumShots: document.getElementById('quantumShots'),
      quantumShotsValue: document.getElementById('quantumShotsValue'),
      conformerSamples: document.getElementById('conformerSamples'),
      conformerSamplesValue: document.getElementById('conformerSamplesValue'),

      systemLogs: document.getElementById('systemLogs'),
      overallProgressBar: document.getElementById('overallProgressBar'),
      overallProgress: document.getElementById('overallProgress'),
      sequenceProgressBar: document.getElementById('sequenceProgressBar'),
      sequenceProgress: document.getElementById('sequenceProgress'),
      msaProgressBar: document.getElementById('msaProgressBar'),
      msaProgress: document.getElementById('msaProgress'),
      quantumProgressBar: document.getElementById('quantumProgressBar'),
      quantumProgress: document.getElementById('quantumProgress'),
      inferenceProgressBar: document.getElementById('inferenceProgressBar'),
      inferenceProgress: document.getElementById('inferenceProgress'),
      postProcessProgressBar: document.getElementById('postProcessProgressBar'),
      postProcessProgress: document.getElementById('postProcessProgress'),

      mainWorkerStatus: document.getElementById('mainWorkerStatus'),
      msaWorkerStatus: document.getElementById('msaWorkerStatus'),
      quantumWorkerStatus: document.getElementById('quantumWorkerStatus'),
      postWorkerStatus: document.getElementById('postWorkerStatus'),

      viewer3D: document.getElementById('viewer3D'),
      confidenceChart: document.getElementById('confidenceChart'),
      paeHeatmap: document.getElementById('paeHeatmap'),
      distogramChart: document.getElementById('distogramChart'),

      statAvgPlddt: document.getElementById('statAvgPlddt'),
      statAvgPae: document.getElementById('statAvgPae'),
      statTmScore: document.getElementById('statTmScore'),
      statTotalEnergy: document.getElementById('statTotalEnergy'),
      statComputeTime: document.getElementById('statComputeTime'),
      statSeqLength: document.getElementById('statSeqLength')
    };
  }

  attachEventListeners() {
    this.elements.sidebarToggle?.addEventListener('click', () => this.toggleSidebar());

    this.elements.navItems.forEach(item => {
      item.addEventListener('click', () => this.switchPage(item.dataset.page));
    });

    this.elements.proteinSequence?.addEventListener('input', () => this.updateSequenceStats());
    this.elements.dnaSequence?.addEventListener('input', () => this.updateSequenceStats());
    this.elements.rnaSequence?.addEventListener('input', () => this.updateSequenceStats());

    this.elements.ligandUploadZone?.addEventListener('click', () => this.elements.ligandFileInput.click());
    this.elements.ligandUploadZone?.addEventListener('dragover', (e) => this.handleDragOver(e));
    this.elements.ligandUploadZone?.addEventListener('dragleave', () => this.handleDragLeave());
    this.elements.ligandUploadZone?.addEventListener('drop', (e) => this.handleDrop(e));
    this.elements.ligandFileInput?.addEventListener('change', (e) => this.handleFileSelect(e));

    this.elements.validateSequencesBtn?.addEventListener('click', () => this.validateSequences());
    this.elements.proceedToParamsBtn?.addEventListener('click', () => this.proceedToParams());
    this.elements.backToInputBtn?.addEventListener('click', () => this.switchPage('input'));
    this.elements.startComputationBtn?.addEventListener('click', () => this.startComputation());
    this.elements.abortComputationBtn?.addEventListener('click', () => this.abortComputation());
    this.elements.loadExampleBtn?.addEventListener('click', () => this.loadExample());
    this.elements.clearSequencesBtn?.addEventListener('click', () => this.clearSequences());

    this.elements.numRecycles?.addEventListener('input', (e) => {
      this.elements.numRecyclesValue.textContent = e.target.value;
    });
    this.elements.numDiffusionSamples?.addEventListener('input', (e) => {
      this.elements.numDiffusionSamplesValue.textContent = e.target.value;
    });
    this.elements.quantumOptLevel?.addEventListener('input', (e) => {
      this.elements.quantumOptLevelValue.textContent = e.target.value;
    });
    this.elements.quantumShots?.addEventListener('input', (e) => {
      this.elements.quantumShotsValue.textContent = e.target.value;
    });
    this.elements.conformerSamples?.addEventListener('input', (e) => {
      this.elements.conformerSamplesValue.textContent = e.target.value;
    });

    document.querySelectorAll('.collapsible-header').forEach(header => {
      header.addEventListener('click', () => {
        header.parentElement.classList.toggle('open');
      });
    });
  }

  async connectWebSocket() {
    try {
      this.ws.on('connected', () => this.addLog('INFO', 'WebSocket kapcsolat létrejött'));
      this.ws.on('disconnected', () => this.addLog('WARN', 'WebSocket kapcsolat megszakadt'));
      this.ws.on('jobCompleted', (data) => this.handleJobCompleted(data));
      this.ws.on('jobFailed', (data) => this.handleJobFailed(data));
      this.ws.on('jobProgress', (data) => this.handleJobProgress(data));
      this.ws.on('error', (error) => this.addLog('ERROR', `WebSocket hiba: ${error.message}`));

      await this.ws.connect();
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      this.addLog('ERROR', 'WebSocket csatlakozás sikertelen');
    }
  }

  toggleSidebar() {
    this.elements.sidebar?.classList.toggle('open');
  }

  switchPage(pageId) {
    this.elements.pageContents.forEach(page => page.classList.remove('active'));
    document.getElementById(`page-${pageId}`)?.classList.add('active');

    this.elements.navItems.forEach(item => {
      item.classList.remove('active');
      if (item.dataset.page === pageId) {
        item.classList.add('active');
      }
    });

    const pageTitles = {
      input: 'Szekvencia Bemenet',
      parameters: 'Paraméterek',
      monitoring: 'Monitorozás',
      results: 'Eredmények',
      quantum: 'Kvantum Konfiguráció',
      analysis: 'Strukturális Analízis',
      verification: 'Formális Verifikáció',
      history: 'Előzmények',
      settings: 'Rendszer Beállítások',
      help: 'Súgó & Dokumentáció'
    };

    if (this.elements.pageTitle) this.elements.pageTitle.textContent = pageTitles[pageId];
    if (this.elements.breadcrumbCurrent) this.elements.breadcrumbCurrent.textContent = pageTitles[pageId];
    this.currentPage = pageId;
  }

  updateSequenceStats() {
    const proteinSeq = this.elements.proteinSequence?.value || '';
    const dnaSeq = this.elements.dnaSequence?.value || '';
    const rnaSeq = this.elements.rnaSequence?.value || '';

    const proteinResult = this.validateProteinSequence(proteinSeq);
    if (this.elements.proteinLength) this.elements.proteinLength.textContent = proteinResult.length;
    if (this.elements.proteinChains) this.elements.proteinChains.textContent = proteinResult.chains;
    if (this.elements.proteinValid) {
      this.elements.proteinValid.textContent = proteinResult.valid ? 'Érvényes' : 'Érvénytelen';
      this.elements.proteinValid.style.color = proteinResult.valid ? 'var(--success)' : 'var(--error)';
    }
    if (this.elements.proteinError) {
      if (proteinResult.valid) {
        this.elements.proteinError.classList.add('hidden');
      } else {
        this.elements.proteinError.querySelector('span').textContent = proteinResult.error;
        this.elements.proteinError.classList.remove('hidden');
      }
    }

    const dnaResult = this.validateDNASequence(dnaSeq);
    if (this.elements.dnaLength) this.elements.dnaLength.textContent = dnaResult.length;
    if (this.elements.gcContent) this.elements.gcContent.textContent = dnaResult.gcContent + '%';

    const rnaResult = this.validateRNASequence(rnaSeq);
    if (this.elements.rnaLength) this.elements.rnaLength.textContent = rnaResult.length;
    if (this.elements.auContent) this.elements.auContent.textContent = rnaResult.auContent + '%';
  }

  validateProteinSequence(sequence) {
    if (!sequence) return { valid: false, error: 'A fehérje szekvencia megadása kötelező', length: 0, chains: 0 };

    const lines = sequence.trim().split('\n');
    if (lines.length < 2) return { valid: false, error: 'Érvénytelen FASTA formátum', length: 0, chains: 0 };
    if (!lines[0].startsWith('>')) return { valid: false, error: 'A fehérje szekvencia címkéje hiányzik', length: 0, chains: 0 };

    const seq = lines.slice(1).join('').toUpperCase();
    const validChars = /^[ACDEFGHIKLMNPQRSTVWY]+$/;
    if (!validChars.test(seq)) return { valid: false, error: 'A fehérje szekvencia érvénytelen karaktereket tartalmaz', length: 0, chains: 0 };

    return { valid: true, length: seq.length, chains: 1 };
  }

  validateDNASequence(sequence) {
    if (!sequence) return { valid: true, length: 0, gcContent: 0 };

    const lines = sequence.trim().split('\n');
    if (lines.length < 2) return { valid: false, error: 'Érvénytelen FASTA formátum', length: 0, gcContent: 0 };
    if (!lines[0].startsWith('>')) return { valid: false, error: 'A DNS szekvencia címkéje hiányzik', length: 0, gcContent: 0 };

    const seq = lines.slice(1).join('').toUpperCase();
    const validChars = /^[ATGC]+$/;
    if (!validChars.test(seq)) return { valid: false, error: 'A DNS szekvencia érvénytelen karaktereket tartalmaz', length: 0, gcContent: 0 };

    const gcCount = (seq.match(/[GC]/g) || []).length;
    const gcPercentage = seq.length > 0 ? ((gcCount / seq.length) * 100).toFixed(1) : 0;

    return { valid: true, length: seq.length, gcContent: gcPercentage };
  }

  validateRNASequence(sequence) {
    if (!sequence) return { valid: true, length: 0, auContent: 0 };

    const lines = sequence.trim().split('\n');
    if (lines.length < 2) return { valid: false, error: 'Érvénytelen FASTA formátum', length: 0, auContent: 0 };
    if (!lines[0].startsWith('>')) return { valid: false, error: 'A RNS szekvencia címkéje hiányzik', length: 0, auContent: 0 };

    const seq = lines.slice(1).join('').toUpperCase();
    const validChars = /^[AUGC]+$/;
    if (!validChars.test(seq)) return { valid: false, error: 'A RNS szekvencia érvénytelen karaktereket tartalmaz', length: 0, auContent: 0 };

    const auCount = (seq.match(/[AU]/g) || []).length;
    const auPercentage = seq.length > 0 ? ((auCount / seq.length) * 100).toFixed(1) : 0;

    return { valid: true, length: seq.length, auContent: auPercentage };
  }

  handleDragOver(e) {
    e.preventDefault();
    this.elements.ligandUploadZone?.classList.add('dragover');
  }

  handleDragLeave() {
    this.elements.ligandUploadZone?.classList.remove('dragover');
  }

  handleDrop(e) {
    e.preventDefault();
    this.elements.ligandUploadZone?.classList.remove('dragover');
    this.handleFiles(e.dataTransfer.files);
  }

  handleFileSelect(e) {
    this.handleFiles(e.target.files);
  }

  handleFiles(files) {
    Array.from(files).forEach(file => {
      if (!this.uploadedFiles.find(f => f.name === file.name && f.size === file.size)) {
        this.uploadedFiles.push(file);
        this.addFileToList(file);
      }
    });
  }

  addFileToList(file) {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    fileItem.innerHTML = `
      <div class="file-info">
        <div class="file-icon"><i class="fas fa-file"></i></div>
        <div class="file-details">
          <div class="file-name">${file.name}</div>
          <div class="file-size">${this.formatFileSize(file.size)}</div>
        </div>
      </div>
      <div class="file-actions">
        <button class="btn btn-icon btn-secondary" onclick="app.removeFile('${file.name}', ${file.size})">
          <i class="fas fa-trash"></i>
        </button>
      </div>
    `;
    this.elements.ligandFileList?.appendChild(fileItem);
  }

  removeFile(name, size) {
    this.uploadedFiles = this.uploadedFiles.filter(file => !(file.name === name && file.size === size));
    this.elements.ligandFileList.innerHTML = '';
    this.uploadedFiles.forEach(file => this.addFileToList(file));
  }

  formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  async validateSequences() {
    const proteinSeq = this.elements.proteinSequence?.value || '';
    const proteinResult = this.validateProteinSequence(proteinSeq);

    if (!proteinResult.valid) {
      this.addLog('ERROR', proteinResult.error);
      return;
    }

    try {
      this.addLog('INFO', 'Szekvencia validálás folyamatban...');
      const result = await this.api.validateSequence(proteinSeq, 'protein');
      this.addLog('SUCCESS', 'Szekvencia validálás sikeres');
      console.log('Validation result:', result);
    } catch (error) {
      this.addLog('ERROR', `Validálás sikertelen: ${error.message}`);
    }
  }

  proceedToParams() {
    const proteinResult = this.validateProteinSequence(this.elements.proteinSequence?.value || '');
    if (!proteinResult.valid) {
      this.addLog('ERROR', 'Érvényes fehérje szekvencia szükséges a folytatáshoz');
      return;
    }
    this.switchPage('parameters');
  }

  async startComputation() {
    if (this.currentJobId) {
      this.addLog('WARN', 'Már fut egy számítás!');
      return;
    }

    const proteinSeq = this.elements.proteinSequence?.value || '';
    const dnaSeq = this.elements.dnaSequence?.value || '';
    const rnaSeq = this.elements.rnaSequence?.value || '';
    const sequenceData = {
      proteinSequence: proteinSeq,
      dnaSequence: dnaSeq || null,
      rnaSequence: rnaSeq || null,
      numRecycles: parseInt(this.elements.numRecycles?.value || 10),
      numDiffusionSamples: parseInt(this.elements.numDiffusionSamples?.value || 5),
      seed: this.elements.seedValue?.value || null,
      maxTemplateDate: this.elements.maxTemplateDate?.value || null,
      quantumBackend: this.elements.quantumBackend?.value || 'simulator',
      quantumOptLevel: parseInt(this.elements.quantumOptLevel?.value || 3),
      quantumShots: parseInt(this.elements.quantumShots?.value || 1024),
      quantumErrorMitigation: this.elements.quantumErrorMitigation?.checked !== false,
      conformerSamples: parseInt(this.elements.conformerSamples?.value || 2000),
      energyThreshold: parseFloat(this.elements.energyThreshold?.value || 10.0),
      ligands: []
    };

    try {
      this.addLog('INFO', 'Predikció indítása...');
      const response = await this.api.predict(sequenceData);

      this.currentJobId = response.job_id;
      this.addLog('SUCCESS', `Predikció elindítva (Job ID: ${this.currentJobId})`);

      await this.ws.joinJobChannel(this.currentJobId);
      this.addLog('INFO', 'WebSocket csatorna csatlakozva');

      this.switchPage('monitoring');
      this.elements.startComputationBtn.disabled = true;
      this.elements.abortComputationBtn.style.display = 'flex';

      this.pollJobStatus();
    } catch (error) {
      this.addLog('ERROR', `Predikció indítása sikertelen: ${error.message}`);
    }
  }

  async pollJobStatus() {
    if (!this.currentJobId) return;

    try {
      const status = await this.api.getJobStatus(this.currentJobId);
      this.updateJobStatus(status);

      if (status.status === 'processing' || status.status === 'queued') {
        setTimeout(() => this.pollJobStatus(), 2000);
      }
    } catch (error) {
      console.error('Failed to poll job status:', error);
    }
  }

  updateJobStatus(status) {
    console.log('Job status update:', status);

  }

  handleJobProgress(data) {
    console.log('Job progress update:', data);
    if (data.job_id !== this.currentJobId) return;

    const progress = data.progress || {};

    if (progress.stage === 'initializing') {
      this.addLog('INFO', 'Inicializálás...');
      this.updateProgress('overall', 10);
    } else if (progress.stage === 'computing') {
      this.addLog('INFO', 'Számítás folyamatban...');
      this.updateProgress('overall', 50);
      this.updateProgress('inference', 50);
    } else if (progress.stage === 'finalizing') {
      this.addLog('INFO', 'Befejezés...');
      this.updateProgress('overall', 90);
      this.updateProgress('postProcess', 90);
    }

    if (progress.progress) {
      this.updateProgress('overall', progress.progress);
    }
  }

  handleJobCompleted(data) {
    console.log('Job completed:', data);
    this.addLog('SUCCESS', 'Számítás sikeresen befejezve!');

    this.elements.startComputationBtn.disabled = false;
    this.elements.abortComputationBtn.style.display = 'none';

    if (data.result) {
      this.displayResults(data.result);
      this.switchPage('results');
    }
  }

  handleJobFailed(data) {
    console.error('Job failed:', data);
    this.addLog('ERROR', `Számítás sikertelen: ${data.error || 'Ismeretlen hiba'}`);

    this.elements.startComputationBtn.disabled = false;
    this.elements.abortComputationBtn.style.display = 'none';
  }

  async abortComputation() {
    this.addLog('WARN', 'Számítás megszakítva');
    this.ws.leaveJobChannel();
    this.currentJobId = null;
    this.elements.startComputationBtn.disabled = false;
    this.elements.abortComputationBtn.style.display = 'none';
    this.resetProgress();
  }

  updateProgressBars(progress) {
    const setProgress = (bar, text, value) => {
      if (bar) bar.style.width = value + '%';
      if (text) text.textContent = Math.round(value) + '%';
    };

    setProgress(this.elements.overallProgressBar, this.elements.overallProgress, progress.overall || 0);
    setProgress(this.elements.sequenceProgressBar, this.elements.sequenceProgress, progress.sequence || 0);
    setProgress(this.elements.msaProgressBar, this.elements.msaProgress, progress.msa || 0);
    setProgress(this.elements.quantumProgressBar, this.elements.quantumProgress, progress.quantum || 0);
    setProgress(this.elements.inferenceProgressBar, this.elements.inferenceProgress, progress.inference || 0);
    setProgress(this.elements.postProcessProgressBar, this.elements.postProcessProgress, progress.postProcess || 0);
  }

  resetProgress() {
    this.updateProgressBars({
      overall: 0,
      sequence: 0,
      msa: 0,
      quantum: 0,
      inference: 0,
      postProcess: 0
    });
  }

  displayResults(result) {
    console.log('Displaying results:', result);

    if (result.metrics) {
      if (this.elements.statAvgPlddt) this.elements.statAvgPlddt.textContent = result.metrics.avg_plddt?.toFixed(1) || '-';
      if (this.elements.statAvgPae) this.elements.statAvgPae.textContent = result.metrics.avg_pae?.toFixed(1) + ' Å' || '-';
      if (this.elements.statTmScore) this.elements.statTmScore.textContent = result.metrics.tm_score?.toFixed(2) || '-';
      if (this.elements.statTotalEnergy) this.elements.statTotalEnergy.textContent = result.metrics.total_energy?.toFixed(1) || '-';
    }

    if (result.structure && this.elements.viewer3D) {
      this.render3DStructure(result.structure);
    }

    if (result.confidence && this.elements.confidenceChart) {
      this.renderConfidenceChart(result.confidence);
    }

    if (result.pae && this.elements.paeHeatmap) {
      this.renderPAEHeatmap(result.pae);
    }
  }

  render3DStructure(pdbData) {
    if (typeof $3Dmol === 'undefined') return;

    const viewer = $3Dmol.createViewer(this.elements.viewer3D, { backgroundColor: "0x000000" });
    viewer.addModel(pdbData, "pdb");
    viewer.setStyle({}, {cartoon: {color: 'spectrum'}});
    viewer.zoomTo();
    viewer.render();
    this.viewer3D = viewer;
  }

  renderConfidenceChart(confidenceData) {
    if (typeof Chart === 'undefined') return;

    const ctx = this.elements.confidenceChart.getContext('2d');

    if (this.charts.confidence) {
      this.charts.confidence.destroy();
    }

    this.charts.confidence = new Chart(ctx, {
      type: 'line',
      data: {
        labels: confidenceData.map((_, i) => i + 1),
        datasets: [{
          label: 'pLDDT Confidence',
          data: confidenceData,
          borderColor: 'rgb(10, 97, 247)',
          backgroundColor: 'rgba(10, 97, 247, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#f0f0f0' }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Aminosav Pozíció', color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.1)' },
            ticks: { color: '#64748b' }
          },
          y: {
            title: { display: true, text: 'pLDDT Érték', color: '#94a3b8' },
            min: 0,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.1)' },
            ticks: { color: '#64748b' }
          }
        }
      }
    });
  }

  renderPAEHeatmap(paeData) {
    console.log('Rendering PAE heatmap:', paeData);
  }

  loadExample() {
    if (this.elements.proteinSequence) {
      this.elements.proteinSequence.value = ">Protein_1\nMKWVTFISLLLLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL";
    }
    this.updateSequenceStats();
    this.addLog('INFO', 'Példa szekvencia betöltve');
  }

  clearSequences() {
    if (this.elements.proteinSequence) this.elements.proteinSequence.value = '';
    if (this.elements.dnaSequence) this.elements.dnaSequence.value = '';
    if (this.elements.rnaSequence) this.elements.rnaSequence.value = '';
    if (this.elements.ligandFileList) this.elements.ligandFileList.innerHTML = '';
    this.uploadedFiles = [];
    this.updateSequenceStats();
    this.addLog('INFO', 'Szekvenciák törölve');
  }

  addLog(level, message) {
    const timestamp = new Date().toLocaleTimeString('hu-HU');
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${level.toLowerCase()}`;
    logEntry.innerHTML = `<span class="log-time">[${timestamp}]</span> <span class="log-level">[${level}]</span> ${message}`;

    if (this.elements.systemLogs) {
      this.elements.systemLogs.appendChild(logEntry);
      this.elements.systemLogs.scrollTop = this.elements.systemLogs.scrollHeight;
    }
  }

  updateProgress(type, percentage) {
    const progressMap = {
      'overall': { bar: this.elements.overallProgressBar, text: this.elements.overallProgress },
      'sequence': { bar: this.elements.sequenceProgressBar, text: this.elements.sequenceProgress },
      'msa': { bar: this.elements.msaProgressBar, text: this.elements.msaProgress },
      'quantum': { bar: this.elements.quantumProgressBar, text: this.elements.quantumProgress },
      'inference': { bar: this.elements.inferenceProgressBar, text: this.elements.inferenceProgress },
      'postProcess': { bar: this.elements.postProcessProgressBar, text: this.elements.postProcessProgress }
    };

    const elements = progressMap[type];
    if (elements) {
      if (elements.bar) elements.bar.style.width = `${percentage}%`;
      if (elements.text) elements.text.textContent = `${percentage}%`;
    }
  }
}

let app;
document.addEventListener('DOMContentLoaded', () => {
  app = new AlphaFold3App();
  console.log('AlphaFold3 App initialized');
});