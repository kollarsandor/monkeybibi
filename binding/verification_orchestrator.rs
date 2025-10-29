use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use tokio::runtime::Runtime;
use reqwest;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub prover: String,
    pub theorem_name: String,
    pub verified: bool,
    pub proof_steps: usize,
    pub duration_ms: u128,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProverConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub timeout_seconds: u64,
    pub enabled: bool,
}

pub struct VerificationOrchestrator {
    provers: Vec<ProverConfig>,
    results: Arc<Mutex<Vec<VerificationResult>>>,
    max_parallel: usize,
}

impl VerificationOrchestrator {
    pub fn new(max_parallel: usize) -> Self {
        let provers = vec![
            ProverConfig {
                name: "Coq".to_string(),
                command: "coqc".to_string(),
                args: vec!["-Q".to_string(), ".".to_string(), "JADED".to_string()],
                timeout_seconds: 300,
                enabled: true,
            },
            ProverConfig {
                name: "Lean4".to_string(),
                command: "lean".to_string(),
                args: vec!["--run".to_string()],
                timeout_seconds: 300,
                enabled: true,
            },
            ProverConfig {
                name: "Agda".to_string(),
                command: "agda".to_string(),
                args: vec!["--safe".to_string()],
                timeout_seconds: 300,
                enabled: true,
            },
            ProverConfig {
                name: "Isabelle".to_string(),
                command: "isabelle".to_string(),
                args: vec!["build".to_string(), "-D".to_string(), ".".to_string()],
                timeout_seconds: 600,
                enabled: true,
            },
            ProverConfig {
                name: "Z3".to_string(),
                command: "z3".to_string(),
                args: vec!["-smt2".to_string()],
                timeout_seconds: 60,
                enabled: true,
            },
            ProverConfig {
                name: "CVC5".to_string(),
                command: "cvc5".to_string(),
                args: vec!["--lang".to_string(), "smt2".to_string()],
                timeout_seconds: 60,
                enabled: true,
            },
            ProverConfig {
                name: "Dafny".to_string(),
                command: "dafny".to_string(),
                args: vec!["/compile:0".to_string()],
                timeout_seconds: 300,
                enabled: true,
            },
            ProverConfig {
                name: "F*".to_string(),
                command: "fstar".to_string(),
                args: vec!["--cache_checked_modules".to_string()],
                timeout_seconds: 300,
                enabled: true,
            },
            ProverConfig {
                name: "ACL2".to_string(),
                command: "acl2".to_string(),
                args: vec![],
                timeout_seconds: 300,
                enabled: true,
            },
            ProverConfig {
                name: "PVS".to_string(),
                command: "pvs".to_string(),
                args: vec!["-batch".to_string()],
                timeout_seconds: 300,
                enabled: true,
            },
        ];

        VerificationOrchestrator {
            provers,
            results: Arc::new(Mutex::new(Vec::new())),
            max_parallel,
        }
    }

    pub fn verify_file(&self, prover: &ProverConfig, file_path: &str) -> VerificationResult {
        let start = Instant::now();
        
        let mut cmd = Command::new(&prover.command);
        cmd.args(&prover.args)
           .arg(file_path)
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());

        match cmd.spawn() {
            Ok(mut child) => {
                let timeout = Duration::from_secs(prover.timeout_seconds);
                let start_time = Instant::now();

                loop {
                    match child.try_wait() {
                        Ok(Some(status)) => {
                            let duration = start.elapsed();
                            
                            if status.success() {
                                let stdout = child.stdout.take().unwrap();
                                let reader = BufReader::new(stdout);
                                let lines: Vec<String> = reader.lines()
                                    .filter_map(|l| l.ok())
                                    .collect();
                                
                                return VerificationResult {
                                    prover: prover.name.clone(),
                                    theorem_name: file_path.to_string(),
                                    verified: true,
                                    proof_steps: lines.len(),
                                    duration_ms: duration.as_millis(),
                                    error_message: None,
                                };
                            } else {
                                let stderr = child.stderr.take().unwrap();
                                let reader = BufReader::new(stderr);
                                let error: String = reader.lines()
                                    .filter_map(|l| l.ok())
                                    .collect::<Vec<_>>()
                                    .join("\n");
                                
                                return VerificationResult {
                                    prover: prover.name.clone(),
                                    theorem_name: file_path.to_string(),
                                    verified: false,
                                    proof_steps: 0,
                                    duration_ms: duration.as_millis(),
                                    error_message: Some(error),
                                };
                            }
                        }
                        Ok(None) => {
                            if start_time.elapsed() > timeout {
                                let _ = child.kill();
                                return VerificationResult {
                                    prover: prover.name.clone(),
                                    theorem_name: file_path.to_string(),
                                    verified: false,
                                    proof_steps: 0,
                                    duration_ms: timeout.as_millis(),
                                    error_message: Some("Timeout".to_string()),
                                };
                            }
                            thread::sleep(Duration::from_millis(100));
                        }
                        Err(e) => {
                            return VerificationResult {
                                prover: prover.name.clone(),
                                theorem_name: file_path.to_string(),
                                verified: false,
                                proof_steps: 0,
                                duration_ms: start.elapsed().as_millis(),
                                error_message: Some(e.to_string()),
                            };
                        }
                    }
                }
            }
            Err(e) => {
                VerificationResult {
                    prover: prover.name.clone(),
                    theorem_name: file_path.to_string(),
                    verified: false,
                    proof_steps: 0,
                    duration_ms: start.elapsed().as_millis(),
                    error_message: Some(e.to_string()),
                }
            }
        }
    }

    pub fn verify_all(&self, file_paths: Vec<String>) -> Vec<VerificationResult> {
        let mut handles = vec![];
        let results = self.results.clone();

        for prover in self.provers.iter().filter(|p| p.enabled) {
            for file_path in &file_paths {
                let prover_clone = prover.clone();
                let file_clone = file_path.clone();
                let results_clone = results.clone();

                let handle = thread::spawn(move || {
                    let result = VerificationOrchestrator::verify_single(
                        &prover_clone,
                        &file_clone
                    );
                    
                    let mut res = results_clone.lock().unwrap();
                    res.push(result);
                });

                handles.push(handle);

                if handles.len() >= self.max_parallel {
                    for h in handles.drain(..) {
                        let _ = h.join();
                    }
                }
            }
        }

        for h in handles {
            let _ = h.join();
        }

        let final_results = results.lock().unwrap();
        final_results.clone()
    }

    fn verify_single(prover: &ProverConfig, file_path: &str) -> VerificationResult {
        let orch = VerificationOrchestrator::new(1);
        orch.verify_file(prover, file_path)
    }

    pub fn aggregate_results(&self, results: &[VerificationResult]) -> AggregateReport {
        let total = results.len();
        let verified = results.iter().filter(|r| r.verified).count();
        let failed = total - verified;
        let total_duration: u128 = results.iter().map(|r| r.duration_ms).sum();
        let avg_duration = if total > 0 { total_duration / total as u128 } else { 0 };

        let by_prover: std::collections::HashMap<String, (usize, usize)> = 
            results.iter().fold(std::collections::HashMap::new(), |mut acc, r| {
                let entry = acc.entry(r.prover.clone()).or_insert((0, 0));
                entry.0 += 1;
                if r.verified {
                    entry.1 += 1;
                }
                acc
            });

        AggregateReport {
            total_verifications: total,
            verified_count: verified,
            failed_count: failed,
            average_duration_ms: avg_duration,
            results_by_prover: by_prover,
        }
    }

    pub async fn notify_backend(&self, results: &[VerificationResult]) -> Result<(), Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let julia_url = "http://0.0.0.0:6000/verification_results";
        
        let payload = serde_json::to_string(results)?;
        
        let response = client.post(julia_url)
            .header("Content-Type", "application/json")
            .body(payload)
            .send()
            .await?;

        if response.status().is_success() {
            println!("✓ Verification results sent to Julia backend");
        } else {
            eprintln!("✗ Failed to send results: {}", response.status());
        }

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AggregateReport {
    pub total_verifications: usize,
    pub verified_count: usize,
    pub failed_count: usize,
    pub average_duration_ms: u128,
    pub results_by_prover: std::collections::HashMap<String, (usize, usize)>,
}

#[tokio::main]
async fn main() {
    println!("JADED Verification Orchestrator");
    println!("================================\n");

    let orchestrator = VerificationOrchestrator::new(8);
    
    let test_files = vec![
        "model/fv.coq".to_string(),
    ];

    println!("Running verification across {} provers...\n", orchestrator.provers.len());
    
    let start = Instant::now();
    let results = orchestrator.verify_all(test_files);
    let duration = start.elapsed();

    println!("\nVerification Results:");
    println!("=====================\n");
    
    for result in &results {
        let status = if result.verified { "✓" } else { "✗" };
        println!("{} {} - {} ({} ms, {} steps)",
                 status,
                 result.prover,
                 result.theorem_name,
                 result.duration_ms,
                 result.proof_steps);
        
        if let Some(ref error) = result.error_message {
            if !error.is_empty() {
                println!("  Error: {}", error.lines().next().unwrap_or("Unknown"));
            }
        }
    }

    let report = orchestrator.aggregate_results(&results);
    
    println!("\nAggregate Report:");
    println!("=================");
    println!("Total verifications: {}", report.total_verifications);
    println!("Verified: {}", report.verified_count);
    println!("Failed: {}", report.failed_count);
    println!("Average duration: {} ms", report.average_duration_ms);
    println!("Total time: {:.2} s", duration.as_secs_f64());
    
    println!("\nBy Prover:");
    for (prover, (total, verified)) in report.results_by_prover {
        println!("  {}: {}/{} verified", prover, verified, total);
    }

    if let Err(e) = orchestrator.notify_backend(&results).await {
        eprintln!("\nWarning: Could not notify backend: {}", e);
    }

    let success_rate = (report.verified_count as f64 / report.total_verifications as f64) * 100.0;
    println!("\nOverall success rate: {:.1}%", success_rate);
    
    if report.verified_count == report.total_verifications {
        println!("\n✓ All verifications passed!");
        std::process::exit(0);
    } else {
        println!("\n✗ Some verifications failed");
        std::process::exit(1);
    }
}
