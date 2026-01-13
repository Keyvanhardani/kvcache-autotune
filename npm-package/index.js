/**
 * KVCache Auto-Tuner - JavaScript Wrapper
 *
 * This package provides a JavaScript wrapper for the kvat Python CLI.
 * It requires Python 3.9+ and pip to be installed on the system.
 */

const { spawn, execSync } = require('child_process');
const path = require('path');

/**
 * Check if Python and kvat are installed
 * @returns {boolean} True if kvat is available
 */
function isKvatInstalled() {
  try {
    execSync('python -m kvat --version', { stdio: 'pipe' });
    return true;
  } catch {
    try {
      execSync('python3 -m kvat --version', { stdio: 'pipe' });
      return true;
    } catch {
      return false;
    }
  }
}

/**
 * Get the Python command to use
 * @returns {string} 'python' or 'python3'
 */
function getPythonCommand() {
  try {
    execSync('python --version', { stdio: 'pipe' });
    return 'python';
  } catch {
    return 'python3';
  }
}

/**
 * Install kvat Python package
 * @param {boolean} full - Install with full dependencies
 * @returns {Promise<void>}
 */
async function installKvat(full = true) {
  const python = getPythonCommand();
  const pkg = full ? 'kvat[full]' : 'kvat';

  return new Promise((resolve, reject) => {
    const proc = spawn(python, ['-m', 'pip', 'install', pkg], {
      stdio: 'inherit'
    });

    proc.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Failed to install kvat (exit code ${code})`));
      }
    });
  });
}

/**
 * Run kvat tune command
 * @param {string} modelId - HuggingFace model ID
 * @param {Object} options - Tuning options
 * @param {string} [options.profile='chat-agent'] - Profile name
 * @param {string} [options.device='cuda'] - Device (cuda, cpu, mps)
 * @param {string} [options.outputDir='./kvat_results'] - Output directory
 * @param {boolean} [options.verbose=false] - Verbose output
 * @returns {Promise<Object>} Tuning result
 */
async function tune(modelId, options = {}) {
  const {
    profile = 'chat-agent',
    device = 'cuda',
    outputDir = './kvat_results',
    verbose = false
  } = options;

  const python = getPythonCommand();
  const args = ['-m', 'kvat', 'tune', modelId, '--profile', profile, '-o', outputDir];

  if (verbose) {
    args.push('-v');
  }

  return new Promise((resolve, reject) => {
    let stdout = '';
    let stderr = '';

    const proc = spawn(python, args, {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
      if (verbose) {
        process.stdout.write(data);
      }
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
      if (verbose) {
        process.stderr.write(data);
      }
    });

    proc.on('close', (code) => {
      if (code === 0) {
        resolve({
          success: true,
          outputDir,
          stdout,
          stderr
        });
      } else {
        reject(new Error(`kvat tune failed (exit code ${code}): ${stderr}`));
      }
    });
  });
}

/**
 * Run kvat info command
 * @returns {Promise<string>} System info
 */
async function info() {
  const python = getPythonCommand();

  return new Promise((resolve, reject) => {
    const proc = spawn(python, ['-m', 'kvat', 'info'], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(`kvat info failed: ${stderr}`));
      }
    });
  });
}

/**
 * Run arbitrary kvat command
 * @param {string[]} args - Command arguments
 * @returns {Promise<{stdout: string, stderr: string, code: number}>}
 */
async function run(args) {
  const python = getPythonCommand();

  return new Promise((resolve, reject) => {
    const proc = spawn(python, ['-m', 'kvat', ...args], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      resolve({ stdout, stderr, code });
    });
  });
}

module.exports = {
  isKvatInstalled,
  installKvat,
  tune,
  info,
  run,
  getPythonCommand
};
