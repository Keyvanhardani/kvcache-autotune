#!/usr/bin/env node

/**
 * KVCache Auto-Tuner - Post-install Script
 *
 * Checks for Python and provides installation instructions.
 */

const { execSync } = require('child_process');

function checkPython() {
  try {
    const version = execSync('python --version', { stdio: 'pipe' }).toString().trim();
    return { available: true, version, command: 'python' };
  } catch {
    try {
      const version = execSync('python3 --version', { stdio: 'pipe' }).toString().trim();
      return { available: true, version, command: 'python3' };
    } catch {
      return { available: false };
    }
  }
}

function checkKvat(pythonCmd) {
  try {
    execSync(`${pythonCmd} -m kvat --version`, { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

function main() {
  console.log('');
  console.log('KVCache Auto-Tuner (kvat) - npm package installed');
  console.log('='.repeat(50));
  console.log('');

  const python = checkPython();

  if (!python.available) {
    console.log('WARNING: Python is not installed or not in PATH.');
    console.log('');
    console.log('kvat requires Python 3.9+ to run. Please install Python first:');
    console.log('  - Windows: https://www.python.org/downloads/');
    console.log('  - macOS: brew install python3');
    console.log('  - Linux: sudo apt install python3 python3-pip');
    console.log('');
    return;
  }

  console.log(`Python found: ${python.version}`);

  const kvatInstalled = checkKvat(python.command);

  if (kvatInstalled) {
    console.log('kvat Python package: Installed');
    console.log('');
    console.log('You can now use kvat:');
    console.log('  kvat tune gpt2 --profile ci-micro');
    console.log('  kvat info');
  } else {
    console.log('kvat Python package: Not installed');
    console.log('');
    console.log('To install the Python package, run:');
    console.log(`  ${python.command} -m pip install kvat[full]`);
    console.log('');
    console.log('Or use the JavaScript API:');
    console.log('  const kvat = require("kvat");');
    console.log('  await kvat.installKvat();');
  }

  console.log('');
  console.log('Documentation: https://github.com/Keyvanhardani/kvcache-autotune');
  console.log('');
}

main();
