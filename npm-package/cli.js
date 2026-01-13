#!/usr/bin/env node

/**
 * KVCache Auto-Tuner CLI Wrapper
 *
 * This script forwards all arguments to the Python kvat CLI.
 * Usage: kvat <command> [options]
 */

const { spawn } = require('child_process');
const { getPythonCommand, isKvatInstalled } = require('./index');

async function main() {
  const args = process.argv.slice(2);

  // Check if kvat is installed
  if (!isKvatInstalled()) {
    console.error('Error: kvat Python package is not installed.');
    console.error('');
    console.error('Please install it first:');
    console.error('  pip install kvat[full]');
    console.error('');
    console.error('Or use the JavaScript API to install:');
    console.error('  const kvat = require("kvat");');
    console.error('  await kvat.installKvat();');
    process.exit(1);
  }

  const python = getPythonCommand();

  // Forward all arguments to Python kvat
  const proc = spawn(python, ['-m', 'kvat', ...args], {
    stdio: 'inherit'
  });

  proc.on('close', (code) => {
    process.exit(code);
  });

  proc.on('error', (err) => {
    console.error('Failed to start kvat:', err.message);
    process.exit(1);
  });
}

main();
